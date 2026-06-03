import csv
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Rotation as R
from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive

from .config import CSV_COLUMNS, PBVSConfig, TargetPose, vec6_to_str
from .geometry import (
    compute_L2s,
    compute_L2_b,
    compute_N2s,
    get_tag_3d_corners,
    inv_T,
    project_3d_to_2d,
)
from .psmc import PSMCPDProxy
from .vision import AprilTagEstimator, draw_axis, draw_axis_colored


class PBVSController:
    def __init__(self, robot_ip: str, intrinsics,
                 hand_eye_calib: np.ndarray,
                 config: PBVSConfig = None):
        self.cfg = config or PBVSConfig()
        self.e_T_c = hand_eye_calib
        self.R_etc = self.e_T_c[0:3, 0:3]
        self.p_etc = self.e_T_c[:3, 3]
        self.rtde_freq = 60.0
        self.dt = 1.0 / self.rtde_freq

        self.rtde_c = RTDEControl(robot_ip, self.rtde_freq)
        receive_vars = ["actual_TCP_pose", "actual_TCP_speed"]
        self.rtde_r = RTDEReceive(
            robot_ip, self.rtde_freq,
            receive_vars,
            True, False, 60
        )
        self.estimator = AprilTagEstimator(self.cfg, intrinsics)

        self.targets: List[TargetPose] = []
        self.cur_target: Optional[TargetPose] = None
        self.cur_target_idx = 0

        self._active_T_des: Optional[np.ndarray] = None
        self._last_s_d = np.zeros(6)
        self._last_s_dot_d = np.zeros(6)
        self._last_s_ddot_d = np.zeros(6)
        self._last_accel_saturated = False

        self.stable_cnt = 0
        self._u_c_integrated = np.zeros(6)

        self._psmc = PSMCPDProxy(
            n=6,
            K=self.cfg.kp,
            B=self.cfg.kd,
            A=self.cfg.accel_limit,
            H=self.cfg.proxy_H,
            dt=self.dt
        )
        self._sopd_A = self.cfg.accel_limit.copy()
        self._sopd_a_star = np.zeros(6)
        self._sopd_Phi = np.zeros(6)

        self._error_log: list = []
        self._t0: float = 0.0
        self._proxy_T_cam: Optional[np.ndarray] = None

        self._csv_file = None
        self._csv_writer = None
        self._frame_idx = 0

        self._cur_u_c = np.zeros(6)
        self._last_u_c = np.zeros(6)
        self._last_u_dot_c = np.zeros(6)
        self._u_c_measured_filtered = np.zeros(6)
        self._has_u_c_measured_filtered = False
        self._last_s_for_diff: Optional[np.ndarray] = None
        self._last_s_diff_time: Optional[float] = None
        self._s_dot_diff_filtered = np.zeros(6)
        self._u_o_filtered = np.zeros(6)
        self._last_u_o_for_diff: Optional[np.ndarray] = None
        self._last_u_o_diff_time: Optional[float] = None
        self._u_dot_o_filtered = np.zeros(6)
        self._last_detection = None

    def _open_csv(self):
        if not self.cfg.enable_csv_logging:
            return
        path = Path(self.cfg.csv_save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._csv_file = open(path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(
            self._csv_file, fieldnames=CSV_COLUMNS, extrasaction="ignore"
        )
        self._csv_writer.writeheader()
        print(f"📄 CSV: {path}")

    def _close_csv(self):
        if self._csv_file:
            self._csv_file.flush()
            self._csv_file.close()
            self._csv_file = self._csv_writer = None
            print(f"✅ CSV 已保存: {self.cfg.csv_save_path}")

    def _write_csv_row(self, row: dict):
        if not self._csv_writer:
            return
        if callable(row):
            row = row()
        clean = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in row.items()}
        self._csv_writer.writerow(clean)
        if self._frame_idx % self.cfg.csv_flush_interval == 0:
            self._csv_file.flush()

    def set_targets(self, targets: List[TargetPose]):
        self.targets = targets
        print(f"✓ 已加载 {len(targets)} 个目标位姿")

    def _switch_target(self, idx: int) -> bool:
        if idx >= len(self.targets):
            return False
        self.cur_target_idx = idx
        self.cur_target = self.targets[idx]
        self._active_T_des = self.cur_target.T_des.copy()
        self._reset_controller_state()
        print(f"\n🎯 切换目标: {self.cur_target.name}")
        return True

    def _reset_controller_state(self):
        self.stable_cnt = 0
        self._u_c_integrated = np.zeros(6)
        self._psmc.reset()
        self._sopd_a_star = np.zeros(6)
        self._sopd_Phi = np.zeros(6)
        self._proxy_T_cam = None
        self._cur_u_c = np.zeros(6)
        self._last_u_c = np.zeros(6)
        self._last_u_dot_c = np.zeros(6)
        self._u_c_measured_filtered = np.zeros(6)
        self._has_u_c_measured_filtered = False
        self._last_s_for_diff = None
        self._last_s_diff_time = None
        self._s_dot_diff_filtered = np.zeros(6)
        self._u_o_filtered = np.zeros(6)
        self._last_u_o_for_diff = None
        self._last_u_o_diff_time = None
        self._u_dot_o_filtered = np.zeros(6)
        self._last_detection = None
        self._last_s_d = np.zeros(6)
        self._last_s_dot_d = np.zeros(6)
        self._last_s_ddot_d = np.zeros(6)
        self._last_accel_saturated = False

    def _desired_T(self) -> np.ndarray:
        if self._active_T_des is not None:
            return self._active_T_des
        if self.cur_target is not None:
            return self.cur_target.T_des
        return np.eye(4)

    def _fixed_reference(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.cur_target is None:
            T = np.eye(4)
            z = np.zeros(6)
            return T, z.copy(), z.copy(), z.copy()

        T0 = self.cur_target.T_des.copy()
        self._active_T_des = T0
        q_des = Rotation.from_matrix(T0[:3, :3]).as_quat()
        if q_des[3] < 0:
            q_des = -q_des
        s_d = np.concatenate([T0[:3, 3], q_des[:3]])
        z = np.zeros(6)
        return T0, s_d, z.copy(), z.copy()

    def _compute_feature_error(self, T_current: np.ndarray,
                               R_base_cam: np.ndarray,
                               s_d_override: np.ndarray = None):
        T_des = self._desired_T()
        T_err = T_des @ inv_T(T_current)
        q_des = Rotation.from_matrix(T_des[:3, :3]).as_quat()
        q_oc = Rotation.from_matrix(T_current[:3, :3]).as_quat()
        if q_des[3] < 0:
            q_des = -q_des
        if np.dot(q_oc, q_des) < 0:
            q_oc = -q_oc
        c_p_oc = T_current[:3, 3]
        s_d = np.concatenate([T_des[:3, 3], q_des[:3]])
        Ls = compute_L2s(c_p_oc, q_oc, R_base_cam)
        if s_d_override is not None:
            s_d = np.asarray(s_d_override, dtype=float).reshape(6).copy()

        s = np.concatenate([c_p_oc, q_oc[:3]])
        e = s_d - s

        try:
            Ls_inv = np.linalg.inv(Ls)
        except np.linalg.LinAlgError:
            Ls_inv = np.linalg.pinv(Ls)

        return s, s_d, e, q_oc, c_p_oc, Ls, Ls_inv, T_err

    def _current_u_c(self) -> np.ndarray:
        return self._last_u_c.copy()

    def _filter_u_c_measured(self, u_c_measured: np.ndarray) -> np.ndarray:
        u_c_measured = np.asarray(u_c_measured, dtype=float).reshape(6)
        if not self._has_u_c_measured_filtered:
            self._u_c_measured_filtered = u_c_measured.copy()
            self._has_u_c_measured_filtered = True
        else:
            tau = 0.10
            beta = tau / (tau + self.dt)
            self._u_c_measured_filtered = (
                beta * self._u_c_measured_filtered
                + (1.0 - beta) * u_c_measured
            )
        return self._u_c_measured_filtered.copy()

    def _compute_s_dot_by_difference(self, s: np.ndarray) -> np.ndarray:
        """Eq. (42a): obtain s_dot from a filtered visual-feature difference."""
        s = np.asarray(s, dtype=float).reshape(6)
        now = time.perf_counter()
        if self._last_s_for_diff is None or self._last_s_diff_time is None:
            self._s_dot_diff_filtered = np.zeros(6)
        else:
            dt = max(now - self._last_s_diff_time, 1e-4)
            s_dot_raw = (s - self._last_s_for_diff) / dt
            tau = 0.04
            beta = tau / (tau + dt)
            self._s_dot_diff_filtered = (
                beta * self._s_dot_diff_filtered
                + (1.0 - beta) * s_dot_raw
            )
        self._last_s_for_diff = s.copy()
        self._last_s_diff_time = now
        return self._s_dot_diff_filtered.copy()

    def _filter_u_o(self, u_o_raw: np.ndarray) -> np.ndarray:
        u_o = np.asarray(u_o_raw, dtype=float).reshape(6).copy()
        deadband = np.array([0.015, 0.015, 0.015, 0.05, 0.05, 0.05])
        limits = np.array([0.50, 0.50, 0.50, 2.0, 2.0, 2.0])
        u_o[np.abs(u_o) < deadband] = 0.0
        u_o = np.clip(u_o, -limits, limits)
        tau = 0.12
        beta = tau / (tau + self.dt)
        self._u_o_filtered = beta * self._u_o_filtered + (1.0 - beta) * u_o
        return self._u_o_filtered.copy()

    def _compute_u_dot_o_by_difference(self, u_o: np.ndarray) -> np.ndarray:
        u_o = np.asarray(u_o, dtype=float).reshape(6)
        now = time.perf_counter()
        if self._last_u_o_for_diff is None or self._last_u_o_diff_time is None:
            self._u_dot_o_filtered = np.zeros(6)
        else:
            dt = max(now - self._last_u_o_diff_time, 1e-4)
            u_dot_o_raw = (u_o - self._last_u_o_for_diff) / dt
            limits = np.array([2.0, 2.0, 2.0, 8.0, 8.0, 8.0])
            u_dot_o_raw = np.clip(u_dot_o_raw, -limits, limits)
            tau = 0.20
            beta = tau / (tau + dt)
            self._u_dot_o_filtered = (
                beta * self._u_dot_o_filtered
                + (1.0 - beta) * u_dot_o_raw
            )
        self._last_u_o_for_diff = u_o.copy()
        self._last_u_o_diff_time = now
        return self._u_dot_o_filtered.copy()

    def _compute_u_dot_c(self, Phi: np.ndarray, q_oc: np.ndarray,
                  c_p_oc: np.ndarray, Ls: np.ndarray,
                  Ls_inv: np.ndarray, u_c: np.ndarray,
                  u_o: np.ndarray, u_dot_o: np.ndarray,
                  N: np.ndarray, R_base_cam: np.ndarray) -> np.ndarray:
        """Eq. (25): u_dot_c = L^{-1}(alpha_c - b - N u_dot_o)."""
        b = compute_L2_b(c_p_oc, q_oc, u_c, u_o, R_base_cam)
        return Ls_inv @ (Phi - b - N @ u_dot_o)

    @staticmethod
    def _clip_3d(x: np.ndarray, A: float) -> np.ndarray:
        norm = float(np.linalg.norm(x))
        if norm <= A or A == float("inf"):
            return x.copy()
        return (A / max(norm, A)) * x

    def _clip_camera_acceleration(self, u_dot_c: np.ndarray) -> np.ndarray:
        clipped = np.zeros_like(u_dot_c)
        clipped[:3] = self._clip_3d(u_dot_c[:3], float(self._sopd_A[0]))
        clipped[3:] = self._clip_3d(u_dot_c[3:], float(self._sopd_A[3]))
        return clipped

    def _is_camera_accel_saturated(self, u_dot_c: np.ndarray) -> bool:
        return (
            np.linalg.norm(u_dot_c[:3]) > float(self._sopd_A[0])
            or np.linalg.norm(u_dot_c[3:]) > float(self._sopd_A[3])
        )

    def _proxy_feature_to_T_cam(self, proxy_pos: np.ndarray) -> Optional[np.ndarray]:
        try:
            t_proxy = proxy_pos[:3]
            qv_proxy = proxy_pos[3:6]
            qv_ns = np.dot(qv_proxy, qv_proxy)
            if qv_ns > 1.0:
                qv_proxy = qv_proxy / np.sqrt(qv_ns + 1e-12)
                qv_ns = np.dot(qv_proxy, qv_proxy)
            q0_proxy = np.sqrt(max(0.0, 1.0 - qv_ns))
            q_s = np.array([qv_proxy[0], qv_proxy[1], qv_proxy[2], q0_proxy])
            T_err_proxy = np.eye(4)
            T_err_proxy[:3, :3] = Rotation.from_quat(q_s).as_matrix()
            T_err_proxy[:3, 3] = t_proxy
            T_proxy_cam = inv_T(T_err_proxy) @ self._desired_T()
            return T_proxy_cam if T_proxy_cam[2, 3] > 0 else None
        except Exception:
            return None

    def _compute_control(self, T_current: np.ndarray,
                         R_base_cam: np.ndarray,
                         u_c_measured: np.ndarray = None,
                         s_dot_d: np.ndarray = None,
                         s_ddot_d: np.ndarray = None,
                         s_d_ref: np.ndarray = None):
        if s_dot_d is None:
            s_dot_d = np.zeros(6)
        if s_ddot_d is None:
            s_ddot_d = np.zeros(6)

        s, s_d_base, _, q_oc, c_p_oc, Ls, Ls_inv, T_err = self._compute_feature_error(
            T_current, R_base_cam, s_d_override=s_d_ref
        )
        s_d_cmd = s_d_base
        s_dot_d_cmd = s_dot_d
        s_ddot_d_cmd = s_ddot_d
        e = s_d_cmd - s
        u_c_cmd = self._current_u_c()
        if u_c_measured is None:
            u_c = u_c_cmd
        else:
            # Use the measured camera twist only as a low-pass correction.
            # This keeps Eq. (42b)'s robot feedback while avoiding raw RTDE noise.
            u_c_measured_filtered = self._filter_u_c_measured(u_c_measured)
            measured_weight = 0.0
            u_c = ((1.0 - measured_weight) * u_c_cmd
                   + measured_weight * u_c_measured_filtered)
        s_dot_by_interaction_matrix = Ls @ u_c
        s_dot_by_difference = self._compute_s_dot_by_difference(s)
        K = np.diag(self.cfg.kp)
        B = np.diag(self.cfg.kd)
        N = compute_N2s(q_oc, R_base_cam)
        try:
            N_inv = np.linalg.inv(N)
        except np.linalg.LinAlgError:
            N_inv = np.linalg.pinv(N)
        # Eq. (42g): uo = N^{-1}(s_dot - L uc).
        u_o = self._filter_u_o(N_inv @ (s_dot_by_difference - s_dot_by_interaction_matrix))
        # u_o = N_inv @ (s_dot_by_difference - s_dot_by_interaction_matrix)
        edot = s_dot_d_cmd - s_dot_by_interaction_matrix
        u_dot_o = self._compute_u_dot_o_by_difference(u_o)
        mode = self.cfg.controller_mode.upper()
        proxy_H_cmd = np.full(6, float("nan"))
        if mode == "SOPD":
            # Baseline: alpha_c = Kp(s* - s) + Kd(s_dot* - s_dot).
            Phi_fb_raw = K @ e + B @ edot
            a_star_raw = Phi_fb_raw.copy()
            self._sopd_a_star = a_star_raw.copy()
            Phi = Phi_fb_raw + s_ddot_d_cmd
            self._sopd_Phi = Phi.copy()
            self._last_accel_saturated = False

        elif mode == "SOPD_SAT":
            # Eq. (28)-style naive baseline: clip camera acceleration u_dot_c.
            Phi_fb_raw = K @ e + B @ edot
            a_star_raw = Phi_fb_raw.copy()
            self._sopd_a_star = a_star_raw.copy()
            Phi = Phi_fb_raw + s_ddot_d_cmd
            self._sopd_Phi = Phi.copy()
            u_dot_c_raw = self._compute_u_dot_c(Phi, q_oc, c_p_oc, Ls, Ls_inv,
                                                u_c, u_o, u_dot_o, N, R_base_cam)
            u_dot_c = self._clip_camera_acceleration(u_dot_c_raw)
            self._last_accel_saturated = self._is_camera_accel_saturated(u_dot_c_raw)

        else:
            self._psmc.H = self.cfg.proxy_H.copy()
            proxy_H_cmd = self.cfg.proxy_H.copy()
            # Eq. (42h): b = b(s, qc, uc, uo).
            b = compute_L2_b(c_p_oc, q_oc, u_c, u_o, R_base_cam)
            # Eq. (42i)-(42n): proxy update and projection of u_dot_c*.
            u_dot_c = self._psmc.compute(
                s=s,
                s_dot=s_dot_by_interaction_matrix,
                s_d=s_d_cmd,
                s_dot_d=s_dot_d_cmd,
                L=Ls,
                L_inv=Ls_inv,
                b=b,
                N=N,
                u_dot_o=u_dot_o,
            )
            a_star_raw = self._psmc.u_dot_c_star.copy()
            Phi = self._psmc.alpha_c.copy()
            self._last_accel_saturated = self._psmc.is_accel_saturated
        if mode == "SOPD":
            u_dot_c = self._compute_u_dot_c(Phi, q_oc, c_p_oc, Ls, Ls_inv,
                                            u_c, u_o, u_dot_o, N, R_base_cam)

        # Eq. (42o): uc = uc,prv + T u_dot_c.
        self._u_c_integrated += u_dot_c * self.dt

        return self._u_c_integrated, u_dot_c, s, s_d_cmd, e, edot, s_dot_by_interaction_matrix, a_star_raw, Phi, proxy_H_cmd

    def _u_c_to_tcp_twist_base(self, u_c: np.ndarray, tcp_pose: np.ndarray) -> np.ndarray:
        R_base_tcp = R.from_rotvec(tcp_pose[3:]).as_matrix()
        p_base_ec = R_base_tcp @ self.p_etc

        v_c_base = u_c[:3]
        omega_base = u_c[3:]

        v_tcp_base = v_c_base - np.cross(omega_base, p_base_ec)

        return np.concatenate([v_tcp_base, omega_base])

    def _tcp_twist_base_to_u_c(self, tcp_twist: np.ndarray, tcp_pose: np.ndarray) -> np.ndarray:
        R_base_tcp = R.from_rotvec(tcp_pose[3:]).as_matrix()
        p_base_ec = R_base_tcp @ self.p_etc

        v_tcp_base = tcp_twist[:3]
        omega_base = tcp_twist[3:]
        v_c_base = v_tcp_base + np.cross(omega_base, p_base_ec)

        return np.concatenate([v_c_base, omega_base])

    def _detect_or_reuse_tag(self, gray_img):
        should_detect = (
            self._last_detection is None
            or self._frame_idx % self.cfg.detect_stride == 0
        )
        if should_detect:
            detection = self.estimator.detect(gray_img)
            if detection[0] is not None:
                self._last_detection = detection
            else:
                self._last_detection = None
            return detection
        return self._last_detection

    def process_step(self, gray_img, tcp_pose: np.ndarray = None,
                     tcp_speed: np.ndarray = None):
        T_cur, corners, R_cur, t_cur = self._detect_or_reuse_tag(gray_img)
        
        if T_cur is None:
            self._last_u_c = np.zeros(6)
            self._last_u_dot_c = np.zeros(6)
            self._last_s_for_diff = None
            self._write_lost_frame_row(tcp_pose)
            return None, None, False, None, None, None

        _, s_d_ref, s_dot_d, s_ddot_d = self._fixed_reference()
        self._last_s_d = s_d_ref.copy()
        self._last_s_dot_d = s_dot_d.copy()
        self._last_s_ddot_d = s_ddot_d.copy()

        if tcp_pose is None:
            actual_pose = np.array(self.rtde_r.getActualTCPPose())
        else:
            actual_pose = np.asarray(tcp_pose, dtype=float)
        if tcp_speed is None:
            try:
                actual_speed = np.array(self.rtde_r.getActualTCPSpeed())
            except Exception:
                actual_speed = None
        else:
            actual_speed = np.asarray(tcp_speed, dtype=float)
        u_c_measured = None
        if actual_speed is not None:
            u_c_measured = self._tcp_twist_base_to_u_c(actual_speed, actual_pose)
        R_base_tcp = R.from_rotvec(actual_pose[3:]).as_matrix()
        R_base_cam = R_base_tcp @ self.R_etc

        u_c, u_dot_c, s, s_d, e, edot, s_dot, a_star, Phi, proxy_H_cmd = self._compute_control(
            T_cur,
            R_base_cam,
            u_c_measured=u_c_measured,
            s_dot_d=s_dot_d,
            s_ddot_d=s_ddot_d,
            s_d_ref=s_d_ref,
        )

        self._last_s_d = s_d.copy()
        self._last_s_dot_d = s_dot_d.copy()
        self._last_s_ddot_d = s_ddot_d.copy()
        self._last_u_c = u_c.copy()
        self._last_u_dot_c = u_dot_c.copy()
        self._cur_u_c = u_c.copy()

        T_err = self._desired_T() @ inv_T(T_cur)
        t_err_vec = T_err[:3, 3]
        R_err_rot = Rotation.from_matrix(T_err[:3, :3])
        err_pos = float(np.linalg.norm(t_err_vec))
        err_rot = float(np.linalg.norm(R_err_rot.as_rotvec()))
        r_euler = R_err_rot.as_euler('xyz', degrees=True)

        mode = self.cfg.controller_mode.upper()
        self._proxy_T_cam = None
        if "PSMC" in mode:
            self._proxy_T_cam = self._proxy_feature_to_T_cam(self._psmc.proxy_position)

        v_ctrl = self._u_c_to_tcp_twist_base(u_c, actual_pose)
        converged_now = (err_pos < self.cfg.pos_threshold
                         and err_rot < self.cfg.rot_threshold)
        if converged_now:
            self.stable_cnt += 1
            if (self.cfg.slow_after_convergence
                    and self.stable_cnt > self.cfg.convergence_slowdown_frames):
                v_ctrl *= self.cfg.convergence_velocity_scale
        else:
            self.stable_cnt = 0
        converged = self.stable_cnt >= self.cfg.stable_frames

        log_needed = self.cfg.enable_memory_log or self._csv_writer is not None
        if log_needed and "PSMC" in mode:
            Phi_log = Phi.copy()
            astar_log = a_star.copy()
            proxy_s_log = self._psmc.proxy_position
            proxy_b_log = self._psmc.proxy_offset
            sat_a = self._last_accel_saturated
        elif log_needed:
            Phi_log = self._sopd_Phi.copy()
            astar_log = self._sopd_a_star.copy()
            proxy_s_log = np.full(6, float("nan"))
            proxy_b_log = np.full(6, float("nan"))
            sat_a = self._last_accel_saturated
        else:
            sat_a = self._last_accel_saturated

        t_now = time.time() - self._t0
        if self.cfg.enable_memory_log:
            des_corners_px = np.full((4, 2), float("nan"))
            if self.cur_target is not None:
                des_corners_3d = get_tag_3d_corners(self.cfg.tag_size, self._desired_T())
                if np.all(des_corners_3d[:, 2] > 0):
                    des_corners_px = project_3d_to_2d(des_corners_3d, self.estimator.K)

            self._error_log.append({
                "t": t_now,
                "err_pos": err_pos * 1000.0,
                "err_rot": float(np.rad2deg(err_rot)),
                "ex": float(t_err_vec[0]*1000), "ey": float(t_err_vec[1]*1000),
                "ez": float(t_err_vec[2]*1000),
                "rx": float(r_euler[0]), "ry": float(r_euler[1]), "rz": float(r_euler[2]),
                "Phi0": float(Phi_log[0]), "Phi1": float(Phi_log[1]), "Phi2": float(Phi_log[2]),
                "Phi3": float(Phi_log[3]), "Phi4": float(Phi_log[4]), "Phi5": float(Phi_log[5]),
                "udotc0": float(u_dot_c[0]), "udotc1": float(u_dot_c[1]), "udotc2": float(u_dot_c[2]),
                "udotc3": float(u_dot_c[3]), "udotc4": float(u_dot_c[4]), "udotc5": float(u_dot_c[5]),
                "as0": float(astar_log[0]), "as1": float(astar_log[1]), "as2": float(astar_log[2]),
                "as3": float(astar_log[3]), "as4": float(astar_log[4]), "as5": float(astar_log[5]),
                "proxy_s0": float(proxy_s_log[0]), "proxy_s1": float(proxy_s_log[1]), "proxy_s2": float(proxy_s_log[2]),
                "proxy_s3": float(proxy_s_log[3]), "proxy_s4": float(proxy_s_log[4]), "proxy_s5": float(proxy_s_log[5]),
                "h0": float(proxy_H_cmd[0]), "h1": float(proxy_H_cmd[1]), "h2": float(proxy_H_cmd[2]),
                "h3": float(proxy_H_cmd[3]), "h4": float(proxy_H_cmd[4]), "h5": float(proxy_H_cmd[5]),
                "sat_a": int(sat_a),
                "target": self.cur_target.name,
                "cx0": float(corners[0,0]), "cy0": float(corners[0,1]),
                "cx1": float(corners[1,0]), "cy1": float(corners[1,1]),
                "cx2": float(corners[2,0]), "cy2": float(corners[2,1]),
                "cx3": float(corners[3,0]), "cy3": float(corners[3,1]),
                "dcx0": float(des_corners_px[0,0]), "dcy0": float(des_corners_px[0,1]),
                "dcx1": float(des_corners_px[1,0]), "dcy1": float(des_corners_px[1,1]),
                "dcx2": float(des_corners_px[2,0]), "dcy2": float(des_corners_px[2,1]),
                "dcx3": float(des_corners_px[3,0]), "dcy3": float(des_corners_px[3,1]),
                "tcp_x": float(tcp_pose[0]) if tcp_pose is not None else float("nan"),
                "tcp_y": float(tcp_pose[1]) if tcp_pose is not None else float("nan"),
                "tcp_z": float(tcp_pose[2]) if tcp_pose is not None else float("nan"),
            })

        self._write_csv_row(lambda: {
            "t": t_now, "frame_idx": self._frame_idx, "target": self.cur_target.name,
            "err_pos_mm": err_pos*1000.0,
            "ex_mm": float(t_err_vec[0]*1000), "ey_mm": float(t_err_vec[1]*1000),
            "ez_mm": float(t_err_vec[2]*1000),
            "err_rot_deg": float(np.rad2deg(err_rot)),
            "rx_deg": float(r_euler[0]), "ry_deg": float(r_euler[1]),
            "rz_deg": float(r_euler[2]),
            "s0": float(s[0]), "s1": float(s[1]), "s2": float(s[2]),
            "s3": float(s[3]), "s4": float(s[4]), "s5": float(s[5]),
            "sstar0": float(s_d[0]), "sstar1": float(s_d[1]),
            "sstar2": float(s_d[2]), "sstar3": float(s_d[3]),
            "sstar4": float(s_d[4]), "sstar5": float(s_d[5]),
            "sdotstar0": float(s_dot_d[0]),
            "sdotstar1": float(s_dot_d[1]),
            "sdotstar2": float(s_dot_d[2]),
            "sdotstar3": float(s_dot_d[3]),
            "sdotstar4": float(s_dot_d[4]),
            "sdotstar5": float(s_dot_d[5]),
            "sddotstar0": float(s_ddot_d[0]),
            "sddotstar1": float(s_ddot_d[1]),
            "sddotstar2": float(s_ddot_d[2]),
            "sddotstar3": float(s_ddot_d[3]),
            "sddotstar4": float(s_ddot_d[4]),
            "sddotstar5": float(s_ddot_d[5]),
            "edot0": float(edot[0]), "edot1_val": float(edot[1]),
            "edot2_val": float(edot[2]), "edot3_val": float(edot[3]),
            "edot4_val": float(edot[4]), "edot5_val": float(edot[5]),
            "astar0": float(a_star[0]), "astar1": float(a_star[1]),
            "astar2": float(a_star[2]), "astar3": float(a_star[3]),
            "astar4": float(a_star[4]), "astar5": float(a_star[5]),
            "Phi0": float(Phi_log[0]), "Phi1": float(Phi_log[1]),
            "Phi2": float(Phi_log[2]), "Phi3": float(Phi_log[3]),
            "Phi4": float(Phi_log[4]), "Phi5": float(Phi_log[5]),
            "accel_saturated": int(sat_a),
            "proxy_s0": float(proxy_s_log[0]), "proxy_s1": float(proxy_s_log[1]),
            "proxy_s2": float(proxy_s_log[2]), "proxy_s3": float(proxy_s_log[3]),
            "proxy_s4": float(proxy_s_log[4]), "proxy_s5": float(proxy_s_log[5]),
            "proxy_H0": float(proxy_H_cmd[0]), "proxy_H1": float(proxy_H_cmd[1]),
            "proxy_H2": float(proxy_H_cmd[2]), "proxy_H3": float(proxy_H_cmd[3]),
            "proxy_H4": float(proxy_H_cmd[4]), "proxy_H5": float(proxy_H_cmd[5]),
            "proxy_b0": float(proxy_b_log[0]), "proxy_b1": float(proxy_b_log[1]),
            "proxy_b2": float(proxy_b_log[2]), "proxy_b3": float(proxy_b_log[3]),
            "proxy_b4": float(proxy_b_log[4]), "proxy_b5": float(proxy_b_log[5]),
            "uc0": float(u_c[0]), "uc1": float(u_c[1]),
            "uc2": float(u_c[2]), "uc3": float(u_c[3]),
            "uc4": float(u_c[4]), "uc5": float(u_c[5]),
            "vtcp0": float(v_ctrl[0]), "vtcp1": float(v_ctrl[1]),
            "vtcp2": float(v_ctrl[2]), "vtcp3": float(v_ctrl[3]),
            "vtcp4": float(v_ctrl[4]), "vtcp5": float(v_ctrl[5]),
            "vtcp_lin_norm": float(np.linalg.norm(v_ctrl[:3])),
            "vtcp_ang_norm": float(np.linalg.norm(v_ctrl[3:])),
            "udotc0": float(u_dot_c[0]), "udotc1": float(u_dot_c[1]),
            "udotc2": float(u_dot_c[2]), "udotc3": float(u_dot_c[3]),
            "udotc4": float(u_dot_c[4]), "udotc5": float(u_dot_c[5]),
            "tcp_x": float(tcp_pose[0]) if tcp_pose is not None else float("nan"),
            "tcp_y": float(tcp_pose[1]) if tcp_pose is not None else float("nan"),
            "tcp_z": float(tcp_pose[2]) if tcp_pose is not None else float("nan"),
            "tcp_rx": float(tcp_pose[3]) if tcp_pose is not None else float("nan"),
            "tcp_ry": float(tcp_pose[4]) if tcp_pose is not None else float("nan"),
            "tcp_rz": float(tcp_pose[5]) if tcp_pose is not None else float("nan"),
            "cx0": float(corners[0,0]), "cy0": float(corners[0,1]),
            "cx1": float(corners[1,0]), "cy1": float(corners[1,1]),
            "cx2": float(corners[2,0]), "cy2": float(corners[2,1]),
            "cx3": float(corners[3,0]), "cy3": float(corners[3,1]),
            "stable_count": self.stable_cnt, "converged": int(converged),
            "mode": mode,
        })
        self._frame_idx += 1

        return v_ctrl, (err_pos, err_rot), converged, corners, R_cur, t_cur

    def _write_lost_frame_row(self, tcp_pose=None):
        if not self._csv_writer:
            self._frame_idx += 1
            return
        nan = float("nan")
        row = {col: nan for col in CSV_COLUMNS}
        row.update({
            "t": time.time() - self._t0,
            "frame_idx": self._frame_idx,
            "target": self.cur_target.name if self.cur_target else "",
            "mode": self.cfg.controller_mode.upper(),
            "converged": 0,
            "stable_count": self.stable_cnt,
        })
        if tcp_pose is not None:
            for i, k in enumerate(["tcp_x", "tcp_y", "tcp_z", "tcp_rx", "tcp_ry", "tcp_rz"]):
                row[k] = float(tcp_pose[i])
        self._write_csv_row(row)
        self._frame_idx += 1

    def plot_error_history(self):
        from .plotting import plot_error_history
        return plot_error_history(self)

    def plot_trajectory_figure(self):
        from .plotting import plot_trajectory_figure
        return plot_trajectory_figure(self)

    def run(self, pipeline, init_pose, move_acc: float = 8.0):
        if not self.targets:
            print("❌ 未设置目标")
            return

        print("🤖 机器人复位中...")
        self.rtde_c.moveL(init_pose, 1.2, 1.0)
        self._switch_target(0)

        mode = self.cfg.controller_mode.upper()
        is_psmc = "PSMC" in mode

        print("\n" + "="*60)
        print(f"🎮 SO-PBVS | {mode}")
        print(f"   Kp={vec6_to_str(self.cfg.kp, 3)}")
        print(f"   Kd={vec6_to_str(self.cfg.kd, 3)}")
        print(f"   A ={vec6_to_str(self.cfg.accel_limit, 3)}")
        print(f"   Detect stride: {self.cfg.detect_stride} | "
              f"Visualization: {'ON' if self.cfg.enable_visualization else 'OFF'}"
              f" stride={self.cfg.visualization_stride}")
        if is_psmc:
            print(f"   H ={vec6_to_str(self.cfg.proxy_H, 3)} s")
        print("   按 'N' 切换目标 | 'Q' 退出")
        print("="*60)

        self._t0 = time.time()
        self._error_log.clear()
        self._frame_idx = 0
        self._open_csv()

        try:
            while True:
                now = time.time()
                if self.cfg.max_runtime > 0 and (now - self._t0) > self.cfg.max_runtime:
                    print(f"\n⏰ 定时停止")
                    break

                t_start = self.rtde_c.initPeriod()
                frames = pipeline.wait_for_frames()
                cf = frames.get_color_frame()
                if not cf:
                    continue
                img = np.asanyarray(cf.get_data())
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                try:
                    tcp_pose_now = np.array(self.rtde_r.getActualTCPPose())
                    tcp_speed_now = np.array(self.rtde_r.getActualTCPSpeed())
                except Exception:
                    tcp_pose_now = None
                    tcp_speed_now = None

                v_cmd, errs, converged, corners, R_cur, t_cur = self.process_step(
                    gray, tcp_pose=tcp_pose_now, tcp_speed=tcp_speed_now
                )

                if v_cmd is not None:
                    self.rtde_c.speedL(v_cmd, move_acc, self.dt)
                    if self._frame_idx % self.cfg.status_print_interval == 0:
                        status = "CONVERGED" if converged else "Running"
                        ep, er = errs
                        sat_s = "SAT" if self._last_accel_saturated else "---"
                        print(f"\r[{mode}] "
                              f"{self.cur_target.name} | "
                              f"P:{ep*1000:.1f}mm R:{np.rad2deg(er):.1f}°"
                              f" | [{sat_s}] {status}   ", end="")
                else:
                    self.rtde_c.speedStop()
                    if self._frame_idx % self.cfg.status_print_interval == 0:
                        print("\r⚠️ Tag Lost...                                ", end="")

                if (self.cfg.enable_visualization
                        and self._frame_idx % self.cfg.visualization_stride == 0):
                    self._visualize(img, corners, v_cmd, R_cur, t_cur)

                key = cv2.waitKey(1) & 0xFF if self.cfg.enable_visualization else 0
                if key == ord('q'):
                    break
                elif key == ord('n'):
                    self.rtde_c.speedStop()
                    next_idx = (self.cur_target_idx + 1) % len(self.targets)
                    self._switch_target(next_idx)
                    time.sleep(0.5)

                self.rtde_c.waitPeriod(t_start)

        except KeyboardInterrupt:
            print("\n用户中断")
        finally:
            self.rtde_c.speedStop()
            self.rtde_c.stopScript()
            print(f"\n控制结束")
            self._close_csv()
            if self.cfg.enable_final_plots:
                self.plot_error_history()
                self.plot_trajectory_figure()

    def _visualize(self, img, corners, v_cmd, R_cur, t_cur):
        vis = img.copy()
        K = self.estimator.K
        mode = self.cfg.controller_mode.upper()
        is_psmc = "PSMC" in mode
        h_img, w_img = vis.shape[:2]

        if corners is not None and R_cur is not None:
            pts = corners.astype(int)
            for i in range(4):
                cv2.line(vis, tuple(pts[i]), tuple(pts[(i+1) % 4]), (0, 255, 0), 2)
            ori = draw_axis(vis, K, R_cur, t_cur, length=0.03)
            if ori is not None:
                cv2.putText(vis, "Current", tuple(ori+[10, -10]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
                eu = Rotation.from_matrix(R_cur).as_euler('xyz', degrees=True)
                cv2.putText(vis, f"T:[{t_cur[0]:.3f},{t_cur[1]:.3f},{t_cur[2]:.3f}]",
                            tuple(ori+[10, 5]), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(vis, f"R:[{eu[0]:.1f},{eu[1]:.1f},{eu[2]:.1f}]deg",
                            tuple(ori+[10, 18]), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)

        proxy_center_px = None
        if is_psmc and self._proxy_T_cam is not None:
            T_proxy = self._proxy_T_cam
            if T_proxy[2, 3] > 0:
                c3d = get_tag_3d_corners(self.cfg.tag_size, T_proxy)
                if np.all(c3d[:, 2] > 0):
                    c2d = project_3d_to_2d(c3d, K)
                    in_v = np.all((c2d[:, 0] >= -50) & (c2d[:, 0] < w_img+50) &
                                  (c2d[:, 1] >= -50) & (c2d[:, 1] < h_img+50))
                    if in_v:
                        pp = c2d.astype(int)
                        for i in range(4):
                            self._dashed_line(vis, tuple(pp[i]), tuple(pp[(i+1) % 4]),
                                              (0, 200, 255), 2, 8)
                        ori_p = draw_axis_colored(vis, K, T_proxy[:3, :3], T_proxy[:3, 3],
                                                  length=0.025,
                                                  colors=((0, 120, 200), (0, 200, 100), (200, 160, 0)))
                        if ori_p is not None:
                            sat = self._psmc.is_accel_saturated
                            cv2.putText(vis, f"Proxy {'[SAT]' if sat else ''}",
                                        tuple(ori_p+[10, -10]), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                        (0, 100, 255) if sat else (0, 200, 255), 1, cv2.LINE_AA)
                        proxy_center_px = np.mean(c2d, axis=0).astype(int)

        if self.cur_target is not None:
            Td = self._desired_T()
            Rd, td = Td[:3, :3], Td[:3, 3]
            c3d = get_tag_3d_corners(self.cfg.tag_size, Td)
            if np.all(c3d[:, 2] > 0):
                c2d = project_3d_to_2d(c3d, K)
                in_v = np.all((c2d[:, 0] >= -50) & (c2d[:, 0] < w_img+50) &
                              (c2d[:, 1] >= -50) & (c2d[:, 1] < h_img+50))
                if in_v:
                    pd2 = c2d.astype(int)
                    for i in range(4):
                        self._dashed_line(vis, tuple(pd2[i]), tuple(pd2[(i+1) % 4]),
                                          (255, 60, 60), 2, 10)
                    ori_d = draw_axis(vis, K, Rd, td, length=0.03)
                    if ori_d is not None:
                        cv2.putText(vis, "Desired", tuple(ori_d+[10, -10]),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 60, 60), 1, cv2.LINE_AA)
                    if corners is not None:
                        cur_c = np.mean(corners, axis=0).astype(int)
                        des_c = np.mean(pd2, axis=0).astype(int)
                        cv2.arrowedLine(vis, tuple(cur_c), tuple(des_c), (255, 255, 0), 2, tipLength=0.2)

        if proxy_center_px is not None and corners is not None:
            cur_c = np.mean(corners, axis=0).astype(int)
            dist = np.linalg.norm(proxy_center_px.astype(float) - cur_c.astype(float))
            if dist > 3.0:
                cv2.arrowedLine(vis, tuple(proxy_center_px), tuple(cur_c),
                                (0, 255, 128), 2, tipLength=0.25)
                proxy_bn = np.linalg.norm(self._psmc.proxy_offset)
                mid = ((proxy_center_px + cur_c) // 2).tolist()
                cv2.putText(vis, f"|proxy_b|={proxy_bn:.3f}", (mid[0]+5, mid[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 128), 1, cv2.LINE_AA)

        if v_cmd is not None:
            cv2.arrowedLine(vis, (w_img//2, h_img//2),
                            (int(w_img//2 + v_cmd[0]*500), int(h_img//2 + v_cmd[1]*500)),
                            (0, 255, 255), 3, tipLength=0.3)
            cv2.putText(vis, f"Vel:{np.linalg.norm(v_cmd[:3])*1000:.1f}mm/s",
                        (10, h_img-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(vis, f"[{mode}] {self.cur_target.name if self.cur_target else ''}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        hud_y = 45
        if is_psmc:
            sat = self._last_accel_saturated
            cv2.putText(vis, f"Accel-SAT: {'YES' if sat else 'no '}",
                        (10, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 0, 255) if sat else (0, 200, 80), 1, cv2.LINE_AA)
            cv2.putText(vis,
                        f"|s_p|={np.linalg.norm(self._psmc.proxy_position):.4f}  "
                        f"|proxy_b|={np.linalg.norm(self._psmc.proxy_offset):.4f}",
                        (10, hud_y+18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 0), 1, cv2.LINE_AA)
        else:
            sat = self._last_accel_saturated
            cv2.putText(vis, f"Accel-SAT: {'YES' if sat else 'no '}",
                        (10, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 0, 255) if sat else (0, 200, 80), 1, cv2.LINE_AA)

        cv2.imshow("SO-PBVS View", vis)

    def _dashed_line(self, img, pt1, pt2, color, thickness, dash_len):
        dist = np.linalg.norm(np.array(pt1) - np.array(pt2))
        dashes = max(int(dist / dash_len), 1)
        for i in range(dashes):
            s = (int(pt1[0] + (pt2[0]-pt1[0]) * i / dashes),
                 int(pt1[1] + (pt2[1]-pt1[1]) * i / dashes))
            e = (int(pt1[0] + (pt2[0]-pt1[0]) * (i+0.5) / dashes),
                 int(pt1[1] + (pt2[1]-pt1[1]) * (i+0.5) / dashes))
            cv2.line(img, s, e, color, thickness)
