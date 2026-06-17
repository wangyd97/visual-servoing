import csv
import time
from pathlib import Path
from typing import List, Optional

import cv2 # pyright: ignore[reportMissingImports]
import numpy as np
from rtde_control import RTDEControlInterface as RTDEControl # pyright: ignore[reportMissingImports]
from rtde_receive import RTDEReceiveInterface as RTDEReceive # pyright: ignore[reportMissingImports]

from .config import PBVSConfig, TargetPose
from .geometry import (
    compute_L,
    compute_b,
    compute_N,
    get_tag_3d_corners,
    inv_T,
    project_3d_to_2d,
)
from .psmc import PSMCPDProxy
from .Mathematic import (
    euler_xyz_from_matrix,
    matrix_from_quat,
    matrix_from_rotvec,
    quat_from_matrix,
    rotvec_from_matrix,
)
from .vision import AprilTagEstimator, draw_axis, draw_axis_colored


class PBVSController:
    def __init__(self, robot_ip: str, intrinsics,
                 hand_eye_calib: np.ndarray,
                 config: PBVSConfig = None):
        self.cfg = config or PBVSConfig()
        self.e_T_c = hand_eye_calib
        self.e_R_c = self.e_T_c[0:3, 0:3]
        # e_T_c contains e_p_ce = p_c - p_e expressed in frame e.
        # Use p_ec = p_e - p_c to match the manuscript convention.
        self.e_p_ec = -self.e_T_c[:3, 3]
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
        self._error_log: list = []
        self._t0: float = 0.0
        self._proxy_T_cam: Optional[np.ndarray] = None

        self._frame_idx = 0

        self._last_u_c = np.zeros(6)
        self._last_u_o = np.zeros(6)
        self._last_R_base_cam: Optional[np.ndarray] = None
        self._last_s_for_diff: Optional[np.ndarray] = None
        self._last_s_diff_time: Optional[float] = None
        self._s_dot_diff_filtered = np.zeros(6)
        self._u_o_filtered = np.zeros(6)
        self._last_u_o_for_diff: Optional[np.ndarray] = None
        self._last_u_o_diff_time: Optional[float] = None
        self._u_dot_o_filtered = np.zeros(6)
        self._last_detection = None
        self._last_c_q_oc: Optional[np.ndarray] = None
        self._last_s = np.full(6, float("nan"))
        self._last_s_d = np.full(6, float("nan"))
        self._last_s_dot = np.zeros(6)
        self._s_lowpass: Optional[np.ndarray] = None
        self._tcp_pose_cmd_est: Optional[np.ndarray] = None
        self._R_base_cam_lowpass: Optional[np.ndarray] = None

    def set_targets(self, targets: List[TargetPose]):
        self.targets = targets

    def _switch_target(self, idx: int) -> bool:
        if idx >= len(self.targets):
            return False
        self.cur_target_idx = idx
        self.cur_target = self.targets[idx]
        self._active_T_des = self.cur_target.T_des.copy()
        self._reset_controller_state()
        return True

    def _reset_controller_state(self):
        self.stable_cnt = 0
        self._u_c_integrated = np.zeros(6)
        self._psmc.reset()
        self._proxy_T_cam = None
        self._last_u_c = np.zeros(6)
        self._last_u_o = np.zeros(6)
        self._last_R_base_cam = None
        self._last_s_for_diff = None
        self._last_s_diff_time = None
        self._s_dot_diff_filtered = np.zeros(6)
        self._u_o_filtered = np.zeros(6)
        self._last_u_o_for_diff = None
        self._last_u_o_diff_time = None
        self._u_dot_o_filtered = np.zeros(6)
        self._last_detection = None
        self._last_c_q_oc = None
        self._last_s = np.full(6, float("nan"))
        self._last_s_d = np.full(6, float("nan"))
        self._last_s_dot = np.zeros(6)
        self._s_lowpass = None
        self._tcp_pose_cmd_est = None
        self._R_base_cam_lowpass = None
        self._last_accel_saturated = False

    def _desired_T(self) -> np.ndarray:
        if self._active_T_des is not None:
            return self._active_T_des
        if self.cur_target is not None:
            return self.cur_target.T_des
        return np.eye(4)

    @staticmethod
    def _nearer_quat(q_ref: np.ndarray, q: np.ndarray) -> np.ndarray:
        q_ref = np.asarray(q_ref, dtype=float).reshape(4)
        q = np.asarray(q, dtype=float).reshape(4).copy()
        if float(q_ref @ q) < 0.0:
            q = -q
        return q

    def _make_c_q_oc_continuous(self, c_q_oc: np.ndarray, update: bool = True) -> np.ndarray:
        c_q_oc = np.asarray(c_q_oc, dtype=float).reshape(4).copy()
        if self._last_c_q_oc is not None:
            c_q_oc = self._nearer_quat(self._last_c_q_oc, c_q_oc)
        if update:
            self._last_c_q_oc = c_q_oc.copy()
        return c_q_oc

    def _compute_feature_error(self, T_current: np.ndarray,
                               R_base_cam: np.ndarray):
        T_des = self._desired_T()
        q_des = quat_from_matrix(T_des[:3, :3])
        c_q_oc = quat_from_matrix(T_current[:3, :3])
        c_q_oc = self._make_c_q_oc_continuous(c_q_oc)
        # print(f"c_q_oc (quat): {c_q_oc}")
        c_p_oc = T_current[:3, 3]
        s_d = np.concatenate([T_des[:3, 3], q_des[1:4]])
        s = np.concatenate([c_p_oc, c_q_oc[1:4]])
        L = compute_L(c_p_oc, c_q_oc, R_base_cam)

        try:
            L_inv = np.linalg.inv(L)
        except np.linalg.LinAlgError:
            L_inv = np.linalg.pinv(L)

        return s, s_d, c_q_oc, c_p_oc, L, L_inv

    def _filter_feature_s(self, s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=float).reshape(6)
        if not self.cfg.enable_feature_lowpass or self.cfg.feature_lowpass_tau <= 0.0:
            self._s_lowpass = s.copy()
            return s.copy()
        if self._s_lowpass is None:
            self._s_lowpass = s.copy()
            return s.copy()

        tau = self.cfg.feature_lowpass_tau
        beta = tau / (tau + self.dt)
        s_filtered = beta * self._s_lowpass + (1.0 - beta) * s
        qv_norm = float(np.linalg.norm(s_filtered[3:6]))
        if qv_norm > 1.0:
            s_filtered[3:6] /= qv_norm + 1e-12
        self._s_lowpass = s_filtered.copy()
        return s_filtered

    @staticmethod
    def _feature_to_pose_parts(s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        c_p_oc = np.asarray(s[:3], dtype=float).copy()
        qv = np.asarray(s[3:6], dtype=float).copy()
        qv_norm_sq = float(qv @ qv)
        if qv_norm_sq > 1.0:
            qv /= np.sqrt(qv_norm_sq + 1e-12)
            qv_norm_sq = float(qv @ qv)
        q0 = np.sqrt(max(0.0, 1.0 - qv_norm_sq))
        return c_p_oc, np.array([q0, qv[0], qv[1], qv[2]], dtype=float)

    def _compute_s_dot_by_difference(self, s: np.ndarray) -> np.ndarray:
        """Eq. (42a): obtain s_dot from filtered visual-feature difference."""
        s = np.asarray(s, dtype=float).reshape(6)
        now = time.perf_counter()
        if self._last_s_for_diff is None or self._last_s_diff_time is None:
            self._s_dot_diff_filtered = np.zeros(6)
        else:
            dt = max(now - self._last_s_diff_time, 1e-4)
            s_dot_raw = (s - self._last_s_for_diff) / dt
            tau = 0.10
            beta = tau / (tau + dt)
            self._s_dot_diff_filtered = (
                beta * self._s_dot_diff_filtered
                + (1.0 - beta) * s_dot_raw
            )
        self._last_s_for_diff = s.copy()
        self._last_s_diff_time = now
        return self._s_dot_diff_filtered.copy()

    def _filter_u_o(self, u_o_raw: np.ndarray) -> np.ndarray:
        u_o = u_o_raw.copy()
        tau = 0.05
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

    def _compute_u_dot_c(self, alpha_c: np.ndarray, c_q_oc: np.ndarray,
                  c_p_oc: np.ndarray, L: np.ndarray,
                  L_inv: np.ndarray, u_c: np.ndarray,
                  u_o: np.ndarray, u_dot_o: np.ndarray,
                  N: np.ndarray, R_base_cam: np.ndarray) -> np.ndarray:
        """Eq. (25): u_dot_c = L^{-1}(alpha_c - b - N u_dot_o)."""
        b = compute_b(c_p_oc, c_q_oc, u_c, u_o, R_base_cam)
        # b = np.zeros(6)
        return L_inv @ (alpha_c - b - N @ u_dot_o)

    def _clip_direct_camera_acceleration(self, u_dot_c: np.ndarray) -> tuple[np.ndarray, bool]:
        limits = np.asarray(self.cfg.accel_limit, dtype=float).reshape(6)
        if np.all(np.isinf(limits)):
            return u_dot_c, False

        clipped = np.clip(u_dot_c, -limits, limits)
        saturated = bool(np.any(np.abs(clipped - u_dot_c) > 1e-12))
        return clipped, saturated

    def _proxy_feature_to_T_cam(self, proxy_pos: np.ndarray) -> Optional[np.ndarray]:
        try:
            t_proxy = proxy_pos[:3]
            qv_proxy = proxy_pos[3:6]
            qv_ns = np.dot(qv_proxy, qv_proxy)
            if qv_ns > 1.0:
                qv_proxy = qv_proxy / np.sqrt(qv_ns + 1e-12)
                qv_ns = np.dot(qv_proxy, qv_proxy)
            q0_proxy = np.sqrt(max(0.0, 1.0 - qv_ns))
            q_s = np.array([q0_proxy, qv_proxy[0], qv_proxy[1], qv_proxy[2]])
            T_err_proxy = np.eye(4)
            T_err_proxy[:3, :3] = matrix_from_quat(q_s)
            T_err_proxy[:3, 3] = t_proxy
            T_proxy_cam = inv_T(T_err_proxy) @ self._desired_T()
            return T_proxy_cam if T_proxy_cam[2, 3] > 0 else None
        except Exception:
            return None

    def _compute_control(self, T_current: np.ndarray,
                         R_base_cam: np.ndarray,
                         s_dot_d: np.ndarray = None):
        if s_dot_d is None:
            s_dot_d = np.zeros(6)

        s, s_d, c_q_oc, c_p_oc, L, L_inv = self._compute_feature_error(
            T_current, R_base_cam
        )
        s = self._filter_feature_s(s)
        c_p_oc, c_q_oc = self._feature_to_pose_parts(s)
        L = compute_L(c_p_oc, c_q_oc, R_base_cam)
        try:
            L_inv = np.linalg.inv(L)
        except np.linalg.LinAlgError:
            L_inv = np.linalg.pinv(L)
        
        e = s_d - s
        u_c = self._last_u_c.copy()
        
        N = compute_N(c_q_oc, R_base_cam)
        try:
            N_inv = np.linalg.inv(N)
        except np.linalg.LinAlgError:
            N_inv = np.linalg.pinv(N)
        s_dot_by_difference = self._compute_s_dot_by_difference(s)
        u_o = self._filter_u_o(N_inv @ (s_dot_by_difference - L @ u_c)) # This part is calculated by eq. (26)
        # u_o = N_inv @ (s_dot_by_difference - L @ u_c)
        u_dot_o = self._compute_u_dot_o_by_difference(u_o)
        u_o = np.zeros(6)  
        u_dot_o = np.zeros(6)  
        s_dot_by_interaction_matrix = L @ u_c + N @ u_o
        K = np.diag(self.cfg.kp)
        B = np.diag(self.cfg.kd)
        # Eq. (42g): uo = N^{-1}(s_dot - L uc).
        edot = s_dot_d - s_dot_by_interaction_matrix
        self._last_s = s.copy()
        self._last_s_d = s_d.copy()
        self._last_s_dot = s_dot_by_interaction_matrix.copy()
        self._last_u_o = u_o.copy()
        mode = self.cfg.controller_mode.upper()

        if mode == "SOPD":
            # Eq. (24): alpha_c = K(s_d - s) + B(s_dot_d - s_dot).
            alpha_c = K @ e + B @ edot
            # Eq. (25): u_dot_c = L^-1(alpha_c - b - N u_dot_o).
            u_dot_c = self._compute_u_dot_c(
                alpha_c, c_q_oc, c_p_oc, L, L_inv, u_c, u_o, u_dot_o, N, R_base_cam
            )
            u_dot_c, self._last_accel_saturated = self._clip_direct_camera_acceleration(u_dot_c)

        elif "PSMC" in mode:
            # Eq. (42h): b = b(s, qc, uc, uo).
            b = compute_b(c_p_oc, c_q_oc, u_c, u_o, R_base_cam)
            # b = np.zeros(6)  
            # Eq. (42i)-(42n): proxy update and projection of u_dot_c*.
            u_dot_c = self._psmc.compute(
                s=s,
                s_dot=s_dot_by_interaction_matrix,
                s_d=s_d,
                s_dot_d=s_dot_d,
                L=L,
                L_inv=L_inv,
                b=b,
                N=N,
                u_dot_o=u_dot_o,
                u_c=u_c
            )
            self._last_accel_saturated = self._psmc.is_accel_saturated

        else:
            raise ValueError(f"Unsupported controller mode: {mode}")

        # Eq. (42o): uc = uc,prv + T u_dot_c.
        self._u_c_integrated += u_dot_c * self.dt

        return self._u_c_integrated, u_dot_c

    def _u_c_to_tcp_twist_base(self, u_c: np.ndarray, tcp_pose: np.ndarray) -> np.ndarray:
        R_base_tcp = matrix_from_rotvec(tcp_pose[3:])
        b_p_ec = R_base_tcp @ self.e_p_ec

        v_c_base = u_c[:3]
        omega_base = u_c[3:]

        v_tcp_base = v_c_base + np.cross(omega_base, b_p_ec)

        return np.concatenate([v_tcp_base, omega_base])

    def _control_tcp_pose(self, actual_pose: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if not self.cfg.use_commanded_tcp_pose_estimate:
            return actual_pose
        if actual_pose is None:
            return None if self._tcp_pose_cmd_est is None else self._tcp_pose_cmd_est.copy()
        actual_pose = np.asarray(actual_pose, dtype=float).reshape(6)
        if self._tcp_pose_cmd_est is None:
            self._tcp_pose_cmd_est = actual_pose.copy()
        return self._tcp_pose_cmd_est.copy()

    def _advance_commanded_tcp_pose(self, v_cmd: Optional[np.ndarray]):
        if not self.cfg.use_commanded_tcp_pose_estimate or self._tcp_pose_cmd_est is None:
            return
        if v_cmd is None:
            return

        v_cmd = np.asarray(v_cmd, dtype=float).reshape(6)
        pose = self._tcp_pose_cmd_est.copy()
        pose[:3] += v_cmd[:3] * self.dt

        R_base_tcp = matrix_from_rotvec(pose[3:])
        R_next = matrix_from_rotvec(v_cmd[3:] * self.dt) @ R_base_tcp
        pose[3:] = rotvec_from_matrix(R_next)
        self._tcp_pose_cmd_est = pose

    def _filter_R_base_cam(self, R_base_cam: np.ndarray) -> np.ndarray:
        if not self.cfg.enable_Rc_lowpass or self.cfg.Rc_lowpass_tau <= 0.0:
            self._R_base_cam_lowpass = R_base_cam.copy()
            return R_base_cam.copy()
        if self._R_base_cam_lowpass is None:
            self._R_base_cam_lowpass = R_base_cam.copy()
            return R_base_cam.copy()

        alpha = self.dt / (self.cfg.Rc_lowpass_tau + self.dt)
        R_prev = self._R_base_cam_lowpass
        rel_rotvec = rotvec_from_matrix(R_base_cam @ R_prev.T)
        R_filtered = matrix_from_rotvec(alpha * rel_rotvec) @ R_prev
        self._R_base_cam_lowpass = R_filtered.copy()
        return R_filtered

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
                     log_tcp_pose: np.ndarray = None):
        T_cur, corners, R_cur, t_cur = self._detect_or_reuse_tag(gray_img)
        
        if T_cur is None:
            self._last_u_c = np.zeros(6)
            self._last_s_for_diff = None
            self._s_dot_diff_filtered = np.zeros(6)
            self._last_c_q_oc = None
            self._frame_idx += 1
            return None, None, False, None, None, None

        if tcp_pose is None:
            actual_pose = np.array(self.rtde_r.getActualTCPPose())
        else:
            actual_pose = np.asarray(tcp_pose, dtype=float)
        logged_pose = (
            np.asarray(log_tcp_pose, dtype=float)
            if log_tcp_pose is not None else actual_pose
        )
        R_base_tcp = matrix_from_rotvec(actual_pose[3:])
        R_base_cam = R_base_tcp @ self.e_R_c
        R_base_cam = self._filter_R_base_cam(R_base_cam)
        self._last_R_base_cam = R_base_cam.copy()

        u_c, u_dot_c = self._compute_control(T_cur, R_base_cam)
        self._last_u_c = u_c.copy()

        T_err = self._desired_T() @ inv_T(T_cur)
        t_err_vec = T_err[:3, 3]
        err_pos = float(np.linalg.norm(t_err_vec))
        err_rot = float(np.linalg.norm(rotvec_from_matrix(T_err[:3, :3])))
        r_euler = euler_xyz_from_matrix(T_err[:3, :3], degrees=True)

        mode = self.cfg.controller_mode.upper()
        self._proxy_T_cam = None
        if "PSMC" in mode:
            self._proxy_T_cam = self._proxy_feature_to_T_cam(self._psmc.proxy_position)

        v_ctrl = self._u_c_to_tcp_twist_base(u_c, actual_pose)
        # v_ctrl = np.array([0.005, 0, 0, 0, 0, 0])  # temporarily disable motion for better debugging
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
                "udotc0": float(u_dot_c[0]), "udotc1": float(u_dot_c[1]), "udotc2": float(u_dot_c[2]),
                "udotc3": float(u_dot_c[3]), "udotc4": float(u_dot_c[4]), "udotc5": float(u_dot_c[5]),
                "target": self.cur_target.name,
                "cx0": float(corners[0,0]), "cy0": float(corners[0,1]),
                "cx1": float(corners[1,0]), "cy1": float(corners[1,1]),
                "cx2": float(corners[2,0]), "cy2": float(corners[2,1]),
                "cx3": float(corners[3,0]), "cy3": float(corners[3,1]),
                "dcx0": float(des_corners_px[0,0]), "dcy0": float(des_corners_px[0,1]),
                "dcx1": float(des_corners_px[1,0]), "dcy1": float(des_corners_px[1,1]),
                "dcx2": float(des_corners_px[2,0]), "dcy2": float(des_corners_px[2,1]),
                "dcx3": float(des_corners_px[3,0]), "dcy3": float(des_corners_px[3,1]),
                "tcp_x": float(logged_pose[0]),
                "tcp_y": float(logged_pose[1]),
                "tcp_z": float(logged_pose[2]),
            })
            row = self._error_log[-1]
            row.update({
                "frame_idx": int(self._frame_idx),
                "controller_mode": mode,
                "accel_saturated": int(self._last_accel_saturated),
                "tcp_x": float(logged_pose[0]),
                "tcp_y": float(logged_pose[1]),
                "tcp_z": float(logged_pose[2]),
                "tcp_rx": float(logged_pose[3]),
                "tcp_ry": float(logged_pose[4]),
                "tcp_rz": float(logged_pose[5]),
                "ctrl_tcp_x": float(actual_pose[0]),
                "ctrl_tcp_y": float(actual_pose[1]),
                "ctrl_tcp_z": float(actual_pose[2]),
                "ctrl_tcp_rx": float(actual_pose[3]),
                "ctrl_tcp_ry": float(actual_pose[4]),
                "ctrl_tcp_rz": float(actual_pose[5]),
            })
            is_psmc = "PSMC" in mode
            proxy_position = self._psmc.proxy_position if is_psmc else np.full(6, float("nan"))
            proxy_star = self._psmc.s_p_star.copy() if is_psmc else np.full(6, float("nan"))
            proxy_offset = self._psmc.proxy_offset if is_psmc else np.full(6, float("nan"))
            for i in range(6):
                row[f"s{i}"] = float(self._last_s[i])
                row[f"sd{i}"] = float(self._last_s_d[i])
                row[f"sdot{i}"] = float(self._last_s_dot[i])
                row[f"uc{i}"] = float(u_c[i])
                row[f"udotc{i}"] = float(u_dot_c[i])
                row[f"uo{i}"] = float(self._last_u_o[i])
                row[f"sp{i}"] = float(proxy_position[i])
                row[f"spstar{i}"] = float(proxy_star[i])
                row[f"proxy_offset{i}"] = float(proxy_offset[i])

        self._frame_idx += 1

        return v_ctrl, (err_pos, err_rot), converged, corners, R_cur, t_cur

    def plot_error_history(self):
        from .plotting import plot_error_history
        return plot_error_history(self)

    def plot_trajectory_figure(self):
        from .plotting import plot_trajectory_figure
        return plot_trajectory_figure(self)

    def save_log_csv(self):
        if not self._error_log:
            print("No log data to save.")
            return
        if not self.cfg.log_save_path:
            return

        log_path = Path(self.cfg.log_save_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(self._error_log[0].keys())
        for row in self._error_log[1:]:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)

        with log_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._error_log)
        print(f"Log saved: {log_path}")

    def run(self, pipeline, init_pose, move_acc: float = 10.0):
        if not self.targets:
            print("Nothing detected!")
            return

        self.rtde_c.moveL(init_pose, 1.2, 1.0)
        self._switch_target(0)

        mode = self.cfg.controller_mode.upper()

        self._t0 = time.time()
        self._error_log.clear()
        self._frame_idx = 0

        try:
            while True:
                now = time.time()
                if self.cfg.max_runtime > 0 and (now - self._t0) > self.cfg.max_runtime:
                    print("\nMax runtime reached; stopping.")
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
                except Exception:
                    tcp_pose_now = None
                tcp_pose_ctrl = self._control_tcp_pose(tcp_pose_now)

                v_cmd, errs, converged, corners, R_cur, t_cur = self.process_step(
                    gray, tcp_pose=tcp_pose_ctrl, log_tcp_pose=tcp_pose_now
                )

                if v_cmd is not None:
                    self.rtde_c.speedL(v_cmd, move_acc, self.dt)
                    self._advance_commanded_tcp_pose(v_cmd)
                    if self._frame_idx % self.cfg.status_print_interval == 0:
                        status = "CONVERGED" if converged else "Running"
                        ep, er = errs
                        sat_s = "SAT" if self._last_accel_saturated else "---"
                        print(f"\r[{mode}] "
                              f"{self.cur_target.name} | "
                              f"P:{ep*1000:.1f}mm R:{np.rad2deg(er):.1f}deg"
                              f" | [{sat_s}] {status}   ", end="", flush=True)
                else:
                    self.rtde_c.speedStop()
                    if self._frame_idx % self.cfg.status_print_interval == 0:
                        print("\rTag lost...                                ", end="", flush=True)

                if (self.cfg.enable_visualization
                        and self._frame_idx % self.cfg.visualization_stride == 0):
                    self._visualize(img, corners, v_cmd, R_cur, t_cur)

                key = cv2.waitKey(1) & 0xFF if self.cfg.enable_visualization else 0
                if key == ord('q'):
                    break

                self.rtde_c.waitPeriod(t_start)

        except KeyboardInterrupt:
            print("\nInterrupted by user.")
        finally:
            self.rtde_c.speedStop()
            self.rtde_c.stopScript()
            print("\nControl stopped.")
            self.save_log_csv()
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
                eu = euler_xyz_from_matrix(R_cur, degrees=True)
                cv2.putText(vis, f"T:[{t_cur[0]:.3f},{t_cur[1]:.3f},{t_cur[2]:.3f}]",
                            tuple(ori+[10, 5]), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(vis, f"R:[{eu[0]:.1f},{eu[1]:.1f},{eu[2]:.1f}]deg",
                            tuple(ori+[10, 18]), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)

            if self._last_R_base_cam is not None:
                v_o_base = self._last_u_o[:3]
                v_o_norm = float(np.linalg.norm(v_o_base))
                if v_o_norm > 1e-4:
                    center_3d = np.asarray(t_cur, dtype=float).reshape(3)
                    v_o_cam_dir = self._last_R_base_cam.T @ (v_o_base / v_o_norm)
                    arrow_len_m = 0.06
                    arrow_3d = np.vstack([
                        center_3d,
                        center_3d + arrow_len_m * v_o_cam_dir,
                    ])
                    if np.all(arrow_3d[:, 2] > 0):
                        arrow_2d = project_3d_to_2d(arrow_3d, K).astype(int)
                        p0 = tuple(arrow_2d[0])
                        p1 = tuple(arrow_2d[1])
                        cv2.arrowedLine(vis, p0, p1, (255, 0, 255), 2, tipLength=0.25)
                        cv2.putText(vis, f"u_o {v_o_norm:.3f}m/s",
                                    (p1[0] + 6, p1[1] - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                                    (255, 0, 255), 1, cv2.LINE_AA)

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
