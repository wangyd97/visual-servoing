from dataclasses import dataclass

import numpy as np

from .Mathematic import to_vec3, to_vec6, vec6_from_pos_rot


@dataclass
class PBVSConfig:
    tag_size: float = 0.05
    detect_stride: int = 1
    apriltag_nthreads: int = 1
    apriltag_quad_decimate: float = 1.0
    apriltag_quad_sigma: float = 0.4
    enable_visualization: bool = True
    visualization_stride: int = 1

    pos_threshold: float = 0.005
    rot_threshold: float = 0.02
    stable_frames: int = 10
    slow_after_convergence: bool = False
    convergence_slowdown_frames: int = 5
    convergence_velocity_scale: float = 0.1

    max_runtime: float = 0.0

    max_linear_vel: float = 0.80
    max_angular_vel: float = 0.80
    enable_feature_lowpass: bool = True
    feature_lowpass_tau: float = 0.02
    use_commanded_tcp_pose_estimate: bool = False
    enable_Rc_lowpass: bool = False
    Rc_lowpass_tau: float = 0.01

    controller_mode: str = "SOPDPSMC"

    # R6 parameter order: [x, y, z, rx, ry, rz]
    kp: object = 1.0
    kd: object = 2.0

    accel_limit: object = None
    accel_limit_pos: object = float("inf")
    accel_limit_rot: object = float("inf")

    proxy_H: object = None
    proxy_H_pos: float = 0.8
    proxy_H_rot: float = 0.8

    plot_save_path: str = "pbvs_error_plot.png"
    trajectory_plot_save_path: str = ""
    log_save_path: str = "log.csv"
    enable_memory_log: bool = True
    enable_final_plots: bool = True
    status_print_interval: int = 1

    def __post_init__(self):
        self.detect_stride = max(1, int(self.detect_stride))
        self.apriltag_nthreads = max(1, int(self.apriltag_nthreads))
        self.apriltag_quad_decimate = max(1.0, float(self.apriltag_quad_decimate))
        self.apriltag_quad_sigma = max(0.0, float(self.apriltag_quad_sigma))
        self.visualization_stride = max(1, int(self.visualization_stride))
        self.slow_after_convergence = bool(self.slow_after_convergence)
        self.convergence_slowdown_frames = max(0, int(self.convergence_slowdown_frames))
        self.convergence_velocity_scale = float(np.clip(self.convergence_velocity_scale, 0.0, 1.0))
        self.enable_feature_lowpass = bool(self.enable_feature_lowpass)
        self.feature_lowpass_tau = max(0.0, float(self.feature_lowpass_tau))
        self.use_commanded_tcp_pose_estimate = bool(self.use_commanded_tcp_pose_estimate)
        self.enable_Rc_lowpass = bool(self.enable_Rc_lowpass)
        self.Rc_lowpass_tau = max(0.0, float(self.Rc_lowpass_tau))
        self.enable_memory_log = bool(self.enable_memory_log)
        self.enable_final_plots = bool(self.enable_final_plots)
        self.status_print_interval = max(1, int(self.status_print_interval))

        self.kp = to_vec6(self.kp, "kp")
        self.kd = to_vec6(self.kd, "kd")

        if self.accel_limit is None:
            default_accel = float("inf") if self.controller_mode.upper() == "SOPD" else 1.5
            accel_pos_value = default_accel if self.accel_limit_pos is None else self.accel_limit_pos
            accel_rot_value = default_accel if self.accel_limit_rot is None else self.accel_limit_rot
            accel_pos = to_vec3(accel_pos_value, "accel_limit_pos")
            accel_rot = to_vec3(accel_rot_value, "accel_limit_rot")
            self.accel_limit = np.concatenate([accel_pos, accel_rot])
        else:
            self.accel_limit = to_vec6(self.accel_limit, "accel_limit")
        self.accel_limit_pos = self.accel_limit[:3].copy()
        self.accel_limit_rot = self.accel_limit[3:].copy()

        if self.proxy_H is None:
            self.proxy_H = vec6_from_pos_rot(self.proxy_H_pos, self.proxy_H_rot)
        else:
            self.proxy_H = to_vec6(self.proxy_H, "proxy_H")
            self.proxy_H_pos = float(np.mean(self.proxy_H[:3]))
            self.proxy_H_rot = float(np.mean(self.proxy_H[3:]))


@dataclass
class TargetPose:
    name: str
    desired_rotation: np.ndarray
    desired_translation: np.ndarray

    def __post_init__(self):
        self.T_des = np.eye(4)
        self.T_des[:3, :3] = self.desired_rotation
        self.T_des[:3, 3] = self.desired_translation
