import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from e1.config import PBVSConfig, TargetPose
    from e1.Mathematic import matrix_from_quat
else:
    from .config import PBVSConfig, TargetPose
    from .Mathematic import matrix_from_quat


ENABLE_VISUALIZATION = True
ENABLE_MEMORY_LOG = True
ENABLE_FINAL_PLOTS = True
STATUS_PRINT_INTERVAL = 30
APRILTAG_NTHREADS = 2
APRILTAG_QUAD_DECIMATE = 2.0

ROBOT_IP = "10.31.17.47"


def hand_eye_matrix() -> np.ndarray:
    # e_T_c describes the camera frame expressed in the end-effector frame.

    return np.array([
        [0.0074, -0.9994, -0.0329, -0.0715],
        [1.0000,  0.0074, -0.0010, -0.0328],
        [0.0012, -0.0329,  0.9995,  0.0499],
        [0.0000,  0.0000,  0.0000,  1.0000],
    ])


def method_params(method: str) -> dict:
    method = method.upper()
    presets = {
       
        "RS": dict(
            controller_mode="SOPD",
            kp=150*np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            kd=30*np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            max_linear_vel=float("inf"),
            max_angular_vel=float("inf"),
        ),
        "RSC": dict(
            controller_mode="SOPD",
            kp=150*np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            kd=30*np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            accel_limit_pos=8 * np.array([1.0, 1.0, 1.0]),
            accel_limit_rot=8 * np.array([1.0, 1.0, 1.0]),
            max_linear_vel=float("inf"),
            max_angular_vel=float("inf"),
        ),

        "RL": dict(
            controller_mode="SOPD",
            kp=150*np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            kd=70*np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            max_linear_vel=float("inf"),
            max_angular_vel=float("inf"),
        ),
        "P": dict(
            controller_mode="SOPDPSMC",
            kp=150 * np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            kd=30 * np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            proxy_H= 0.5 * np.array([1,1,1,1,1,1]),
            accel_limit_pos= 8 * np.array([1.0, 1.0, 1.0]),
            accel_limit_rot= 8 * np.array([1.0, 1.0, 1.0]),
            max_linear_vel=float("inf"),
            max_angular_vel=float("inf"),
        ),
    }
    if method not in presets:
        raise ValueError(f"Unknown method: {method}")
    return presets[method]


def output_stem(method: str) -> str:
    method = method.upper()
    return "SOPDPSMC" if method == "P" else f"SOPD_{method}"


def build_config(method: str, runtime: float) -> PBVSConfig:
    project_dir = Path(__file__).resolve().parent
    figure_dir = project_dir / "figures"
    data_dir = project_dir / "data"
    file_stem = output_stem(method)
    params = method_params(method)


    return PBVSConfig(
        tag_size=0.08,
        detect_stride=1,
        apriltag_nthreads=APRILTAG_NTHREADS,
        apriltag_quad_decimate=APRILTAG_QUAD_DECIMATE,
        enable_visualization=ENABLE_VISUALIZATION,
        visualization_stride=1,
        pos_threshold=0.002,
        rot_threshold=0.01,
        slow_after_convergence=False,
        max_runtime=runtime,
        plot_save_path=str(figure_dir / f"{file_stem}.png"),
        trajectory_plot_save_path=str(figure_dir / f"{file_stem}_trajectory.png"),
        log_save_path=str(data_dir / f"log_{method.upper()}_exp1_2.csv"),
        enable_memory_log=ENABLE_MEMORY_LOG,
        enable_final_plots=ENABLE_FINAL_PLOTS,
        status_print_interval=STATUS_PRINT_INTERVAL,
        **params,
    )


def build_targets():
    # desired_quaternion = np.array([0.970, 0.171, 0.171, 0.030])  # q=[qw,qx,qy,qz]
    desired_quaternion = np.array([1, 0, 0, 0])  # q=[qw,qx,qy,qz]
    base_rotation = matrix_from_quat(desired_quaternion)
    base_translation = np.array([0.00, 0.00, 0.25])
    return [
        TargetPose(
            name="Large_error_start",
            desired_rotation=base_rotation,
            desired_translation=base_translation,
        ),
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Experiment I/II/III visual servoing comparison."
    )
    parser.add_argument(
        "--method",
        choices=["RS", "RSC", "RL", "P"],
        default="RL",
        help="Controller preset to run.",
    )
    parser.add_argument(
        "--runtime",
        type=float,
        default=None,
        help="Recording duration in seconds. Use 0 for manual stop.",
    )
    args = parser.parse_args()
    runtime = args.runtime
    if runtime is None:
        runtime = 8.0

    if __package__ in (None, ""):
        from e1.vision import init_realsense
    else:
        from .vision import init_realsense

    pipeline, intr = init_realsense()
    intrinsics_params = (intr.fx, intr.fy, intr.ppx, intr.ppy)

    if __package__ in (None, ""):
        from e1.controller import PBVSController
    else:
        from .controller import PBVSController
    controller = PBVSController(
        robot_ip=ROBOT_IP,
        intrinsics=intrinsics_params,
        hand_eye_calib=hand_eye_matrix(),
        config=build_config(args.method, runtime),
    )
    controller.set_targets(build_targets())

    init_pose = np.array([
        -0.20588217, -0.05717437,  0.420001008,
        -2.50455863, -1.8897531,  -0.01089382,
    ])

    # init_pose = np.array([
    #     -0.25588217, -0.15717437,  0.250001008,
    #     -2.50455863, -1.8897531,  -0.01089382,
    # ])
    
    # init_pose = np.array([
    #     -0.25588217, -0.15717437,  0.250001008,
    #     3.10924706, -0.43501290,  0.01221673,
    # ])
    controller.run(pipeline, init_pose)


if __name__ == "__main__":
    main()
