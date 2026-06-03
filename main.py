#updated on 2026-06-02:
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from e1.config import PBVSConfig, TargetPose
    from e1.controller import PBVSController
    from e1.vision import init_realsense
else:
    from .config import PBVSConfig, TargetPose
    from .controller import PBVSController
    from .vision import init_realsense


ENABLE_VISUALIZATION = True
ENABLE_CSV_LOGGING = True
ENABLE_MEMORY_LOG = True
ENABLE_FINAL_PLOTS = True
STATUS_PRINT_INTERVAL = 30
APRILTAG_NTHREADS = 2
APRILTAG_QUAD_DECIMATE = 2.0

ROBOT_IP = "10.31.17.25"


def hand_eye_matrix() -> np.ndarray:
    return np.array([
        [0.0074, -0.9994, -0.0329, -0.0715],
        [1.0000,  0.0074, -0.0010, -0.0328],
        [0.0012, -0.0329,  0.9995,  0.0499],
        [0.0000,  0.0000,  0.0000,  1.0000],
    ])


def method_params(method: str) -> dict:
    method = method.upper()
    presets = {
        "R1": dict(
            controller_mode="SOPD",
            kp=[36.0] * 6,
            kd=[16.0] * 6,
            max_linear_vel=float("inf"),
            max_angular_vel=float("inf"),
        ),
        "R2": dict(
            controller_mode="SOPD",
            kp=[40.0, 40.0, 40.0, 40.0, 40.0, 40.0],
            kd=[15.0, 15.0, 15.0, 15.0, 15.0, 15.0],
            max_linear_vel=float("inf"),
            max_angular_vel=float("inf"),
        ),

        "R3": dict(
            controller_mode="SOPD",
            kp=[80.0, 80.0, 80.0, 80.0, 80.0, 80.0],
            kd=[25.0, 25.0, 25.0, 25.0, 25.0, 25.0],
            max_linear_vel=float("inf"),
            max_angular_vel=float("inf"),
        ),
        "RS": dict(
            controller_mode="SOPD_SAT",
            kp=[80.0, 80.0, 80.0, 80.0, 80.0, 80.0],
            kd=[15.0, 15.0, 15.0, 15.0, 15.0, 15.0],
            accel_limit_pos=[5.0, 5.0, 5.0],
            accel_limit_rot=[10.0, 10.0, 10.0],
            max_linear_vel=float("inf"),
            max_angular_vel=float("inf"),
        ),
        "P": dict(
            controller_mode="SOPDPSMC",
            kp=[300.0] * 6,
            kd=[5.0] * 6,
            proxy_H=[0.65, 0.65, 0.65, 0.65, 0.65, 0.65],
            accel_limit_pos=[15.0, 15.0, 15.0],
            accel_limit_rot=[30.0, 30.0, 30.0],
            max_linear_vel=float("inf"),
            max_angular_vel=float("inf"),
        ),
    }
    if method not in presets:
        raise ValueError(f"Unknown method: {method}")
    return presets[method]


def output_stem(method: str) -> str:
    method = method.upper()
    if method == "RS":
        return "SOPD_SAT"
    return "SOPDPSMC" if method == "P" else f"SOPD_{method}"


def build_config(method: str, runtime: float) -> PBVSConfig:
    project_dir = Path(__file__).resolve().parent
    figure_dir = project_dir / "exp_figures"
    csv_dir = project_dir / "csv_data"
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
        csv_save_path=str(csv_dir / f"{file_stem}.csv"),
        enable_csv_logging=ENABLE_CSV_LOGGING,
        enable_memory_log=ENABLE_MEMORY_LOG,
        enable_final_plots=ENABLE_FINAL_PLOTS,
        status_print_interval=STATUS_PRINT_INTERVAL,
        **params,
    )


def build_targets():
    return [
        TargetPose(
            name="Large_error_start",
            desired_rotation=Rotation.from_euler(
                "xyz", (0, 0, 0), degrees=True
            ).as_matrix(),
            desired_translation=np.array([0.00, 0.00, 0.30]),
        ),
        TargetPose(
            name="Large_error_target_2",
            desired_rotation=Rotation.from_euler(
                "xyz", (0, -10, 0), degrees=True
            ).as_matrix(),
            desired_translation=np.array([0.00, 0.02, 0.15]),
        ),
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Experiment I: large-error regulation comparison."
    )
    parser.add_argument(
        "--method",
        choices=["R1", "R2", "R3", "RS", "P"],
        default="R3",
        help="Controller preset to run.",
    )
    parser.add_argument(
        "--runtime",
        type=float,
        default=8.0,
        help="Recording duration in seconds. Use 0 for manual stop.",
    )
    args = parser.parse_args()

    pipeline, intr = init_realsense()
    intrinsics_params = (intr.fx, intr.fy, intr.ppx, intr.ppy)
    print(f"相机内参: {intrinsics_params}")
    print(f"Experiment I method: {args.method}")
    print(f"CSV output stem: {output_stem(args.method)}")

    controller = PBVSController(
        robot_ip=ROBOT_IP,
        intrinsics=intrinsics_params,
        hand_eye_calib=hand_eye_matrix(),
        config=build_config(args.method, args.runtime),
    )
    controller.set_targets(build_targets())

    init_pose = np.array([
        -0.20588217, -0.05717437,  0.450001008,
        -2.50455863, -1.8897531,  -0.01089382,
    ])
    controller.run(pipeline, init_pose)


if __name__ == "__main__":
    main()
