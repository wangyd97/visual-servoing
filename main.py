#updated on 2026-06-02:
import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from e1.config import PBVSConfig, TargetPose
    from e1.controller import PBVSController
    from e1.rotation_utils import matrix_from_quat, matrix_from_rotvec
    from e1.vision import init_realsense
else:
    from .config import PBVSConfig, TargetPose
    from .controller import PBVSController
    from .rotation_utils import matrix_from_quat, matrix_from_rotvec
    from .vision import init_realsense


ENABLE_VISUALIZATION = True
ENABLE_MEMORY_LOG = True
ENABLE_FINAL_PLOTS = True
STATUS_PRINT_INTERVAL = 30
APRILTAG_NTHREADS = 2
APRILTAG_QUAD_DECIMATE = 2.0
ENABLE_SMALL_STEP_EXPERIMENT = False

SMALL_STEP_START_TIME = 3.0
SMALL_STEP_PERIOD = 0.80
SMALL_STEP_TRANSLATION = 0.015
SMALL_STEP_ROT_DEG = 3.0
SMALL_STEP_TRANS_CYCLES = 3
SMALL_STEP_ROT_CYCLES = 3

ROBOT_IP = "10.31.17.94"


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
            kd=40*np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            max_linear_vel=float("inf"),
            max_angular_vel=float("inf"),
        ),
        "RSC": dict(
            controller_mode="SOPD",
            kp=150*np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            kd=40*np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            accel_limit_pos=5 * np.array([1.0, 1.0, 1.0]),
            accel_limit_rot=10 * np.array([1.0, 1.0, 1.0]),
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
            kd=25 * np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
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


def build_config(method: str, runtime: float, small_step: bool = False) -> PBVSConfig:
    project_dir = Path(__file__).resolve().parent
    figure_dir = project_dir / "exp_figures"
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
        auto_switch_targets=small_step,
        auto_switch_start_time=SMALL_STEP_START_TIME,
        auto_switch_period=SMALL_STEP_PERIOD,
        plot_save_path=str(figure_dir / f"{file_stem}.png"),
        trajectory_plot_save_path=str(figure_dir / f"{file_stem}_trajectory.png"),
        log_save_path=str(figure_dir / f"log_{method.upper()}_test.csv"),
        enable_memory_log=ENABLE_MEMORY_LOG,
        enable_final_plots=ENABLE_FINAL_PLOTS,
        status_print_interval=STATUS_PRINT_INTERVAL,
        **params,
    )


def build_small_step_sequence():
    sequence = [("Reference_0", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))]
    for i in range(SMALL_STEP_TRANS_CYCLES):
        sequence.extend([
            (f"Step_x_plus_{i + 1}", (SMALL_STEP_TRANSLATION, 0.0, 0.0), (0.0, 0.0, 0.0)),
            (f"Reference_x_plus_{i + 1}", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            (f"Step_x_minus_{i + 1}", (-SMALL_STEP_TRANSLATION, 0.0, 0.0), (0.0, 0.0, 0.0)),
            (f"Reference_x_minus_{i + 1}", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ])
    for i in range(SMALL_STEP_ROT_CYCLES):
        sequence.extend([
            (f"Step_ry_plus_{i + 1}", (0.0, 0.0, 0.0), (0.0, SMALL_STEP_ROT_DEG, 0.0)),
            (f"Reference_ry_plus_{i + 1}", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            (f"Step_ry_minus_{i + 1}", (0.0, 0.0, 0.0), (0.0, -SMALL_STEP_ROT_DEG, 0.0)),
            (f"Reference_ry_minus_{i + 1}", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ])
    return sequence


def build_targets(small_step: bool = False):
    # desired_quaternion = np.array([0.970, 0.171, 0.171, 0.030])  # q=[qw,qx,qy,qz]
    desired_quaternion = np.array([1, 0, 0, 0])  # q=[qw,qx,qy,qz]
    base_rotation = matrix_from_quat(desired_quaternion)
    base_translation = np.array([0.00, 0.00, 0.25])
    if not small_step:
        return [
            TargetPose(
                name="Large_error_start",
                desired_rotation=base_rotation,
                desired_translation=base_translation,
            ),
        ]

    targets = []
    for name, step_translation, step_rotation_deg in build_small_step_sequence():
        step_rotation = matrix_from_rotvec(np.deg2rad(np.asarray(step_rotation_deg, dtype=float)))
        targets.append(
            TargetPose(
                name=name,
                desired_rotation=step_rotation @ base_rotation,
                desired_translation=base_translation + np.asarray(step_translation, dtype=float),
            )
        )
    return targets


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
        runtime = 28.0 if ENABLE_SMALL_STEP_EXPERIMENT else 8.0

    pipeline, intr = init_realsense()
    intrinsics_params = (intr.fx, intr.fy, intr.ppx, intr.ppy)
    if ENABLE_SMALL_STEP_EXPERIMENT:
        print("Experiment II small-step mode: ON")
        print(f"Step sequence starts at {SMALL_STEP_START_TIME:.2f}s, period {SMALL_STEP_PERIOD:.2f}s")
        print(f"Translation step: {SMALL_STEP_TRANSLATION * 1000.0:.1f} mm")
        print(f"Rotation step: {SMALL_STEP_ROT_DEG:.1f} deg")
    else:
        print("Small-step mode: OFF")

    controller = PBVSController(
        robot_ip=ROBOT_IP,
        intrinsics=intrinsics_params,
        hand_eye_calib=hand_eye_matrix(),
        config=build_config(args.method, runtime, small_step=ENABLE_SMALL_STEP_EXPERIMENT),
    )
    controller.set_targets(build_targets(small_step=ENABLE_SMALL_STEP_EXPERIMENT))

    init_pose = np.array([
        -0.20588217, -0.05717437,  0.420001008,
        -2.50455863, -1.8897531,  -0.01089382,
    ])
    controller.run(pipeline, init_pose)


if __name__ == "__main__":
    main()
