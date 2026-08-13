# visual-servoing
Responsive yet overdamped visual servoing, with high-speed dynamics and safety.

## Main Files

| File | Description |
| --- | --- |
| [`main.py`](main.py) | Configures and starts the visual servoing experiment. |
| [`config.py`](config.py) | Defines controller, vision, target, and logging parameters. |
| [`controller.py`](controller.py) | Runs AprilTag-based PBVS control and sends velocity commands to the UR robot. |
| [`vision.py`](vision.py) | Initializes the RealSense camera and estimates AprilTag poses. |
| [`geometry.py`](geometry.py) | Provides transformations, projections, and visual-servoing matrices. |
| [`psmc.py`](psmc.py) | Implements the proxy-based controller with acceleration constraints. |
| [`Mathematic.py`](Mathematic.py) | Provides quaternion, rotation, and vector utilities. |
| [`Tools/hand_eye_calibration.py`](Tools/hand_eye_calibration.py) | Calibrates the transformation between the robot end effector and camera. |
