from .config import PBVSConfig, TargetPose

__all__ = ["PBVSConfig", "TargetPose", "PBVSController", "init_realsense"]


def __getattr__(name):
    if name == "PBVSController":
        from .controller import PBVSController
        return PBVSController
    if name == "init_realsense":
        from .vision import init_realsense
        return init_realsense
    raise AttributeError(name)
