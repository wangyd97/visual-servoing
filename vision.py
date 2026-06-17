import sys

import cv2
import numpy as np
import pyrealsense2 as rs
from pupil_apriltags import Detector

from .config import PBVSConfig
from .geometry import project_3d_to_2d


def init_realsense():
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        print("No RealSense device detected.", flush=True)
        sys.exit(1)

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
    try:
        profile = pipeline.start(cfg)
        sensor = pipeline.get_active_profile().get_device().query_sensors()[1]
        sensor.set_option(rs.option.enable_auto_exposure, 0)
        sensor.set_option(rs.option.exposure, 100)
        intr = (profile.get_stream(rs.stream.color)
                .as_video_stream_profile().get_intrinsics())
        return pipeline, intr
    except RuntimeError as e:
        print(f"RealSense initialization failed: {e}", flush=True)
        sys.exit(1)



def draw_axis(img, K, Rot, t, length=0.05):
    axis = np.float32([[0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, length]])
    T = np.eye(4); T[:3, :3] = Rot; T[:3, 3] = t
    pts = (T @ np.hstack([axis, np.ones((4, 1))]).T).T[:, :3]
    if np.all(pts[:, 2] > 0):
        ip = project_3d_to_2d(pts, K).astype(int)
        o = tuple(ip[0])
        cv2.line(img, o, tuple(ip[1]), (0, 0, 255), 3)
        cv2.line(img, o, tuple(ip[2]), (0, 255, 0), 3)
        cv2.line(img, o, tuple(ip[3]), (255, 0, 0), 3)
        return ip[0]
    return None


def draw_axis_colored(img, K, Rot, t, length=0.05,
                      colors=((0, 0, 255), (0, 255, 0), (255, 0, 0))):
    axis = np.float32([[0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, length]])
    T = np.eye(4); T[:3, :3] = Rot; T[:3, 3] = t
    pts = (T @ np.hstack([axis, np.ones((4, 1))]).T).T[:, :3]
    if np.all(pts[:, 2] > 0):
        ip = project_3d_to_2d(pts, K).astype(int)
        o = tuple(ip[0])
        cv2.line(img, o, tuple(ip[1]), colors[0], 2)
        cv2.line(img, o, tuple(ip[2]), colors[1], 2)
        cv2.line(img, o, tuple(ip[3]), colors[2], 2)
        return ip[0]
    return None




class AprilTagEstimator:
    def __init__(self, config: PBVSConfig, intrinsics):
        self.detector = Detector(
            families='tag36h11',
            nthreads=config.apriltag_nthreads,
            quad_decimate=config.apriltag_quad_decimate,
            quad_sigma=config.apriltag_quad_sigma,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )
        fx, fy, cx, cy = intrinsics
        self.camera_params = [fx, fy, cx, cy]
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        self.tag_size = config.tag_size

    def detect(self, gray):
        dets = self.detector.detect(
            gray, estimate_tag_pose=True,
            camera_params=self.camera_params,
            tag_size=self.tag_size
        )
        if dets:
            d = dets[0]
            T = np.eye(4)
            T[:3, :3] = d.pose_R
            T[:3, 3] = d.pose_t.flatten()
            return T, d.corners, d.pose_R, d.pose_t.flatten()
        return None, None, None, None
