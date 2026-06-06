"""
Eye-in-Hand Extrinsic Calibration
RealSense D435i + UR3e  (RTDE 版本)

标定配置：Eye-in-Hand（相机安装在末端执行器上）
目标：求解相机坐标系在末端执行器坐标系中的位姿 e_T_c

原理：AX = XB  (Tsai-Lenz 方法)
  A = 相邻两帧间 末端执行器 的相对运动  (base_T_end)
  B = 相邻两帧间 标定板     的相对运动  (camera_T_target)
  X = 待求的手眼矩阵 e_T_c，其中平移项为 e_p_ce = p_c - p_e

依赖安装:
  pip install pyrealsense2 opencv-python opencv-contrib-python numpy
  pip install ur-rtde        # 替换原 urx，用于 UR3e 实时数据读取
"""
from __future__ import annotations

import cv2
import numpy as np
import time
import os
import json
from pathlib import Path
# ──────────────────────────────────────────────
#  RTDE（ur_rtde）读取 UR3e 实时位姿
# ──────────────────────────────────────────────
try:
    from rtde_receive import RTDEReceiveInterface as RTDEReceive
    HAS_RTDE = True
except ImportError:
    HAS_RTDE = False
    print("[WARN] ur-rtde not installed. Robot poses must be entered manually.")

try:
    import pyrealsense2 as rs
    HAS_RS = True
except ImportError:
    HAS_RS = False
    print("[WARN] pyrealsense2 not installed. Camera will use OpenCV fallback.")


# ══════════════════════════════════════════════
#  全局配置
# ══════════════════════════════════════════════
class Config:
    # ── 标定板（仅棋盘格）────────────────────────
    SQUARE_SIZE_M  = 0.025           # 格子边长（米）
    CHESS_ROWS     = 6               # 内角点行数
    CHESS_COLS     = 9               # 内角点列数

    # ── 采集参数 ────────────────────────────────
    MIN_POSES      = 10              # 最少采集位姿数
    SAVE_DIR       = Path("calib_data")

    # ── UR3e 网络 ───────────────────────────────
    ROBOT_IP       = "10.31.17.190"

    # ── RTDE 采样频率（Hz）──────────────────────
    RTDE_FREQUENCY = 500.0

    # ── 手眼标定算法 ────────────────────────────
    HANDEYE_METHOD = cv2.CALIB_HAND_EYE_TSAI


# ══════════════════════════════════════════════
#  RealSense D435i 封装
# ══════════════════════════════════════════════
class RealSenseCamera:
    def __init__(self, width=1280, height=720, fps=30):
        if not HAS_RS:
            raise RuntimeError("pyrealsense2 not available.")

        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16,  fps)
        profile = self.pipeline.start(cfg)

        color_stream = profile.get_stream(rs.stream.color)
        intr = color_stream.as_video_stream_profile().get_intrinsics()

        self.K = np.array([
            [intr.fx,       0, intr.ppx],
            [      0, intr.fy, intr.ppy],
            [      0,       0,         1],
        ], dtype=np.float64)
        self.dist = np.array(intr.coeffs, dtype=np.float64)

        print(f"[Camera] D435i ready  |  fx={intr.fx:.2f}  fy={intr.fy:.2f}")
        print(f"         cx={intr.ppx:.2f}  cy={intr.ppy:.2f}")
        print(f"         dist={self.dist}")

        self.align = rs.align(rs.stream.color)
        time.sleep(1)  # 暖机

    def get_frame(self):
        frames      = self.pipeline.wait_for_frames()
        aligned     = self.align.process(frames)
        color_frame = aligned.get_color_frame()
        if not color_frame:
            return None
        return np.asanyarray(color_frame.get_data())

    def stop(self):
        self.pipeline.stop()


# ══════════════════════════════════════════════
#  UR3e 机械臂封装（RTDE 版本）
# ══════════════════════════════════════════════
class UR3eRobot:
    """
    通过 RTDE（ur_rtde）读取末端执行器在 base 坐标系下的位姿（4×4 齐次矩阵）。

    RTDE 接口直接提供 actual_TCP_pose = [x, y, z, rx, ry, rz]（轴角，单位 m / rad），
    与 URx 的 getl() 格式完全一致，因此矩阵转换逻辑保持不变。

    若无 ur-rtde 或无法连接，自动回退到手动输入模式。
    """

    def __init__(self, ip: str = None, frequency: float = 500.0):
        self.rtde_r = None

        if not HAS_RTDE:
            print("[Robot] ur-rtde 未安装，将使用手动输入模式。")
            return

        if not ip:
            print("[Robot] 未指定 IP，将使用手动输入模式。")
            return

        try:
            self.rtde_r = RTDEReceive(
                ip,
                frequency,
                ["actual_TCP_pose"],
                True,
                False,
            )
            if not self.rtde_r.isConnected():
                raise RuntimeError("RTDE 连接建立失败（isConnected() = False）")
            print(f"[Robot] RTDE 已连接  UR3e @ {ip}  (freq={frequency} Hz)")
        except Exception as e:
            print(f"[Robot] RTDE 连接失败: {e}")
            print("[Robot] 回退到手动输入模式。")
            self.rtde_r = None

    def get_pose_matrix(self) -> np.ndarray:
        """返回 base_T_end (4×4 齐次矩阵)"""
        if self.rtde_r is not None and self.rtde_r.isConnected():
            pose = self.rtde_r.getActualTCPPose()
            return self._vec_to_mat(pose)
        else:
            return self._manual_input()

    @staticmethod
    def _vec_to_mat(pose) -> np.ndarray:
        """[x, y, z, rx, ry, rz]（轴角）→ 4×4 齐次矩阵"""
        x, y, z, rx, ry, rz = pose
        rvec = np.array([rx, ry, rz], dtype=np.float64)
        angle = np.linalg.norm(rvec)
        if angle < 1e-10:
            R = np.eye(3, dtype=np.float64)
        else:
            R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3,  3] = [x, y, z]
        return T

    @staticmethod
    def _manual_input() -> np.ndarray:
        print("\n[Manual] Enter end-effector pose [x y z rx ry rz]")
        print("         Units: meters, radians (axis-angle)")
        vals = list(map(float, input("  > ").strip().split()))
        assert len(vals) == 6, "Exactly 6 values required."
        return UR3eRobot._vec_to_mat(vals)

    def disconnect(self):
        if self.rtde_r is not None:
            try:
                self.rtde_r.disconnect()
                print("[Robot] RTDE 已断开。")
            except Exception:
                pass
            self.rtde_r = None


# ══════════════════════════════════════════════
#  标定板检测器（仅棋盘格）
# ══════════════════════════════════════════════
class BoardDetector:
    def __init__(self, cfg: Config, K, dist):
        self.cfg  = cfg
        self.K    = K
        self.dist = dist

        # 棋盘格内角点排列 (cols, rows)
        self.pattern = (cfg.CHESS_COLS, cfg.CHESS_ROWS)
        # 世界坐标系下棋盘格内角点坐标 (Z=0)
        objp = np.zeros((cfg.CHESS_ROWS * cfg.CHESS_COLS, 3), np.float32)
        objp[:, :2] = (
            np.mgrid[0:cfg.CHESS_COLS, 0:cfg.CHESS_ROWS].T.reshape(-1, 2)
            * cfg.SQUARE_SIZE_M
        )
        self.objp = objp

    def detect(self, image):
        """
        检测棋盘格，返回 (rvec, tvec, vis_image)
        若检测失败，rvec, tvec 为 None
        """
        vis = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        ok, corners = cv2.findChessboardCorners(gray, self.pattern, flags)
        if not ok:
            return None, None, vis

        # 亚像素细化
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        cv2.drawChessboardCorners(vis, self.pattern, corners, ok)

        # 解算位姿
        ok, rvec, tvec = cv2.solvePnP(self.objp, corners, self.K, self.dist)
        if not ok:
            return None, None, vis

        # 绘制坐标系轴
        cv2.drawFrameAxes(vis, self.K, self.dist, rvec, tvec, 0.05)
        return rvec, tvec, vis


# ══════════════════════════════════════════════
#  手眼标定主流程
# ══════════════════════════════════════════════
class EyeInHandCalibration:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        cfg.SAVE_DIR.mkdir(exist_ok=True)

        print("=" * 60)
        print("  Eye-in-Hand Calibration  (RTDE 版本)")
        print("  RealSense D435i  +  UR3e")
        print("=" * 60)

        self.cam   = RealSenseCamera()
        self.robot = UR3eRobot(ip=cfg.ROBOT_IP, frequency=cfg.RTDE_FREQUENCY)
        self.det   = BoardDetector(cfg, self.cam.K, self.cam.dist)

        # 采集缓冲
        self.R_g2b: list[np.ndarray] = []   # base_T_end 的旋转部分
        self.t_g2b: list[np.ndarray] = []   # base_T_end 的平移部分
        self.R_t2c: list[np.ndarray] = []   # camera_T_target 的旋转部分
        self.t_t2c: list[np.ndarray] = []   # camera_T_target 的平移部分

    # ── 交互采集 ────────────────────────────────
    def collect(self) -> int:
        print(f"\n[Collect] Move the robot to ≥{self.cfg.MIN_POSES} different poses.")
        print("  SPACE → capture   |   u → undo   |   q → done & calibrate\n")

        idx = 0
        while True:
            img = self.cam.get_frame()
            if img is None:
                continue

            rvec, tvec, vis = self.det.detect(img)

            # HUD
            n_color = (0, 200, 50) if rvec is not None else (30, 30, 220)
            cv2.putText(vis, f"Captured: {idx}/{self.cfg.MIN_POSES}",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, n_color, 2)
            hint = "Board OK – SPACE to capture" if rvec is not None else "Board not detected"
            cv2.putText(vis, hint, (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, n_color, 2)

            cv2.imshow("Eye-in-Hand Calibration", vis)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord('u') and idx > 0:
                for lst in [self.R_g2b, self.t_g2b, self.R_t2c, self.t_t2c]:
                    lst.pop()
                idx -= 1
                print(f"  [Undo] Total: {idx}")

            elif key == ord(' '):
                if rvec is None:
                    print("  [Skip] Board not detected – move to a better angle.")
                    continue

                print(f"\n  [Pose {idx+1}] Reading robot pose via RTDE...")
                base_T_end = self.robot.get_pose_matrix()

                Rg = base_T_end[:3, :3]
                tg = base_T_end[:3, 3].reshape(3, 1)
                Rt, _ = cv2.Rodrigues(rvec)
                tt     = tvec.reshape(3, 1)

                self.R_g2b.append(Rg);  self.t_g2b.append(tg)
                self.R_t2c.append(Rt);  self.t_t2c.append(tt)

                # 保存采集图像
                cv2.imwrite(str(self.cfg.SAVE_DIR / f"pose_{idx:03d}.png"), img)
                idx += 1
                print(f"  Saved. base_T_end =\n{base_T_end}")

        cv2.destroyAllWindows()
        return idx

    # ── 执行标定 ────────────────────────────────
    def calibrate(self) -> np.ndarray | None:
        n = len(self.R_g2b)
        print(f"\n[Calibrate] Running cv2.calibrateHandEye with {n} poses ...")

        if n < self.cfg.MIN_POSES:
            print(f"[ERROR] Need ≥{self.cfg.MIN_POSES} poses, only got {n}.")
            return None

        R_c2e, t_c2e = cv2.calibrateHandEye(
            self.R_g2b, self.t_g2b,
            self.R_t2c, self.t_t2c,
            method=self.cfg.HANDEYE_METHOD,
        )

        e_T_c = np.eye(4, dtype=np.float64)
        e_T_c[:3, :3] = R_c2e
        e_T_c[:3,  3] = t_c2e.flatten()

        self._print_result(e_T_c)
        self._save_result(e_T_c)
        return e_T_c

    @staticmethod
    def _print_result(e_T_c: np.ndarray):
        np.set_printoptions(precision=4, suppress=True)
        print("\n" + "=" * 60)
        print("  Result:  e_T_c  (camera frame expressed in end-effector frame, 4×4)")
        print("=" * 60)
        print("\n# 手眼标定矩阵")
        print("e_T_c = np.array([")
        for row in e_T_c:
            print("    [" + ", ".join(f"{v:8.4f}" for v in row) + "],")
        print("])")

        t = e_T_c[:3, 3]
        R = e_T_c[:3, :3]
        rvec, _ = cv2.Rodrigues(R)
        angle_deg = np.degrees(np.linalg.norm(rvec))
        print(f"\n  Translation  : x={t[0]:.4f}  y={t[1]:.4f}  z={t[2]:.4f}  m")
        print(f"  Rotation     : axis-angle magnitude = {angle_deg:.2f} deg")

        # ZYX Euler
        sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
        if sy > 1e-6:
            roll  = np.degrees(np.arctan2( R[2,1], R[2,2]))
            pitch = np.degrees(np.arctan2(-R[2,0], sy))
            yaw   = np.degrees(np.arctan2( R[1,0], R[0,0]))
        else:
            roll  = np.degrees(np.arctan2(-R[1,2], R[1,1]))
            pitch = np.degrees(np.arctan2(-R[2,0], sy))
            yaw   = 0.0
        print(f"  Euler ZYX    : roll={roll:.2f}°  pitch={pitch:.2f}°  yaw={yaw:.2f}°")

    def _save_result(self, e_T_c: np.ndarray):
        np.save(str(self.cfg.SAVE_DIR / "e_T_c.npy"), e_T_c)

        data = {
            "description"  : "Eye-in-Hand: e_T_c, camera frame expressed in end-effector frame",
            "sensor"       : "RealSense D435i",
            "robot"        : "UR3e",
            "interface"    : "RTDE (ur_rtde)",
            "method"       : "Tsai (cv2.CALIB_HAND_EYE_TSAI)",
            "camera_matrix": self.cam.K.tolist(),
            "dist_coeffs"  : self.cam.dist.tolist(),
            "e_T_c"        : e_T_c.tolist(),
        }
        json_path = self.cfg.SAVE_DIR / "e_T_c.json"
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"\n[Saved] {self.cfg.SAVE_DIR/'e_T_c.npy'}  and  {json_path}")

    def run(self) -> np.ndarray | None:
        try:
            n = self.collect()
            return self.calibrate() if n >= self.cfg.MIN_POSES else None
        finally:
            self.cam.stop()
            self.robot.disconnect()


# ══════════════════════════════════════════════
#  已知标定结果：变换示例
# ══════════════════════════════════════════════
def demo_with_known_result():
    """
    使用已知手眼矩阵，演示坐标变换链：
      camera → end-effector → base
    """
    e_T_c = np.array([
        [-0.0331, -0.9956, -0.0873, -0.0471],
        [-0.9994, -0.0325, -0.0079, -0.0371],
        [ 0.0050, -0.0875, -0.9962, -0.0413],
        [ 0.0000,  0.0000,  0.0000,  1.0000],
    ])

    print("=" * 60)
    print("  Demo: Coordinate Transform with Known e_T_c")
    print("=" * 60)

    np.set_printoptions(precision=4, suppress=True)
    print("\nHand-Eye Matrix (e_T_c), camera frame expressed in end-effector frame:")
    print(e_T_c)

    R = e_T_c[:3, :3]
    err = np.linalg.norm(R @ R.T - np.eye(3))
    print(f"\nOrthogonality check  ||R·Rᵀ - I|| = {err:.2e}  (should be ~0)")

    base_T_end = np.array([
        [ 0.9659,  0.0000,  0.2588, 0.300],
        [ 0.0000,  1.0000,  0.0000, 0.100],
        [-0.2588,  0.0000,  0.9659, 0.500],
        [ 0.0000,  0.0000,  0.0000, 1.000],
    ])
    print("\nCurrent base_T_end (example):")
    print(base_T_end)

    p_cam = np.array([0.05, -0.02, 0.35, 1.0])

    base_T_cam = base_T_end @ e_T_c
    p_base     = base_T_cam @ p_cam

    print(f"\nPoint in camera frame     : {p_cam[:3]}")
    print(f"Point in end-effector frame: {(e_T_c @ p_cam)[:3]}")
    print(f"Point in base frame        : {p_base[:3]}")

    cam_T_end = np.linalg.inv(e_T_c)
    print("\nInverse  cam_T_end  (camera → end-effector):")
    print(cam_T_end)

    return e_T_c, base_T_cam


# ══════════════════════════════════════════════
#  离线模式：从保存的 JSON 重新标定
# ══════════════════════════════════════════════
def offline_calibrate(json_path: str) -> np.ndarray:
    """
    从已保存的 poses JSON 文件重新运行标定（无需相机和机械臂）。

    JSON 格式:
    {
      "poses": [
        {
          "R_g2b": [[...3x3...]],
          "t_g2b": [x, y, z],
          "R_t2c": [[...3x3...]],
          "t_t2c": [x, y, z]
        },
        ...
      ]
    }
    """
    with open(json_path) as f:
        data = json.load(f)

    R_g, t_g, R_t, t_t = [], [], [], []
    for p in data["poses"]:
        R_g.append(np.array(p["R_g2b"], dtype=np.float64))
        t_g.append(np.array(p["t_g2b"], dtype=np.float64).reshape(3, 1))
        R_t.append(np.array(p["R_t2c"], dtype=np.float64))
        t_t.append(np.array(p["t_t2c"], dtype=np.float64).reshape(3, 1))

    R_c2e, t_c2e = cv2.calibrateHandEye(
        R_g, t_g, R_t, t_t,
        method=cv2.CALIB_HAND_EYE_TSAI,
    )
    e_T_c = np.eye(4)
    e_T_c[:3, :3] = R_c2e
    e_T_c[:3,  3] = t_c2e.flatten()

    EyeInHandCalibration._print_result(e_T_c)
    return e_T_c


# ══════════════════════════════════════════════
#  入口
# ══════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Eye-in-Hand Calibration: RealSense D435i + UR3e (RTDE)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["collect", "demo", "offline"],
        default="demo",
        help=(
            "collect  – 交互采集位姿并标定（需要相机和机械臂）\n"
            "demo     – 用已知 e_T_c 演示坐标变换（无需硬件）\n"
            "offline  – 从已保存的 JSON 重新标定"
        ),
    )
    parser.add_argument(
        "--json", default="", help="poses JSON 路径（仅 offline 模式）"
    )
    args = parser.parse_args()

    if args.mode == "collect":
        cfg   = Config()
        calib = EyeInHandCalibration(cfg)
        result = calib.run()
        if result is not None:
            print("\n[Done] 标定完成！")

    elif args.mode == "demo":
        demo_with_known_result()

    elif args.mode == "offline":
        if not args.json:
            print("[ERROR] --json <path> 是 offline 模式的必要参数。")
        else:
            offline_calibrate(args.json)
