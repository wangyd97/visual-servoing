import numpy as np


def matrix_from_rotvec(rotvec: np.ndarray) -> np.ndarray:
    rotvec = np.asarray(rotvec, dtype=float).reshape(3)
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-12:
        return np.eye(3) + skew(rotvec)
    axis = rotvec / theta
    K = skew(axis)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def rotvec_from_matrix(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=float).reshape(3, 3)
    cos_theta = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))
    if theta < 1e-12:
        return 0.5 * np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ])
    if np.pi - theta < 1e-6:
        axis = np.empty(3)
        axis[0] = np.sqrt(max(0.0, (R[0, 0] + 1.0) * 0.5))
        axis[1] = np.sqrt(max(0.0, (R[1, 1] + 1.0) * 0.5))
        axis[2] = np.sqrt(max(0.0, (R[2, 2] + 1.0) * 0.5))
        if R[0, 1] < 0.0:
            axis[1] = -axis[1]
        if R[0, 2] < 0.0:
            axis[2] = -axis[2]
        norm = np.linalg.norm(axis)
        if norm < 1e-12:
            axis = np.array([1.0, 0.0, 0.0])
        else:
            axis /= norm
        return theta * axis
    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ]) / (2.0 * np.sin(theta))
    return theta * axis


def quat_from_matrix(R: np.ndarray) -> np.ndarray:
    """Return paper-order unit quaternion q=[w,x,y,z]."""
    R = np.asarray(R, dtype=float).reshape(3, 3)
    tr = float(np.trace(R))
    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) * 2.0
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(max(0.0, 1.0 + R[1, 1] - R[0, 0] - R[2, 2])) * 2.0
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(max(0.0, 1.0 + R[2, 2] - R[0, 0] - R[1, 1])) * 2.0
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    q = np.array([qw, qx, qy, qz], dtype=float)
    return q / (np.linalg.norm(q) + 1e-12)


def matrix_from_quat(q: np.ndarray) -> np.ndarray:
    """Build rotation matrix from paper-order quaternion q=[w,x,y,z]."""
    q = np.asarray(q, dtype=float).reshape(4)
    q = q / (np.linalg.norm(q) + 1e-12)
    w, x, y, z = q
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ])


def matrix_from_euler_xyz(angles, degrees: bool = False) -> np.ndarray:
    angles = np.asarray(angles, dtype=float).reshape(3)
    if degrees:
        angles = np.deg2rad(angles)
    ax, ay, az = angles
    cx, sx = np.cos(ax), np.sin(ax)
    cy, sy = np.cos(ay), np.sin(ay)
    cz, sz = np.cos(az), np.sin(az)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return Rz @ Ry @ Rx


def euler_xyz_from_matrix(R: np.ndarray, degrees: bool = False) -> np.ndarray:
    R = np.asarray(R, dtype=float).reshape(3, 3)
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    if sy > 1e-9:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0.0
    angles = np.array([x, y, z], dtype=float)
    return np.rad2deg(angles) if degrees else angles


def skew(v: np.ndarray) -> np.ndarray:
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])
