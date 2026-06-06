import numpy as np


def sinc(x: float) -> float:
    """sin(x) / x with a smooth small-angle expansion."""
    x = float(x)
    x2 = x * x
    if abs(x) < 1e-8:
        return 1.0 - x2 / 6.0 + x2 * x2 / 120.0
    return np.sin(x) / x


def normalize_quat(q: np.ndarray) -> np.ndarray:
    """Normalize a paper-order quaternion q=[w,x,y,z]."""
    q = np.asarray(q, dtype=float).reshape(4).copy()
    norm = float(np.linalg.norm(q))
    if norm <= 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / norm


def quat_inv(q: np.ndarray) -> np.ndarray:
    """Quaternion inverse for q=[w,x,y,z]."""
    q = normalize_quat(q)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Quaternion product a*b for paper-order quaternions q=[w,x,y,z]."""
    aw, ax, ay, az = normalize_quat(a)
    bw, bx, by, bz = normalize_quat(b)
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        ax * bw + aw * bx - az * by + ay * bz,
        ay * bw + az * bx + aw * by - ax * bz,
        az * bw - ay * bx + ax * by + aw * bz,
    ], dtype=float)


def quat_nearer(q_ref: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Flip q if -q is closer to q_ref."""
    q_ref = normalize_quat(q_ref)
    q = normalize_quat(q)
    return -q if float(q_ref @ q) < 0.0 else q


def quat_from_rotvec(rotvec: np.ndarray) -> np.ndarray:
    """Build q=[w,x,y,z] from a rotation vector using professor-style sinc."""
    rotvec = np.asarray(rotvec, dtype=float).reshape(3)
    half_theta = 0.5 * float(np.linalg.norm(rotvec))
    q = np.array([
        np.cos(half_theta),
        0.5 * rotvec[0] * sinc(half_theta),
        0.5 * rotvec[1] * sinc(half_theta),
        0.5 * rotvec[2] * sinc(half_theta),
    ], dtype=float)
    return normalize_quat(q)


def quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    """Convert q=[w,x,y,z] to the shortest rotation vector."""
    q = normalize_quat(q)
    qv = q[1:4]
    rx = float(np.linalg.norm(qv))
    if rx == 0.0:
        return np.zeros(3)

    theta = 2.0 * np.arctan2(rx, q[0])
    if theta >= np.pi:
        theta -= 2.0 * np.pi
    return theta * qv / rx


def matrix_from_rotvec(rotvec: np.ndarray) -> np.ndarray:
    return matrix_from_quat(quat_from_rotvec(rotvec))


def rotvec_from_matrix(R: np.ndarray) -> np.ndarray:
    return quat_to_rotvec(quat_from_matrix(R))


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
    return normalize_quat(q)


def matrix_from_quat(q: np.ndarray) -> np.ndarray:
    """Build rotation matrix from paper-order quaternion q=[w,x,y,z]."""
    q = normalize_quat(q)
    w, x, y, z = q
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ])


# Just for logging, not used in control
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
