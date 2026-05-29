import numpy as np


def skew(p: np.ndarray) -> np.ndarray:
    return np.array([
        [0,     -p[2],  p[1]],
        [p[2],   0,    -p[0]],
        [-p[1],  p[0],  0   ]
    ])


def inv_T(T: np.ndarray) -> np.ndarray:
    Rot, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = Rot.T
    Ti[:3, 3] = -Rot.T @ t
    return Ti


def project_3d_to_2d(pts: np.ndarray, K: np.ndarray) -> np.ndarray:
    p = K @ pts.T
    return (p[:2] / p[2]).T


def get_tag_3d_corners(tag_size: float, T_tag: np.ndarray) -> np.ndarray:
    h = tag_size / 2.0
    c = np.array([[-h, -h, 0], [h, -h, 0], [h, h, 0], [-h, h, 0]])
    ch = np.hstack([c, np.ones((4, 1))])
    return (T_tag @ ch.T).T[:, :3]


def B1_mat(q: np.ndarray) -> np.ndarray:
    """B1(q)，q=[qx,qy,qz,qw]，对应 qv_dot = B1(q) * c_omega_cb。"""
    q0, qv = q[3], q[:3]
    return 0.5 * (-q0 * np.eye(3) + skew(qv))


def compute_L2_hat(c_t_o: np.ndarray, q_oc: np.ndarray) -> np.ndarray:
    """PDF eq.(9) 的 Lhat(s)，输入 twist 为 camera-frame [c_v_cb, c_omega_cb]。"""
    return np.block([
        [-np.eye(3),         skew(c_t_o)    ],
        [np.zeros((3, 3)),   B1_mat(q_oc)   ]
    ])


def compute_L2s(c_t_o: np.ndarray, q_oc: np.ndarray,
                R_base_cam: np.ndarray) -> np.ndarray:
    """PDF eq.(8): L(s,qc)=Lhat(s) diag(Rc.T, Rc.T)，输入 twist 为 base-frame uc。"""
    Rt = R_base_cam.T
    return compute_L2_hat(c_t_o, q_oc) @ np.block([
        [Rt, np.zeros((3, 3))],
        [np.zeros((3, 3)), Rt],
    ])


def compute_L2_b(c_t_o: np.ndarray, q_oc: np.ndarray,
                 u_c: np.ndarray,
                 R_base_cam: np.ndarray) -> np.ndarray:
    """PDF eq.(11) 的 b(s,qc,uc)，输入 uc=[v_c,omega_c] 均在 base frame 表示。"""
    v_base = u_c[:3]
    omega_base = u_c[3:]
    omega_cam = R_base_cam.T @ omega_base
    b_trans = (
        2.0 * R_base_cam.T @ np.cross(omega_base, v_base)
        + skew(omega_cam) @ skew(omega_cam) @ c_t_o
    )
    b_rot = -0.25 * float(omega_base @ omega_base) * q_oc[:3]
    return np.concatenate([b_trans, b_rot])
