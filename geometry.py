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
    """Rotational block in Eq. (20): -c_Q_oc / 2, q=[qx,qy,qz,qw]."""
    q0, qv = q[3], q[:3]
    return 0.5 * (-q0 * np.eye(3) + skew(qv))


def compute_L2_hat(c_t_o: np.ndarray, q_oc: np.ndarray) -> np.ndarray:
    """Camera-frame interaction matrix block from Eq. (20)."""
    return np.block([
        [-np.eye(3),         skew(c_t_o)    ],
        [np.zeros((3, 3)),   B1_mat(q_oc)   ]
    ])


def compute_L2s(c_t_o: np.ndarray, q_oc: np.ndarray,
                R_base_cam: np.ndarray) -> np.ndarray:
    """Eq. (20): L(s,qc)=Lhat(s) diag(Rc.T, Rc.T), input is base-frame uc."""
    Rt = R_base_cam.T
    return compute_L2_hat(c_t_o, q_oc) @ np.block([
        [Rt, np.zeros((3, 3))],
        [np.zeros((3, 3)), Rt],
    ])


def compute_N2s(q_oc: np.ndarray, R_base_cam: np.ndarray) -> np.ndarray:
    """Eq. (21): N(s,qc), input is base-frame object twist uo."""
    Rt = R_base_cam.T
    return np.block([
        [Rt, np.zeros((3, 3))],
        [np.zeros((3, 3)), -B1_mat(q_oc) @ Rt],
    ])


def compute_L2_b(c_t_o: np.ndarray, q_oc: np.ndarray,
                 u_c: np.ndarray, u_o: np.ndarray,
                 R_base_cam: np.ndarray) -> np.ndarray:
    """Eq. (23): b(s,qc,uc,uo), with uc and uo expressed in base frame."""
    v_c_base = u_c[:3]
    omega_c_base = u_c[3:]
    v_o_base = u_o[:3]
    omega_o_base = u_o[3:]
    omega_c_cam = R_base_cam.T @ omega_c_base
    omega_rel_base = omega_o_base - omega_c_base
    N = compute_N2s(q_oc, R_base_cam)
    b_trans = (
        N[:3, :3] @ (2.0 * np.cross(v_o_base - v_c_base, omega_c_base))
        + skew(omega_c_cam) @ skew(omega_c_cam) @ c_t_o
    )
    b_rot = (
        N[3:, 3:] @ np.cross(omega_o_base, omega_c_base)
        - 0.25 * float(omega_rel_base @ omega_rel_base) * q_oc[:3]
    )
    return np.concatenate([b_trans, b_rot])
