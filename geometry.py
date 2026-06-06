import numpy as np

from .rotation_utils import skew


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


def compute_c_Q_oc(c_q_oc: np.ndarray) -> np.ndarray:
    """Quaternion matrix c_Q_oc used in Eqs. (20) and (21), q=[qw,qx,qy,qz]."""
    q0, qv = c_q_oc[0], c_q_oc[1:4]
    return q0 * np.eye(3) - skew(qv)


def compute_L(c_p_oc: np.ndarray, c_q_oc: np.ndarray,
              R_base_cam: np.ndarray) -> np.ndarray:
    """Eq. (20): L(s,qc), input is base-frame camera twist uc."""
    Rt = R_base_cam.T
    c_Q_oc = compute_c_Q_oc(c_q_oc)
    return np.block([
        [-Rt,       skew(c_p_oc) @ Rt],
        [np.zeros((3, 3)), -0.5 * c_Q_oc @ Rt]
    ])


def compute_N(c_q_oc: np.ndarray, R_base_cam: np.ndarray) -> np.ndarray:
    """Eq. (21): N(s,qc), input is base-frame object twist uo."""
    Rt = R_base_cam.T
    c_Q_oc = compute_c_Q_oc(c_q_oc)
    return np.block([
        [Rt, np.zeros((3, 3))],
        [np.zeros((3, 3)), 0.5 * c_Q_oc @ Rt],
    ])


def compute_b(c_p_oc: np.ndarray, c_q_oc: np.ndarray,
              u_c: np.ndarray, u_o: np.ndarray,
              R_base_cam: np.ndarray) -> np.ndarray:
    """Eq. (23): b(s,qc,uc,uo), with uc and uo expressed in base frame."""
    v_c_base = u_c[:3]
    omega_c_base = u_c[3:]
    v_o_base = u_o[:3]
    omega_o_base = u_o[3:]
    omega_c_cam = R_base_cam.T @ omega_c_base
    omega_rel_base = omega_o_base - omega_c_base
    N = compute_N(c_q_oc, R_base_cam)
    # Eq. (23): b = N[2(vo-vc)x omega_c, omega_o x omega_c]^T
    #              + [[Rc.T omega_c x]^2 c_p_oc,
    #                 -||omega_o-omega_c||^2 c_q_oc / 4]^T.
    return (
        N @ np.concatenate([
            2.0 * np.cross(v_o_base - v_c_base, omega_c_base),
            np.cross(omega_o_base, omega_c_base),
        ])
        + np.concatenate([
            skew(omega_c_cam) @ skew(omega_c_cam) @ c_p_oc,
            -0.25 * float(omega_rel_base @ omega_rel_base) * c_q_oc[1:4],
        ])
    )
