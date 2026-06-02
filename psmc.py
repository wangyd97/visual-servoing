import numpy as np

from .config import to_vec6


class PSMCPDProxy:
    def __init__(self, n, K, B, A, H, dt):
        self.n = n
        self.K = to_vec6(K, "K")
        self.B = to_vec6(B, "B")
        self.A = to_vec6(A, "A")
        self.H = to_vec6(H, "H")
        self.T = float(dt)

        self.s_p_prv = np.zeros(n)
        self.s_p_star = np.zeros(n)
        self.s_p = np.zeros(n)
        self.alpha_c_star = np.zeros(n)
        self.alpha_c = np.zeros(n)
        self.u_dot_c_star = np.zeros(n)
        self.u_dot_c = np.zeros(n)
        self._initialized = False

    @staticmethod
    def _clip_3d(x: np.ndarray, A: float) -> np.ndarray:
        norm = float(np.linalg.norm(x))
        if norm <= A or A == float("inf"):
            return x.copy()
        return (A / max(norm, A)) * x

    def _clip_camera_acceleration(self, u_dot_c: np.ndarray) -> np.ndarray:
        clipped = np.zeros_like(u_dot_c)
        clipped[:3] = self._clip_3d(u_dot_c[:3], float(self.A[0]))
        clipped[3:] = self._clip_3d(u_dot_c[3:], float(self.A[3]))
        return clipped

    def reset(self):
        self.s_p_prv[:] = 0.0
        self.s_p_star[:] = 0.0
        self.s_p[:] = 0.0
        self.alpha_c_star[:] = 0.0
        self.alpha_c[:] = 0.0
        self.u_dot_c_star[:] = 0.0
        self.u_dot_c[:] = 0.0
        self._initialized = False

    def compute(self, s, s_dot, s_d, s_dot_d, L, L_inv, b, N, u_dot_o):
        K, B, H, T = self.K, self.B, self.H, self.T
        if not self._initialized:
            self.s_p_prv = np.asarray(s, dtype=float).reshape(self.n).copy()
            self._initialized = True

        # Eq. (42i): s_p* = (I + H/T)^-1 (s_d + H s_dot_d + H s_p,prv/T).
        self.s_p_star = (s_d + H * s_dot_d + H * self.s_p_prv / T) / (1.0 + H / T)

        # Eq. (42j): alpha_c* = (K + B/T)s_p* - Ks - B(s_dot + s_p,prv/T).
        self.alpha_c_star = (
            (K + B / T) * self.s_p_star
            - K * s
            - B * (s_dot + self.s_p_prv / T)
        )

        # Eq. (42k): u_dot_c* = L^-1(alpha_c* - b - N u_dot_o).
        self.u_dot_c_star = L_inv @ (self.alpha_c_star - b - N @ u_dot_o)

        # Eq. (42l): u_dot_c = Pi_A(u_dot_c*).
        # Translational and rotational 3D vectors are norm-clipped separately:
        # clip(x) = A x / max(||x||, A), not element-wise clipping.
        self.u_dot_c = self._clip_camera_acceleration(self.u_dot_c_star)

        # Eq. (42m): alpha_c = alpha_c* + L(u_dot_c - u_dot_c*).
        self.alpha_c = self.alpha_c_star + L @ (self.u_dot_c - self.u_dot_c_star)

        # Eq. (42n): s_p = s_p* + (K + B/T)^-1(alpha_c - alpha_c*).
        denom = K + B / T
        self.s_p = self.s_p_star + np.where(
            np.abs(denom) > 1e-15,
            (self.alpha_c - self.alpha_c_star) / denom,
            0.0,
        )
        self.s_p_prv = self.s_p.copy()
        return self.u_dot_c

    @property
    def is_accel_saturated(self):
        return (
            np.linalg.norm(self.u_dot_c_star[:3]) > float(self.A[0])
            or np.linalg.norm(self.u_dot_c_star[3:]) > float(self.A[3])
        )

    @property
    def proxy_position(self):
        return self.s_p.copy()

    @property
    def proxy_offset(self):
        return self.s_p - self.s_p_star
