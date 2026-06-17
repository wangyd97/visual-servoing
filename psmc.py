import numpy as np

from .Mathematic import to_vec6


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

    def _clip_camera_acceleration(self, x: np.ndarray) -> np.ndarray:
        # x = np.asarray(x, dtype=float).reshape(6)
        A_t = float(np.mean(self.A[:3]))
        A_r = float(np.mean(self.A[3:]))
        # Radial projection onto {a | ||a_t||^2/A_t^2 + ||a_r||^2/A_r^2 <= 1}.
        linear_ratio = 0.0 if A_t == float("inf") else float(np.linalg.norm(x[:3])) / A_t
        angular_ratio = 0.0 if A_r == float("inf") else float(np.linalg.norm(x[3:])) / A_r
        normalized_size = float(np.sqrt(linear_ratio**2 + angular_ratio**2))
        scale = max(normalized_size, 1.0)
        # print("scale", x/ scale)
        return x / scale
    

    def _clip_versa(self, x: np.ndarray, a: float, b: float) -> np.ndarray:
        clipped = np.zeros_like(x)
        lam = 1.0
        lam = min(lam, a / max(abs(x[0]),a))
        lam = min(lam, a / max(abs(x[1]),a))
        lam = min(lam, a / max(abs(x[2]),a))
        lam = min(lam, b / max(abs(x[3]),b))
        lam = min(lam, b / max(abs(x[4]),b))
        lam = min(lam, b / max(abs(x[5]),b))
        clipped = x * lam

        return clipped

    @staticmethod
    def _quat_from_qv(qv: np.ndarray) -> np.ndarray:
        """Build q=[qw,qx,qy,qz] from its vector part."""
        qv = np.asarray(qv, dtype=float).reshape(3).copy()
        qv_norm_sq = float(qv @ qv)
        if qv_norm_sq > 1.0:
            qv /= np.sqrt(qv_norm_sq + 1e-12)
            qv_norm_sq = float(qv @ qv)
        qw = np.sqrt(max(0.0, 1.0 - qv_norm_sq))
        return np.array([qw, qv[0], qv[1], qv[2]], dtype=float)

    @classmethod
    def _nearer_quat_vector(cls, qv_ref: np.ndarray, qv: np.ndarray) -> np.ndarray:
        q_ref = cls._quat_from_qv(qv_ref)
        q = cls._quat_from_qv(qv)
        if float(q_ref @ q) < 0.0:
            q = -q
        return q[1:4].copy()

    @classmethod
    def _make_rotation_feature_nearer(cls, x_ref: np.ndarray, x: np.ndarray) -> np.ndarray:
        out = np.asarray(x, dtype=float).reshape(-1).copy()
        out[3:6] = cls._nearer_quat_vector(x_ref[3:6], out[3:6])
        return out

    def reset(self):
        self.s_p_prv[:] = 0.0
        self.s_p_star[:] = 0.0
        self.s_p[:] = 0.0
        self.alpha_c_star[:] = 0.0
        self.alpha_c[:] = 0.0
        self.u_dot_c_star[:] = 0.0
        self.u_dot_c[:] = 0.0
        self._initialized = False

    def compute(self, s, s_dot, s_d, s_dot_d, L, L_inv, b, N, u_dot_o, u_c):
        # K, B, H are stored as diagonal entries in the config, and used here
        # as diagonal matrices to match the notation in Eq. (42).
        LIMITING_ALPHA = False
        # b=np.zeros(self.n)
        K = np.diag(self.K)
        B = np.diag(self.B)
        H = np.diag(self.H)
        I = np.eye(self.n)
        T = self.T
        if not self._initialized:
            # Reset the proxy feature by the measured feature s in the first loop.
            # This avoids starting Eq. (42i) from the zero proxy state after reset().
            measured_s = s.copy()

            
            self.s_p_prv = measured_s.copy()
            self.s_p_star = measured_s.copy()
            self.s_p = measured_s.copy()
            self._initialized = True

        # Eq : s_p* = (I + H/T)^-1 (s_d + H s_dot_d + (H/T)s_p,prv).
        self.s_p_star = self.s_p_prv + (
            s_d + self.H * s_dot_d  - self.s_p_prv
        ) / (1.0 + self.H / T)
        # Eq: choose quaternion sign closer to s_p,prv.
        self.s_p_star = self._make_rotation_feature_nearer(
            self.s_p_prv,
            self.s_p_star,
        )

        # Eq: alpha_c* = (K + B/T)s_p* - Ks - B(s_dot + s_p,prv/T).
        self.alpha_c_star = (
            (K + B / T) @ self.s_p_star
            - K @ s
            - B @ (s_dot + self.s_p_prv / T)
        )
        if LIMITING_ALPHA:
        # When setting the LIMITING_ALPHA=False, one limit the acceleration in feature space but not the task-space
            self.alpha_c = self._clip_camera_acceleration(self.alpha_c_star)
            self.u_dot_c = L_inv @ (self.alpha_c - b - N @ u_dot_o)
        else:
        # Eq: u_dot_c* = L^-1(alpha_c* - b - N u_dot_o).
            self.u_dot_c_star = L_inv @ (self.alpha_c_star - b - N @ u_dot_o)
        # Eq: u_dot_c = Pi_A(u_dot_c*).
            self.u_dot_c = self._clip_camera_acceleration(self.u_dot_c_star)
            tmp1 =u_c + T * self.u_dot_c
            # print(tmp1)
            tmp1 = self._clip_versa(tmp1,999,999)
            self.u_dot_c = (tmp1 - u_c) / T
        # Eq: alpha_c = alpha_c* + L(u_dot_c - u_dot_c*).
            self.alpha_c = self.alpha_c_star + L @ (self.u_dot_c - self.u_dot_c_star)

        # Eq: s_p = s_p* + (K + B/T)^-1(alpha_c - alpha_c*).
        self.s_p = self.s_p_star + np.linalg.solve(
            K + B / T,
            self.alpha_c - self.alpha_c_star,
        )
        # Eq: choose quaternion sign closer to s_p*.
        self.s_p = self._make_rotation_feature_nearer(
            self.s_p_star,
            self.s_p,
        )
        self.s_p_prv = self.s_p.copy()
        return self.u_dot_c

    @property
    def is_accel_saturated(self):
        A_t = float(np.mean(self.A[:3]))
        A_r = float(np.mean(self.A[3:]))
        linear_ratio = 0.0 if A_t == float("inf") else float(np.linalg.norm(self.u_dot_c_star[:3])) / A_t
        angular_ratio = 0.0 if A_r == float("inf") else float(np.linalg.norm(self.u_dot_c_star[3:])) / A_r
        return float(np.sqrt(linear_ratio**2 + angular_ratio**2)) > 1.0

    @property
    def proxy_position(self):
        return self.s_p.copy()

    @property
    def proxy_offset(self):
        return self.s_p - self.s_p_star
