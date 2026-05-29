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
        self.b_km1 = np.zeros(n); self.q_k = np.zeros(n)
        self.a_k = np.zeros(n); self.a_star_k = np.zeros(n)
        self._initialized = False

    def reset(self):
        self.b_km1[:] = 0; self.q_k[:] = 0
        self.a_k[:] = 0; self.a_star_k[:] = 0
        self._initialized = True

    def compute(self, p, pd, p_dot, pd_dot):
        K, B, H, T = self.K, self.B, self.H, self.T
        if not self._initialized:
            self.reset()
        sigma = (pd - p) + H * (pd_dot - p_dot)
        a_star = ((B + K*T) / (H + T)) * sigma + ((K*H - B) / (H + T)) * self.b_km1
        self.a_star_k = a_star.copy()
        a = np.clip(a_star, -self.A, self.A)
        self.a_k = a.copy()
        denom = B + K * T
        b_k = np.where(
            np.abs(denom) > 1e-15,
            (B * self.b_km1 + T * a) / denom,
            0.0
        )
        self.q_k = p + b_k
        self.b_km1 = b_k.copy()
        return a

    @property
    def is_accel_saturated(self):
        return bool(np.any(np.abs(self.a_star_k) > self.A))

    @property
    def proxy_position(self):
        return self.q_k.copy()
