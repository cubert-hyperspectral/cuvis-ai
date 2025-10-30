import torch
import torch.nn as nn

from cuvis_ai.node import LabelLike, MetaLike, Node, NodeOutput
from cuvis_ai.utils.torch import _flatten_bhwc


# ---------- Shared base ----------
class RXBase(Node):
    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        super().__init__(eps=eps)

    @property
    def input_dim(self) -> tuple[int, int, int, int]:
        return (-1, -1, -1, -1)  # B,H,W,C

    @property
    def output_dim(self) -> tuple[int, int, int, int]:
        return (-1, -1, -1, 1)  # B,H,W,1

    @staticmethod
    def _quad_form_solve(Xc: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
        """
        Xc: (B,N,C) centered; cov: (C,C) or (B,C,C)
        returns md2: (B,N)
        """
        B, N, C = Xc.shape
        rhs = Xc.transpose(1, 2)  # (B,C,N)
        covB = cov if cov.dim() == 3 else cov.unsqueeze(0).expand(B, C, C)
        y = torch.linalg.solve(covB, rhs)  # (B,C,N)
        md2 = (rhs * y).sum(dim=1)  # (B,N)
        return md2

    # def serialize(self) -> Dict[str, Any]:
    #     return {"class": f"{self.__class__.__module__}.{self.__class__.__name__}",
    #             "config": {"eps": self.eps},
    #             "state_dict": self.state_dict()}

    # def load(self, payload: Dict[str, Any]) -> None:
    #     cfg = payload.get("config", {})
    #     self.eps = cfg.get("eps", self.eps)
    #     self.load_state_dict(payload.get("state_dict", {}), strict=False)


# ---------- Trained/global variant ----------
class RXGlobal(RXBase):
    """
    Uses global μ, Σ (estimated from train). Optionally trainable.
    """

    def __init__(
        self, eps: float = 1e-6, trainable_stats: bool = False, cache_inverse: bool = True
    ):
        super().__init__(eps)
        self.trainable_stats = trainable_stats
        self.cache_inverse = cache_inverse
        # global stats
        self.register_buffer("mu", None)  # (C,)
        self.register_buffer("cov", None)  # (C,C)
        self.register_buffer("cov_inv", None)  # (C,C)
        # streaming accumulators
        self.register_buffer("_mean", None)
        self.register_buffer("_M2", None)
        self._n = 0
        self._fitted = False

    @torch.no_grad()
    def update(self, batch_bhwc: torch.Tensor):
        X = _flatten_bhwc(batch_bhwc).reshape(-1, batch_bhwc.shape[-1])  # (M,C)
        m = X.shape[0]
        if m <= 1:
            return
        mean_b = X.mean(0)
        M2_b = (X - mean_b).T @ (X - mean_b)
        if self._n == 0:
            self._n, self._mean, self._M2 = m, mean_b, M2_b
        else:
            n, tot = self._n, self._n + m
            delta = mean_b - self._mean
            new_mean = self._mean + delta * (m / tot)
            outer = torch.outer(delta, delta) * (n * m / tot)
            self._n, self._mean, self._M2 = tot, new_mean, self._M2 + M2_b + outer
        self._fitted = False

    @torch.no_grad()
    def finalize(self):
        if self._n <= 1:
            raise ValueError("Not enough samples to finalize.")
        mu = self._mean.clone()
        cov = self._M2 / (self._n - 1)
        if self.eps > 0:
            cov = cov + self.eps * torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
        # assign as params or buffers
        if self.trainable_stats:
            self.mu = nn.Parameter(mu, requires_grad=True)
            self.cov = nn.Parameter(cov, requires_grad=True)
            self.cov_inv = (
                nn.Parameter(torch.linalg.pinv(cov), requires_grad=True)
                if self.cache_inverse
                else None
            )
        else:
            self.mu, self.cov = mu, cov
            self.cov_inv = torch.linalg.pinv(cov) if self.cache_inverse else None
        self._fitted = True
        return self

    @torch.no_grad()
    def fit(self, data_bhwc: torch.Tensor):
        self.reset()
        self.update(data_bhwc)
        self.finalize()
        return self

    def reset(self):
        self.mu = None
        self.cov = None
        self.cov_inv = None
        self._n = 0
        self._mean = None
        self._M2 = None
        self._fitted = False

    def forward(self, x_bhwc: torch.Tensor) -> dict[str, torch.Tensor]:
        if not self._fitted or self.mu is None:
            raise RuntimeError("RXGlobal not finalized. Call update()/finalize() or fit().")
        B, H, W, C = x_bhwc.shape
        N = H * W
        X = x_bhwc.view(B, N, C)
        Xc = X - self.mu.to(X.dtype).to(X.device)
        if self.cov_inv is not None:
            cov_inv = self.cov_inv.to(X.dtype).to(X.device)
            md2 = torch.einsum("bnc,cd,bnd->bn", Xc, cov_inv, Xc)  # (B,N)
        else:
            md2 = self._quad_form_solve(Xc, self.cov.to(X.dtype).to(X.device))
        scores = md2.view(B, H, W)
        return {"out": scores}


# ---------- Per-batch/stateless variant ----------
class RXPerBatch(RXBase):
    """
    Computes μ, Σ per image in the batch on the fly; no fit/finalize.
    """

    def forward(
        self,
        x: torch.Tensor,
        y: LabelLike = None,
        m: MetaLike = None,
        **_: MetaLike,
    ) -> NodeOutput:
        B, H, W, C = x.shape
        N = H * W
        X_flat = _flatten_bhwc(x)  # (B,N,C)
        mu = X_flat.mean(1, keepdim=True)  # (B,1,C)
        Xc = X_flat - mu
        cov = torch.matmul(Xc.transpose(1, 2), Xc) / max(N - 1, 1)  # (B,C,C)
        if self.eps > 0:
            eye = torch.eye(C, device=x.device, dtype=x.dtype).expand(B, C, C)
            cov = cov + self.eps * eye
        md2 = self._quad_form_solve(Xc, cov)  # (B,N)
        scores = md2.view(B, H, W)
        return scores.unsqueeze(-1), y, m

    def serialize(self, serial_dir: str) -> dict:
        return {
            "config": self._config,
            "state_dict": self.rx.state_dict(),
        }

    def load(self, params: dict, serial_dir: str) -> None:
        config = params.get("config", {})
        if config and config != self._config:
            self.rx = RXPerBatch(**config)
            self._config = dict(config)
        state = params.get("state_dict", {})
        if state:
            self.rx.load_state_dict(state, strict=False)
