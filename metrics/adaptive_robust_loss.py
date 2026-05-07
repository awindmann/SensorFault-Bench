"""Barron's adaptive robust loss (learned per forecast coordinate [horizon, feature]).

Reference: Barron, 2019
Paper: https://arxiv.org/abs/1701.03077
Repo: https://github.com/jonbarron/robust_loss_pytorch
"""

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_SOFTPLUS_SHIFT = math.log(math.expm1(1.0))


def _log_safe(x: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.clamp_max(x, 33e37))


def _log1p_safe(x: torch.Tensor) -> torch.Tensor:
    return torch.log1p(torch.clamp_max(x, 33e37))


def affine_sigmoid(logits: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Map reals to (lo, hi), where 0 maps to (lo+hi)/2."""
    if not lo < hi:
        raise ValueError(f"`lo` ({lo}) must be < `hi` ({hi})")
    return torch.sigmoid(logits) * (hi - lo) + lo


def inv_affine_sigmoid(probs: float, lo: float, hi: float) -> torch.Tensor:
    """Inverse of affine_sigmoid."""
    if not lo < hi:
        raise ValueError(f"`lo` ({lo}) must be < `hi` ({hi})")
    return torch.logit(torch.tensor((probs - lo) / (hi - lo)))


def affine_softplus(x: torch.Tensor, lo: float, ref: float) -> torch.Tensor:
    """Map reals to (lo, inf), where 0 maps to ref."""
    if not lo < ref:
        raise ValueError(f"`lo` ({lo}) must be < `ref` ({ref})")
    return (ref - lo) * nn.functional.softplus(x + _SOFTPLUS_SHIFT) + lo


def _interpolate1d(
    x: torch.Tensor,
    values: torch.Tensor,
    tangents: torch.Tensor,
) -> torch.Tensor:
    """1D cubic Hermite spline interpolation with linear extrapolation."""
    x_lo = torch.floor(
        torch.clamp(x, 0, values.shape[0] - 2)
    ).to(torch.int64)
    x_hi = x_lo + 1
    t = x - x_lo.to(x.dtype)

    t_sq = t**2
    t_cu = t * t_sq
    h01 = -2.0 * t_cu + 3.0 * t_sq
    h00 = 1.0 - h01
    h11 = t_cu - t_sq
    h10 = h11 - t_sq + t

    value_before = tangents[0] * t + values[0]
    value_after = tangents[-1] * (t - 1.0) + values[-1]

    value_mid = (
        values[x_lo] * h00
        + values[x_hi] * h01
        + tangents[x_lo] * h10
        + tangents[x_hi] * h11
    )
    return torch.where(
        t < 0.0, value_before, torch.where(t > 1.0, value_after, value_mid)
    )


_SPLINE_CPU: dict[str, torch.Tensor] | None = None
_SPLINE_CACHE: dict[tuple[torch.dtype, torch.device], dict[str, torch.Tensor]] = {}


def _load_spline_cpu() -> dict[str, torch.Tensor]:
    global _SPLINE_CPU
    if _SPLINE_CPU is not None:
        return _SPLINE_CPU
    spline_path = Path(__file__).parent / "resources" / "partition_spline.npz"
    with np.load(spline_path, allow_pickle=False) as f:
        _SPLINE_CPU = {
            "x_scale": torch.tensor(f["x_scale"]),
            "values": torch.tensor(f["values"]),
            "tangents": torch.tensor(f["tangents"]),
        }
    return _SPLINE_CPU


def _get_spline_for(x: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return spline data on the same dtype/device as x, with caching."""
    key = (x.dtype, x.device)
    cached = _SPLINE_CACHE.get(key)
    if cached is not None:
        return cached
    base = _load_spline_cpu()
    cached = {k: v.to(dtype=x.dtype, device=x.device) for k, v in base.items()}
    _SPLINE_CACHE[key] = cached
    return cached


def _partition_spline_curve(alpha: torch.Tensor) -> torch.Tensor:
    """Non-linear transform of alpha for spline lookup."""
    return torch.where(
        alpha < 4,
        (2.25 * alpha - 4.5) / (torch.abs(alpha - 2) + 0.25) + alpha + 2,
        5.0 / 18.0 * _log_safe(4 * alpha - 15) + 8,
    )


def log_base_partition_function(alpha: torch.Tensor) -> torch.Tensor:
    """Spline-backed approximation to log Z(alpha)."""
    spline = _get_spline_for(alpha)
    x = _partition_spline_curve(alpha)
    return _interpolate1d(
        x * spline["x_scale"],
        spline["values"],
        spline["tangents"],
    )


def barron_lossfun(
    x: torch.Tensor, alpha: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Barron's general robust loss rho(x, alpha, scale)."""
    squared_scaled_x = (x / scale) ** 2
    loss_two = 0.5 * squared_scaled_x
    loss_zero = _log1p_safe(0.5 * squared_scaled_x)

    machine_epsilon = torch.tensor(
        np.finfo(np.float32).eps,
        device=x.device,
        dtype=x.dtype,
    )
    beta_safe = torch.maximum(machine_epsilon, torch.abs(alpha - 2.0))
    alpha_safe = torch.where(alpha >= 0, 1.0, -1.0) * torch.maximum(
        machine_epsilon,
        torch.abs(alpha),
    )
    loss_otherwise = (beta_safe / alpha_safe) * (
        torch.pow(squared_scaled_x / beta_safe + 1.0, 0.5 * alpha) - 1.0
    )
    return torch.where(
        alpha == 0,
        loss_zero,
        torch.where(alpha == 2, loss_two, loss_otherwise),
    )


def nll(
    x: torch.Tensor, alpha: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Negative log-likelihood of the Barron distribution."""
    loss = barron_lossfun(x, alpha, scale)
    log_partition = torch.log(scale) + log_base_partition_function(alpha)
    return loss + log_partition


def validate_rloss_params(
    *,
    alpha_lo: float,
    alpha_hi: float,
    alpha_init: float,
    scale_lo: float,
    scale_init: float,
    param_scope: str,
) -> None:
    """Validate adaptive robust loss configuration. Raises on invalid values."""
    for name, val in [
        ("rloss_alpha_lo", alpha_lo),
        ("rloss_alpha_hi", alpha_hi),
        ("rloss_alpha_init", alpha_init),
        ("rloss_scale_lo", scale_lo),
        ("rloss_scale_init", scale_init),
    ]:
        if not isinstance(val, (int, float)) or not np.isfinite(val):
            raise ValueError(f"{name} must be a finite number, got {val!r}")

    if alpha_lo <= 0:
        raise ValueError(f"rloss_alpha_lo must be > 0, got {alpha_lo}")
    if alpha_hi >= 2:
        raise ValueError(f"rloss_alpha_hi must be < 2, got {alpha_hi}")
    if alpha_lo >= alpha_hi:
        raise ValueError(
            f"rloss_alpha_lo ({alpha_lo}) must be < rloss_alpha_hi ({alpha_hi})"
        )
    if not (alpha_lo < alpha_init < alpha_hi):
        raise ValueError(
            f"rloss_alpha_init ({alpha_init}) must be in "
            f"(rloss_alpha_lo, rloss_alpha_hi) = ({alpha_lo}, {alpha_hi})"
        )
    if scale_lo <= 0:
        raise ValueError(f"rloss_scale_lo must be > 0, got {scale_lo}")
    if scale_init <= scale_lo:
        raise ValueError(
            f"rloss_scale_init ({scale_init}) must be > rloss_scale_lo ({scale_lo})"
        )
    if param_scope != "per_horizon_feature":
        raise ValueError(
            f"rloss_param_scope must be 'per_horizon_feature', got '{param_scope}'"
        )


class AdaptiveRobustLoss(nn.Module):
    """Trainable adaptive robust loss with learned per-dimension shape and scale.

    forward(pred, target) -> Tensor[batch], matching stateless loss callables.
    Internals run in float32 regardless of input precision.
    """

    def __init__(
        self,
        num_dims: int,
        alpha_lo: float,
        alpha_hi: float,
        alpha_init: float,
        scale_lo: float,
        scale_init: float,
    ):
        super().__init__()
        validate_rloss_params(
            alpha_lo=alpha_lo,
            alpha_hi=alpha_hi,
            alpha_init=alpha_init,
            scale_lo=scale_lo,
            scale_init=scale_init,
            param_scope="per_horizon_feature",
        )
        if num_dims < 1:
            raise ValueError(f"num_dims must be >= 1, got {num_dims}")

        self.num_dims = num_dims
        self.alpha_lo = alpha_lo
        self.alpha_hi = alpha_hi
        self.scale_lo = scale_lo
        self.scale_init = scale_init

        latent_alpha_init = inv_affine_sigmoid(alpha_init, lo=alpha_lo, hi=alpha_hi)
        self.latent_alpha = nn.Parameter(
            latent_alpha_init.detach()
            .float()
            .unsqueeze(0)
            .expand(1, num_dims)
            .clone(),
            requires_grad=True,
        )
        self.latent_scale = nn.Parameter(
            torch.zeros(1, num_dims, dtype=torch.float32),
            requires_grad=True,
        )

        _load_spline_cpu()

    def get_alpha(self) -> torch.Tensor:
        """Effective alpha values, shape [1, num_dims]."""
        return affine_sigmoid(self.latent_alpha, lo=self.alpha_lo, hi=self.alpha_hi)

    def get_scale(self) -> torch.Tensor:
        """Effective scale values, shape [1, num_dims]."""
        return affine_softplus(self.latent_scale, lo=self.scale_lo, ref=self.scale_init)

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute per-batch mean adaptive NLL loss."""
        residual = pred - target
        batch_size = residual.shape[0]

        residual_flat = residual.reshape(batch_size, -1)
        if residual_flat.shape[1] != self.num_dims:
            raise ValueError(
                f"Expected {self.num_dims} forecast dimensions, "
                f"got {residual_flat.shape[1]} "
                f"(input shape {residual.shape})"
            )

        orig_dtype = residual_flat.dtype
        residual_fp32 = residual_flat.float()

        alpha = self.get_alpha()
        scale = self.get_scale()

        loss_per_element = nll(residual_fp32, alpha, scale)
        loss_per_batch = loss_per_element.mean(dim=1)

        return loss_per_batch.to(orig_dtype)
