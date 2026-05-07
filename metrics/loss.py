import torch
import torch.nn as nn
from typing import Callable, Dict


def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean Squared Error."""
    return (pred - target).pow(2).mean(dim=(1, 2))


def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean Absolute Error."""
    return (pred - target).abs().mean(dim=(1, 2))


def mape(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Mean Absolute Percentage Error."""
    return ((pred - target).abs() / (target.abs() + eps)).mean(dim=(1, 2))


def smape(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Symmetric Mean Absolute Percentage Error."""
    return (2 * (pred - target).abs() / (pred.abs() + target.abs() + eps)).mean(dim=(1, 2))


BASE_LOSSES: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "MSE": mse,
    "MAE": mae,
    "MAPE": mape,
    "SMAPE": smape,
}

BASE_LOSS_NAMES = tuple(BASE_LOSSES.keys())


def resolve_stateless_loss(
    loss_name: str,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    if not isinstance(loss_name, str):
        raise TypeError(f"loss_name must be a string, received {type(loss_name)}.")

    token = loss_name.strip()
    if not token:
        raise ValueError("loss_name must be a non-empty string.")

    key_upper = token.upper()

    # Case-insensitive lookup
    for name, fn in BASE_LOSSES.items():
        if name.upper() == key_upper:
            return fn

    if key_upper == "ADAPTIVEROBUSTLOSS":
        raise ValueError(
            "AdaptiveRobustLoss requires explicit rloss_* parameters and "
            "must be built via build_loss(), not resolve_stateless_loss()."
        )

    raise KeyError(f"Unknown loss function '{loss_name}'.")


# Type alias for loss objects: either a stateless callable or a trainable Module.
LossType = Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | nn.Module

RLOSS_PARAM_KEYS = (
    "rloss_alpha_lo",
    "rloss_alpha_hi",
    "rloss_alpha_init",
    "rloss_scale_lo",
    "rloss_scale_init",
    "rloss_param_scope",
)


def build_loss(
    loss_name: str,
    *,
    d_seq_out: int | None = None,
    d_target_features: int | None = None,
    rloss_alpha_lo: float | None = None,
    rloss_alpha_hi: float | None = None,
    rloss_alpha_init: float | None = None,
    rloss_scale_lo: float | None = None,
    rloss_scale_init: float | None = None,
    rloss_param_scope: str | None = None,
) -> LossType:
    """Build a loss function — stateless callable or trainable AdaptiveRobustLoss.

    For stateless losses, delegates to resolve_stateless_loss().
    For AdaptiveRobustLoss, constructs and returns the nn.Module.
    """
    if not isinstance(loss_name, str) or not loss_name.strip():
        raise ValueError(f"loss_name must be a non-empty string, got {loss_name!r}")

    is_adaptive = loss_name.strip().upper() == "ADAPTIVEROBUSTLOSS"
    rloss_kwargs = {
        "rloss_alpha_lo": rloss_alpha_lo,
        "rloss_alpha_hi": rloss_alpha_hi,
        "rloss_alpha_init": rloss_alpha_init,
        "rloss_scale_lo": rloss_scale_lo,
        "rloss_scale_init": rloss_scale_init,
        "rloss_param_scope": rloss_param_scope,
    }
    has_any_rloss = any(v is not None for v in rloss_kwargs.values())

    if is_adaptive:
        # Require ALL rloss_* params
        missing = [k for k, v in rloss_kwargs.items() if v is None]
        if missing:
            raise ValueError(
                f"AdaptiveRobustLoss requires all rloss_* parameters. "
                f"Missing: {', '.join(missing)}"
            )
        if d_seq_out is None or d_target_features is None:
            raise ValueError(
                "AdaptiveRobustLoss requires d_seq_out and d_target_features."
            )

        from metrics.adaptive_robust_loss import AdaptiveRobustLoss

        if str(rloss_param_scope) != "per_horizon_feature":
            raise ValueError(
                f"rloss_param_scope must be 'per_horizon_feature', got '{rloss_param_scope}'"
            )
        num_dims = d_seq_out * d_target_features

        # AdaptiveRobustLoss.__init__ validates all rloss_* params.
        return AdaptiveRobustLoss(
            num_dims=num_dims,
            alpha_lo=float(rloss_alpha_lo),
            alpha_hi=float(rloss_alpha_hi),
            alpha_init=float(rloss_alpha_init),
            scale_lo=float(rloss_scale_lo),
            scale_init=float(rloss_scale_init),
        )
    else:
        if has_any_rloss:
            present = [k for k, v in rloss_kwargs.items() if v is not None]
            raise ValueError(
                f"rloss_* parameters are only valid with loss='AdaptiveRobustLoss'. "
                f"Got loss='{loss_name}' with: {', '.join(present)}"
            )
        return resolve_stateless_loss(loss_name)
