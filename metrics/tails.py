import math

import torch


def tail_count(n_values: int, alpha: float) -> int:
    """Compute worst-tail count for VaR/CVaR: max(1, ceil((1-alpha) * n))."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}.")
    if n_values <= 0:
        raise ValueError(f"n_values must be positive; got {n_values}.")
    return max(1, math.ceil((1.0 - float(alpha)) * int(n_values)))


def worst_tail(values: torch.Tensor, alpha: float) -> torch.Tensor:
    """Return the worst tail (lowest values) used by robustness VaR/CVaR."""
    flat = values.flatten()
    if flat.numel() <= 0:
        raise ValueError("values must contain at least one element.")
    k = tail_count(int(flat.numel()), alpha)
    return torch.topk(flat, k, largest=False).values


def tail_var(values: torch.Tensor, alpha: float) -> torch.Tensor:
    """VaR over the worst tail: max(worst_tail(values, alpha))."""
    return worst_tail(values, alpha).max()


def tail_cvar(values: torch.Tensor, alpha: float) -> torch.Tensor:
    """CVaR over the worst tail: mean(worst_tail(values, alpha))."""
    return worst_tail(values, alpha).mean()
