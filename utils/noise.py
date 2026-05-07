"""Shared raw-space relative-noise transform for randomized training.

Reference: Yoon et al., 2022
Paper: https://proceedings.mlr.press/v151/yoon22a.html
Repo: https://github.com/tetrzim/robust-probabilistic-forecasting
"""

import torch


def apply_raw_space_noise(
    x: torch.Tensor,
    eps: torch.Tensor,
    noise_std: float,
    input_means: torch.Tensor,
    input_stds: torch.Tensor,
    channel_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply raw-space multiplicative relative noise (Yoon et al., 2022).

    Formula: destandardize -> x_raw * (1 + noise_std * eps) -> re-standardize.
    Handles 2D (time, channels) and 3D (batch, time, channels) inputs.
    """
    ndim = x.ndim
    if ndim == 2:
        view_shape = (1, -1)
    elif ndim == 3:
        view_shape = (1, 1, -1)
    else:
        raise ValueError(f"apply_raw_space_noise expects 2D or 3D input; got {ndim}D.")
    means = input_means.to(x.device, dtype=x.dtype).view(*view_shape)
    stds = input_stds.to(x.device, dtype=x.dtype).view(*view_shape)
    x_raw = x * stds + means
    x_raw_noisy = x_raw * (1 + noise_std * eps)
    if channel_mask is not None:
        mask = channel_mask.to(x.device, dtype=x.dtype).view(*view_shape)
        x_raw_noisy = x_raw * (1 - mask) + x_raw_noisy * mask
    return (x_raw_noisy - means) / stds
