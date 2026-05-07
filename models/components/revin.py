"""Reversible Instance Normalization layer.

Reference: Kim et al., 2021
Paper: https://openreview.net/forum?id=cGDAkQo1C0p
Repo: https://github.com/ts-kim/RevIN
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from utils.parsing import parse_feature_indices


class RevIN(nn.Module):
    """Reversible Instance Normalization (RevIN).

    Expected tensor layout is ``(batch, sequence, feature)``.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        if not isinstance(num_features, int):
            raise ValueError(f"num_features must be an int, got {type(num_features).__name__}.")
        if num_features <= 0:
            raise ValueError(f"num_features must be > 0, got {num_features}.")
        if not isinstance(affine, bool):
            raise ValueError(f"affine must be a bool, got {type(affine).__name__}.")
        try:
            eps_value = float(eps)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"eps must be a positive float, got {eps!r}.") from exc
        if eps_value <= 0.0:
            raise ValueError(f"eps must be > 0, got {eps_value}.")

        self.num_features = num_features
        self.eps = eps_value
        self.affine = affine
        self._mean: torch.Tensor | None = None
        self._stdev: torch.Tensor | None = None

        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.num_features))
            self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(
        self,
        x: torch.Tensor,
        mode: str,
        target_indices: Sequence[int] | None = None,
    ) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise ValueError("RevIN expects torch.Tensor input.")
        if x.ndim < 2:
            raise ValueError(
                f"RevIN expects input with at least 2 dimensions, got shape {tuple(x.shape)}."
            )

        mode_token = str(mode).strip()
        if mode_token == "norm":
            self._validate_feature_dim(x, expected_features=self.num_features, context="norm")
            self._cache_statistics(x)
            return self._normalize(x)
        if mode_token == "denorm":
            if self._mean is None or self._stdev is None:
                raise RuntimeError("RevIN denorm called before norm; normalization stats are missing.")
            mean, stdev, affine_weight, affine_bias = self._resolve_denorm_context(
                x, target_indices
            )
            return self._denormalize(x, mean, stdev, affine_weight, affine_bias)

        raise ValueError(f"Unknown RevIN mode '{mode_token}'. Expected 'norm' or 'denorm'.")

    @staticmethod
    def _validate_feature_dim(
        x: torch.Tensor,
        *,
        expected_features: int,
        context: str,
    ) -> None:
        actual = int(x.size(-1))
        if actual != expected_features:
            raise ValueError(
                f"RevIN {context} expects feature dimension {expected_features}, got {actual}."
            )

    def _cache_statistics(self, x: torch.Tensor) -> None:
        reduce_dims = tuple(range(1, x.ndim - 1))
        if len(reduce_dims) == 0:
            raise ValueError(
                "RevIN requires at least one non-batch/non-feature axis for normalization."
            )
        mean = torch.mean(x, dim=reduce_dims, keepdim=True)
        var = torch.var(x, dim=reduce_dims, keepdim=True, unbiased=False)
        self._mean = mean.detach()
        self._stdev = torch.sqrt(var + self.eps).detach()

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self._mean is None or self._stdev is None:
            raise RuntimeError("RevIN normalization stats are missing.")
        x_norm = (x - self._mean) / self._stdev
        if self.affine:
            x_norm = x_norm * self.affine_weight
            x_norm = x_norm + self.affine_bias
        return x_norm

    def _resolve_denorm_context(
        self,
        x: torch.Tensor,
        target_indices: Sequence[int] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if self._mean is None or self._stdev is None:
            raise RuntimeError("RevIN denormalization stats are missing.")
        predicted_features = int(x.size(-1))

        if target_indices is not None:
            indices = parse_feature_indices(
                target_indices,
                n_features=self.num_features,
                key="target_indices",
            )
            if predicted_features != len(indices):
                raise ValueError(
                    "RevIN denorm feature mismatch: predicted feature dimension "
                    f"{predicted_features} does not match target subset length {len(indices)}."
                )

            index_tensor = torch.as_tensor(indices, device=x.device, dtype=torch.long)
            mean_subset = torch.index_select(self._mean, dim=-1, index=index_tensor)
            stdev_subset = torch.index_select(self._stdev, dim=-1, index=index_tensor)
            affine_weight_subset = None
            affine_bias_subset = None
            if self.affine:
                affine_weight_subset = torch.index_select(
                    self.affine_weight,
                    dim=0,
                    index=index_tensor,
                )
                affine_bias_subset = torch.index_select(
                    self.affine_bias,
                    dim=0,
                    index=index_tensor,
                )
            return mean_subset, stdev_subset, affine_weight_subset, affine_bias_subset

        if predicted_features != self.num_features:
            raise ValueError(
                "RevIN denorm feature mismatch: predicted feature dimension "
                f"{predicted_features} does not match full input dimension {self.num_features} "
                "and no target_indices were provided."
            )
        affine_weight = self.affine_weight if self.affine else None
        affine_bias = self.affine_bias if self.affine else None
        return self._mean, self._stdev, affine_weight, affine_bias

    def _denormalize(
        self,
        x: torch.Tensor,
        mean: torch.Tensor,
        stdev: torch.Tensor,
        affine_weight: torch.Tensor | None,
        affine_bias: torch.Tensor | None,
    ) -> torch.Tensor:
        x_denorm = x
        if self.affine:
            if affine_weight is None or affine_bias is None:
                raise RuntimeError("RevIN affine parameters are missing for denormalization.")
            x_denorm = x_denorm - affine_bias
            x_denorm = x_denorm / (affine_weight + self.eps * self.eps)
        x_denorm = x_denorm * stdev
        x_denorm = x_denorm + mean
        return x_denorm


__all__ = ["RevIN"]
