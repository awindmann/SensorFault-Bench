# ModernTCN model.
#
# Reference: Luo and Wang, 2024
# Paper: https://openreview.net/forum?id=vpJMJerXHU
# Repo: https://github.com/luodhhh/ModernTCN
# Upstream license: MIT License.
# Source provenance: compact long-term-forecasting implementation adapted to
# the repository's PyTorch Lightning style.

import torch
import torch.nn as nn

from models.base_module import BaseLitModule
from utils.parsing import (
    parse_required_bool,
    parse_required_dropout,
    parse_required_odd_positive_int,
    parse_required_positive_int,
)

_UNSUPPORTED_MODERNTCN_KEYS = frozenset(
    {
        "dims",
        "dw_dims",
        "stem_ratio",
        "downsample_ratio",
        "use_multi_scale",
        "small_kernel_merged",
        "call_structural_reparam",
        "decomposition",
        "kernel_size",
        "revin",
        "affine",
        "subtract_last",
        "freq",
    }
)


class ReparamLargeKernelConv(nn.Module):
    def __init__(
        self,
        *,
        channels: int,
        large_size: int,
        small_size: int,
    ) -> None:
        super().__init__()
        self.large_branch = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=large_size,
                stride=1,
                padding=large_size // 2,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm1d(channels),
        )
        self.small_branch = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=small_size,
                stride=1,
                padding=small_size // 2,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm1d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.large_branch(x) + self.small_branch(x)


class ModernTCNBlock(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        n_vars: int,
        ffn_ratio: int,
        large_size: int,
        small_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.n_vars = int(n_vars)
        self.d_ff = int(d_model) * int(ffn_ratio)

        self.temporal_conv = ReparamLargeKernelConv(
            channels=self.n_vars * self.d_model,
            large_size=large_size,
            small_size=small_size,
        )
        self.temporal_norm = nn.BatchNorm1d(self.d_model)

        self.ffn_features_in = nn.Conv1d(
            in_channels=self.n_vars * self.d_model,
            out_channels=self.n_vars * self.d_ff,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=self.n_vars,
        )
        self.ffn_features_activation = nn.GELU()
        self.ffn_features_out = nn.Conv1d(
            in_channels=self.n_vars * self.d_ff,
            out_channels=self.n_vars * self.d_model,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=self.n_vars,
        )
        self.ffn_features_dropout_1 = nn.Dropout(dropout)
        self.ffn_features_dropout_2 = nn.Dropout(dropout)

        self.ffn_variables_in = nn.Conv1d(
            in_channels=self.n_vars * self.d_model,
            out_channels=self.n_vars * self.d_ff,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=self.d_model,
        )
        self.ffn_variables_activation = nn.GELU()
        self.ffn_variables_out = nn.Conv1d(
            in_channels=self.n_vars * self.d_ff,
            out_channels=self.n_vars * self.d_model,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=self.d_model,
        )
        self.ffn_variables_dropout_1 = nn.Dropout(dropout)
        self.ffn_variables_dropout_2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        batch_size, n_vars, d_model, patch_count = x.shape

        x = x.reshape(batch_size, n_vars * d_model, patch_count)
        x = self.temporal_conv(x)
        x = x.reshape(batch_size * n_vars, d_model, patch_count)
        x = self.temporal_norm(x)
        x = x.reshape(batch_size, n_vars * d_model, patch_count)
        x = self.ffn_features_dropout_1(self.ffn_features_in(x))
        x = self.ffn_features_activation(x)
        x = self.ffn_features_dropout_2(self.ffn_features_out(x))
        x = x.reshape(batch_size, n_vars, d_model, patch_count)

        x = x.permute(0, 2, 1, 3)
        x = x.reshape(batch_size, d_model * n_vars, patch_count)
        x = self.ffn_variables_dropout_1(self.ffn_variables_in(x))
        x = self.ffn_variables_activation(x)
        x = self.ffn_variables_dropout_2(self.ffn_variables_out(x))
        x = x.reshape(batch_size, d_model, n_vars, patch_count)
        x = x.permute(0, 2, 1, 3)

        return residual + x


class ModernTCNForecastHead(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        patch_count: int,
        d_seq_out: int,
        n_vars: int,
        individual: bool,
        head_dropout: float,
    ) -> None:
        super().__init__()
        self.n_vars = int(n_vars)
        self.individual = bool(individual)
        self._flattened_dim = int(d_model) * int(patch_count)

        if self.individual:
            self.linears = nn.ModuleList(
                [nn.Linear(self._flattened_dim, int(d_seq_out)) for _ in range(self.n_vars)]
            )
            self.dropouts = nn.ModuleList(
                [nn.Dropout(head_dropout) for _ in range(self.n_vars)]
            )
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(self._flattened_dim, int(d_seq_out))
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.individual:
            outputs = []
            for idx in range(self.n_vars):
                flattened = x[:, idx, :, :].reshape(x.size(0), self._flattened_dim)
                projected = self.linears[idx](flattened)
                outputs.append(self.dropouts[idx](projected))
            return torch.stack(outputs, dim=1)

        flattened = self.flatten(x)
        projected = self.linear(flattened)
        return self.dropout(projected)


class ModernTCN(BaseLitModule):
    def __init__(
        self,
        d_model=None,
        num_blocks=None,
        large_size=None,
        small_size=None,
        ffn_ratio=None,
        patch_size=None,
        patch_stride=None,
        dropout=None,
        head_dropout=None,
        individual=None,
        loss="MSE",
        **kwargs,
    ) -> None:
        unsupported_keys = sorted(kwargs.keys() & _UNSUPPORTED_MODERNTCN_KEYS)
        if unsupported_keys:
            raise ValueError(
                "ModernTCN received unsupported upstream-specific argument(s): "
                + ", ".join(unsupported_keys)
                + "."
            )
        super().__init__(**kwargs)
        self.model_architecture = "ModernTCN"

        self.d_model = parse_required_positive_int(d_model, key="d_model")
        self.num_blocks = parse_required_positive_int(num_blocks, key="num_blocks")
        self.large_size = parse_required_odd_positive_int(large_size, key="large_size")
        self.small_size = parse_required_odd_positive_int(small_size, key="small_size")
        self.ffn_ratio = parse_required_positive_int(ffn_ratio, key="ffn_ratio")
        self.patch_size = parse_required_positive_int(patch_size, key="patch_size")
        self.patch_stride = parse_required_positive_int(patch_stride, key="patch_stride")
        self.dropout = parse_required_dropout(dropout, key="dropout")
        self.head_dropout = parse_required_dropout(head_dropout, key="head_dropout")
        self.individual = parse_required_bool(
            individual,
            key="individual",
            context="ModernTCN",
        )

        if self.patch_stride > self.patch_size:
            raise ValueError(
                "patch_stride must be <= patch_size; got "
                f"patch_stride={self.patch_stride}, patch_size={self.patch_size}."
            )
        if self.patch_size > self.d_seq_in:
            raise ValueError(
                f"patch_size must be <= d_seq_in; got patch_size={self.patch_size}, "
                f"d_seq_in={self.d_seq_in}."
            )
        if self.small_size > self.large_size:
            raise ValueError(
                "small_size must be <= large_size; got "
                f"small_size={self.small_size}, large_size={self.large_size}."
            )

        self.patch_count = self.d_seq_in // self.patch_stride
        self.stem = nn.Sequential(
            nn.Conv1d(
                1,
                self.d_model,
                kernel_size=self.patch_size,
                stride=self.patch_stride,
                bias=True,
            ),
            nn.BatchNorm1d(self.d_model),
        )
        self.blocks = nn.ModuleList(
            [
                ModernTCNBlock(
                    d_model=self.d_model,
                    n_vars=self.d_input_features,
                    ffn_ratio=self.ffn_ratio,
                    large_size=self.large_size,
                    small_size=self.small_size,
                    dropout=self.dropout,
                )
                for _ in range(self.num_blocks)
            ]
        )
        self.forecast_head = ModernTCNForecastHead(
            d_model=self.d_model,
            patch_count=self.patch_count,
            d_seq_out=self.d_seq_out,
            n_vars=self.d_input_features,
            individual=self.individual,
            head_dropout=self.head_dropout,
        )
        if self.d_input_features == self.d_output_features:
            self.output_projection = None
        else:
            self.output_projection = nn.Linear(
                self.d_input_features,
                self.d_output_features,
            )
        self.loss_fn = self._build_loss_fn(loss)

    def _validate_input(self, x: torch.Tensor) -> None:
        if x.dim() != 3:
            raise ValueError(
                f"ModernTCN expects input with shape (B, T, D); got {tuple(x.shape)}."
            )
        if x.size(1) != self.d_seq_in:
            raise ValueError(
                f"ModernTCN expects sequence length {self.d_seq_in}, got {x.size(1)}."
            )
        if x.size(2) != self.d_input_features:
            raise ValueError(
                "ModernTCN expects "
                f"{self.d_input_features} input features, got {x.size(2)}."
            )

    def encode_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input(x)
        normalized = self._revin_norm_inputs(x)
        batch_size = normalized.size(0)

        stem_input = normalized.permute(0, 2, 1).reshape(
            batch_size * self.d_input_features,
            1,
            self.d_seq_in,
        )
        pad_len = self.patch_size - self.patch_stride
        if pad_len > 0:
            stem_input = nn.functional.pad(stem_input, (0, pad_len), mode="replicate")

        stem_output = self.stem(stem_input)
        if stem_output.size(-1) != self.patch_count:
            raise ValueError(
                "ModernTCN stem patch axis does not match configured patch_count: "
                f"{stem_output.size(-1)} != {self.patch_count}."
            )

        features = stem_output.reshape(
            batch_size,
            self.d_input_features,
            self.d_model,
            self.patch_count,
        )
        for block in self.blocks:
            features = block(features)
        return features

    def _shared_step(self, x, y):
        features = self.encode_backbone_features(x)
        y_pred_raw = self.forecast_head(features).permute(0, 2, 1)

        if self.output_projection is not None:
            y_pred_raw = self.output_projection(y_pred_raw)

        y_pred = self.project_targets(y_pred_raw)
        y_pred = self._revin_denorm_targets(y_pred)
        loss = self.loss_fn(y_pred, y).mean() if y is not None else None
        return {
            "pred": y_pred,
            "target": y,
            "loss": loss,
        }
