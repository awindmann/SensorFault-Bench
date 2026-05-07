# TSMixer model.
#
# Reference: Chen et al., 2023
# Paper: https://arxiv.org/abs/2303.06053
# Repo: https://github.com/google-research/google-research/tree/master/tsmixer
# Upstream copyright: Copyright 2026 The Google Research Authors.
# Upstream license: Apache License, Version 2.0.
# Source provenance: translated from upstream Google Research
# `tsmixer_basic/models/tsmixer.py` into the repository's PyTorch Lightning
# style.

import torch.nn as nn

from models.base_module import BaseLitModule
from utils.parsing import require_tsmixer_hparams


class _FlattenBatchNorm1d(nn.Module):
    """Batch norm over flattened temporal-channel axis."""

    def __init__(
        self,
        *,
        d_seq_in: int,
        d_features: int,
        eps: float = 1e-3,
        momentum: float = 0.01,
    ) -> None:
        super().__init__()
        self.d_seq_in = d_seq_in
        self.d_features = d_features
        # Keras BatchNormalization defaults: epsilon=1e-3, momentum=0.99.
        # PyTorch uses the inverse momentum convention for running stats:
        # torch_momentum = 1 - keras_momentum.
        self.norm = nn.BatchNorm1d(
            d_seq_in * d_features,
            eps=eps,
            momentum=momentum,
        )

    def forward(self, x):
        batch_size = x.size(0)
        flat = x.reshape(batch_size, self.d_seq_in * self.d_features)
        flat = self.norm(flat)
        return flat.reshape(batch_size, self.d_seq_in, self.d_features)


class _MixerBlock(nn.Module):
    def __init__(
        self,
        *,
        d_seq_in: int,
        d_features: int,
        ff_dim: int,
        dropout: float,
        norm_type: str,
        activation: str,
    ) -> None:
        super().__init__()
        self.temporal_norm = _build_norm(
            norm_type=norm_type,
            d_seq_in=d_seq_in,
            d_features=d_features,
        )
        self.temporal_mixing = nn.Linear(d_seq_in, d_seq_in)
        self.temporal_activation = _build_activation(activation)
        self.temporal_dropout = nn.Dropout(dropout)

        self.feature_norm = _build_norm(
            norm_type=norm_type,
            d_seq_in=d_seq_in,
            d_features=d_features,
        )
        self.feature_fc1 = nn.Linear(d_features, ff_dim)
        self.activation = _build_activation(activation)
        self.feature_dropout1 = nn.Dropout(dropout)
        self.feature_fc2 = nn.Linear(ff_dim, d_features)
        self.feature_dropout2 = nn.Dropout(dropout)

        _init_dense_like_keras(self.temporal_mixing)
        _init_dense_like_keras(self.feature_fc1)
        _init_dense_like_keras(self.feature_fc2)

    def forward(self, x):
        temporal = self.temporal_norm(x)
        temporal = self.temporal_mixing(temporal.transpose(1, 2)).transpose(1, 2)
        temporal = self.temporal_activation(temporal)
        temporal = self.temporal_dropout(temporal)
        residual = x + temporal

        feature = self.feature_norm(residual)
        feature = self.feature_fc1(feature)
        feature = self.activation(feature)
        feature = self.feature_dropout1(feature)
        feature = self.feature_fc2(feature)
        feature = self.feature_dropout2(feature)
        return residual + feature


def _build_norm(*, norm_type: str, d_seq_in: int, d_features: int) -> nn.Module:
    if norm_type == "L":
        # Keras LayerNormalization default epsilon=1e-3.
        return nn.LayerNorm((d_seq_in, d_features), eps=1e-3)
    if norm_type == "B":
        return _FlattenBatchNorm1d(
            d_seq_in=d_seq_in,
            d_features=d_features,
            eps=1e-3,
            momentum=0.01,
        )
    raise ValueError("norm_type must be exactly 'L' or 'B'.")


def _build_activation(activation: str) -> nn.Module:
    if activation == "relu":
        return nn.ReLU()
    if activation == "gelu":
        return nn.GELU()
    raise ValueError("activation must be exactly 'relu' or 'gelu'.")


def _init_dense_like_keras(layer: nn.Linear) -> None:
    """Match Keras Dense defaults: glorot_uniform kernel + zero bias."""
    nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class TSMixer(BaseLitModule):
    def __init__(
        self,
        n_block=None,
        ff_dim=None,
        dropout=None,
        norm_type=None,
        activation=None,
        loss="MSE",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_architecture = "TSMixer"

        parsed = require_tsmixer_hparams(
            {
                "n_block": n_block,
                "ff_dim": ff_dim,
                "dropout": dropout,
                "norm_type": norm_type,
                "activation": activation,
            }
        )

        n_block = parsed["n_block"]
        ff_dim = parsed["ff_dim"]
        dropout_rate = parsed["dropout"]
        norm_type = parsed["norm_type"]
        activation_name = parsed["activation"]

        self.mixer_blocks = nn.ModuleList(
            [
                _MixerBlock(
                    d_seq_in=self.d_seq_in,
                    d_features=self.d_input_features,
                    ff_dim=ff_dim,
                    dropout=dropout_rate,
                    norm_type=norm_type,
                    activation=activation_name,
                )
                for _ in range(n_block)
            ]
        )
        self.temporal_projection = nn.Linear(self.d_seq_in, self.d_seq_out)
        _init_dense_like_keras(self.temporal_projection)
        if self.target_indices is None and self.d_input_features != self.d_output_features:
            self.output_projection = nn.Linear(
                self.d_input_features,
                self.d_output_features,
            )
            _init_dense_like_keras(self.output_projection)
        else:
            self.output_projection = nn.Identity()

        self.loss_fn = self._build_loss_fn(loss)

    def _shared_step(self, x, y):
        x = self._revin_norm_inputs(x)
        hidden = x
        for block in self.mixer_blocks:
            hidden = block(hidden)

        y_pred_raw = self.temporal_projection(hidden.transpose(1, 2)).transpose(1, 2)
        y_pred_raw = self.output_projection(y_pred_raw)
        y_pred = self.project_targets(y_pred_raw)
        y_pred = self._revin_denorm_targets(y_pred)

        loss = self.loss_fn(y_pred, y).mean() if y is not None else None
        return {
            "pred": y_pred,
            "target": y,
            "loss": loss,
        }
