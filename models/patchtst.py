# PatchTST model.
#
# Reference: Nie et al., 2023
# Paper: https://openreview.net/forum?id=Jbdc0vTOcol
# Repo: https://github.com/thuml/Time-Series-Library
# Upstream license: MIT License.
# Source provenance: adapted to this repository using local
# Time-Series-Library-derived components.

import torch
import torch.nn as nn

from models.base_module import BaseLitModule
from models.components.attention import AttentionLayer, FullAttention
from models.components.embedding import PatchEmbedding
from models.components.encoder_decoder import Encoder, EncoderLayer
from utils.parsing import (
    parse_required_choice,
    parse_required_dropout,
    parse_required_positive_int,
)


class FlattenHead(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        patch_count: int,
        d_seq_out: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(int(d_model) * int(patch_count), int(d_seq_out))
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_vars, d_model, patch_count)
        x = self.flatten(x)
        x = self.linear(x)
        return self.dropout(x)


class PatchTST(BaseLitModule):
    def __init__(
        self,
        d_model=None,
        d_ff=None,
        n_layers_enc=None,
        n_heads=None,
        patch_len=None,
        stride=None,
        dropout=None,
        factor=None,
        activation=None,
        loss="MSE",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_architecture = "PatchTST"

        self.d_model = parse_required_positive_int(d_model, key="d_model")
        self.d_ff = parse_required_positive_int(d_ff, key="d_ff")
        self.n_layers_enc = parse_required_positive_int(n_layers_enc, key="n_layers_enc")
        self.n_heads = parse_required_positive_int(n_heads, key="n_heads")
        self.patch_len = parse_required_positive_int(patch_len, key="patch_len")
        self.stride = parse_required_positive_int(stride, key="stride")
        self.factor = parse_required_positive_int(factor, key="factor")
        self.dropout = parse_required_dropout(dropout, key="dropout")
        self.activation = parse_required_choice(
            activation,
            key="activation",
            allowed=("gelu", "relu"),
        )
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model must be divisible by n_heads; got d_model={self.d_model}, "
                f"n_heads={self.n_heads}."
            )
        if self.patch_len > self.d_seq_in:
            raise ValueError(
                f"patch_len must be <= d_seq_in; got patch_len={self.patch_len}, "
                f"d_seq_in={self.d_seq_in}."
            )

        self.patch_count = ((self.d_seq_in + self.stride - self.patch_len) // self.stride) + 1

        self.patch_embedding = PatchEmbedding(
            d_model=self.d_model,
            patch_len=self.patch_len,
            stride=self.stride,
            padding=self.stride,
            patch_num=self.patch_count,
            dropout=self.dropout,
        )
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            self.factor,
                            attention_dropout=self.dropout,
                            output_attention=False,
                            res_attention=True,
                        ),
                        self.d_model,
                        self.n_heads,
                    ),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation,
                    norm="BatchNorm",
                )
                for _ in range(self.n_layers_enc)
            ],
        )
        self.flatten_head = FlattenHead(
            d_model=self.d_model,
            patch_count=self.patch_count,
            d_seq_out=self.d_seq_out,
            dropout=self.dropout,
        )
        if self.d_input_features == self.d_output_features:
            self.output_projection = None
        else:
            self.output_projection = nn.Linear(
                self.d_input_features,
                self.d_output_features,
            )
        self._norm_eps = 1e-5
        self.loss_fn = self._build_loss_fn(loss)

    def _shared_step(self, x, y):
        # x: (B, d_seq_in, d_input_features)
        encoded_tokens, n_vars, means, stdev = self._encode_patch_tokens_with_stats(x)

        batch_size = x.size(0)
        encoded = encoded_tokens.reshape(
            batch_size,
            n_vars,
            encoded_tokens.size(1),
            encoded_tokens.size(2),
        )
        encoded = encoded.permute(0, 1, 3, 2)
        y_pred_raw = self.flatten_head(encoded).permute(0, 2, 1)

        y_pred_raw = y_pred_raw * stdev
        y_pred_raw = y_pred_raw + means

        if self.output_projection is not None:
            y_pred_raw = self.output_projection(y_pred_raw)

        y_pred = self.project_targets(y_pred_raw)
        loss = self.loss_fn(y_pred, y).mean() if y is not None else None
        return {
            "pred": y_pred,
            "target": y,
            "loss": loss,
        }

    def _encode_patch_tokens_with_stats(self, x: torch.Tensor):
        """Return encoded patch tokens and normalization stats."""
        means = x.mean(1, keepdim=True).detach()
        centered = x - means
        stdev = torch.sqrt(
            torch.var(centered, dim=1, keepdim=True, unbiased=False) + self._norm_eps
        )
        stdev = stdev.detach()
        normalized = centered / stdev

        patch_input = normalized.permute(0, 2, 1)
        patch_tokens, n_vars = self.patch_embedding(patch_input)
        encoded_tokens, _ = self.encoder(patch_tokens, attn_mask=None)
        if encoded_tokens.size(1) != self.patch_count:
            raise ValueError(
                "PatchTST encoder patch axis does not match configured patch_count: "
                f"{encoded_tokens.size(1)} != {self.patch_count}."
            )
        return encoded_tokens, n_vars, means, stdev
