# DLinear model.
#
# Reference: Zeng et al., 2023
# Paper: https://arxiv.org/abs/2205.13504
# Repo: https://github.com/thuml/Time-Series-Library
# Upstream license: MIT License.
# Source provenance: adapted from the Time-Series-Library DLinear
# implementation into the repository's PyTorch Lightning style.

import torch
import torch.nn as nn

from models.base_module import BaseLitModule



class DLinear(BaseLitModule):
    """DLinear model.
    Args:
        moving_avg: int, moving average window size
        individual: bool, whether to use individual weights for each feature
        init_weights: bool, whether to initialize weights uniformly
        loss: str, loss function
    """
    def __init__(self, moving_avg=25, individual=False, init_weights=False,
                 loss="MSE", **kwargs):
        super().__init__(**kwargs)
        self.model_architecture = "DLinear"
        
        self.moving_avg = moving_avg
        self.decompsition = series_decomp(self.moving_avg)
        self.individual = individual
        self.init_weights = init_weights

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.d_input_features):
                self.Linear_Seasonal.append(
                    nn.Linear(self.d_seq_in, self.d_seq_out)
                )
                self.Linear_Trend.append(
                    nn.Linear(self.d_seq_in, self.d_seq_out)
                )
                if self.init_weights:
                    self.Linear_Seasonal[i].weight = nn.Parameter(
                        (1 / self.d_seq_in) * torch.ones([self.d_seq_out, self.d_seq_in])
                    )
                    self.Linear_Trend[i].weight = nn.Parameter(
                        (1 / self.d_seq_in) * torch.ones([self.d_seq_out, self.d_seq_in])
                    )
        else:
            self.Linear_Seasonal = nn.Linear(self.d_seq_in, self.d_seq_out)
            self.Linear_Trend = nn.Linear(self.d_seq_in, self.d_seq_out)

            if self.init_weights:
                self.Linear_Seasonal.weight = nn.Parameter(
                    (1 / self.d_seq_in) * torch.ones([self.d_seq_out, self.d_seq_in])
                )
                self.Linear_Trend.weight = nn.Parameter(
                    (1 / self.d_seq_in) * torch.ones([self.d_seq_out, self.d_seq_in])
                )

        self.loss_fn = self._build_loss_fn(loss)
        if self.d_output_features == self.d_input_features:
            self.output_projection = nn.Identity()
        else:
            self.output_projection = nn.Linear(self.d_input_features, self.d_output_features)

    def _shared_step(self, x, y):
        # x: (batch_size, d_seq_in, d_features)
        # y: (batch_size, d_seq_out, d_features)
        x = self._revin_norm_inputs(x)
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)

        if self.individual:
            seasonal_output = torch.zeros(
                [seasonal_init.size(0), seasonal_init.size(1), self.d_seq_out],
                dtype=seasonal_init.dtype
            ).to(seasonal_init.device)
            trend_output = torch.zeros(
                [trend_init.size(0), trend_init.size(1), self.d_seq_out],
                dtype=trend_init.dtype
            ).to(trend_init.device)

            for i in range(self.d_input_features):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](
                    seasonal_init[:, i, :]
                )
                trend_output[:, i, :] = self.Linear_Trend[i](
                    trend_init[:, i, :]
                )
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        y_pred = seasonal_output + trend_output
        y_pred = y_pred.permute(0, 2, 1).contiguous()
        y_pred = self.output_projection(y_pred)
        y_pred = self.project_targets(y_pred)
        y_pred = self._revin_denorm_targets(y_pred)

        loss = self.loss_fn(y_pred, y).mean() if y is not None else None
        return {
            "pred": y_pred,
            "target": y,
            "loss": loss,
        }


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean
