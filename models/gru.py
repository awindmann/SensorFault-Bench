import torch
import torch.nn as nn

from models.base_module import BaseLitModule



class GRU(BaseLitModule):
    """GRU model.
    Args:
        d_hidden (int): hidden dimension
        n_layers (int): number of layers
        bidirectional (bool): whether to use bidirectional GRU
        dropout (float): dropout rate
        autoregressive (bool): whether to predict one output at a time
        loss (str): name of the loss function, defaults to MSE
    """
    def __init__(self, d_hidden, n_layers=1, bidirectional=False, dropout=0.5, autoregressive=False, loss="MSE", **kwargs):
        super().__init__(**kwargs)
        self.model_architecture = "GRU"

        self.gru = nn.GRU(
            input_size=self.d_input_features,
            hidden_size=d_hidden,
            num_layers=n_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
        
        if not autoregressive:
            self.fc = nn.Linear(d_hidden * (bidirectional + 1), self.d_output_features * self.d_seq_out)
        else:
            self.fc = nn.Linear(d_hidden * (bidirectional + 1), self.d_input_features)
        self.autoregressive = autoregressive
        if self.autoregressive and self.target_indices is None:
            raise ValueError("Autoregressive GRU requires targets to be part of the input feature set.")

        self.loss_fn = self._build_loss_fn(loss)

    def _extract_top_hidden(self, h):
        num_directions = 2 if self.gru.bidirectional else 1
        h = h.view(
            self.gru.num_layers,
            num_directions,
            h.size(1),
            self.gru.hidden_size,
        )
        h_top = h[-1]
        if num_directions == 2:
            return torch.cat([h_top[0], h_top[1]], dim=1)
        return h_top[0]

    def decode(self, x):
        if self.autoregressive:
            raise ValueError("GRU decode does not support autoregressive=True.")
        b_size = x.size(0)
        _, h = self.gru(x)
        h = self._extract_top_hidden(h)
        return self.fc(h).view(b_size, self.d_seq_out, self.d_output_features)

    def _shared_step(self, x, y):
        x = self._revin_norm_inputs(x)
        b_size = x.size(0)

        if not self.autoregressive:
            y_pred_raw = self.decode(x)
            y_pred = self.project_targets(y_pred_raw)
        else:
            y_pred_steps = []
            _, h = self.gru(x)
            gru_input = x[:, -1:, :]
            raw_step = self.fc(self._extract_top_hidden(h)).unsqueeze(1)
            proj_step = self.project_targets(raw_step)
            y_pred_steps.append(proj_step)
            gru_input = self.prepare_autoregressive_input(proj_step, gru_input)
            for _ in range(self.d_seq_out - 1):
                _, h = self.gru(gru_input, h)
                raw_step = self.fc(self._extract_top_hidden(h)).unsqueeze(1)
                proj_step = self.project_targets(raw_step)
                y_pred_steps.append(proj_step)
                gru_input = self.prepare_autoregressive_input(proj_step, gru_input)
            y_pred = torch.cat(y_pred_steps, dim=1)
        y_pred = self._revin_denorm_targets(y_pred)
        
        loss = self.loss_fn(y_pred, y).mean() if y is not None else None
        return {
            "pred": y_pred,
            "target": y,
            "loss": loss,
        }
