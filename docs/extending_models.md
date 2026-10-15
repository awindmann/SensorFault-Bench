# Extending Models

A forecasting architecture is a PyTorch Lightning module that the benchmark
trains, selects, and evaluates in the same way as the built-in architectures.
This guide describes the interface, works through a small linear model, lists
the configuration entries that register it, and shows how to evaluate a model
whose weights were trained elsewhere.

## Interface

A model inherits from `BaseLitModule` in
[`models/base_module.py`](../models/base_module.py). The pipeline constructs it
with the shape of the forecasting task and the values of one configuration
from the search grid.

| Constructor argument | Meaning |
| --- | --- |
| `d_input_features` | Number of input channels |
| `d_target_features` | Number of forecast targets |
| `d_seq_in` | Input window length |
| `d_seq_out` | Forecast horizon |
| `target_indices` | Positions of the forecast targets among the input channels |
| `loss` | Name of the training loss, `MSE` in the benchmark |
| Further keyword arguments | Values of one configuration from the search grid |

The model implements `_shared_step(x, y)`. It receives the input `x` of shape
(batch, `d_seq_in`, `d_input_features`) and the targets `y`, which are `None`
when the model only forecasts. The method returns a dictionary with the forecast
`pred` of shape (batch, `d_seq_out`, `d_target_features`) and the mean loss
`loss`, which is `None` without targets. The base class provides training,
validation, checkpoints, and the Adam optimizer. Evaluation calls `forward()`,
which returns `_shared_step(x, None)["pred"]`, so the scored faults reach the
model only through `x`.

Two helpers of the base class keep a model compatible with the benchmark.
`project_targets` selects the forecast targets when the model forecasts every
input channel. A model that calls `_revin_norm_inputs` on its input and
`_revin_denorm_targets` on its forecast also supports the RevIN method.

## Worked Example

The model below maps the input history of each channel linearly to its
forecast. If the targets are not among the input channels, a second linear
layer maps the input channels to the targets.

<!-- model-example:start -->
```python
from torch import nn

from models.base_module import BaseLitModule


class LinearForecaster(BaseLitModule):
    """Map the history of each input channel linearly to its forecast."""

    def __init__(self, loss="MSE", **kwargs):
        super().__init__(**kwargs)
        self.model_architecture = "LinearForecaster"
        self.time_projection = nn.Linear(self.d_seq_in, self.d_seq_out)
        if self.d_output_features == self.d_input_features:
            self.channel_projection = nn.Identity()
        else:
            self.channel_projection = nn.Linear(self.d_input_features, self.d_output_features)
        self.loss_fn = self._build_loss_fn(loss)

    def _shared_step(self, x, y):
        x = self._revin_norm_inputs(x)
        y_pred = self.time_projection(x.transpose(1, 2)).transpose(1, 2)
        y_pred = self.channel_projection(y_pred)
        y_pred = self.project_targets(y_pred)
        y_pred = self._revin_denorm_targets(y_pred)
        loss = self.loss_fn(y_pred, y).mean() if y is not None else None
        return {"pred": y_pred, "loss": loss}
```
<!-- model-example:end -->

`model_architecture` repeats the class name, which serves as the architecture
key in the configuration files and on the command line. `_build_loss_fn`
returns one loss value per window, so `_shared_step` averages it.

## Registration

1. Save the class as `models/linear_forecaster.py` and export it from
   [`models/__init__.py`](../models/__init__.py) with
   `from .linear_forecaster import LinearForecaster` and an entry in
   `__all__`.
2. Add the search grid to
   [`configs/baseline_hparams.yaml`](../configs/baseline_hparams.yaml). Every
   key becomes a constructor argument, including the learning rate `lr` that a
   trainable model needs:

   ```yaml
   LinearForecaster:
     lr: [0.001, 0.0001]
   ```

3. Register the architecture in
   [`configs/benchmark_scope.yaml`](../configs/benchmark_scope.yaml), which
   decides where it may run and how analysis labels it. Under `architectures`,
   add the key to `display_order`, a label to `display_labels`, a role under
   `roles`, and a family under `families`. The role `method_comparison`
   applies the robustness-improvement methods to the architecture, while
   `baseline_only_reference` keeps it a reference. Under
   `methods.method_architecture_applicability`, add the key to `baseline` and
   to every method that should apply to it:

   ```yaml
   architectures:
     display_order: [DLinear, GRU, ModernTCN, PatchTST, TSMixer, SeasonalNaive, Chronos2, LinearForecaster]
     display_labels:
       LinearForecaster: "Linear forecaster"
     roles:
       method_comparison: [DLinear, GRU, ModernTCN, PatchTST, TSMixer, LinearForecaster]
     families:
       LinearForecaster: "Fully Connected"
   methods:
     method_architecture_applicability:
       baseline: [DLinear, GRU, ModernTCN, PatchTST, TSMixer, SeasonalNaive, Chronos2, LinearForecaster]
   ```

4. Add the key to `BENCHMARK_ARCHITECTURES` in
   [`configs/defaults.yaml`](../configs/defaults.yaml) only if runs without
   `--model` should include it.

A first run on ETTh1 checks the integration in a few minutes on a CPU:

```bash
uv run python run_training.py --data-files ETTh1 --model LinearForecaster --method baseline --max-hp-trials-per-model 1 --max-epochs 1
uv run python run_testing.py --data-files ETTh1 --model LinearForecaster --method baseline --max-hp-trials-per-model 1
uv run python run_analysis.py --data-files ETTh1 --model LinearForecaster --method baseline --max-hp-trials-per-model 1
```

The full evaluation and the comparison with the default configuration follow
[Evaluating a new model or
method](../README.md#evaluating-a-new-model-or-method).

## Models Trained Elsewhere

A model whose weights were trained outside the benchmark enters in one of two
ways. To fine-tune it, a trainable model such as `LinearForecaster` loads the
weights in its constructor, and the search then trains from them. To evaluate
it as it is, the model keeps its weights fixed, as in the example below. Its
search grid passes the path of the weights file:

```yaml
PretrainedLinear:
  weights_path: ["/path/to/pretrained_linear.pt"]
```

<!-- pretrained-model-example:start -->
```python
import torch

from models.linear_forecaster import LinearForecaster


class PretrainedLinear(LinearForecaster):
    """Evaluate a LinearForecaster whose weights were trained elsewhere."""

    uses_base_optimizer = False
    max_fit_epochs = 1

    def __init__(self, weights_path, loss="MSE", **kwargs):
        super().__init__(loss=loss, **kwargs)
        self.model_architecture = "PretrainedLinear"
        self.load_state_dict(torch.load(weights_path, map_location="cpu"))
        self.automatic_optimization = False

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.0)

    def training_step(self, batch, _):
        x, y = batch
        loss = self._shared_step(x, y)["loss"]
        self.log("train_loss", loss)
        # A step without gradients leaves the weights unchanged and advances the
        # global step, so Lightning saves the checkpoint that testing loads.
        self.optimizers().step()
        return loss
```
<!-- pretrained-model-example:end -->

`uses_base_optimizer = False` removes the optimizer settings of the
benchmark, and `max_fit_epochs = 1` limits training to the single pass that
records the validation loss and saves the checkpoint. Testing rebuilds the
model from its hyperparameters before it loads that checkpoint, so the weights
file must still exist at the same path. The architecture takes the role
`baseline_only_reference` in `configs/benchmark_scope.yaml`.

The training data of such a model must exclude the test part of every
benchmark dataset. A report on the model also names the fault operators that
it saw in training (see
[Fault Exposure in Training](benchmark_protocol.md#fault-exposure-in-training)).

## Included Architectures

The benchmark ships these architecture keys:

- `DLinear`
- `GRU`
- `ModernTCN`
- `PatchTST`
- `TSMixer`
- `SeasonalNaive`
- `Chronos2`

`SeasonalNaive` and `Chronos2` are reference architectures that take only the
baseline. The robustness-improvement methods apply to `DLinear`, `GRU`,
`ModernTCN`, `PatchTST`, and `TSMixer`.
