# Extending Methods

A robustness-improvement method changes how a forecasting model is trained or
combines fitted models. The benchmark compares every method with the baseline
forecasting model of the same dataset and architecture. A method consists of a
recipe in `configs/pipelines/` and, when the existing options do not express
it, code that the recipe switches on.

## Method Kinds

The `pipeline_kind` of a recipe selects how the pipeline runs it.

| Kind | Effect | Where new behavior goes | Built-in examples |
| --- | --- | --- | --- |
| `train` | Changes how a model is fitted | `datamodule_kwargs` reach `TSDataModule` in [`data/data_module.py`](../data/data_module.py), and `model_kwargs` reach the model and `BaseLitModule` in [`models/base_module.py`](../models/base_module.py) | `randomized_training`, `adversarial_training`, `adaptive_robust_loss`, `revin`, `fault_augmentation` |
| `wrap` | Combines or wraps selected baseline forecasting models without training them again | A wrapper in [`improvements/`](../improvements/), for example [`improvements/ensemble.py`](../improvements/ensemble.py) | `ensemble`, `randomized_smoothing` |

A third kind, `finetune`, continues training from the selected baseline
forecasting model.
No built-in method uses it.

## Worked Example

The recipe below defines a method that trains with the mean absolute error
instead of the mean squared error. It needs no new code, because the loss is
an existing constructor argument of every model.

<!-- method-recipe-example:start -->
```yaml
# configs/pipelines/mae_training.yaml
# Train with the mean absolute error instead of the mean squared error.
pipeline_id: "mae_training"
pipeline_method: "mae_training"
pipeline_kind: "train"

recipe_params: {}

model_kwargs:
  loss: "MAE"

model_hparams:
  mode: "inherit_baseline"
```
<!-- method-recipe-example:end -->

| Field | Meaning |
| --- | --- |
| `pipeline_method` | Method key used on the command line and in the configuration files |
| `pipeline_id` | Name of one recipe variant. A placeholder such as `{smoothing_noise_std}` inserts the value of a recipe parameter, and every parameter that changes training must appear in it |
| `pipeline_kind` | `train`, `wrap`, or `finetune`, as described above |
| `recipe_params` | Parameters of the method, each with a `default` and an optional search `grid` |
| `datamodule_kwargs`, `model_kwargs` | Values passed to the data module or to the model. A string such as `"{smoothing_noise_std}"` inserts a recipe parameter |
| `model_hparams.mode` | `inherit_baseline` reuses the hyperparameters of the selected baseline of the same architecture, so the search covers only the recipe parameters. The baseline itself uses `baseline_grid` |

[`configs/pipelines/randomized_training.yaml`](../configs/pipelines/randomized_training.yaml)
shows a recipe with a parameter grid, and
[`configs/pipelines/ensemble.yaml`](../configs/pipelines/ensemble.yaml) shows a
`wrap` recipe.

## Registration

1. Save the recipe under `configs/pipelines/` and add the method key to
   `PIPELINE_RECIPE_PATHS_BY_METHOD` in
   [`pipelines/recipes.py`](../pipelines/recipes.py).
2. Add the key to `BENCHMARK_METHODS` in
   [`configs/defaults.yaml`](../configs/defaults.yaml). The entry points load
   the recipes of this list, so a method runs only when it is listed. Runs
   without `--method` then include it as well.
3. Add the key to `methods.display_order` and `methods.display_labels` in
   [`configs/benchmark_scope.yaml`](../configs/benchmark_scope.yaml), and list
   the architectures it applies to under
   `methods.method_architecture_applicability`.
4. Add a display label to `CORE_METHOD_DISPLAY` in
   [`configs/reporting/core_figures.yaml`](../configs/reporting/core_figures.yaml).
   Analysis computes paired deltas only for methods listed there.
5. Add a group with one help entry per recipe parameter to
   [`configs/cli_help.yaml`](../configs/cli_help.yaml). Because every recipe
   parameter becomes a command-line option, the entry points raise an error
   when one lacks help text.

A first run on ETTh1 checks the integration in a few minutes on a CPU:

```bash
uv run python run_training.py --data-files ETTh1 --model DLinear --method baseline mae_training --max-hp-trials-per-model 1 --max-epochs 1
uv run python run_testing.py --data-files ETTh1 --model DLinear --method baseline mae_training --max-hp-trials-per-model 1
uv run python run_analysis.py --data-files ETTh1 --model DLinear --method baseline mae_training --max-hp-trials-per-model 1
```

The analysis table `tables/improvement_deltas_selected.csv` then holds the
paired deltas of the method. The full evaluation and the comparison with the
default configuration follow [Evaluating a new model or
method](../README.md#evaluating-a-new-model-or-method).

## Included Methods

The benchmark ships:

- `baseline`
- `randomized_training`
- `adversarial_training`
- `adaptive_robust_loss`
- `ensemble`
- `randomized_smoothing`
- `fault_augmentation`
- `revin`

## Run Identity

Every selectable run logs identity tags, which training, testing, model
selection, and analysis use to match runs and trace their lineage.

| Field | Meaning |
| --- | --- |
| `pipeline_method` | Method key that the command line and the benchmark scope select, such as `baseline` or `ensemble` |
| `pipeline_id` | Recipe variant within the method, naming every parameter that affects training |
| `pipeline_kind` | How the pipeline trains and loads the run: `train`, `finetune`, or `wrap` |
| `robustness_method` | Method key in evaluation and reporting, `baseline` for baseline runs and equal to `pipeline_method` otherwise |

Display names belong in `configs/benchmark_scope.yaml` and the reporting
configuration, not in these tags. Changing an identity tag breaks
selection, lineage checks, checkpoint loading, and compatibility with existing
MLflow runs.

## Rules for Comparable Methods

- Every recipe parameter that affects training appears in `pipeline_id`,
  unless all variants fix it to the same value.
- Wrapper methods keep their `loader_kind` metadata explicit, so testing can
  rebuild the selected run.
- Selection uses validation data only, while reporting uses test data only.
- A method and its paired baseline share the same `data_config_signature`.
- Missing baselines, missing tags, unsupported pairs, and unknown methods raise
  before execution. Unknown method names have no aliases or fallback loaders.

## Paired Transfer Inputs

Fault augmentation is the evaluated use of the training-only transfer fault
families, but a method can also use them as paired supervision. Each transfer
operator takes a clean input window and returns the corrupted window, the
unchanged target, and the affected channels. A method can train on such pairs,
for example to reconstruct the clean history before forecasting or to keep
internal representations consistent between the clean and corrupted inputs.
The scored scenarios in `PERTURBATION_SCENARIOS` stay reserved for evaluation,
so a method that trains on pairs must draw them from the transfer families
only.

The snippet below builds one pair per transfer family from the recipe in
`configs/pipelines/fault_augmentation.yaml`. Its input windows are standardized
and indexed by time step and channel, with all randomness drawn from the
explicit generator.

<!-- paired-transfer-inputs:start -->
```python
import torch
import yaml

from data.perturbations import PERTURBATION_REGISTRY

with open("configs/pipelines/fault_augmentation.yaml", encoding="utf-8") as handle:
    recipe = yaml.safe_load(handle)
transfer_families = recipe["train_fault_profiles"]["holdout"]["scenarios"]

generator = torch.Generator().manual_seed(0)
clean = torch.randn(96, 7, generator=generator)
target = torch.randn(96, 7, generator=generator)
continuous_channels = list(range(clean.shape[1]))

pairs = []
for family in transfer_families:
    operator = PERTURBATION_REGISTRY[family](channel_frac=0.25)
    corrupted, unchanged_target, channels = operator(
        clean, target, 0.8, generator, continuous_channels, [],
    )
    pairs.append(
        {"family": family, "clean": clean, "corrupted": corrupted, "channels": channels}
    )
```
<!-- paired-transfer-inputs:end -->
