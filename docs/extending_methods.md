# Extending Methods

Robustness-improvement methods are selected through `pipeline_method` values and
recipe YAML.

## Owner Paths

1. Add a recipe under `configs/pipelines/`.
2. Register the benchmark recipe lookup in `pipelines/recipes.py`.
3. Register or extend the implementation in `improvements/` when the method is
   a wrapper or reusable improvement.
4. Use the existing train-kind, finetune-kind, or wrap-kind runner paths.
5. Add the method to `BENCHMARK_METHODS` in `configs/defaults.yaml` only if it
   should be part of default benchmark execution.
6. Add display labels and method-architecture applicability to
   `configs/benchmark_scope.yaml` only when the method is benchmark.
7. Update reporting labels in `configs/reporting/core_figures.yaml` when the
   method should appear in benchmark analysis figures.

## Benchmark Built-Ins

The benchmark currently ships:

- `baseline`
- `randomized_training`
- `adversarial_training`
- `adaptive_robust_loss`
- `ensemble`
- `randomized_smoothing`
- `fault_augmentation`
- `revin`

## Run Identity

Every selectable run must log stable identity tags. These values are used for
training lineage, testing dispatch, model selection, and analysis joins.

| Field | Meaning |
| --- | --- |
| `pipeline_method` | Method family selected by CLI and benchmark scope, such as `baseline` or `ensemble`. |
| `pipeline_id` | Concrete recipe variant within a method family. Include training-affecting parameter choices here. |
| `pipeline_kind` | Runner and loader contract: `train`, `finetune`, or `wrap`. |
| `robustness_method` | Evaluation and reporting method key. Use `baseline` for baseline runs and match `pipeline_method` for non-baseline runs. |

Put human-readable names in `configs/benchmark_scope.yaml` or reporting config,
not in these tags. Changing identity tags breaks selection, lineage checks,
checkpoint loading, and compatibility with existing MLflow runs.

## Implementation Rules

- Training-affecting recipe params must be encoded in `pipeline_id`, unless
  they are fixed singleton fairness knobs.
- Wrapper methods must keep `loader_kind` metadata explicit so testing dispatch
  can reconstruct the selected run.
- Selection remains validation-only. Final reporting remains test-only.
- Baseline and improvement comparisons must preserve `data_config_signature`
  comparability.
- Missing baselines, missing tags, unsupported pairs, and unknown methods must
  raise before execution.
- Do not add compatibility shims, aliases, or fallback loaders for unknown
  method names.
