# Reproducing the Benchmark

This guide describes how to run the full benchmark and where its outputs go.
The workflow reads the dataset keys from `configs/defaults.yaml` and the display
labels of the benchmark scope from `configs/benchmark_scope.yaml`.

## Setup

```bash
uv sync
```

Place the four benchmark datasets under `DATA_ROOT`, which defaults to
`data/processed`. [`data/README.md`](../data/README.md) describes how to
acquire, preprocess, and validate them.

## Source Validation

The pytest suite under `tests/` checks the benchmark code and the worked
examples of the guides:

```bash
uv run pytest -q
```

## Full Benchmark Commands

Train every dataset, architecture, and applicable method of the benchmark:

```bash
uv run python run_training.py
```

This command uses the benchmark defaults. The batch size of each dataset comes
from `configs/dataset_windows.yaml`, which sets 64 for `BeijingAir_Tiantan` and
`Penmanshiel_Hourly_WT08` and 16 for `ETTh1` and `traffic`.

Evaluate the selected models under the scored scenarios:

```bash
uv run python run_testing.py --full-coverage
```

Build the tables and figures across datasets from the tested models:

```bash
uv run python run_analysis.py --full-coverage
```

By default, testing and analysis accept partial coverage. With
`--full-coverage`, they raise an error when a run of the benchmark scope is
missing.

For a first run on a small scope, use the smoke benchmark of the README
[quick start](../README.md#quick-start). It trains one DLinear baseline on
ETTh1 for one epoch and evaluates it under two scenarios.

## Command-Line Options

`uv run python run_training.py --help` lists every option with its purpose and
default, grouped by topic and with one group per method recipe.
`run_testing.py` and `run_analysis.py` accept the same options.
The help text lives in `configs/cli_help.yaml`, while the defaults live in
`configs/defaults.yaml` and in the recipes under `configs/pipelines/`. A recipe
option given on the command line fixes that parameter instead of searching its
grid.

## Method Execution Order

`BENCHMARK_METHODS` in `configs/defaults.yaml` lists the methods that
`python run_training.py` runs without `--method`:

1. `baseline`
2. `randomized_training`
3. `adversarial_training`
4. `adaptive_robust_loss`
5. `ensemble`
6. `randomized_smoothing`
7. `fault_augmentation`
8. `revin`

Training runs `baseline` first, because the other methods build on its runs,
and the wrapping methods `ensemble` and `randomized_smoothing` last, because
they wrap fitted models. The remaining methods run in the listed order.

At full scope, training applies each method only to the architectures that
`configs/benchmark_scope.yaml` lists for it. The baseline covers all seven
architectures, whereas the robustness-improvement methods cover the five
method-comparison architectures, with `revin` excluding `PatchTST`.

## Runtime Profiles

Local profile:

```bash
uv run python run_training.py --data-root data/processed --logdir runs
```

VM or object-store profile:

```bash
uv run python run_training.py \
  --data-root s3://<bucket>/<prefix> \
  --logdir <mlflow-tracking-uri> \
  --minio-endpoint <object-store-endpoint>
```

The local profile keeps the empty default of `MINIO_ENDPOINT` in
`configs/defaults.yaml`. A MinIO store needs `--minio-endpoint`, whereas S3
stores that rely on provider defaults can leave it empty. Credentials come from
the shell, the environment, or a secret manager of the VM, never from committed
YAML. Changing `DATA_ROOT`, `LOGDIR`, `MINIO_ENDPOINT`, or credentials does not
change the data configuration signature or the run identity.

## Artifacts

Training and testing log MLflow runs under `LOGDIR`, which defaults to the
local store `runs`. `run_testing.py` attaches the degradation results to each
tested run, including the clean samples, scenario samples, and scenario
summaries.

`run_analysis.py` creates a meta-analysis MLflow run with:

- `tables/analysis.csv`
- `tables/full_results.csv`
- `tables/improvement_deltas_selected.csv`
- `tables/method_scenario_family_delta.csv`
- `tables/method_scenario_family_summary.csv`
- `tables/pipeline_method_delta_results.csv`
- `tables/scenario_samples.csv`
- `tables/testing_coverage.csv`
- `tables/figures_manifest.csv`
- `tables/forecast_extremes.csv` when the rendering of extreme forecasts is
  enabled and extreme samples can be rendered
- `figures/...`
- `config/meta_analysis_args.yaml`

Exact figure filenames are recorded in `tables/figures_manifest.csv`.

## Results Schema

`tables/full_results.csv` has one row per tested model. The key columns are:

| Column | Meaning |
| --- | --- |
| `dataset`, `model_architecture` | Dataset and architecture keys |
| `robustness_method`, `pipeline_id` | `baseline` or the method key, and the selected recipe variant |
| `run_id` | MLflow run of the selected model |
| `data_config_signature` | Hash of the data configuration that determines the splits |
| `eval_data_seed` | Seed of the sampled test windows and the fault randomness |
| `MSE_test` | Clean MSE |
| `D_w`, `D_w_CI_lo`, `D_w_CI_hi` | Worst-scenario degradation and its 95% bootstrap interval |
| `MSE_w` | Worst-scenario fault-time MSE. `err_pert_ws` holds the same value, with its interval in `err_pert_ws_CI_lo` and `err_pert_ws_CI_hi` |
| `worst_scenario` | Key of the scenario with the largest degradation |
| `D_mean`, `err_pert_mean` | Means of scenario degradation and scenario MSE over the scored scenarios, reported as diagnostics |
| `n_test_samples` | Number of sampled test windows |
| `robustness_scoring_semantics` | Severity sampling distribution of the evaluation, for example `uniform_severity_degradation` |

`tables/improvement_deltas_selected.csv` has one row per selected
robustness-improved variant and architecture. The key columns are:

| Column | Meaning |
| --- | --- |
| `dataset`, `model_architecture`, `robustness_method`, `pipeline_id` | Identity of the robustness-improved variant |
| `run_id`, `matched_baseline_run_id` | Runs of the robustness-improved variant and its paired baseline forecasting model |
| `MSE_test_improved`, `D_w_improved`, `MSE_w_improved` | Quantities of the robustness-improved variant |
| `MSE_test_baseline`, `D_w_baseline`, `MSE_w_baseline` | Quantities of the paired baseline forecasting model |
| `delta_MSE_test`, `delta_D_w`, `delta_MSE_w` | Paired deltas, method minus baseline. Negative values indicate improvement |
| `pct_change_MSE_test`, `pct_change_D_w`, `pct_change_MSE_w` | Paired deltas in percent of the baseline value |

`tables/pipeline_method_delta_results.csv` averages the paired deltas of each
method over architectures per dataset, with bootstrap intervals over the
architecture pairs.

## Paper Artifact

The result bundle of the NeurIPS 2026 paper is committed under
`paper_artifact/`. Its guide,
[`paper_artifact/README.md`](../paper_artifact/README.md), maps each section of
the paper to its bundled results, the commands that reproduce them, and the
compute accounting of its experiments. `MANIFEST.json` stores the SHA-256
checksum and byte size of each bundled file.

The optional folder `paper_artifact/R/` contains the R and TikZ scripts of the
paper figures, their curated inputs, and the rendered PDFs. Its R and TeX
dependencies, which `uv sync` does not install, are documented in
`paper_artifact/R/README.md`.

[`forecast_plot_replay.md`](forecast_plot_replay.md) regenerates the forecast
plots of single test windows from tested runs or from exported traces.

## Troubleshooting

**Accelerator selection.** `--accelerator` defaults to `auto`, which lets
PyTorch Lightning pick an available accelerator. Pass `--accelerator cpu` to
force CPU execution, or `--accelerator gpu --devices 1` to require a GPU. The
benchmark trains on one device.

**DataLoader shared-memory errors.** Data loading uses 12 worker processes by
default. If workers fail with shared-memory or open-file errors, pass
`--num-workers 0` or a smaller worker count. Inside a container, a larger
shared-memory size also helps.

**Memory pressure.** Testing downloads each selected checkpoint into a
temporary directory and deletes the file after loading it. If the system
temporary directory is memory-backed, point `TMPDIR` to a directory on disk.
A smaller `--batch-size` lowers accelerator memory use. The benchmark batch
sizes come from `configs/dataset_windows.yaml`.

**MLflow store location.** `--logdir` selects the tracking store. A local path,
such as the default `runs`, becomes a file store, whereas an `http://` or
`https://` URI targets a tracking server. Runs are grouped into experiments
named `<experiment-prefix>-<dataset>`, where the prefix comes from
`--mlflow-experiment-prefix`. Training, testing, and analysis must use the same
`--logdir` and prefix, or testing reports that it found no experiment with that
prefix. Browse a local store with:

```bash
uv run mlflow ui --backend-store-uri runs
```

**Chronos-2 weights.** The first Chronos-2 run downloads `amazon/chronos-2` at
the revision pinned in `configs/baseline_hparams.yaml` from the Hugging Face
Hub into the Hugging Face cache, which `HF_HOME` relocates. For machines
without network access, fill the cache once on a connected machine, copy it,
and set `HF_HUB_OFFLINE=1`:

```bash
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('amazon/chronos-2', revision='0f8a440441931157957e2be1a9bce66627d99c76')"
```

A failed download raises `Failed to load Chronos2Pipeline`.

**Reruns.** Training skips a configuration when its experiment already holds
a finished run with the same run signature, so an interrupted campaign resumes
where it stopped. `--rerun` deletes the existing runs of each requested
signature and trains them again. Testing likewise skips selected models that are
already fully tested in the current evaluation context, which `--rerun`
evaluates again. Runs that are still active with the same signature raise an
error.

**Coverage errors.** With `--full-coverage`, testing raises
`Testing expected-vs-seen coverage mismatch detected` and lists the missing
architecture, method, or variant runs. Train the missing runs, or narrow the
scope with `--data-files`, `--model`, and `--method`. Without
`--full-coverage`, testing and analysis drop incomplete method families and
print what they dropped. Analysis also raises under `--full-coverage` when the
configured scenarios lie outside `configs/reporting/core_figures.yaml`, as
described in [`sensitivity_analyses.md`](sensitivity_analyses.md).

**Dataset validation failures.** `scripts/benchmark_dataset_contracts.py` and
the preprocessing scripts compare a staged file with the registry metadata,
including the filename, row count, channel count, column order, and timestamp
interval. A mismatch usually means a different upstream version or a wrong
source file. Download the source again as described in
[`data/README.md`](../data/README.md) and repeat the validation.
