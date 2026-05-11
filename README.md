# SensorFault-Bench

SensorFault-Bench contains the benchmark code for training forecasting
models and evaluating them under sensor-fault perturbations.

## Dataset Collection

SensorFault-Bench evaluates four benchmark datasets under one sensor-fault
robustness protocol:

- `BeijingAir_Tiantan`: derived Beijing Air Tiantan PM2.5 time series record,
  published at
  <https://www.kaggle.com/datasets/sensorfaultbench/beijing-air-tiantan-pm2-5-time-series-dataset>.
- `Penmanshiel_Hourly_WT08`: derived Penmanshiel WT08 hourly SCADA time series
  record, published at
  <https://www.kaggle.com/datasets/sensorfaultbench/penmanshiel-wt08-hourly-scada-time-series-dataset>.
- `ETTh1`: existing ETT benchmark dataset, acquired from the public ETT source
  at <https://github.com/zhouhaoyi/ETDataset/tree/main/ETT-small>.
- `traffic`: existing Traffic benchmark dataset, acquired from the public
  upstream CSV at
  <https://huggingface.co/datasets/thuml/Time-Series-Library/blob/main/traffic/traffic.csv>
  and converted to the benchmark Parquet runtime format.

The two derived Kaggle records are licensed under CC BY-SA 4.0 and have
separate Croissant metadata records. Detailed source terms, expected filenames,
access paths, and validation commands are specified in
[`data/README.md`](data/README.md).

## Benchmark Workflow

The benchmark entry points are:

- `run_training.py`: train the full benchmark scope or an explicit
  dataset, architecture, and method subset.
- `run_testing.py`: select current winners and evaluate them under the
  canonical perturbation protocol.
- `run_analysis.py`: build cross-dataset analysis tables and figures from
  tested winners.

Install dependencies and run a small local smoke:

Before running the commands, place the required dataset files under `DATA_ROOT`
as described in `data/README.md`.

```bash
uv sync
uv run python run_training.py \
  --data-files ETTh1 \
  --model DLinear \
  --method baseline \
  --max-epochs 1 \
  --max-hp-trials-per-model 1 \
  --logdir runs
uv run python run_testing.py \
  --data-files ETTh1 \
  --model DLinear \
  --method baseline \
  --perturbation-scenarios missing_data noise \
  --max-hp-trials-per-model 1 \
  --logdir runs
uv run python run_analysis.py \
  --data-files ETTh1 \
  --model DLinear \
  --method baseline \
  --perturbation-scenarios missing_data noise \
  --max-hp-trials-per-model 1 \
  --logdir runs
```

Use `--help` on each entry point for sample limits, hardware options, storage
configuration, rerun behavior, and other advanced controls.

Run the full benchmark scope from configured defaults:

```bash
uv run python run_training.py
uv run python run_testing.py --full-coverage
uv run python run_analysis.py --full-coverage
```

Coverage is partial by default for testing and analysis. Pass
`--full-coverage` when you want strict benchmark coverage checks so missing
benchmark coverage raises.

The full training command expands across all benchmark datasets, architectures, and
methods in `configs/defaults.yaml`. It is a compute-heavy benchmark bundle, not a
quick smoke check. Dataset-specific benchmark batch sizes are owned by
`configs/dataset_windows.yaml`, explicit `--batch-size` CLI values override
those YAML defaults for scoped local runs.

`run_analysis.py` logs tables under the MLflow artifact path `tables/`, figures
under `figures/`, the figure index at `tables/figures_manifest.csv`, and the
analysis invocation under `config/meta_analysis_args.yaml`. The default local
MLflow tracking store is `runs`.

## Testing

The source repository includes a pytest suite under `tests/`. It validates the
benchmark code, release surface, paper-artifact manifest, and public-safe
documentation. Run it with:

```bash
uv sync
uv run pytest -q
```

### Paper Artifact

The submitted repository includes `paper_artifact/MANIFEST.json` as the frozen
result-export identity for the paper. The core bundle contains small
paper-facing tables, a public-safe evaluation-context snapshot, curated
forecast-example sample identities, and file checksums. The optional
`paper_artifact/R/` subartifact contains the R and TikZ scripts, curated figure
inputs, and polished PDFs for the paper figures. R and TeX are not part of the
Python benchmark environment.

The artifact intentionally excludes checkpoints, MLflow run stores, raw and
processed datasets, broad sample-level dumps, raw private analysis configs,
generic generated figure binaries, full forecast trace CSV dumps outside the
curated R figure extracts, and replay PDFs.

### Forecast Plot Replay

Rerunning training, testing, and analysis is the benchmark reproduction path.
Forecast plots can also be regenerated for curated sample IDs, or for selected
samples from `tables/scenario_samples.csv` or `tables/forecast_extremes.csv`.
Use `scripts/render_forecast_plots.py from-runs` when trained benchmark models,
MLflow runs, checkpoints, and the configured datasets are available. This mode
rebuilds the requested sample forecasts and writes portable trace CSVs by
default. Use `scripts/render_forecast_plots.py from-traces` only to render
existing forecast trace CSVs, including the curated R figure extracts, without
MLflow, MinIO, checkpoints, or processed dataset access.
Detailed commands are in
[`docs/reproducing_benchmark.md`](docs/reproducing_benchmark.md).

## Perturbation Scenarios

Sensor-fault evaluation samples a severity in `[0, 1]` and applies the selected
operator to a severity-coupled channel subset. Scenario membership and order are
configured by `PERTURBATION_SCENARIOS` in `configs/defaults.yaml`. Operator
semantics live in `data/perturbations.py`.

Default benchmark scenarios:

- Value faults: `drift`, `attenuation`, `noise`, `spike`
- Timing faults: `time_stretch`, `time_compress`
- Availability faults: `stuck_sensor`, `missing_data`

Fault augmentation uses the configured training-only transfer fault families
`\mathcal{P}_{\mathrm{trans}}`. In code, these are represented
as train-side holdout profiles in
[`configs/pipelines/fault_augmentation.yaml`](configs/pipelines/fault_augmentation.yaml).
They are disjoint from the scored benchmark scenarios and the configured profile
grid evaluates:

- `holdout_simple`: `linear_drift`, `scaling`, `trimming_constant`, `packet_loss`
- `holdout_varying`: `nonlinear_drift`, `time_varying_scaling`,
  `trimming_varying`, `packet_loss`

Discrete-state operators `wrong_state` and `chattering` are implemented for
datasets with discrete channels, but are not part of the default benchmark
scenario set.

## Benchmark Scope

Datasets:

- `BeijingAir_Tiantan`
- `Penmanshiel_Hourly_WT08`
- `ETTh1`
- `traffic`

Built-in forecasting architectures:

- `DLinear`
- `GRU`
- `ModernTCN`
- `PatchTST`
- `TSMixer`
- `SeasonalNaive`
- `Chronos2`

Benchmark methods:

- `baseline`
- `randomized_training`
- `adversarial_training`
- `adaptive_robust_loss`
- `ensemble`
- `randomized_smoothing`
- `fault_augmentation`
- `revin`

Runtime membership and execution order are owned by `configs/defaults.yaml`.
The list above is the full training method execution order.
Display labels, display order, architecture roles, method applicability,
and reporting labels are owned by `configs/benchmark_scope.yaml`.

## Data

Raw and processed datasets are not redistributed by default. Dataset keys
resolve to filenames under `DATA_ROOT`, which defaults to `data/processed`.
Acquisition instructions, source links, derived-dataset validation, and dataset
terms are documented in `data/README.md`. Provenance notebooks for the two
derived benchmark datasets live under `notebooks/provenance/`.

## Runtime Profiles

Local profile:

```bash
uv run python run_training.py --data-root data/processed --logdir runs
```

VM S3/MinIO profile:

```bash
uv run python run_training.py \
  --data-root s3://<bucket>/<prefix> \
  --logdir <mlflow-tracking-uri> \
  --minio-endpoint <object-store-endpoint>
```

The local profile writes MLflow runs and analysis artifacts under `runs`.
The VM profile writes runs to the configured MLflow tracking URI and reads
datasets from the configured S3-compatible `DATA_ROOT`. Credentials must come
from the shell, environment, or VM secret manager.

## Documentation

- `docs/reproducing_benchmark.md` documents the full benchmark workflow and
  artifact locations, including object-store configuration.
- `docs/extending_datasets.md` documents dataset registry and window defaults.
- `docs/extending_models.md` documents model integration.
- `docs/extending_methods.md` documents robustness-method integration.
- `THIRD_PARTY_NOTICES.md` records source attribution and dataset
  source notices.

## License

Repository code and documentation are licensed under Apache-2.0. See
`LICENSE`. Dataset licenses and source terms are separate from the repository
code license. See `data/README.md`. Third-party source attributions and
carry-over notices are recorded in `THIRD_PARTY_NOTICES.md`.
