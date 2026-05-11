# Reproducing The Benchmark

This guide documents the repository workflow for the benchmark scope. The
workflow uses the tracked source tree, the dataset keys in
`configs/defaults.yaml`, and the benchmark-scope display metadata in
`configs/benchmark_scope.yaml`.

## Setup

```bash
uv sync
```

Place the four benchmark datasets under `DATA_ROOT`, which defaults to
`data/processed`. See `data/README.md` for acquisition, preprocessing, and
derived-dataset validation.

## Source Validation

The source repository includes a pytest suite under `tests/`. It validates the
benchmark code, release surface, paper-artifact manifest, and public-safe
documentation:

```bash
uv run pytest -q
```

## Full Benchmark Commands

Train all benchmark datasets, architectures, and applicable methods:

```bash
uv run python run_training.py
```

The commands above use the configured benchmark defaults. Dataset-specific
benchmark batch sizes are owned by `configs/dataset_windows.yaml`: batch size
64 for `BeijingAir_Tiantan` and `Penmanshiel_Hourly_WT08`, and batch size 16 for
`ETTh1` and `traffic`.

Evaluate current winners under the canonical perturbation protocol:

```bash
uv run python run_testing.py --full-coverage
```

Build cross-dataset analysis artifacts from tested winners:

```bash
uv run python run_analysis.py --full-coverage
```

Coverage is partial by default for testing and analysis. Pass
`--full-coverage` when you want strict benchmark coverage checks so missing
benchmark coverage raises.

For partial local smoke checks, narrow the scope explicitly:

```bash
uv run python run_training.py \
  --data-files ETTh1 \
  --model DLinear \
  --method baseline \
  --max-epochs 1 \
  --max-hp-trials-per-model 1 \
  --n-train-samples 8 \
  --n-val-samples 4 \
  --n-test-samples 4 \
  --input-len 12 \
  --target-len 3 \
  --batch-size 4 \
  --num-workers 0 \
  --accelerator cpu \
  --devices 1 \
  --mlflow-experiment-prefix benchmark-smoke \
  --logdir runs/smoke \
  --rerun
uv run python run_testing.py \
  --data-files ETTh1 \
  --model DLinear \
  --method baseline \
  --max-hp-trials-per-model 1 \
  --n-train-samples 8 \
  --n-val-samples 4 \
  --n-test-samples 4 \
  --input-len 12 \
  --target-len 3 \
  --batch-size 4 \
  --num-workers 0 \
  --accelerator cpu \
  --devices 1 \
  --perturbation-scenarios missing_data noise \
  --bootstrap-ci-resamples 10 \
  --mlflow-experiment-prefix benchmark-smoke \
  --logdir runs/smoke \
  --rerun
uv run python run_analysis.py \
  --data-files ETTh1 \
  --model DLinear \
  --method baseline \
  --max-hp-trials-per-model 1 \
  --n-train-samples 8 \
  --n-val-samples 4 \
  --n-test-samples 4 \
  --input-len 12 \
  --target-len 3 \
  --batch-size 4 \
  --perturbation-scenarios missing_data noise \
  --bootstrap-ci-resamples 10 \
  --mlflow-experiment-prefix benchmark-smoke \
  --logdir runs/smoke
```

## Method Execution Order

`python run_training.py` expands methods in the `BENCHMARK_METHODS` order from
`configs/defaults.yaml`:

1. `baseline`
2. `randomized_training`
3. `adversarial_training`
4. `adaptive_robust_loss`
5. `ensemble`
6. `randomized_smoothing`
7. `fault_augmentation`
8. `revin`

The full-scope runner applies the benchmark method-architecture applicability
matrix from `configs/benchmark_scope.yaml`. Baseline covers all benchmark
architectures. Non-baseline methods cover the method-comparison architectures,
with `revin` excluding `PatchTST`.

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

The local profile uses the empty `MINIO_ENDPOINT` default from
`configs/defaults.yaml`. Credentials must come from the shell, environment, or
VM secret manager. For MinIO, pass `--minio-endpoint` explicitly. Standard
S3-compatible deployments that rely on provider defaults can leave it empty.

## Artifacts

Training and testing log MLflow runs under the configured `LOGDIR`. The default
local store is `runs`.

`run_testing.py` writes degradation artifacts on tested winner runs, including
clean samples, scenario samples, and scenario summaries.

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
- `tables/forecast_extremes.csv` when forecast-extreme rendering is enabled
  and renderable extreme samples are available
- `figures/...`
- `config/meta_analysis_args.yaml`

Exact figure filenames are recorded in `tables/figures_manifest.csv`.

## Paper Artifact

The submitted paper evidence bundle is committed under `paper_artifact/`. Its
`MANIFEST.json` names the frozen source exports, records the source repository
commit observed at build time, lists the copied table files, and stores SHA-256
checksums and byte sizes for each bundled file. The paired
`config/eval_context.json` is generated from allowlisted analysis fields, not
from the raw `config/meta_analysis_args.yaml` files.

The core paper artifact is a provenance and table bundle. It includes curated
forecast-example sample identities under `forecast_examples/manifest.json`, but
it does not redistribute checkpoints, raw datasets, processed runtime datasets,
MLflow run stores, broad sample-level dumps, raw private configs, generic
generated figure binaries, full forecast trace CSV dumps outside the curated R
figure extracts, or replay PDFs.

The optional `paper_artifact/R/` subartifact is narrower: it contains the R and
TikZ scripts, curated figure-level inputs, and polished PDFs for the submitted
paper figures. These R and TeX dependencies are documented in
`paper_artifact/R/README.md` and are not installed by `uv sync`.

## Forecast Plot Replay

Use `scripts/render_forecast_plots.py from-runs` to regenerate selected
forecast plots from tested MLflow runs and trained benchmark models. This mode
queries the selected run, loads the matching checkpoint and dataset, rebuilds
the requested clean and perturbed samples, and writes fresh per-sample forecast
trace CSVs by default. It does not require trace CSVs as input.

For already curated baseline-winner samples, pass the compact sample identity
directly. The format is
`ARCH:SCENARIO:PERT_IDX:SAMPLE_ID:SOURCE_IDX:SEVERITY[:SAMPLE_SCORE]`:

```bash
uv run python scripts/render_forecast_plots.py from-runs \
  --tracking-uri <mlflow-tracking-uri> \
  --minio-endpoint <object-store-endpoint> \
  --experiment-prefix improv-11 \
  --data-root <processed-data-root> \
  --dataset BeijingAir_Tiantan \
  --batch-size 64 \
  --n-test-samples 10000 \
  --eval-data-seed 1806770612 \
  --output-dir forecast_plots_final_traces/beijing_air \
  --sample PatchTST:drift:0:2329:572:0.596497 \
  --sample DLinear:spike:3:8609:3644:0.492074 \
  --export-traces
```

`--export-traces` is enabled by default. Pass `--no-export-traces` only when
the portable replay artifacts are not needed. The public repository defaults
target local smoke runs, so plots from the frozen benchmark run store need the
matching `--tracking-uri`, `--minio-endpoint`, `--experiment-prefix`, and
`--data-root` values. Use `--samples-csv` instead of compact `--sample` when a
selected row must carry `run_id`, `robustness_method`, `pipeline_method`, or
per-row runtime overrides.

To select samples from analysis artifacts, pass either
`tables/scenario_samples.csv` or `tables/forecast_extremes.csv`:

```bash
uv run python scripts/render_forecast_plots.py from-runs \
  --dataset ETTh1 \
  --model DLinear \
  --method baseline \
  --scenario-samples-csv <analysis-artifacts>/tables/scenario_samples.csv \
  --output-dir forecast_plots
```

Use `--forecast-extremes-csv` with `--extreme-kind`, `--extreme-rank`,
`--sample-id`, `--pert-idx`, or `--limit` when only a filtered subset of the
extreme-sample table should be rendered.

Use `scripts/render_forecast_plots.py from-traces` only to replay trace CSVs
that were already exported by `from-runs` or by an earlier plotting run. This
path renders without MLflow, checkpoints, or dataset files:

```bash
uv run python scripts/render_forecast_plots.py from-traces \
  --trace-root paper_artifact/R/data/forecast_plots_final_traces \
  --output-dir forecast_plot_replay \
  --format html \
  --preserve-layout \
  --missing-robustness-method baseline
```

Trace CSVs produced by current tooling contain the `robustness_method` column.
Older trace CSVs, including the bundled curated R figure extracts, must pass the
method label explicitly with `--missing-robustness-method`. Otherwise rendering
raises instead of guessing the method.
