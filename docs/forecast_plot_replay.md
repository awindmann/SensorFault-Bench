# Forecast Plot Replay

`scripts/render_forecast_plots.py` regenerates the forecast plots of single
test windows, such as the examples in the README and the paper. Its `from-runs`
mode rebuilds the windows from tested runs, whereas its `from-traces` mode
replays exported trace files without access to the run store.

## From Tested Runs

`scripts/render_forecast_plots.py from-runs` regenerates forecast plots from
tested MLflow runs and the trained benchmark models. It queries the selected
run, loads the matching checkpoint and dataset, rebuilds the requested clean and
perturbed samples, and by default writes one forecast trace CSV per sample. It
needs no trace CSVs as input.

A known sample of a selected baseline forecasting model can be passed directly
as a compact identity of the form
`ARCH:SCENARIO:PERT_IDX:SAMPLE_ID:SOURCE_IDX:SEVERITY[:SAMPLE_SCORE]`:

```bash
uv run python scripts/render_forecast_plots.py from-runs \
  --tracking-uri <mlflow-tracking-uri> \
  --minio-endpoint <object-store-endpoint> \
  --experiment-prefix <experiment-prefix> \
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
the trace files for a later replay are not needed. Because the repository
defaults point to local smoke runs, plots from the run store of the default
configuration need matching values for `--tracking-uri`, `--minio-endpoint`,
`--experiment-prefix`, and `--data-root`. When a selected sample must carry its
own `run_id`, `robustness_method`, `pipeline_method`, or runtime overrides, list
it in `--samples-csv` instead of passing a compact `--sample`.

To select samples from the analysis tables, pass either
`tables/scenario_samples.csv` or `tables/forecast_extremes.csv`:

```bash
uv run python scripts/render_forecast_plots.py from-runs \
  --dataset ETTh1 \
  --model DLinear \
  --method baseline \
  --scenario-samples-csv <analysis-artifacts>/tables/scenario_samples.csv \
  --output-dir forecast_plots
```

To render only part of the table of extreme samples, combine
`--forecast-extremes-csv` with `--extreme-kind`, `--extreme-rank`,
`--sample-id`, `--pert-idx`, or `--limit`.

## From Exported Traces

`scripts/render_forecast_plots.py from-traces` replays trace CSVs that
`from-runs` or an earlier plotting run exported. It renders without MLflow,
checkpoints, or dataset files:

```bash
uv run python scripts/render_forecast_plots.py from-traces \
  --trace-root paper_artifact/R/data/forecast_plots_final_traces \
  --output-dir forecast_plot_replay \
  --format html \
  --preserve-layout \
  --missing-robustness-method baseline
```

Trace CSVs that this script exports contain a `robustness_method` column. For
trace CSVs without it, such as the curated traces of the R figures,
`--missing-robustness-method` must name the method, because rendering raises
an error rather than guessing it.
