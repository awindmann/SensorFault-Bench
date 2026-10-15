# Changelog

All notable changes to SensorFault-Bench are recorded here. Version numbers
follow [semantic versioning](https://semver.org/).

## 1.0.0

Camera-ready release for the NeurIPS 2026 Evaluations & Datasets Track paper
"Benchmarking Sensor-Fault Robustness in Forecasting". The results of the
default configuration are unchanged, as are its scored scenarios, selection
rules, and scoring.

### Added

- `paper_artifact/supplementary_tables/`, the tables that the paper added in
  its camera-ready revision. Its `README.md` describes each table.
- `data/croissant/`, the Croissant metadata of the two curated datasets with
  their Responsible AI fields.
- The sensitivity scenarios `negative_drift`, `negative_spike`, and
  `local_timing_warp`.
- The severity sampling distribution `beta_1_2`, which
  `PERTURBATION_SEVERITY_DISTRIBUTION` selects. The default stays `uniform`.
- `docs/benchmark_protocol.md`, `docs/sensitivity_analyses.md`, a results
  schema and troubleshooting in `docs/reproducing_benchmark.md`, and paired
  transfer inputs in `docs/extending_methods.md`.
- `CITATION.cff`, this changelog, and continuous integration.
- `docs/using_fault_operators.md`, a guide to uses of the fault operators
  outside the benchmark, such as pretraining or stress tests calibrated to
  observed faults. The protocol section "Fault Exposure in Training" asks
  reports to name the fault operators that a model saw in training, including
  pretraining.
- `notebooks/tutorial/quickstart.ipynb`, a CPU tutorial that shows the scored
  scenarios, runs a small benchmark, and reads the reported quantities.
- `configs/cli_help.yaml`, the help text of every command-line option. The
  `--help` output of the three entry points groups the options by topic and
  shows their defaults.

### Changed

- The README is restructured around the key findings, the results of the
  default configuration, and a guide to evaluating a new model or method.
  `data/README.md` adds download commands for the two curated datasets.
- The paper artifact is version 2,
  `sensorfault-bench-neurips2026-paper-artifact-v2`.
- The folders in `paper_artifact/tables/` carry the name of their evaluated
  setting, for example `default_configuration/` instead of
  `primary_clean_2026-04-26_10k/`. `paper_artifact/MANIFEST.json` records the
  export date and source path of each folder. The tables are unchanged.
- The figure scripts and PDFs in `paper_artifact/R/` match the camera-ready
  figures. Their inputs are unchanged.
- `paper_artifact/README.md` maps each section of the paper to its bundled
  results and the commands that reproduce them, and states that the paper
  corresponds to release `v1.0.0`.
- The compute accounting moved from `evidence/compute_accounting/` to
  `paper_artifact/compute_accounting/`. It also covers the seed reruns, ETTh1
  task settings, and sensitivity checks of the camera-ready revision. Its values
  for the default configuration are unchanged.
  `scripts/compute_accounting_ledger.py` writes it from the campaigns listed in
  `configs/reporting/compute_accounting.yaml`.
- The extension guides describe the interfaces, work through one example each,
  and list every configuration entry that a new architecture, method, or
  dataset needs. `docs/extending_models.md` shows how to evaluate a model whose
  weights were trained elsewhere.
- The architecture families of the analysis plots are configured in
  `configs/benchmark_scope.yaml`, and the trajectory legends use the method
  labels of `configs/reporting/core_figures.yaml`. Analysis therefore accepts
  new architectures and methods without code changes.
- `instantiate_perturbations` in `data/data_module.py` is now public.
- `TSDataModule` no longer accepts `severity_laws`.
  `PERTURBATION_SEVERITY_DISTRIBUTION` sets the severity sampling distribution.
- `DatasetSpec` requires `input_channels`.

### Fixed

- Forecast replay binds the same randomized smoothing noise to each window as
  degradation scoring does. `scripts/render_forecast_plots.py` and the
  rendering of extreme forecasts in `run_analysis.py` therefore reproduce the
  scored forecasts of randomized smoothing runs instead of failing the score
  check.
- With `STRICT_IID: true`, every sampled window lies inside its own block
  together with its target horizon. No two windows therefore share a time step.
  The results of the default configuration are unchanged because the option is
  off by default and degradation evaluation rejects it.
- `scripts/build_paper_artifact.py --force` builds the new bundle in a staging
  directory and replaces the existing bundle only after the build succeeds. A
  failed build leaves the previous bundle unchanged.
- The artifact builder and the test suite run on Python 3.10, the minimum
  version declared in `pyproject.toml`. Continuous integration tests Python
  3.10, 3.11, and 3.12.
- Training, testing, analysis, and forecast replay stop when the input or
  target channel order recorded on a run differs from the dataset
  specification. The data configuration signature compares channel sets, so a
  run trained under another channel order was previously scored with the wrong
  channel mapping. Training also counted such a run as a completed candidate.
- Analysis and repeated testing accept runs that were reevaluated with fewer
  scenarios, as `docs/sensitivity_analyses.md` describes. They previously
  stopped on the metrics that the earlier evaluation had logged for the dropped
  scenarios.
- `run_training.py` trains `baseline` first and wrapping methods last, whatever
  the order of `--method`, so a method listed before `baseline` no longer fails
  on a new experiment. `run_analysis.py` stops before reading any runs and
  names the missing `baseline` when `--method` omits it.
- Training stops with a clear error when the validation split holds fewer
  windows than one batch. Validation scores full batches only, so such a split
  previously ended in an error about the missing early-stopping metric.
- Testing keeps the end time of each evaluated run. Because the end time breaks
  ties between equal validation scores, the new end time that evaluation used
  to record could make a repeated test select another winner after
  `--refresh-selection-metrics`.
- Scoring accepts bootstrap percentile intervals that exclude the point
  estimate, which a small `BOOTSTRAP_CI_RESAMPLES` can produce. Such intervals
  previously stopped scoring.
- Fault augmentation runs with data loader workers under the `spawn` start
  method, the default on macOS and Windows. Its severity sampler previously
  could not be passed to the workers.
- Randomized training stops when its noise level is not finite, for example
  `nan`. Such a run previously trained without noise.

## 0.1.0

State of the repository during the review period.
