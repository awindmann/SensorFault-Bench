# Paper Artifact

This directory is the entry point for the NeurIPS 2026 paper "Benchmarking
Sensor-Fault Robustness in Forecasting". The paper corresponds to release
`v1.0.0` of this repository. Later releases can extend the benchmark, but a
checkout of `v1.0.0` always restores the code, results, and evidence of the
paper.

## Paper to Repository Map

| Paper section | Bundled results | Commands |
| --- | --- | --- |
| "Results" and "Extended results" | `tables/default_configuration/`, `paper_inputs/` | [Reproducing the paper](../README.md#reproducing-the-paper) |
| "Sensitivity to the evaluation data seed", "Selector pressure", and "Sensitivity to the number of affected channels" | `tables/eval_data_seed_*/`, `tables/selector_perturbed_validation/`, `tables/fixed_channel_fraction_*/` | [Evaluation data seed, selector, and channel count](../docs/sensitivity_analyses.md#evaluation-data-seed-selector-and-channel-count) |
| "Sensitivity to severity weighting" | `supplementary_tables/severity_weighting_*.csv` | [Severity weighting](../docs/sensitivity_analyses.md#severity-weighting) |
| "Sensitivity to fault direction" | `supplementary_tables/fault_direction_architecture.csv` | [Fault direction](../docs/sensitivity_analyses.md#fault-direction) |
| "Dependence on TimeCompress" | `supplementary_tables/timing_*.csv` | [Timing](../docs/sensitivity_analyses.md#timing) |
| "ETTh1 target scope and horizon ablation" | `supplementary_tables/etth1_task_*.csv` | [ETTh1 target scope and horizon](../docs/sensitivity_analyses.md#etth1-target-scope-and-horizon) |
| "Training seed sensitivity" | `supplementary_tables/training_seed_*.csv`, `supplementary_tables/patchtst_clean_validation_near_tie*.csv` | [Training and search seeds](../docs/sensitivity_analyses.md#training-and-search-seeds) |
| "Full experimental details" | `supplementary_tables/selected_clean_validation_hyperparameters.csv`, `compute_accounting/` | [Compute accounting](#compute-accounting) |
| Figures | `R/figures/` | [Figure scripts](R/README.md) |

## Compute Accounting

[`compute_accounting/`](compute_accounting/) records the runs, evaluation
episodes, and GPU-hours of the paper's experiments: the default configuration,
the reruns with training and search seeds 1001 and 1002, the three additional
ETTh1 task settings, and the evaluations for the sensitivity checks. An overhead
audit covers the remaining recorded project compute.
[`scripts/compute_accounting_ledger.py`](../scripts/compute_accounting_ledger.py)
writes the folder's four CSV files in one read-only pass over the MLflow
experiments and analysis windows of
[`configs/reporting/compute_accounting.yaml`](../configs/reporting/compute_accounting.yaml).
Before it writes, the default configuration must reproduce the compute
accounting of version 0.1.0, which the script reads from that release. Running
the script needs access to the tracking server that holds the recorded runs.

## Bundle Contents

- `MANIFEST.json`: the machine-readable index of the bundle, with checksums,
  the dataset scope, source identifiers, generation commands, and the content
  that the bundle leaves out.
- `tables/`: the CSV analysis tables, one folder per evaluated setting.
  `default_configuration/` holds the results of the default configuration. The
  other folders hold the sensitivity checks of evaluation data seeds, selector
  pressure, and fixed fractions of affected channels.
- `paper_inputs/`: the CSV inputs of the paper's tables and figures, derived
  from the tables of the default configuration.
- `supplementary_tables/`: the hyperparameters selected on clean validation
  data, a diagnostic of near-ties in clean validation loss for PatchTST, and the
  architecture and method results for training and search seeds 42, 1001,
  and 1002. Further tables cover the checks of severity weighting, fault
  direction, timing, and the ETTh1 task. Its `README.md` describes each file.
- `compute_accounting/`: the compute accounting described above.
- `config/eval_context.json`: the evaluation context of the bundled analysis
  sources.
- `forecast_examples/manifest.json`: the sample identities of the curated
  forecast examples, with a record of which traces the bundle contains.
- `R/`: the scripts that reproduce the paper figures, together with their input
  CSVs, the curated forecast traces, and the rendered PDFs.

The artifact excludes checkpoints, MLflow run stores, raw and processed
datasets, complete outputs per sample, raw analysis configurations, forecast
traces beyond the curated R extracts, and complete export trees. The two
curated datasets are published as separate dataset records with their own
metadata (see `data/README.md` in the repository root).

## Paper Inputs

The files in `paper_inputs/` are the small CSV inputs from which the
paper's tables and figures are built. The artifact builder derives them from the
tables of the default configuration and lists them in `MANIFEST.json` with row
counts, checksums, and source tables.

## Provenance

All paths needed to inspect the artifact are relative to this directory. Each
folder in `tables/` carries the name of its evaluated setting, which
`MANIFEST.json` uses as the source identifier and which stays fixed across
builds. For each source, the manifest also records the export date and the path
from which the build read the tables. These records serve to audit the build,
while inspecting the results needs only the directory tree described above.

## Verification

`MANIFEST.json` and `supplementary_tables/manifest.json` record the SHA-256
checksum and byte size of every bundled file.
