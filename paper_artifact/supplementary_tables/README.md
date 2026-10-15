# Supplementary Tables

This directory holds the hyperparameters selected on clean validation data, the
results for training and search seeds 42, 1001, and 1002, and the tables of the
paper's sensitivity checks.

- `selected_clean_validation_hyperparameters.csv` gives, for each of the four
  datasets and seven forecasting architectures, the baseline configuration that
  clean validation selected at seed 42. Its `hyperparameters_json` column holds
  the exact model hyperparameters and the MSE training loss. The Chronos-2 rows
  report the model revision pinned in `configs/baseline_hparams.yaml`. The runs
  behind these rows loaded the default branch of `amazon/chronos-2`,
  whose weights and configuration are byte-identical to that revision.
- `training_seed_architecture_results.csv` gives all architecture results for
  training and search seeds 42, 1001, and 1002, with data splits and evaluation
  sampling fixed within each dataset. SeasonalNaive and Chronos-2 are fixed
  references that do not depend on the training seed.
- `training_seed_method_results.csv` gives every selected method-baseline pair
  for the same three seeds. It covers all seven robustness-improvement methods
  on the architectures they apply to. RevIN has four pairs per dataset because
  the PatchTST baseline already uses reversible instance normalization.
- `patchtst_clean_validation_near_ties.csv` gives the 14 PatchTST
  configurations on Beijing Air Tiantan at seed 1001 whose clean validation
  loss lies within an exploratory band of 3% above the best one.
  `patchtst_clean_validation_near_tie_summary.csv` summarizes the two branches
  that the paper describes. The diagnostic shows an association and does not
  isolate width, learning rate, or model selection as a cause.

The sensitivity tables report the paper's checks of the benchmark design. Each
table sets the evaluation of the default configuration beside the varied one.

- Severity weighting, which samples severity from Beta(1,2) instead of the
  uniform distribution: `severity_weighting_architecture.csv`,
  `severity_weighting_method_pairs.csv`,
  `severity_weighting_method_means.csv`, `severity_weighting_ranks.csv`, and
  `severity_weighting_exceptions.csv`.
- Fault direction, which replaces Drift and Spike with their negative
  counterparts: `fault_direction_architecture.csv`.
- Timing, which replaces TimeCompress with a local timing warp or omits it
  from the scored set: `timing_local_warp_architecture.csv`,
  `timing_time_compress_clipping.csv`,
  `timing_time_compress_omission_architecture.csv`, and
  `timing_time_compress_omission_method_pairs.csv`.
- ETTh1 target scope and horizon: `etth1_task_architecture.csv` and
  `etth1_task_contrasts.csv`.

`docs/sensitivity_analyses.md` gives the commands that reproduce each check.

`manifest.json` records the row counts, checksums, and exact scope of the
tables. It identifies each build input by its repository path when the
repository ships the input, and otherwise by a source identifier and its
SHA-256 checksum. The manifest omits run identifiers, service details, local
paths, and campaign labels.
