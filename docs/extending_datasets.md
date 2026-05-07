# Extending Datasets

## Owner Paths

1. Add the dataset metadata to `data/datasets/specs.py`.
2. Add dataset window and benchmark batch-size defaults to
   `configs/dataset_windows.yaml`.
3. Add the dataset to `configs/defaults.yaml` only if it should enter the benchmark
   default benchmark scope.
4. Add display metadata to `configs/benchmark_scope.yaml` only when the dataset is
   part of the benchmark reporting scope.
5. Update `data/README.md` with acquisition instructions, expected filename,
   source terms, and validation commands.

## Required Metadata

Each registered dataset must define:

- a stable dataset key,
- a relative filename under `DATA_ROOT`,
- split mode,
- target alias,
- input and target channel metadata,
- continuous and discrete channel metadata.

Do not infer channel typing from observed values at runtime. Missing or
ambiguous metadata should raise during registry or datamodule validation.
Any split-relevant dataset change must be covered by the
`data_config_signature` used for run comparability.
