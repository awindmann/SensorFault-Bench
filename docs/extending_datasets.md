# Extending Datasets

The benchmark protocol applies to any multivariate time series whose channels
are typed as continuous or discrete. A new dataset needs a data file under
`DATA_ROOT`, an entry in the dataset registry, and window defaults. Results on
a new dataset describe that dataset and are not comparable with the default
configuration on its four datasets.

## Data File

The loader reads CSV and Parquet files with one column per channel and one row
per time step, in time order and at a fixed interval. A CSV file starts with a
timestamp column in the format `YYYY-MM-DD HH:MM:SS`, which the loader uses as
the index:

<!-- dataset-file-example:start -->
```text
date,inlet_temperature,outlet_temperature,flow_rate,pump_state
2024-01-01 00:00:00,20.06,25.31,3.56,0
2024-01-01 01:00:00,21.23,25.94,3.36,1
```
<!-- dataset-file-example:end -->

Continuous channels hold numbers, while discrete channels hold integer codes,
such as the state of a pump. The loader does not fill missing values, so gaps
must be filled or removed before registration. A missing value inside a window
propagates into the loss and stops training.

## Registration

1. Save the file under `DATA_ROOT`, by default `data/processed/`.
2. Add an entry to `DATASET_REGISTRY` in
   [`data/datasets/specs.py`](../data/datasets/specs.py):

   <!-- dataset-spec-example:start -->
   ```python
   DatasetSpec(
       key="MyPlant",
       path="my_plant.csv",
       split_mode="temporal",
       input_channels=("inlet_temperature", "outlet_temperature", "flow_rate", "pump_state"),
       channel_groups={"outlet": ("outlet_temperature",)},
       default_target="outlet",
       description="Hourly process data of one plant, forecasting the outlet temperature.",
       continuous_channels=("inlet_temperature", "outlet_temperature", "flow_rate"),
       discrete_channels=("pump_state",),
   ),
   ```
   <!-- dataset-spec-example:end -->

3. Add the window lengths and the batch size to
   [`configs/dataset_windows.yaml`](../configs/dataset_windows.yaml):

   ```yaml
   MyPlant:
     input_len: 48
     target_len: 12
     batch_size: 32
   ```

4. Add a display label to `CORE_FIGURE_DATASET_SPEC` in
   [`configs/reporting/core_figures.yaml`](../configs/reporting/core_figures.yaml),
   for example `"MyPlant=My plant"`. Analysis rejects datasets without a label.

| Field | Meaning |
| --- | --- |
| `key` | Dataset key for `--data-files` |
| `path` | File name relative to `DATA_ROOT` |
| `split_mode` | `temporal` splits one series in time order into training, validation, and test parts. `across_batches` and `within_batches` serve datasets made of separate runs, which `batch_column` identifies |
| `input_channels` | Columns the model receives, in order. Required, because every run records this order |
| `channel_groups`, `default_target` | Named groups of forecast targets and the group forecast by default. `--data-targets` selects another group |
| `continuous_channels`, `discrete_channels` | Partition of the input channels by type. The scored scenarios act on continuous channels, except MissingData, which acts on all channels |
| `description` | Short description of the source and the task |

Channel types come from this entry and are never inferred from the data. An
entry with missing or contradictory channels raises when the registry loads.
The dataset key, the channels, the window lengths, and the split settings
enter the `data_config_signature`, which keeps runs on different data
configurations apart.

A first run checks the integration:

```bash
uv run python run_training.py --data-files MyPlant --model DLinear --method baseline --max-hp-trials-per-model 1 --max-epochs 1
uv run python run_testing.py --data-files MyPlant --model DLinear --method baseline --max-hp-trials-per-model 1
uv run python run_analysis.py --data-files MyPlant --model DLinear --method baseline --max-hp-trials-per-model 1
```

## Datasets for the Benchmark Itself

A dataset proposed as part of the benchmark also needs:

- acquisition steps, source terms, and validation commands in
  [`data/README.md`](../data/README.md)
- a validation entry in
  [`scripts/benchmark_dataset_contracts.py`](../scripts/benchmark_dataset_contracts.py)
- an entry in `DATA_FILES` of
  [`configs/defaults.yaml`](../configs/defaults.yaml)
- display metadata in
  [`configs/benchmark_scope.yaml`](../configs/benchmark_scope.yaml)
