# Data Preparation

The benchmark resolves datasets by key and filename under `DATA_ROOT`.
The default is `DATA_ROOT=data/processed`. Raw and processed datasets are not
redistributed by default.

## Runtime Profiles

Local profile:

```bash
--data-root data/processed --logdir runs
```

VM or object-store profile:

```bash
--data-root s3://<bucket>/<prefix> \
--logdir <mlflow-tracking-uri> \
--minio-endpoint <object-store-endpoint>
```

The local profile uses the empty `MINIO_ENDPOINT` default from
`configs/defaults.yaml`. Credentials must be supplied through the shell or VM
secret manager, not through committed YAML. For MinIO, pass `--minio-endpoint`
explicitly. Standard S3-compatible deployments that rely on provider defaults
can leave it empty. Changing `DATA_ROOT`, `LOGDIR`, `MINIO_ENDPOINT`, or
credentials does not change the data signature or run identity.

## Benchmark Datasets

| Key | Expected filename under `DATA_ROOT` | Source | Rows | Modeled channels | Target alias |
| --- | --- | --- | ---: | ---: | --- |
| `ETTh1` | `ETTh1.csv` | ETT benchmark artifact | 17420 | 7 | `all` |
| `traffic` | `traffic.parquet` | PeMS traffic benchmark derivative | 17544 | 862 | `all` |
| `BeijingAir_Tiantan` | `beijing_air_tiantan.parquet` | Beijing Multi-Site Air Quality, Tiantan station | 24038 | 12 | `pm25` |
| `Penmanshiel_Hourly_WT08` | `penmanshiel_hourly_wt08.parquet` | Penmanshiel wind-farm SCADA, WT08 | 25867 | 65 | `power` |

`ETTh1` uses a `date` column followed by `HUFL`, `HULL`, `MUFL`, `MULL`,
`LUFL`, `LULL`, and `OT`. `traffic` uses 862 numeric detector channels, named
`0` through `860` plus `OT`. `BeijingAir_Tiantan` uses continuous channels
`PM2.5`, `PM10`, `SO2`, `NO2`, `CO`, `O3`, `TEMP`, `PRES`, `DEWP`, `RAIN`,
`WSPM`, and discrete channel `wd`. `Penmanshiel_Hourly_WT08` uses the 65
continuous WT08 SCADA channels declared in `data/datasets/specs.py`.

## Dataset Setup

Save the four benchmark files under `DATA_ROOT`. The default local layout is:

```text
data/processed/ETTh1.csv
data/processed/traffic.parquet
data/processed/beijing_air_tiantan.parquet
data/processed/penmanshiel_hourly_wt08.parquet
```

To run the benchmark, save the files in the layout above. `ETTh1.csv` is
consumed directly and must contain the expected source columns. The other three
benchmark datasets are consumed as Parquet files.

`traffic` has a two-file role. The upstream source is the public
`traffic.csv`, while the registered benchmark key `traffic` resolves only to
`traffic.parquet`. Run `scripts/preprocess_traffic.py` to validate the upstream
CSV and create the canonical Parquet runtime file without changing the sensor
series.

If you do not already have the required Parquet files, generate them from the
raw sources with the preprocessing scripts in
[Benchmark Dataset Validation](#benchmark-dataset-validation). If you already
have ready-to-use Parquet files, validate the staged files with the commands
below.

Ready-to-use processed records for the two derived single-target datasets are
published separately:

- `BeijingAir_Tiantan`: <https://www.kaggle.com/datasets/sensorfaultbench/beijing-air-tiantan-pm2-5-time-series-dataset>
- `Penmanshiel_Hourly_WT08`: <https://www.kaggle.com/datasets/sensorfaultbench/penmanshiel-wt08-hourly-scada-time-series-dataset>

These Kaggle-hosted derived dataset records are licensed under CC BY-SA 4.0.
Their upstream sources retain the source-specific terms listed below.

Download the Parquet file from each record, place it under `DATA_ROOT` with the
expected filename above, then run the validation commands below.

```bash
uv run python scripts/preprocess_beijing_air_tiantan.py \
  --output data/processed/beijing_air_tiantan.parquet \
  --validate-existing

uv run python scripts/preprocess_penmanshiel_hourly_wt08.py \
  --output data/processed/penmanshiel_hourly_wt08.parquet \
  --validate-existing

uv run python scripts/preprocess_traffic.py \
  --output data/processed/traffic.parquet \
  --validate-existing

uv run python scripts/benchmark_dataset_contracts.py \
  --dataset ETTh1 \
  --output data/processed/ETTh1.csv \
  --validate-existing
```

These commands provide row-count and channel-count validation against the
dataset registry metadata, expected filename, column order, timestamp interval
and integrity, finite continuous values, and discrete value domains where a
dataset has discrete channels.

Two checksum surfaces are used by the public dataset materials:

- Runtime validation: add `--require-checksums` to validate the canonical
  dataframe-content SHA256 recorded in
  `scripts/benchmark_dataset_contracts.py`. This checks table content, not
  Parquet file bytes.
- Dataset-record fixity: standalone dataset records may include
  `checksums.sha256` files generated with
  `scripts/generate_dataset_checksums.py`. These are exact file-byte fixity
  checks for `sha256sum -c` or `shasum -a 256 -c`. They verify that a downloaded
  Parquet or CSV file matches the uploaded file.

Parquet writer metadata, writer-version differences, timestamp physical units,
and creation-time metadata are not part of the benchmark dataset identity. A
newly reprocessed Parquet container can therefore pass content validation
without being byte-identical to an uploaded dataset file.

## Sources And Terms

Raw and processed benchmark datasets are not redistributed by this repository
by default. Users are responsible for following the external source terms for
each dataset.

- `ETTh1.csv`: acquire from the ETT benchmark data archive, commonly mirrored
  with the ETT forecasting benchmark at
  <https://github.com/zhouhaoyi/ETDataset/tree/main/ETT-small>. Place the file
  directly under `DATA_ROOT`.
- `traffic`: acquire the standard long-term-forecasting `traffic.csv` file
  with columns `date`, `0` through `860`, and `OT` from the THUML
  Time-Series-Library dataset collection at
  <https://huggingface.co/datasets/thuml/Time-Series-Library/blob/main/traffic/traffic.csv>.
  The source collection is archived at
  <https://doi.org/10.5281/zenodo.4656132>.
  This is the exact THUML Traffic CSV used by this benchmark. It starts at
  `2016-07-01 02:00:00`, ends at `2018-07-02 01:00:00`, and has no internal
  hourly timestamp gaps. Convert `traffic.csv` to the benchmark Parquet file
  with `scripts/preprocess_traffic.py`. The repository does not redistribute a
  new Traffic dataset record. It records the upstream source, validates the
  public CSV, and uses `traffic.parquet` only as the benchmark runtime format.
  For a local convenience run, you can use the original CSV directly by editing
  the `traffic` `DatasetSpec` in `data/datasets/specs.py` from
  `path="traffic.parquet"` to `path="traffic.csv"` and placing the upstream file
  at `DATA_ROOT/traffic.csv`. Leave the key, split mode, target alias, and
  channel metadata unchanged. The CSV loader parses the first `date` column as
  the datetime index and drops it, so the modeled channels remain `0` through
  `860` plus `OT`. Convert with `scripts/preprocess_traffic.py` and restore
  `path="traffic.parquet"` before running the benchmark dataset validation
  commands or presenting results as the canonical `traffic.parquet` setup,
  because the validation contract expects the Parquet runtime filename.
- `BeijingAir_Tiantan`: acquire the Beijing Multi-Site Air Quality source from
  the UCI Machine Learning Repository at
  <https://doi.org/10.24432/C5RK5G>. Use the Tiantan station source file
  `PRSA_Data_Tiantan_20130301-20170228.csv` for the derived Tiantan slice.
- `Penmanshiel_Hourly_WT08`: acquire the Penmanshiel wind-farm SCADA source
  from the Zenodo v3 record at <https://zenodo.org/records/16807304>, DOI
  <https://doi.org/10.5281/zenodo.16807304>. Use the
  `Penmanshiel_SCADA_2016_WT01-10_3107.zip` through
  `Penmanshiel_SCADA_2022_WT01-10_4462.zip` files. Place those WT01-10 ZIPs
  in one source directory, and the preprocessing script extracts the derived
  WT08 slice.

The repository code license does not grant rights to external datasets. Cite
and comply with each upstream dataset source separately.

## Benchmark Dataset Validation

Validate the staged ETTh1 file and the derived Beijing Air Tiantan, Penmanshiel
WT08, and traffic benchmark files with `scripts/benchmark_dataset_contracts.py`.

The preprocessing scripts support two modes: preparing a processed benchmark
file from an upstream raw source, and validating an already staged processed
file. They print a JSON summary with the dataset key, filename, row count,
channel count, and canonical dataframe-content SHA256.

Prepare the derived datasets from raw sources:

```bash
uv run python scripts/preprocess_beijing_air_tiantan.py \
  --raw-source <path-to-PRSA_Data_Tiantan_20130301-20170228.csv> \
  --output data/processed/beijing_air_tiantan.parquet

uv run python scripts/preprocess_penmanshiel_hourly_wt08.py \
  --raw-source <path-to-penmanshiel-wt08-source> \
  --output data/processed/penmanshiel_hourly_wt08.parquet

uv run python scripts/preprocess_traffic.py \
  --raw-source <path-to-traffic.csv-or-directory> \
  --output data/processed/traffic.parquet
```

Accepted raw-source paths:

- `preprocess_beijing_air_tiantan.py` accepts the Tiantan station CSV, the UCI
  PRSA ZIP, or a directory containing exactly one of those files.
- `preprocess_penmanshiel_hourly_wt08.py` accepts a directory containing the
  required 2016-2022 `Penmanshiel_SCADA_*_WT01-10_*.zip` files, or one such ZIP
  file beside the rest of that subset.
- `preprocess_traffic.py` accepts `traffic.csv`, or a directory containing
  exactly one `traffic.csv`.
