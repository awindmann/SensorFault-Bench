# Data Preparation

The benchmark finds each dataset by its key and filename under `DATA_ROOT`,
which defaults to `data/processed`. The repository itself contains no raw or
processed dataset files.

## Download

The commands below place the four benchmark files under the default
`DATA_ROOT`. ETTh1 and Traffic come from their public sources, with Traffic
converted to the benchmark Parquet file. The two curated datasets come from
their Kaggle records, which need no Kaggle account.

```bash
mkdir -p data/processed data/raw
curl -L -o data/processed/ETTh1.csv https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv
curl -L -o data/raw/traffic.csv https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/traffic/traffic.csv
uv run python scripts/preprocess_traffic.py --raw-source data/raw/traffic.csv --output data/processed/traffic.parquet
curl -L -o beijing_air_tiantan.zip https://www.kaggle.com/api/v1/datasets/download/sensorfaultbench/beijing-air-tiantan-pm2-5-time-series-dataset
unzip -j beijing_air_tiantan.zip beijing_air_tiantan.parquet -d data/processed
curl -L -o penmanshiel_hourly_wt08.zip https://www.kaggle.com/api/v1/datasets/download/sensorfaultbench/penmanshiel-wt08-hourly-scada-time-series-dataset
unzip -j penmanshiel_hourly_wt08.zip penmanshiel_hourly_wt08.parquet -d data/processed
```

Then validate the four files. With `--require-checksums`, validation also
compares the checksum of the table content.

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

The validation compares each file with the dataset registry. Besides
row-count and channel-count checks, it covers the filename, the column order,
the timestamp interval and integrity, the finiteness of continuous values, and
the value domains of discrete channels. Datasets on an object store, such as
S3, are configured with the
[runtime profiles](../docs/reproducing_benchmark.md#runtime-profiles).

## Benchmark Datasets

| Key | Expected filename under `DATA_ROOT` | Source | Rows | Modeled channels | Target alias |
| --- | --- | --- | ---: | ---: | --- |
| `ETTh1` | `ETTh1.csv` | ETT benchmark artifact | 17420 | 7 | `all` |
| `traffic` | `traffic.parquet` | PeMS traffic benchmark derivative | 17544 | 862 | `all` |
| `BeijingAir_Tiantan` | `beijing_air_tiantan.parquet` | Beijing Multi-Site Air Quality, Tiantan station | 24038 | 12 | `pm25` |
| `Penmanshiel_Hourly_WT08` | `penmanshiel_hourly_wt08.parquet` | Penmanshiel wind-farm SCADA, WT08 | 25867 | 65 | `power` |

`ETTh1` uses a `date` column followed by `HUFL`, `HULL`, `MUFL`, `MULL`,
`LUFL`, `LULL`, and `OT`. `traffic` uses 862 numeric detector channels, named
`0` through `860` plus `OT`. `BeijingAir_Tiantan` uses the discrete channel `wd`
and the continuous channels `PM2.5`, `PM10`, `SO2`, `NO2`, `CO`, `O3`, `TEMP`,
`PRES`, `DEWP`, `RAIN`, and `WSPM`. `Penmanshiel_Hourly_WT08` uses the 65
continuous WT08 SCADA channels declared in `data/datasets/specs.py`.

## File Layout

Save the four benchmark files under `DATA_ROOT`. The default local layout is:

```text
data/processed/ETTh1.csv
data/processed/traffic.parquet
data/processed/beijing_air_tiantan.parquet
data/processed/penmanshiel_hourly_wt08.parquet
```

The benchmark reads `ETTh1.csv` directly, so the file must keep its source
columns, whereas it reads the other three datasets as Parquet files.

Traffic involves two files. The upstream source is the public `traffic.csv`,
whereas the benchmark key `traffic` resolves only to `traffic.parquet`.
`scripts/preprocess_traffic.py` validates the upstream CSV and converts it to
this Parquet file without changing the sensor series.

To rebuild the Parquet files from the raw sources instead, use the
preprocessing scripts in
[Benchmark Dataset Validation](#benchmark-dataset-validation).

The commands under [Download](#download) fetch and validate the processed
records of the two curated single-target datasets, which are published
separately on Kaggle:

- `BeijingAir_Tiantan`:
  <https://www.kaggle.com/datasets/sensorfaultbench/beijing-air-tiantan-pm2-5-time-series-dataset>
- `Penmanshiel_Hourly_WT08`:
  <https://www.kaggle.com/datasets/sensorfaultbench/penmanshiel-wt08-hourly-scada-time-series-dataset>

The Croissant metadata of both records is in [`croissant/`](croissant/). It
includes the Responsible AI fields, which the Croissant export of Kaggle leaves
out.

Both records are licensed under CC BY-SA 4.0, whereas their upstream sources
keep their own terms, which are listed below.

Two kinds of checksums protect the dataset files:

- **Content checksum.** With `--require-checksums`, validation compares the
  dataframe-content SHA256 recorded in
  `scripts/benchmark_dataset_contracts.py`. It covers the table content and
  ignores the bytes of the Parquet file.
- **File checksum.** A published dataset record can include a
  `checksums.sha256` file generated with
  `scripts/generate_dataset_checksums.py`. Its file-byte fixity checks, run with
  `sha256sum -c` or `shasum -a 256 -c`, confirm that a downloaded Parquet or CSV
  file matches the uploaded one.

The identity of a benchmark dataset excludes Parquet writer metadata, writer
versions, the physical units of timestamps, and creation times. A Parquet file
that is processed again can therefore pass content validation without matching
an uploaded file byte for byte.

## Sources and Terms

Raw and processed benchmark datasets are not redistributed with this
repository. Users must follow the terms of each external source.

- `ETTh1.csv`: acquire the file from the ETT archive at
  <https://github.com/zhouhaoyi/ETDataset/tree/main/ETT-small>, which
  accompanies the ETT forecasting benchmark, and place it directly under
  `DATA_ROOT`.
- `traffic`: acquire `traffic.csv` of the standard long-term forecasting
  benchmark, with the columns `date`, `0` through `860`, and `OT`, from the
  THUML Time-Series-Library dataset collection at
  <https://huggingface.co/datasets/thuml/Time-Series-Library/blob/main/traffic/traffic.csv>.
  The collection is archived at <https://doi.org/10.5281/zenodo.4656132>.
  The benchmark uses exactly this CSV, which runs from `2016-07-01 02:00:00` to
  `2018-07-02 01:00:00` without gaps in its hourly timestamps.
  `scripts/preprocess_traffic.py` validates it and converts it to
  `traffic.parquet`, the format in which the benchmark reads Traffic. The
  repository records the upstream source and publishes no Traffic dataset
  record of its own.

  For a quick local run, the original CSV can also be read directly. Change
  `path="traffic.parquet"` to `path="traffic.csv"` in the `traffic`
  `DatasetSpec` of `data/datasets/specs.py`, place the upstream file at
  `DATA_ROOT/traffic.csv`, and leave the key, split mode, target alias, and
  channel metadata unchanged. The CSV loader parses the first column, `date`,
  as the datetime index and drops it, so the modeled channels remain `0`
  through `860` plus `OT`. Because the dataset validation expects the Parquet
  file, restore `path="traffic.parquet"` and convert the CSV with
  `scripts/preprocess_traffic.py` before running the validation commands or
  reporting results for the benchmark setup.
- `BeijingAir_Tiantan`: acquire the Beijing Multi-Site Air Quality source from
  the UCI Machine Learning Repository at <https://doi.org/10.24432/C5RK5G>. The
  curated Beijing Air Tiantan dataset derives from the Tiantan station file
  `PRSA_Data_Tiantan_20130301-20170228.csv`.
- `Penmanshiel_Hourly_WT08`: acquire the Penmanshiel wind-farm SCADA source
  from the Zenodo v3 record at <https://zenodo.org/records/16807304>, DOI
  <https://doi.org/10.5281/zenodo.16807304>. Place the ZIP files
  `Penmanshiel_SCADA_2016_WT01-10_3107.zip` through
  `Penmanshiel_SCADA_2022_WT01-10_4462.zip` in one source directory, from which
  the preprocessing script extracts the curated Penmanshiel WT08 dataset.

The repository code license does not grant rights to external datasets. Cite
each upstream source and follow its terms separately.

## Benchmark Dataset Validation

`scripts/benchmark_dataset_contracts.py` validates the staged ETTh1 file and
the processed files of Beijing Air Tiantan, Penmanshiel WT08, and Traffic.

The preprocessing scripts either prepare a processed benchmark file from its raw
upstream source or validate a processed file that is already staged. Both modes
print a JSON summary with the dataset key, filename, row count, channel count,
and dataframe-content SHA256.

Prepare the processed files from the raw sources:

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

Paths accepted by `--raw-source`:

- `preprocess_beijing_air_tiantan.py` accepts the Tiantan station CSV, the UCI
  PRSA ZIP, or a directory containing exactly one of those files.
- `preprocess_penmanshiel_hourly_wt08.py` accepts a directory containing the
  required 2016-2022 `Penmanshiel_SCADA_*_WT01-10_*.zip` files, or one such ZIP
  file beside the rest of that subset.
- `preprocess_traffic.py` accepts `traffic.csv`, or a directory containing
  exactly one `traffic.csv`.
