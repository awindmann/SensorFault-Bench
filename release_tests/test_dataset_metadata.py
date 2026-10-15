import json
from pathlib import Path

import yaml

from config_loader import load_dataset_windows, load_defaults
from data.datasets import DATASET_REGISTRY
from release_tests.repository_files import private_value_offenders


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BENCHMARK_DATASETS = {
    "ETTh1": "ETTh1.csv",
    "traffic": "traffic.parquet",
    "BeijingAir_Tiantan": "beijing_air_tiantan.parquet",
    "Penmanshiel_Hourly_WT08": "penmanshiel_hourly_wt08.parquet",
}
PROVENANCE_NOTEBOOKS = {
    "beijing_air_tiantan_derivation.ipynb",
    "penmanshiel_hourly_wt08_derivation.ipynb",
}
CROISSANT_FILES = {
    "BeijingAir_Tiantan": "data/croissant/beijing_air_tiantan.json",
    "Penmanshiel_Hourly_WT08": "data/croissant/penmanshiel_hourly_wt08.json",
}
RAI_FIELDS = (
    "rai:dataLimitations",
    "rai:dataBiases",
    "rai:personalSensitiveInformation",
    "rai:dataUseCases",
    "rai:dataSocialImpact",
    "rai:hasSyntheticData",
)


def test_provenance_notebooks_are_allowlisted_benchmark_safe_and_markdown_only():
    provenance_dir = REPO_ROOT / "notebooks/provenance"
    notebooks = {path.name: path for path in provenance_dir.glob("*.ipynb")}
    assert set(notebooks) == PROVENANCE_NOTEBOOKS

    for notebook_path in notebooks.values():
        assert notebook_path.stat().st_size < 30000
        text = notebook_path.read_text(encoding="utf-8")
        assert private_value_offenders(text) == ()
        assert "Original source:" in text
        assert "Benchmark derived dataset:" in text
        assert "## Slice Rationale" in text
        assert "## EDA Summary" in text
        assert "## Benchmark Validation" in text
        payload = json.loads(text)
        for cell in payload["cells"]:
            assert cell["cell_type"] == "markdown"
            assert "outputs" not in cell
            assert "execution_count" not in cell


def test_benchmark_yaml_configs_are_private_value_free():
    for yaml_path in (REPO_ROOT / "configs").rglob("*.yaml"):
        config_text = yaml_path.read_text(encoding="utf-8")
        yaml.safe_load(config_text)
        assert private_value_offenders(config_text) == (), yaml_path


def test_croissant_metadata_matches_curated_records_and_cites_the_paper():
    guide = (REPO_ROOT / "data/README.md").read_text(encoding="utf-8")
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    bibtex = readme.split("```bibtex\n", 1)[1].split("```", 1)[0].strip()

    assert sorted(path.name for path in (REPO_ROOT / "data/croissant").iterdir()) == sorted(
        Path(path).name for path in CROISSANT_FILES.values()
    )
    for key, path in CROISSANT_FILES.items():
        metadata = json.loads((REPO_ROOT / path).read_text(encoding="utf-8"))
        record_url, _, version = metadata["url"].partition("/versions/")
        guide_entry = guide.split(f"- `{key}`:", 1)[1].split("\n- ", 1)[0]

        assert record_url in guide_entry, path
        assert version == str(metadata["version"]), path
        assert metadata["license"]["name"] == "CC BY-SA 4.0", path
        assert metadata["dct:conformsTo"] == "http://mlcommons.org/croissant/RAI/1.0", path
        for field in RAI_FIELDS:
            assert field in metadata and str(metadata[field]).strip(), (path, field)
        assert metadata["citeAs"] == bibtex, path


def test_shipped_defaults_point_to_local_benchmark_paths():
    defaults = load_defaults()

    assert defaults["DATA_ROOT"] == "data/processed"
    assert defaults["LOGDIR"] == "runs"
    assert defaults["MINIO_ENDPOINT"] == ""


def test_benchmark_dataset_registry_and_window_defaults_are_limited_to_benchmark_scope():
    expected_keys = set(EXPECTED_BENCHMARK_DATASETS)
    defaults = load_defaults()

    assert set(DATASET_REGISTRY.keys()) == expected_keys
    assert set(load_dataset_windows(defaults=defaults)) == expected_keys


def test_data_guide_lists_benchmark_filenames_and_validation_surface():
    guide = (REPO_ROOT / "data/README.md").read_text(encoding="utf-8")

    for key, filename in EXPECTED_BENCHMARK_DATASETS.items():
        assert key in guide
        assert filename in guide
    for required_phrase in (
        "row-count and channel-count",
        "validation",
        "scripts/benchmark_dataset_contracts.py",
        "dataframe-content SHA256",
        "file-byte fixity",
        "scripts/generate_dataset_checksums.py",
        "https://www.kaggle.com/datasets/sensorfaultbench/beijing-air-tiantan-pm2-5-time-series-dataset",
        "https://www.kaggle.com/datasets/sensorfaultbench/penmanshiel-wt08-hourly-scada-time-series-dataset",
    ):
        assert required_phrase in guide


def test_data_guide_documents_traffic_runtime_and_csv_paths():
    guide = (REPO_ROOT / "data/README.md").read_text(encoding="utf-8")
    sources = guide[guide.index("## Sources and Terms") :]
    traffic_section = sources[
        sources.index("- `traffic`:")
        : sources.index("- `BeijingAir_Tiantan`:")
    ]

    assert "scripts/preprocess_traffic.py" in traffic_section
    assert "traffic.parquet" in traffic_section
    assert "DATA_ROOT/traffic.csv" in traffic_section
