"""Checks that the README tables, counts, and commands match their sources of truth.

The README restates the benchmark scope, the scored scenarios, the transfer
families, the compute budget, and the results of the default configuration.
Each statement is compared with the configuration, registry, artifact, or
generator that owns it.
"""

from __future__ import annotations

import csv
import inspect
import re
from pathlib import Path

import yaml

from config_loader import load_defaults
from data.perturbations import PERTURBATION_REGISTRY
from scripts.benchmark_dataset_contracts import CONTRACTS_BY_KEY
from scripts.render_readme_results import render_readme


REPO_ROOT = Path(__file__).resolve().parents[1]
README_PATH = REPO_ROOT / "README.md"
# The README budget statements describe the seed-42 default configuration.
DEFAULT_CONFIGURATION_CAMPAIGN = "reference_seed42"
PAPER_SMOKE_LISTING = (
    "uv sync",
    "uv run python run_training.py --data-files ETTh1 --model DLinear --method baseline "
    "--max-epochs 1 --max-hp-trials-per-model 1 --logdir runs",
    "uv run python run_testing.py --data-files ETTh1 --model DLinear --method baseline "
    "--perturbation-scenarios missing_data noise --max-hp-trials-per-model 1 --logdir runs",
    "uv run python run_analysis.py --data-files ETTh1 --model DLinear --method baseline "
    "--perturbation-scenarios missing_data noise --max-hp-trials-per-model 1 --logdir runs",
)
# Constructor parameters that hold the severity endpoints. None marks an
# endpoint that the operator fixes at zero.
ENDPOINT_PARAMETERS = {
    "drift": ("min_offset", "max_offset"),
    "attenuation": ("min_factor", "max_factor"),
    "noise": ("min_sd", "max_sd"),
    "spike": ("min_hickup", "max_hickup"),
    "time_stretch": ("min_rate", "max_rate"),
    "time_compress": ("min_rate", "max_rate"),
    "stuck_sensor": (None, "max_duration_frac"),
    "missing_data": (None, "max_gap_frac"),
}
CHANNEL_SCOPE_LABELS = {"continuous": "continuous subset", "all": "all channels"}
INCLUSION_MARKS = {True: "✓", False: "–"}


def _readme() -> str:
    return README_PATH.read_text(encoding="utf-8")


def _yaml(path: str) -> dict[str, object]:
    return yaml.safe_load((REPO_ROOT / path).read_text(encoding="utf-8"))


def _pairs(entries: list[str]) -> dict[str, str]:
    return dict(entry.split("=", 1) for entry in entries)


def _table(header_prefix: str) -> list[list[str]]:
    lines = [line.strip() for line in _readme().splitlines()]
    (start,) = [index for index, line in enumerate(lines) if line.startswith(header_prefix)]
    rows = []
    for line in lines[start + 2:]:
        if not line.startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip("|").split("|")])
    return rows


def _code_tokens(cell: str) -> list[str]:
    return re.findall(r"`([^`]+)`", cell)


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def test_readme_generated_blocks_are_current():
    text = _readme()

    assert render_readme(text) == text


def test_readme_scope_strip_counts_match_configs():
    scope = _yaml("configs/benchmark_scope.yaml")
    match = re.search(
        r"(\d+) datasets · (\d+) sensor-fault scenarios · (\d+) forecasting architectures · "
        r"(\d+) robustness-improvement methods</sub>",
        _readme(),
    )

    assert match is not None
    assert [int(value) for value in match.groups()] == [
        len(scope["datasets"]["display_order"]),
        len(load_defaults()["PERTURBATION_SCENARIOS"]),
        len(scope["architectures"]["display_order"]),
        len([method for method in scope["methods"]["display_order"] if method != "baseline"]),
    ]


def test_readme_scenario_table_matches_registry_and_reporting_config():
    figures = _yaml("configs/reporting/core_figures.yaml")
    display = _pairs(figures["CORE_SCENARIO_DISPLAY"])
    classes = {
        scenario: scenario_class
        for scenario_class, members in _pairs(figures["CORE_SCENARIO_GROUPS"]).items()
        for scenario in members.split(",")
    }
    rows = _table("| Class | Scenario |")
    keys = [_code_tokens(row[1])[0] for row in rows]

    assert keys == load_defaults()["PERTURBATION_SCENARIOS"]
    for key, (row_class, scenario, _construction, severity_range, channels, _behavior) in zip(
        keys, rows
    ):
        operator = PERTURBATION_REGISTRY[key]
        parameters = inspect.signature(operator.__init__).parameters
        low_name, high_name = ENDPOINT_PARAMETERS[key]
        expected = (
            0.0 if low_name is None else float(parameters[low_name].default),
            float(parameters[high_name].default),
        )
        documented = re.search(r"(\d+(?:\.\d+)?) to (\d+(?:\.\d+)?)$", severity_range)

        assert row_class == classes[key]
        assert scenario == f"{display[key]} (`{key}`)"
        assert documented is not None, severity_range
        assert (float(documented.group(1)), float(documented.group(2))) == expected
        assert channels == CHANNEL_SCOPE_LABELS[operator.channel_scope]


def test_readme_transfer_families_match_recipe_and_stay_disjoint():
    recipe = _yaml("configs/pipelines/fault_augmentation.yaml")
    profiles = recipe["train_fault_profiles"]
    grid = recipe["recipe_params"]["train_perturbation_profile"]["grid"]
    sentence = re.search(
        r"seven training-only transfer fault families,(.*?)Its hyperparameter",
        _readme(),
        re.S,
    )

    assert sentence is not None
    documented = _code_tokens(sentence.group(1))
    assert documented == profiles["holdout"]["scenarios"]
    assert set(documented) == {
        scenario for profile in grid for scenario in profiles[profile]["scenarios"]
    }
    assert not set(documented) & set(load_defaults()["PERTURBATION_SCENARIOS"])
    assert "two profiles" in _normalized(_readme()) and len(grid) == 2


def test_readme_dataset_table_matches_registry_contracts_and_windows():
    scope = _yaml("configs/benchmark_scope.yaml")
    windows = _yaml("configs/dataset_windows.yaml")
    rows = _table("| Key | Dataset |")

    assert [_code_tokens(row[0])[0] for row in rows] == scope["datasets"]["display_order"]
    for row in rows:
        key = _code_tokens(row[0])[0]
        contract = CONTRACTS_BY_KEY[key]
        targets = contract.channel_count if contract.target_alias == "all" else 1

        assert row[1] == scope["datasets"]["display_labels"][key]
        assert int(row[3]) == contract.channel_count
        assert int(row[4]) == targets
        assert int(row[5]) == windows[key]["input_len"]
        assert int(row[6]) == windows[key]["target_len"]


def test_readme_architecture_table_matches_scope_roles():
    scope = _yaml("configs/benchmark_scope.yaml")
    architectures = scope["architectures"]
    baseline_architectures = set(scope["methods"]["method_architecture_applicability"]["baseline"])
    method_comparison_architectures = set(architectures["roles"]["method_comparison"])
    rows = _table("| Key | Architecture | Architecture comparison | Method-baseline comparison |")

    assert [_code_tokens(row[0])[0] for row in rows] == architectures["display_order"]
    for row in rows:
        key = _code_tokens(row[0])[0]
        assert row[1] == architectures["display_labels"][key]
        assert row[2] == INCLUSION_MARKS[key in baseline_architectures]
        assert row[3] == INCLUSION_MARKS[key in method_comparison_architectures]


def test_readme_method_table_matches_recipes_and_applicability():
    scope = _yaml("configs/benchmark_scope.yaml")
    methods = scope["methods"]
    architecture_labels = scope["architectures"]["display_labels"]
    rows = _table("| Key | Method | Kind | Applies to |")

    assert [_code_tokens(row[0])[0] for row in rows] == methods["display_order"]
    assert set(methods["display_order"]) == set(load_defaults()["BENCHMARK_METHODS"])
    for row in rows:
        key = _code_tokens(row[0])[0]
        recipe = _yaml(f"configs/pipelines/{key}.yaml")

        assert row[1] == methods["display_labels"][key]
        assert row[2] == recipe["pipeline_kind"]
        assert row[3].split(", ") == [
            architecture_labels[architecture]
            for architecture in methods["method_architecture_applicability"][key]
        ]


def test_readme_smoke_commands_match_the_paper_listing():
    text = _readme()
    start = text.index("```bash", text.index("Run the smoke benchmark.")) + len("```bash")
    block = text[start:text.index("```", start)]

    assert tuple(line.strip() for line in block.strip().splitlines()) == PAPER_SMOKE_LISTING


def test_readme_compute_figures_match_the_ledger():
    hours: dict[str, float] = {}
    with (REPO_ROOT / "paper_artifact/compute_accounting/compute_aggregates.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        for row in csv.DictReader(handle):
            if (
                row["campaign_id"] == DEFAULT_CONFIGURATION_CAMPAIGN
                and row["group"] == "training_by_dataset"
            ):
                assert row["key"] not in hours, row["key"]
                hours[row["key"]] = float(row["hours"])
    assert set(hours) == set(CONTRACTS_BY_KEY)
    total = round(sum(hours.values()))
    starter = round(hours["BeijingAir_Tiantan"] + hours["ETTh1"])
    text = _normalized(_readme())

    assert f"recorded {total} training GPU-hours, most of them on Traffic" in text
    assert max(hours, key=hours.get) == "traffic"
    assert f"about {starter} of the {total} recorded training GPU-hours" in text


def test_readme_default_configuration_states_the_test_window_budget():
    with (REPO_ROOT / "paper_artifact/paper_inputs/main_table_backbone.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        budgets = {int(row["n_test_samples"]) for row in csv.DictReader(handle)}

    assert budgets == {10000}
    assert "evaluated on 10,000 sampled test windows per dataset" in _normalized(_readme())


TREE_ENTRY = re.compile(r"^((?:│   |    )*)(?:├── |└── )(\S+)")


def _layout_paths(block: str) -> list[str]:
    """Return the repository-relative paths of a tree drawn with box characters."""
    root, *lines = block.strip().splitlines()
    assert root == "SensorFault-Bench/"
    parents: list[str] = []
    paths = []
    for line in lines:
        entry = TREE_ENTRY.match(line)
        if entry is None:
            assert line.strip() == "│", line
            continue
        depth = len(entry.group(1)) // 4
        name = entry.group(2).rstrip("/")
        parents = [*parents[:depth], name]
        paths.append("/".join(parents))
    return paths


def test_readme_repository_layout_paths_exist():
    text = _readme()
    start = text.index("```text", text.index("## Repository layout")) + len("```text")
    block = text[start:text.index("```", start)]
    paths = _layout_paths(block)

    assert "run_training.py" in paths
    assert "data/perturbations.py" in paths
    for path in paths:
        assert (REPO_ROOT / path).exists(), path


def _section(text: str, heading: str) -> str:
    start = text.index(heading)
    end = text.find("\n## ", start + len(heading))
    return text[start:] if end == -1 else text[start:end]


def test_readme_comparison_rule_matches_manifest_defaults_and_schema():
    section = _normalized(_section(_readme(), "## Evaluating a new model or method"))
    defaults = load_defaults()
    manifest = yaml.safe_load((REPO_ROOT / "paper_artifact/MANIFEST.json").read_text(encoding="utf-8"))
    schema = (REPO_ROOT / "docs/reproducing_benchmark.md").read_text(encoding="utf-8")

    assert "`data_config_signatures`" in section
    assert set(manifest["data_config_signatures"]) == set(defaults["DATA_FILES"])
    assert all(len(signatures) == 1 for signatures in manifest["data_config_signatures"].values())
    assert f"`n_test_samples` is {defaults['N_TEST_SAMPLES']:,}" in section
    assert f"seed {defaults['SEED']}" in section
    assert "`robustness_scoring_semantics` is `uniform_severity_degradation`" in section
    for column in ("data_config_signature", "n_test_samples", "robustness_scoring_semantics"):
        assert f"`{column}`" in schema
    for table in ("full_results.csv", "improvement_deltas_selected.csv", "pipeline_method_delta_results.csv"):
        assert f"tables/{table}" in section
        assert table in schema


def test_download_commands_fetch_the_registered_dataset_files():
    from data.datasets import DATASET_REGISTRY

    data_root = load_defaults()["DATA_ROOT"]
    readme = _readme()
    data_guide = (REPO_ROOT / "data/README.md").read_text(encoding="utf-8")
    record_links = dict(
        re.findall(r"`(\w+)`:\s+<(https://doi\.org/10\.5281/zenodo\.\d+)>", data_guide)
    )

    etth1_path = f"{data_root}/{DATASET_REGISTRY.get('ETTh1').path}"
    assert f"curl -L -o {etth1_path} https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv" in readme
    assert f"--output {etth1_path} --validate-existing" in readme
    assert set(record_links) == {"BeijingAir_Tiantan", "Penmanshiel_Hourly_WT08"}
    for key, record_link in record_links.items():
        filename = DATASET_REGISTRY.get(key).path
        record_id = record_link.rsplit(".", 1)[1]
        download = f"curl -L -o {data_root}/{filename} https://zenodo.org/records/{record_id}/files/{filename}\n"
        assert download in data_guide, key
        assert record_link in readme
