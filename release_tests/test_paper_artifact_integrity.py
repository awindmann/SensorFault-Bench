"""Integrity checks for the bundled paper artifact.

Every file recorded in `paper_artifact/MANIFEST.json` and in the
supplementary-table manifest must exist with the recorded SHA-256 checksum, byte
size, and CSV row count. The artifact directory may contain no unrecorded file.
The released clean-validation hyperparameters must reproduce the recorded run
signatures of the selected baseline forecasting models, and the selected
Chronos-2 configurations must report the revision that the grid pins.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

from pipelines.signatures import build_signature
from scripts.build_supplementary_tables import (
    BASELINE_GRID_PATH,
    DATASETS,
    SELECTED_HPARAMS_FILENAME,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPO_ROOT / "paper_artifact"
SUPPLEMENTARY_ROOT = ARTIFACT_ROOT / "supplementary_tables"
RUN_LEDGER_PATH = ARTIFACT_ROOT / "compute_accounting" / "compute_ledger_official_runs.csv"
# The released hyperparameters belong to the seed-42 default configuration.
DEFAULT_CONFIGURATION_CAMPAIGN = "reference_seed42"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_row_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.reader(handle)) - 1


def _artifact_record_groups(manifest: dict[str, Any]) -> list[list[dict[str, Any]]]:
    return [
        manifest["files"],
        manifest["derived_paper_inputs"],
        manifest["supplementary_tables"]["files"],
        manifest["r_figure_artifact"]["files"],
        manifest["compute_accounting"]["files"],
    ]


def _assert_record_matches(record: dict[str, Any], path: Path) -> None:
    assert path.is_file(), record["path"]
    assert path.stat().st_size == record["size_bytes"], record["path"]
    assert _sha256(path) == record["sha256"], record["path"]
    if "row_count" in record:
        assert _csv_row_count(path) == record["row_count"], record["path"]


def _non_ignored_artifact_files() -> set[str]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "paper_artifact"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return {
        (REPO_ROOT / line).relative_to(ARTIFACT_ROOT).as_posix()
        for line in result.stdout.splitlines()
        if line.strip() and (REPO_ROOT / line).is_file()
    }


def test_manifest_records_match_file_bytes_sizes_and_row_counts():
    manifest = _load_json(ARTIFACT_ROOT / "MANIFEST.json")

    for group in _artifact_record_groups(manifest):
        assert group
        for record in group:
            _assert_record_matches(record, ARTIFACT_ROOT / record["path"])


def test_supplementary_table_records_match_file_bytes_sizes_and_row_counts():
    supplementary_manifest = _load_json(SUPPLEMENTARY_ROOT / "manifest.json")

    assert supplementary_manifest["files"]
    for record in supplementary_manifest["files"]:
        _assert_record_matches(record, SUPPLEMENTARY_ROOT / record["path"])


def test_artifact_directory_contains_only_recorded_files():
    manifest = _load_json(ARTIFACT_ROOT / "MANIFEST.json")
    recorded = {
        record["path"]
        for group in _artifact_record_groups(manifest)
        for record in group
    }

    assert _non_ignored_artifact_files() == recorded | {"MANIFEST.json"}


def test_artifact_id_agrees_across_manifests():
    manifest = _load_json(ARTIFACT_ROOT / "MANIFEST.json")
    artifact_id = manifest["artifact_id"]

    assert artifact_id.endswith(f"-v{manifest['artifact_version']}")
    assert _load_json(ARTIFACT_ROOT / "config" / "eval_context.json")["artifact_id"] == (
        artifact_id
    )
    assert _load_json(ARTIFACT_ROOT / "forecast_examples" / "manifest.json")[
        "artifact_id"
    ] == artifact_id


def test_selected_hyperparameters_reproduce_the_recorded_run_signatures():
    # The ledger records one row for each timing span, so a run can repeat.
    with RUN_LEDGER_PATH.open(newline="", encoding="utf-8") as handle:
        winners: dict[tuple[str, str], dict[str, tuple[str, str]]] = {}
        for row in csv.DictReader(handle):
            if (
                row["campaign_id"] == DEFAULT_CONFIGURATION_CAMPAIGN
                and row["pipeline_method"] == "baseline"
                and row["best_model"] == "true"
            ):
                winners.setdefault(
                    (row["dataset"], row["model_architecture"]),
                    {},
                )[row["run_id"]] = (row["data_config_signature"], row["signature"])
    selected_path = SUPPLEMENTARY_ROOT / "selected_clean_validation_hyperparameters.csv"
    with selected_path.open(newline="", encoding="utf-8") as handle:
        selected_rows = list(csv.DictReader(handle))

    assert {
        (row["dataset"], row["model_architecture"]) for row in selected_rows
    } == set(winners)
    for row in selected_rows:
        ((data_config_signature, signature),) = winners[
            (row["dataset"], row["model_architecture"])
        ].values()
        assert build_signature(
            row["model_architecture"],
            row["dataset"],
            json.loads(row["hyperparameters_json"]),
            pipeline_id="baseline",
            data_config_signature=data_config_signature,
        ) == signature, (row["dataset"], row["model_architecture"])


def test_selected_chronos2_configurations_report_the_pinned_revision():
    with (REPO_ROOT / BASELINE_GRID_PATH).open(encoding="utf-8") as handle:
        chronos_grid = yaml.safe_load(handle)["Chronos2"]
    path = SUPPLEMENTARY_ROOT / SELECTED_HPARAMS_FILENAME
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["model_architecture"] == "Chronos2"
        ]

    assert len(rows) == len(DATASETS)
    for row in rows:
        assert json.loads(row["hyperparameters_json"]) == {
            "chronos_model_id": chronos_grid["chronos_model_id"][0],
            "chronos_model_revision": chronos_grid["chronos_model_revision"][0],
            "loss": "MSE",
        }
