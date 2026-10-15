"""Build the supplementary tables of the paper artifact."""

from __future__ import annotations

import csv
import hashlib
import io
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any
import zipfile

import yaml

from pipelines.signatures import build_signature


SUPPLEMENTARY_TABLES_DIRNAME = "supplementary_tables"
SUPPLEMENTARY_TABLES_MANIFEST = "manifest.json"

DATASETS = (
    "BeijingAir_Tiantan",
    "Penmanshiel_Hourly_WT08",
    "ETTh1",
    "traffic",
)
DATASET_LABELS = {
    "BeijingAir_Tiantan": "Beijing Air Tiantan",
    "Penmanshiel_Hourly_WT08": "Penmanshiel WT08",
    "ETTh1": "ETTh1",
    "traffic": "Traffic",
}
ARCHITECTURES = (
    "SeasonalNaive",
    "DLinear",
    "GRU",
    "ModernTCN",
    "TSMixer",
    "PatchTST",
    "Chronos2",
)
ARCHITECTURE_LABELS = {
    "SeasonalNaive": "SeasonalNaive",
    "DLinear": "DLinear",
    "GRU": "GRU",
    "ModernTCN": "ModernTCN",
    "TSMixer": "TSMixer",
    "PatchTST": "PatchTST",
    "Chronos2": "Chronos-2",
}
LEARNED_ARCHITECTURES = frozenset(ARCHITECTURES[1:6])
METHODS = (
    "randomized_training",
    "adversarial_training",
    "fault_augmentation",
    "revin",
    "adaptive_robust_loss",
    "ensemble",
    "randomized_smoothing",
)
METHOD_LABELS = {
    "randomized_training": "randomized training",
    "adversarial_training": "PGD adversarial training",
    "fault_augmentation": "fault augmentation",
    "revin": "RevIN",
    "adaptive_robust_loss": "adaptive robust loss",
    "ensemble": "ensemble aggregation",
    "randomized_smoothing": "randomized smoothing",
}
SCENARIOS = frozenset(
    {
        "drift",
        "attenuation",
        "noise",
        "spike",
        "time_stretch",
        "time_compress",
        "stuck_sensor",
        "missing_data",
    }
)


@dataclass(frozen=True)
class FrozenSource:
    """A frozen build input, staged under its neutral file name."""

    source_id: str
    filename: str
    sha256: str


@dataclass(frozen=True)
class TrainingSeedSource:
    seed: int
    source: FrozenSource


TRAINING_SEED_SOURCES = (
    TrainingSeedSource(
        seed=42,
        source=FrozenSource(
            source_id="training_seed_42_tables",
            filename="training_seed_42_tables.zip",
            sha256="5701ba7a356b1ee2a3bc8c99fef7a5befba59658c570e6bdbc3b0354279416a0",
        ),
    ),
    TrainingSeedSource(
        seed=1001,
        source=FrozenSource(
            source_id="training_seed_1001_tables",
            filename="training_seed_1001_tables.zip",
            sha256="a3ce51cb6452ef5dd1c135afbf2beeff7b08b50a40141a4d54402c14eccf5901",
        ),
    ),
    TrainingSeedSource(
        seed=1002,
        source=FrozenSource(
            source_id="training_seed_1002_tables",
            filename="training_seed_1002_tables.zip",
            sha256="52ce6039f859b5f37f7d750a70c9b9f4d39afc8bfc368a177cc617f089c925c6",
        ),
    ),
)
PATCHTST_DIAGNOSTIC_SOURCE = FrozenSource(
    source_id="patchtst_near_tie_scope_runs",
    filename="patchtst_near_tie_scope_runs.csv",
    sha256="7b185236c1cc83784ecb42ca0a09b46e1e471043e8f0c18981bc7415dc117f60",
)


@dataclass(frozen=True)
class SensitivityTable:
    filename: str
    role: str
    source: FrozenSource


# Each sensitivity table republishes one verified evidence table without its
# run identifiers and without the constant scope flag of the paired-method
# tables. All published values are copied cell by cell.
SENSITIVITY_TABLES = (
    SensitivityTable(
        filename="severity_weighting_architecture.csv",
        role="architecture results under uniform and Beta(1,2) severity sampling",
        source=FrozenSource(
            source_id="severity_weighting_architecture_evidence",
            filename="severity-weighting-architecture.csv",
            sha256="496c8bfcae4d714149c07de993f7f08cb7c9b6266c85429976ce96bec4d55247",
        ),
    ),
    SensitivityTable(
        filename="severity_weighting_method_pairs.csv",
        role="method-baseline deltas under uniform and Beta(1,2) severity sampling",
        source=FrozenSource(
            source_id="severity_weighting_method_pairs_evidence",
            filename="severity-weighting-method-pairs.csv",
            sha256="af4fcb66c6edecc9c1ec1afe8befeb0a216e377578b9f2cdfa61e82d0d81f52b",
        ),
    ),
    SensitivityTable(
        filename="severity_weighting_method_means.csv",
        role="dataset-level mean method-baseline deltas under both severity distributions",
        source=FrozenSource(
            source_id="severity_weighting_method_means_evidence",
            filename="severity-weighting-method-means.csv",
            sha256="1868b53e961f52010f60dc4a01d6840f473c8fb48531e3114aad23204b0c8f3c",
        ),
    ),
    SensitivityTable(
        filename="severity_weighting_ranks.csv",
        role="rank agreement and leader changes under Beta(1,2) severity sampling",
        source=FrozenSource(
            source_id="severity_weighting_ranks_evidence",
            filename="severity-weighting-ranks.csv",
            sha256="1e3e57adb60a12868c0c40c02cea05fbea09213535a6fdf3363ffdf6e39026c6",
        ),
    ),
    SensitivityTable(
        filename="severity_weighting_exceptions.csv",
        role="method effects whose sign changes under Beta(1,2) severity sampling",
        source=FrozenSource(
            source_id="severity_weighting_exceptions_evidence",
            filename="severity-weighting-exceptions.csv",
            sha256="05199a5e4ab70e09a09ff89cd7bea01ad3b7dd0b66fb5e7b6c3e149ba8051ff1",
        ),
    ),
    SensitivityTable(
        filename="fault_direction_architecture.csv",
        role="architecture results with Drift and Spike in the negative direction",
        source=FrozenSource(
            source_id="fault_direction_architecture_evidence",
            filename="fault-direction-architecture.csv",
            sha256="464f20dc196a07ad2433b1cfb7876d3aa12b8b3ee0e7c338e5d757602620a3dc",
        ),
    ),
    SensitivityTable(
        filename="timing_local_warp_architecture.csv",
        role="architecture results with the local timing warp in place of TimeCompress",
        source=FrozenSource(
            source_id="timing_local_warp_architecture_evidence",
            filename="local-warp-architecture.csv",
            sha256="2b5a70b7f148cc40950640461b16691060e26141bb55347aedcf85aebd41d36e",
        ),
    ),
    SensitivityTable(
        filename="timing_time_compress_clipping.csv",
        role="share of TimeCompress output positions held at the window endpoint",
        source=FrozenSource(
            source_id="timing_time_compress_clipping_evidence",
            filename="timecompress-clipping.csv",
            sha256="ed1e7d7728fd249f5a26ee49756c5549e339a2d75f12c87d728e6bbf4e95b7a4",
        ),
    ),
    SensitivityTable(
        filename="timing_time_compress_omission_architecture.csv",
        role="architecture results with TimeCompress omitted from the scored set",
        source=FrozenSource(
            source_id="timing_time_compress_omission_architecture_evidence",
            filename="timecompress-omission-architecture.csv",
            sha256="e2f59845bc668c794673470751a9ffbaa37c8fed73c0d1a2ac5271c782375c67",
        ),
    ),
    SensitivityTable(
        filename="timing_time_compress_omission_method_pairs.csv",
        role="method-baseline deltas with TimeCompress omitted from the scored set",
        source=FrozenSource(
            source_id="timing_time_compress_omission_method_pairs_evidence",
            filename="timecompress-omission-method-pairs.csv",
            sha256="9f6b19d70d4d8391b793b9be28052d0c9963c6e20eed4cdbbc7d2c3b4efee6f4",
        ),
    ),
    SensitivityTable(
        filename="etth1_task_architecture.csv",
        role="ETTh1 architecture results across target scope and horizon",
        source=FrozenSource(
            source_id="etth1_task_architecture_evidence",
            filename="etth1-task-architecture.csv",
            sha256="355e50fa74a0a7629f192eb538255452c6676e0c9bfdca7f1b14d318b2be0964",
        ),
    ),
    SensitivityTable(
        filename="etth1_task_contrasts.csv",
        role="ETTh1 contrasts between target scopes and between horizons",
        source=FrozenSource(
            source_id="etth1_task_contrasts_evidence",
            filename="etth1-task-contrasts.csv",
            sha256="0e946f21bc5df5d9958f6c56035cdc316ed52c5fa39ba4f65222680f98d50224",
        ),
    ),
)
SENSITIVITY_EXCLUDED_COLUMNS = frozenset(
    {
        "run_id",
        "matched_baseline_run_id",
        "method_run_id",
        "baseline_run_id",
        "requested_scope",
    }
)
BASELINE_GRID_PATH = "configs/baseline_hparams.yaml"
RUN_LEDGER_PATH = "paper_artifact/compute_accounting/compute_ledger_official_runs.csv"
# Run signatures exclude the Chronos-2 revision as a provenance-only key (see
# pipelines/signatures.py). The reference baseline runs loaded the default
# branch of amazon/chronos-2, whose weights and configuration are byte-identical
# to the revision pinned in the grid. The selected configuration reports that
# pinned revision, so the grid must provide it.
PINNED_REVISION_GRID_KEYS = {
    "Chronos2": frozenset({"chronos_model_revision"}),
}

ARCHITECTURE_RESULTS_FILENAME = "training_seed_architecture_results.csv"
METHOD_RESULTS_FILENAME = "training_seed_method_results.csv"
SELECTED_HPARAMS_FILENAME = "selected_clean_validation_hyperparameters.csv"
PATCHTST_DIAGNOSTIC_FILENAME = "patchtst_clean_validation_near_ties.csv"
PATCHTST_SUMMARY_FILENAME = "patchtst_clean_validation_near_tie_summary.csv"

ARCHITECTURE_RESULT_FIELDS = (
    "training_seed",
    "dataset",
    "dataset_label",
    "model_architecture",
    "architecture_label",
    "data_config_signature",
    "eval_data_seed",
    "n_test_samples",
    "selection_metric",
    "best_val_loss",
    "MSE_c",
    "D_w",
    "D_w_CI_lo",
    "D_w_CI_hi",
    "D_mean",
    "D_mean_CI_lo",
    "D_mean_CI_hi",
    "MSE_w",
    "MSE_w_CI_lo",
    "MSE_w_CI_hi",
    "worst_scenario",
    "robustness_scoring_semantics",
)
METHOD_RESULT_FIELDS = (
    "training_seed",
    "dataset",
    "dataset_label",
    "model_architecture",
    "architecture_label",
    "robustness_method",
    "method_label",
    "method_variant_id",
    "data_config_signature",
    "eval_data_seed",
    "n_test_samples",
    "selection_metric",
    "selection_value",
    "MSE_c_method",
    "MSE_c_baseline",
    "delta_MSE_c",
    "pct_change_MSE_c",
    "D_w_method",
    "D_w_method_CI_lo",
    "D_w_method_CI_hi",
    "D_w_baseline",
    "delta_D_w",
    "pct_change_D_w",
    "D_mean_method",
    "D_mean_method_CI_lo",
    "D_mean_method_CI_hi",
    "D_mean_baseline",
    "delta_D_mean",
    "pct_change_D_mean",
    "MSE_w_method",
    "MSE_w_method_CI_lo",
    "MSE_w_method_CI_hi",
    "MSE_w_baseline",
    "delta_MSE_w",
    "pct_change_MSE_w",
    "worst_scenario_method",
    "worst_scenario_baseline",
    "robustness_scoring_semantics",
)
PATCHTST_RESULT_FIELDS = (
    "training_seed",
    "dataset",
    "dataset_label",
    "model_architecture",
    "architecture_label",
    "rank_clean_validation",
    "best_val_loss",
    "best_val_loss_relative_gap",
    "is_canonical_winner",
    "learning_rate",
    "d_model",
    "d_ff",
    "n_layers_enc",
    "n_heads",
    "patch_len",
    "stride",
    "dropout",
    "factor",
    "activation",
    "MSE_c",
    "D_w",
    "D_mean",
    "MSE_w",
    "worst_scenario",
    "worst_scenario_family",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _staged_source(source_dir: Path, source: FrozenSource) -> Path:
    path = source_dir / source.filename
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing staged source {source.source_id}: expected {source.filename} "
            f"in {source_dir}."
        )
    digest = _sha256(path)
    if digest != source.sha256:
        raise ValueError(
            f"Staged source {source.source_id} has SHA-256 {digest}, "
            f"expected {source.sha256}."
        )
    return path


def _read_archive_csv(
    archive: zipfile.ZipFile,
    member: str,
) -> tuple[list[dict[str, str]], str]:
    payload = archive.read(member)
    rows = [
        {key: value for key, value in row.items() if key}
        for row in csv.DictReader(io.StringIO(payload.decode("utf-8")))
        if row.get("dataset") in DATASETS
    ]
    return rows, _sha256_bytes(payload)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty supplementary table: {path.name}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_sensitivity_table(source_path: Path, output_path: Path) -> int:
    with source_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Sensitivity evidence {source_path.name} has no header.")
        fields = tuple(
            name
            for name in reader.fieldnames
            if name not in SENSITIVITY_EXCLUDED_COLUMNS
        )
        rows = [{field: row[field] for field in fields} for row in reader]
    _write_csv(output_path, rows, fields)
    return len(rows)


def _close(left: str | float, right: str | float) -> bool:
    return math.isclose(
        float(left),
        float(right),
        rel_tol=1e-10,
        abs_tol=1e-12,
    )


def _load_baseline_grid(path: Path) -> dict[str, dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a mapping.")
    for architecture, grid in payload.items():
        if not isinstance(architecture, str) or not isinstance(grid, dict):
            raise ValueError(f"Malformed architecture grid in {path}.")
    return payload


def _load_run_signatures(path: Path) -> dict[str, str]:
    signatures: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            run_id = row["run_id"]
            signature = row["signature"]
            previous = signatures.setdefault(run_id, signature)
            if previous != signature:
                raise ValueError(f"Run {run_id} has inconsistent recorded signatures.")
    return signatures


def _recover_selected_hparams(
    *,
    baseline_rows: list[dict[str, str]],
    grids: dict[str, dict[str, Any]],
    run_signatures: dict[str, str],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in sorted(
        baseline_rows,
        key=lambda item: (
            DATASETS.index(item["dataset"]),
            ARCHITECTURES.index(item["model_architecture"]),
        ),
    ):
        architecture = row["model_architecture"]
        grid = grids.get(architecture)
        if grid is None:
            raise ValueError(f"Missing hyperparameter grid for {architecture}.")
        target_signature = run_signatures.get(row["run_id"])
        if target_signature is None:
            raise ValueError(
                "The canonical baseline winner is absent from the recorded run ledger: "
                f"{row['dataset']} / {architecture}."
            )

        pinned_keys = PINNED_REVISION_GRID_KEYS.get(architecture, frozenset())
        if not pinned_keys <= set(grid):
            raise ValueError(
                f"The {architecture} grid lacks the pinned keys "
                f"{sorted(pinned_keys - set(grid))}."
            )
        keys = tuple(grid)
        choices = tuple(
            value if isinstance(value, list) else [value]
            for value in grid.values()
        )
        matches: list[dict[str, Any]] = []
        for values in itertools.product(*choices):
            hparams = dict(zip(keys, values))
            hparams["loss"] = "MSE"
            signature = build_signature(
                architecture,
                row["dataset"],
                hparams,
                pipeline_id="baseline",
                data_config_signature=row["data_config_signature"],
            )
            if signature == target_signature:
                matches.append(hparams)
        if len(matches) != 1:
            raise ValueError(
                "Expected one signature-matched clean-validation configuration for "
                f"{row['dataset']} / {architecture}, found {len(matches)}."
            )
        output.append(
            {
                "training_seed": 42,
                "dataset": row["dataset"],
                "dataset_label": DATASET_LABELS[row["dataset"]],
                "model_architecture": architecture,
                "architecture_label": ARCHITECTURE_LABELS[architecture],
                "selection_metric": "clean_validation_MSE",
                "best_val_loss": row["best_val_loss"],
                "hyperparameters_json": json.dumps(
                    matches[0],
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        )
    if len(output) != len(DATASETS) * len(ARCHITECTURES):
        raise ValueError(f"Expected 28 selected configurations, found {len(output)}.")
    return output


def _architecture_result(seed: int, row: dict[str, str]) -> dict[str, Any]:
    architecture = row["model_architecture"]
    return {
        "training_seed": seed,
        "dataset": row["dataset"],
        "dataset_label": DATASET_LABELS[row["dataset"]],
        "model_architecture": architecture,
        "architecture_label": ARCHITECTURE_LABELS[architecture],
        "data_config_signature": row["data_config_signature"],
        "eval_data_seed": row["eval_data_seed"],
        "n_test_samples": row["n_test_samples"],
        "selection_metric": "clean_validation_MSE",
        "best_val_loss": row["best_val_loss"],
        "MSE_c": row["MSE_test"],
        "D_w": row["D_w"],
        "D_w_CI_lo": row["D_w_CI_lo"],
        "D_w_CI_hi": row["D_w_CI_hi"],
        "D_mean": row["D_mean"],
        "D_mean_CI_lo": row["D_mean_CI_lo"],
        "D_mean_CI_hi": row["D_mean_CI_hi"],
        "MSE_w": row["err_pert_ws"],
        "MSE_w_CI_lo": row["err_pert_ws_CI_lo"],
        "MSE_w_CI_hi": row["err_pert_ws_CI_hi"],
        "worst_scenario": row["worst_scenario"],
        "robustness_scoring_semantics": row["robustness_scoring_semantics"],
    }


def _method_result(seed: int, row: dict[str, str]) -> dict[str, Any]:
    architecture = row["model_architecture"]
    method = row["pipeline_method"]
    return {
        "training_seed": seed,
        "dataset": row["dataset"],
        "dataset_label": DATASET_LABELS[row["dataset"]],
        "model_architecture": architecture,
        "architecture_label": ARCHITECTURE_LABELS[architecture],
        "robustness_method": method,
        "method_label": METHOD_LABELS[method],
        "method_variant_id": row["pipeline_id"],
        "data_config_signature": row["data_config_signature"],
        "eval_data_seed": row["eval_data_seed"],
        "n_test_samples": row["n_test_samples"],
        "selection_metric": row["selection_metric"],
        "selection_value": row["selection_value"],
        "MSE_c_method": row["MSE_test_improved"],
        "MSE_c_baseline": row["MSE_test_baseline"],
        "delta_MSE_c": row["delta_MSE_test"],
        "pct_change_MSE_c": row["pct_change_MSE_test"],
        "D_w_method": row["D_w_improved"],
        "D_w_method_CI_lo": row["D_w_CI_lo"],
        "D_w_method_CI_hi": row["D_w_CI_hi"],
        "D_w_baseline": row["D_w_baseline"],
        "delta_D_w": row["delta_D_w"],
        "pct_change_D_w": row["pct_change_D_w"],
        "D_mean_method": row["D_mean_improved"],
        "D_mean_method_CI_lo": row["D_mean_CI_lo"],
        "D_mean_method_CI_hi": row["D_mean_CI_hi"],
        "D_mean_baseline": row["D_mean_baseline"],
        "delta_D_mean": row["delta_D_mean"],
        "pct_change_D_mean": row["pct_change_D_mean"],
        "MSE_w_method": row["err_pert_ws_improved"],
        "MSE_w_method_CI_lo": row["err_pert_ws_CI_lo"],
        "MSE_w_method_CI_hi": row["err_pert_ws_CI_hi"],
        "MSE_w_baseline": row["err_pert_ws_baseline"],
        "delta_MSE_w": row["delta_err_pert_ws"],
        "pct_change_MSE_w": row["pct_change_err_pert_ws"],
        "worst_scenario_method": row["worst_scenario_improved"],
        "worst_scenario_baseline": row["worst_scenario_baseline"],
        "robustness_scoring_semantics": row["robustness_scoring_semantics"],
    }


def _verify_seed_source(
    *,
    seed: int,
    full_rows: list[dict[str, str]],
    delta_rows: list[dict[str, str]],
    scenario_rows: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(full_rows) != 164 or len(delta_rows) != 136:
        raise ValueError(
            f"Seed {seed} has unexpected result counts: "
            f"{len(full_rows)} full rows and {len(delta_rows)} method pairs."
        )
    if len(scenario_rows) != len(full_rows) * len(SCENARIOS):
        raise ValueError(f"Seed {seed} has incomplete scenario rows.")

    row_index = {
        (row["dataset"], row["data_config_signature"], row["run_id"]): row
        for row in full_rows
    }
    if len(row_index) != len(full_rows):
        raise ValueError(f"Seed {seed} has duplicate selected-result identities.")

    scenario_index: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in scenario_rows:
        key = (row["dataset"], row["data_config_signature"], row["run_id"])
        scenario_index.setdefault(key, []).append(row)

    baselines: dict[tuple[str, str, str], dict[str, str]] = {}
    architecture_results: list[dict[str, Any]] = []
    for row in full_rows:
        if row["pipeline_method"] != row["robustness_method"]:
            raise ValueError(f"Seed {seed} has inconsistent method labels.")
        if row["model_architecture"] not in ARCHITECTURES:
            raise ValueError(f"Seed {seed} has an unexpected architecture.")
        if row["n_test_samples"] != "10000":
            raise ValueError(f"Seed {seed} does not use 10,000 test windows.")
        if row["robustness_scoring_semantics"] != "uniform_severity_degradation":
            raise ValueError(f"Seed {seed} has unexpected scoring semantics.")
        if row["perturbation_scenarios_count"] != "8":
            raise ValueError(f"Seed {seed} does not contain eight scenarios.")

        result_key = (row["dataset"], row["data_config_signature"], row["run_id"])
        scenarios = scenario_index.get(result_key, [])
        if len(scenarios) != len(SCENARIOS):
            raise ValueError(f"Seed {seed} has incomplete scenarios for {result_key}.")
        if {item["scenario"] for item in scenarios} != SCENARIOS:
            raise ValueError(f"Seed {seed} has an unexpected scenario set.")
        if not _close(row["D_w"], max(float(item["D"]) for item in scenarios)):
            raise ValueError(f"Seed {seed} has an inconsistent D_w maximum.")
        if not _close(
            row["err_pert_ws"],
            max(float(item["err_pert"]) for item in scenarios),
        ):
            raise ValueError(f"Seed {seed} has an inconsistent MSE_w maximum.")

        if row["pipeline_method"] == "baseline":
            if row["pipeline_id"] != "baseline" or row["pipeline_kind"] != "train":
                raise ValueError(f"Seed {seed} has malformed baseline identity.")
            baseline_key = (
                row["dataset"],
                row["data_config_signature"],
                row["model_architecture"],
            )
            if baseline_key in baselines:
                raise ValueError(f"Seed {seed} has duplicate baseline winners.")
            baselines[baseline_key] = row
            architecture_results.append(_architecture_result(seed, row))
        elif row["pipeline_method"] not in METHODS:
            raise ValueError(f"Seed {seed} has an unexpected robustness method.")

    if len(baselines) != len(DATASETS) * len(ARCHITECTURES):
        raise ValueError(f"Seed {seed} does not contain 28 baseline winners.")

    method_results: list[dict[str, Any]] = []
    context_fields = (
        "data_config_signature",
        "eval_data_seed",
        "test_metric",
        "n_test_samples",
        "robustness_scoring_semantics",
        "perturbation_channel_fraction_max",
        "perturbation_scenarios_signature",
    )
    quantity_fields = ("MSE_test", "D_w", "err_pert_ws")
    for row in delta_rows:
        if row["pipeline_method"] not in METHODS:
            raise ValueError(f"Seed {seed} has an unexpected method pair.")
        baseline_key = (
            row["dataset"],
            row["data_config_signature"],
            row["model_architecture"],
        )
        baseline = baselines[baseline_key]
        improved = row_index[
            (row["dataset"], row["data_config_signature"], row["run_id"])
        ]
        if row["matched_baseline_run_id"] != baseline["run_id"]:
            raise ValueError(f"Seed {seed} has a mismatched paired baseline.")
        for field in context_fields:
            if row[field] != improved[field] or row[field] != baseline[field]:
                raise ValueError(f"Seed {seed} has inconsistent {field} context.")
        for field in quantity_fields:
            if not _close(row[f"{field}_improved"], improved[field]):
                raise ValueError(f"Seed {seed} has an inconsistent method result.")
            if not _close(row[f"{field}_baseline"], baseline[field]):
                raise ValueError(f"Seed {seed} has an inconsistent baseline result.")
            expected_delta = float(improved[field]) - float(baseline[field])
            if not _close(row[f"delta_{field}"], expected_delta):
                raise ValueError(f"Seed {seed} has an inconsistent paired delta.")
        method_results.append(_method_result(seed, row))

    for dataset in DATASETS:
        for method in METHODS:
            rows = [
                row
                for row in method_results
                if row["dataset"] == dataset and row["robustness_method"] == method
            ]
            expected = LEARNED_ARCHITECTURES - (
                {"PatchTST"} if method == "revin" else set()
            )
            if {row["model_architecture"] for row in rows} != expected:
                raise ValueError(
                    f"Seed {seed} has incomplete {method} coverage on {dataset}."
                )
    return architecture_results, method_results


def _patchtst_diagnostic(source_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    with source_path.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    selected = [
        row
        for row in source_rows
        if row["scored"] == "True" and float(row["best_val_loss_rel_gap"]) <= 0.03
    ]
    if len(selected) != 14:
        raise ValueError(f"Expected 14 PatchTST near-tie rows, found {len(selected)}.")

    output: list[dict[str, Any]] = []
    for row in sorted(selected, key=lambda item: int(item["rank_clean_val"])):
        output.append(
            {
                "training_seed": int(row["seed_master"]),
                "dataset": row["dataset"],
                "dataset_label": DATASET_LABELS[row["dataset"]],
                "model_architecture": row["model_architecture"],
                "architecture_label": ARCHITECTURE_LABELS[row["model_architecture"]],
                "rank_clean_validation": row["rank_clean_val"],
                "best_val_loss": row["best_val_loss"],
                "best_val_loss_relative_gap": row["best_val_loss_rel_gap"],
                "is_canonical_winner": row["is_canonical_winner"],
                "learning_rate": row["learning_rate"],
                "d_model": row["d_model"],
                "d_ff": row["d_ff"],
                "n_layers_enc": row["n_layers_enc"],
                "n_heads": row["n_heads"],
                "patch_len": row["patch_len"],
                "stride": row["stride"],
                "dropout": row["dropout"],
                "factor": row["factor"],
                "activation": row["activation"],
                "MSE_c": row["MSE_c"],
                "D_w": row["D_w"],
                "D_mean": row["D_mean"],
                "MSE_w": row["MSE_w"],
                "worst_scenario": row["worst_scenario_name"],
                "worst_scenario_family": row["worst_scenario_family"],
            }
        )

    summaries: list[dict[str, Any]] = []
    for fragile_branch in (True, False):
        group = [
            row
            for row in output
            if (
                int(row["d_model"]) == 128
                and float(row["learning_rate"]) == 0.001
            )
            == fragile_branch
        ]
        expected_scenario = "spike" if fragile_branch else "noise"
        expected_count = 4 if fragile_branch else 10
        if len(group) != expected_count:
            raise ValueError("PatchTST diagnostic branch count changed.")
        if {row["worst_scenario"] for row in group} != {expected_scenario}:
            raise ValueError("PatchTST diagnostic branch scenario changed.")
        summaries.append(
            {
                "branch": (
                    "d_model_128_and_learning_rate_0.001"
                    if fragile_branch
                    else "other_near_tied_configurations"
                ),
                "n_configurations": len(group),
                "mean_MSE_c": mean(float(row["MSE_c"]) for row in group),
                "mean_D_w": mean(float(row["D_w"]) for row in group),
                "mean_MSE_w": mean(float(row["MSE_w"]) for row in group),
                "worst_scenario": expected_scenario,
            }
        )
    if [round(row["mean_D_w"], 3) for row in summaries] != [2.388, 1.332]:
        raise ValueError("PatchTST diagnostic degradation means changed.")
    if [round(row["mean_MSE_c"], 3) for row in summaries] != [0.340, 0.336]:
        raise ValueError("PatchTST diagnostic clean-MSE means changed.")
    return output, summaries


def _write_readme(output_root: Path) -> Path:
    path = output_root / "README.md"
    path.write_text(
        """# Supplementary Tables

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
""",
        encoding="utf-8",
    )
    return path


def _output_record(
    *,
    output_root: Path,
    path: Path,
    role: str,
    row_count: int | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": path.relative_to(output_root).as_posix(),
        "role": role,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if row_count is not None:
        record["row_count"] = row_count
    return record


def build_supplementary_tables(
    *,
    source_dir: Path,
    repo_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Build the supplementary tables and return their manifest.

    ``source_dir`` holds the frozen inputs under their neutral file names. Each
    staged file must match its pinned SHA-256 checksum. The search grid and the
    run-signature ledger are read from the shipped files under ``repo_root``.
    """
    source_dir = source_dir.resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Staged source directory does not exist: {source_dir}")
    output_root.mkdir(parents=True, exist_ok=True)
    source_records: list[dict[str, Any]] = []
    architecture_results: list[dict[str, Any]] = []
    method_results: list[dict[str, Any]] = []
    seed_42_baselines: list[dict[str, str]] | None = None

    members = (
        "artifacts/tables/full_results.csv",
        "artifacts/tables/improvement_deltas_selected.csv",
        "artifacts/tables/scenario_summary.csv",
    )
    for source in TRAINING_SEED_SOURCES:
        archive_path = _staged_source(source_dir, source.source)
        member_hashes: dict[str, str] = {}
        with zipfile.ZipFile(archive_path) as archive:
            full_rows, member_hashes[members[0]] = _read_archive_csv(archive, members[0])
            delta_rows, member_hashes[members[1]] = _read_archive_csv(archive, members[1])
            scenario_rows, member_hashes[members[2]] = _read_archive_csv(archive, members[2])
        architecture_seed_rows, method_seed_rows = _verify_seed_source(
            seed=source.seed,
            full_rows=full_rows,
            delta_rows=delta_rows,
            scenario_rows=scenario_rows,
        )
        architecture_results.extend(architecture_seed_rows)
        method_results.extend(method_seed_rows)
        if source.seed == 42:
            seed_42_baselines = [
                row for row in full_rows if row["pipeline_method"] == "baseline"
            ]
        source_records.append(
            {
                "source_id": source.source.source_id,
                "role": f"training and search seed {source.seed} result source",
                "sha256": source.source.sha256,
                "members": [
                    {"path": member, "sha256": member_hashes[member]}
                    for member in members
                ],
            }
        )

    if len(architecture_results) != 84 or len(method_results) != 408:
        raise ValueError("Training-seed release tables have unexpected total row counts.")
    if seed_42_baselines is None:
        raise ValueError("The canonical seed-42 baseline source was not loaded.")

    grid_path = repo_root / BASELINE_GRID_PATH
    ledger_path = repo_root / RUN_LEDGER_PATH
    selected_hparams = _recover_selected_hparams(
        baseline_rows=seed_42_baselines,
        grids=_load_baseline_grid(grid_path),
        run_signatures=_load_run_signatures(ledger_path),
    )
    source_records.extend(
        [
            {
                "role": "baseline hyperparameter search grid",
                "path": BASELINE_GRID_PATH,
                "sha256": _sha256(grid_path),
            },
            {
                "role": "canonical baseline run-signature ledger",
                "path": RUN_LEDGER_PATH,
                "sha256": _sha256(ledger_path),
            },
        ]
    )

    patchtst_path = _staged_source(source_dir, PATCHTST_DIAGNOSTIC_SOURCE)
    patchtst_rows, patchtst_summary = _patchtst_diagnostic(patchtst_path)
    source_records.append(
        {
            "source_id": PATCHTST_DIAGNOSTIC_SOURCE.source_id,
            "role": "PatchTST clean-validation near-tie diagnostic",
            "sha256": PATCHTST_DIAGNOSTIC_SOURCE.sha256,
        }
    )

    architecture_path = output_root / ARCHITECTURE_RESULTS_FILENAME
    method_path = output_root / METHOD_RESULTS_FILENAME
    hparams_path = output_root / SELECTED_HPARAMS_FILENAME
    patchtst_output_path = output_root / PATCHTST_DIAGNOSTIC_FILENAME
    patchtst_summary_path = output_root / PATCHTST_SUMMARY_FILENAME
    _write_csv(architecture_path, architecture_results, ARCHITECTURE_RESULT_FIELDS)
    _write_csv(method_path, method_results, METHOD_RESULT_FIELDS)
    _write_csv(
        hparams_path,
        selected_hparams,
        (
            "training_seed",
            "dataset",
            "dataset_label",
            "model_architecture",
            "architecture_label",
            "selection_metric",
            "best_val_loss",
            "hyperparameters_json",
        ),
    )
    _write_csv(patchtst_output_path, patchtst_rows, PATCHTST_RESULT_FIELDS)
    _write_csv(
        patchtst_summary_path,
        patchtst_summary,
        (
            "branch",
            "n_configurations",
            "mean_MSE_c",
            "mean_D_w",
            "mean_MSE_w",
            "worst_scenario",
        ),
    )
    sensitivity_records: list[dict[str, Any]] = []
    for table in SENSITIVITY_TABLES:
        output_path = output_root / table.filename
        row_count = _write_sensitivity_table(
            _staged_source(source_dir, table.source),
            output_path,
        )
        source_records.append(
            {
                "source_id": table.source.source_id,
                "role": f"verified evidence table: {table.role}",
                "sha256": table.source.sha256,
            }
        )
        sensitivity_records.append(
            _output_record(
                output_root=output_root,
                path=output_path,
                role=table.role,
                row_count=row_count,
            )
        )
    readme_path = _write_readme(output_root)

    output_records = [
        _output_record(
            output_root=output_root,
            path=hparams_path,
            role="selected clean-validation hyperparameters",
            row_count=len(selected_hparams),
        ),
        _output_record(
            output_root=output_root,
            path=architecture_path,
            role="complete three-training-seed architecture results",
            row_count=len(architecture_results),
        ),
        _output_record(
            output_root=output_root,
            path=method_path,
            role="complete three-training-seed method-baseline results",
            row_count=len(method_results),
        ),
        _output_record(
            output_root=output_root,
            path=patchtst_output_path,
            role="PatchTST clean-validation near-tie configurations",
            row_count=len(patchtst_rows),
        ),
        _output_record(
            output_root=output_root,
            path=patchtst_summary_path,
            role="PatchTST clean-validation near-tie branch summary",
            row_count=len(patchtst_summary),
        ),
        *sensitivity_records,
        _output_record(
            output_root=output_root,
            path=readme_path,
            role="supplementary-table documentation",
        ),
    ]
    manifest = {
        "description": (
            "Supplementary tables of selected clean-validation hyperparameters, "
            "training-seed sensitivity, method-baseline effects, the PatchTST "
            "near-tie diagnostic, and the benchmark-design sensitivity checks."
        ),
        "training_seeds": [source.seed for source in TRAINING_SEED_SOURCES],
        "datasets": list(DATASETS),
        "architectures": list(ARCHITECTURES),
        "robustness_methods": list(METHODS),
        "selection_metric": "clean_validation_MSE",
        "evaluation_windows_per_scenario": 10000,
        "sources": source_records,
        "files": output_records,
    }
    manifest_path = output_root / SUPPLEMENTARY_TABLES_MANIFEST
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


__all__ = [
    "ARCHITECTURES",
    "ARCHITECTURE_RESULTS_FILENAME",
    "BASELINE_GRID_PATH",
    "DATASETS",
    "FrozenSource",
    "METHODS",
    "METHOD_RESULTS_FILENAME",
    "PATCHTST_DIAGNOSTIC_FILENAME",
    "PATCHTST_DIAGNOSTIC_SOURCE",
    "PATCHTST_SUMMARY_FILENAME",
    "RUN_LEDGER_PATH",
    "SENSITIVITY_EXCLUDED_COLUMNS",
    "SENSITIVITY_TABLES",
    "SELECTED_HPARAMS_FILENAME",
    "SUPPLEMENTARY_TABLES_DIRNAME",
    "SUPPLEMENTARY_TABLES_MANIFEST",
    "TRAINING_SEED_SOURCES",
    "build_supplementary_tables",
]
