"""Checks that the commands in docs/sensitivity_analyses.md match the bundled results.

The identity checks rebuild the documented arguments through the testing entry
point and compare the data configuration signature and the evaluation-data seed
with the bundled result rows. No data file and no tracking store is needed.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shlex
from pathlib import Path

import pandas as pd
import pytest

from config_loader import load_defaults
from data.datasets import resolve_with_defaults
from data.perturbations import PERTURBATION_REGISTRY
from pipelines.signatures import compute_data_config_signature
from run_analysis import prepare_analysis_entrypoint_args
from run_testing import prepare_testing_entrypoint_args
from utils.parsing import parse_perturbation_severity_distribution, resolve_dataset_window_args
from utils.rng import derive_seed


REPO_ROOT = Path(__file__).resolve().parents[1]
DOC_PATH = REPO_ROOT / "docs" / "sensitivity_analyses.md"
EVIDENCE_ROOT = REPO_ROOT / "paper_artifact" / "supplementary_tables"


def _documented_commands(script: str) -> list[list[str]]:
    prefix = f"uv run python {script} "
    return [
        shlex.split(line.strip()[len(prefix):])
        for line in DOC_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith(prefix)
    ]


def _flag_values(cli_args: list[str], flag: str) -> list[str]:
    index = cli_args.index(flag)
    values: list[str] = []
    for token in cli_args[index + 1:]:
        if token.startswith("--"):
            break
        values.append(token)
    return values


def _identity(cli_args: list[str]) -> tuple[str, str, int]:
    args, _recipe_specs, window_defaults = prepare_testing_entrypoint_args(cli_args)
    (dataset_spec,) = resolve_with_defaults(
        datasets=args.data_files,
        targets=args.data_targets,
        data_root=args.data_root,
    )
    dataset_args = resolve_dataset_window_args(
        args,
        dataset_spec=dataset_spec,
        dataset_window_defaults=window_defaults,
        explicit_arg_overrides=args._window_arg_overrides,
    )
    signature = compute_data_config_signature(dataset_spec=dataset_spec, args=dataset_args)
    data_base_seed = args.data_split_seed if args.data_split_seed is not None else args.seed
    data_seed = derive_seed(int(data_base_seed), f"data:{dataset_spec.key}:{signature}")
    return dataset_spec.key, signature, data_seed


def _read_rows(filename: str) -> list[dict[str, str]]:
    with (EVIDENCE_ROOT / filename).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_training_seed_commands_reproduce_the_bundled_identities():
    commands = [
        command
        for command in _documented_commands("run_testing.py")
        if "--seed" in command
    ]
    bundled = {
        (row["dataset"], int(row["training_seed"])): row
        for row in _read_rows("training_seed_architecture_results.csv")
    }

    assert len(commands) == 4
    for command in commands:
        dataset, signature, data_seed = _identity(command)
        seed = int(_flag_values(command, "--seed")[0])
        row = bundled[(dataset, seed)]
        assert signature == row["data_config_signature"]
        assert data_seed == int(row["eval_data_seed"])
        assert int(_flag_values(command, "--eval-data-seed")[0]) == data_seed


@pytest.mark.parametrize(
    ("target_scope", "horizon"),
    [("all", 96), ("all", 6), ("ot", 96), ("ot", 6)],
)
def test_etth1_task_cells_reproduce_the_bundled_identities(target_scope, horizon):
    cell = f"{target_scope}_h{horizon}"
    rows = [row for row in _read_rows("etth1_task_architecture.csv") if row["cell"] == cell]
    command = [
        "--data-files", "ETTh1",
        "--data-targets", target_scope,
        "--input-len", "96",
        "--target-len", str(horizon),
    ]

    dataset, signature, data_seed = _identity(command)

    assert dataset == "ETTh1"
    assert {row["data_config_signature"] for row in rows} == {signature}
    assert {int(row["eval_data_seed"]) for row in rows} == {data_seed}


def test_documented_etth1_example_uses_the_cell_flags():
    for script in ("run_training.py", "run_testing.py", "run_analysis.py"):
        (command,) = [
            command
            for command in _documented_commands(script)
            if "etth1-ot-h6" in command
        ]
        assert _flag_values(command, "--data-targets") == ["ot"]
        assert _flag_values(command, "--input-len") == ["96"]
        assert _flag_values(command, "--target-len") == ["6"]


def test_documented_scenario_substitutions_keep_the_scored_positions():
    scored = list(load_defaults()["PERTURBATION_SCENARIOS"])
    substitutions = {
        "negative_drift": "drift",
        "negative_spike": "spike",
        "local_timing_warp": "time_compress",
    }
    scenario_lists = [
        _flag_values(command, "--perturbation-scenarios")
        for script in ("run_testing.py", "run_analysis.py")
        for command in _documented_commands(script)
        if "--perturbation-scenarios" in command
    ]

    assert len(scenario_lists) == 4
    for scenarios in scenario_lists:
        assert len(scenarios) == len(scored)
        assert all(name in PERTURBATION_REGISTRY for name in scenarios)
        for position, (name, reference) in enumerate(zip(scenarios, scored)):
            assert name == reference or substitutions[name] == scored[position]


def test_documented_severity_distribution_is_supported():
    distributions = {
        _flag_values(command, "--perturbation-severity-distribution")[0]
        for script in ("run_testing.py", "run_analysis.py")
        for command in _documented_commands(script)
        if "--perturbation-severity-distribution" in command
    }

    assert distributions == {"beta_1_2"}
    assert parse_perturbation_severity_distribution("beta_1_2") == "beta_1_2"


def test_time_compress_omission_snippet_reproduces_the_bundled_seed_42_rows():
    text = DOC_PATH.read_text(encoding="utf-8")
    block = re.search(
        r"<!-- time-compress-omission:start -->\n```python\n(.*?)```\n"
        r"<!-- time-compress-omission:end -->",
        text,
        re.S,
    )
    assert block is not None
    namespace: dict[str, object] = {}
    previous_cwd = Path.cwd()
    os.chdir(REPO_ROOT)
    try:
        exec(block.group(1), namespace)
    finally:
        os.chdir(previous_cwd)
    recomputed = namespace["without_time_compress"]

    bundled = pd.read_csv(EVIDENCE_ROOT / "timing_time_compress_omission_architecture.csv")
    bundled = bundled[bundled["training_seed"] == 42]
    merged = recomputed.merge(
        bundled,
        on=["dataset", "model_architecture"],
        suffixes=("_recomputed", "_bundled"),
    )

    assert len(merged) == len(bundled) == 28
    for quantity in ("D_w_without_time_compress", "MSE_w_without_time_compress"):
        assert (merged[f"{quantity}_recomputed"] == merged[f"{quantity}_bundled"]).all()


DESIGN_CHECK_FLAGS = ("--eval-data-seed", "--improvement-selection-mode", "--fixed-channel-fraction")


def _evaluation_setting(args) -> tuple[str | None, str, float | None]:
    return (
        None if args.eval_data_seed is None else str(args.eval_data_seed),
        args.improvement_selection_mode,
        args.fixed_channel_fraction,
    )


def test_design_check_commands_match_the_bundled_evaluation_contexts():
    contexts = json.loads(
        (REPO_ROOT / "paper_artifact" / "config" / "eval_context.json").read_text(encoding="utf-8")
    )["contexts"]
    bundled = {
        (
            context["eval_data_seed"],
            context["improvement_selection_mode"],
            context.get("fixed_channel_fraction"),
        ): source_id
        for source_id, context in contexts.items()
        if source_id != "default_configuration"
    }
    commands = [
        command
        for command in _documented_commands("run_testing.py")
        if "--seed" not in command and any(flag in command for flag in DESIGN_CHECK_FLAGS)
    ]

    settings = [
        _evaluation_setting(prepare_testing_entrypoint_args(command)[0])
        for command in commands
    ]

    assert sorted(bundled[setting] for setting in settings) == sorted(bundled.values())
    for command in _documented_commands("run_analysis.py"):
        if "--seed" not in command and any(flag in command for flag in DESIGN_CHECK_FLAGS):
            assert _evaluation_setting(prepare_analysis_entrypoint_args(command)) in bundled
