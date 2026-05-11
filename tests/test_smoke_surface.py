from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_SHARED_OPTIONS = {
    "--data-files": "ETTh1",
    "--max-hp-trials-per-model": "1",
    "--n-train-samples": "8",
    "--n-val-samples": "4",
    "--n-test-samples": "4",
    "--input-len": "12",
    "--target-len": "3",
    "--mlflow-experiment-prefix": "benchmark-smoke",
    "--logdir": "runs/smoke",
}
TRAINING_AND_TESTING_SCOPE_OPTIONS = {
    "--model": "DLinear",
    "--method": "baseline",
    "--batch-size": "4",
    "--num-workers": "0",
    "--accelerator": "cpu",
    "--devices": "1",
}


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def _bash_block_after(path: str, marker: str) -> str:
    text = _read(path)
    marker_index = text.index(marker)
    block_start = text.index("```bash", marker_index) + len("```bash")
    block_end = text.index("```", block_start)
    return text[block_start:block_end].strip()


def _commands_from_block(block: str) -> list[list[str]]:
    commands: list[str] = []
    current = ""
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.endswith("\\"):
            current += line[:-1].strip() + " "
            continue
        commands.append((current + line).strip())
        current = ""
    if current:
        raise AssertionError(f"Unterminated shell continuation: {current}")
    return [shlex.split(command) for command in commands]


def _command_for(commands: list[list[str]], entrypoint: str) -> list[str]:
    for command in commands:
        if entrypoint in command:
            return command
    raise AssertionError(f"Missing smoke command for {entrypoint}.")


def _assert_option(tokens: list[str], option: str, expected_value: str) -> None:
    assert option in tokens
    index = tokens.index(option)
    assert index + 1 < len(tokens), f"{option} is missing a value."
    assert tokens[index + 1] == expected_value


@pytest.mark.parametrize("entrypoint", ("run_training.py", "run_testing.py", "run_analysis.py"))
def test_benchmark_entrypoint_help_commands_exit_successfully(entrypoint: str):
    result = subprocess.run(
        [sys.executable, entrypoint, "--help"],
        cwd=REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert f"usage: {entrypoint}" in result.stdout
    assert "--data-root" in result.stdout
    assert "--logdir" in result.stdout
    assert "--minio-endpoint" in result.stdout
    if entrypoint == "run_analysis.py":
        for training_flag in (
            "--rerun",
            "--max-epochs",
            "--save-checkpoint",
        ):
            assert training_flag in result.stdout


@pytest.mark.parametrize(
    ("path", "marker"),
    (
        ("README.md", "Install dependencies and run a small local smoke:"),
        ("docs/reproducing_benchmark.md", "For partial local smoke checks"),
    ),
)
def test_documented_smoke_commands_are_bounded_and_pair_entrypoints(
    path: str,
    marker: str,
):
    commands = _commands_from_block(_bash_block_after(path, marker))
    training = _command_for(commands, "run_training.py")
    testing = _command_for(commands, "run_testing.py")
    analysis = _command_for(commands, "run_analysis.py")

    if path == "README.md":
        for command in (training, testing, analysis):
            _assert_option(command, "--data-files", "ETTh1")
            _assert_option(command, "--logdir", "runs")
            _assert_option(command, "--model", "DLinear")
            _assert_option(command, "--method", "baseline")
            _assert_option(command, "--max-hp-trials-per-model", "1")
        _assert_option(training, "--max-epochs", "1")
        for command in (testing, analysis):
            assert command[
                command.index("--perturbation-scenarios")
                + 1 : command.index("--perturbation-scenarios")
                + 3
            ] == ["missing_data", "noise"]
            assert "--full-coverage" not in command
            assert "--no-full-coverage" not in command
        return

    for option, value in SMOKE_SHARED_OPTIONS.items():
        _assert_option(training, option, value)
        _assert_option(testing, option, value)
        _assert_option(analysis, option, value)

    for command in (training, testing):
        for option, value in TRAINING_AND_TESTING_SCOPE_OPTIONS.items():
            _assert_option(command, option, value)
        assert "--rerun" in command
    _assert_option(analysis, "--model", "DLinear")
    _assert_option(analysis, "--method", "baseline")
    _assert_option(analysis, "--batch-size", "4")

    _assert_option(training, "--max-epochs", "1")

    for command in (testing, analysis):
        _assert_option(command, "--bootstrap-ci-resamples", "10")
        assert command[command.index("--perturbation-scenarios") + 1 : command.index("--perturbation-scenarios") + 3] == [
            "missing_data",
            "noise",
        ]
        assert "--full-coverage" not in command
        assert "--no-full-coverage" not in command

    assert "--benchmark-architectures" not in analysis
    assert "--benchmark-methods" not in analysis


def test_documented_full_benchmark_commands_force_strict_coverage():
    commands = _commands_from_block(
        _bash_block_after(
            "README.md",
            "Run the full benchmark scope from configured defaults:",
        )
    )
    testing = _command_for(commands, "run_testing.py")
    analysis = _command_for(commands, "run_analysis.py")

    assert "--full-coverage" in testing
    assert "--full-coverage" in analysis
