"""Checks the smoke and full benchmark commands that README.md and the reproduction guide document."""

from __future__ import annotations

import shlex
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


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


def test_reproduction_guide_points_to_the_single_smoke_recipe():
    guide = _read("docs/reproducing_benchmark.md")

    assert "[quick start](../README.md#quick-start)" in guide
    assert "--n-train-samples" not in guide


def test_documented_full_benchmark_commands_force_strict_coverage():
    commands = _commands_from_block(
        _bash_block_after(
            "README.md",
            "## Reproducing the paper",
        )
    )
    testing = _command_for(commands, "run_testing.py")
    analysis = _command_for(commands, "run_analysis.py")

    assert "--full-coverage" in testing
    assert "--full-coverage" in analysis
