"""Checks that the figure-CSV generator rebuilds the shipped paper inputs.

The generator reads the bundled tables with the default pandas float parser,
which rounds some values one unit in the last place differently on different
CPU architectures. The rebuilt files therefore match the shipped ones in every
header, text cell, and row, and in every number to RELATIVE_TOLERANCE. The
integrity test pins the shipped bytes themselves.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

from scripts.generate_paper_figure_csvs import (
    BACKBONE_SCENARIO_FILENAME,
    DEFAULT_BENCHMARK_SCOPE_CONFIG,
    DEFAULT_CORE_FIGURES_CONFIG,
    MAIN_TABLE_FILENAME,
    PGD_TRAJECTORY_FILENAME,
    generate_paper_figure_csvs,
)


# Far above the parser rounding of about 1e-16 and far below any change of the
# generated rows, columns, or formulas.
RELATIVE_TOLERANCE = 1e-12
ABSOLUTE_TOLERANCE = 1e-15


def _as_float(cell: str) -> float | None:
    try:
        return float(cell)
    except ValueError:
        return None


def _assert_csv_matches(rebuilt_path: Path, shipped_path: Path) -> None:
    with rebuilt_path.open(newline="", encoding="utf-8") as handle:
        rebuilt = list(csv.reader(handle))
    with shipped_path.open(newline="", encoding="utf-8") as handle:
        shipped = list(csv.reader(handle))

    assert rebuilt[0] == shipped[0], shipped_path.name
    assert len(rebuilt) == len(shipped), shipped_path.name
    for row_index, (rebuilt_row, shipped_row) in enumerate(zip(rebuilt, shipped)):
        assert len(rebuilt_row) == len(shipped_row), (shipped_path.name, row_index)
        for column, rebuilt_cell, shipped_cell in zip(shipped[0], rebuilt_row, shipped_row):
            if rebuilt_cell == shipped_cell:
                continue
            location = f"{shipped_path.name} row {row_index} column {column}"
            rebuilt_value = _as_float(rebuilt_cell)
            shipped_value = _as_float(shipped_cell)
            assert rebuilt_value is not None and shipped_value is not None, (
                f"{location}: {rebuilt_cell!r} != {shipped_cell!r}"
            )
            assert math.isclose(
                rebuilt_value,
                shipped_value,
                rel_tol=RELATIVE_TOLERANCE,
                abs_tol=ABSOLUTE_TOLERANCE,
            ), f"{location}: {rebuilt_cell} != {shipped_cell}"


def test_generate_paper_figure_csvs_rebuilds_the_shipped_paper_inputs(tmp_path):
    artifact_root = Path(__file__).resolve().parents[1] / "paper_artifact"

    output_paths = generate_paper_figure_csvs(
        artifact_root=artifact_root,
        output_dir=tmp_path / "paper_inputs",
        core_figures_config=DEFAULT_CORE_FIGURES_CONFIG,
        benchmark_scope_config=DEFAULT_BENCHMARK_SCOPE_CONFIG,
    )

    assert set(output_paths) == {
        BACKBONE_SCENARIO_FILENAME,
        MAIN_TABLE_FILENAME,
        PGD_TRAJECTORY_FILENAME,
    }
    for filename, path in output_paths.items():
        _assert_csv_matches(path, artifact_root / "paper_inputs" / filename)
