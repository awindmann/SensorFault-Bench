from __future__ import annotations

from pathlib import Path

from scripts.generate_paper_figure_csvs import (
    BACKBONE_SCENARIO_FILENAME,
    DEFAULT_BENCHMARK_SCOPE_CONFIG,
    DEFAULT_CORE_FIGURES_CONFIG,
    MAIN_TABLE_FILENAME,
    PGD_TRAJECTORY_FILENAME,
    generate_paper_figure_csvs,
)


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
        shipped_path = artifact_root / "paper_inputs" / filename
        assert path.read_bytes() == shipped_path.read_bytes(), filename
