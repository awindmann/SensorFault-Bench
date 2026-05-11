import math

import pandas as pd
import pytest

from metrics.reference_normalized import (
    compute_reference_normalized_diagnostics,
    summarize_reference_normalized_anchor,
)


EPS = 1e-8


def _make_grid(rows):
    return pd.DataFrame(
        rows,
        columns=[
            "scenario",
            "severity_bin_idx",
            "severity",
            "n_samples",
            "err_pert_mean",
        ],
    )


def _base_reference_grid() -> pd.DataFrame:
    return _make_grid(
        [
            ("drift", 0, 0.5, 16, 10.0),
            ("drift", 1, 1.0, 16, 14.0),
            ("noise", 0, 0.5, 16, 8.0),
            ("noise", 1, 1.0, 16, 12.0),
        ]
    )


def _base_target_grid() -> pd.DataFrame:
    return _make_grid(
        [
            ("drift", 0, 0.5, 16, 5.0),
            ("drift", 1, 1.0, 16, 7.0),
            ("noise", 0, 0.5, 16, 12.0),
            ("noise", 1, 1.0, 16, 18.0),
        ]
    )


def test_compute_reference_normalized_diagnostics_matches_formulas():
    diagnostics = compute_reference_normalized_diagnostics(
        target_grid_df=_base_target_grid(),
        target_clean_error=4.0,
        reference_grid_df=_base_reference_grid(),
        reference_clean_error=6.0,
        eps=EPS,
        target_label="target_run",
        reference_label="anchor_run",
    )

    scenario_df = diagnostics.scenario_df.sort_values("scenario").reset_index(drop=True)
    assert list(scenario_df["scenario"]) == ["drift", "noise"]
    assert math.isclose(scenario_df.loc[0, "scenario_mCE_snaive"], 0.5, rel_tol=0.0)
    assert math.isclose(scenario_df.loc[1, "scenario_mCE_snaive"], 1.5, rel_tol=0.0)
    assert math.isclose(
        scenario_df.loc[0, "scenario_relative_mCE_snaive"],
        8.0 / 18.0,
        rel_tol=0.0,
    )
    assert math.isclose(
        scenario_df.loc[1, "scenario_relative_mCE_snaive"],
        13.0 / 7.0,
        rel_tol=0.0,
    )
    assert math.isclose(scenario_df.loc[0, "scenario_degref_snaive"], 1.0, rel_tol=0.0)
    assert math.isclose(
        scenario_df.loc[1, "scenario_degref_snaive"],
        8.0 / 22.0,
        rel_tol=0.0,
    )
    assert math.isclose(diagnostics.mCE_snaive, 1.0, rel_tol=0.0)
    assert math.isclose(
        diagnostics.relative_mCE_snaive,
        ((8.0 / 18.0) + (13.0 / 7.0)) / 2.0,
        rel_tol=0.0,
    )
    assert math.isclose(
        diagnostics.degref_snaive,
        (1.0 + (8.0 / 22.0)) / 2.0,
        rel_tol=0.0,
    )
    assert math.isclose(diagnostics.mCE_ws_snaive, 1.5, rel_tol=0.0)
    assert math.isclose(
        diagnostics.relative_mCE_ws_snaive,
        13.0 / 7.0,
        rel_tol=0.0,
    )
    assert math.isclose(diagnostics.degref_ws_snaive, 8.0 / 22.0, rel_tol=0.0)
    assert math.isclose(
        scenario_df.loc[0, "scenario_CE_cell_meansev_snaive"],
        ((5.0 + EPS) / (10.0 + EPS) + (7.0 + EPS) / (14.0 + EPS)) / 2.0,
        rel_tol=0.0,
    )
    assert math.isclose(
        scenario_df.loc[1, "scenario_CE_cell_meansev_snaive"],
        ((12.0 + EPS) / (8.0 + EPS) + (18.0 + EPS) / (12.0 + EPS)) / 2.0,
        rel_tol=0.0,
    )
    assert math.isclose(
        scenario_df.loc[0, "scenario_S_cell_meansev_snaive"],
        ((10.0 + EPS) / (5.0 + EPS) + (14.0 + EPS) / (7.0 + EPS)) / 2.0,
        rel_tol=0.0,
    )
    assert math.isclose(
        scenario_df.loc[1, "scenario_S_cell_meansev_snaive"],
        ((8.0 + EPS) / (12.0 + EPS) + (12.0 + EPS) / (18.0 + EPS)) / 2.0,
        rel_tol=0.0,
    )
    assert math.isclose(
        diagnostics.CE_cell_ws_meansev_snaive,
        scenario_df["scenario_CE_cell_meansev_snaive"].max(),
        rel_tol=0.0,
    )
    assert math.isclose(
        diagnostics.S_cell_ws_meansev_snaive,
        scenario_df["scenario_S_cell_meansev_snaive"].min(),
        rel_tol=0.0,
    )
    assert diagnostics.ce_family_supported is True
    assert diagnostics.relative_family_supported is True
    assert diagnostics.degref_family_supported is True
    assert diagnostics.cell_family_supported is True


def test_compute_reference_normalized_diagnostics_clips_non_positive_target_degradation():
    target_grid = _make_grid(
        [
            ("drift", 0, 0.5, 16, 2.0),
            ("drift", 1, 1.0, 16, 3.0),
        ]
    )
    reference_grid = _make_grid(
        [
            ("drift", 0, 0.5, 16, 8.0),
            ("drift", 1, 1.0, 16, 9.0),
        ]
    )

    diagnostics = compute_reference_normalized_diagnostics(
        target_grid_df=target_grid,
        target_clean_error=3.0,
        reference_grid_df=reference_grid,
        reference_clean_error=4.0,
        eps=EPS,
        target_label="target_run",
        reference_label="anchor_run",
    )

    assert diagnostics.scenario_df.loc[0, "scenario_degref_snaive"] == 1.0
    assert diagnostics.degref_snaive == 1.0


def test_summarize_reference_normalized_anchor_returns_counts():
    summary = summarize_reference_normalized_anchor(
        reference_grid_df=_base_reference_grid(),
        reference_clean_error=6.0,
        reference_label="anchor_run",
    )

    assert summary.n_scenarios == 2
    assert summary.n_severity_levels == 2
    assert summary.ce_family_supported is True
    assert summary.relative_family_supported is True
    assert summary.degref_family_supported is True
    assert summary.cell_family_supported is True


def test_compute_reference_normalized_diagnostics_leaves_only_degref_family_unset_when_reference_degradation_is_non_positive():
    reference_grid = _make_grid(
        [
            ("drift", 0, 0.5, 16, 4.0),
            ("drift", 1, 1.0, 16, 6.0),
        ]
    )

    diagnostics = compute_reference_normalized_diagnostics(
        target_grid_df=_make_grid(
            [
                ("drift", 0, 0.5, 16, 7.0),
                ("drift", 1, 1.0, 16, 8.0),
            ]
        ),
        target_clean_error=2.0,
        reference_grid_df=reference_grid,
        reference_clean_error=6.0,
        eps=EPS,
        target_label="target_run",
        reference_label="anchor_run",
    )

    assert diagnostics.ce_family_supported is True
    assert diagnostics.relative_family_supported is True
    assert diagnostics.degref_family_supported is False
    assert diagnostics.cell_family_supported is True
    assert not diagnostics.scenario_df["scenario_mCE_snaive"].isna().any()
    assert not diagnostics.scenario_df["scenario_relative_mCE_snaive"].isna().any()
    assert diagnostics.scenario_df["scenario_degref_snaive"].isna().all()
    assert math.isnan(diagnostics.degref_snaive)
    assert math.isnan(diagnostics.degref_ws_snaive)


def test_compute_reference_normalized_diagnostics_leaves_relative_and_degref_families_unset_when_relative_denominator_is_non_positive():
    reference_grid = _make_grid(
        [
            ("drift", 0, 0.5, 16, 2.0),
            ("drift", 1, 1.0, 16, 3.0),
        ]
    )

    diagnostics = compute_reference_normalized_diagnostics(
        target_grid_df=_make_grid(
            [
                ("drift", 0, 0.5, 16, 4.0),
                ("drift", 1, 1.0, 16, 5.0),
            ]
        ),
        target_clean_error=1.0,
        reference_grid_df=reference_grid,
        reference_clean_error=5.0,
        eps=EPS,
        target_label="target_run",
        reference_label="anchor_run",
    )

    assert diagnostics.ce_family_supported is True
    assert diagnostics.relative_family_supported is False
    assert diagnostics.degref_family_supported is False
    assert diagnostics.cell_family_supported is True
    assert not diagnostics.scenario_df["scenario_mCE_snaive"].isna().any()
    assert diagnostics.scenario_df["scenario_relative_mCE_snaive"].isna().all()
    assert diagnostics.scenario_df["scenario_degref_snaive"].isna().all()
    assert math.isnan(diagnostics.relative_mCE_snaive)
    assert math.isnan(diagnostics.relative_mCE_ws_snaive)
    assert math.isnan(diagnostics.degref_snaive)
    assert math.isnan(diagnostics.degref_ws_snaive)


def test_compute_reference_normalized_diagnostics_leaves_only_cell_family_supported_when_ce_denominator_is_non_positive():
    reference_grid = _make_grid(
        [
            ("drift", 0, 0.5, 16, 0.0),
            ("drift", 1, 1.0, 16, 0.0),
        ]
    )

    diagnostics = compute_reference_normalized_diagnostics(
        target_grid_df=_make_grid(
            [
                ("drift", 0, 0.5, 16, 4.0),
                ("drift", 1, 1.0, 16, 5.0),
            ]
        ),
        target_clean_error=1.0,
        reference_grid_df=reference_grid,
        reference_clean_error=0.0,
        eps=EPS,
        target_label="target_run",
        reference_label="anchor_run",
    )

    assert diagnostics.ce_family_supported is False
    assert diagnostics.relative_family_supported is False
    assert diagnostics.degref_family_supported is False
    assert diagnostics.cell_family_supported is True
    assert diagnostics.scenario_df["scenario_mCE_snaive"].isna().all()
    assert diagnostics.scenario_df["scenario_relative_mCE_snaive"].isna().all()
    assert diagnostics.scenario_df["scenario_degref_snaive"].isna().all()
    assert not diagnostics.scenario_df["scenario_CE_cell_meansev_snaive"].isna().any()
    assert not diagnostics.scenario_df["scenario_S_cell_meansev_snaive"].isna().any()
    assert math.isnan(diagnostics.mCE_snaive)
    assert math.isnan(diagnostics.relative_mCE_snaive)
    assert math.isnan(diagnostics.degref_snaive)
    assert not math.isnan(diagnostics.CE_cell_ws_meansev_snaive)
    assert not math.isnan(diagnostics.S_cell_ws_meansev_snaive)


def test_compute_reference_normalized_diagnostics_raises_on_mismatched_scenarios():
    reference_grid = _base_reference_grid()
    target_grid = _base_target_grid().replace({"noise": "spike"})

    with pytest.raises(ValueError, match="identical scenario sets"):
        compute_reference_normalized_diagnostics(
            target_grid_df=target_grid,
            target_clean_error=4.0,
            reference_grid_df=reference_grid,
            reference_clean_error=6.0,
            eps=EPS,
            target_label="target_run",
            reference_label="anchor_run",
        )


def test_compute_reference_normalized_diagnostics_raises_on_mismatched_severity_bin_idx_sets():
    reference_grid = _base_reference_grid()
    target_grid = _base_target_grid().copy()
    target_grid.loc[target_grid["severity_bin_idx"] == 1, "severity_bin_idx"] = 2

    with pytest.raises(ValueError, match="identical severity_bin_idx sets"):
        compute_reference_normalized_diagnostics(
            target_grid_df=target_grid,
            target_clean_error=4.0,
            reference_grid_df=reference_grid,
            reference_clean_error=6.0,
            eps=EPS,
            target_label="target_run",
            reference_label="anchor_run",
        )


def test_compute_reference_normalized_diagnostics_raises_on_mismatched_severity_values():
    reference_grid = _base_reference_grid()
    target_grid = _base_target_grid().copy()
    target_grid.loc[target_grid["severity_bin_idx"] == 1, "severity"] = 0.9

    with pytest.raises(ValueError, match="identical numeric severity values"):
        compute_reference_normalized_diagnostics(
            target_grid_df=target_grid,
            target_clean_error=4.0,
            reference_grid_df=reference_grid,
            reference_clean_error=6.0,
            eps=EPS,
            target_label="target_run",
            reference_label="anchor_run",
        )


def test_compute_reference_normalized_diagnostics_raises_on_mismatched_cell_coverage():
    target_grid = _base_target_grid().loc[
        lambda df: ~((df["scenario"] == "noise") & (df["severity_bin_idx"] == 1))
    ]

    with pytest.raises(ValueError, match="identical scenario/severity-bin coverage"):
        compute_reference_normalized_diagnostics(
            target_grid_df=target_grid,
            target_clean_error=4.0,
            reference_grid_df=_base_reference_grid(),
            reference_clean_error=6.0,
            eps=EPS,
            target_label="target_run",
            reference_label="anchor_run",
        )


def test_compute_reference_normalized_diagnostics_raises_on_jointly_incomplete_scenario_grid():
    target_grid = _base_target_grid().loc[
        lambda df: ~((df["scenario"] == "noise") & (df["severity_bin_idx"] == 1))
    ]
    reference_grid = _base_reference_grid().loc[
        lambda df: ~((df["scenario"] == "noise") & (df["severity_bin_idx"] == 1))
    ]

    with pytest.raises(ValueError, match="full severity_bin_idx grid"):
        compute_reference_normalized_diagnostics(
            target_grid_df=target_grid,
            target_clean_error=4.0,
            reference_grid_df=reference_grid,
            reference_clean_error=6.0,
            eps=EPS,
            target_label="target_run",
            reference_label="anchor_run",
        )


def test_compute_reference_normalized_diagnostics_raises_on_mismatched_n_samples():
    reference_grid = _base_reference_grid().copy()
    reference_grid.loc[
        (reference_grid["scenario"] == "noise")
        & (reference_grid["severity_bin_idx"] == 1),
        "n_samples",
    ] = 12

    with pytest.raises(ValueError, match="matching n_samples per severity bin"):
        compute_reference_normalized_diagnostics(
            target_grid_df=_base_target_grid(),
            target_clean_error=4.0,
            reference_grid_df=reference_grid,
            reference_clean_error=6.0,
            eps=EPS,
            target_label="target_run",
            reference_label="anchor_run",
        )


def test_compute_reference_normalized_diagnostics_raises_on_empty_grid():
    empty_grid = _make_grid([])

    with pytest.raises(ValueError, match="is empty"):
        compute_reference_normalized_diagnostics(
            target_grid_df=empty_grid,
            target_clean_error=1.0,
            reference_grid_df=_base_reference_grid(),
            reference_clean_error=6.0,
            eps=EPS,
            target_label="target_run",
            reference_label="anchor_run",
        )


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (
            lambda df: df.drop(columns=["n_samples"]),
            "missing columns",
        ),
        (
            lambda df: pd.concat([df, df.iloc[[0]]], ignore_index=True),
            "duplicate \\(scenario, severity_bin_idx\\) rows",
        ),
        (
            lambda df: df.assign(
                severity=df["severity"].where(df.index != 0, 0.7)
            ),
            "maps severity_bin_idx to multiple severity values",
        ),
        (
            lambda df: df.assign(
                scenario=df["scenario"].where(df.index != 0, "   ")
            ),
            "empty scenario values",
        ),
        (
            lambda df: df.assign(
                n_samples=df["n_samples"].where(df.index != 0, 16.5)
            ),
            "non-integer n_samples values",
        ),
        (
            lambda df: df.assign(
                n_samples=df["n_samples"].where(df.index != 0, 0)
            ),
            "non-positive n_samples values",
        ),
        (
            lambda df: df.assign(
                severity_bin_idx=df["severity_bin_idx"].where(df.index != 0, 0.5)
            ),
            "non-integer severity_bin_idx values",
        ),
    ],
)
def test_compute_reference_normalized_diagnostics_rejects_malformed_target_grid(
    mutator,
    match,
):
    with pytest.raises(ValueError, match=match):
        compute_reference_normalized_diagnostics(
            target_grid_df=mutator(_base_target_grid()),
            target_clean_error=4.0,
            reference_grid_df=_base_reference_grid(),
            reference_clean_error=6.0,
            eps=EPS,
            target_label="target_run",
            reference_label="anchor_run",
        )


def test_compute_reference_normalized_diagnostics_single_scenario_single_severity():
    target_grid = _make_grid([("drift", 0, 0.5, 16, 5.0)])
    reference_grid = _make_grid([("drift", 0, 0.5, 16, 10.0)])

    diagnostics = compute_reference_normalized_diagnostics(
        target_grid_df=target_grid,
        target_clean_error=2.0,
        reference_grid_df=reference_grid,
        reference_clean_error=4.0,
        eps=EPS,
        target_label="target_run",
        reference_label="anchor_run",
    )

    assert math.isclose(diagnostics.mCE_snaive, 0.5, rel_tol=0.0)
    assert math.isclose(diagnostics.relative_mCE_snaive, (5.0 - 2.0) / (10.0 - 4.0), rel_tol=0.0)
    assert len(diagnostics.scenario_df) == 1


def test_compute_reference_normalized_diagnostics_raises_on_non_finite_clean_error():
    with pytest.raises(ValueError, match="clean error must be finite"):
        compute_reference_normalized_diagnostics(
            target_grid_df=_base_target_grid(),
            target_clean_error=float("inf"),
            reference_grid_df=_base_reference_grid(),
            reference_clean_error=6.0,
            eps=EPS,
            target_label="target_run",
            reference_label="anchor_run",
        )


@pytest.mark.parametrize("bad_eps", [0.0, -1.0, float("inf")])
def test_compute_reference_normalized_diagnostics_rejects_non_positive_or_non_finite_eps(
    bad_eps,
):
    with pytest.raises(ValueError, match="eps must be finite and strictly positive"):
        compute_reference_normalized_diagnostics(
            target_grid_df=_base_target_grid(),
            target_clean_error=4.0,
            reference_grid_df=_base_reference_grid(),
            reference_clean_error=6.0,
            eps=bad_eps,
            target_label="target_run",
            reference_label="anchor_run",
        )
