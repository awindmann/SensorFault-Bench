import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
from plotly.subplots import make_subplots

from typing import Mapping, Sequence, Iterable, Tuple
import re

from utils.parsing import (
    assert_no_duplicate_rows,
    padded_feature_names,
    require_dataframe_columns,
    require_nonempty_string_series,
    require_numeric_series,
    validate_raw_display_id_values,
)
from visualizations.semantics import (
    PlotSemanticsRecord,
    require_plot_semantics_mapping,
)

_MAX_FORECAST_DISPLAY_CHANNELS = 12

# ---------------------------------------------------------------------------
# Shared figure style constants.
# Palette: Okabe-Ito (colorblind-safe).
# Two font tiers: _PUB_ for core benchmark figures (print at 5.5 in text width),
#                 _DIAG_ for diagnostic/audit figures (screen viewing).
# ---------------------------------------------------------------------------
_FONT_FAMILY = "Serif"
_GRID_COLOR = "rgba(0,0,0,0.12)"
_PARETO_LINE = dict(width=2, color="grey", dash="dot")
_OKABE_ITO: Sequence[str] = [
    "#0072B2",  # blue
    "#D55E00",  # red-orange
    "#009E73",  # green
    "#E69F00",  # orange
    "#56B4E9",  # light blue
    "#CC79A7",  # pink
    "#F0E442",  # yellow
    "#999999",  # gray
]

# Paper-tier (print at ~100 dpi → 550 px = 5.5 in).
_TEXT_WIDTH_PX = 550
_SINGLE_HEIGHT_PX = 380
_PANEL_HEIGHT_PX = 320
_PUB_FONT = dict(size=10, family=_FONT_FAMILY)
_PUB_TICK_SIZE = 9
_PUB_ANNOTATION_SIZE = 9
_PUB_MARKER_SIZE = 8
_PUB_LEGEND_SIZE = 9

# Diagnostic-tier (browser / interactive viewing).
_DIAG_FONT = dict(size=16, family=_FONT_FAMILY)
_DIAG_FONT_COMPACT = dict(size=14, family=_FONT_FAMILY)
_DIAG_TITLE_SIZE = 20
_DIAG_TITLE_SIZE_COMPACT = 18
_DIAG_ANNOTATION_SIZE = 12
_FORECAST_BASE_HEIGHT_PX = 400
_FORECAST_BASE_TOP_MARGIN_PX = 50
_FORECAST_TITLE_EXTRA_LINE_PX = 34

pio.renderers.default = "browser"
try:
    pio.kaleido.scope.mathjax = None  # Disable MathJax in static export
except Exception:
    pass  # Kaleido is not available during import. write_image handles export-time failures.


_PERT_ORDER: Sequence[str] = [
    "Drift", "Attenuation", "Noise", "StuckSensor", "MissingData",
    "TimeCompress", "TimeStretch", "Spike",
    "WrongState", "Chattering",
]

ROBUSTNESS_METRIC_DISPLAY = {
    "mCE_snaive": "Mean CE vs SeasonalNaive",
    "relative_mCE_snaive": "Mean Relative CE vs SeasonalNaive",
    "degref_snaive": "Mean DegRef vs SeasonalNaive",
    "mCE_ws_snaive": "Worst-Scenario CE vs SeasonalNaive",
    "relative_mCE_ws_snaive": "Worst-Scenario Relative CE vs SeasonalNaive",
    "degref_ws_snaive": "Worst-Scenario DegRef vs SeasonalNaive",
    "CE_cell_ws_meansev_snaive": "Worst-Scenario Cell CE vs SeasonalNaive",
    "S_cell_ws_meansev_snaive": "Worst-Scenario Cell S vs SeasonalNaive",
    "scenario_mCE_snaive": "Scenario CE vs SeasonalNaive",
    "scenario_relative_mCE_snaive": "Scenario Relative CE vs SeasonalNaive",
    "scenario_degref_snaive": "Scenario DegRef vs SeasonalNaive",
    "scenario_CE_cell_meansev_snaive": "Scenario Cell CE vs SeasonalNaive",
    "scenario_S_cell_meansev_snaive": "Scenario Cell S vs SeasonalNaive",
    "D": "Scenario Degradation",
    "err_pert": "Scenario Perturbed Error",
    "D_w": "Worst-Scenario Degradation",
    "D_mean": "Mean Degradation",
    "err_pert_ws": "Worst-Scenario Error",
}

# ---------------------------------------------------------------------------
# Display name helpers
# ---------------------------------------------------------------------------


def _html_break_line_count(text: str) -> int:
    """Count explicit HTML line breaks used in Plotly text fields."""
    return max(1, len(re.split(r"<br\s*/?>", str(text), flags=re.IGNORECASE)))


def _pretty(name: str) -> str:
    return "".join(w.capitalize() for w in name.replace("-", "_").split("_"))


def robustness_metric_display_name(metric_key: str) -> str:
    key = str(metric_key)
    title = ROBUSTNESS_METRIC_DISPLAY.get(key)
    if title is None:
        raise ValueError(
            f"Unknown robustness metric key '{key}'. "
            f"Known keys: {sorted(ROBUSTNESS_METRIC_DISPLAY.keys())}."
        )
    return title


def metric_display_name(metric_key: str) -> str:
    """Resolve one supported metric key to an explicit display label."""
    key = str(metric_key).strip()
    if not key:
        raise ValueError("Metric display name requires a non-empty metric key.")
    if key in ROBUSTNESS_METRIC_DISPLAY:
        return robustness_metric_display_name(key)
    explicit_display = {
        "best_val_loss": "Best Val Loss",
        "err_clean": "Clean Error",
        "err_clean_global": "Global Clean Error",
        "err_pert_mean": "Mean Perturbed Error",
        "rho_eff": "Effective Robustness Residual",
        "tau_mean": "Mean Corrupted-Error Effect",
    }
    if key in explicit_display:
        return explicit_display[key]
    if key.endswith("_w"):
        base_metric = key.removesuffix("_w")
        return f"{metric_display_name(base_metric)} at Worst Scenario"
    if key.endswith("_test"):
        base_metric = key.removesuffix("_test")
        return f"{metric_display_name(base_metric)} (Test)"
    if key.endswith("_val"):
        base_metric = key.removesuffix("_val")
        return f"{metric_display_name(base_metric)} (Val)"
    uppercase_metrics = {"MSE", "MAE", "MAPE", "SMAPE"}
    if key in uppercase_metrics:
        return key
    return " ".join(part.capitalize() for part in key.replace("_", " ").split())


def delta_metric_display_name(metric_key: str) -> str:
    """Resolve one supported delta metric key to an explicit display label."""
    key = str(metric_key).strip()
    if not key:
        raise ValueError("Delta metric display name requires a non-empty metric key.")
    return f"Δ {metric_display_name(key)}"


def _pretty_metric_title(col_name: str, include_mean_hint: bool = True) -> str:
    """Derive a human-friendly metric title from a dataframe column name.
    Examples:
    - "MSE_test_mean" -> "MSE (Mean)"
    - "MAE_val_std"   -> "MAE (Std)"
    - "MAE_test"      -> "MAE"
    """
    base = col_name
    mean_flag = False
    std_flag = False
    for suffix in ("_mean", "_std"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            if suffix == "_mean":
                mean_flag = True
            else:
                std_flag = True
            break
    for split in ("_test", "_val"):
        if base.endswith(split):
            base = base[: -len(split)]
            break
    title = base.replace("_", " ")
    # Preserve common uppercase metric abbreviations
    if title.lower() in {"mse", "mae", "mape", "smape"}:
        title = title.upper()
    else:
        title = " ".join(w.capitalize() for w in title.split())
    if include_mean_hint and mean_flag:
        return f"{title} (Mean)"
    if include_mean_hint and std_flag:
        return f"{title} (Std)"
    return title


def _normalize_scenario_name(name: str) -> str:
    if name is None:
        return ""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _pretty_robust_title(col_name: str, include_mean_hint: bool = True) -> str:
    """Human-friendly robustness title from column name.
    Examples:
    - "D_w_mean" -> "Worst-Scenario Degradation" or "Worst-Scenario Degradation (Mean)"
    - "err_pert_ws_mean" -> "Worst-Scenario Error (Mean)"
    - "mCE_ws_snaive_mean" -> "Worst-Scenario CE vs SeasonalNaive (Mean)"
    Raises on unknown robustness metric keys.
    """
    base = str(col_name)
    summary_suffix: str | None = None
    for suffix, suffix_label in (("_mean", "Mean"), ("_std", "Std")):
        if base.endswith(suffix):
            candidate = base[: -len(suffix)]
            # Only strip aggregation suffixes when the remaining key is a known metric.
            if candidate in ROBUSTNESS_METRIC_DISPLAY:
                base = candidate
                summary_suffix = suffix_label
            break
    title = robustness_metric_display_name(base)
    if include_mean_hint and summary_suffix is not None:
        return f"{title} ({summary_suffix})"
    return title


def _trajectory_output_display_map() -> dict[str, str]:
    return {
        "adaptive_robust_loss": "Adaptive Robust Loss",
        "revin": "RevIN",
        "ensemble": "Ensemble",
        "randomized_training": "Randomized Training",
        "adversarial_training": "PGD adversarial training",
        "fault_augmentation": "Fault Augmentation",
        "randomized_smoothing": "Randomized Smoothing",
    }


def trajectory_output_label_for_method(method: str) -> str:
    method_key = str(method).strip()
    if not method_key:
        raise ValueError(
            "Cannot build trajectory legend label for empty robustness method."
        )
    display_name = _trajectory_output_display_map().get(method_key)
    if display_name is None:
        raise ValueError(
            "Cannot build trajectory legend label for unknown robustness method "
            f"'{method_key}'."
        )
    return f"+ {display_name}"


def _score_axis_range(
    values: Iterable[float],
) -> Tuple[float, float]:
    """Compute axis range for robustness scores.

    The default range is [0, 1] and expands if data falls outside.
    """
    arr = np.asarray(list(values), dtype=float).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 1.0

    lower = float(np.min(finite))
    upper = float(np.max(finite))

    # Anchor at [0, 1] as the baseline, expand if data exceeds.
    lower = min(lower, 0.0)
    upper = max(upper, 1.0)

    span = upper - lower
    pad = 0.05 * span
    if lower < 0.0:
        lower -= pad
    upper += pad
    return lower, upper


def _require_plot_semantics_record(
    semantics: PlotSemanticsRecord,
    *,
    context: str,
) -> PlotSemanticsRecord:
    if not isinstance(semantics, PlotSemanticsRecord):
        raise ValueError(
            f"{context} requires PlotSemanticsRecord metadata; got "
            f"{type(semantics).__name__}."
        )
    return semantics


def _numeric_axis_range(
    values: Iterable[float],
    *,
    include_zero: bool = False,
) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=float).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        if include_zero:
            return -1.0, 1.0
        return 0.0, 1.0
    lower = float(np.min(finite))
    upper = float(np.max(finite))
    if include_zero:
        lower = min(lower, 0.0)
        upper = max(upper, 0.0)
    span = upper - lower
    pad = 0.05 * span if span > 0 else 0.05 * max(abs(lower), abs(upper), 1.0)
    return lower - pad, upper + pad


def _delta_axis_range(
    values: Iterable[float],
    *,
    neutral_value: float,
) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=float).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return neutral_value - 1.0, neutral_value + 1.0
    abs_max = float(np.max(np.abs(finite - neutral_value)))
    if abs_max <= 0.0:
        abs_max = 1.0
    bound = 1.05 * abs_max
    return neutral_value - bound, neutral_value + bound


def _axis_range_from_semantics(
    values: Iterable[float],
    semantics: PlotSemanticsRecord,
    *,
    include_zero: bool = False,
) -> Tuple[float, float]:
    record = _require_plot_semantics_record(
        semantics,
        context="Axis range computation",
    )
    if record.axis_family == "score":
        return _score_axis_range(values)
    if record.axis_family == "delta":
        neutral_value = 0.0 if record.neutral_value is None else float(record.neutral_value)
        return _delta_axis_range(values, neutral_value=neutral_value)
    return _numeric_axis_range(values, include_zero=include_zero)


def _axis_is_minimize(semantics: PlotSemanticsRecord) -> bool:
    record = _require_plot_semantics_record(
        semantics,
        context="Axis direction resolution",
    )
    return record.direction == "minimize"


def _axis_better_annotation(
    semantics: PlotSemanticsRecord,
    *,
    axis: str,
    reversed_axis: bool = False,
) -> str:
    record = _require_plot_semantics_record(
        semantics,
        context="Axis direction annotation",
    )
    if axis not in {"x", "y"}:
        raise ValueError(f"Unknown axis {axis!r} for direction annotation.")
    positive_direction_is_better = record.direction == "maximize"
    if reversed_axis:
        positive_direction_is_better = not positive_direction_is_better
    if axis == "x":
        return "Better →" if positive_direction_is_better else "Better ←"
    return "Better ↑" if positive_direction_is_better else "Better ↓"


def _directional_colorscale(semantics: PlotSemanticsRecord) -> str:
    record = _require_plot_semantics_record(
        semantics,
        context="Color scale selection",
    )
    if record.direction == "minimize":
        return "RdYlGn_r"
    return "RdYlGn"


def _utility_aligned_delta_multiplier(semantics: PlotSemanticsRecord) -> float:
    record = _require_plot_semantics_record(
        semantics,
        context="Delta utility alignment",
    )
    if record.axis_family != "delta":
        raise ValueError(
            "Delta utility alignment requires axis_family='delta', got "
            f"{record.axis_family!r}."
        )
    return -1.0 if record.direction == "minimize" else 1.0


# ---------------------------------------------------------------------------
# Perturbation & scenario plots
# ---------------------------------------------------------------------------


def plot_perturbation_curves(
    df: pd.DataFrame,
    model_name: str,
    *,
    value_col: str,
    y_semantics: PlotSemanticsRecord,
    idx_to_name: Mapping[int, str] | None = None,
    title: str | None = None,
    neutral_anchor_value: float | None = None,
) -> go.Figure:
    """Plot scenario profiles for each perturbation type across severity levels.

    Args:
        df: DataFrame containing `pert_idx`, `severity`, and one metric column.
        model_name: Name of the evaluated model used in plot titles.
        value_col: Metric column to aggregate and plot.
        y_semantics: Explicit plotting semantics for the metric on the y-axis.
        idx_to_name: Optional mapping from perturbation index to name.
        title: Custom plot title. Defaults to a generic message when omitted.
        neutral_anchor_value: Optional explicit clean-reference anchor value
            added at severity 0 for every plotted scenario.
    """
    df = df.copy()
    semantics = _require_plot_semantics_record(
        y_semantics,
        context="plot_perturbation_curves",
    )
    required_cols = {"pert_idx", "severity", value_col}
    missing_cols = sorted(required_cols - set(df.columns))
    if missing_cols:
        raise KeyError(
            f"Expected perturbation-curve columns {sorted(required_cols)}; "
            f"missing {missing_cols}."
        )
    df.rename(
        columns={
            "severity": "_severity_level",
            value_col: "_metric_value",
        },
        inplace=True,
    )
    df["pert"] = (
        df["pert_idx"].map(idx_to_name) if idx_to_name else
        "Pert " + df["pert_idx"].astype(str)
    )
    df["pert"] = df["pert"].apply(_pretty)

    agg_df = (
        df.groupby(["pert", "_severity_level"], as_index=False)
        .agg(metric_value=("_metric_value", "mean"))
    )
    agg_df.rename(columns={"_severity_level": "severity"}, inplace=True)

    if neutral_anchor_value is not None:
        neutral_value = float(neutral_anchor_value)
        if not np.isfinite(neutral_value):
            raise ValueError(
                "plot_perturbation_curves neutral_anchor_value must be finite."
            )
        anchor_rows: list[dict[str, float | str]] = []
        for pert_name, pert_df in agg_df.groupby("pert", sort=False):
            severity_values = pert_df["severity"].to_numpy(dtype=float)
            if np.isclose(severity_values, 0.0, rtol=0.0, atol=1e-12).any():
                continue
            anchor_rows.append(
                {
                    "pert": str(pert_name),
                    "severity": 0.0,
                    "metric_value": neutral_value,
                }
            )
        if anchor_rows:
            agg_df = pd.concat([pd.DataFrame(anchor_rows), agg_df], ignore_index=True)

    y_range = _axis_range_from_semantics(
        agg_df["metric_value"].to_numpy(dtype=float),
        semantics,
    )

    fig = go.Figure()
    present_perturbations = set(agg_df["pert"].astype(str).tolist())
    ordered_perturbations = [
        pert_name for pert_name in _PERT_ORDER if pert_name in present_perturbations
    ]
    ordered_perturbations.extend(
        sorted(present_perturbations - set(ordered_perturbations))
    )
    for pert_name in ordered_perturbations:
        pert_df = agg_df[agg_df["pert"] == pert_name].sort_values("severity")
        if not pert_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=pert_df["severity"],
                    y=pert_df["metric_value"],
                    mode="lines",
                    name=pert_name,
                )
            )

    title_text = title if title else f"{model_name} Across Severity Levels"
    fig.update_layout(
        width=600, height=350,
        title=dict(text=title_text, y=0.95, x=0.5, xanchor="center", yanchor="top"),
        xaxis_title="Severity Level",
        yaxis_title=semantics.display_label,
        yaxis_range=list(y_range),
        font=_DIAG_FONT,
        title_font=dict(size=_DIAG_TITLE_SIZE),
        margin=dict(l=5, t=50, b=5, r=5),
        template="plotly_white",
        showlegend=True,
        legend_traceorder="normal",
    )
    return fig


def plot_scenario_radar(
    traces: Mapping[str, Mapping[str, float]] | Sequence[Tuple[str, Mapping[str, float]]],
    *,
    title: str | None = None,
    scenario_order: Sequence[str] | None = None,
    radial_range: Tuple[float, float] | None = None,
    fill_opacity: float = 0.25,
    color_sequence: Sequence[str] | None = None,
) -> go.Figure:
    """Render a spider/radar chart of scenario metric values.

    Args:
        traces: Mapping or sequence of (label, scenario->value) pairs. Values outside the provided scenarios are ignored. Missing scenarios are skipped.
        title: Optional plot title.
        scenario_order: Order of scenarios around the radar. Defaults to perturbation order.
        radial_range: (min, max) range for the radial axis.
        fill_opacity: alpha for the polygon fill.
    """
    if isinstance(traces, Mapping):
        items: Iterable[Tuple[str, Mapping[str, float]]] = traces.items()
    else:
        items = traces

    if scenario_order is None:
        scenario_order = [name for name in _PERT_ORDER]
    normalized_order = [_normalize_scenario_name(name) for name in scenario_order]

    def _order_by_normalized(names: Iterable[str]) -> list[str]:
        norm_map: dict[str, str] = {}
        for name in names:
            norm = _normalize_scenario_name(name)
            if norm not in norm_map:
                norm_map[norm] = name
        ordered: list[str] = []
        for norm in normalized_order:
            if norm in norm_map:
                ordered.append(norm_map.pop(norm))
        ordered.extend(name for _, name in sorted(norm_map.items()))
        return ordered

    # Collect all scenarios present in traces to avoid empty axes.
    present = set()
    normalized: list[Tuple[str, list[str], list[float]]] = []
    for label, values in items:
        available = {scenario: float(val) for scenario, val in values.items() if val is not None}
        if not available:
            continue
        present.update(available.keys())
        ordered_theta = _order_by_normalized(available.keys())
        ordered_r = [available[s] for s in ordered_theta]
        normalized.append((label, ordered_theta, ordered_r))

    if not normalized:
        fig = go.Figure()
        fig.update_layout(title=title or "Scenario Metric Profile (No Data)")
        return fig

    # Restrict theta to scenarios that appear somewhere.
    keep_theta = _order_by_normalized(present)

    fig = go.Figure()
    palette = list(color_sequence) if color_sequence is not None else px.colors.qualitative.Plotly
    fill_alpha = max(0.0, min(1.0, float(fill_opacity)))
    for idx, (label, theta_vals, r_vals) in enumerate(normalized):
        theta_filtered = [s for s in keep_theta if s in theta_vals]
        if not theta_filtered:
            continue
        value_map = dict(zip(theta_vals, r_vals))
        ordered_r = [value_map[s] for s in theta_filtered]
        # Close the polygon by repeating the first point.
        display_theta = [_pretty(s.replace(" ", "_")) for s in theta_filtered]
        closed_theta = display_theta + [display_theta[0]]
        closed_r = ordered_r + [ordered_r[0]]
        base_color = palette[idx % len(palette)] if palette else None
        fillcolor = None
        if base_color:
            if base_color.startswith("#") and len(base_color) == 7:
                r = int(base_color[1:3], 16)
                g = int(base_color[3:5], 16)
                b = int(base_color[5:7], 16)
                fillcolor = f"rgba({r},{g},{b},{fill_alpha})"
            elif base_color.startswith("rgb"):
                components = base_color.rstrip(")").split("(")[1]
                parts = components.split(",")
                if len(parts) >= 3:
                    r, g, b = parts[:3]
                    fillcolor = f"rgba({r.strip()},{g.strip()},{b.strip()},{fill_alpha})"
        if fillcolor is None:
            fillcolor = f"rgba(31,119,180,{fill_alpha})"
        hover_template = f"{label} — value=%{{r:.4f}}<extra></extra>"
        fig.add_trace(
            go.Scatterpolar(
                r=closed_r,
                theta=closed_theta,
                mode="lines+markers",
                name=label,
                fill="toself",
                fillcolor=fillcolor,
                line=dict(color=base_color) if base_color else None,
                marker=dict(color=base_color) if base_color else None,
                opacity=1.0,
                hovertemplate=hover_template,
            )
        )

    if radial_range:
        r_min, r_max = radial_range
    else:
        all_vals = [val for _, _, values in normalized for val in values]
        r_min, r_max = _score_axis_range(all_vals)

    default_title = "Scenario Metric Profile"
    title_text = default_title if title is None else title
    fig.update_layout(
        title=title_text,
        polar=dict(
            radialaxis=dict(range=[r_min, r_max], tickfont=dict(size=12), ticks="outside", showline=True),
            angularaxis=dict(direction="clockwise"),
        ),
        showlegend=len(normalized) > 1,
        margin=dict(l=40, r=40, t=60, b=40),
        font=_DIAG_FONT,
        title_font=dict(size=_DIAG_TITLE_SIZE),
    )
    return fig


# ---------------------------------------------------------------------------
# Pareto & ranked performance plots
# ---------------------------------------------------------------------------


def _pareto_mask(df: pd.DataFrame, cols: list[str], minimize_mask: list[bool]) -> pd.Series:
    """Return boolean mask of Pareto-optimal rows for objectives in cols."""
    arr = df[cols].to_numpy(dtype=float).copy()
    for j, minimize in enumerate(minimize_mask):
        if not minimize:  # convert maximize to minimize by negation
            arr[:, j] = -arr[:, j]
    n = len(arr)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        dominated = (arr <= arr[i]).all(axis=1) & (arr < arr[i]).any(axis=1)
        dominated[i] = False
        if dominated.any():
            keep[i] = False
    return pd.Series(keep, index=df.index)


def plot_pareto(
    data_df: pd.DataFrame,
    *,
    perf_col: str,
    robust_col: str,
    x_semantics: PlotSemanticsRecord,
    y_semantics: PlotSemanticsRecord,
    model_col: str = "model",
    arch_col: str | None = None,
    perf_std_col: str | None = None,
    robust_std_col: str | None = None,
    perf_lower_is_better: bool = True,
    flip_perf_axis: bool = False,
    show_std: bool = False,
    x_include_zero: bool = False,
    title: str = "Performance vs Robustness",
    x_title: str | None = None,
    y_title: str | None = None,
) -> go.Figure:
    df = data_df.copy()
    x_record = _require_plot_semantics_record(x_semantics, context="plot_pareto x-axis")
    y_record = _require_plot_semantics_record(y_semantics, context="plot_pareto y-axis")
    perf_minimize = _axis_is_minimize(x_record)
    if bool(perf_lower_is_better) != perf_minimize:
        raise ValueError(
            "plot_pareto received inconsistent performance direction metadata: "
            f"perf_lower_is_better={perf_lower_is_better} but x_semantics.direction="
            f"{x_record.direction!r}."
        )
    for c in [perf_col, robust_col, model_col]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    color_col = arch_col if arch_col and arch_col in df.columns else model_col

    df["_pareto"] = _pareto_mask(
        df,
        [perf_col, robust_col],
        [perf_minimize, _axis_is_minimize(y_record)],
    )

    scatter_kwargs = dict(
        x=perf_col, y=robust_col, color=color_col, text=model_col,
        hover_data={perf_col: ":.4f", robust_col: ":.4f", model_col: True, color_col: True},
        template="plotly_white",
    )
    has_std = show_std and (perf_std_col in df.columns if perf_std_col else False) and (
        robust_std_col in df.columns if robust_std_col else False
    )
    if has_std:
        fig = px.scatter(df, error_x=perf_std_col, error_y=robust_std_col, **scatter_kwargs)
    else:
        fig = px.scatter(df, **scatter_kwargs)
    fig.update_traces(textposition="top center", cliponaxis=False)

    # Pareto frontier line
    x_range_base = list(
        _axis_range_from_semantics(
            df[perf_col].to_numpy(dtype=float),
            x_record,
            include_zero=x_include_zero,
        )
    )
    x_reversed = flip_perf_axis and perf_minimize
    if x_reversed:
        x_range_base = [x_range_base[1], x_range_base[0]]

    pf = df[df["_pareto"]].sort_values(
        perf_col,
        ascending=not x_reversed,
    )
    if not pf.empty:
        fig.add_trace(
            go.Scatter(
                x=pf[perf_col],
                y=pf[robust_col],
                mode="lines",
                line=_PARETO_LINE,
                name="Pareto frontier",
                hoverinfo="skip",
                showlegend=True,
            )
        )

    fig.update_xaxes(
        title_text=x_title or x_record.display_label,
        range=x_range_base,
        dtick=None,
        ticks="outside",
        ticklen=6,
        tickwidth=1,
        showline=True,
        mirror=True,
        zeroline=False,
        showgrid=True,
        gridcolor=_GRID_COLOR,
    )
    fig.update_yaxes(
        title_text=y_title or y_record.display_label,
        range=list(
            _axis_range_from_semantics(
                df[robust_col].to_numpy(dtype=float),
                y_record,
            )
        ),
        dtick=None,
        ticks="outside",
        ticklen=6,
        tickwidth=1,
        showline=True,
        mirror=True,
        zeroline=False,
        showgrid=True,
        gridcolor=_GRID_COLOR,
    )

    fig.update_layout(
        title=title,
        legend_title_text="Architecture" if color_col == arch_col else "Model",
        margin=dict(l=10, r=10, t=50, b=10),
        font=_DIAG_FONT,
        title_font=dict(size=_DIAG_TITLE_SIZE),
    )

    # Direction annotations
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=1 if x_reversed else 0,
        y=-0.01,
        text=_axis_better_annotation(
            x_record,
            axis="x",
            reversed_axis=x_reversed,
        ),
        showarrow=False,
        xanchor="right" if x_reversed else "left",
        yanchor="top",
        yshift=-18,
        font=dict(size=_DIAG_ANNOTATION_SIZE),
    )
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=-0.04,
        y=0.92,
        text=_axis_better_annotation(
            y_record,
            axis="y",
        ),
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        xshift=-25,
        font=dict(size=_DIAG_ANNOTATION_SIZE),
    )
    return fig


def compute_ranked_performance_robustness_summary(
    df_long: pd.DataFrame,
    *,
    dataset_col: str,
    model_col: str,
    perf_col: str,
    robust_col: str,
) -> pd.DataFrame:
    required_cols = {dataset_col, model_col, perf_col, robust_col}
    missing_cols = sorted(required_cols - set(df_long.columns))
    if missing_cols:
        raise ValueError(
            "Cannot compute ranked performance/robustness summary: missing columns "
            f"{missing_cols}."
        )
    if df_long.empty:
        raise ValueError(
            "Cannot compute ranked performance/robustness summary from an empty dataframe."
        )

    df = df_long.loc[:, [dataset_col, model_col, perf_col, robust_col]].copy()
    perf_values = require_numeric_series(
        df[perf_col],
        column_name=perf_col,
        context="Ranked performance/robustness summary",
        allow_nan=False,
        allow_infinite=False,
    )
    robust_values = require_numeric_series(
        df[robust_col],
        column_name=robust_col,
        context="Ranked performance/robustness summary",
        allow_nan=False,
        allow_infinite=False,
    )
    df[perf_col] = perf_values.to_numpy(dtype=float)
    df[robust_col] = robust_values.to_numpy(dtype=float)
    df["perf_rank"] = df.groupby(dataset_col)[perf_col].rank(
        method="average",
        ascending=True,
    )
    return (
        df.groupby(model_col, dropna=False)
        .agg(
            avg_perf_rank=("perf_rank", "mean"),
            mean_metric_value=(robust_col, "mean"),
            n_datasets=(dataset_col, "nunique"),
        )
        .reset_index()
    )


def _ranked_performance_plot_df(
    df_long: pd.DataFrame,
    *,
    dataset_col: str,
    model_col: str,
    arch_col: str | None,
    perf_col: str,
    robust_col: str,
    robust_semantics: PlotSemanticsRecord,
) -> tuple[pd.DataFrame, str]:
    plot_df = compute_ranked_performance_robustness_summary(
        df_long,
        dataset_col=dataset_col,
        model_col=model_col,
        perf_col=perf_col,
        robust_col=robust_col,
    )
    color_col = model_col
    if arch_col is not None:
        if arch_col not in df_long.columns:
            raise ValueError(
                f"Cannot build ranked Pareto plot: missing grouping column '{arch_col}'."
            )
        model_arch = df_long.loc[:, [model_col, arch_col]].drop_duplicates().copy()
        missing_group_mask = model_arch[arch_col].isna()
        if missing_group_mask.any():
            examples = (
                model_arch.loc[missing_group_mask, [model_col, arch_col]]
                .head(5)
                .to_dict(orient="records")
            )
            raise ValueError(
                "Cannot build ranked Pareto plot: missing grouping values found for "
                f"'{arch_col}'. Examples: {examples}."
            )
        model_arch[arch_col] = model_arch[arch_col].astype(str).str.strip()
        empty_group_mask = model_arch[arch_col] == ""
        if empty_group_mask.any():
            examples = (
                model_arch.loc[empty_group_mask, [model_col, arch_col]]
                .head(5)
                .to_dict(orient="records")
            )
            raise ValueError(
                "Cannot build ranked Pareto plot: empty grouping values found for "
                f"'{arch_col}'. Examples: {examples}."
            )
        duplicate_models = model_arch.duplicated(subset=[model_col], keep=False)
        if duplicate_models.any():
            examples = (
                model_arch.loc[duplicate_models, [model_col, arch_col]]
                .head(5)
                .to_dict(orient="records")
            )
            raise ValueError(
                "Cannot build ranked Pareto plot: multiple grouping values found for "
                f"'{model_col}'. Examples: {examples}."
            )
        plot_df = plot_df.merge(model_arch, on=model_col, how="left", validate="1:1")
        color_col = arch_col
    plot_df["_pareto"] = _pareto_mask(
        plot_df,
        ["avg_perf_rank", "mean_metric_value"],
        [True, _axis_is_minimize(robust_semantics)],
    )
    return plot_df, color_col


def _ranked_performance_hovertemplate(
    *,
    model_col: str,
    color_label: str,
    y_label: str,
) -> str:
    return (
        f"{model_col}: %{{customdata[0]}}<br>"
        f"{color_label}: %{{customdata[1]}}<br>"
        "Average Performance Rank: %{x:.2f}<br>"
        f"{y_label}: %{{y:.3f}}<br>"
        "Datasets: %{customdata[2]}<extra></extra>"
    )


def _ranked_performance_axes(
    *,
    plot_dfs: Sequence[pd.DataFrame],
    robust_semantics: PlotSemanticsRecord,
) -> tuple[list[float], list[float]]:
    valid_plot_dfs = [df for df in plot_dfs if not df.empty]
    if not valid_plot_dfs:
        raise ValueError("Cannot compute ranked Pareto axes from empty plot data.")
    x_vals = np.concatenate(
        [df["avg_perf_rank"].to_numpy(dtype=float) for df in valid_plot_dfs]
    )
    y_vals = np.concatenate(
        [df["mean_metric_value"].to_numpy(dtype=float) for df in valid_plot_dfs]
    )
    max_rank = int(np.ceil(float(np.nanmax(x_vals)))) if x_vals.size else 1
    return [max_rank + 0.3, 0.7], list(
        _axis_range_from_semantics(y_vals, robust_semantics)
    )


def _ranked_performance_color_map(values: Sequence[str]) -> dict[str, str]:
    unique_values = [str(value) for value in values]
    return {
        value: _OKABE_ITO[idx % len(_OKABE_ITO)]
        for idx, value in enumerate(unique_values)
    }


def plot_ranked_performance_robustness_pareto(
    df_long: pd.DataFrame,
    *,
    dataset_col: str,
    model_col: str,
    arch_col: str | None = None,
    perf_col: str,
    robust_col: str,
    y_semantics: PlotSemanticsRecord,
    x_title: str | None = None,
    y_title: str | None = None,
) -> go.Figure:
    plot_df, color_col = _ranked_performance_plot_df(
        df_long,
        dataset_col=dataset_col,
        model_col=model_col,
        arch_col=arch_col,
        perf_col=perf_col,
        robust_col=robust_col,
        robust_semantics=y_semantics,
    )
    color_label = "Architecture" if color_col == arch_col else "Model"
    color_values = sorted(plot_df[color_col].astype(str).str.strip().unique())
    color_map = _ranked_performance_color_map(color_values)

    hovertemplate = _ranked_performance_hovertemplate(
        model_col=model_col,
        color_label=color_label,
        y_label=y_title or y_semantics.display_label,
    )

    fig = go.Figure()
    for group_value in color_values:
        group_df = plot_df.loc[
            plot_df[color_col].astype(str).str.strip() == group_value
        ].sort_values(model_col)
        if group_df.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=group_df["avg_perf_rank"],
                y=group_df["mean_metric_value"],
                mode="markers+text",
                text=group_df[model_col],
                textposition="top center",
                textfont=dict(size=_PUB_TICK_SIZE),
                cliponaxis=False,
                marker=dict(size=_PUB_MARKER_SIZE, color=color_map[group_value]),
                name=group_value,
                customdata=np.column_stack(
                    [
                        group_df[model_col].astype(str),
                        group_df[color_col].astype(str),
                        group_df["n_datasets"].astype(int),
                    ]
                ),
                hovertemplate=hovertemplate,
            )
        )

    pareto_df = plot_df.loc[plot_df["_pareto"]].sort_values("avg_perf_rank")
    if not pareto_df.empty:
        fig.add_trace(
            go.Scatter(
                x=pareto_df["avg_perf_rank"],
                y=pareto_df["mean_metric_value"],
                mode="lines",
                line=_PARETO_LINE,
                name="Pareto frontier",
                hoverinfo="skip",
                showlegend=True,
            )
        )

    x_range, y_range = _ranked_performance_axes(
        plot_dfs=[plot_df],
        robust_semantics=y_semantics,
    )
    fig.update_xaxes(
        title_text=(
            x_title
            or f"Average {_pretty_metric_title(perf_col, include_mean_hint=False)} Rank"
        ),
        title_font=dict(size=_PUB_FONT["size"]),
        dtick=1,
        range=x_range,
        ticks="outside",
        ticklen=4,
        tickwidth=1,
        tickfont=dict(size=_PUB_TICK_SIZE),
        showline=True,
        mirror=True,
        zeroline=False,
        showgrid=True,
        gridcolor=_GRID_COLOR,
    )
    fig.update_yaxes(
        title_text=y_title or y_semantics.display_label,
        title_font=dict(size=_PUB_FONT["size"]),
        range=y_range,
        ticks="outside",
        ticklen=4,
        tickwidth=1,
        tickfont=dict(size=_PUB_TICK_SIZE),
        showline=True,
        mirror=True,
        zeroline=False,
        showgrid=True,
        gridcolor=_GRID_COLOR,
    )
    fig.update_layout(
        template="plotly_white",
        width=_TEXT_WIDTH_PX,
        height=_SINGLE_HEIGHT_PX,
        autosize=False,
        legend_title_text=color_label,
        legend=dict(font=dict(size=_PUB_LEGEND_SIZE)),
        margin=dict(l=10, r=10, t=10, b=10),
        font=_PUB_FONT,
    )
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=1,
        y=-0.01,
        text="Better \u2192",
        showarrow=False,
        xanchor="right",
        yanchor="top",
        yshift=-14,
        font=dict(size=_PUB_ANNOTATION_SIZE),
    )
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=-0.04,
        y=0.92,
        text=_axis_better_annotation(y_semantics, axis="y"),
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        xshift=-20,
        font=dict(size=_PUB_ANNOTATION_SIZE),
    )
    return fig


def plot_ranked_performance_robustness_pareto_panels(
    panel_frames: Sequence[tuple[str, pd.DataFrame]],
    *,
    dataset_col: str,
    model_col: str,
    arch_col: str | None = None,
    perf_col: str,
    robust_col: str,
    y_semantics: PlotSemanticsRecord,
    x_title: str | None = None,
    y_title: str | None = None,
) -> go.Figure:
    if len(panel_frames) != 2:
        raise ValueError(
            "Ranked performance/robustness panel plot expects exactly two panels."
        )

    panel_plot_dfs: list[tuple[str, pd.DataFrame, str]] = []
    for panel_title, panel_df in panel_frames:
        if panel_df.empty:
            raise ValueError(
                f"Ranked performance/robustness panel '{panel_title}' is empty."
            )
        plot_df, color_col = _ranked_performance_plot_df(
            panel_df,
            dataset_col=dataset_col,
            model_col=model_col,
            arch_col=arch_col,
            perf_col=perf_col,
            robust_col=robust_col,
            robust_semantics=y_semantics,
        )
        panel_plot_dfs.append((panel_title, plot_df, color_col))

    color_col = panel_plot_dfs[0][2]
    if any(panel_color_col != color_col for _, _, panel_color_col in panel_plot_dfs[1:]):
        raise ValueError("Ranked performance/robustness panels disagree on color grouping.")
    color_label = "Architecture" if color_col == arch_col else "Model"
    color_values = sorted(
        {
            str(value).strip()
            for _, plot_df, _ in panel_plot_dfs
            for value in plot_df[color_col].astype(str)
        }
    )
    color_map = _ranked_performance_color_map(color_values)
    x_range, y_range = _ranked_performance_axes(
        plot_dfs=[plot_df for _, plot_df, _ in panel_plot_dfs],
        robust_semantics=y_semantics,
    )
    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.12,
        subplot_titles=[panel_title for panel_title, _, _ in panel_plot_dfs],
    )
    hovertemplate = _ranked_performance_hovertemplate(
        model_col=model_col,
        color_label=color_label,
        y_label=y_title or y_semantics.display_label,
    )
    shown_legend_groups: set[str] = set()
    pareto_legend_shown = False

    for col_idx, (_, plot_df, _) in enumerate(panel_plot_dfs, start=1):
        for group_value in color_values:
            group_df = plot_df.loc[
                plot_df[color_col].astype(str).str.strip() == group_value
            ].sort_values(model_col)
            if group_df.empty:
                continue
            showlegend = group_value not in shown_legend_groups
            fig.add_trace(
                go.Scatter(
                    x=group_df["avg_perf_rank"],
                    y=group_df["mean_metric_value"],
                    mode="markers+text",
                    text=group_df[model_col],
                    textposition="top center",
                    textfont=dict(size=_PUB_TICK_SIZE),
                    cliponaxis=False,
                    marker=dict(
                        size=_PUB_MARKER_SIZE,
                        color=color_map[group_value],
                    ),
                    name=group_value,
                    legendgroup=group_value,
                    showlegend=showlegend,
                    customdata=np.column_stack(
                        [
                            group_df[model_col].astype(str),
                            group_df[color_col].astype(str),
                            group_df["n_datasets"].astype(int),
                        ]
                    ),
                    hovertemplate=hovertemplate,
                ),
                row=1,
                col=col_idx,
            )
            shown_legend_groups.add(group_value)
        pareto_df = plot_df.loc[plot_df["_pareto"]].sort_values("avg_perf_rank")
        if not pareto_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=pareto_df["avg_perf_rank"],
                    y=pareto_df["mean_metric_value"],
                    mode="lines",
                    line=_PARETO_LINE,
                    name="Pareto frontier",
                    legendgroup="pareto_frontier",
                    showlegend=not pareto_legend_shown,
                    hoverinfo="skip",
                ),
                row=1,
                col=col_idx,
            )
            pareto_legend_shown = True
        fig.update_xaxes(
            title_text=(
                x_title
                or f"Average {_pretty_metric_title(perf_col, include_mean_hint=False)} Rank"
            ),
            title_font=dict(size=_PUB_FONT["size"]),
            dtick=1,
            range=x_range,
            ticks="outside",
            ticklen=4,
            tickwidth=1,
            tickfont=dict(size=_PUB_TICK_SIZE),
            showline=True,
            mirror=True,
            zeroline=False,
            showgrid=True,
            gridcolor=_GRID_COLOR,
            row=1,
            col=col_idx,
        )
        fig.update_yaxes(
            title_text=(y_title or y_semantics.display_label) if col_idx == 1 else None,
            title_font=dict(size=_PUB_FONT["size"]),
            range=y_range,
            ticks="outside",
            ticklen=4,
            tickwidth=1,
            tickfont=dict(size=_PUB_TICK_SIZE),
            showline=True,
            mirror=True,
            zeroline=False,
            showgrid=True,
            gridcolor=_GRID_COLOR,
            row=1,
            col=col_idx,
        )

    for annotation in fig.layout.annotations:
        annotation.font = dict(size=_PUB_FONT["size"], family=_FONT_FAMILY)

    fig.update_layout(
        template="plotly_white",
        width=_TEXT_WIDTH_PX,
        height=_PANEL_HEIGHT_PX,
        autosize=False,
        legend_title_text=color_label,
        legend=dict(font=dict(size=_PUB_LEGEND_SIZE)),
        margin=dict(l=10, r=10, t=28, b=10),
        font=_PUB_FONT,
    )
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=1,
        y=-0.04,
        text="Better \u2192",
        showarrow=False,
        xanchor="right",
        yanchor="top",
        font=dict(size=_PUB_ANNOTATION_SIZE),
    )
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=-0.02,
        y=0.92,
        text=_axis_better_annotation(y_semantics, axis="y"),
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        xshift=-8,
        font=dict(size=_PUB_ANNOTATION_SIZE),
    )
    return fig


def plot_pareto_dataset_panels(
    panel_frames: Sequence[tuple[str, pd.DataFrame]],
    *,
    perf_col: str,
    robust_col: str,
    x_semantics: PlotSemanticsRecord,
    y_semantics: PlotSemanticsRecord,
    model_col: str = "model",
    arch_col: str | None = None,
    perf_lower_is_better: bool = True,
    x_title: str | None = None,
    y_title: str | None = None,
    show_group_legend: bool = False,
    show_pareto_legend: bool = True,
) -> go.Figure:
    """Paper-tier subplot grid with one raw-performance-vs-robustness Pareto panel per dataset."""
    x_record = _require_plot_semantics_record(
        x_semantics,
        context="plot_pareto_dataset_panels x-axis",
    )
    y_record = _require_plot_semantics_record(
        y_semantics,
        context="plot_pareto_dataset_panels y-axis",
    )
    perf_minimize = _axis_is_minimize(x_record)
    if bool(perf_lower_is_better) != perf_minimize:
        raise ValueError(
            "plot_pareto_dataset_panels received inconsistent performance direction "
            f"metadata: perf_lower_is_better={perf_lower_is_better} but "
            f"x_semantics.direction={x_record.direction!r}."
        )
    n_panels = len(panel_frames)
    if n_panels < 1 or n_panels > 4:
        raise ValueError(
            "Dataset Pareto panel plot expects between 1 and 4 panels, "
            f"got {n_panels}."
        )
    for panel_title, panel_df in panel_frames:
        if panel_df.empty:
            raise ValueError(
                f"Dataset Pareto panel '{panel_title}' is empty."
            )

    color_col = model_col
    if arch_col is not None:
        for _, panel_df in panel_frames:
            if arch_col not in panel_df.columns:
                raise ValueError(
                    f"Dataset Pareto panel plot: missing grouping column '{arch_col}'."
                )
        color_col = arch_col
    color_label = "Architecture" if color_col == arch_col else "Model"

    color_values = sorted(
        {
            str(v).strip()
            for _, pdf in panel_frames
            for v in pdf[color_col].astype(str)
        }
    )
    color_map = _ranked_performance_color_map(color_values)

    # Shared y-axis range across all panels.
    all_robust = np.concatenate(
        [pdf[robust_col].to_numpy(dtype=float) for _, pdf in panel_frames]
    )
    y_range = list(_axis_range_from_semantics(all_robust, y_record))

    n_rows, n_cols = 1, n_panels

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        shared_yaxes=True,
        horizontal_spacing=0.04 if n_cols >= 3 else 0.12,
        vertical_spacing=0.0,
        subplot_titles=[title for title, _ in panel_frames],
    )

    shown_legend_groups: set[str] = set()
    pareto_legend_shown = not show_pareto_legend

    for panel_idx, (_, panel_df) in enumerate(panel_frames):
        row = panel_idx // n_cols + 1
        col = panel_idx % n_cols + 1
        df = panel_df.copy()
        df["_pareto"] = _pareto_mask(
            df,
            [perf_col, robust_col],
            [perf_minimize, _axis_is_minimize(y_record)],
        )

        for group_value in color_values:
            group_df = df.loc[
                df[color_col].astype(str).str.strip() == group_value
            ].sort_values(model_col)
            if group_df.empty:
                continue
            showlegend = group_value not in shown_legend_groups
            fig.add_trace(
                go.Scatter(
                    x=group_df[perf_col],
                    y=group_df[robust_col],
                    mode="markers+text",
                    text=group_df[model_col],
                    textposition="top center",
                    textfont=dict(size=_PUB_TICK_SIZE - 1),
                    cliponaxis=False,
                    marker=dict(
                        size=_PUB_MARKER_SIZE,
                        color=color_map[group_value],
                    ),
                    name=group_value,
                    legendgroup=group_value,
                    showlegend=showlegend and show_group_legend,
                    hovertemplate=(
                        f"{model_col}: %{{text}}<br>"
                        f"{color_label}: {group_value}<br>"
                        f"{perf_col}: %{{x:.4f}}<br>"
                        f"{robust_col}: %{{y:.3f}}<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )
            shown_legend_groups.add(group_value)

        pf = df.loc[df["_pareto"]].sort_values(
            perf_col, ascending=perf_lower_is_better,
        )
        if not pf.empty:
            fig.add_trace(
                go.Scatter(
                    x=pf[perf_col],
                    y=pf[robust_col],
                    mode="lines",
                    line=_PARETO_LINE,
                    name="Pareto frontier",
                    legendgroup="pareto_frontier",
                    showlegend=not pareto_legend_shown,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )
            pareto_legend_shown = True

        # Per-panel x range (dataset-local performance scale).
        x_vals = df[perf_col].to_numpy(dtype=float)
        x_range = list(
            _axis_range_from_semantics(
                x_vals,
                x_record,
            )
        )

        is_bottom_row = row == n_rows
        is_left_col = col == 1
        fig.update_xaxes(
            title_text=(x_title or x_record.display_label) if is_bottom_row else None,
            title_font=dict(size=_PUB_FONT["size"]),
            range=x_range,
            ticks="outside",
            ticklen=4,
            tickwidth=1,
            tickfont=dict(size=_PUB_TICK_SIZE),
            showline=True,
            mirror=True,
            zeroline=False,
            showgrid=True,
            gridcolor=_GRID_COLOR,
            row=row,
            col=col,
        )
        fig.update_yaxes(
            title_text=(y_title or y_record.display_label) if is_left_col else None,
            title_font=dict(size=_PUB_FONT["size"]),
            range=y_range,
            ticks="outside",
            ticklen=4,
            tickwidth=1,
            tickfont=dict(size=_PUB_TICK_SIZE),
            showline=True,
            mirror=True,
            zeroline=False,
            showgrid=True,
            gridcolor=_GRID_COLOR,
            row=row,
            col=col,
        )

    # Style subplot titles.
    for annotation in fig.layout.annotations:
        annotation.font = dict(size=_PUB_FONT["size"], family=_FONT_FAMILY)

    fig.update_layout(
        template="plotly_white",
        width=(
            _TEXT_WIDTH_PX
            if n_cols <= 2
            else int(_TEXT_WIDTH_PX * (n_cols / 2.0))
        ),
        height=_SINGLE_HEIGHT_PX,
        autosize=False,
        legend_title_text=color_label if show_group_legend else None,
        legend=dict(font=dict(size=_PUB_LEGEND_SIZE)),
        margin=dict(l=10, r=10, t=28, b=28),
        font=_PUB_FONT,
    )
    return fig


def plot_heatmap(
    data_df: pd.DataFrame,
    title: str,
    xlabel: str = "",
    ylabel: str = "",
    cbar_label: str = "Correlation",
    text_format: str = ".2f",
    color_scale: str = "RdYlGn_r",
    color_midpoint: float | None = 0,
    color_range: list[float] | None = [-1, 1],
    p_values_df: pd.DataFrame | None = None,
) -> go.Figure:
    """
    Plots a heatmap with options for color scale and significance highlighting.
    """
    custom_text = data_df.map(lambda x: f"{x:{text_format}}")
    if p_values_df is not None:
        significant = p_values_df < 0.05
        custom_text = custom_text + significant.map(lambda x: "*" if x else "")

    fig = px.imshow(
        data_df,
        text_auto=False,  # We use a custom text trace
        aspect="auto",
        labels=dict(x=xlabel, y=ylabel, color=cbar_label),
        title=title,
        color_continuous_scale=color_scale,
        color_continuous_midpoint=color_midpoint,
        range_color=color_range,
    )
    fig.update_traces(text=custom_text, texttemplate="%{text}")
    fig.update_layout(
        width=800, height=600,
        title=dict(y=0.95, x=0.5, xanchor="center", yanchor="top"),
        font=_DIAG_FONT,
        title_font=dict(size=_DIAG_TITLE_SIZE),
        margin=dict(l=5, t=50, b=5, r=5),
        template="plotly_white",
    )
    return fig


def plot_robustness_vs_performance(
    metric_name: str,
    model_name: str,
    pert_names: list[str],
    metric_values: list[float],
    performance_values: list[float],
) -> go.Figure:
    """
    Plots a scatter of a robustness metric vs. clean performance for each perturbation type.
    """
    plot_df = pd.DataFrame({
        "pert_name": pert_names,
        "performance": performance_values,
        "metric": metric_values,
    })

    fig = px.scatter(
        plot_df,
        x="performance",
        y="metric",
        text="pert_name",
        labels={
            "performance": "Mean Clean Error",
            "metric": metric_name
        },
        title=f"{metric_name} vs. Clean Error for {model_name}",
    )

    fig.update_traces(textposition='top center')
    fig.update_layout(
        width=600, height=450,
        title=dict(y=0.95, x=0.5, xanchor="center", yanchor="top"),
        font=_DIAG_FONT,
        title_font=dict(size=_DIAG_TITLE_SIZE),
        margin=dict(l=5, t=50, b=5, r=5),
        template="plotly_white",
    )
    return fig


def plot_perturbed_vs_clean_error(
    df: pd.DataFrame,
    *,
    model_col: str,
    clean_col: str,
    pert_col: str,
    color_col: str | None = None,
    legend_title: str | None = None,
    title: str = "Perturbed vs Clean Error",
    x_title: str | None = None,
    y_title: str | None = None,
    marker_symbol: str = "circle",
) -> go.Figure:
    base_cols = [model_col, clean_col, pert_col]
    if color_col and color_col in df.columns and color_col not in base_cols:
        base_cols.append(color_col)
    plot_df = df[base_cols].dropna(subset=[clean_col, pert_col]).copy()
    if plot_df.empty:
        fig = go.Figure()
        fig.update_layout(title=f"{title} (No Data)")
        return fig

    if color_col and color_col in df.columns:
        plot_df[color_col] = df.loc[plot_df.index, color_col]

    scatter_kwargs = dict(
        data_frame=plot_df,
        x=clean_col,
        y=pert_col,
        text=model_col,
        template="plotly_white",
        hover_data=None,
    )

    if color_col and color_col in plot_df.columns:
        scatter_kwargs["color"] = color_col

    fig = px.scatter(**scatter_kwargs)
    fig.update_traces(textposition="top center", cliponaxis=False)
    fig.update_traces(marker=dict(symbol=marker_symbol), selector=dict(mode="markers+text"))

    fig.update_traces(hoverinfo="skip", hovertemplate=None)

    combined = pd.concat(
        [plot_df[clean_col].astype(float), plot_df[pert_col].astype(float)],
        ignore_index=True,
    )
    min_val = float(np.nanmin(combined)) if not combined.empty else 0.0
    max_val = float(np.nanmax(combined)) if not combined.empty else 1.0
    span = max_val - min_val
    pad = 0.05 * span if span > 0 else 0.05 * max(abs(max_val), 1.0)
    lower = min_val - pad
    upper = max_val + pad

    fig.add_trace(
        go.Scatter(
            x=[lower, upper],
            y=[lower, upper],
            mode="lines",
            line=dict(color="grey", dash="dash"),
            name="Parity",
            hoverinfo="skip",
            showlegend=True,
        )
    )

    fig.update_xaxes(
        title_text=x_title or "Clean Error",
        range=[lower, upper],
        ticks="outside",
        ticklen=6,
        tickwidth=1,
        showline=True,
        mirror=True,
        zeroline=False,
        showgrid=True,
        gridcolor=_GRID_COLOR,
    )
    fig.update_yaxes(
        title_text=y_title or "Perturbed Error",
        range=[lower, upper],
        ticks="outside",
        ticklen=6,
        tickwidth=1,
        showline=True,
        mirror=True,
        zeroline=False,
        showgrid=True,
        gridcolor=_GRID_COLOR,
    )

    if legend_title is None and color_col and color_col in plot_df.columns:
        legend_title = "Architecture" if color_col == "architecture_family" else color_col.replace("_", " ").title()

    fig.update_layout(
        width=600,
        height=450,
        title=title,
        legend_title_text=legend_title,
        margin=dict(l=5, t=50, b=5, r=5),
        font=_DIAG_FONT,
        title_font=dict(size=_DIAG_TITLE_SIZE),
    )
    return fig


def plot_perturbed_vs_clean_error_panels(
    panel_frames: Sequence[tuple[str, pd.DataFrame]],
    *,
    model_col: str,
    clean_col: str,
    pert_col: str,
    x_semantics: PlotSemanticsRecord,
    y_semantics: PlotSemanticsRecord,
    color_col: str | None = None,
    x_title: str | None = None,
    y_title: str | None = None,
    marker_symbol: str = "circle",
    show_group_legend: bool = False,
) -> go.Figure:
    if not panel_frames:
        raise ValueError("Perturbed-vs-clean panel plot requires at least one panel.")
    if len(panel_frames) > 4:
        raise ValueError(
            "Perturbed-vs-clean panel plot supports at most 4 panels, "
            f"got {len(panel_frames)}."
        )

    x_record = _require_plot_semantics_record(
        x_semantics,
        context="plot_perturbed_vs_clean_error_panels x-axis",
    )
    y_record = _require_plot_semantics_record(
        y_semantics,
        context="plot_perturbed_vs_clean_error_panels y-axis",
    )
    if not _axis_is_minimize(x_record) or not _axis_is_minimize(y_record):
        raise ValueError(
            "plot_perturbed_vs_clean_error_panels expects lower-is-better semantics "
            "for both clean and perturbed error axes."
        )

    color_key = color_col if color_col is not None else model_col
    color_values: set[str] = set()
    prepared_frames: list[tuple[str, pd.DataFrame]] = []
    for panel_title, panel_df in panel_frames:
        require_dataframe_columns(
            panel_df,
            {model_col, clean_col, pert_col},
            context="Perturbed-vs-clean panel input",
        )
        plot_cols = [model_col, clean_col, pert_col]
        if color_col is not None:
            if color_col not in panel_df.columns:
                raise ValueError(
                    "Perturbed-vs-clean panel plot is missing color column "
                    f"'{color_col}'."
                )
            plot_cols.append(color_col)
        plot_df = panel_df[plot_cols].dropna(subset=[clean_col, pert_col]).copy()
        if plot_df.empty:
            raise ValueError(
                f"Perturbed-vs-clean panel '{panel_title}' is empty after metric filtering."
            )
        plot_df[model_col] = plot_df[model_col].astype(str).str.strip()
        if color_key not in plot_df.columns:
            plot_df[color_key] = plot_df[model_col]
        plot_df[color_key] = plot_df[color_key].astype(str).str.strip()
        color_values.update(plot_df[color_key].tolist())
        prepared_frames.append((panel_title, plot_df))

    color_map = _ranked_performance_color_map(sorted(color_values))
    n_panels = len(prepared_frames)
    fig = make_subplots(
        rows=1,
        cols=n_panels,
        shared_yaxes=False,
        horizontal_spacing=0.04 if n_panels >= 3 else 0.12,
        subplot_titles=[title for title, _ in prepared_frames],
    )

    shown_legend_groups: set[str] = set()
    for panel_idx, (_, panel_df) in enumerate(prepared_frames):
        col = panel_idx + 1
        for group_value in sorted(panel_df[color_key].unique()):
            group_df = panel_df.loc[
                panel_df[color_key].astype(str).str.strip() == group_value
            ].sort_values(model_col)
            if group_df.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=group_df[clean_col],
                    y=group_df[pert_col],
                    mode="markers+text",
                    text=group_df[model_col],
                    textposition="top center",
                    textfont=dict(size=_PUB_TICK_SIZE - 1),
                    cliponaxis=False,
                    marker=dict(
                        size=_PUB_MARKER_SIZE,
                        color=color_map[group_value],
                        symbol=marker_symbol,
                    ),
                    name=group_value,
                    legendgroup=group_value,
                    showlegend=show_group_legend and group_value not in shown_legend_groups,
                    hovertemplate=(
                        f"{model_col}: %{{text}}<br>"
                        f"{clean_col}: %{{x:.4f}}<br>"
                        f"{pert_col}: %{{y:.4f}}<extra></extra>"
                    ),
                ),
                row=1,
                col=col,
            )
            shown_legend_groups.add(group_value)

        panel_min, panel_max = _numeric_axis_range(
            pd.concat(
                [
                    panel_df[clean_col].astype(float),
                    panel_df[pert_col].astype(float),
                ],
                ignore_index=True,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[panel_min, panel_max],
                y=[panel_min, panel_max],
                mode="lines",
                line=dict(color="grey", dash="dash"),
                name="Parity",
                hoverinfo="skip",
                showlegend=False,
            ),
            row=1,
            col=col,
        )

        _, yref = _subplot_axis_refs(row=1, col=col, cols=n_panels)
        fig.update_xaxes(
            title_text=x_title or x_record.display_label,
            title_font=dict(size=_PUB_FONT["size"]),
            range=[panel_min, panel_max],
            ticks="outside",
            ticklen=4,
            tickwidth=1,
            tickfont=dict(size=_PUB_TICK_SIZE),
            showline=True,
            mirror=True,
            zeroline=False,
            showgrid=True,
            gridcolor=_GRID_COLOR,
            scaleanchor=yref,
            scaleratio=1,
            constrain="domain",
            row=1,
            col=col,
        )
        fig.update_yaxes(
            title_text=(y_title or y_record.display_label) if col == 1 else None,
            title_font=dict(size=_PUB_FONT["size"]),
            range=[panel_min, panel_max],
            ticks="outside",
            ticklen=4,
            tickwidth=1,
            tickfont=dict(size=_PUB_TICK_SIZE),
            showline=True,
            mirror=True,
            zeroline=False,
            showgrid=True,
            gridcolor=_GRID_COLOR,
            row=1,
            col=col,
        )

    for annotation in fig.layout.annotations:
        annotation.font = dict(size=_PUB_FONT["size"], family=_FONT_FAMILY)

    fig.update_layout(
        template="plotly_white",
        width=(
            _TEXT_WIDTH_PX
            if n_panels <= 2
            else int(_TEXT_WIDTH_PX * (n_panels / 2.0))
        ),
        height=_SINGLE_HEIGHT_PX,
        autosize=False,
        legend_title_text=color_key.replace("_", " ").title() if show_group_legend else None,
        legend=dict(font=dict(size=_PUB_LEGEND_SIZE)),
        margin=dict(l=10, r=10, t=28, b=28),
        font=_PUB_FONT,
    )
    return fig


def plot_error_distribution_overview(
    df: pd.DataFrame,
    *,
    dataset: str,
    facet_col: str = "model_variant",
    value_col: str = "error_value",
    error_kind_col: str = "error_kind",
    max_facets: int = 24,
) -> go.Figure:
    """Plot raw error-distribution violins per model variant for one dataset."""
    if not str(dataset).strip():
        raise ValueError("plot_error_distribution_overview requires non-empty dataset.")
    if max_facets <= 0:
        raise ValueError("plot_error_distribution_overview requires max_facets > 0.")

    required_cols = {facet_col, value_col, error_kind_col}
    missing_cols = sorted(required_cols - set(df.columns))
    if missing_cols:
        raise ValueError(
            "plot_error_distribution_overview is missing required columns: "
            f"{missing_cols}."
        )
    if df.empty:
        raise ValueError("plot_error_distribution_overview received an empty dataframe.")

    plot_df = df.copy()
    facet_values = plot_df[facet_col].astype(str).str.strip()
    if (facet_values == "").any():
        raise ValueError(
            f"plot_error_distribution_overview has empty facet values in '{facet_col}'."
        )
    plot_df[facet_col] = facet_values

    error_kinds = plot_df[error_kind_col].astype(str).str.strip()
    if (error_kinds == "").any():
        raise ValueError(
            f"plot_error_distribution_overview has empty error_kind values in '{error_kind_col}'."
        )
    plot_df[error_kind_col] = error_kinds
    expected_error_kinds = {"err_clean", "err_pert"}
    unexpected_error_kinds = sorted(
        set(plot_df[error_kind_col].unique()) - expected_error_kinds
    )
    if unexpected_error_kinds:
        raise ValueError(
            "plot_error_distribution_overview has unsupported error kinds: "
            f"{unexpected_error_kinds}."
        )

    value_numeric = require_numeric_series(
        plot_df[value_col],
        column_name=value_col,
        context="plot_error_distribution_overview",
        allow_nan=False,
        allow_infinite=False,
    )
    plot_df[value_col] = value_numeric

    unique_facets = sorted(plot_df[facet_col].dropna().unique())
    if len(unique_facets) > max_facets:
        raise ValueError(
            "plot_error_distribution_overview exceeds facet limit: "
            f"{len(unique_facets)} > {max_facets}."
        )

    per_facet_kind_counts = plot_df.groupby(facet_col, dropna=False)[error_kind_col].nunique()
    missing_kind_facets = per_facet_kind_counts[per_facet_kind_counts != 2]
    if not missing_kind_facets.empty:
        examples = [str(name) for name in missing_kind_facets.index[:5]]
        raise ValueError(
            "plot_error_distribution_overview requires both err_clean and err_pert "
            f"for every facet. Invalid facets: {examples}."
        )

    facet_wrap = 4
    fig = px.violin(
        plot_df,
        x=error_kind_col,
        y=value_col,
        color=error_kind_col,
        facet_col=facet_col,
        facet_col_wrap=facet_wrap,
        points=False,
        box=True,
        category_orders={error_kind_col: ["err_clean", "err_pert"]},
        color_discrete_map={
            "err_clean": "#1f77b4",
            "err_pert": "#d62728",
        },
        title=f"{dataset}: Error Distributions by Model Variant",
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    facet_count = int(len(unique_facets))
    n_rows = int(np.ceil(facet_count / float(facet_wrap)))
    fig.update_layout(
        width=max(900, min(1800, 350 * min(facet_wrap, max(1, facet_count)))),
        height=max(500, min(2600, 320 * max(1, n_rows))),
        margin=dict(l=10, r=10, t=60, b=20),
        legend_title_text="Error Kind",
        template="plotly_white",
        font=dict(size=13, family=_FONT_FAMILY),
        title_font=dict(size=_DIAG_TITLE_SIZE_COMPACT),
    )
    fig.update_xaxes(title_text="")
    fig.update_yaxes(title_text="Error")
    return fig


# ---------------------------------------------------------------------------
# Improvement trajectory plots
# ---------------------------------------------------------------------------


def _trajectory_perf_axis_range(
    values: Iterable[float],
    *,
    perf_lower_is_better: bool,
    flip_perf_axis: bool,
) -> list[float]:
    arr = np.asarray(list(values), dtype=float).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise ValueError("Cannot compute trajectory performance axis from empty data.")
    lower = float(np.min(finite))
    upper = float(np.max(finite))
    pad = 0.05 * (upper - lower) if upper > lower else 0.05 * max(abs(upper), 1.0)
    axis_range = [lower - pad, upper + pad]
    if flip_perf_axis and perf_lower_is_better:
        axis_range = [axis_range[1], axis_range[0]]
    return axis_range


def _prepare_trajectory_frame_pair(
    backbone_df: pd.DataFrame,
    improvement_df: pd.DataFrame,
    *,
    perf_col: str,
    robust_col: str,
    dataset_col: str,
    baseline_backbone_col: str,
    improvement_name_col: str,
    method_col: str,
    backbone_col: str,
    require_signature: bool,
    signature_col: str,
    improvement_identity_key_cols: Sequence[str] | None,
    improvement_join_key_cols: Sequence[str] | None,
    context: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline_required_cols = {
        dataset_col,
        baseline_backbone_col,
        perf_col,
        robust_col,
    }
    improvement_required_cols = {
        dataset_col,
        backbone_col,
        method_col,
        improvement_name_col,
        perf_col,
        robust_col,
    }
    if require_signature:
        baseline_required_cols.add(signature_col)
        improvement_required_cols.add(signature_col)
    if improvement_identity_key_cols is not None:
        improvement_required_cols.update(improvement_identity_key_cols)
    if improvement_join_key_cols is not None:
        improvement_required_cols.update(improvement_join_key_cols)

    require_dataframe_columns(
        backbone_df,
        baseline_required_cols,
        context=f"{context}: baseline input",
    )
    require_dataframe_columns(
        improvement_df,
        improvement_required_cols,
        context=f"{context}: improvement input",
    )

    baseline = backbone_df.dropna(subset=[perf_col, robust_col]).copy()
    improvement = improvement_df.dropna(subset=[perf_col, robust_col]).copy()

    baseline_sample_cols = [col for col in (dataset_col, baseline_backbone_col) if col in baseline.columns]
    improvement_sample_cols = [
        col
        for col in (dataset_col, backbone_col, improvement_name_col, method_col)
        if col in improvement.columns
    ]

    if not baseline.empty:
        baseline["_trajectory_dataset"] = require_nonempty_string_series(
            baseline,
            dataset_col,
            context=f"{context}: baseline input",
            sample_cols=baseline_sample_cols,
        )
        baseline["_trajectory_backbone_key"] = require_nonempty_string_series(
            baseline,
            baseline_backbone_col,
            context=f"{context}: baseline input",
            sample_cols=baseline_sample_cols,
        )
        if require_signature:
            baseline["_trajectory_signature"] = require_nonempty_string_series(
                baseline,
                signature_col,
                context=f"{context}: baseline input",
                sample_cols=baseline_sample_cols,
            )

    if not improvement.empty:
        improvement["_trajectory_dataset"] = require_nonempty_string_series(
            improvement,
            dataset_col,
            context=f"{context}: improvement input",
            sample_cols=improvement_sample_cols,
        )
        improvement["_trajectory_backbone_key"] = require_nonempty_string_series(
            improvement,
            backbone_col,
            context=f"{context}: improvement input",
            sample_cols=improvement_sample_cols,
        )
        improvement["_trajectory_group"] = require_nonempty_string_series(
            improvement,
            method_col,
            context=f"{context}: improvement input",
            sample_cols=improvement_sample_cols,
        )
        improvement["_trajectory_variant"] = require_nonempty_string_series(
            improvement,
            improvement_name_col,
            context=f"{context}: improvement input",
            sample_cols=improvement_sample_cols,
        )
        if require_signature:
            improvement["_trajectory_signature"] = require_nonempty_string_series(
                improvement,
                signature_col,
                context=f"{context}: improvement input",
                sample_cols=improvement_sample_cols,
            )

    baseline_join_key_cols = [dataset_col]
    if require_signature:
        baseline_join_key_cols.append(signature_col)
    baseline_join_key_cols.append(baseline_backbone_col)
    if not baseline.empty:
        assert_no_duplicate_rows(
            baseline,
            baseline_join_key_cols,
            context=(
                f"{context}: baseline rows must be unique per {baseline_join_key_cols}"
            ),
        )

    if improvement_identity_key_cols is None:
        improvement_identity_key_cols = [
            dataset_col,
            method_col,
            improvement_name_col,
            backbone_col,
        ]
    else:
        improvement_identity_key_cols = list(improvement_identity_key_cols)
    if improvement_join_key_cols is None:
        improvement_join_key_cols = [
            dataset_col,
            *([signature_col] if require_signature else []),
            method_col,
            improvement_name_col,
            backbone_col,
        ]
    else:
        improvement_join_key_cols = list(improvement_join_key_cols)
    if not improvement.empty:
        assert_no_duplicate_rows(
            improvement,
            improvement_identity_key_cols,
            context=(
                f"{context}: improvement rows must be unique per "
                f"{improvement_identity_key_cols}"
            ),
        )
        assert_no_duplicate_rows(
            improvement,
            improvement_join_key_cols,
            context=(
                f"{context}: improvement rows must be unique per "
                f"{improvement_join_key_cols}"
            ),
        )

    merge_key_cols = ["_trajectory_dataset"]
    if require_signature:
        merge_key_cols.append("_trajectory_signature")
    merge_key_cols.append("_trajectory_backbone_key")
    if not improvement.empty:
        baseline_key_df = baseline[merge_key_cols].drop_duplicates().copy()
        improvement_with_match = improvement.merge(
            baseline_key_df.assign(_trajectory_match=True),
            on=merge_key_cols,
            how="left",
        )
        unmatched_improvement = improvement_with_match.loc[
            improvement_with_match["_trajectory_match"] != True
        ]
        if not unmatched_improvement.empty:
            example_cols = [dataset_col]
            if require_signature:
                example_cols.append(signature_col)
            example_cols.extend(
                [method_col, improvement_name_col, backbone_col]
            )
            example_cols = list(dict.fromkeys(example_cols))
            examples = (
                unmatched_improvement[example_cols]
                .drop_duplicates()
                .head(5)
                .to_dict(orient="records")
            )
            raise ValueError(
                f"{context}: improvement rows do not match baseline trajectory join keys. "
                f"Examples: {examples}."
            )
    merged = improvement.merge(
        baseline[merge_key_cols + [perf_col, robust_col]],
        on=merge_key_cols,
        how="inner",
        suffixes=("_improved", "_baseline"),
    )
    return baseline, improvement, merged


def _trajectory_pareto_df(
    backbone_df: pd.DataFrame,
    improvement_df: pd.DataFrame,
    *,
    perf_col: str,
    robust_col: str,
    perf_lower_is_better: bool,
    robust_minimize: bool,
) -> pd.DataFrame:
    all_points = pd.concat(
        [
            backbone_df[[perf_col, robust_col]],
            improvement_df[[perf_col, robust_col]],
        ],
        ignore_index=True,
    ).dropna(subset=[perf_col, robust_col])
    if all_points.empty:
        return all_points
    all_points = all_points.copy()
    all_points["_pareto"] = _pareto_mask(
        all_points,
        [perf_col, robust_col],
        [perf_lower_is_better, robust_minimize],
    )
    return all_points.loc[all_points["_pareto"]].sort_values(
        perf_col,
        ascending=perf_lower_is_better,
    )


def _trajectory_color_map(
    merged_frames: Sequence[pd.DataFrame],
) -> dict[str, str]:
    group_values = sorted(
        {
            str(value).strip()
            for merged in merged_frames
            for value in merged["_trajectory_group"].astype(str)
        }
    )
    return _ranked_performance_color_map(group_values)


def _subplot_axis_refs(
    *,
    row: int,
    col: int,
    cols: int,
) -> tuple[str, str]:
    axis_idx = (row - 1) * cols + col
    axis_suffix = "" if axis_idx == 1 else str(axis_idx)
    return f"x{axis_suffix}", f"y{axis_suffix}"


def _add_trajectory_pair_traces(
    fig: go.Figure,
    merged: pd.DataFrame,
    *,
    perf_col: str,
    robust_col: str,
    color_map: Mapping[str, str],
    row: int,
    col: int,
    cols: int,
    show_text_labels: bool,
) -> None:
    xref, yref = _subplot_axis_refs(row=row, col=col, cols=cols)
    for _, record in merged.iterrows():
        group_value = str(record["_trajectory_group"])
        color = color_map[group_value]
        method_line = ""
        if "robustness_method" in record.index and pd.notna(record["robustness_method"]):
            method_line = f"Method: {str(record['robustness_method'])}<br>"

        fig.add_trace(
            go.Scatter(
                x=[record[f"{perf_col}_baseline"]],
                y=[record[f"{robust_col}_baseline"]],
                mode="markers+text" if show_text_labels else "markers",
                text=[record["_trajectory_backbone_key"]] if show_text_labels else None,
                textposition="top center",
                textfont=dict(size=_PUB_TICK_SIZE - 1),
                cliponaxis=False,
                marker=dict(
                    size=10,
                    color=color,
                    line=dict(color=color, width=1),
                    symbol="circle",
                ),
                name=f"{group_value} (baseline)",
                legendgroup=f"backbone-{group_value}",
                showlegend=False,
                hovertemplate=(
                    f"<b>{record['_trajectory_backbone_key']}</b> (baseline)<br>"
                    f"{_pretty_metric_title(perf_col, False)}: "
                    f"{record[f'{perf_col}_baseline']:.4f}<br>"
                    f"{_pretty_robust_title(robust_col, False)}: "
                    f"{record[f'{robust_col}_baseline']:.4f}<br>"
                    "<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=[record[f"{perf_col}_improved"]],
                y=[record[f"{robust_col}_improved"]],
                mode="markers",
                marker=dict(
                    size=10,
                    color="white",
                    line=dict(color=color, width=2),
                    symbol="circle",
                ),
                name=f"{group_value} (robustness)",
                legendgroup=f"backbone-{group_value}",
                showlegend=False,
                hovertemplate=(
                    f"<b>{record['_trajectory_variant']}</b><br>"
                    f"Backbone: {record['_trajectory_backbone_key']}<br>"
                    f"{method_line}"
                    f"{_pretty_metric_title(perf_col, False)}: "
                    f"{record[f'{perf_col}_improved']:.4f}<br>"
                    f"{_pretty_robust_title(robust_col, False)}: "
                    f"{record[f'{robust_col}_improved']:.4f}<br>"
                    f"Δ Perf: "
                    f"{record[f'{perf_col}_improved'] - record[f'{perf_col}_baseline']:.4f}<br>"
                    f"Δ Robust: "
                    f"{record[f'{robust_col}_improved'] - record[f'{robust_col}_baseline']:.4f}<br>"
                    "<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )
        fig.add_annotation(
            x=record[f"{perf_col}_improved"],
            y=record[f"{robust_col}_improved"],
            ax=record[f"{perf_col}_baseline"],
            ay=record[f"{robust_col}_baseline"],
            xref=xref,
            yref=yref,
            axref=xref,
            ayref=yref,
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor=color,
            opacity=0.6,
        )


def _add_trajectory_legend_traces(
    fig: go.Figure,
    *,
    color_map: Mapping[str, str],
    show_group_legend: bool,
    baseline_output_label: str,
    robustness_output_label: str,
    show_pareto: bool,
    row: int,
    col: int,
) -> None:
    if show_group_legend:
        for group_value in sorted(color_map):
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(
                        size=10,
                        color=color_map[group_value],
                        line=dict(color=color_map[group_value], width=1),
                        symbol="circle",
                    ),
                    name=group_value,
                    legendgroup=f"backbone-{group_value}",
                    showlegend=True,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )
    else:
        baseline_color = next(iter(color_map.values()), _OKABE_ITO[0])
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(
                    size=10,
                    color=baseline_color,
                    line=dict(color=baseline_color, width=1),
                    symbol="circle",
                ),
                name=baseline_output_label,
                legendgroup="trajectory_baseline_output",
                showlegend=True,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(
                size=10,
                color="white",
                line=dict(color="#7f7f7f", width=2),
                symbol="circle",
            ),
            name=str(robustness_output_label),
            legendgroup="trajectory_output",
            showlegend=True,
            hoverinfo="skip",
        ),
        row=row,
        col=col,
    )
    if show_pareto:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line=_PARETO_LINE,
                name="Pareto frontier",
                legendgroup="pareto_frontier",
                showlegend=True,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )


def _validate_trajectory_panel_spec(
    panel_spec: Sequence[tuple[str, str]],
    *,
    context: str,
) -> list[tuple[str, str]]:
    if not panel_spec:
        raise ValueError(f"{context}: panel spec must be non-empty.")
    normalized_spec: list[tuple[str, str]] = []
    seen_dataset_keys: set[str] = set()
    for dataset_key, display_title in panel_spec:
        dataset_text = str(dataset_key).strip()
        title_text = str(display_title).strip()
        if not dataset_text:
            raise ValueError(f"{context}: panel spec contains an empty dataset key.")
        if not title_text:
            raise ValueError(
                f"{context}: panel spec for dataset '{dataset_text}' has an empty title."
            )
        if dataset_text in seen_dataset_keys:
            raise ValueError(
                f"{context}: panel spec contains duplicate dataset key '{dataset_text}'."
            )
        seen_dataset_keys.add(dataset_text)
        normalized_spec.append((dataset_text, title_text))
    if len(normalized_spec) > 4:
        raise ValueError(
            f"{context}: panel spec supports at most 4 datasets, got {len(normalized_spec)}."
        )
    return normalized_spec


def _trajectory_subplot_geometry(
    n_panels: int,
) -> tuple[int, int, int, float, float]:
    if n_panels == 1:
        return 1, 1, _SINGLE_HEIGHT_PX, 0.0, 0.0
    if n_panels == 2:
        return 1, 2, _PANEL_HEIGHT_PX, 0.12, 0.0
    if n_panels in {3, 4}:
        return 2, 2, _TEXT_WIDTH_PX, 0.12, 0.18
    raise ValueError(
        f"Trajectory subplot layout requires 1-4 panels, got {n_panels}."
    )


def plot_improvement_trajectory_subplots(
    backbone_df: pd.DataFrame,
    improvement_df: pd.DataFrame,
    *,
    panel_spec: Sequence[tuple[str, str]],
    perf_col: str,
    robust_col: str,
    x_semantics: PlotSemanticsRecord,
    y_semantics: PlotSemanticsRecord,
    dataset_col: str = "dataset",
    model_col: str = "model",
    method_col: str = "robustness_method",
    backbone_col: str = "backbone_architecture",
    baseline_backbone_col: str | None = None,
    improvement_name_col: str | None = None,
    perf_lower_is_better: bool = True,
    flip_perf_axis: bool = False,
    show_pareto: bool = True,
    show_group_legend: bool = False,
    show_text_labels: bool = True,
    robustness_output_label: str = "Robustness method output",
    x_title: str | None = None,
    y_title: str | None = None,
    require_signature: bool = True,
    signature_col: str = "data_config_signature",
    improvement_identity_key_cols: Sequence[str] | None = None,
    improvement_join_key_cols: Sequence[str] | None = None,
    context: str = "plot_improvement_trajectory_subplots",
) -> go.Figure:
    x_record = _require_plot_semantics_record(
        x_semantics,
        context="plot_improvement_trajectory_subplots x-axis",
    )
    y_record = _require_plot_semantics_record(
        y_semantics,
        context="plot_improvement_trajectory_subplots y-axis",
    )
    perf_minimize = _axis_is_minimize(x_record)
    if bool(perf_lower_is_better) != perf_minimize:
        raise ValueError(
            "plot_improvement_trajectory_subplots received inconsistent performance "
            f"direction metadata: perf_lower_is_better={perf_lower_is_better} but "
            f"x_semantics.direction={x_record.direction!r}."
        )
    baseline_backbone_col = baseline_backbone_col or model_col
    improvement_name_col = improvement_name_col or model_col
    validated_panel_spec = _validate_trajectory_panel_spec(
        panel_spec,
        context=context,
    )
    panel_dataset_keys = [dataset_key for dataset_key, _ in validated_panel_spec]
    allowed_dataset_keys = set(panel_dataset_keys)

    require_dataframe_columns(
        backbone_df,
        {dataset_col},
        context=f"{context}: baseline input",
    )
    require_dataframe_columns(
        improvement_df,
        {dataset_col},
        context=f"{context}: improvement input",
    )

    baseline_dataset_values = set(
        backbone_df[dataset_col].dropna().astype(str).str.strip().tolist()
    )
    improvement_dataset_values = set(
        improvement_df[dataset_col].dropna().astype(str).str.strip().tolist()
    )
    unexpected_dataset_values = sorted(
        (baseline_dataset_values | improvement_dataset_values) - allowed_dataset_keys
    )
    if unexpected_dataset_values:
        raise ValueError(
            f"{context}: input includes datasets outside the panel spec {unexpected_dataset_values}."
        )

    present_dataset_keys = [
        dataset_key
        for dataset_key in panel_dataset_keys
        if dataset_key in baseline_dataset_values or dataset_key in improvement_dataset_values
    ]
    if not present_dataset_keys:
        raise ValueError(
            f"{context}: no panel datasets remain after filtering to the panel spec."
        )

    panel_data: list[tuple[str, str, pd.DataFrame, pd.DataFrame, pd.DataFrame]] = []
    for dataset_key, panel_title in validated_panel_spec:
        if dataset_key not in present_dataset_keys:
            continue
        panel_backbone_df = backbone_df.loc[
            backbone_df[dataset_col].astype(str).str.strip() == dataset_key
        ].copy()
        panel_improvement_df = improvement_df.loc[
            improvement_df[dataset_col].astype(str).str.strip() == dataset_key
        ].copy()
        if panel_backbone_df.empty:
            raise ValueError(
                f"{context}: dataset '{dataset_key}' has no baseline rows."
            )
        if panel_improvement_df.empty:
            raise ValueError(
                f"{context}: dataset '{dataset_key}' has no improvement rows."
            )
        prepared_backbone_df, prepared_improvement_df, merged = _prepare_trajectory_frame_pair(
            panel_backbone_df,
            panel_improvement_df,
            perf_col=perf_col,
            robust_col=robust_col,
            dataset_col=dataset_col,
            baseline_backbone_col=baseline_backbone_col,
            improvement_name_col=improvement_name_col,
            method_col=method_col,
            backbone_col=backbone_col,
            require_signature=require_signature,
            signature_col=signature_col,
            improvement_identity_key_cols=improvement_identity_key_cols,
            improvement_join_key_cols=improvement_join_key_cols,
            context=f"{context}: dataset '{dataset_key}'",
        )
        if prepared_backbone_df.empty:
            raise ValueError(
                f"{context}: dataset '{dataset_key}' has no baseline rows after metric filtering."
            )
        if prepared_improvement_df.empty:
            raise ValueError(
                f"{context}: dataset '{dataset_key}' has no improvement rows after metric filtering."
            )
        if merged.empty:
            raise ValueError(
                f"{context}: dataset '{dataset_key}' has no merged trajectory rows."
            )
        panel_data.append(
            (
                dataset_key,
                panel_title,
                prepared_backbone_df,
                prepared_improvement_df,
                merged,
            )
        )

    n_panels = len(panel_data)
    rows, cols, height_px, horizontal_spacing, vertical_spacing = (
        _trajectory_subplot_geometry(n_panels)
    )
    subplot_titles = [panel_title for _, panel_title, _, _, _ in panel_data]
    if rows * cols > n_panels:
        subplot_titles.extend([""] * (rows * cols - n_panels))
    fig = make_subplots(
        rows=rows,
        cols=cols,
        shared_yaxes=True,
        horizontal_spacing=horizontal_spacing,
        vertical_spacing=vertical_spacing,
        subplot_titles=subplot_titles,
    )

    color_map = _trajectory_color_map([merged for _, _, _, _, merged in panel_data])
    all_robust_values = np.concatenate(
        [
            np.concatenate(
                [
                    merged[f"{robust_col}_baseline"].to_numpy(dtype=float),
                    merged[f"{robust_col}_improved"].to_numpy(dtype=float),
                ]
            )
            for _, _, _, _, merged in panel_data
        ]
    )
    y_range = list(_axis_range_from_semantics(all_robust_values, y_record))

    pareto_legend_needed = False
    for panel_idx, (_, _, panel_backbone_df, panel_improvement_df, merged) in enumerate(panel_data):
        row = panel_idx // cols + 1
        col = panel_idx % cols + 1
        _add_trajectory_pair_traces(
            fig,
            merged,
            perf_col=perf_col,
            robust_col=robust_col,
            color_map=color_map,
            row=row,
            col=col,
            cols=cols,
            show_text_labels=show_text_labels,
        )
        if show_pareto:
            pareto_df = _trajectory_pareto_df(
                panel_backbone_df,
                panel_improvement_df,
                perf_col=perf_col,
                robust_col=robust_col,
                perf_lower_is_better=perf_minimize,
                robust_minimize=_axis_is_minimize(y_record),
            )
            if not pareto_df.empty:
                pareto_legend_needed = True
                fig.add_trace(
                    go.Scatter(
                        x=pareto_df[perf_col],
                        y=pareto_df[robust_col],
                        mode="lines",
                        line=_PARETO_LINE,
                        name="Pareto frontier",
                        legendgroup="pareto_frontier",
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=row,
                    col=col,
                )

        x_values = np.concatenate(
            [
                merged[f"{perf_col}_baseline"].to_numpy(dtype=float),
                merged[f"{perf_col}_improved"].to_numpy(dtype=float),
            ]
        )
        x_range = _trajectory_perf_axis_range(
            x_values,
            perf_lower_is_better=perf_minimize,
            flip_perf_axis=flip_perf_axis,
        )
        is_bottom_row = row == rows
        is_left_col = col == 1
        fig.update_xaxes(
            title_text=(
                x_title or x_record.display_label
            ) if is_bottom_row else None,
            title_font=dict(size=_PUB_FONT["size"]),
            range=x_range,
            ticks="outside",
            ticklen=4,
            tickwidth=1,
            tickfont=dict(size=_PUB_TICK_SIZE),
            showline=True,
            mirror=True,
            zeroline=False,
            showgrid=True,
            gridcolor=_GRID_COLOR,
            row=row,
            col=col,
        )
        fig.update_yaxes(
            title_text=(
                y_title or y_record.display_label
            ) if is_left_col else None,
            title_font=dict(size=_PUB_FONT["size"]),
            range=y_range,
            ticks="outside",
            ticklen=4,
            tickwidth=1,
            tickfont=dict(size=_PUB_TICK_SIZE),
            showline=True,
            mirror=True,
            zeroline=False,
            showgrid=True,
            gridcolor=_GRID_COLOR,
            row=row,
            col=col,
        )

    for empty_idx in range(n_panels, rows * cols):
        row = empty_idx // cols + 1
        col = empty_idx % cols + 1
        fig.update_xaxes(visible=False, row=row, col=col)
        fig.update_yaxes(visible=False, row=row, col=col)

    _add_trajectory_legend_traces(
        fig,
        color_map=color_map,
        show_group_legend=show_group_legend,
        baseline_output_label="Baseline",
        robustness_output_label=robustness_output_label,
        show_pareto=show_pareto and pareto_legend_needed,
        row=1,
        col=1,
    )

    for annotation in fig.layout.annotations:
        annotation.font = dict(size=_PUB_FONT["size"], family=_FONT_FAMILY)

    fig.update_layout(
        template="plotly_white",
        width=_TEXT_WIDTH_PX,
        height=height_px,
        autosize=False,
        legend_title_text="Backbone" if show_group_legend else None,
        legend=dict(font=dict(size=_PUB_LEGEND_SIZE)),
        margin=dict(l=10, r=10, t=28, b=28),
        font=_PUB_FONT,
        hovermode="closest",
        title=None,
    )
    return fig


def plot_improvement_trajectories(
    backbone_df: pd.DataFrame,
    improvement_df: pd.DataFrame,
    *,
    perf_col: str,
    robust_col: str,
    x_semantics: PlotSemanticsRecord,
    y_semantics: PlotSemanticsRecord,
    dataset_col: str = "dataset",
    model_col: str = "model",
    method_col: str = "robustness_method",
    backbone_col: str = "backbone_architecture",
    perf_lower_is_better: bool = True,
    flip_perf_axis: bool = False,
    show_pareto: bool = True,
    show_group_legend: bool = True,
    show_text_labels: bool = False,
    robustness_output_label: str = "Robustness method output",
    title: str = "Improvement Trajectories",
    x_title: str | None = None,
    y_title: str | None = None,
    baseline_backbone_col: str | None = None,
    improvement_name_col: str | None = None,
    require_signature: bool = False,
    signature_col: str = "data_config_signature",
    improvement_identity_key_cols: Sequence[str] | None = None,
    improvement_join_key_cols: Sequence[str] | None = None,
) -> go.Figure:
    """Plot trajectories showing backbone→improvement movement in performance-robustness space.

    Args:
        backbone_df: DataFrame with backbone (backbone) results
        improvement_df: DataFrame with improvement results
        perf_col: Performance metric column name
        robust_col: Robustness metric column name
        dataset_col: Dataset identifier column
        model_col: Model identifier column
        method_col: Grouping column used for backbone color coding
        backbone_col: Backbone architecture column in improvement_df
        perf_lower_is_better: Whether lower performance values are better
        flip_perf_axis: Whether to flip x-axis direction
        show_pareto: Whether to show Pareto frontier
        title: Plot title
        x_title: X-axis title
        y_title: Y-axis title
    """
    x_record = _require_plot_semantics_record(
        x_semantics,
        context="plot_improvement_trajectories x-axis",
    )
    y_record = _require_plot_semantics_record(
        y_semantics,
        context="plot_improvement_trajectories y-axis",
    )
    perf_minimize = _axis_is_minimize(x_record)
    if bool(perf_lower_is_better) != perf_minimize:
        raise ValueError(
            "plot_improvement_trajectories received inconsistent performance direction "
            f"metadata: perf_lower_is_better={perf_lower_is_better} but "
            f"x_semantics.direction={x_record.direction!r}."
        )
    baseline_backbone_col = baseline_backbone_col or model_col
    improvement_name_col = improvement_name_col or model_col
    if method_col not in improvement_df.columns:
        raise ValueError(
            f"plot_improvement_trajectories is missing grouping column '{method_col}'."
        )
    prepared_backbone_df, prepared_improvement_df, merged = _prepare_trajectory_frame_pair(
        backbone_df,
        improvement_df,
        perf_col=perf_col,
        robust_col=robust_col,
        dataset_col=dataset_col,
        baseline_backbone_col=baseline_backbone_col,
        improvement_name_col=improvement_name_col,
        method_col=method_col,
        backbone_col=backbone_col,
        require_signature=require_signature,
        signature_col=signature_col,
        improvement_identity_key_cols=improvement_identity_key_cols,
        improvement_join_key_cols=improvement_join_key_cols,
        context="plot_improvement_trajectories",
    )
    if merged.empty:
        fig = go.Figure()
        fig.update_layout(title=f"{title} (No Data)")
        return fig

    fig = make_subplots(rows=1, cols=1)
    color_map = _trajectory_color_map([merged])
    _add_trajectory_pair_traces(
        fig,
        merged,
        perf_col=perf_col,
        robust_col=robust_col,
        color_map=color_map,
        row=1,
        col=1,
        cols=1,
        show_text_labels=show_text_labels,
    )
    pareto_legend_needed = False
    if show_pareto:
        pareto_df = _trajectory_pareto_df(
            prepared_backbone_df,
            prepared_improvement_df,
            perf_col=perf_col,
            robust_col=robust_col,
            perf_lower_is_better=perf_minimize,
            robust_minimize=_axis_is_minimize(y_record),
        )
        if not pareto_df.empty:
            pareto_legend_needed = True
            fig.add_trace(
                go.Scatter(
                    x=pareto_df[perf_col],
                    y=pareto_df[robust_col],
                    mode="lines",
                    line=_PARETO_LINE,
                    name="Pareto frontier",
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    _add_trajectory_legend_traces(
        fig,
        color_map=color_map,
        show_group_legend=show_group_legend,
        baseline_output_label="Baseline",
        robustness_output_label=robustness_output_label,
        show_pareto=show_pareto and pareto_legend_needed,
        row=1,
        col=1,
    )

    x_range = _trajectory_perf_axis_range(
        np.concatenate(
            [
                merged[f"{perf_col}_baseline"].to_numpy(dtype=float),
                merged[f"{perf_col}_improved"].to_numpy(dtype=float),
            ]
        ),
        perf_lower_is_better=perf_minimize,
        flip_perf_axis=flip_perf_axis,
    )

    fig.update_xaxes(
        title_text=x_title or x_record.display_label,
        range=x_range,
        showline=True,
        mirror=True,
        showgrid=True,
        gridcolor=_GRID_COLOR,
    )
    fig.update_yaxes(
        title_text=y_title or y_record.display_label,
        range=list(
            _axis_range_from_semantics(
                np.concatenate(
                    [
                        merged[f"{robust_col}_baseline"].to_numpy(dtype=float),
                        merged[f"{robust_col}_improved"].to_numpy(dtype=float),
                    ]
                ),
                y_record,
            )
        ),
        showline=True,
        mirror=True,
        showgrid=True,
        gridcolor=_GRID_COLOR,
    )

    fig.update_layout(
        title=title,
        template="plotly_white",
        font=_DIAG_FONT,
        title_font=dict(size=_DIAG_TITLE_SIZE),
        margin=dict(l=10, r=10, t=50, b=10),
        hovermode="closest",
        legend_title_text="Backbone (filled markers)",
    )

    return fig


# ---------------------------------------------------------------------------
# Improvement delta & comparison plots
# ---------------------------------------------------------------------------


def plot_improvement_deltas_heatmap(
    deltas_df: pd.DataFrame,
    *,
    metric_cols: list[str],
    metric_semantics: Mapping[str, PlotSemanticsRecord],
    row_id_cols: list[str] = ["robustness_method", "backbone_architecture"],
    title: str = "Improvement Method Effectiveness",
    annotate_values: bool = True,
    significance_col: str | None = None,
    colorbar_label: str = "Utility-aligned Δ",
) -> go.Figure:
    """Plot heatmap showing improvement deltas across methods and metrics.

    Args:
        deltas_df: DataFrame with delta columns (e.g., delta_MSE_test, delta_D_w)
        metric_cols: List of delta column names to display
        metric_semantics: Explicit delta-direction semantics for every metric column
        row_id_cols: Columns to use for row identification
        title: Plot title
        annotate_values: Whether to show values in cells
        significance_col: Optional column with p-values for significance markers
        colorbar_label: Colorbar title.
    """
    plot_df = deltas_df.copy()
    if plot_df.empty:
        raise ValueError(
            "Cannot plot improvement deltas heatmap from an empty dataframe."
        )
    if not metric_cols:
        raise ValueError(
            "Cannot plot improvement deltas heatmap without metric columns."
        )
    missing_row_id_cols = [col for col in row_id_cols if col not in plot_df.columns]
    if missing_row_id_cols:
        raise ValueError(
            "Cannot plot improvement deltas heatmap: missing row identifier columns "
            f"{missing_row_id_cols}."
        )
    missing_metric_cols = [col for col in metric_cols if col not in plot_df.columns]
    if missing_metric_cols:
        raise ValueError(
            "Cannot plot improvement deltas heatmap: missing metric columns "
            f"{missing_metric_cols}."
        )
    resolved_metric_semantics = require_plot_semantics_mapping(
        metric_semantics,
        required_keys=metric_cols,
        context="plot_improvement_deltas_heatmap",
    )

    if len(row_id_cols) == 1:
        plot_df["_row_label"] = plot_df[row_id_cols[0]].astype(str)
    else:
        plot_df["_row_label"] = plot_df[row_id_cols].apply(
            lambda row: " | ".join(str(val) for val in row), axis=1
        )

    pivot_raw = plot_df.set_index("_row_label")[metric_cols].copy()
    if pivot_raw.isna().any().any():
        missing_mask = pivot_raw.isna()
        examples: list[dict[str, str]] = []
        for row_label, col_name in list(zip(*np.where(missing_mask.to_numpy())))[:5]:
            examples.append(
                {
                    "row": str(pivot_raw.index[row_label]),
                    "metric": str(metric_cols[col_name]),
                }
            )
        raise ValueError(
            "Cannot plot improvement deltas heatmap: missing metric values detected. "
            f"Examples: {examples}."
        )
    pivot_color = pivot_raw.copy()
    for col_name in metric_cols:
        pivot_color[col_name] = (
            pivot_color[col_name].astype(float)
            * _utility_aligned_delta_multiplier(resolved_metric_semantics[col_name])
        )

    display_cols = [
        resolved_metric_semantics[col_name].display_label
        for col_name in metric_cols
    ]
    pivot_raw.columns = display_cols
    pivot_color.columns = display_cols

    abs_max = max(
        abs(float(np.nanmin(pivot_color.values))),
        abs(float(np.nanmax(pivot_color.values))),
    )
    if not np.isfinite(abs_max) or abs_max <= 0:
        abs_max = 1.0
    color_range = [-abs_max, abs_max]

    annotations_text = None
    if annotate_values:
        annotations_text = pivot_raw.map(lambda x: f"{x:.3f}" if pd.notna(x) else "")

        if significance_col and significance_col in plot_df.columns:
            sig_df = plot_df.set_index("_row_label")[[significance_col]]
            for idx in annotations_text.index:
                for col_idx, col in enumerate(annotations_text.columns):
                    if idx in sig_df.index and sig_df.loc[idx, significance_col] < 0.05:
                        annotations_text.loc[idx, col] += "*"

    fig = px.imshow(
        pivot_color,
        text_auto=False,
        aspect="auto",
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0,
        range_color=color_range,
        labels=dict(color=colorbar_label),
        title=title,
    )

    if annotations_text is not None:
        fig.update_traces(text=annotations_text.values, texttemplate="%{text}")

    fig.update_layout(
        width=max(600, len(display_cols) * 100),
        height=max(400, len(pivot_color) * 40),
        title=dict(y=0.95, x=0.5, xanchor="center", yanchor="top"),
        font=_DIAG_FONT_COMPACT,
        title_font=dict(size=_DIAG_TITLE_SIZE_COMPACT),
        margin=dict(l=150, t=80, b=80, r=50),
        template="plotly_white",
    )

    return fig


def plot_method_delta_pair_subplots(
    deltas_df: pd.DataFrame,
    *,
    perf_delta_col: str,
    robust_delta_col: str,
    perf_semantics: PlotSemanticsRecord,
    robust_semantics: PlotSemanticsRecord,
    robust_higher_is_better: bool = True,
    method_col: str = "robustness_method",
    baseline_col: str = "backbone_architecture",
    title: str = "Improvement Method Deltas",
) -> go.Figure:
    """Plot two baseline x method heatmap panels: performance and robustness."""
    plot_df = deltas_df.copy()
    perf_record = _require_plot_semantics_record(
        perf_semantics,
        context="plot_method_delta_pair_subplots performance semantics",
    )
    robust_record = _require_plot_semantics_record(
        robust_semantics,
        context="plot_method_delta_pair_subplots robustness semantics",
    )
    if _axis_is_minimize(robust_record) == bool(robust_higher_is_better):
        raise ValueError(
            "plot_method_delta_pair_subplots received inconsistent robustness direction "
            f"metadata: robust_higher_is_better={robust_higher_is_better} but "
            f"robust_semantics.direction={robust_record.direction!r}."
        )
    if plot_df.empty:
        raise ValueError(
            "Cannot plot method delta pair subplots from an empty dataframe."
        )

    required_cols = [baseline_col, method_col, perf_delta_col, robust_delta_col]
    missing_cols = [col for col in required_cols if col not in plot_df.columns]
    if missing_cols:
        raise ValueError(
            "Cannot plot method delta pair subplots: missing required columns "
            f"{missing_cols}."
        )

    for label_col in (baseline_col, method_col):
        missing_label = plot_df[label_col].isna()
        if missing_label.any():
            examples = (
                plot_df.loc[missing_label, [baseline_col, method_col]]
                .drop_duplicates()
                .head(5)
                .to_dict(orient="records")
            )
            raise ValueError(
                "Cannot plot method delta pair subplots: missing axis labels for "
                f"'{label_col}'. Examples: {examples}."
            )
        plot_df[label_col] = plot_df[label_col].astype(str).str.strip()
        empty_label = plot_df[label_col] == ""
        if empty_label.any():
            examples = (
                plot_df.loc[empty_label, [baseline_col, method_col]]
                .drop_duplicates()
                .head(5)
                .to_dict(orient="records")
            )
            raise ValueError(
                "Cannot plot method delta pair subplots: empty axis labels for "
                f"'{label_col}'. Examples: {examples}."
            )

    for delta_col in (perf_delta_col, robust_delta_col):
        plot_df[delta_col] = pd.to_numeric(plot_df[delta_col], errors="raise")
        if plot_df[delta_col].isna().any():
            examples = (
                plot_df.loc[
                    plot_df[delta_col].isna(),
                    [baseline_col, method_col, "pipeline_id"]
                    if "pipeline_id" in plot_df.columns
                    else [baseline_col, method_col],
                ]
                .drop_duplicates()
                .head(5)
                .to_dict(orient="records")
            )
            raise ValueError(
                "Cannot plot method delta pair subplots: missing delta values for required "
                f"column '{delta_col}'. Examples: {examples}."
            )
        finite_mask = np.isfinite(plot_df[delta_col].to_numpy(dtype=float))
        if not finite_mask.all():
            examples = (
                plot_df.loc[
                    ~finite_mask,
                    [baseline_col, method_col, "pipeline_id"]
                    if "pipeline_id" in plot_df.columns
                    else [baseline_col, method_col],
                ]
                .drop_duplicates()
                .head(5)
                .to_dict(orient="records")
            )
            raise ValueError(
                "Cannot plot method delta pair subplots: non-finite delta values for required "
                f"column '{delta_col}'. Examples: {examples}."
            )

    duplicate_mask = plot_df.duplicated([baseline_col, method_col], keep=False)
    if duplicate_mask.any():
        examples = (
            plot_df.loc[duplicate_mask, [baseline_col, method_col, "pipeline_id"]]
            if "pipeline_id" in plot_df.columns
            else plot_df.loc[duplicate_mask, [baseline_col, method_col]]
        )
        examples = (
            examples
            .drop_duplicates()
            .head(5)
            .to_dict(orient="records")
        )
        raise ValueError(
            "Cannot plot method delta pair subplots: duplicate (baseline, method) rows "
            f"Examples: {examples}."
        )

    perf_metric_name = perf_delta_col.removeprefix("delta_")
    perf_label = perf_record.display_label
    robust_metric_name = robust_delta_col.removeprefix("delta_")
    robust_title = robust_record.display_label
    robust_label = robust_record.display_label

    baseline_order = sorted(plot_df[baseline_col].unique().tolist())
    method_order = sorted(plot_df[method_col].unique().tolist())

    perf_pivot = (
        plot_df.pivot(index=baseline_col, columns=method_col, values=perf_delta_col)
        .reindex(index=baseline_order, columns=method_order)
        .astype(float)
    )
    robust_pivot = (
        plot_df.pivot(index=baseline_col, columns=method_col, values=robust_delta_col)
        .reindex(index=baseline_order, columns=method_order)
        .astype(float)
    )

    perf_raw = perf_pivot.to_numpy(dtype=float)
    robust_raw = robust_pivot.to_numpy(dtype=float)
    perf_color = perf_raw
    robust_color = robust_raw

    def _format_delta_text(values: np.ndarray) -> np.ndarray:
        text = np.full(values.shape, "", dtype=object)
        finite_mask = np.isfinite(values)
        text[finite_mask] = [f"{float(value):.3f}" for value in values[finite_mask]]
        return text

    def _abs_symmetric_bound(values: np.ndarray) -> float:
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            return 1.0
        abs_max = float(np.max(np.abs(finite_values)))
        return 1.0 if abs_max <= 0.0 else abs_max

    perf_text = _format_delta_text(perf_raw)
    robust_text = _format_delta_text(robust_raw)
    perf_abs = _abs_symmetric_bound(perf_color)
    robust_abs = _abs_symmetric_bound(robust_color)

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.16,
        subplot_titles=[perf_label, robust_label],
    )
    fig.add_trace(
        go.Heatmap(
            x=method_order,
            y=baseline_order,
            z=perf_color,
            text=perf_text,
            texttemplate="%{text}",
            colorscale=_directional_colorscale(perf_record),
            zmid=0.0,
            zmin=-perf_abs,
            zmax=perf_abs,
            colorbar=dict(title=perf_record.display_label, x=0.44),
            hovertemplate=(
                "Baseline: %{y}<br>"
                "Method: %{x}<br>"
                f"Δ {perf_metric_name}: %{{text}}<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            x=method_order,
            y=baseline_order,
            z=robust_color,
            text=robust_text,
            texttemplate="%{text}",
            colorscale=_directional_colorscale(robust_record),
            zmid=0.0,
            zmin=-robust_abs,
            zmax=robust_abs,
            colorbar=dict(title=robust_record.display_label, x=1.02),
            hovertemplate=(
                "Baseline: %{y}<br>"
                "Method: %{x}<br>"
                f"Δ {robust_title}: %{{text}}<extra></extra>"
            ),
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="Improvement Method", tickangle=-35, row=1, col=1)
    fig.update_xaxes(title_text="Improvement Method", tickangle=-35, row=1, col=2)
    fig.update_yaxes(title_text="Baseline Backbone", autorange="reversed", row=1, col=1)
    fig.update_yaxes(title_text="Baseline Backbone", autorange="reversed", row=1, col=2)
    fig.update_layout(
        width=max(1100, 220 * max(1, len(method_order)) + 380),
        height=max(440, 46 * max(1, len(baseline_order)) + 160),
        title=dict(y=0.95, x=0.5, xanchor="center", yanchor="top"),
        font=_DIAG_FONT_COMPACT,
        title_font=dict(size=_DIAG_TITLE_SIZE_COMPACT),
        margin=dict(l=120, t=100, b=90, r=90),
        showlegend=False,
        title_text=title,
        template="plotly_white",
    )
    return fig


def plot_improvement_comparison(
    deltas_long_df: pd.DataFrame,
    *,
    method_col: str = "robustness_method",
    metric_col: str = "metric_name",
    value_col: str = "delta_value",
    title: str = "Improvement Method Comparison",
    reference_line: float = 0.0,
) -> go.Figure:
    """Plot boxplot comparing delta distributions across improvement methods.

    Args:
        deltas_long_df: Long-form DataFrame with one row per observation
        method_col: Column containing improvement method names
        metric_col: Column containing metric names
        value_col: Column containing delta values
        title: Plot title
        reference_line: Y-value for reference line (typically 0)
    """
    plot_df = deltas_long_df.copy()
    plot_df["_metric_display"] = plot_df[metric_col].apply(metric_display_name)

    fig = px.box(
        plot_df,
        x=method_col,
        y=value_col,
        color=method_col,
        facet_col="_metric_display",
        facet_col_wrap=3,
        title=title,
        labels={
            value_col: "Δ Value",
            method_col: "Method",
            "_metric_display": "Metric",
        },
    )

    fig.add_hline(
        y=reference_line,
        line_dash="dash",
        line_color="black",
        opacity=0.5,
        annotation_text="No change",
        annotation_position="right",
    )

    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

    fig.update_layout(
        width=1200,
        height=600,
        title=dict(y=0.97, x=0.5, xanchor="center", yanchor="top"),
        font=_DIAG_FONT_COMPACT,
        title_font=dict(size=_DIAG_TITLE_SIZE_COMPACT),
        margin=dict(l=50, t=100, b=80, r=50),
        template="plotly_white",
        showlegend=False,
    )

    fig.update_xaxes(tickangle=-45)

    return fig


def plot_per_method_delta_scatter(
    deltas_df: pd.DataFrame,
    *,
    delta_perf_col: str,
    delta_robust_col: str,
    method_name: str,
    perf_semantics: PlotSemanticsRecord,
    robust_semantics: PlotSemanticsRecord,
    robust_higher_is_better: bool = True,
    arch_col: str = "architecture_family",
    perf_lower_is_better: bool = True,
    normalize_perf: bool = True,
    symbol_col: str | None = None,
    x_title: str | None = None,
    y_title: str | None = None,
) -> go.Figure:
    df = deltas_df.dropna(subset=[delta_perf_col, delta_robust_col]).copy()
    perf_record = _require_plot_semantics_record(
        perf_semantics,
        context="plot_per_method_delta_scatter performance semantics",
    )
    robust_record = _require_plot_semantics_record(
        robust_semantics,
        context="plot_per_method_delta_scatter robustness semantics",
    )
    if bool(perf_lower_is_better) != _axis_is_minimize(perf_record):
        raise ValueError(
            "plot_per_method_delta_scatter received inconsistent performance direction "
            f"metadata: perf_lower_is_better={perf_lower_is_better} but "
            f"perf_semantics.direction={perf_record.direction!r}."
        )
    if _axis_is_minimize(robust_record) == bool(robust_higher_is_better):
        raise ValueError(
            "plot_per_method_delta_scatter received inconsistent robustness direction "
            f"metadata: robust_higher_is_better={robust_higher_is_better} but "
            f"robust_semantics.direction={robust_record.direction!r}."
        )
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title=f"{method_name}: no data", template="plotly_white")
        return fig

    if arch_col not in df.columns:
        arch_col = "backbone_architecture" if "backbone_architecture" in df.columns else None

    x = df[delta_perf_col].astype(float).to_numpy()
    if normalize_perf:
        raise ValueError(
            "plot_per_method_delta_scatter no longer supports normalize_perf=True. "
            "Pass raw delta semantics explicitly."
        )
    df["_x"] = x
    df["_y"] = df[delta_robust_col].astype(float)

    fig = px.scatter(
        df,
        x="_x",
        y="_y",
        color=arch_col if arch_col else None,
        symbol=symbol_col if symbol_col and symbol_col in df.columns else None,
        hover_data={
            "robustness_method": True if "robustness_method" in df.columns else False,
            "pipeline_id": True if "pipeline_id" in df.columns else False,
            "model_architecture": True if "model_architecture" in df.columns else False,
        },
        template="plotly_white",
        labels={"_x": x_title or perf_record.display_label,
                "_y": y_title or robust_record.display_label,
                (arch_col or "color"): "Architecture"},
        title=f"{method_name}: Performance vs Robustness change",
    )
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.3)
    fig.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.3)
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    return fig


def plot_method_win_rate_heatmap(
    win_rate_df: pd.DataFrame,
    *,
    row_col: str = "dataset",
    col_col: str = "robustness_method",
    value_col: str = "win_rate_pct",
    title: str = "Method Win Rate Heatmap",
) -> go.Figure:
    plot_df = win_rate_df[[row_col, col_col, value_col]].copy()
    plot_df[value_col] = pd.to_numeric(plot_df[value_col], errors="raise")
    if plot_df[value_col].isna().any():
        raise ValueError("Win rate data contains NaN values after numeric conversion.")
    if plot_df.empty:
        fig = go.Figure()
        fig.update_layout(title=f"{title} (No Data)")
        return fig

    pivot_df = plot_df.pivot(index=row_col, columns=col_col, values=value_col)
    text_values = pivot_df.map(lambda x: f"{x:.1f}%" if pd.notna(x) else "")

    fig = px.imshow(
        pivot_df,
        text_auto=False,
        aspect="auto",
        color_continuous_scale="YlGn",
        range_color=[0, 100],
        labels=dict(x="Method", y="Dataset", color="% Beating Baseline"),
        title=title,
    )
    fig.update_traces(text=text_values.values, texttemplate="%{text}")
    fig.update_layout(
        width=max(700, 130 * max(1, len(pivot_df.columns))),
        height=max(420, 90 * max(1, len(pivot_df.index))),
        title=dict(y=0.95, x=0.5, xanchor="center", yanchor="top"),
        font=_DIAG_FONT_COMPACT,
        title_font=dict(size=_DIAG_TITLE_SIZE_COMPACT),
        margin=dict(l=120, t=80, b=80, r=60),
        template="plotly_white",
    )
    return fig


def plot_selection_margin(
    margin_df: pd.DataFrame,
    *,
    dataset_col: str = "dataset",
    method_col: str = "robustness_method",
    margin_col: str = "selection_margin",
    title: str = "Representative Selection Margin",
) -> go.Figure:
    required = [dataset_col, method_col, margin_col]
    plot_df = margin_df[required].copy()
    plot_df[margin_col] = pd.to_numeric(plot_df[margin_col], errors="raise")
    if plot_df[margin_col].isna().any():
        raise ValueError("Selection margin data contains NaN values after numeric conversion.")
    if plot_df.empty:
        fig = go.Figure()
        fig.update_layout(title=f"{title} (No Data)")
        return fig

    has_metric_col = "selection_metric_name" in margin_df.columns
    has_scores = (
        "winner_score" in margin_df.columns
        and "runner_score" in margin_df.columns
    )
    if has_metric_col:
        plot_df["selection_metric_name"] = margin_df["selection_metric_name"].astype(str)
    if has_scores:
        plot_df["winner_score"] = pd.to_numeric(margin_df["winner_score"], errors="raise")
        plot_df["runner_score"] = pd.to_numeric(margin_df["runner_score"], errors="raise")

    plot_df["label"] = (
        plot_df[dataset_col].astype(str) + " | " + plot_df[method_col].astype(str)
    )
    if has_metric_col:
        plot_df["label"] = (
            plot_df["label"] + " | " + plot_df["selection_metric_name"].astype(str)
        )
    plot_df = plot_df.sort_values(margin_col, ascending=False)

    fig = px.bar(
        plot_df,
        x="label",
        y=margin_col,
        color=dataset_col,
        hover_data={
            "selection_metric_name": True if has_metric_col else False,
            "winner_score": True if has_scores else False,
            "runner_score": True if has_scores else False,
        },
        title=title,
        labels={
            "label": (
                "Dataset | Method | Selection Metric"
                if has_metric_col
                else "Dataset | Method"
            ),
            margin_col: "Validation Margin (Runner-up - Winner)",
            dataset_col: "Dataset",
        },
        template="plotly_white",
    )
    fig.add_hline(y=0.0, line_dash="dash", line_color="black", opacity=0.4)
    fig.update_layout(
        width=max(800, 80 * max(1, len(plot_df))),
        height=500,
        title=dict(y=0.95, x=0.5, xanchor="center", yanchor="top"),
        font=_DIAG_FONT_COMPACT,
        title_font=dict(size=_DIAG_TITLE_SIZE_COMPACT),
        margin=dict(l=80, t=80, b=140, r=40),
    )
    fig.update_xaxes(tickangle=-35)
    return fig


def plot_scenario_delta_heatmap(
    scenario_delta_df: pd.DataFrame,
    *,
    row_id_cols: Sequence[str] = ("dataset", "robustness_method"),
    scenario_col: str = "scenario",
    value_col: str = "delta_metric_value",
    value_semantics: PlotSemanticsRecord,
    title: str = "Scenario Delta vs Baseline",
    color_label: str | None = None,
) -> go.Figure:
    value_record = _require_plot_semantics_record(
        value_semantics,
        context="plot_scenario_delta_heatmap value semantics",
    )
    required_cols = [*row_id_cols, scenario_col, value_col]
    plot_df = scenario_delta_df[required_cols].copy()
    plot_df[value_col] = pd.to_numeric(plot_df[value_col], errors="raise")
    if plot_df[value_col].isna().any():
        raise ValueError("Scenario delta data contains NaN values after numeric conversion.")
    if plot_df.empty:
        fig = go.Figure()
        fig.update_layout(title=f"{title} (No Data)")
        return fig

    if len(row_id_cols) == 1:
        plot_df["_row_label"] = plot_df[row_id_cols[0]].astype(str)
    else:
        plot_df["_row_label"] = plot_df[list(row_id_cols)].apply(
            lambda row: " | ".join(str(v) for v in row), axis=1
        )

    pivot_df = plot_df.pivot(
        index="_row_label",
        columns=scenario_col,
        values=value_col,
    )
    text_values = pivot_df.map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
    abs_max = max(abs(float(np.nanmin(pivot_df.values))), abs(float(np.nanmax(pivot_df.values))))
    if not np.isfinite(abs_max) or abs_max <= 0:
        abs_max = 1.0

    fig = px.imshow(
        pivot_df,
        text_auto=False,
        aspect="auto",
        color_continuous_scale=_directional_colorscale(value_record),
        color_continuous_midpoint=0.0,
        range_color=[-abs_max, abs_max],
        labels=dict(
            x="Scenario",
            y="Dataset | Method",
            color=color_label or value_record.display_label,
        ),
        title=title,
    )
    fig.update_traces(text=text_values.values, texttemplate="%{text}")
    fig.update_layout(
        width=max(900, 120 * max(1, len(pivot_df.columns))),
        height=max(450, 80 * max(1, len(pivot_df.index))),
        title=dict(y=0.95, x=0.5, xanchor="center", yanchor="top"),
        font=_DIAG_FONT_COMPACT,
        title_font=dict(size=_DIAG_TITLE_SIZE_COMPACT),
        margin=dict(l=170, t=80, b=80, r=60),
        template="plotly_white",
    )
    return fig


def _normalize_display_mapping(
    mapping: Mapping[str, str],
    *,
    context: str,
) -> dict[str, str]:
    if not isinstance(mapping, Mapping) or not mapping:
        raise ValueError(f"{context} display mapping must be a non-empty mapping.")
    normalized: dict[str, str] = {}
    seen_labels: set[str] = set()
    for raw_key, display_label in mapping.items():
        raw_text = str(raw_key).strip()
        label_text = str(display_label).strip()
        if not raw_text:
            raise ValueError(f"{context} contains an empty raw id.")
        if not label_text:
            raise ValueError(f"{context} contains an empty display label.")
        if raw_text in normalized:
            raise ValueError(f"{context} contains duplicate raw id '{raw_text}'.")
        if label_text in seen_labels:
            raise ValueError(
                f"{context} contains duplicate display label '{label_text}'."
            )
        seen_labels.add(label_text)
        normalized[raw_text] = label_text
    return normalized


def _normalize_ordered_display_spec(
    spec: Sequence[tuple[str, str]],
    *,
    context: str,
) -> list[tuple[str, str]]:
    if not spec:
        raise ValueError(f"{context} must be non-empty.")
    normalized_map = _normalize_display_mapping(dict(spec), context=context)
    return [(raw_key, normalized_map[raw_key]) for raw_key, _ in spec]


def _plot_grouped_scenario_heatmap(
    scenario_df: pd.DataFrame,
    *,
    dataset_spec: Sequence[tuple[str, str]],
    row_display: Mapping[str, str],
    scenario_display_order: Sequence[str],
    scenario_display: Mapping[str, str],
    scenario_groups: Mapping[str, Sequence[str]],
    value_col: str,
    value_semantics: PlotSemanticsRecord,
    dataset_col: str,
    row_col: str,
    scenario_col: str,
    row_id_label: str,
    row_axis_title: str,
    row_hover_label: str,
    colorbar_label: str,
    context: str,
    horizontal_spacing: float = 0.035,
) -> go.Figure:
    value_record = _require_plot_semantics_record(
        value_semantics,
        context=f"{context} value semantics",
    )
    normalized_dataset_spec = _normalize_ordered_display_spec(
        dataset_spec,
        context=f"{context} dataset_spec",
    )
    normalized_row_display = _normalize_display_mapping(
        row_display,
        context=f"{context} row_display",
    )
    normalized_scenario_display = _normalize_display_mapping(
        scenario_display,
        context=f"{context} scenario_display",
    )
    ordered_scenarios = [str(value).strip() for value in scenario_display_order]
    if ordered_scenarios != list(normalized_scenario_display):
        raise ValueError(
            f"{context} scenario order must match scenario_display keys exactly."
        )

    required_cols = {dataset_col, row_col, scenario_col, value_col}
    missing_cols = sorted(required_cols - set(scenario_df.columns))
    if missing_cols:
        raise ValueError(
            f"{context} is missing required columns: {missing_cols}."
        )
    if scenario_df.empty:
        raise ValueError(f"{context} received an empty dataframe.")

    plot_df = scenario_df.copy()
    for column in (dataset_col, row_col, scenario_col):
        values = plot_df[column].astype(str).str.strip()
        if (values == "").any():
            raise ValueError(f"{context} has empty values in '{column}'.")
        plot_df[column] = values
    plot_df[value_col] = pd.to_numeric(plot_df[value_col], errors="raise")
    finite_mask = np.isfinite(plot_df[value_col].to_numpy(dtype=float))
    if not finite_mask.all():
        raise ValueError(f"{context} has non-finite '{value_col}' values.")

    dataset_keys = [raw_key for raw_key, _ in normalized_dataset_spec]
    row_keys = list(normalized_row_display)
    validate_raw_display_id_values(
        plot_df[dataset_col].tolist(),
        raw_ids=dataset_keys,
        display_mapping=dict(normalized_dataset_spec),
        context=context,
        id_label="dataset",
    )
    validate_raw_display_id_values(
        plot_df[row_col].tolist(),
        raw_ids=row_keys,
        display_mapping=normalized_row_display,
        context=context,
        id_label=row_id_label,
    )
    validate_raw_display_id_values(
        plot_df[scenario_col].tolist(),
        raw_ids=ordered_scenarios,
        display_mapping=normalized_scenario_display,
        context=context,
        id_label="scenario",
    )

    filtered = plot_df.copy()
    if filtered.empty:
        raise ValueError(
            f"{context} has no rows in the requested dataset/{row_id_label} scope."
        )
    assert_no_duplicate_rows(
        filtered,
        [dataset_col, row_col, scenario_col],
        context=(
            f"{context} requires unique "
            f"({dataset_col}, {row_col}, {scenario_col}) rows"
        ),
    )

    consumed: list[str] = []
    for group_name, scenario_keys in scenario_groups.items():
        scenario_list = [str(value).strip() for value in scenario_keys]
        if not scenario_list:
            raise ValueError(f"{context} group '{group_name}' is empty.")
        consumed.extend(scenario_list)
    if consumed != ordered_scenarios:
        raise ValueError(
            f"{context} scenario_groups do not cover scenario_display_order exactly."
        )

    finite_values = filtered[value_col].to_numpy(dtype=float)
    z_range = _axis_range_from_semantics(finite_values, value_record)

    subplot_titles = [label for _, label in normalized_dataset_spec]
    fig = make_subplots(
        rows=1,
        cols=len(dataset_keys),
        shared_xaxes=False,
        shared_yaxes=True,
        horizontal_spacing=horizontal_spacing,
        vertical_spacing=0.0,
        subplot_titles=subplot_titles,
    )

    finite_cell_count = 0
    x_positions = list(range(len(ordered_scenarios)))
    y_positions = list(range(len(row_keys)))
    for panel_idx, (dataset_key, _) in enumerate(normalized_dataset_spec):
        row = 1
        col = panel_idx + 1
        dataset_df = filtered.loc[filtered[dataset_col] == dataset_key].copy()
        matrix = np.full((len(row_keys), len(ordered_scenarios)), np.nan, dtype=float)
        unavailable_mask = np.zeros_like(matrix, dtype=bool)
        hover_customdata = np.empty(
            (len(row_keys), len(ordered_scenarios), 3),
            dtype=object,
        )
        dataset_label = dict(normalized_dataset_spec)[dataset_key]
        for row_idx, row_key in enumerate(row_keys):
            row_label = normalized_row_display[row_key]
            for scenario_idx, scenario_key in enumerate(ordered_scenarios):
                hover_customdata[row_idx, scenario_idx, 0] = (
                    normalized_scenario_display[scenario_key]
                )
                hover_customdata[row_idx, scenario_idx, 1] = row_label
                hover_customdata[row_idx, scenario_idx, 2] = dataset_label
        for row_idx, row_key in enumerate(row_keys):
            scoped_df = dataset_df.loc[dataset_df[row_col] == row_key].copy()
            if scoped_df.empty:
                unavailable_mask[row_idx, :] = True
                continue
            present_scenarios = set(scoped_df[scenario_col].tolist())
            if present_scenarios != set(ordered_scenarios):
                missing_scenarios = sorted(set(ordered_scenarios) - present_scenarios)
                extra_scenarios = sorted(present_scenarios - set(ordered_scenarios))
                raise ValueError(
                    f"{context} requires full scenario coverage for available cells. "
                    f"Dataset={dataset_key}, {row_id_label}={row_key}, "
                    f"missing={missing_scenarios}, extra={extra_scenarios}."
                )
            ordered_row_df = (
                scoped_df.set_index(scenario_col)
                .loc[ordered_scenarios]
                .reset_index()
            )
            values = ordered_row_df[value_col].to_numpy(dtype=float)
            matrix[row_idx, :] = values
            finite_cell_count += int(np.isfinite(values).sum())
        text = np.full(matrix.shape, "", dtype=object)
        finite_entries = np.isfinite(matrix)
        text[finite_entries] = [f"{float(value):.2f}" for value in matrix[finite_entries]]

        fig.add_trace(
            go.Heatmap(
                x=x_positions,
                y=y_positions,
                z=matrix,
                customdata=hover_customdata,
                coloraxis="coloraxis",
                text=text,
                texttemplate="%{text}",
                hovertemplate=(
                    "Dataset: %{customdata[2]}<br>"
                    f"{row_hover_label}: %{{customdata[1]}}<br>"
                    "Scenario: %{customdata[0]}<br>"
                    f"{value_record.display_label}: %{{z:.3f}}<extra></extra>"
                ),
                xgap=1,
                ygap=1,
                showscale=False,
            ),
            row=row,
            col=col,
        )
        if unavailable_mask.any():
            fig.add_trace(
                go.Heatmap(
                    x=x_positions,
                    y=y_positions,
                    z=np.where(unavailable_mask, 1.0, np.nan),
                    text=np.where(unavailable_mask, "NA", ""),
                    texttemplate="%{text}",
                    colorscale=[[0.0, "#D0D0D0"], [1.0, "#D0D0D0"]],
                    zmin=0.0,
                    zmax=1.0,
                    showscale=False,
                    hoverinfo="skip",
                    xgap=1,
                    ygap=1,
                ),
                row=row,
                col=col,
            )

        fig.update_xaxes(
            tickmode="array",
            tickvals=x_positions,
            ticktext=[normalized_scenario_display[key] for key in ordered_scenarios],
            tickangle=-35,
            title_text=None,
            row=row,
            col=col,
        )
        fig.update_yaxes(
            tickmode="array",
            tickvals=y_positions,
            ticktext=[normalized_row_display[key] for key in row_keys],
            autorange="reversed",
            title_text=row_axis_title if col == 1 else None,
            showticklabels=col == 1,
            row=row,
            col=col,
        )

        xref, yref = _subplot_axis_refs(row=row, col=col, cols=len(dataset_keys))
        start_idx = 0
        for group_name, scenario_keys in scenario_groups.items():
            group_len = len(tuple(scenario_keys))
            end_idx = start_idx + group_len
            fig.add_annotation(
                x=float(start_idx + (group_len - 1) / 2.0),
                y=-1.05,
                xref=xref,
                yref=yref,
                text=str(group_name),
                showarrow=False,
                font=dict(size=_PUB_FONT["size"], family=_FONT_FAMILY),
            )
            if end_idx < len(ordered_scenarios):
                fig.add_vline(
                    x=float(end_idx - 0.5),
                    line_color="rgba(0,0,0,0.45)",
                    line_width=1,
                    row=row,
                    col=col,
                )
            start_idx = end_idx

    if finite_cell_count == 0:
        raise ValueError(f"{context} has no finite cells.")

    coloraxis = dict(
        colorscale=_directional_colorscale(value_record),
        cmin=float(z_range[0]),
        cmax=float(z_range[1]),
        colorbar=dict(title=colorbar_label, thickness=14, len=0.85),
    )
    if value_record.axis_family == "delta":
        neutral_value = 0.0 if value_record.neutral_value is None else float(value_record.neutral_value)
        coloraxis["cmid"] = neutral_value

    fig.update_layout(
        template="plotly_white",
        width=max(_TEXT_WIDTH_PX * 2, 260 * len(dataset_keys)),
        height=max(_PANEL_HEIGHT_PX + 110, 34 * len(row_keys) + 150),
        autosize=False,
        margin=dict(l=76, r=28, t=28, b=62),
        font=_PUB_FONT,
        coloraxis=coloraxis,
        title=None,
    )
    for annotation in fig.layout.annotations:
        annotation.font = dict(size=_PUB_FONT["size"], family=_FONT_FAMILY)
    return fig


def plot_method_scenario_delta_heatmap(
    scenario_delta_df: pd.DataFrame,
    *,
    dataset_spec: Sequence[tuple[str, str]],
    method_display: Mapping[str, str],
    scenario_display_order: Sequence[str],
    scenario_display: Mapping[str, str],
    scenario_groups: Mapping[str, Sequence[str]],
    value_col: str,
    value_semantics: PlotSemanticsRecord,
    dataset_col: str = "dataset",
    method_col: str = "robustness_method",
    scenario_col: str = "scenario",
    colorbar_label: str = "Delta Scenario Degradation",
    horizontal_spacing: float = 0.035,
) -> go.Figure:
    value_record = _require_plot_semantics_record(
        value_semantics,
        context="plot_method_scenario_delta_heatmap value semantics",
    )
    if value_record.axis_family != "delta":
        raise ValueError(
            "plot_method_scenario_delta_heatmap requires delta semantics."
        )
    return _plot_grouped_scenario_heatmap(
        scenario_delta_df,
        dataset_spec=dataset_spec,
        row_display=method_display,
        scenario_display_order=scenario_display_order,
        scenario_display=scenario_display,
        scenario_groups=scenario_groups,
        value_col=value_col,
        value_semantics=value_semantics,
        dataset_col=dataset_col,
        row_col=method_col,
        scenario_col=scenario_col,
        row_id_label="robustness_method",
        row_axis_title="Method",
        row_hover_label="Method",
        colorbar_label=colorbar_label,
        context="plot_method_scenario_delta_heatmap",
        horizontal_spacing=horizontal_spacing,
    )


def plot_backbone_scenario_heatmap(
    scenario_df: pd.DataFrame,
    *,
    dataset_spec: Sequence[tuple[str, str]],
    backbone_display: Mapping[str, str],
    scenario_display_order: Sequence[str],
    scenario_display: Mapping[str, str],
    scenario_groups: Mapping[str, Sequence[str]],
    value_col: str,
    value_semantics: PlotSemanticsRecord,
    dataset_col: str = "dataset",
    backbone_col: str = "model_architecture",
    scenario_col: str = "scenario",
    colorbar_label: str = "Scenario Degradation",
    horizontal_spacing: float = 0.035,
) -> go.Figure:
    return _plot_grouped_scenario_heatmap(
        scenario_df,
        dataset_spec=dataset_spec,
        row_display=backbone_display,
        scenario_display_order=scenario_display_order,
        scenario_display=scenario_display,
        scenario_groups=scenario_groups,
        value_col=value_col,
        value_semantics=value_semantics,
        dataset_col=dataset_col,
        row_col=backbone_col,
        scenario_col=scenario_col,
        row_id_label="model_architecture",
        row_axis_title="Backbone",
        row_hover_label="Backbone",
        colorbar_label=colorbar_label,
        context="plot_backbone_scenario_heatmap",
        horizontal_spacing=horizontal_spacing,
    )


def plot_forecast_extreme(
    time_index: Sequence[float] | Sequence[int],
    target: Sequence[float],
    prediction_perturbed: Sequence[float],
    *,
    prediction_clean: Sequence[float] | None = None,
    clean_input: Sequence[Sequence[float]] | Sequence[float] | None = None,
    perturbed_input: Sequence[Sequence[float]] | Sequence[float] | None = None,
    input_time_index: Sequence[float] | Sequence[int] | None = None,
    input_feature_names: Sequence[str] | None = None,
    target_feature_names: Sequence[str] | None = None,
    affected_feature_names: Sequence[str] | None = None,
    title: str | None = None,
    scenario: str | None = None,
    severity: float | None = None,
    sample_score: float | None = None,
    score_metric: str | None = None,
    target_label: str = "Target",
    clean_prediction_label: str = "Prediction (Clean)",
    perturbed_prediction_label: str = "Prediction (Perturbed)",
) -> go.Figure:
    max_display_channels = _MAX_FORECAST_DISPLAY_CHANNELS

    def _ensure_2d(values):
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 1:
            arr = arr[:, None]
        return arr

    def _norm(name: str | None) -> str:
        return name.strip().casefold() if isinstance(name, str) and name.strip() else ""

    def _default_time(length: int) -> np.ndarray:
        return np.arange(1, length + 1, dtype=float)

    def _channel_opacity(n: int, *, lo: float = 0.15, hi: float = 0.8) -> float:
        return max(lo, min(hi, 5.0 / max(n, 1)))

    output_time = np.asarray(time_index, dtype=float)
    target_arr = _ensure_2d(target)
    pert_prediction_arr = _ensure_2d(prediction_perturbed)
    if target_arr.shape != pert_prediction_arr.shape:
        raise ValueError(
            "plot_forecast_extreme requires target and prediction_perturbed to share "
            f"the same shape; received {target_arr.shape} vs {pert_prediction_arr.shape}."
        )
    clean_prediction_arr = None
    if prediction_clean is not None:
        clean_prediction_arr = _ensure_2d(prediction_clean)
        if clean_prediction_arr.shape != target_arr.shape:
            raise ValueError(
                "plot_forecast_extreme requires prediction_clean to share the same "
                f"shape as target; received {clean_prediction_arr.shape} vs "
                f"{target_arr.shape}."
            )

    if input_feature_names is None:
        input_names: list[str] = []
    else:
        input_names = list(input_feature_names)
    clean_arr = _ensure_2d(clean_input) if clean_input is not None else None
    pert_arr = _ensure_2d(perturbed_input) if perturbed_input is not None else None
    clean_time = None
    if clean_arr is not None:
        clean_time = (
            np.asarray(input_time_index, dtype=float)
            if input_time_index is not None
            else _default_time(clean_arr.shape[0])
        )

    original_target_count = int(target_arr.shape[1])
    target_names_all = padded_feature_names(
        target_feature_names,
        original_target_count,
        prefix="Target",
    )
    target_name_norms_all = [_norm(name) for name in target_names_all]
    affected_name_values = [] if affected_feature_names is None else list(affected_feature_names)
    affected_name_keys_all = {
        _norm(name) for name in affected_name_values if _norm(name)
    }
    input_name_norms_all = [_norm(name) for name in input_names]
    input_index_by_name: dict[str, int] = {}
    for idx, norm_name in enumerate(input_name_norms_all):
        if norm_name and norm_name not in input_index_by_name:
            input_index_by_name[norm_name] = idx

    display_target_indices = list(range(original_target_count))
    selected_input_indices: list[int] | None = None
    channel_selection_note: str | None = None
    if original_target_count > max_display_channels:
        affected_target_indices = [
            idx
            for idx, norm_name in enumerate(target_name_norms_all)
            if norm_name and norm_name in affected_name_keys_all
        ]
        channel_errors = np.nanmean(np.abs(pert_prediction_arr - target_arr), axis=0)
        channel_spans = (
            np.maximum(np.nanmax(target_arr, axis=0), np.nanmax(pert_prediction_arr, axis=0))
            - np.minimum(np.nanmin(target_arr, axis=0), np.nanmin(pert_prediction_arr, axis=0))
        )

        def _descending_priority(value: float) -> float:
            return -float(value) if np.isfinite(value) else float("inf")

        dynamic_order = sorted(
            range(original_target_count),
            key=lambda idx: (
                _descending_priority(channel_errors[idx]),
                _descending_priority(channel_spans[idx]),
                int(idx),
            ),
        )
        ordered_indices: list[int] = []
        for idx in affected_target_indices + dynamic_order:
            if idx in ordered_indices:
                continue
            ordered_indices.append(int(idx))
            if len(ordered_indices) >= max_display_channels:
                break
        display_target_indices = ordered_indices
        selection_reason = (
            "affected channels first, then largest perturbed-error channels"
            if affected_target_indices
            else "largest perturbed-error channels"
        )
        channel_selection_note = (
            f"Showing {len(display_target_indices)} of {original_target_count} "
            f"target channels ({selection_reason})."
        )
        mapped_input_indices = [
            input_index_by_name[norm_name]
            for norm_name in (target_name_norms_all[idx] for idx in display_target_indices)
            if norm_name in input_index_by_name
        ]
        if mapped_input_indices:
            selected_input_indices = mapped_input_indices
        elif clean_arr is not None and clean_arr.shape[1] == original_target_count:
            selected_input_indices = list(display_target_indices)
        elif clean_arr is not None:
            clean_arr = None
            pert_arr = None
            input_names = []
            channel_selection_note = (
                channel_selection_note.rstrip(".")
                + " Input history omitted because the target/input channel mapping "
                "is unavailable."
            )

    if display_target_indices != list(range(original_target_count)):
        target_arr = target_arr[:, display_target_indices]
        pert_prediction_arr = pert_prediction_arr[:, display_target_indices]
        if clean_prediction_arr is not None:
            clean_prediction_arr = clean_prediction_arr[:, display_target_indices]
        target_feature_names = [target_names_all[idx] for idx in display_target_indices]
    else:
        target_feature_names = target_names_all

    if selected_input_indices is not None and clean_arr is not None:
        clean_arr = clean_arr[:, selected_input_indices]
        input_names = [
            input_names[idx]
            if idx < len(input_names) and input_names[idx]
            else f"Input {idx + 1}"
            for idx in selected_input_indices
        ]
        if pert_arr is not None:
            pert_arr = pert_arr[:, selected_input_indices]

    target_name_keys = {_norm(name) for name in target_feature_names if _norm(name)}
    target_name_lookup = {
        _norm(name): name
        for name in target_feature_names
        if isinstance(name, str) and name.strip()
    }
    affected_name_keys = {
        _norm(name) for name in affected_name_values if _norm(name)
    }
    normalized_inputs = [_norm(name) for name in input_names]

    n_input_channels = clean_arr.shape[1] if clean_arr is not None else 0
    many_inputs = n_input_channels > 10
    n_target_inputs = sum(1 for key in normalized_inputs if key in target_name_keys)
    input_target_opacity = _channel_opacity(n_target_inputs) if n_target_inputs > 1 else 1.0
    first_target_input = True

    input_traces: list[go.Scatter] = []
    if clean_arr is not None:
        for idx in range(clean_arr.shape[1]):
            name = input_names[idx] if idx < len(input_names) and input_names[idx] else f"Input {idx + 1}"
            norm_name = normalized_inputs[idx] if idx < len(normalized_inputs) else ""
            legend_label = name if name else f"Input {idx + 1}"
            is_target_channel = norm_name in target_name_keys
            if is_target_channel:
                legend_label = target_name_lookup.get(norm_name, legend_label)
                input_color = (
                    f"rgba(0, 0, 0, {input_target_opacity})"
                    if n_target_inputs > 1
                    else "rgb(0, 0, 0)"
                )
                input_width = 1 if many_inputs else 2
                line_style = dict(color=input_color, width=input_width)
                show_legend = first_target_input and n_target_inputs > 1
                if first_target_input and n_target_inputs > 5:
                    if original_target_count != n_target_inputs:
                        legend_label = (
                            f"Input ({n_target_inputs} of {original_target_count} target ch)"
                        )
                    else:
                        legend_label = f"Input ({n_target_inputs} target ch)"
                first_target_input = False
            else:
                line_style = dict(color="rgba(128, 128, 128, 0.55)", width=1)
                show_legend = not many_inputs
            input_traces.append(
                go.Scatter(
                    x=clean_time,
                    y=clean_arr[:, idx],
                    mode="lines",
                    name=legend_label,
                    line=line_style,
                    legendgroup="inputs_target" if is_target_channel else "inputs",
                    legendrank=50 + idx,
                    showlegend=show_legend,
                )
            )

    perturbed_traces: list[go.Scatter] = []
    if pert_arr is not None:
        if clean_time is None:
            clean_time = (
                np.asarray(input_time_index, dtype=float)
                if input_time_index is not None
                else _default_time(pert_arr.shape[0])
            )
        affected_indices = {
            idx for idx, key in enumerate(normalized_inputs) if key in affected_name_keys
        }
        diff_indices = set()
        if clean_arr is not None and clean_arr.shape == pert_arr.shape:
            diff_mask = np.any(np.abs(pert_arr - clean_arr) > 1e-8, axis=0)
            diff_indices = {int(i) for i, flagged in enumerate(diff_mask) if flagged}
        candidate_indices = sorted(affected_indices or diff_indices)
        n_perturbed = len(candidate_indices)
        many_perturbed = n_perturbed > 10
        pert_opacity = (
            _channel_opacity(n_perturbed, lo=0.3, hi=0.9)
            if many_perturbed
            else 0.9
        )
        pert_width = 1 if many_perturbed else 2
        pert_legend = f"Perturbed Input ({n_perturbed} ch)" if n_perturbed > 5 else None
        for pos, idx in enumerate(candidate_indices):
            if idx >= pert_arr.shape[1]:
                continue
            name = input_names[idx] if idx < len(input_names) and input_names[idx] else f"Input {idx + 1}"
            trace_name = pert_legend or f"{name} (Perturbed)"
            perturbed_traces.append(
                go.Scatter(
                    x=clean_time,
                    y=pert_arr[:, idx],
                    mode="lines",
                    name=trace_name,
                    line=dict(
                        color=f"rgba(214, 39, 40, {pert_opacity})",
                        width=pert_width,
                        dash="dashdot",
                    ),
                    legendgroup="perturbed",
                    legendrank=40 + pos,
                    showlegend=(pos == 0) if pert_legend else True,
                )
            )

    target_name_values = [] if target_feature_names is None else list(target_feature_names)
    target_names = [name for name in target_name_values if isinstance(name, str) and name]
    n_targets = target_arr.shape[1]
    multi_target = n_targets > 1

    if multi_target:
        target_opacity = _channel_opacity(n_targets)
        target_width = 1.5 if n_targets <= 10 else 1
        target_color = f"rgba(0, 0, 0, {target_opacity})"
        pert_pred_color = f"rgba(31, 119, 180, {target_opacity})"
        clean_pred_color = f"rgba(255, 127, 14, {target_opacity})"
        if original_target_count != n_targets:
            target_legend = f"Target ({n_targets} of {original_target_count} ch)"
            pert_pred_legend = (
                f"Prediction (Perturbed, {n_targets} of {original_target_count} ch)"
            )
            clean_pred_legend = (
                f"Prediction (Clean, {n_targets} of {original_target_count} ch)"
            )
        else:
            target_legend = f"Target ({n_targets} ch)" if n_targets > 5 else target_label
            pert_pred_legend = (
                f"Prediction (Perturbed, {n_targets} ch)"
                if n_targets > 5
                else perturbed_prediction_label
            )
            clean_pred_legend = (
                f"Prediction (Clean, {n_targets} ch)"
                if n_targets > 5
                else clean_prediction_label
            )
    else:
        target_color = "rgb(0, 0, 0)"
        pert_pred_color = "rgb(31, 119, 180)"
        clean_pred_color = "rgb(255, 127, 14)"
        target_width = 2
        target_legend = target_names[0] if target_names else target_label
        target_name = target_names[0] if target_names else None
        pert_pred_legend = (
            f"{target_name} (Pred Perturbed)"
            if target_name is not None
            else perturbed_prediction_label
        )
        clean_pred_legend = (
            f"{target_name} (Pred Clean)"
            if target_name is not None
            else clean_prediction_label
        )

    target_traces: list[go.Scatter] = []
    for idx in range(n_targets):
        target_traces.append(
            go.Scatter(
                x=output_time,
                y=target_arr[:, idx],
                mode="lines",
                name=target_legend,
                line=dict(color=target_color, width=target_width),
                legendgroup="target",
                legendrank=10,
                showlegend=(idx == 0),
            )
        )

    pert_prediction_traces: list[go.Scatter] = []
    for idx in range(pert_prediction_arr.shape[1]):
        pert_prediction_traces.append(
            go.Scatter(
                x=output_time,
                y=pert_prediction_arr[:, idx],
                mode="lines",
                name=pert_pred_legend,
                line=dict(color=pert_pred_color, width=target_width),
                legendgroup="prediction_perturbed",
                legendrank=20,
                showlegend=(idx == 0),
            )
        )

    clean_prediction_traces: list[go.Scatter] = []
    if clean_prediction_arr is not None:
        for idx in range(clean_prediction_arr.shape[1]):
            clean_prediction_traces.append(
                go.Scatter(
                    x=output_time,
                    y=clean_prediction_arr[:, idx],
                    mode="lines",
                    name=clean_pred_legend,
                    line=dict(
                        color=clean_pred_color,
                        width=target_width,
                        dash="dash",
                    ),
                    legendgroup="prediction_clean",
                    legendrank=15,
                    showlegend=(idx == 0),
                )
            )

    if scenario is None:
        raise ValueError("plot_forecast_extreme requires a perturbation scenario name.")
    if severity is None:
        raise ValueError("plot_forecast_extreme requires a severity value.")
    severity_value = float(severity)
    if not np.isfinite(severity_value):
        raise ValueError(f"Severity value must be finite, received {severity_value}.")
    if sample_score is not None and not np.isfinite(float(sample_score)):
        raise ValueError(f"Sample score must be finite, received {sample_score}.")

    fig = go.Figure()
    for trace in input_traces:
        fig.add_trace(trace)
    for trace in perturbed_traces:
        fig.add_trace(trace)
    for trace in clean_prediction_traces:
        fig.add_trace(trace)
    for trace in target_traces:
        fig.add_trace(trace)
    for trace in pert_prediction_traces:
        fig.add_trace(trace)

    default_title = f"Forecast vs Target | Scenario: {scenario}, Severity: {severity_value:.3f}"
    if sample_score is not None:
        metric_label = score_metric or "score"
        default_title += f" | {metric_label}={float(sample_score):.4f}"
    final_title = default_title if title is None else title
    if channel_selection_note:
        final_title = f"{final_title}<br><sup>{channel_selection_note}</sup>"
    title_line_count = _html_break_line_count(final_title)
    title_extra_lines = max(0, title_line_count - 1)
    figure_height = _FORECAST_BASE_HEIGHT_PX + (
        _FORECAST_TITLE_EXTRA_LINE_PX * title_extra_lines
    )
    top_margin = _FORECAST_BASE_TOP_MARGIN_PX + (
        _FORECAST_TITLE_EXTRA_LINE_PX * title_extra_lines
    )
    title_size = (
        _DIAG_TITLE_SIZE_COMPACT if title_line_count > 1 else _DIAG_TITLE_SIZE
    )

    if output_time.size:
        fig.add_vline(
            x=output_time[0] - 0.5,
            line=dict(color="rgba(0,0,0,0.35)", dash="dash"),
            annotation_text="Forecast start",
            annotation_position="top left",
        )

    fig.update_layout(
        title=final_title,
        template="plotly_white",
        autosize=True,
        height=figure_height,
        font=_DIAG_FONT,
        title_font=dict(size=title_size),
        margin=dict(l=10, r=10, t=top_margin, b=10),
        legend=dict(
            traceorder="normal",
            x=1.02,
            xanchor="left",
            y=1.0,
            yanchor="top",
        ),
    )
    fig.update_xaxes(
        title_text="Time",
        showgrid=True,
        gridcolor=_GRID_COLOR,
        ticks="outside",
        ticklen=6,
        tickwidth=1,
        mirror=True,
    )
    fig.update_yaxes(
        title_text="Value",
        showgrid=True,
        gridcolor=_GRID_COLOR,
        ticks="outside",
        ticklen=6,
        tickwidth=1,
        mirror=True,
    )
    return fig


# ---------------------------------------------------------------------------
# Coverage plots
# ---------------------------------------------------------------------------


def plot_testing_coverage_heatmap(
    coverage_fractions: Mapping[Tuple[str, str], Tuple[int, int]],
    *,
    title: str = "Testing Coverage (seen / expected)",
) -> go.Figure:
    """Build an architecture x method heatmap from coverage audit fractions.

    Parameters
    ----------
    coverage_fractions
        ``{(arch, method): (seen, expected), ...}`` as returned by the
        coverage audit.
    title
        Plot title.

    Returns
    -------
    go.Figure
    """
    if not coverage_fractions:
        raise ValueError("coverage_fractions is empty; nothing to plot.")

    architectures = sorted({arch for arch, _ in coverage_fractions})
    methods = sorted({method for _, method in coverage_fractions})
    n_arch, n_meth = len(architectures), len(methods)
    arch_idx = {a: i for i, a in enumerate(architectures)}
    meth_idx = {m: i for i, m in enumerate(methods)}

    z = np.full((n_arch, n_meth), np.nan)
    text = np.full((n_arch, n_meth), "", dtype=object)
    for (arch, method), (seen, expected) in coverage_fractions.items():
        r, c = arch_idx[arch], meth_idx[method]
        z[r, c] = seen / expected if expected > 0 else np.nan
        text[r, c] = f"{seen}/{expected}"

    fig = go.Figure(
        data=go.Heatmap(
            z=z, x=methods, y=architectures, text=text,
            texttemplate="%{text}",
            colorscale="RdYlGn", zmin=0, zmax=1,
            colorbar=dict(title="Coverage"),
        ),
    )
    fig.update_layout(
        title=dict(text=title, y=0.95, x=0.5, xanchor="center", yanchor="top"),
        xaxis=dict(title="Method", type="category"),
        yaxis=dict(title="Architecture", type="category", autorange="reversed"),
        width=max(400, 120 * n_meth + 160),
        height=max(300, 60 * n_arch + 140),
        font=_DIAG_FONT_COMPACT,
        title_font=dict(size=_DIAG_TITLE_SIZE_COMPACT),
        margin=dict(l=5, t=50, b=5, r=5),
        template="plotly_white",
    )
    return fig
