from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

_PLOT_DIRECTION_TOKENS = frozenset(("minimize", "maximize"))
_PLOT_AXIS_FAMILY_TOKENS = frozenset(("numeric", "score", "delta"))


@dataclass(frozen=True)
class PlotSemanticsRecord:
    """Explicit per-axis plotting semantics shared across analysis and plotting."""

    direction: str
    axis_family: str
    neutral_value: float | None
    display_label: str

    def __post_init__(self) -> None:
        direction = str(self.direction).strip().lower()
        if direction not in _PLOT_DIRECTION_TOKENS:
            raise ValueError(
                "PlotSemanticsRecord.direction must be one of "
                f"{sorted(_PLOT_DIRECTION_TOKENS)}; got {self.direction!r}."
            )
        axis_family = str(self.axis_family).strip().lower()
        if axis_family not in _PLOT_AXIS_FAMILY_TOKENS:
            raise ValueError(
                "PlotSemanticsRecord.axis_family must be one of "
                f"{sorted(_PLOT_AXIS_FAMILY_TOKENS)}; got {self.axis_family!r}."
            )
        display_label = str(self.display_label).strip()
        if not display_label:
            raise ValueError("PlotSemanticsRecord.display_label must be non-empty.")
        neutral_value = self.neutral_value
        if axis_family == "delta":
            if neutral_value is None:
                raise ValueError(
                    "PlotSemanticsRecord with axis_family='delta' requires "
                    "neutral_value=0."
                )
            neutral_value = float(neutral_value)
            if not np.isfinite(neutral_value):
                raise ValueError(
                    "PlotSemanticsRecord.neutral_value must be finite when provided."
                )
            if not np.isclose(neutral_value, 0.0, rtol=0.0, atol=1e-12):
                raise ValueError(
                    "PlotSemanticsRecord with axis_family='delta' requires "
                    "neutral_value=0."
                )
        elif neutral_value is not None:
            neutral_value = float(neutral_value)
            if not np.isfinite(neutral_value):
                raise ValueError(
                    "PlotSemanticsRecord.neutral_value must be finite when provided."
                )
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "axis_family", axis_family)
        object.__setattr__(self, "display_label", display_label)
        object.__setattr__(self, "neutral_value", neutral_value)


def require_plot_semantics_mapping(
    semantics_by_key: Mapping[str, PlotSemanticsRecord],
    *,
    required_keys: Sequence[str],
    context: str,
) -> dict[str, PlotSemanticsRecord]:
    """Require one complete plot-semantics mapping for the requested metric keys."""
    if not isinstance(semantics_by_key, Mapping):
        raise ValueError(f"{context} plot semantics must be a mapping.")
    resolved: dict[str, PlotSemanticsRecord] = {}
    missing: list[str] = []
    for key in required_keys:
        if key not in semantics_by_key:
            missing.append(str(key))
            continue
        record = semantics_by_key[key]
        if not isinstance(record, PlotSemanticsRecord):
            raise ValueError(
                f"{context} plot semantics for '{key}' must be a PlotSemanticsRecord."
            )
        resolved[str(key)] = record
    if missing:
        raise ValueError(
            f"{context} plot semantics are missing required keys: {', '.join(missing)}."
        )
    return resolved
