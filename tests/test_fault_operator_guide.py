"""Runs the calibrated-pairs snippet of docs/using_fault_operators.md."""

from __future__ import annotations

import os
import re
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
GUIDE_PATH = REPO_ROOT / "docs" / "using_fault_operators.md"


def _run_calibrated_pairs_snippet() -> dict[str, object]:
    block = re.search(
        r"<!-- calibrated-pairs:start -->\n```python\n(.*?)```\n<!-- calibrated-pairs:end -->",
        GUIDE_PATH.read_text(encoding="utf-8"),
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
    return namespace


def test_calibrated_pairs_snippet_applies_calibrated_operators():
    namespace = _run_calibrated_pairs_snippet()
    operators = namespace["operators"]
    pairs = namespace["pairs"]
    windows = namespace["windows"]

    assert operators["drift"].max_off == 1.5
    assert operators["stuck_sensor"].max_frac == 0.3
    assert len(pairs) == len(windows) * len(operators)
    n_channels = windows.shape[-1]
    for pair in pairs:
        channels = list(pair["channels"])
        unaffected = [channel for channel in range(n_channels) if channel not in channels]

        assert 1 <= len(channels) <= 2, pair["operator"]
        assert torch.equal(pair["corrupted"][:, unaffected], pair["clean"][:, unaffected])
        assert not torch.equal(pair["corrupted"][:, channels], pair["clean"][:, channels])
