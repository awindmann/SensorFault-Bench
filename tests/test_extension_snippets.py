"""Runs the paired transfer-input snippet documented in docs/extending_methods.md."""

from __future__ import annotations

import os
import re
from pathlib import Path

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DOC_PATH = REPO_ROOT / "docs" / "extending_methods.md"


def _run_snippet() -> dict[str, object]:
    block = re.search(
        r"<!-- paired-transfer-inputs:start -->\n```python\n(.*?)```\n"
        r"<!-- paired-transfer-inputs:end -->",
        DOC_PATH.read_text(encoding="utf-8"),
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


def test_paired_transfer_snippet_builds_one_pair_per_transfer_family():
    namespace = _run_snippet()
    recipe = yaml.safe_load(
        (REPO_ROOT / "configs/pipelines/fault_augmentation.yaml").read_text(encoding="utf-8")
    )
    families = recipe["train_fault_profiles"]["holdout"]["scenarios"]
    clean = namespace["clean"]
    pairs = namespace["pairs"]

    assert [pair["family"] for pair in pairs] == families
    for pair in pairs:
        channels = list(pair["channels"])
        unaffected = [channel for channel in range(clean.shape[1]) if channel not in channels]

        assert pair["clean"] is clean
        assert pair["corrupted"].shape == clean.shape
        assert channels and set(channels) <= set(range(clean.shape[1]))
        assert torch.equal(pair["corrupted"][:, unaffected], clean[:, unaffected])
        assert not torch.equal(pair["corrupted"][:, channels], clean[:, channels]), pair["family"]
    assert namespace["unchanged_target"] is namespace["target"]
