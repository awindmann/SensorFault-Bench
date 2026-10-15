"""Ties docs/using_fault_operators.md to the operator registry, configs, and citation metadata."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from config_loader import load_defaults
from data.perturbations import PERTURBATION_REGISTRY


REPO_ROOT = Path(__file__).resolve().parents[1]
GUIDE_PATH = REPO_ROOT / "docs" / "using_fault_operators.md"


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def _operator_groups() -> dict[str, list[str]]:
    block = re.search(
        r"<!-- operator-groups:start -->\n(.*?)<!-- operator-groups:end -->",
        GUIDE_PATH.read_text(encoding="utf-8"),
        re.S,
    )
    assert block is not None
    groups: dict[str, list[str]] = {}
    for line in block.group(1).strip().splitlines()[2:]:
        group, operators, _role = (cell.strip() for cell in line.strip("|").split("|"))
        groups[group] = re.findall(r"`([^`]+)`", operators)
    return groups


def _sensitivity_scenarios() -> set[str]:
    commands = re.findall(r"--perturbation-scenarios ([a-z_ ]+)", _read("docs/sensitivity_analyses.md"))
    scored = set(load_defaults()["PERTURBATION_SCENARIOS"])
    return {name for command in commands for name in command.split()} - scored


def _f2f_adapted_operators() -> list[str]:
    source = _read("data/perturbations.py")
    classes = re.findall(r"^# F2F:.*\nclass (\w+)", source, flags=re.MULTILINE)
    name_by_class = {cls.__name__: name for name, cls in PERTURBATION_REGISTRY.items()}
    return [name_by_class[cls] for cls in classes]


def test_operator_groups_partition_the_registry_and_follow_their_owners():
    groups = _operator_groups()
    recipe = yaml.safe_load(_read("configs/pipelines/fault_augmentation.yaml"))
    listed = [name for operators in groups.values() for name in operators]

    assert sorted(listed) == sorted(PERTURBATION_REGISTRY)
    assert groups["Scored scenarios"] == load_defaults()["PERTURBATION_SCENARIOS"]
    assert groups["Transfer fault families"] == recipe["train_fault_profiles"]["holdout"]["scenarios"]
    assert set(groups["Sensitivity scenarios"]) == _sensitivity_scenarios()
    assert set(groups["Discrete-state operators"]) == {
        name for name, cls in PERTURBATION_REGISTRY.items() if cls.channel_scope == "discrete"
    }


def test_failure_mode_attribution_matches_source_comments_and_citation_metadata():
    adapted = _f2f_adapted_operators()
    guide = _normalized(GUIDE_PATH.read_text(encoding="utf-8"))
    notices = _normalized(_read("THIRD_PARTY_NOTICES.md"))
    citation = yaml.safe_load(_read("CITATION.cff"))
    (reference,) = citation["references"]
    guide_sentence = re.search(r"The operators ([^.]*) adapt failure modes from Brandt", guide)
    notices_sentence = re.search(r"adapted for the operators ([^.]*)\. These operators", notices)

    assert adapted and "packet_loss" not in adapted
    for sentence in (guide_sentence, notices_sentence):
        assert sentence is not None
        assert sorted(re.findall(r"`([^`]+)`", sentence.group(1))) == sorted(adapted)
    assert reference["authors"][0]["family-names"] == "Brandt"
    assert f"https://doi.org/{reference['doi']}" in guide
    assert reference["title"] in guide
    assert "fault operators" in citation["message"]
