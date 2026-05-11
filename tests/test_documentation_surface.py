from __future__ import annotations

import re
from pathlib import Path

import yaml

from config_loader import load_defaults


REPO_ROOT = Path(__file__).resolve().parents[1]
DOC_PATHS = (
    "README.md",
    "data/README.md",
    "docs/reproducing_benchmark.md",
    "docs/extending_datasets.md",
    "docs/extending_models.md",
    "docs/extending_methods.md",
    "THIRD_PARTY_NOTICES.md",
)


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def _benchmark_scope() -> dict[str, object]:
    return yaml.safe_load(_read("configs/benchmark_scope.yaml"))


def test_required_benchmark_docs_exist_and_are_linked_from_readme():
    readme = _read("README.md")

    for path in DOC_PATHS:
        assert (REPO_ROOT / path).exists(), path
    for path in DOC_PATHS[1:]:
        assert path in readme


def test_docs_name_benchmark_entrypoints_and_configured_scope():
    docs_blob = "\n".join(_read(path) for path in DOC_PATHS)
    benchmark_scope = _benchmark_scope()

    for entrypoint in ("run_training.py", "run_testing.py", "run_analysis.py"):
        assert entrypoint in docs_blob
        assert (REPO_ROOT / entrypoint).exists()
    for command in (
        "uv run python run_training.py",
        "uv run python run_testing.py",
        "uv run python run_analysis.py",
    ):
        assert command in docs_blob
    for dataset in benchmark_scope["datasets"]["display_order"]:
        assert dataset in docs_blob
    for architecture in benchmark_scope["architectures"]["display_order"]:
        assert architecture in docs_blob
    for method in load_defaults()["BENCHMARK_METHODS"]:
        assert method in docs_blob


def test_extension_docs_name_canonical_owner_paths():
    dataset_doc = _read("docs/extending_datasets.md")
    model_doc = _read("docs/extending_models.md")
    method_doc = _read("docs/extending_methods.md")

    for phrase in (
        "data/datasets/specs.py",
        "configs/dataset_windows.yaml",
        "data_config_signature",
    ):
        assert phrase in dataset_doc
    for phrase in (
        "models/__init__.py",
        "configs/baseline_hparams.yaml",
    ):
        assert phrase in model_doc
    for phrase in (
        "configs/pipelines/",
        "BENCHMARK_METHODS",
        "configs/reporting/core_figures.yaml",
    ):
        assert phrase in method_doc


def test_markdown_commands_reference_shipped_entrypoints():
    command_pattern = re.compile(r"(?:uv run )?python ([A-Za-z0-9_./-]+\.py)")

    for path in DOC_PATHS:
        for match in command_pattern.finditer(_read(path)):
            target = match.group(1)
            assert (REPO_ROOT / target).exists(), f"{path}: {target}"
