from __future__ import annotations

import re
from pathlib import Path

import yaml

from config_loader import load_defaults
from release_tests.repository_files import private_value_offenders


REPO_ROOT = Path(__file__).resolve().parents[1]
DOC_PATHS = (
    "README.md",
    "data/README.md",
    "docs/benchmark_protocol.md",
    "docs/reproducing_benchmark.md",
    "docs/sensitivity_analyses.md",
    "docs/extending_datasets.md",
    "docs/extending_models.md",
    "docs/extending_methods.md",
    "docs/using_fault_operators.md",
    "docs/forecast_plot_replay.md",
    "paper_artifact/README.md",
    "paper_artifact/R/README.md",
    "THIRD_PARTY_NOTICES.md",
)
# The README links every file of DOC_PATHS. The text checks also read two
# documents that the README does not link.
TEXT_DOC_PATHS = (
    *DOC_PATHS,
    "paper_artifact/supplementary_tables/README.md",
    "CHANGELOG.md",
)
README_ASSET_MAX_BYTES = 1024 * 1024
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)\)")
HTML_REFERENCE = re.compile(r'(?:href|src)="([^"]+)"')
FENCED_BLOCK = re.compile(r"^\s*```.*?^\s*```", re.MULTILINE | re.DOTALL)


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


def _heading_anchor(heading: str) -> str:
    text = re.sub(r"[`*_]", "", heading.strip().lower())
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def _anchors(path: Path) -> set[str]:
    text = FENCED_BLOCK.sub("", path.read_text(encoding="utf-8"))
    return {
        _heading_anchor(match.group(1))
        for match in re.finditer(r"^#{1,6} (.+)$", text, flags=re.MULTILINE)
    }


def _relative_references(path: str) -> list[str]:
    text = FENCED_BLOCK.sub("", _read(path))
    references = MARKDOWN_LINK.findall(text) + HTML_REFERENCE.findall(text)
    return [
        reference
        for reference in references
        if not reference.startswith(("http://", "https://", "mailto:"))
    ]


def test_relative_doc_links_and_anchors_resolve():
    for path in DOC_PATHS:
        doc_dir = (REPO_ROOT / path).parent
        for reference in _relative_references(path):
            target, _, anchor = reference.partition("#")
            resolved = (doc_dir / target).resolve() if target else REPO_ROOT / path
            assert resolved.exists(), f"{path}: {reference}"
            if anchor:
                assert resolved.suffix == ".md", f"{path}: {reference}"
                assert anchor in _anchors(resolved), f"{path}: {reference}"


def test_readme_images_are_small_pngs_with_alt_text():
    readme = _read("README.md")
    images = re.findall(r"<img src=\"(docs/assets/[^\"]+)\"[^>]*alt=\"([^\"]+)\"", readme)

    assert {source for source, _alt in images} == {
        "docs/assets/sensorfault-bench-overview.png",
        "docs/assets/clean-mse-vs-worst-scenario-degradation.png",
        "docs/assets/sensor-faults-in-action.png",
    }
    for source, alt in images:
        asset = REPO_ROOT / source
        assert asset.read_bytes()[:8] == PNG_SIGNATURE, source
        assert asset.stat().st_size <= README_ASSET_MAX_BYTES, source
        assert len(alt.split()) >= 10, source


def test_markdown_commands_reference_shipped_entrypoints():
    command_pattern = re.compile(r"(?:uv run )?python ([A-Za-z0-9_./-]+\.py)")

    for path in TEXT_DOC_PATHS:
        for match in command_pattern.finditer(_read(path)):
            target = match.group(1)
            assert (REPO_ROOT / target).exists(), f"{path}: {target}"


def test_reproducing_benchmark_docs_match_default_method_order():
    defaults = load_defaults()
    doc = _read("docs/reproducing_benchmark.md")

    assert "uv run python run_training.py" in doc
    assert "uv run python run_testing.py" in doc
    assert "uv run python run_analysis.py" in doc
    documented_order = re.findall(
        r"^\d+\. `([^`]+)`$",
        doc,
        flags=re.MULTILINE,
    )
    assert documented_order == defaults["BENCHMARK_METHODS"]


def test_benchmark_docs_are_private_value_free():
    for path in (*TEXT_DOC_PATHS, "CITATION.cff", "pyproject.toml", ".github/workflows/tests.yml"):
        text = _read(path)
        assert private_value_offenders(text) == (), path


def test_data_guide_records_external_source_terms_and_code_license_split():
    guide = _read("data/README.md")

    for phrase in (
        "Raw and processed benchmark datasets are not redistributed",
        "https://github.com/zhouhaoyi/ETDataset",
        "https://huggingface.co/datasets/thuml/Time-Series-Library/blob/main/traffic/traffic.csv",
        "https://doi.org/10.5281/zenodo.4656132",
        "https://doi.org/10.24432/C5RK5G",
        "https://doi.org/10.5281/zenodo.16807304",
        "CC BY-SA 4.0",
        "repository code license does not grant rights to external datasets",
    ):
        assert phrase in guide


def test_data_guide_names_benchmark_validation_surface():
    guide = _read("data/README.md")

    for phrase in (
        "scripts/benchmark_dataset_contracts.py",
        "row-count and channel-count",
    ):
        assert phrase in guide


def test_third_party_notices_record_code_and_dataset_sources():
    notices = _read("THIRD_PARTY_NOTICES.md")

    for phrase in (
        "Time-Series-Library",
        "MIT License",
        "Copyright (c) 2021 THUML @ Tsinghua University",
        "TSMixer",
        "Apache-2.0",
        "Copyright 2026 The Google Research Authors",
        "PyTorch Lightning style",
        "Chronos Forecasting",
        "chronos-forecasting==2.2.2",
        "Randomized Smoothing For Regression",
        "Adaptive Robust Loss",
        "Robustness Metric Comparisons",
        "mCE-style reference-normalized diagnostics",
        "mPC-style and rPC-style mean perturbed performance diagnostics",
        "Taori-style effective robustness residuals",
        "Plotting code is repository-owned",
        "Michaelis et al., 2019",
        "Sensor Failure Modes",
        "data/README.md",
    ):
        assert phrase in notices
