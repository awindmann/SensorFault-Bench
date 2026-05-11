from __future__ import annotations

import re
import subprocess
import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_BENCHMARK_DOCS = {
    "README.md",
    "data/README.md",
    "docs/reproducing_benchmark.md",
    "docs/extending_datasets.md",
    "docs/extending_models.md",
    "docs/extending_methods.md",
    "THIRD_PARTY_NOTICES.md",
}
BENCHMARK_ENTRYPOINTS = {
    "run_training.py",
    "run_testing.py",
    "run_analysis.py",
}
ALLOWED_MANIFESTS = {"pyproject.toml", "uv.lock"}
FORBIDDEN_DEPENDENCY_MANIFEST_FILENAMES = {
    "constraints.txt",
    "environment.yaml",
    "environment.yml",
    "Pipfile",
    "Pipfile.lock",
    "poetry.lock",
    "poetry.toml",
    "requirements.txt",
    "requirements-dev.txt",
    "requirements-test.txt",
    "requirements_full.txt",
    "setup.cfg",
    "setup.py",
}
MAX_REPOSITORY_FILE_BYTES = 5 * 1024 * 1024


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def _repo_files() -> tuple[str, ...]:
    """Return the current repository-surface file list.

    This gate audits the current non-ignored source tree. Files that are absent
    from the candidate tree are outside this check.
    """
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    paths: list[str] = []
    for raw_path in result.stdout.splitlines():
        path = raw_path.strip()
        if not path:
            continue
        full_path = REPO_ROOT / path
        if full_path.is_file():
            paths.append(path)
    return tuple(sorted(dict.fromkeys(paths)))


def test_repository_surface_file_list_has_no_forbidden_paths_or_runtime_state():
    repo_files = _repo_files()
    forbidden_names = {
        ".DS_Store",
    }
    forbidden_prefixes = (
        ".pytest_cache/",
        ".ruff_cache/",
        ".venv/",
        ".vscode/",
        "data/processed/",
        "data/raw/",
        "exports/",
        "figures/",
        "logs/",
        "reports/",
        "runs/",
        "supplementary/",
        "tables/",
    )
    forbidden_suffixes = (".ckpt", ".tar.gz", ".tar.zst", ".zip")

    for path in repo_files:
        assert path not in forbidden_names
        assert not path.startswith(forbidden_prefixes), path
        assert not path.endswith(forbidden_suffixes), path

    root_run_scripts = {
        path
        for path in repo_files
        if Path(path).parent == Path(".")
        and Path(path).name.startswith("run_")
        and Path(path).suffix == ".py"
    }
    assert root_run_scripts == BENCHMARK_ENTRYPOINTS

    dependency_manifest_offenders = [
        path
        for path in repo_files
        if Path(path).name in FORBIDDEN_DEPENDENCY_MANIFEST_FILENAMES
        or (
            Path(path).name.startswith("requirements")
            and Path(path).suffix == ".txt"
        )
    ]
    assert dependency_manifest_offenders == []
    assert ALLOWED_MANIFESTS.issubset(repo_files)


def test_repository_surface_has_required_benchmark_docs_and_entrypoints():
    repo_files = set(_repo_files())

    for path in REQUIRED_BENCHMARK_DOCS | BENCHMARK_ENTRYPOINTS:
        assert path in repo_files

    ignored_lines = {
        line.strip()
        for line in _read(".gitignore").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    for docs_path in REQUIRED_BENCHMARK_DOCS:
        assert docs_path not in ignored_lines
    assert "docs/" not in ignored_lines
    assert "/docs/" not in ignored_lines


def test_gitignore_excludes_runtime_cache_and_data_state_paths():
    forbidden_local_paths = (
        ".pytest_cache/v/cache/nodeids",
        ".ruff_cache/CACHEDIR.TAG",
        ".venv/pyvenv.cfg",
        ".vscode/launch.json",
        "config/meta_analysis_args.yaml",
        "data/processed/sample.parquet",
        "data/raw/source.csv",
        "figures/figure.pdf",
        "runs/mlruns.db",
        "local-output.tar.zst",
        "source-archive.tar.gz",
        "tables/analysis.csv",
    )
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", *forbidden_local_paths],
        cwd=REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    ignored = set(result.stdout.splitlines())
    assert ignored == set(forbidden_local_paths)


def test_repository_surface_files_are_size_bounded():
    oversized = [
        (path, (REPO_ROOT / path).stat().st_size)
        for path in _repo_files()
        if (REPO_ROOT / path).stat().st_size > MAX_REPOSITORY_FILE_BYTES
    ]

    assert oversized == []


def test_license_file_and_package_metadata_are_distribution_ready():
    license_path = REPO_ROOT / "LICENSE"
    assert license_path.is_file(), "Benchmark requires a selected root LICENSE."

    license_text = license_path.read_text(encoding="utf-8").strip()
    assert license_text
    placeholder_terms = ("TODO", "TBD", "selected license", "placeholder")
    lowered_license = license_text.lower()
    for term in placeholder_terms:
        assert term not in lowered_license
    assert "Apache License" in license_text
    assert "Version 2.0" in license_text
    assert re.search(r"^Copyright\s+\d{4}\s+\S", license_text, re.MULTILINE) is None

    project = tomllib.loads(_read("pyproject.toml"))["project"]
    assert project["license"] == "Apache-2.0"
    assert project["license-files"] == ["LICENSE"]

    readme = _read("README.md")
    assert "LICENSE" in readme
    assert "Apache-2.0" in readme
    assert "License blocker" not in readme
