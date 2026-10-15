"""Checks that the citation metadata agrees across CITATION.cff, README.md, and pyproject.toml."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_CFF_KEYS = {
    "cff-version",
    "message",
    "title",
    "authors",
    "version",
    "license",
    "repository-code",
    "preferred-citation",
}


def _citation() -> dict[str, object]:
    return yaml.safe_load((REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8"))


def _readme() -> str:
    return (REPO_ROOT / "README.md").read_text(encoding="utf-8")


def _project() -> dict[str, object]:
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]


def _bibtex_fields() -> dict[str, str]:
    block = re.search(r"```bibtex\n(@inproceedings\{[^,]+,\n.*?\n\})\n```", _readme(), re.S)
    assert block is not None
    return dict(re.findall(r"^\s+(\w+)=\{(.*)\},?$", block.group(1), flags=re.MULTILINE))


def _names(authors: list[dict[str, str]]) -> list[str]:
    return [f"{author['given-names']} {author['family-names']}" for author in authors]


def test_citation_file_has_required_keys_and_matches_package_metadata():
    citation = _citation()
    project = _project()

    assert REQUIRED_CFF_KEYS <= set(citation)
    assert citation["cff-version"] == "1.2.0"
    assert citation["version"] == project["version"]
    assert citation["license"] == project["license"]
    assert citation["repository-code"] == project["urls"]["Repository"]
    assert _names(citation["authors"]) == [author["name"] for author in project["authors"]]
    assert all(author["affiliation"] for author in citation["authors"])


def test_readme_bibtex_matches_preferred_citation():
    preferred = _citation()["preferred-citation"]
    fields = _bibtex_fields()
    bibtex_authors = [
        " ".join(reversed(name.split(", ", 1))) for name in fields["author"].split(" and ")
    ]

    assert preferred["type"] == "conference-paper"
    assert fields["title"] == preferred["title"]
    assert fields["booktitle"] == preferred["collection-title"]
    assert int(fields["year"]) == preferred["year"]
    assert fields["url"] == preferred["url"] == _project()["urls"]["Paper"]
    assert bibtex_authors == _names(preferred["authors"]) == _names(_citation()["authors"])


def test_readme_author_line_matches_citation_order():
    author_line = re.search(r'<p align="center">(Alexander [^<]+)<br>', _readme())

    assert author_line is not None
    readme_authors = [
        re.sub(r"[¹²³⁴⁵˒]", "", name).strip() for name in author_line.group(1).split(" · ")
    ]
    assert readme_authors == _names(_citation()["authors"])


def test_readme_badges_match_package_metadata():
    readme = _readme()
    project = _project()
    release = re.search(r"img\.shields\.io/badge/release-v(\d+\.\d+\.\d+)-", readme)
    python = re.search(r"img\.shields\.io/badge/python-3\.(\d+)%20to%203\.(\d+)-", readme)

    assert release is not None
    assert release.group(1) == project["version"]
    assert python is not None
    lowest, highest = map(int, python.groups())
    assert project["requires-python"] == f">=3.{lowest},<3.{highest + 1}"


def test_changelog_names_the_package_version():
    changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    versions = re.findall(r"^## (\d+\.\d+\.\d+)$", changelog, flags=re.MULTILINE)

    assert versions[0] == _project()["version"]
    assert versions == sorted(versions, key=lambda value: tuple(map(int, value.split("."))))[::-1]
