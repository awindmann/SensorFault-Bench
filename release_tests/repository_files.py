"""File listing and text checks shared by the release tests.

The file listing calls `git ls-files`, so the checks that use it need a git
checkout and fail in a copy without `.git`.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from scripts.build_paper_artifact import PRIVATE_VALUE_PATTERNS


REPO_ROOT = Path(__file__).resolve().parents[1]
MAX_REPOSITORY_FILE_BYTES = 5 * 1024 * 1024
# Documentation may link the repositories in KNOWN_PUBLIC_GITHUB_REPOS, so the
# allowlist replaces the two GitHub locator patterns of the artifact builder.
GITHUB_LOCATOR_LABELS = (
    "GitHub SSH repository locator",
    "GitHub HTTPS repository locator",
)
GITHUB_REPO_URL_REGEX = re.compile(
    r"(?i)\b(?:git@github\.com:|https?://github\.com/)"
    r"(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+)(?:\.git)?\b"
)
KNOWN_PUBLIC_GITHUB_REPOS = {
    "amazon-science/chronos-forecasting",
    "arekavandi/certified_adv_rregression",
    "awindmann/sensorfault-bench",
    "bethgelab/robust-detection-benchmark",
    "google-research/google-research",
    "hendrycks/robustness",
    "jbrandt97/faultstofeatures",
    "jonbarron/robust_loss_pytorch",
    "luodhhh/moderntcn",
    "madrylab/mnist_challenge",
    "modestyachts/imagenet-testbed",
    "pandas-dev/pandas",
    "tetrzim/robust-probabilistic-forecasting",
    "thuml/time-series-library",
    "ts-kim/revin",
    "zhouhaoyi/etdataset",
}


def _text_value_patterns() -> tuple[tuple[str, re.Pattern[str]], ...]:
    missing = set(GITHUB_LOCATOR_LABELS) - set(PRIVATE_VALUE_PATTERNS)
    if missing:
        raise KeyError(
            "PRIVATE_VALUE_PATTERNS no longer defines the GitHub locator "
            f"patterns {sorted(missing)}."
        )
    return tuple(
        (label, regex)
        for label, regex in PRIVATE_VALUE_PATTERNS.items()
        if label not in GITHUB_LOCATOR_LABELS
    )


TEXT_VALUE_PATTERNS = _text_value_patterns()


def private_value_offenders(text: str) -> tuple[str, ...]:
    """Return the labels of the private-value classes that occur in the text."""
    offenders = [label for label, regex in TEXT_VALUE_PATTERNS if regex.search(text)]
    for match in GITHUB_REPO_URL_REGEX.finditer(text):
        owner = match.group("owner").lower()
        repo = match.group("repo").removesuffix(".git").lower()
        if f"{owner}/{repo}" not in KNOWN_PUBLIC_GITHUB_REPOS:
            offenders.append("unapproved GitHub repository URL")
            break
    return tuple(offenders)


def list_repo_files() -> tuple[str, ...]:
    """Return the tracked and untracked files that git does not ignore.

    This check audits the current non-ignored source tree. Files that are
    absent from the working tree are outside it.
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
