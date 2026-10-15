"""Render the generated result blocks of README.md from the bundled paper artifact.

README.md contains three generated blocks:

- ``key-findings``: the headline findings with numbers from the results of
  the default configuration and the paired method deltas.
- ``baseline-results``: per-dataset leaders and the full baseline
  architecture table.
- ``method-results``: the mean paired deltas of every robustness-improvement
  method on every dataset.

Each block sits between ``<!-- <name>:start -->`` and ``<!-- <name>:end -->``.
The renderer raises when the bundled tables no longer support a ranking, sign,
or dominant scenario class that a finding sentence states. The clean-MSE
comparison of Chronos-2 with the last-value naive forecaster has no numeric
threshold, so the finding prints both values.

Usage from the repository root::

    uv run python scripts/render_readme_results.py          # rewrite the blocks
    uv run python scripts/render_readme_results.py --check  # fail on a stale README
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.benchmark_dataset_contracts import CONTRACTS_BY_KEY

README_PATH = Path("README.md")
MAIN_TABLE_PATH = Path("paper_artifact/paper_inputs/main_table_backbone.csv")
METHOD_DELTA_PATH = Path(
    "paper_artifact/tables/default_configuration/pipeline_method_delta_results.csv"
)
SELECTED_HPARAMS_PATH = Path(
    "paper_artifact/supplementary_tables/selected_clean_validation_hyperparameters.csv"
)
BENCHMARK_SCOPE_PATH = Path("configs/benchmark_scope.yaml")
CORE_FIGURES_PATH = Path("configs/reporting/core_figures.yaml")

FOUNDATION_MODEL = "Chronos2"
NAIVE_REFERENCE = "SeasonalNaive"
REFERENCE_TRAINING_SEED = "42"
BLOCK_NAMES = ("key-findings", "baseline-results", "method-results")
NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven"}
ORDINAL_PREFIXES = {1: "", 2: "second-"}


@dataclass(frozen=True)
class ArchitectureResult:
    dataset: str
    architecture: str
    mse_clean: float
    d_w: float
    mse_w: float
    worst_scenario: str


def _read_csv(repo_root: Path, path: Path) -> list[dict[str, str]]:
    full_path = repo_root / path
    if not full_path.is_file():
        raise FileNotFoundError(f"README source table is missing: {path}")
    with full_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_yaml(repo_root: Path, path: Path) -> dict[str, object]:
    return yaml.safe_load((repo_root / path).read_text(encoding="utf-8"))


def _labels_from_pairs(entries: list[str], *, key: str) -> dict[str, str]:
    labels: dict[str, str] = {}
    for entry in entries:
        name, separator, label = entry.partition("=")
        if not separator or not name or not label:
            raise ValueError(f"{key} entry must have the form 'key=Label', got {entry!r}.")
        labels[name] = label
    return labels


def _display_labels(repo_root: Path) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    scope = _read_yaml(repo_root, BENCHMARK_SCOPE_PATH)
    figures = _read_yaml(repo_root, CORE_FIGURES_PATH)
    datasets = dict(scope["datasets"]["display_labels"])
    architectures = dict(scope["architectures"]["display_labels"])
    scenarios = _labels_from_pairs(
        list(figures["CORE_SCENARIO_DISPLAY"]),
        key="CORE_SCENARIO_DISPLAY",
    )
    return datasets, architectures, scenarios


def _scenario_classes(repo_root: Path) -> dict[str, str]:
    figures = _read_yaml(repo_root, CORE_FIGURES_PATH)
    classes: dict[str, str] = {}
    for scenario_class, members in _labels_from_pairs(
        list(figures["CORE_SCENARIO_GROUPS"]),
        key="CORE_SCENARIO_GROUPS",
    ).items():
        for scenario in members.split(","):
            classes[scenario] = scenario_class
    return classes


def _architecture_results(repo_root: Path) -> list[ArchitectureResult]:
    rows = _read_csv(repo_root, MAIN_TABLE_PATH)
    if not rows:
        raise ValueError(f"{MAIN_TABLE_PATH} has no rows.")
    ordered = sorted(rows, key=lambda row: (int(row["dataset_order"]), int(row["backbone_order"])))
    return [
        ArchitectureResult(
            dataset=row["dataset"],
            architecture=row["backbone"],
            mse_clean=float(row["MSE_c"]),
            d_w=float(row["D_w"]),
            mse_w=float(row["MSE_w"]),
            worst_scenario=row["worst_scenario"],
        )
        for row in ordered
    ]


def _datasets_in_order(results: list[ArchitectureResult]) -> list[str]:
    return list(dict.fromkeys(result.dataset for result in results))


def _unique_min(rows: list, value, *, context: str):
    best = min(value(row) for row in rows)
    winners = [row for row in rows if value(row) == best]
    if len(winners) != 1:
        raise ValueError(f"{context}: exact tie for the lowest value {best}.")
    return winners[0]


def _unique_max(rows: list, value, *, context: str):
    best = max(value(row) for row in rows)
    winners = [row for row in rows if value(row) == best]
    if len(winners) != 1:
        raise ValueError(f"{context}: exact tie for the largest value {best}.")
    return winners[0]


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _fmt_delta(value: float) -> str:
    """Format a delta without a sign below the display precision."""
    text = _fmt(value)
    return "0.000" if text == "-0.000" else text


def _dataset_rows(results: list[ArchitectureResult], dataset: str) -> list[ArchitectureResult]:
    return [result for result in results if result.dataset == dataset]


def _single_target_datasets() -> set[str]:
    return {key for key, contract in CONTRACTS_BY_KEY.items() if contract.target_alias != "all"}


def _dominant_worst_scenario_class(
    results: list[ArchitectureResult],
    dataset: str,
    scenario_classes: dict[str, str],
) -> str:
    """Return the class holding the worst scenario of most baseline architectures."""
    counts: dict[str, int] = {}
    rows = _dataset_rows(results, dataset)
    for row in rows:
        scenario_class = scenario_classes[row.worst_scenario]
        counts[scenario_class] = counts.get(scenario_class, 0) + 1
    scenario_class, count = max(counts.items(), key=lambda item: item[1])
    if 2 * count <= len(rows):
        raise ValueError(f"{dataset}: no scenario class holds most worst scenarios ({counts}).")
    return scenario_class


def _selected_naive_periods(repo_root: Path) -> dict[str, int]:
    periods: dict[str, int] = {}
    for row in _read_csv(repo_root, SELECTED_HPARAMS_PATH):
        if row["training_seed"] != REFERENCE_TRAINING_SEED:
            continue
        if row["model_architecture"] != NAIVE_REFERENCE:
            continue
        periods[row["dataset"]] = int(json.loads(row["hyperparameters_json"])["season_length"])
    return periods


def _mean_method_deltas(repo_root: Path) -> dict[tuple[str, str], tuple[float, int]]:
    deltas: dict[tuple[str, str], tuple[float, int]] = {}
    for row in _read_csv(repo_root, METHOD_DELTA_PATH):
        deltas[(row["dataset"], row["robustness_method"])] = (
            float(row["delta_D_w_mean"]),
            int(row["count"]),
        )
    return deltas


def render_baseline_results(repo_root: Path = REPO_ROOT) -> str:
    """Render the baseline-results block from the bundled main table."""
    dataset_labels, architecture_labels, scenario_labels = _display_labels(repo_root)
    results = _architecture_results(repo_root)
    lines = [
        "| Dataset | Lowest clean MSE | Lowest worst-scenario degradation | "
        "Lowest worst-scenario fault-time MSE |",
        "| --- | --- | --- | --- |",
    ]
    for dataset in _datasets_in_order(results):
        rows = _dataset_rows(results, dataset)
        cells = []
        for value, name in (
            (lambda row: row.mse_clean, "clean MSE"),
            (lambda row: row.d_w, "worst-scenario degradation"),
            (lambda row: row.mse_w, "worst-scenario fault-time MSE"),
        ):
            leader = _unique_min(rows, value, context=f"{dataset} {name}")
            cells.append(f"{architecture_labels[leader.architecture]} ({_fmt(value(leader))})")
        lines.append(f"| {dataset_labels[dataset]} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "<details>",
        f"<summary>All {len(results)} baseline forecasting models</summary>",
        "",
        "| Dataset | Architecture | Clean MSE | Worst-scenario degradation | "
        "Worst-scenario fault-time MSE | Worst scenario |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in results:
        lines.append(
            f"| {dataset_labels[row.dataset]} | {architecture_labels[row.architecture]} | "
            f"{_fmt(row.mse_clean)} | {_fmt(row.d_w)} | "
            f"{_fmt(row.mse_w)} | {scenario_labels[row.worst_scenario]} |"
        )
    lines += ["", "</details>"]
    return "\n".join(lines)


def render_key_findings(repo_root: Path = REPO_ROOT) -> str:
    """Render the key-findings block and check the premises of each sentence."""
    dataset_labels, architecture_labels, _scenario_labels = _display_labels(repo_root)
    results = _architecture_results(repo_root)
    datasets = _datasets_in_order(results)

    clean_leaders = {
        dataset: _unique_min(_dataset_rows(results, dataset), lambda row: row.mse_clean,
                             context=f"{dataset} clean MSE")
        for dataset in datasets
    }
    fault_time_leaders = {
        dataset: _unique_min(_dataset_rows(results, dataset), lambda row: row.mse_w,
                             context=f"{dataset} worst-scenario fault-time MSE")
        for dataset in datasets
    }
    sharpest = _unique_max(list(clean_leaders.values()), lambda row: row.d_w,
                           context="clean-MSE leaders by worst-scenario degradation")
    rank = 1 + sum(
        1 for row in _dataset_rows(results, sharpest.dataset) if row.d_w > sharpest.d_w
    )
    if rank not in ORDINAL_PREFIXES:
        raise ValueError(
            f"{sharpest.dataset}: the clean-MSE leader ranks {rank} by worst-scenario "
            "degradation, which does not support 'degrade sharply'."
        )
    disagreements = sum(
        1 for dataset in datasets
        if clean_leaders[dataset].architecture != fault_time_leaders[dataset].architecture
    )
    if disagreements == 0:
        raise ValueError("Clean-MSE and worst-scenario fault-time MSE leaders agree on every dataset.")
    finding_1 = (
        "1. **Low clean error does not imply robustness.** Forecasting architectures favored by "
        "clean MSE can degrade sharply under faults. On "
        f"{dataset_labels[sharpest.dataset]}, {architecture_labels[sharpest.architecture]} has "
        f"the lowest clean MSE ({_fmt(sharpest.mse_clean)}) and the {ORDINAL_PREFIXES[rank]}largest "
        f"worst-scenario degradation ({_fmt(sharpest.d_w)}). The architectures with the lowest "
        "clean MSE and the lowest worst-scenario fault-time MSE differ on "
        f"{NUMBER_WORDS[disagreements]} of the {NUMBER_WORDS[len(datasets)]} datasets."
    )

    single_target = [dataset for dataset in datasets if dataset in _single_target_datasets()]
    all_channel = [dataset for dataset in datasets if dataset not in single_target]
    periods = _selected_naive_periods(repo_root)
    clean_pairs = []
    for dataset in single_target:
        if periods.get(dataset) != 1:
            raise ValueError(
                f"{dataset}: the selected {NAIVE_REFERENCE} period is {periods.get(dataset)!r}, "
                "so it is not the last-value naive forecaster."
            )
        rows = _dataset_rows(results, dataset)
        by_architecture = {row.architecture: row for row in rows}
        foundation = by_architecture[FOUNDATION_MODEL]
        degradation_rank = 1 + sum(1 for row in rows if row.d_w < foundation.d_w)
        if degradation_rank != 2:
            raise ValueError(
                f"{dataset}: {FOUNDATION_MODEL} ranks {degradation_rank} by worst-scenario "
                "degradation, not second lowest."
            )
        clean_pairs.append(
            f"{_fmt(foundation.mse_clean)} against "
            f"{_fmt(by_architecture[NAIVE_REFERENCE].mse_clean)} on {dataset_labels[dataset]}"
        )
    degradation_values = []
    architecture_counts = set()
    for dataset in all_channel:
        rows = _dataset_rows(results, dataset)
        worst = _unique_max(rows, lambda row: row.d_w,
                            context=f"{dataset} worst-scenario degradation")
        if worst.architecture != FOUNDATION_MODEL:
            raise ValueError(
                f"{dataset}: the largest worst-scenario degradation belongs to "
                f"{worst.architecture}, not {FOUNDATION_MODEL}."
            )
        degradation_values.append(_fmt(worst.d_w))
        architecture_counts.add(len(rows))
    if len(architecture_counts) != 1:
        raise ValueError(
            f"The all-channel datasets compare different architecture counts: {architecture_counts}."
        )
    (architecture_count,) = architecture_counts
    foundation_label = architecture_labels[FOUNDATION_MODEL]
    finding_2 = (
        f"2. **{foundation_label} has either high clean error or low robustness "
        f"on each dataset.** {foundation_label} is a pretrained time series foundation model "
        "that the benchmark runs zero-shot, without fine-tuning on the "
        f"{NUMBER_WORDS[len(datasets)]} datasets. On the "
        f"{NUMBER_WORDS[len(single_target)]} single-target datasets, its worst-scenario "
        "degradation is the second lowest, but its clean MSE matches or trails that of the "
        "last-value naive forecaster, which repeats the last observed value "
        f"({' and '.join(clean_pairs)}). On "
        f"{' and '.join(dataset_labels[dataset] for dataset in all_channel)}, "
        f"where all channels are forecast targets, {foundation_label} has the largest "
        f"worst-scenario degradation of the {NUMBER_WORDS[architecture_count]} architectures "
        f"({' and '.join(degradation_values)})."
    )

    deltas = _mean_method_deltas(repo_root)
    scenario_classes = _scenario_classes(repo_root)
    leaders: dict[str, tuple[str, float, int]] = {}
    for dataset in datasets:
        candidates = sorted(
            (method, value, count) for (row_dataset, method), (value, count) in deltas.items()
            if row_dataset == dataset
        )
        leaders[dataset] = _unique_min(
            candidates,
            lambda item: item[1],
            context=f"{dataset} mean paired worst-scenario degradation delta",
        )
    method_labels = {
        "adversarial_training": "PGD adversarial training",
        "randomized_training": "randomized training",
        "fault_augmentation": "fault augmentation",
    }
    value_cells = []
    for method in ("adversarial_training", "randomized_training"):
        led = [dataset for dataset in single_target if leaders[dataset][0] == method]
        if len(led) != 1:
            raise ValueError(f"{method} does not lead on exactly one single-target dataset.")
        if _dominant_worst_scenario_class(results, led[0], scenario_classes) != "Value":
            raise ValueError(f"{led[0]}: value faults do not dominate the worst scenarios.")
        value_cells.append((method, led[0]))
    availability_dataset = "traffic"
    if leaders[availability_dataset][0] != "fault_augmentation":
        raise ValueError(
            f"{availability_dataset}: {leaders[availability_dataset][0]} leads, "
            "not fault_augmentation."
        )
    if _dominant_worst_scenario_class(results, availability_dataset, scenario_classes) != "Availability":
        raise ValueError(f"{availability_dataset}: availability faults do not dominate.")
    named = [*value_cells, ("fault_augmentation", availability_dataset)]
    for _method, dataset in named:
        if leaders[dataset][1] >= 0.0:
            raise ValueError(
                f"{dataset}: the leading mean worst-scenario degradation delta "
                f"{leaders[dataset][1]} is not a reduction."
            )
    pair_counts = {leaders[dataset][2] for _method, dataset in named}
    if len(pair_counts) != 1:
        raise ValueError(f"The named method means average different pair counts: {pair_counts}.")
    (pair_count,) = pair_counts

    def _cell(method: str, dataset: str) -> str:
        return (
            f"{method_labels[method]} leads on {dataset_labels[dataset]} "
            f"({_fmt(leaders[dataset][1])})"
        )

    first, second = (_cell(method, dataset) for method, dataset in value_cells)
    third = _cell("fault_augmentation", availability_dataset)
    finding_3 = (
        "3. **Robustness-improvement methods help selectively.** Paired deltas against the "
        "baseline forecasting model of the same architecture show selective degradation "
        f"reductions. {first[0].upper()}{first[1:]} and "
        f"{second.replace(' leads on ', ' on ', 1)}, where value faults dominate observed "
        f"degradation. {third[0].upper()}{third[1:]}, where "
        "availability faults dominate. Values are mean paired changes in worst-scenario "
        f"degradation over the {NUMBER_WORDS[pair_count]} method-comparison architectures."
    )
    return "\n".join((finding_1, finding_2, finding_3))


def render_method_results(repo_root: Path = REPO_ROOT) -> str:
    """Render the method-results block from the bundled method-delta table."""
    scope = _read_yaml(repo_root, BENCHMARK_SCOPE_PATH)
    dataset_labels = dict(scope["datasets"]["display_labels"])
    method_labels = dict(scope["methods"]["display_labels"])
    dataset_order = list(scope["datasets"]["display_order"])
    method_order = [method for method in scope["methods"]["display_order"] if method != "baseline"]
    rows = {
        (row["dataset"], row["robustness_method"]): row
        for row in _read_csv(repo_root, METHOD_DELTA_PATH)
    }
    expected = {(dataset, method) for dataset in dataset_order for method in method_order}
    if set(rows) != expected:
        raise ValueError(
            f"{METHOD_DELTA_PATH} must hold one row per dataset and method, "
            f"missing {sorted(expected - set(rows))}, unexpected {sorted(set(rows) - expected)}."
        )
    lines = [
        "<details>",
        f"<summary>All {len(rows)} method results</summary>",
        "",
        "| Dataset | Method | Pairs | Clean MSE delta | Worst-scenario degradation delta | "
        "Worst-scenario fault-time MSE delta |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for dataset in dataset_order:
        for method in method_order:
            row = rows[(dataset, method)]
            lines.append(
                f"| {dataset_labels[dataset]} | {method_labels[method]} | {int(row['count'])} | "
                f"{_fmt_delta(float(row['delta_err_clean_mean']))} | "
                f"{_fmt_delta(float(row['delta_D_w_mean']))} | "
                f"{_fmt_delta(float(row['delta_err_pert_ws_mean']))} |"
            )
    lines += ["", "</details>"]
    return "\n".join(lines)


RENDERERS = {
    "key-findings": render_key_findings,
    "baseline-results": render_baseline_results,
    "method-results": render_method_results,
}


def block_pattern(name: str) -> re.Pattern[str]:
    return re.compile(
        rf"(<!-- {re.escape(name)}:start -->\n)(.*?)(\n<!-- {re.escape(name)}:end -->)",
        re.S,
    )


def readme_block(text: str, name: str) -> str:
    matches = block_pattern(name).findall(text)
    if len(matches) != 1:
        raise ValueError(f"README.md must contain exactly one {name} block, found {len(matches)}.")
    return matches[0][1]


def render_readme(text: str, repo_root: Path = REPO_ROOT) -> str:
    for name in BLOCK_NAMES:
        readme_block(text, name)
        rendered = RENDERERS[name](repo_root)
        text = block_pattern(name).sub(lambda match: match.group(1) + rendered + match.group(3), text)
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit with status 1 when README.md differs from the rendered blocks.",
    )
    args = parser.parse_args(argv)
    readme_path = REPO_ROOT / README_PATH
    current = readme_path.read_text(encoding="utf-8")
    rendered = render_readme(current)
    if args.check:
        if rendered != current:
            print("README.md result blocks are stale. Run scripts/render_readme_results.py.")
            return 1
        print("README.md result blocks are current.")
        return 0
    readme_path.write_text(rendered, encoding="utf-8")
    print("Rendered README.md result blocks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
