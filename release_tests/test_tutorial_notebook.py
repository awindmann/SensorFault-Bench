"""Checks the CPU tutorial notebook in notebooks/tutorial/quickstart.ipynb.

The notebook ships with executed outputs. The checks cover the order and
cleanliness of those outputs, the commands and scope it runs, and the operator
interface its scenario gallery uses. No data file and no run store is needed.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import torch

from config_loader import build_parser, load_defaults
from data.data_module import instantiate_perturbations
from data.perturbations import PERTURBATION_REGISTRY
from run_analysis import prepare_analysis_entrypoint_args
from run_testing import prepare_testing_entrypoint_args


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "tutorial" / "quickstart.ipynb"
LOCAL_PATH_PATTERN = re.compile(r"/Users/|/home/|[A-Za-z]:\\|site-packages|/private/|/tmp/")


def _notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def _code_cells() -> list[dict]:
    return [cell for cell in _notebook()["cells"] if cell["cell_type"] == "code"]


def _source(cell: dict) -> str:
    return "".join(cell["source"])


def _assigned_list(name: str) -> list[str]:
    for cell in _code_cells():
        for node in ast.parse(_source(cell)).body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == name for target in node.targets
            ):
                expression = compile(ast.Expression(node.value), name, "eval")
                return eval(expression, {"LOGDIR": "runs/tutorial"})
    raise AssertionError(f"{name} is not assigned in the tutorial notebook.")


def _output_text(output: dict) -> str:
    data = output.get("data", {})
    return "".join(output.get("text", "")) + "".join(data.get("text/plain", "")) + "".join(
        data.get("text/html", "")
    )


def test_tutorial_outputs_are_executed_in_order_without_errors_or_local_paths():
    cells = _code_cells()

    assert [cell["execution_count"] for cell in cells] == list(range(1, len(cells) + 1))
    for cell in cells:
        assert cell["outputs"], cell["id"]
        for output in cell["outputs"]:
            assert output["output_type"] != "error", cell["id"]
            assert output.get("name") != "stderr", cell["id"]
            text = _output_text(output)
            assert not LOCAL_PATH_PATTERN.search(text), cell["id"]
            assert "Traceback" not in text and "Warning" not in text, cell["id"]


def test_tutorial_commands_use_shipped_scripts_and_the_intended_scope():
    sources = "\n".join(_source(cell) for cell in _code_cells())
    scripts = re.findall(r'run\(\s*"([^"]+\.py)"', sources)
    scope = _assigned_list("SCOPE")
    test_windows = _assigned_list("TEST_WINDOWS")
    defaults = load_defaults()

    assert {"run_training.py", "run_testing.py", "run_analysis.py"} <= set(scripts)
    for script in scripts:
        assert (REPO_ROOT / script).is_file(), script

    training = build_parser(defaults).parse_args([*scope, "--max-epochs", "2"])
    testing, _, _ = prepare_testing_entrypoint_args([*scope, *test_windows, "--full-coverage"])
    analysis = prepare_analysis_entrypoint_args([*scope, *test_windows])
    for args in (training, testing, analysis):
        assert args.data_files == ["ETTh1"]
        assert args.model == ["DLinear"]
        assert args.method == ["baseline", "randomized_training"]
        assert args.accelerator == "cpu"
        assert int(args.num_workers) == 0
    assert set(testing.method) <= set(defaults["BENCHMARK_METHODS"])
    assert testing.full_coverage is True
    assert int(testing.n_test_samples) == int(analysis.n_test_samples) == 2000


def test_tutorial_fault_gallery_operators_keep_the_forecast_target_unchanged():
    sources = "\n".join(_source(cell) for cell in _code_cells())
    defaults = load_defaults()
    scenarios = list(defaults["PERTURBATION_SCENARIOS"])
    x = torch.randn(96, 7, generator=torch.Generator().manual_seed(0))
    y = torch.randn(96, 7, generator=torch.Generator().manual_seed(1))

    assert 'SCENARIOS = list(DEFAULTS["PERTURBATION_SCENARIOS"])' in sources
    operators = instantiate_perturbations(
        [PERTURBATION_REGISTRY[name] for name in scenarios],
        registry=PERTURBATION_REGISTRY,
        channel_fraction_max=defaults["PERTURBATION_CHANNEL_FRACTION_MAX"],
    )
    for index, operator in enumerate(operators):
        faulty, target, affected = operator(
            x, y, 0.6, torch.Generator().manual_seed(index), list(range(7)), []
        )
        assert torch.equal(target, y), scenarios[index]
        assert faulty.shape == x.shape, scenarios[index]
        assert affected and set(affected) <= set(range(7)), scenarios[index]
