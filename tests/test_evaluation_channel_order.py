"""Training and evaluation reject runs whose channel order differs from the data."""

from types import SimpleNamespace

import pandas as pd
import pytest

import testing.evaluation as evaluation
from data.data_module import TSDataModule
from data.datasets import ResolvedDatasetSpec, channel_order_tags, spec_to_tags
from models import SeasonalNaive
from pipelines.runner import PipelineRunner
from testing.evaluation import _prepare_model_for_evaluation, load_model_with_loader


def _make_spec(path: str, *, inputs: tuple[str, ...], targets: tuple[str, ...]) -> ResolvedDatasetSpec:
    return ResolvedDatasetSpec(
        key="toy",
        path=path,
        input_channels=inputs,
        target_channels=targets,
        target_alias=None,
        split_mode="temporal",
        description=None,
        batch_column=None,
        continuous_channels=inputs,
        discrete_channels=(),
    )


def _make_datamodule(spec: ResolvedDatasetSpec) -> TSDataModule:
    dm = TSDataModule(
        dataset_spec=spec,
        input_len=4,
        target_len=2,
        n_train_samples=8,
        n_val_samples=4,
        n_test_samples=4,
        perturbation_channel_fraction_max=0.5,
        perturbation_scenarios=("drift",),
        perturbation_severity_distribution="uniform",
        train_split=0.6,
        val_split=0.2,
        purged_fraction=0.0,
        shuffle_batches_before_split=True,
        batch_size=4,
        num_workers=0,
        seed=42,
        strict_iid=False,
    )
    dm.setup()
    return dm


@pytest.fixture
def toy_csv(tmp_path) -> str:
    path = tmp_path / "toy.csv"
    pd.DataFrame(
        {
            "a": [float(idx) for idx in range(200)],
            "b": [float(10 * idx) for idx in range(200)],
            "c": [float(100 * idx) for idx in range(200)],
        }
    ).to_csv(path, index=False)
    return str(path)


def _trained_model(dm: TSDataModule) -> SeasonalNaive:
    return SeasonalNaive(
        d_input_features=dm.n_inputs,
        d_target_features=dm.n_outputs,
        d_seq_in=4,
        d_seq_out=2,
        target_indices=dm.target_column_indices,
        season_length=1,
    )


def _eval_args() -> SimpleNamespace:
    return SimpleNamespace(input_len=4, target_len=2, test_metric="MSE")


def test_prepare_model_for_evaluation_keeps_target_mapping_of_unchanged_order(toy_csv):
    dm = _make_datamodule(_make_spec(toy_csv, inputs=("a", "b"), targets=("a",)))
    model = _trained_model(dm)

    _prepare_model_for_evaluation(model, _eval_args(), dm)

    assert model.target_indices == (0,)


def test_prepare_model_for_evaluation_rejects_reordered_target_mapping(toy_csv):
    train_dm = _make_datamodule(_make_spec(toy_csv, inputs=("a", "b"), targets=("a",)))
    model = _trained_model(train_dm)
    eval_dm = _make_datamodule(_make_spec(toy_csv, inputs=("b", "a"), targets=("a",)))

    with pytest.raises(ValueError, match=r"target_indices \(0,\) do not match datamodule target_indices \(1,\)"):
        _prepare_model_for_evaluation(model, _eval_args(), eval_dm)


def _make_run(spec: ResolvedDatasetSpec) -> SimpleNamespace:
    tags = {
        "model_architecture": "SeasonalNaive",
        "loader_kind": "SeasonalNaive",
        "pipeline_id": "baseline",
        "pipeline_method": "baseline",
        "pipeline_kind": "train",
        "robustness_method": "baseline",
        **spec_to_tags(spec, n_inputs=len(spec.input_channels), n_outputs=len(spec.target_channels)),
    }
    return SimpleNamespace(info=SimpleNamespace(run_id="run_channels"), data=SimpleNamespace(tags=tags))


@pytest.fixture
def stub_checkpoint_loader(monkeypatch, tmp_path):
    loaded_model = SimpleNamespace()
    monkeypatch.setattr(
        evaluation,
        "_load_model_from_run",
        lambda client, run, target_class: (loaded_model, str(tmp_path)),
    )
    return loaded_model


def test_load_model_with_loader_accepts_unchanged_channel_order(toy_csv, stub_checkpoint_loader):
    spec = _make_spec(toy_csv, inputs=("a", "b", "c"), targets=("a", "b"))

    model, _ = load_model_with_loader(object(), _make_run(spec), _eval_args(), SimpleNamespace(dataset_spec=spec))

    assert model is stub_checkpoint_loader


@pytest.mark.parametrize(
    ("eval_inputs", "eval_targets", "message"),
    [
        (("a", "c", "b"), ("a", "b"), "input_channels='a;b;c'"),
        (("a", "b", "c"), ("b", "a"), "target_channels='a;b'"),
    ],
)
def test_load_model_with_loader_rejects_reordered_channels(
    toy_csv,
    stub_checkpoint_loader,
    eval_inputs,
    eval_targets,
    message,
):
    run = _make_run(_make_spec(toy_csv, inputs=("a", "b", "c"), targets=("a", "b")))
    eval_spec = _make_spec(toy_csv, inputs=eval_inputs, targets=eval_targets)

    with pytest.raises(ValueError, match=f"was trained with {message}"):
        load_model_with_loader(object(), run, _eval_args(), SimpleNamespace(dataset_spec=eval_spec))


def test_load_model_with_loader_requires_channel_order_tags(toy_csv, stub_checkpoint_loader):
    spec = _make_spec(toy_csv, inputs=("a", "b"), targets=("a",))
    run = _make_run(spec)
    del run.data.tags["input_channels"]

    with pytest.raises(ValueError, match="missing required 'input_channels' tag"):
        load_model_with_loader(object(), run, _eval_args(), SimpleNamespace(dataset_spec=spec))


class _RunSearchClient:
    def __init__(self, runs):
        self._runs = list(runs)

    def search_runs(self, _experiment_ids, **_kwargs):
        return list(self._runs)


def _finished_run(spec: ResolvedDatasetSpec, *, status: str = "FINISHED") -> SimpleNamespace:
    run = _make_run(spec)
    run.info.status = status
    return run


def _require_training_channel_order(runs, spec: ResolvedDatasetSpec) -> None:
    PipelineRunner(SimpleNamespace(), SimpleNamespace())._require_channel_order_of_finished_runs(
        client=_RunSearchClient(runs),
        experiment_id="experiment",
        dataset_spec=spec,
        architecture="SeasonalNaive",
    )


@pytest.mark.parametrize(
    ("current_inputs", "current_targets", "message"),
    [
        (("c", "b", "a"), ("a", "b"), "input_channels='a;b;c'"),
        (("a", "b", "c"), ("b", "a"), "target_channels='a;b'"),
    ],
)
def test_training_rejects_finished_run_with_reordered_channels(
    toy_csv,
    current_inputs,
    current_targets,
    message,
):
    trained_spec = _make_spec(toy_csv, inputs=("a", "b", "c"), targets=("a", "b"))
    current_spec = _make_spec(toy_csv, inputs=current_inputs, targets=current_targets)

    with pytest.raises(ValueError, match=f"was trained with {message}"):
        _require_training_channel_order([_finished_run(trained_spec)], current_spec)


def test_training_accepts_unchanged_order_and_skips_unfinished_runs(toy_csv):
    spec = _make_spec(toy_csv, inputs=("a", "b", "c"), targets=("a",))
    reordered_spec = _make_spec(toy_csv, inputs=("c", "b", "a"), targets=("a",))

    _require_training_channel_order(
        [_finished_run(spec), _finished_run(reordered_spec, status="FAILED")],
        spec,
    )


def test_channel_order_tags_require_explicit_input_channels():
    spec = _make_spec("toy.csv", inputs=("a", "b"), targets=("b",))
    object.__setattr__(spec, "input_channels", None)

    with pytest.raises(ValueError, match="must list its input_channels"):
        channel_order_tags(spec)


def test_channel_order_tags_record_targets_that_follow_the_inputs_as_empty():
    spec = _make_spec("toy.csv", inputs=("b", "a"), targets=("a",))
    object.__setattr__(spec, "target_channels", None)

    assert channel_order_tags(spec) == {"input_channels": "b;a", "target_channels": ""}
