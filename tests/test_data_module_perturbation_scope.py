import pandas as pd
import pytest

import data.data_module as data_module_module
from data.data_module import TSDataModule
from data.datasets.base import ResolvedDatasetSpec


DEFAULT_TEST_SCENARIOS = (
    "drift",
    "attenuation",
    "noise",
    "stuck_sensor",
    "missing_data",
    "spike",
    "time_stretch",
    "time_compress",
    "wrong_state",
    "chattering",
)


def _make_dataset_spec(tmp_path) -> ResolvedDatasetSpec:
    df = pd.DataFrame(
        {
            "x": list(range(120)),
            "y": list(range(120, 240)),
            "d": [idx % 2 for idx in range(120)],
        }
    )
    path = tmp_path / "toy.csv"
    df.to_csv(path, index=False)
    return ResolvedDatasetSpec(
        key="toy",
        path=str(path),
        input_channels=("x", "y", "d"),
        target_channels=("x",),
        target_alias=None,
        split_mode="temporal",
        description=None,
        batch_column=None,
        continuous_channels=("x", "y"),
        discrete_channels=("d",),
    )


def _make_continuous_only_dataset_spec(tmp_path) -> ResolvedDatasetSpec:
    df = pd.DataFrame(
        {
            "x": list(range(120)),
            "y": list(range(120, 240)),
            "z": list(range(240, 360)),
        }
    )
    path = tmp_path / "toy_continuous.csv"
    df.to_csv(path, index=False)
    return ResolvedDatasetSpec(
        key="toy_continuous",
        path=str(path),
        input_channels=("x", "y", "z"),
        target_channels=("x",),
        target_alias=None,
        split_mode="temporal",
        description=None,
        batch_column=None,
        continuous_channels=("x", "y", "z"),
        discrete_channels=(),
    )


def _make_datamodule(
    tmp_path,
    *,
    p_perturbations="uniform",
    severity_laws=None,
    dataset_spec: ResolvedDatasetSpec | None = None,
    perturbation_scenarios=DEFAULT_TEST_SCENARIOS,
    seed: int = 42,
    val_seed: int | None = None,
    n_test_samples: int = 2,
) -> TSDataModule:
    dataset_spec = _make_dataset_spec(tmp_path) if dataset_spec is None else dataset_spec
    return TSDataModule(
        dataset_spec=dataset_spec,
        input_len=4,
        target_len=2,
        n_train_samples=8,
        n_val_samples=4,
        n_test_samples=n_test_samples,
        perturbation_channel_fraction_max=0.5,
        perturbation_scenarios=perturbation_scenarios,
        train_split=0.7,
        val_split=0.2,
        purged_fraction=0.0,
        shuffle_batches_before_split=True,
        batch_size=4,
        num_workers=0,
        seed=seed,
        val_seed=val_seed,
        strict_iid=False,
        p_perturbations=p_perturbations,
        severity_laws=severity_laws,
    )


def test_datamodule_routes_constructor_by_channel_scope(monkeypatch, tmp_path):
    class FakeContinuous:
        name = "fake_continuous"
        idx = 1000
        channel_scope = "continuous"

        def __init__(self, channel_frac: float = 0.1):
            self.channel_frac = float(channel_frac)

    class FakeAll:
        name = "fake_all"
        idx = 1001
        channel_scope = "all"

        def __init__(self):
            self.created = True

    monkeypatch.setattr(
        data_module_module,
        "PERTURBATION_REGISTRY",
        {
            "fake_continuous": FakeContinuous,
            "fake_all": FakeAll,
        },
    )

    dm = _make_datamodule(
        tmp_path,
        perturbation_scenarios=("fake_continuous", "fake_all"),
    )
    perturbations = dm.pert_sampler.perturbations
    assert len(perturbations) == 2
    assert isinstance(perturbations[0], FakeContinuous)
    assert perturbations[0].channel_frac == pytest.approx(0.5)
    assert isinstance(perturbations[1], FakeAll)


def test_datamodule_rejects_invalid_perturbation_channel_scope(monkeypatch, tmp_path):
    class FakeInvalid:
        name = "fake_invalid"
        idx = 1002
        channel_scope = "invalid_scope"

        def __init__(self):
            self.created = True

    monkeypatch.setattr(
        data_module_module,
        "PERTURBATION_REGISTRY",
        {
            "fake_invalid": FakeInvalid,
        },
    )

    with pytest.raises(ValueError, match="invalid channel_scope"):
        _make_datamodule(tmp_path, perturbation_scenarios=("fake_invalid",))


def test_datamodule_uses_explicit_discrete_channel_typing(tmp_path):
    dm = _make_datamodule(tmp_path)
    dm.setup()
    assert "d" in dm.discrete_channels
    assert "d" not in dm.continuous_channels


def test_datamodule_defaults_untyped_inputs_to_continuous(tmp_path):
    df = pd.DataFrame(
        {
            "x": list(range(120)),
            "y": list(range(120, 240)),
            "d": [idx % 2 for idx in range(120)],
        }
    )
    path = tmp_path / "toy_untyped.csv"
    df.to_csv(path, index=False)
    dataset_spec = ResolvedDatasetSpec(
        key="toy_untyped",
        path=str(path),
        input_channels=("x", "y", "d"),
        target_channels=("x",),
        target_alias=None,
        split_mode="temporal",
        description=None,
        batch_column=None,
    )
    dm = _make_datamodule(
        tmp_path,
        dataset_spec=dataset_spec,
        perturbation_scenarios=("drift",),
    )
    dm.setup()
    assert dm.continuous_channels == ("x", "y", "d")
    assert dm.discrete_channels == ()


def test_datamodule_rejects_invalid_perturbation_probability_string(tmp_path):
    with pytest.raises(
        ValueError,
        match="p_perturbations must be 'uniform' or a numeric weight vector",
    ):
        _make_datamodule(tmp_path, p_perturbations="weighted")


def test_datamodule_rejects_probability_length_mismatch(tmp_path):
    with pytest.raises(ValueError, match="length mismatch"):
        _make_datamodule(tmp_path, p_perturbations=[1.0, 1.0])


def test_datamodule_rejects_negative_probability_weights(tmp_path):
    with pytest.raises(ValueError, match="must be non-negative"):
        _make_datamodule(
            tmp_path,
            p_perturbations=[-1.0] + [1.0] * (len(DEFAULT_TEST_SCENARIOS) - 1),
        )


def test_datamodule_selects_only_requested_scenarios(tmp_path):
    dm = _make_datamodule(
        tmp_path,
        perturbation_scenarios=("drift", "missing_data"),
    )
    assert dm.perturbation_names == ["drift", "missing_data"]
    assert dm.perturbation_name_by_idx == {
        dm.pert_sampler.perturbations[0].idx: "drift",
        dm.pert_sampler.perturbations[1].idx: "missing_data",
    }


def test_datamodule_time_compress_uses_operator_default_endpoint(tmp_path):
    dm = _make_datamodule(
        tmp_path,
        perturbation_scenarios=("time_compress",),
    )
    perturbation = dm.pert_sampler.perturbations[0]
    assert perturbation.name == "time_compress"
    assert perturbation.max_rate == pytest.approx(0.1)


def test_datamodule_rejects_probability_vector_mismatched_to_selected_scenarios(tmp_path):
    with pytest.raises(ValueError, match="length mismatch"):
        _make_datamodule(
            tmp_path,
            perturbation_scenarios=("drift", "missing_data"),
            p_perturbations=[1.0, 1.0, 1.0],
        )


def test_datamodule_rejects_unknown_severity_law_name_for_selected_scenarios(tmp_path):
    with pytest.raises(ValueError, match="unknown perturbation names"):
        _make_datamodule(
            tmp_path,
            perturbation_scenarios=("drift", "missing_data"),
            severity_laws={"noise": lambda rng: 0.5},
        )


def test_datamodule_rejects_unknown_perturbation_scenario(tmp_path):
    with pytest.raises(ValueError, match="Unknown perturbation scenario"):
        _make_datamodule(
            tmp_path,
            perturbation_scenarios=("drift", "not_a_scenario"),
        )


def test_datamodule_rejects_duplicate_perturbation_scenarios(tmp_path):
    with pytest.raises(ValueError, match="duplicate scenario name"):
        _make_datamodule(
            tmp_path,
            perturbation_scenarios=("drift", "drift"),
        )


def test_datamodule_rejects_empty_perturbation_scenarios(tmp_path):
    with pytest.raises(ValueError, match="non-empty list of scenario names"):
        _make_datamodule(
            tmp_path,
            perturbation_scenarios=[],
        )


def test_datamodule_rejects_discrete_scenario_without_discrete_channels(tmp_path):
    with pytest.raises(ValueError, match="does not support configured perturbations"):
        dm = _make_datamodule(
            tmp_path,
            dataset_spec=_make_continuous_only_dataset_spec(tmp_path),
            perturbation_scenarios=("wrong_state",),
        )
        dm.setup()


def test_datamodule_exposes_deterministic_perturbation_scenarios_signature(tmp_path):
    scenarios = ("drift", "noise", "missing_data")
    dm_one = _make_datamodule(tmp_path, perturbation_scenarios=scenarios)
    dm_two = _make_datamodule(tmp_path, perturbation_scenarios=scenarios)
    assert dm_one.perturbation_scenarios_signature == dm_two.perturbation_scenarios_signature


def test_datamodule_val_seed_decouples_val_sampling_from_test_sampling(tmp_path):
    dm_one = _make_datamodule(tmp_path, seed=42, val_seed=17, n_test_samples=8)
    dm_two = _make_datamodule(tmp_path, seed=99, val_seed=17, n_test_samples=8)

    dm_one.setup()
    dm_two.setup()

    assert dm_one.ds_val.sample_idxs.tolist() == dm_two.ds_val.sample_idxs.tolist()
    assert dm_one.ds_test.base_ds.sample_idxs.tolist() != dm_two.ds_test.base_ds.sample_idxs.tolist()
    assert dm_one.ds_test.seed != dm_two.ds_test.seed


def test_datamodule_default_val_seed_preserves_historical_seed_plus_one_contract(tmp_path):
    dm_default = _make_datamodule(tmp_path, seed=42)
    dm_explicit = _make_datamodule(tmp_path, seed=999, val_seed=42)

    dm_default.setup()
    dm_explicit.setup()

    assert dm_default.ds_val.sample_idxs.tolist() == dm_explicit.ds_val.sample_idxs.tolist()
