import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

import models.components.attention as attention_impl
from models.components.attention import ProbAttention
from models.base_module import BaseLitModule
from pipelines.signatures import compute_data_config_signature
from utils.rng import derive_component_seeds, derive_seed, derive_tuning_seed


def _make_args(*, seed=42, data_split_seed=None, n_train_samples=10):
    return SimpleNamespace(
        input_len=4,
        target_len=2,
        train_split=0.7,
        val_split=0.2,
        purged_fraction=0.01,
        shuffle_batches_before_split=True,
        strict_iid=False,
        n_train_samples=n_train_samples,
        n_val_samples=5 if n_train_samples is not None else None,
        n_test_samples=5 if n_train_samples is not None else None,
        seed=seed,
        data_split_seed=data_split_seed,
    )


def _make_dataset_spec(key="demo", *, split_mode="temporal"):
    return SimpleNamespace(
        key=key,
        input_channels=None,
        target_channels=None,
        split_mode=split_mode,
    )


def _reference_temporal_signature(args, *, dataset_key="demo"):
    normalized = {
        "dataset": dataset_key,
        "target_channels": "all",
        "input_channels": "all",
        "input_len": args.input_len,
        "target_len": args.target_len,
        "train_split": args.train_split,
        "val_split": args.val_split,
        "purged_fraction": args.purged_fraction,
        "shuffle_batches_before_split": True,
        "strict_iid": args.strict_iid,
    }
    sampling_enabled = any(
        value is not None
        for value in (args.n_train_samples, args.n_val_samples, args.n_test_samples)
    )
    if sampling_enabled:
        sampling_seed = args.data_split_seed if args.data_split_seed is not None else args.seed
        normalized["sampling_seed"] = sampling_seed
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_data_seed_stable_across_model_variants():
    seeds_a = derive_component_seeds(
        base_seed=123,
        dataset_key="demo",
        data_config_signature="sig",
        architecture="DLinear",
        pipeline_id="baseline",
        pipeline_method="baseline",
        pipeline_kind="train",
    )
    seeds_b = derive_component_seeds(
        base_seed=123,
        dataset_key="demo",
        data_config_signature="sig",
        architecture="GRU",
        pipeline_id="alt",
        pipeline_method="alt",
        pipeline_kind="train",
    )
    assert seeds_a["data_seed"] == seeds_b["data_seed"]
    assert seeds_a["model_seed"] != seeds_b["model_seed"]


def test_model_seed_changes_with_stage_label():
    seeds_pre = derive_component_seeds(
        base_seed=123,
        dataset_key="demo",
        data_config_signature="sig",
        architecture="DLinear",
        pipeline_id="baseline",
        pipeline_method="baseline",
        pipeline_kind="train",
        model_stage="phase_a",
    )
    seeds_ft = derive_component_seeds(
        base_seed=123,
        dataset_key="demo",
        data_config_signature="sig",
        architecture="DLinear",
        pipeline_id="baseline",
        pipeline_method="baseline",
        pipeline_kind="train",
        model_stage="phase_b",
    )
    assert seeds_pre["model_seed"] != seeds_ft["model_seed"]


def test_data_split_seed_overrides_master_seed():
    seeds_master = derive_component_seeds(
        base_seed=42,
        data_base_seed=None,
        dataset_key="demo",
        data_config_signature="sig",
        architecture="DLinear",
        pipeline_id="baseline",
        pipeline_method="baseline",
        pipeline_kind="train",
    )
    seeds_split = derive_component_seeds(
        base_seed=42,
        data_base_seed=999,
        dataset_key="demo",
        data_config_signature="sig",
        architecture="DLinear",
        pipeline_id="baseline",
        pipeline_method="baseline",
        pipeline_kind="train",
    )
    assert seeds_master["data_seed"] != seeds_split["data_seed"]


def test_seed_derivation_rejects_empty_key():
    with pytest.raises(ValueError, match="key must be set"):
        derive_seed(42, "")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("data_config_signature", ""),
        ("architecture", None),
        ("pipeline_id", ""),
        ("pipeline_method", "  "),
        ("pipeline_kind", None),
    ],
)
def test_component_seed_derivation_requires_identity_components(field, value):
    kwargs = {
        "base_seed": 123,
        "dataset_key": "demo",
        "data_config_signature": "sig",
        "architecture": "DLinear",
        "pipeline_id": "baseline",
        "pipeline_method": "baseline",
        "pipeline_kind": "train",
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=f"{field} must be set"):
        derive_component_seeds(**kwargs)


def test_data_config_signature_uses_data_split_seed_when_sampling():
    dataset_spec = _make_dataset_spec()
    args_a = _make_args(seed=42, data_split_seed=123, n_train_samples=10)
    args_b = _make_args(seed=42, data_split_seed=456, n_train_samples=10)
    sig_a = compute_data_config_signature(dataset_spec=dataset_spec, args=args_a)
    sig_b = compute_data_config_signature(dataset_spec=dataset_spec, args=args_b)
    assert sig_a != sig_b


def test_data_config_signature_preserves_cli_seed_token_shape_when_sampling():
    dataset_spec = _make_dataset_spec()
    args_int = _make_args(seed=42, data_split_seed=123, n_train_samples=10)
    args_str = _make_args(seed=42, data_split_seed="123", n_train_samples=10)
    args_null = _make_args(seed=42, data_split_seed="null", n_train_samples=10)

    assert compute_data_config_signature(dataset_spec=dataset_spec, args=args_int) != (
        compute_data_config_signature(dataset_spec=dataset_spec, args=args_str)
    )
    assert compute_data_config_signature(dataset_spec=dataset_spec, args=args_str) == (
        _reference_temporal_signature(args_str)
    )
    assert compute_data_config_signature(dataset_spec=dataset_spec, args=args_null) == (
        _reference_temporal_signature(args_null)
    )


def test_data_config_signature_ignores_shuffle_for_temporal_mode():
    dataset_spec = _make_dataset_spec(split_mode="temporal")
    args_a = _make_args(seed=42, n_train_samples=None)
    args_b = _make_args(seed=42, n_train_samples=None)
    args_b.shuffle_batches_before_split = False
    sig_a = compute_data_config_signature(dataset_spec=dataset_spec, args=args_a)
    sig_b = compute_data_config_signature(dataset_spec=dataset_spec, args=args_b)
    assert sig_a == sig_b


def test_data_config_signature_ignores_purge_for_across_batches_mode():
    dataset_spec = _make_dataset_spec(split_mode="across_batches")
    args_a = _make_args(seed=42, n_train_samples=None)
    args_b = _make_args(seed=42, n_train_samples=None)
    args_b.purged_fraction = 0.5
    sig_a = compute_data_config_signature(dataset_spec=dataset_spec, args=args_a)
    sig_b = compute_data_config_signature(dataset_spec=dataset_spec, args=args_b)
    assert sig_a == sig_b


def test_data_config_signature_changes_with_split_mode():
    args = _make_args(seed=42, n_train_samples=None)
    args.shuffle_batches_before_split = False
    sig_temporal = compute_data_config_signature(
        dataset_spec=_make_dataset_spec(split_mode="temporal"),
        args=args,
    )
    sig_within_batches = compute_data_config_signature(
        dataset_spec=_make_dataset_spec(split_mode="within_batches"),
        args=args,
    )
    assert sig_temporal != sig_within_batches


def test_temporal_signature_matches_reference_contract():
    args = _make_args(seed=42, n_train_samples=None)
    args.shuffle_batches_before_split = False
    signature = compute_data_config_signature(
        dataset_spec=_make_dataset_spec(split_mode="temporal"),
        args=args,
    )
    assert signature == _reference_temporal_signature(args)


def test_temporal_signature_matches_reference_contract_with_sampling_under_current_defaults():
    args = _make_args(seed=42, data_split_seed=None, n_train_samples=10)
    args.shuffle_batches_before_split = False
    signature = compute_data_config_signature(
        dataset_spec=_make_dataset_spec(split_mode="temporal"),
        args=args,
    )
    assert signature == _reference_temporal_signature(args)


def test_data_config_signature_rejects_within_batches_shuffle():
    args = _make_args(seed=42, n_train_samples=None)
    with pytest.raises(ValueError, match="shuffle_batches_before_split must be false"):
        compute_data_config_signature(
            dataset_spec=_make_dataset_spec(split_mode="within_batches"),
            args=args,
        )


def test_tuning_seed_is_deterministic_for_same_scope():
    seed_a = derive_tuning_seed(
        base_seed=123,
        dataset_key="demo",
        architecture="DLinear",
        data_config_signature="sig",
        pipeline_method="adaptive_robust_loss",
        pipeline_kind="train",
        tuning_strategy="random_subgrid",
    )
    seed_b = derive_tuning_seed(
        base_seed=123,
        dataset_key="demo",
        architecture="DLinear",
        data_config_signature="sig",
        pipeline_method="adaptive_robust_loss",
        pipeline_kind="train",
        tuning_strategy="random_subgrid",
    )
    assert seed_a == seed_b


def test_probattention_uses_seed_helper_generator(monkeypatch):
    attention = ProbAttention()
    queries = torch.zeros(1, 1, 4, 2)
    keys = torch.zeros(1, 1, 5, 2)
    captured: dict[str, object] = {}
    seeded_generator = torch.Generator().manual_seed(17)
    real_randint = torch.randint

    def _torch_generator(device):
        captured["generator_device"] = str(device)
        return seeded_generator

    def _randint(high, size, **kwargs):
        captured["generator"] = kwargs.get("generator")
        captured["randint_device"] = kwargs.get("device")
        return real_randint(high, size, **kwargs)

    monkeypatch.setattr(attention_impl, "torch_generator", _torch_generator)
    monkeypatch.setattr(torch, "randint", _randint)

    attention._prob_QK(queries, keys, sample_k=2, n_top=1)

    assert captured["generator"] is seeded_generator
    assert str(captured["generator_device"]) == str(queries.device)
    assert str(captured["randint_device"]) == str(queries.device)


def test_adversarial_attack_generator_uses_model_seed_namespace():
    model = BaseLitModule(
        d_input_features=1,
        d_target_features=1,
        d_seq_in=2,
        d_seq_out=1,
        lr=0.001,
        loss="MSE",
        adversarial_training_mode="pgd_linf",
        advtrain_epsilon=0.1,
        advtrain_step_size=0.02,
        advtrain_attack_steps=1,
        advtrain_random_start=True,
        advtrain_attack_channels="all",
    )

    model.set_model_seed(123)
    first = torch.rand((), generator=model._get_attack_generator())
    model.set_model_seed(123)
    second = torch.rand((), generator=model._get_attack_generator())

    torch.testing.assert_close(first, second)


def test_tuning_seed_changes_when_scope_changes():
    seed_lstm = derive_tuning_seed(
        base_seed=123,
        dataset_key="demo",
        architecture="DLinear",
        data_config_signature="sig",
        pipeline_method="adaptive_robust_loss",
        pipeline_kind="train",
        tuning_strategy="random_subgrid",
    )
    seed_gru = derive_tuning_seed(
        base_seed=123,
        dataset_key="demo",
        architecture="GRU",
        data_config_signature="sig",
        pipeline_method="adaptive_robust_loss",
        pipeline_kind="train",
        tuning_strategy="random_subgrid",
    )
    assert seed_lstm != seed_gru
