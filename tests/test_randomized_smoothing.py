import math
import warnings

import pytest
import torch

from models.base_module import BaseLitModule
from models.randomized_smoothing import RandomizedSmoothing


class _DummyBackbone(BaseLitModule):
    def __init__(self):
        super().__init__(
            d_input_features=2,
            d_target_features=2,
            d_seq_in=2,
            d_seq_out=1,
            target_indices=(0, 1),
            lr_scheduler=False,
        )
        self.loss_fn = lambda pred, target: (pred - target) ** 2

    def _shared_step(self, x, y):
        pred = x[:, -1:, :]
        result = {"pred": pred}
        if y is not None:
            result["loss"] = ((pred - y[:, :1, :]) ** 2).mean()
        return result


class _QueuedBackbone(BaseLitModule):
    def __init__(self, preds: list[torch.Tensor]):
        if not preds:
            raise ValueError("preds must be non-empty.")
        first = torch.as_tensor(preds[0], dtype=torch.float32)
        if first.ndim != 3:
            raise ValueError("Queued predictions must have shape (B, T_out, F).")
        super().__init__(
            d_input_features=first.shape[-1],
            d_target_features=first.shape[-1],
            d_seq_in=2,
            d_seq_out=first.shape[1],
            target_indices=tuple(range(first.shape[-1])),
            lr_scheduler=False,
        )
        self.loss_fn = lambda pred, target: (pred - target) ** 2
        self._preds = [torch.as_tensor(pred, dtype=torch.float32) for pred in preds]
        self._next_idx = 0

    def _shared_step(self, x, y):
        if self._next_idx >= len(self._preds):
            raise AssertionError("Queued backbone received more calls than predictions.")
        pred = self._preds[self._next_idx].to(x.device, dtype=x.dtype)
        self._next_idx += 1
        result = {"pred": pred}
        if y is not None:
            result["loss"] = ((pred - y) ** 2).mean()
        return result


def _make_wrapper(
    *,
    noise_std,
    sample_count,
    seed=None,
    trim_alpha=0.35,
    backbone=None,
):
    wrapper = RandomizedSmoothing(
        wrapped_backbone=backbone or _DummyBackbone(),
        noise_std=noise_std,
        sample_count=sample_count,
        trim_alpha=trim_alpha,
    )
    if seed is not None:
        wrapper.set_noise_generator(torch.Generator().manual_seed(seed))
    return wrapper


def _bind_eval_context(wrapper, *, input_columns, target_columns):
    wrapper.bind_eval_context(
        input_columns=input_columns,
        target_columns=target_columns,
        continuous_channels=input_columns,
        input_means=torch.arange(1, len(input_columns) + 1, dtype=torch.float32),
        input_stds=torch.arange(1, len(input_columns) + 1, dtype=torch.float32),
    )


def _expected_trimmed_prediction(
    *,
    x: torch.Tensor,
    noise_std: float,
    sample_count: int,
    trim_alpha: float,
    seed: int,
) -> torch.Tensor:
    preds = []
    generator = torch.Generator().manual_seed(seed)
    for _ in range(sample_count):
        eps = torch.randn(
            x.shape,
            dtype=x.dtype,
            device=x.device,
            generator=generator,
        )
        preds.append((x + noise_std * eps)[:, -1:, :])
    stacked = torch.stack(preds, dim=0)
    trim_count = math.floor(trim_alpha * sample_count)
    return torch.sort(stacked, dim=0).values[
        trim_count : sample_count - trim_count
    ].mean(dim=0)


def test_randomized_smoothing_uses_additive_gaussian_input_noise():
    wrapper = _make_wrapper(
        noise_std=0.2,
        sample_count=4,
        seed=5,
        trim_alpha=0.25,
    )
    x = torch.tensor(
        [[[-1.0, 1.0], [0.5, -0.5]]],
        dtype=torch.float32,
    )
    y = torch.zeros((1, 1, 2), dtype=torch.float32)

    expected_pred = _expected_trimmed_prediction(
        x=x,
        noise_std=0.2,
        sample_count=4,
        trim_alpha=0.25,
        seed=5,
    )

    outputs = wrapper._shared_step(x, y)

    assert torch.allclose(outputs["pred"], expected_pred)


def test_randomized_smoothing_set_test_mode_propagates_to_backbone():
    backbone = _DummyBackbone()
    wrapper = _make_wrapper(
        noise_std=0.2,
        sample_count=4,
        seed=5,
        trim_alpha=0.25,
        backbone=backbone,
    )

    wrapper.set_test_mode(test_metric="MSE")

    assert wrapper.test_metric == "MSE"
    assert backbone.test_metric == "MSE"


def test_randomized_smoothing_perturbs_all_input_channels():
    wrapper = _make_wrapper(
        noise_std=0.1,
        sample_count=4,
        seed=9,
        trim_alpha=0.25,
    )
    x = torch.tensor(
        [[[0.0, -1.0], [1.0, 0.5]]],
        dtype=torch.float32,
    )
    y = torch.zeros((1, 1, 2), dtype=torch.float32)

    expected_pred = _expected_trimmed_prediction(
        x=x,
        noise_std=0.1,
        sample_count=4,
        trim_alpha=0.25,
        seed=9,
    )

    outputs = wrapper._shared_step(x, y)

    assert torch.allclose(outputs["pred"], expected_pred)


def test_randomized_smoothing_trimmed_aggregation_rejects_outlier_mean():
    preds = [
        torch.tensor([[[-100.0]]]),
        torch.tensor([[[0.0]]]),
        torch.tensor([[[1.0]]]),
        torch.tensor([[[2.0]]]),
        torch.tensor([[[100.0]]]),
    ]
    wrapper = _make_wrapper(
        noise_std=0.0,
        sample_count=5,
        seed=1,
        trim_alpha=0.2,
        backbone=_QueuedBackbone(preds),
    )
    _bind_eval_context(wrapper, input_columns=("target",), target_columns=("target",))

    outputs = wrapper._shared_step(
        torch.zeros((1, 2, 1), dtype=torch.float32),
        None,
    )

    assert torch.allclose(outputs["pred"], torch.tensor([[[1.0]]]))
    assert not torch.allclose(
        outputs["pred"],
        torch.tensor([[[0.6]]]),
    )


def test_randomized_smoothing_trims_coordinatewise_along_sample_axis():
    preds = [
        torch.tensor([[[0.0, 100.0], [5.0, 50.0]]]),
        torch.tensor([[[1.0, 90.0], [4.0, 40.0]]]),
        torch.tensor([[[2.0, 80.0], [3.0, 30.0]]]),
        torch.tensor([[[100.0, 70.0], [2.0, 20.0]]]),
        torch.tensor([[[3.0, -100.0], [1.0, 10.0]]]),
    ]
    wrapper = _make_wrapper(
        noise_std=0.0,
        sample_count=5,
        seed=2,
        trim_alpha=0.2,
        backbone=_QueuedBackbone(preds),
    )
    _bind_eval_context(wrapper, input_columns=("f0", "f1"), target_columns=("f0", "f1"))

    outputs = wrapper._shared_step(
        torch.zeros((1, 2, 2), dtype=torch.float32),
        None,
    )

    expected = torch.tensor([[[2.0, 80.0], [3.0, 30.0]]], dtype=torch.float32)
    assert torch.allclose(outputs["pred"], expected)


def test_randomized_smoothing_recomputes_loss_from_trimmed_prediction():
    preds = [
        torch.tensor([[[0.0]]]),
        torch.tensor([[[2.0]]]),
        torch.tensor([[[10.0]]]),
    ]
    wrapper = _make_wrapper(
        noise_std=0.0,
        sample_count=3,
        seed=3,
        trim_alpha=0.34,
        backbone=_QueuedBackbone(preds),
    )
    _bind_eval_context(wrapper, input_columns=("target",), target_columns=("target",))
    y = torch.zeros((1, 1, 1), dtype=torch.float32)

    outputs = wrapper._shared_step(
        torch.zeros((1, 2, 1), dtype=torch.float32),
        y,
    )

    assert torch.allclose(outputs["pred"], torch.tensor([[[2.0]]]))
    assert outputs["loss"].item() == pytest.approx(4.0)
    assert outputs["loss"].item() != pytest.approx((0.0 + 4.0 + 100.0) / 3.0)


@pytest.mark.parametrize(
    ("trim_alpha", "sample_count", "expected_message"),
    [
        (0.0, 100, "must satisfy 0 < rs_trim_alpha < 0.5"),
        (-0.1, 100, "must satisfy 0 < rs_trim_alpha < 0.5"),
        (0.5, 100, "must satisfy 0 < rs_trim_alpha < 0.5"),
        (float("nan"), 100, "must be finite"),
        (float("inf"), 100, "must be finite"),
        (0.1, 5, "trims zero samples per tail"),
    ],
)
def test_randomized_smoothing_rejects_invalid_trim_alpha(
    trim_alpha,
    sample_count,
    expected_message,
):
    with pytest.raises(ValueError, match=expected_message):
        _make_wrapper(
            noise_std=0.1,
            sample_count=sample_count,
            seed=4,
            trim_alpha=trim_alpha,
        )


def test_randomized_smoothing_trims_independently_across_batches():
    """Verify trimming operates per (B, T_out, F) element, not across batches."""
    # Batch 0 has outliers at the top. Batch 1 has outliers at the bottom.
    preds = [
        torch.tensor([[[100.0]], [[-100.0]]]),  # outlier high b0, outlier low b1
        torch.tensor([[[1.0]], [[1.0]]]),
        torch.tensor([[[2.0]], [[2.0]]]),
        torch.tensor([[[3.0]], [[3.0]]]),
        torch.tensor([[[-100.0]], [[100.0]]]),  # outlier low b0, outlier high b1
    ]
    wrapper = _make_wrapper(
        noise_std=0.0,
        sample_count=5,
        seed=1,
        trim_alpha=0.2,
        backbone=_QueuedBackbone(preds),
    )
    _bind_eval_context(wrapper, input_columns=("f",), target_columns=("f",))

    outputs = wrapper._shared_step(
        torch.zeros((2, 2, 1), dtype=torch.float32),
        None,
    )

    # Both batches should independently trim their own outliers and get mean([1,2,3]) = 2.
    expected = torch.tensor([[[2.0]], [[2.0]]], dtype=torch.float32)
    assert torch.allclose(outputs["pred"], expected)


def test_randomized_smoothing_bound_noise_sample_ids_make_batch_and_singleton_predictions_match():
    wrapper = _make_wrapper(
        noise_std=0.2,
        sample_count=4,
        seed=5,
        trim_alpha=0.25,
    )
    _bind_eval_context(wrapper, input_columns=("target",), target_columns=("target",))
    x_batch = torch.tensor(
        [
            [[1.0], [2.0]],
            [[3.0], [4.0]],
        ],
        dtype=torch.float32,
    )
    y_batch = torch.zeros((2, 1, 1), dtype=torch.float32)

    wrapper.bind_noise_sample_ids([0, 1], context_key="degradation:scenario:0")
    try:
        batch_outputs = wrapper._shared_step(x_batch, y_batch)
    finally:
        wrapper.clear_noise_sample_ids()

    wrapper.bind_noise_sample_ids([1], context_key="degradation:scenario:0")
    try:
        singleton_outputs = wrapper._shared_step(x_batch[1:2], y_batch[1:2])
    finally:
        wrapper.clear_noise_sample_ids()

    assert torch.allclose(batch_outputs["pred"][1:2], singleton_outputs["pred"])
    assert batch_outputs["loss"].item() >= 0.0


def test_randomized_smoothing_rejects_backbone_without_loss_fn():
    class _NoLossBackbone(_DummyBackbone):
        def __init__(self):
            super().__init__()
            del self.loss_fn

    with pytest.raises(ValueError, match="requires wrapped_backbone.loss_fn"):
        RandomizedSmoothing(
            wrapped_backbone=_NoLossBackbone(),
            noise_std=0.1,
            sample_count=5,
            trim_alpha=0.2,
        )


def test_randomized_smoothing_construction_avoids_wrapped_backbone_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        RandomizedSmoothing(
            wrapped_backbone=_DummyBackbone(),
            noise_std=0.1,
            sample_count=5,
            trim_alpha=0.2,
        )

    assert all("wrapped_backbone" not in str(w.message) for w in caught)


def test_randomized_smoothing_requires_explicit_noise_generator():
    wrapper = _make_wrapper(
        noise_std=0.1,
        sample_count=5,
        trim_alpha=0.2,
    )
    _bind_eval_context(wrapper, input_columns=("a", "b"), target_columns=("a", "b"))

    with pytest.raises(ValueError, match="set_noise_generator"):
        wrapper._shared_step(
            torch.zeros((1, 2, 2), dtype=torch.float32),
            torch.zeros((1, 1, 2), dtype=torch.float32),
        )


def test_randomized_smoothing_does_not_require_bound_eval_context():
    wrapper = _make_wrapper(
        noise_std=0.1,
        sample_count=5,
        seed=1,
        trim_alpha=0.2,
    )
    outputs = wrapper._shared_step(
        torch.zeros((1, 2, 2), dtype=torch.float32),
        torch.zeros((1, 1, 2), dtype=torch.float32),
    )

    assert outputs["pred"].shape == (1, 1, 2)


def test_randomized_smoothing_bind_eval_context_is_a_no_op():
    wrapper = _make_wrapper(
        noise_std=0.1,
        sample_count=5,
        seed=1,
        trim_alpha=0.2,
    )
    _bind_eval_context(wrapper, input_columns=("a", "b"), target_columns=("a", "b"))

    outputs = wrapper._shared_step(
        torch.zeros((1, 2, 2), dtype=torch.float32),
        torch.zeros((1, 1, 2), dtype=torch.float32),
    )

    assert outputs["pred"].shape == (1, 1, 2)
