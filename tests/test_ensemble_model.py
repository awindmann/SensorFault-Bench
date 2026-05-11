import torch
import torch.nn as nn
import pytest

from models.base_module import BaseLitModule
from models.ensemble import Ensemble


class _DummyBackbone(BaseLitModule):
    def __init__(self, pred_value: float):
        super().__init__(
            d_input_features=1,
            d_target_features=1,
            d_seq_in=2,
            d_seq_out=1,
            lr_scheduler=False,
            target_indices=(0,),
        )
        self.pred_value = pred_value

    def _shared_step(self, x, y):
        pred = torch.full(
            (x.shape[0], self.d_seq_out, self.d_output_features),
            self.pred_value,
            dtype=x.dtype,
            device=x.device,
        )
        return {"pred": pred, "target": y, "loss": None}


class _ParameterizedBackbone(BaseLitModule):
    def __init__(self):
        super().__init__(
            d_input_features=1,
            d_target_features=1,
            d_seq_in=2,
            d_seq_out=1,
            lr_scheduler=False,
            target_indices=(0,),
        )
        self.linear = nn.Linear(1, 1)

    def _shared_step(self, x, y):
        pred = self.linear(x[:, -1:, :])
        return {"pred": pred, "target": y, "loss": None}


def test_ensemble_median_combines_member_predictions():
    model = Ensemble(
        backbones=[
            _DummyBackbone(0.0),
            _DummyBackbone(10.0),
            _DummyBackbone(100.0),
        ],
        combine_method="median",
    )

    x = torch.zeros(2, 2, 1)
    output = model._shared_step(x, None)

    expected = torch.full((2, 1, 1), 10.0)
    assert torch.equal(output["pred"], expected)


def test_ensemble_median_averages_two_middle_predictions_for_even_member_count():
    model = Ensemble(
        backbones=[
            _DummyBackbone(0.0),
            _DummyBackbone(10.0),
            _DummyBackbone(20.0),
            _DummyBackbone(40.0),
        ],
        combine_method="median",
    )

    x = torch.zeros(2, 2, 1)
    output = model._shared_step(x, None)

    expected = torch.full((2, 1, 1), 15.0)
    assert torch.equal(output["pred"], expected)


def test_ensemble_rejects_invalid_combine_method():
    with pytest.raises(ValueError, match="Unsupported ensemble_combine_method 'average'"):
        Ensemble(
            backbones=[_DummyBackbone(1.0)],
            combine_method="average",
        )


def test_ensemble_set_test_mode_propagates_to_members():
    left = _DummyBackbone(1.0)
    right = _DummyBackbone(2.0)
    model = Ensemble(
        backbones=[left, right],
        combine_method="mean",
    )

    model.set_test_mode(test_metric="MSE")

    assert model.test_metric == "MSE"
    assert left.test_metric == "MSE"
    assert right.test_metric == "MSE"


def test_ensemble_hparams_exclude_live_backbone_modules():
    model = Ensemble(
        backbones=[_ParameterizedBackbone(), _ParameterizedBackbone()],
        combine_method="mean",
    )

    assert "backbones" not in model.hparams
    assert model.hparams["backbone_count"] == 2
    assert model.hparams["backbone_architectures"] == [
        "_ParameterizedBackbone",
        "_ParameterizedBackbone",
    ]
