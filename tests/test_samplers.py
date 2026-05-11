import pytest
import torch

from data.samplers import PerturbationSeveritySampler


class _FakePerturbation:
    def __init__(self, name: str):
        self.name = name


def test_sampler_normalizes_probabilities_and_returns_valid_severity():
    perturbations = [_FakePerturbation("a"), _FakePerturbation("b")]
    sampler = PerturbationSeveritySampler(
        perturbations,
        torch.tensor([2.0, 1.0], dtype=torch.float32),
    )
    assert sampler.prob.tolist() == pytest.approx([2.0 / 3.0, 1.0 / 3.0])

    pert, severity = sampler(torch.Generator().manual_seed(7))
    assert pert.name in {"a", "b"}
    assert 0.0 <= severity <= 1.0


def test_sampler_rejects_probability_length_mismatch():
    perturbations = [_FakePerturbation("a"), _FakePerturbation("b")]
    with pytest.raises(ValueError, match="length mismatch"):
        PerturbationSeveritySampler(
            perturbations,
            torch.tensor([1.0], dtype=torch.float32),
        )


def test_sampler_rejects_negative_probability_weight():
    perturbations = [_FakePerturbation("a"), _FakePerturbation("b")]
    with pytest.raises(ValueError, match="must be non-negative"):
        PerturbationSeveritySampler(
            perturbations,
            torch.tensor([1.0, -0.1], dtype=torch.float32),
        )


def test_sampler_rejects_zero_total_probability_mass():
    perturbations = [_FakePerturbation("a"), _FakePerturbation("b")]
    with pytest.raises(ValueError, match="positive total mass"):
        PerturbationSeveritySampler(
            perturbations,
            torch.tensor([0.0, 0.0], dtype=torch.float32),
        )


def test_sampler_rejects_unknown_severity_law_names():
    perturbations = [_FakePerturbation("a")]
    with pytest.raises(ValueError, match="unknown perturbation names"):
        PerturbationSeveritySampler(
            perturbations,
            torch.tensor([1.0], dtype=torch.float32),
            severity_laws={"missing": lambda _rng: 0.5},
        )


def test_sampler_rejects_non_callable_severity_law():
    perturbations = [_FakePerturbation("a")]
    with pytest.raises(ValueError, match="must be callable"):
        PerturbationSeveritySampler(
            perturbations,
            torch.tensor([1.0], dtype=torch.float32),
            severity_laws={"a": 0.5},
        )


def test_sampler_rejects_out_of_range_severity_from_law():
    perturbations = [_FakePerturbation("a")]
    sampler = PerturbationSeveritySampler(
        perturbations,
        torch.tensor([1.0], dtype=torch.float32),
        severity_laws={"a": lambda _rng: 1.5},
    )
    with pytest.raises(ValueError, match="out-of-range"):
        sampler(torch.Generator().manual_seed(1))


def test_sampler_rejects_non_numeric_severity_from_law():
    perturbations = [_FakePerturbation("a")]
    sampler = PerturbationSeveritySampler(
        perturbations,
        torch.tensor([1.0], dtype=torch.float32),
        severity_laws={"a": lambda _rng: "bad"},
    )
    with pytest.raises(ValueError, match="non-numeric"):
        sampler(torch.Generator().manual_seed(1))


def test_sampler_rejects_non_finite_severity_from_law():
    perturbations = [_FakePerturbation("a")]
    sampler = PerturbationSeveritySampler(
        perturbations,
        torch.tensor([1.0], dtype=torch.float32),
        severity_laws={"a": lambda _rng: float("nan")},
    )
    with pytest.raises(ValueError, match="non-finite"):
        sampler(torch.Generator().manual_seed(1))


def test_sampler_without_grid_uses_continuous_uniform_severity():
    """Regression test: train-time sampling stays continuous U[0,1], not grid."""
    perturbations = [_FakePerturbation("a")]
    sampler = PerturbationSeveritySampler(
        perturbations,
        torch.tensor([1.0], dtype=torch.float32),
    )
    rng = torch.Generator().manual_seed(42)
    severities = set()
    for _ in range(50):
        result = sampler(rng)
        assert 0.0 <= result.severity <= 1.0
        assert result.severity_idx is None
        severities.add(result.severity)
    # Continuous sampling should produce many distinct values (not grid-quantized)
    assert len(severities) > 10


def test_sampler_no_longer_accepts_removed_severity_grid_kwarg():
    perturbations = [_FakePerturbation("a")]

    with pytest.raises(TypeError, match="severity_grid"):
        PerturbationSeveritySampler(
            perturbations,
            torch.tensor([1.0], dtype=torch.float32),
            severity_grid=[0.5, 1.0],
        )
