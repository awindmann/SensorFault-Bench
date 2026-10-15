import pytest
import torch

from data.samplers import (
    PerturbationSeveritySampler,
    beta_1_2_severity,
    resolve_severity_distribution,
    uniform_severity,
)
from utils.parsing import PERTURBATION_SEVERITY_DISTRIBUTIONS


class _FakePerturbation:
    def __init__(self, name: str):
        self.name = name


def test_sampler_normalizes_probabilities_and_returns_valid_severity():
    perturbations = [_FakePerturbation("a"), _FakePerturbation("b")]
    sampler = PerturbationSeveritySampler(
        perturbations,
        torch.tensor([2.0, 1.0], dtype=torch.float32),
        uniform_severity,
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
            uniform_severity,
        )


def test_sampler_rejects_negative_probability_weight():
    perturbations = [_FakePerturbation("a"), _FakePerturbation("b")]
    with pytest.raises(ValueError, match="must be non-negative"):
        PerturbationSeveritySampler(
            perturbations,
            torch.tensor([1.0, -0.1], dtype=torch.float32),
            uniform_severity,
        )


def test_sampler_rejects_zero_total_probability_mass():
    perturbations = [_FakePerturbation("a"), _FakePerturbation("b")]
    with pytest.raises(ValueError, match="positive total mass"):
        PerturbationSeveritySampler(
            perturbations,
            torch.tensor([0.0, 0.0], dtype=torch.float32),
            uniform_severity,
        )


def test_sampler_requires_callable_severity_distribution():
    perturbations = [_FakePerturbation("a")]
    with pytest.raises(ValueError, match="must be callable"):
        PerturbationSeveritySampler(
            perturbations,
            torch.tensor([1.0], dtype=torch.float32),
            None,
        )


def test_sampler_rejects_non_callable_severity_distribution():
    perturbations = [_FakePerturbation("a")]
    with pytest.raises(ValueError, match="must be callable"):
        PerturbationSeveritySampler(
            perturbations,
            torch.tensor([1.0], dtype=torch.float32),
            0.5,
        )


def test_sampler_rejects_out_of_range_severity_from_distribution():
    perturbations = [_FakePerturbation("a")]
    sampler = PerturbationSeveritySampler(
        perturbations,
        torch.tensor([1.0], dtype=torch.float32),
        lambda _rng: 1.5,
    )
    with pytest.raises(ValueError, match="out-of-range"):
        sampler(torch.Generator().manual_seed(1))


def test_sampler_rejects_non_numeric_severity_from_distribution():
    perturbations = [_FakePerturbation("a")]
    sampler = PerturbationSeveritySampler(
        perturbations,
        torch.tensor([1.0], dtype=torch.float32),
        lambda _rng: "bad",
    )
    with pytest.raises(ValueError, match="non-numeric"):
        sampler(torch.Generator().manual_seed(1))


def test_sampler_rejects_non_finite_severity_from_distribution():
    perturbations = [_FakePerturbation("a")]
    sampler = PerturbationSeveritySampler(
        perturbations,
        torch.tensor([1.0], dtype=torch.float32),
        lambda _rng: float("nan"),
    )
    with pytest.raises(ValueError, match="non-finite"):
        sampler(torch.Generator().manual_seed(1))


def test_sampler_without_grid_uses_continuous_uniform_severity():
    """Regression test: train-time sampling stays continuous U[0,1], not grid."""
    perturbations = [_FakePerturbation("a")]
    sampler = PerturbationSeveritySampler(
        perturbations,
        torch.tensor([1.0], dtype=torch.float32),
        uniform_severity,
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
            uniform_severity,
            severity_grid=[0.5, 1.0],
        )


def test_beta_1_2_severity_matches_inverse_cdf_and_consumes_one_draw():
    actual_rng = torch.Generator().manual_seed(314)
    replay_rng = torch.Generator().manual_seed(314)

    actual = beta_1_2_severity(actual_rng)
    uniform_draw = float(torch.rand((), generator=replay_rng).item())
    expected = 1.0 - (1.0 - uniform_draw) ** 0.5

    assert actual == pytest.approx(expected)
    assert torch.equal(actual_rng.get_state(), replay_rng.get_state())


def test_beta_1_2_severity_is_bounded_deterministic_and_has_expected_mean():
    rng = torch.Generator().manual_seed(2718)
    values = torch.tensor([beta_1_2_severity(rng) for _ in range(20_000)])

    assert torch.all((values >= 0.0) & (values <= 1.0))
    assert float(values.mean()) == pytest.approx(1.0 / 3.0, abs=0.01)

    replay_rng = torch.Generator().manual_seed(2718)
    replay = torch.tensor([beta_1_2_severity(replay_rng) for _ in range(20_000)])
    assert torch.equal(values, replay)


@pytest.mark.parametrize(
    ("distribution_id", "expected"),
    [
        ("uniform", uniform_severity),
        ("beta_1_2", beta_1_2_severity),
    ],
)
def test_resolve_severity_distribution_returns_registered_callable(distribution_id, expected):
    assert resolve_severity_distribution(distribution_id) is expected


def test_resolve_severity_distribution_covers_every_parsed_distribution_id():
    resolved = [
        resolve_severity_distribution(distribution_id)
        for distribution_id in PERTURBATION_SEVERITY_DISTRIBUTIONS
    ]

    assert all(callable(distribution) for distribution in resolved)
    assert len(set(resolved)) == len(PERTURBATION_SEVERITY_DISTRIBUTIONS)
