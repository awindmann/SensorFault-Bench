import math
from dataclasses import dataclass
from typing import Any, Callable

import torch

from utils.parsing import parse_perturbation_severity_distribution


##### severity probability #####
def uniform_severity(rng: torch.Generator) -> float:
    """S ~ U(0, 1)."""
    return torch.rand((), generator=rng).item()


def beta_1_2_severity(rng: torch.Generator) -> float:
    """S ~ Beta(1, 2), sampled by inverse CDF from one uniform draw."""
    uniform_value = float(torch.rand((), generator=rng).item())
    return 1.0 - math.sqrt(1.0 - uniform_value)


@dataclass(frozen=True)
class BoundedUniformSeverity:
    """S ~ U(0, bound).

    A module-level callable, so data-loader workers started with `spawn` can
    pickle datasets that hold it.
    """

    bound: float

    def __call__(self, rng: torch.Generator) -> float:
        return float(torch.rand((), generator=rng).item() * self.bound)


def resolve_severity_distribution(
    distribution_id: Any,
    *,
    key: str = "perturbation_severity_distribution",
) -> Callable[[torch.Generator], float]:
    """Resolve one parsed severity-distribution id to its sampling callable."""
    parsed_distribution_id = parse_perturbation_severity_distribution(distribution_id, key=key)
    if parsed_distribution_id == "uniform":
        return uniform_severity
    if parsed_distribution_id == "beta_1_2":
        return beta_1_2_severity
    raise AssertionError(
        f"Unhandled perturbation severity sampling distribution {parsed_distribution_id!r}."
    )


def sample_validated_severity(
    severity_distribution: Callable[[torch.Generator], float],
    rng: torch.Generator,
    *,
    context: str,
) -> float:
    """Sample and validate one finite severity in [0, 1]."""
    if not callable(severity_distribution):
        raise ValueError(
            f"{context} severity sampling distribution must be callable, got "
            f"{type(severity_distribution).__name__}."
        )
    sampled = severity_distribution(rng)
    try:
        severity = float(sampled)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{context} severity sampling distribution returned non-numeric value {sampled!r}."
        ) from exc
    if not math.isfinite(severity):
        raise ValueError(
            f"{context} severity sampling distribution returned non-finite value {severity}."
        )
    if severity < 0.0 or severity > 1.0:
        raise ValueError(
            f"{context} severity sampling distribution returned out-of-range value {severity}."
        )
    return severity


def _require_probability_vector(
    p_perturbation: torch.Tensor,
    *,
    n_expected: int,
) -> torch.Tensor:
    if not isinstance(p_perturbation, torch.Tensor):
        raise ValueError("p_perturbation must be a torch.Tensor.")
    probs = p_perturbation.to(dtype=torch.float32)
    if probs.ndim != 1:
        raise ValueError(
            f"p_perturbation must be a 1D tensor; got shape {tuple(probs.shape)}."
        )
    if probs.numel() != n_expected:
        raise ValueError(
            f"p_perturbation length mismatch: expected {n_expected}, got {probs.numel()}."
        )
    if not torch.isfinite(probs).all():
        raise ValueError("p_perturbation contains non-finite values.")
    if (probs < 0).any():
        raise ValueError("p_perturbation must be non-negative.")
    total = float(probs.sum().item())
    if total <= 0.0:
        raise ValueError("p_perturbation must have positive total mass.")
    return probs / total


##### perturbation-severity sampler #####
@dataclass(frozen=True)
class SampledPerturbationSeverity:
    perturbation: Any
    severity: float
    severity_idx: int | None = None

    def __iter__(self):
        yield self.perturbation
        yield self.severity


def normalize_sampled_perturbation(
    sample: Any,
    *,
    context: str,
) -> SampledPerturbationSeverity:
    if isinstance(sample, SampledPerturbationSeverity):
        return sample
    if isinstance(sample, (tuple, list)):
        if len(sample) == 2:
            perturbation, severity = sample
            severity_idx = None
        elif len(sample) == 3:
            perturbation, severity_idx, severity = sample
        else:
            raise ValueError(
                f"{context} must return either (perturbation, severity) or "
                "(perturbation, severity_idx, severity)."
            )
        normalized_idx: int | None = None
        if severity_idx is not None:
            if isinstance(severity_idx, bool):
                raise ValueError(f"{context} returned invalid boolean severity_idx={severity_idx!r}.")
            try:
                normalized_idx = int(severity_idx)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{context} returned non-integer severity_idx={severity_idx!r}."
                ) from exc
        try:
            normalized_severity = float(severity)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{context} returned non-numeric severity={severity!r}."
            ) from exc
        return SampledPerturbationSeverity(
            perturbation=perturbation,
            severity=normalized_severity,
            severity_idx=normalized_idx,
        )
    raise ValueError(
        f"{context} must return SampledPerturbationSeverity or a tuple/list payload."
    )


class PerturbationSeveritySampler:
    def __init__(
        self,
        perturbations: list,
        p_perturbation: torch.Tensor,
        severity_distribution: Callable[[torch.Generator], float],
    ):
        if not perturbations:
            raise ValueError("perturbations must be a non-empty list.")
        perturbation_names: set[str] = set()
        for perturbation in perturbations:
            name_raw = getattr(perturbation, "name", None)
            if name_raw is None or not str(name_raw).strip():
                raise ValueError(
                    "Each perturbation must define a non-empty 'name' attribute."
                )
            name = str(name_raw).strip()
            if name in perturbation_names:
                raise ValueError(
                    f"Duplicate perturbation name '{name}' in perturbation sampler."
                )
            perturbation_names.add(name)

        self.perturbations = perturbations
        self.prob = _require_probability_vector(
            p_perturbation,
            n_expected=len(perturbations),
        )
        if not callable(severity_distribution):
            raise ValueError(
                "severity_distribution must be callable, got "
                f"{type(severity_distribution).__name__}."
            )
        self.severity_distribution = severity_distribution

    def __call__(self, rng: torch.Generator):
        idx = int(torch.multinomial(self.prob, 1, generator=rng).item())
        pert = self.perturbations[idx]
        sev = sample_validated_severity(
            self.severity_distribution,
            rng,
            context=f"Severity sampling distribution for perturbation '{pert.name}'",
        )
        return SampledPerturbationSeverity(
            perturbation=pert,
            severity=sev,
        )
