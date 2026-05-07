import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping
import torch


##### severity probability #####
def uniform_severity(rng: torch.Generator) -> float:
    """S ~ U(0, 1)."""
    return torch.rand((), generator=rng).item()


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


def _normalize_severity_laws(
    severity_laws: Mapping[str, Callable[[torch.Generator], float]] | None,
    *,
    perturbation_names: set[str],
) -> dict[str, Callable[[torch.Generator], float]]:
    if severity_laws is None:
        return {}
    if not isinstance(severity_laws, Mapping):
        raise ValueError("severity_laws must be a mapping from perturbation name to callable.")

    normalized: dict[str, Callable[[torch.Generator], float]] = {}
    unknown_names: list[str] = []
    for raw_name, law in severity_laws.items():
        name = str(raw_name).strip()
        if not name:
            raise ValueError("severity_laws contains an empty perturbation name key.")
        if name not in perturbation_names:
            unknown_names.append(name)
            continue
        if not callable(law):
            raise ValueError(
                f"severity_laws['{name}'] must be callable, got {type(law).__name__}."
            )
        normalized[name] = law

    if unknown_names:
        known = ", ".join(sorted(perturbation_names))
        raise ValueError(
            "severity_laws contains unknown perturbation names: "
            f"{sorted(unknown_names)}. Known perturbations: {known}."
        )
    return normalized


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
        severity_laws: Mapping[str, Callable[[torch.Generator], float]] | None = None,
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
        self.severity_laws = _normalize_severity_laws(
            severity_laws,
            perturbation_names=perturbation_names,
        )

    def __call__(self, rng: torch.Generator):
        idx = int(torch.multinomial(self.prob, 1, generator=rng).item())
        pert = self.perturbations[idx]
        severity_law = self.severity_laws.get(pert.name)
        if severity_law is None:
            sev = uniform_severity(rng)
        else:
            sev = severity_law(rng)
        try:
            sev = float(sev)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Severity law for perturbation '{pert.name}' returned non-numeric value {sev!r}."
            ) from exc
        if not math.isfinite(sev):
            raise ValueError(
                f"Severity law for perturbation '{pert.name}' returned non-finite value {sev}."
            )
        if sev < 0.0 or sev > 1.0:
            raise ValueError(
                f"Severity law for perturbation '{pert.name}' returned out-of-range value {sev}."
            )
        return SampledPerturbationSeverity(
            perturbation=pert,
            severity=sev,
        )
