"""
Perturbation operators for CPS multivariate time-series.

Each operator is callable:
    x_noisy, y_noisy, affected_channels = op(
        x, y,                               # tensors
        severity,                           # float in [0, 1]
        rng,                                # torch.Generator
        cont_channels, disc_channels        # indices of continuous and discrete channels
    )
"""

import inspect
import json
import math
import numpy as np
import torch
from typing import Any, List, Sequence

from utils.parsing import (
    parse_perturbation_channel_fraction_max,
    parse_perturbation_scenarios,
)


def _parse_channel_fraction_max(value: float) -> float:
    return parse_perturbation_channel_fraction_max(value, key="channel_frac")


def _parse_severity(value: float) -> float:
    severity = float(value)
    if severity < 0.0 or severity > 1.0:
        raise ValueError(f"severity must satisfy 0 <= severity <= 1; got {severity}.")
    return severity


VALID_PERTURBATION_CHANNEL_SCOPES = frozenset({"continuous", "discrete", "all"})


def require_perturbation_channel_scope(
    perturbation: Any,
    *,
    context: str,
) -> str:
    if not hasattr(perturbation, "channel_scope"):
        raise ValueError(f"{context} is missing required channel_scope metadata.")
    scope = getattr(perturbation, "channel_scope")
    if scope not in VALID_PERTURBATION_CHANNEL_SCOPES:
        raise ValueError(
            f"{context} has invalid channel_scope={scope!r}. "
            "Expected one of: 'continuous', 'discrete', 'all'."
        )
    return scope


def _severity_coupled_channel_count(
    *,
    pool_size: int,
    max_fraction: float,
    severity: float,
) -> int:
    if pool_size < 0:
        raise ValueError(f"pool_size must be >= 0; got {pool_size}.")
    if pool_size == 0:
        return 0
    if severity == 0.0:
        return 0
    k_max = int(math.ceil(max_fraction * pool_size))
    if k_max < 1 or k_max > pool_size:
        raise ValueError(
            f"Invalid k_max={k_max} for pool_size={pool_size} and max_fraction={max_fraction}."
        )
    if severity == 1.0:
        return k_max
    return 1 + int(math.floor(severity * (k_max - 1)))


def select_channels(
    channel_pool: List[int],
    max_fraction: float,
    severity: float,
    rng: torch.Generator,
    *,
    max_fraction_validated: bool = False,
    severity_validated: bool = False,
) -> List[int]:
    """Select a severity-coupled subset from a given channel pool."""
    if not channel_pool:
        return []
    if max_fraction_validated:
        try:
            parsed_fraction = float(max_fraction)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "max_fraction must be numeric when max_fraction_validated=True; "
                f"got {max_fraction!r}."
            ) from exc
        if parsed_fraction <= 0.0 or parsed_fraction > 1.0:
            raise ValueError(
                "max_fraction must satisfy 0 < max_fraction <= 1 when "
                f"max_fraction_validated=True; got {parsed_fraction}."
            )
    else:
        parsed_fraction = _parse_channel_fraction_max(max_fraction)
    if severity_validated:
        try:
            parsed_severity = float(severity)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "severity must be numeric when severity_validated=True; "
                f"got {severity!r}."
            ) from exc
        if parsed_severity < 0.0 or parsed_severity > 1.0:
            raise ValueError(
                "severity must satisfy 0 <= severity <= 1 when "
                f"severity_validated=True; got {parsed_severity}."
            )
    else:
        parsed_severity = _parse_severity(severity)
    k = _severity_coupled_channel_count(
        pool_size=len(channel_pool),
        max_fraction=parsed_fraction,
        severity=parsed_severity,
    )
    if k == 0:
        return []
    perm_indices = torch.randperm(len(channel_pool), generator=rng)[:k]
    return [channel_pool[i] for i in perm_indices]


def randint_py(low: int, high: int, rng: torch.Generator) -> int:
    """Generate a single random integer, compatible with torch Generator."""
    return int(torch.randint(low, high, (1,), generator=rng).item())


def lin_interp(severity: float, low: float, high: float) -> float:
    """Linear interpolation."""
    parsed_severity = _parse_severity(severity)
    return low + parsed_severity * (high - low)


def _constructor_default_payload(perturbation_cls: type) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    signature = inspect.signature(perturbation_cls.__init__)
    for name, parameter in signature.parameters.items():
        if name in {"self", "channel_frac"}:
            continue
        if parameter.default is inspect.Parameter.empty:
            continue
        defaults[name] = parameter.default
    return {
        "class": perturbation_cls.__name__,
        "channel_scope": require_perturbation_channel_scope(
            perturbation_cls,
            context=f"perturbation class {perturbation_cls.__name__}",
        ),
        "defaults": defaults,
    }


def build_perturbation_scenario_params_signature(names: Sequence[str]) -> str:
    """Build an order-preserving signature for selected perturbation semantics."""
    scenario_names = parse_perturbation_scenarios(
        names,
        key="perturbation_scenario_params_signature_source",
    )
    payload = []
    for name in scenario_names:
        perturbation_cls = PERTURBATION_REGISTRY.get(name)
        if perturbation_cls is None:
            raise ValueError(
                f"Unknown perturbation scenario '{name}' in "
                "perturbation_scenario_params_signature_source."
            )
        payload.append(
            {
                "name": name,
                **_constructor_default_payload(perturbation_cls),
            }
        )
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


##### continuous channels #####

class Drift:
    """Add a constant offset to a fraction of channels."""
    name = "drift"
    channel_scope = "continuous"

    def __init__(self, channel_frac: float = 0.1, *, min_offset=0.0, max_offset=0.75):
        self.ch_frac = _parse_channel_fraction_max(channel_frac)
        self.min_off = min_offset
        self.max_off = max_offset

    def __call__(self, x, y, severity, rng, cont_channels, _disc_channels):
        if severity == 0: return x, y, []
        offset = lin_interp(severity, self.min_off, self.max_off)
        channels = select_channels(
            cont_channels,
            self.ch_frac,
            severity,
            rng,
            max_fraction_validated=True,
            severity_validated=True,
        )
        if not channels: return x, y, []
        x = x.clone()
        x[:, channels] += offset
        return x, y, channels


# FaultsToFeatures sensor failure modes.
#
# Reference: Brandt et al., 2025
# Paper: https://openreview.net/forum?id=9aElHWiZ72
# Repo: https://github.com/JBrandt97/FaultsToFeatures
# F2F: linear drift failure mode (beta(t) linear).
class LinearDrift:
    """Add a linear time-varying offset beta(t) = rate * t."""
    name = "linear_drift"
    channel_scope = "continuous"

    def __init__(self, channel_frac: float = 0.1, *, min_drift=0.0, max_drift=1.0):
        self.ch_frac = _parse_channel_fraction_max(channel_frac)
        self.min_drift = min_drift
        self.max_drift = max_drift

    def __call__(self, x, y, severity, rng, cont_channels, _disc_channels):
        if severity == 0: return x, y, []
        max_drift = lin_interp(severity, self.min_drift, self.max_drift)
        channels = select_channels(
            cont_channels,
            self.ch_frac,
            severity,
            rng,
            max_fraction_validated=True,
            severity_validated=True,
        )
        if not channels: return x, y, []
        x = x.clone()
        seq_len = x.size(0)
        time_steps = torch.arange(seq_len, device=x.device, dtype=x.dtype)
        denom = float(max(seq_len - 1, 1))
        drift = (max_drift / denom) * time_steps
        x[:, channels] += drift.unsqueeze(-1)
        return x, y, channels


# F2F: nonlinear drift failure mode (beta(t) quadratic).
class NonlinearDrift:
    """Add a quadratic time-varying offset beta(t) = c1*t + c2*t^2."""
    name = "nonlinear_drift"
    channel_scope = "continuous"

    def __init__(
        self,
        channel_frac: float = 0.1,
        *,
        min_linear=0.0,
        max_linear=0.5,
        min_quadratic=0.0,
        max_quadratic=0.5,
    ):
        self.ch_frac = _parse_channel_fraction_max(channel_frac)
        self.min_linear = min_linear
        self.max_linear = max_linear
        self.min_quadratic = min_quadratic
        self.max_quadratic = max_quadratic

    def __call__(self, x, y, severity, rng, cont_channels, _disc_channels):
        if severity == 0: return x, y, []
        linear_mag = lin_interp(severity, self.min_linear, self.max_linear)
        quad_mag = lin_interp(severity, self.min_quadratic, self.max_quadratic)
        channels = select_channels(
            cont_channels,
            self.ch_frac,
            severity,
            rng,
            max_fraction_validated=True,
            severity_validated=True,
        )
        if not channels: return x, y, []
        x = x.clone()
        seq_len = x.size(0)
        time_steps = torch.arange(seq_len, device=x.device, dtype=x.dtype)
        denom_linear = float(max(seq_len - 1, 1))
        denom_quad = float(max((seq_len - 1) ** 2, 1))
        linear_coeff = linear_mag / denom_linear
        quad_coeff = quad_mag / denom_quad
        drift = linear_coeff * time_steps + quad_coeff * (time_steps ** 2)
        x[:, channels] += drift.unsqueeze(-1)
        return x, y, channels


class Attenuation:
    """Multiply continuous channels with a factor in [1, 0]."""
    name = "attenuation"
    channel_scope = "continuous"

    def __init__(self, channel_frac: float = 0.1, *, min_factor=1.0, max_factor=0.25):
        self.ch_frac = _parse_channel_fraction_max(channel_frac)
        self.min_fac = min_factor
        self.max_fac = max_factor

    def __call__(self, x, y, severity, rng, cont_channels, _disc_channels):
        if severity == 0: return x, y, []
        factor = lin_interp(severity, self.min_fac, self.max_fac)
        channels = select_channels(
            cont_channels,
            self.ch_frac,
            severity,
            rng,
            max_fraction_validated=True,
            severity_validated=True,
        )
        if not channels: return x, y, []
        x = x.clone()
        x[:, channels] *= factor
        return x, y, channels


# F2F: scaling failure mode (alpha constant).
class Scaling:
    """Multiply continuous channels with a factor > 1."""
    name = "scaling"
    channel_scope = "continuous"

    def __init__(self, channel_frac: float = 0.1, *, min_factor=1.0, max_factor=2.0):
        self.ch_frac = _parse_channel_fraction_max(channel_frac)
        self.min_fac = min_factor
        self.max_fac = max_factor

    def __call__(self, x, y, severity, rng, cont_channels, _disc_channels):
        if severity == 0: return x, y, []
        factor = lin_interp(severity, self.min_fac, self.max_fac)
        channels = select_channels(
            cont_channels,
            self.ch_frac,
            severity,
            rng,
            max_fraction_validated=True,
            severity_validated=True,
        )
        if not channels: return x, y, []
        x = x.clone()
        x[:, channels] *= factor
        return x, y, channels


# F2F: time-varying scaling failure mode (alpha(t) linear over time).
class TimeVaryingScaling:
    """Apply a linear scaling factor from 1.0 to max over the sequence."""
    name = "time_varying_scaling"
    channel_scope = "continuous"

    def __init__(self, channel_frac: float = 0.1, *, min_factor=1.0, max_factor=2.0):
        self.ch_frac = _parse_channel_fraction_max(channel_frac)
        self.min_fac = min_factor
        self.max_fac = max_factor

    def __call__(self, x, y, severity, rng, cont_channels, _disc_channels):
        if severity == 0: return x, y, []
        max_factor = lin_interp(severity, self.min_fac, self.max_fac)
        channels = select_channels(
            cont_channels,
            self.ch_frac,
            severity,
            rng,
            max_fraction_validated=True,
            severity_validated=True,
        )
        if not channels: return x, y, []
        x = x.clone()
        seq_len = x.size(0)
        time_steps = torch.arange(seq_len, device=x.device, dtype=x.dtype)
        denom = float(max(seq_len - 1, 1))
        scale = 1.0 + (max_factor - 1.0) * (time_steps / denom)
        x[:, channels] = x[:, channels] * scale.unsqueeze(-1)
        return x, y, channels


class Noise:
    """Add white Gaussian noise to continuous channels."""
    name = "noise"
    channel_scope = "continuous"

    def __init__(self, channel_frac: float = 0.1, *, min_sd=0.0, max_sd=1.0):
        self.ch_frac = _parse_channel_fraction_max(channel_frac)
        self.min_sd = min_sd
        self.max_sd = max_sd

    def __call__(self, x, y, severity, rng, cont_channels, _disc_channels):
        if severity == 0: return x, y, []
        sd = lin_interp(severity, self.min_sd, self.max_sd)
        channels = select_channels(
            cont_channels,
            self.ch_frac,
            severity,
            rng,
            max_fraction_validated=True,
            severity_validated=True,
        )
        if not channels: return x, y, []
        noise = torch.zeros_like(x)
        noise[:, channels] = torch.randn((x.size(0), len(channels)), generator=rng, device=x.device) * sd
        return x + noise, y, channels


# F2F: outliers failure mode (gamma(t) delta(t)).
class Outliers:
    """Add occasional random deviations at random time steps."""
    name = "outliers"
    channel_scope = "continuous"

    def __init__(
        self,
        channel_frac: float = 0.1,
        *,
        min_prob=0.0,
        max_prob=0.05,
        min_range=0.0,
        max_range=3.0,
    ):
        self.ch_frac = _parse_channel_fraction_max(channel_frac)
        self.min_prob = min_prob
        self.max_prob = max_prob
        self.min_range = min_range
        self.max_range = max_range

    def __call__(self, x, y, severity, rng, cont_channels, _disc_channels):
        if severity == 0: return x, y, []
        outlier_prob = lin_interp(severity, self.min_prob, self.max_prob)
        outlier_range = lin_interp(severity, self.min_range, self.max_range)
        channels = select_channels(
            cont_channels,
            self.ch_frac,
            severity,
            rng,
            max_fraction_validated=True,
            severity_validated=True,
        )
        if not channels: return x, y, []
        x = x.clone()
        seq_len = x.size(0)
        mask = torch.rand((seq_len, len(channels)), generator=rng, device=x.device) < outlier_prob
        deviations = (
            torch.rand((seq_len, len(channels)), generator=rng, device=x.device, dtype=x.dtype) * 2.0 - 1.0
        )
        deviations = deviations * outlier_range
        x[:, channels] = x[:, channels] + mask.to(x.dtype) * deviations
        hit = mask.any(dim=0)
        actually_affected = [ch for ch, h in zip(channels, hit) if h]
        return x, y, actually_affected


class Spike:
    """Add a high value to a time step of continuous channels."""
    name = "spike"
    channel_scope = "continuous"
    
    def __init__(self, channel_frac: float = 0.1, *, min_hickup=0.0, max_hickup=7.5):
        self.ch_frac = _parse_channel_fraction_max(channel_frac)
        self.min = min_hickup
        self.max = max_hickup

    def __call__(self, x, y, severity, rng, cont_channels, _disc_channels):
        if severity == 0: return x, y, []
        hickup = lin_interp(severity, self.min, self.max)
        channels = select_channels(
            cont_channels,
            self.ch_frac,
            severity,
            rng,
            max_fraction_validated=True,
            severity_validated=True,
        )
        if not channels: return x, y, []
        x = x.clone()
        for ch in channels:
            pos = randint_py(1, x.size(0), rng)
            x[pos, ch] += hickup
        return x, y, channels


class StuckSensor:
    """Hold continuous channels flat for a contiguous window."""
    name = "stuck_sensor"
    channel_scope = "continuous"

    def __init__(self, channel_frac: float = 0.1, *, max_duration_frac=1.0):
        self.ch_frac = _parse_channel_fraction_max(channel_frac)
        self.max_frac = max_duration_frac

    def __call__(self, x, y, severity, rng, cont_channels, _disc_channels):
        if severity == 0: return x, y, []
        max_duration = self.max_frac * (x.size(0) - 1)
        duration_float = lin_interp(severity, 0.0, max_duration)
        duration = int(np.ceil(duration_float))

        channels = select_channels(
            cont_channels,
            self.ch_frac,
            severity,
            rng,
            max_fraction_validated=True,
            severity_validated=True,
        )
        if not channels: return x, y, []
        x = x.clone()
        for ch in channels:
            start = randint_py(1, x.size(0) - duration + 1, rng)
            x[start:start + duration, ch] = x[start - 1, ch]
        return x, y, channels


class PacketLoss:
    """Forward-fill one or more bursty channel-local outages on continuous channels."""
    name = "packet_loss"
    channel_scope = "continuous"

    def __init__(
        self,
        channel_frac: float = 0.1,
        *,
        max_start_prob=0.25,
        max_continue_prob=0.9,
    ):
        self.ch_frac = _parse_channel_fraction_max(channel_frac)
        self.max_start_prob = max_start_prob
        self.max_continue_prob = max_continue_prob

    def __call__(self, x, y, severity, rng, cont_channels, _disc_channels):
        if severity == 0 or x.size(0) <= 1:
            return x, y, []
        start_prob = lin_interp(severity, 0.0, self.max_start_prob)
        continue_prob = lin_interp(severity, 0.0, self.max_continue_prob)
        selected_channels = select_channels(
            cont_channels,
            self.ch_frac,
            severity,
            rng,
            max_fraction_validated=True,
            severity_validated=True,
        )
        if not selected_channels:
            return x, y, []

        x_pert = x.clone()
        for ch in selected_channels:
            # Guarantee at least one burst per selected channel by anchoring a
            # burst start at one position.
            guaranteed_burst_start = randint_py(1, x.size(0), rng)
            in_burst = False
            for pos in range(1, x.size(0)):
                if pos == guaranteed_burst_start:
                    in_burst = True
                elif in_burst:
                    in_burst = (
                        float(torch.rand((), generator=rng, device=x.device).item())
                        < continue_prob
                    )
                else:
                    in_burst = (
                        float(torch.rand((), generator=rng, device=x.device).item())
                        < start_prob
                    )
                if in_burst:
                    x_pert[pos, ch] = x_pert[pos - 1, ch]
        return x_pert, y, selected_channels


class TimeStretch:
    """Stretch the signal by simulating a sensor with higher sampling rate wrongly merged."""
    name = "time_stretch"
    channel_scope = "continuous"

    def __init__(self, channel_frac: float = 0.1, *, min_rate=1.0, max_rate=5.0):
        self.ch_frac = _parse_channel_fraction_max(channel_frac)
        self.min_rate = min_rate
        self.max_rate = max_rate

    def __call__(self, x, y, severity, rng, cont_channels, _disc_channels):
        if severity == 0: return x, y, []
        rate = lin_interp(severity, self.min_rate, self.max_rate)
        # Higher rate -> more samples -> signal stretches -> we move slower through original
        warp_factor = 1.0 / rate
        channels = select_channels(
            cont_channels,
            self.ch_frac,
            severity,
            rng,
            max_fraction_validated=True,
            severity_validated=True,
        )
        if not channels: return x, y, []

        x_pert = x.clone()
        d_in = x.size(0)
        warp_duration = int(np.ceil(d_in / 2))
        warp_start_pos = randint_py(1, d_in - warp_duration + 1, rng)

        original_time_index = np.arange(d_in, dtype=float)
        irreg_time = np.full(warp_duration, warp_factor, dtype=float)
        irreg_time_index = np.cumsum(irreg_time) + warp_start_pos - 1

        window_end = warp_start_pos + warp_duration
        for ch in channels:
            channel_values = x[:, ch].detach().cpu().numpy()
            interp_values_np = np.interp(irreg_time_index, original_time_index, channel_values)
            interp_values = torch.from_numpy(interp_values_np).to(device=x.device, dtype=x.dtype)
            x_pert[warp_start_pos:window_end, ch] = interp_values

            last_sample_time = float(irreg_time_index[-1])
            hold_end = min(d_in, max(window_end, int(np.floor(last_sample_time)) + 1))
            if hold_end > window_end:
                x_pert[window_end:hold_end, ch] = interp_values[-1]

        return x_pert, y, channels


class TimeCompress:
    """Compress the signal by simulating a sensor with lower sampling rate wrongly merged."""
    name = "time_compress"
    channel_scope = "continuous"

    def __init__(self, channel_frac: float = 0.1, *, min_rate=1.0, max_rate=0.1):
        self.ch_frac = _parse_channel_fraction_max(channel_frac)
        self.min_rate = min_rate
        self.max_rate = max_rate

    def __call__(self, x, y, severity, rng, cont_channels, _disc_channels):
        if severity == 0: return x, y, []
        rate = lin_interp(severity, self.min_rate, self.max_rate)
        # Lower rate -> fewer samples -> signal compresses -> we move faster through original
        warp_factor = 1.0 / rate
        channels = select_channels(
            cont_channels,
            self.ch_frac,
            severity,
            rng,
            max_fraction_validated=True,
            severity_validated=True,
        )
        if not channels: return x, y, []

        x_pert = x.clone()
        d_in = x.size(0)
        warp_duration = int(np.ceil(d_in / 2))
        warp_start_pos = randint_py(1, d_in - warp_duration + 1, rng)

        original_time_index = np.arange(d_in, dtype=float)
        irreg_time = np.full(warp_duration, warp_factor, dtype=float)
        irreg_time_index = np.cumsum(irreg_time) + warp_start_pos - 1

        window_end = warp_start_pos + warp_duration
        for ch in channels:
            channel_values = x[:, ch].detach().cpu().numpy()
            interp_values_np = np.interp(irreg_time_index, original_time_index, channel_values)
            interp_values = torch.from_numpy(interp_values_np).to(device=x.device, dtype=x.dtype)
            x_pert[warp_start_pos:window_end, ch] = interp_values

        return x_pert, y, channels


# F2F: trimming failure mode with hard clamping (alpha=0).
class TrimmingConstant:
    """Hard clamp values to mean +/- k*std bounds per channel."""
    name = "trimming_constant"
    channel_scope = "continuous"

    def __init__(self, channel_frac: float = 0.1, *, min_std_mult=3.0, max_std_mult=1.0):
        self.ch_frac = _parse_channel_fraction_max(channel_frac)
        self.min_std_mult = min_std_mult
        self.max_std_mult = max_std_mult

    def __call__(self, x, y, severity, rng, cont_channels, _disc_channels):
        if severity == 0: return x, y, []
        std_mult = lin_interp(severity, self.min_std_mult, self.max_std_mult)
        channels = select_channels(
            cont_channels,
            self.ch_frac,
            severity,
            rng,
            max_fraction_validated=True,
            severity_validated=True,
        )
        if not channels: return x, y, []
        x = x.clone()
        lower = -std_mult
        upper = std_mult
        values = x[:, channels]
        out_of_bounds = (values < lower) | (values > upper)
        x[:, channels] = torch.clamp(values, lower, upper)
        hit = out_of_bounds.any(dim=0)
        actually_affected = [ch for ch, h in zip(channels, hit) if h]
        return x, y, actually_affected


# F2F: trimming failure mode with dampening (0 < alpha < 1).
class TrimmingVarying:
    """Dampen out-of-bound values with a factor alpha."""
    name = "trimming_varying"
    channel_scope = "continuous"

    def __init__(
        self,
        channel_frac: float = 0.1,
        *,
        min_std_mult=3.0,
        max_std_mult=1.0,
        min_damping=1.0,
        max_damping=0.6,
    ):
        self.ch_frac = _parse_channel_fraction_max(channel_frac)
        self.min_std_mult = min_std_mult
        self.max_std_mult = max_std_mult
        self.min_damping = min_damping
        self.max_damping = max_damping

    def __call__(self, x, y, severity, rng, cont_channels, _disc_channels):
        if severity == 0: return x, y, []
        std_mult = lin_interp(severity, self.min_std_mult, self.max_std_mult)
        damping = lin_interp(severity, self.min_damping, self.max_damping)
        channels = select_channels(
            cont_channels,
            self.ch_frac,
            severity,
            rng,
            max_fraction_validated=True,
            severity_validated=True,
        )
        if not channels: return x, y, []
        x = x.clone()
        lower = -std_mult
        upper = std_mult
        values = x[:, channels]
        below = values < lower
        above = values > upper
        out_of_bounds = below | above
        dampened_below = lower + (values - lower) * damping
        dampened_above = upper + (values - upper) * damping
        x[:, channels] = torch.where(
            below,
            dampened_below,
            torch.where(above, dampened_above, values),
        )
        hit = out_of_bounds.any(dim=0)
        actually_affected = [ch for ch, h in zip(channels, hit) if h]
        return x, y, actually_affected


##### discrete channels #####

class WrongState:
    """A discrete channel shows a wrong value."""
    name = "wrong_state"
    channel_scope = "discrete"

    def __init__(self, channel_frac: float = 0.1, *, max_duration_frac=0.5, delta=5):
        self.ch_frac = _parse_channel_fraction_max(channel_frac)
        self.max_duration_frac = max_duration_frac
        self.delta = delta

    def __call__(self, x, y, severity, rng, _cont_channels, disc_channels):
        if severity == 0 or not disc_channels: return x, y, []
        max_duration = self.max_duration_frac * (x.size(0) - 1)
        duration_float = lin_interp(severity, 0.0, max_duration)
        duration = int(np.ceil(duration_float))

        channels = select_channels(
            disc_channels,
            self.ch_frac,
            severity,
            rng,
            max_fraction_validated=True,
            severity_validated=True,
        )
        if not channels: return x, y, []
        x = x.clone()
        for ch in channels:
            start = randint_py(1, x.size(0) - duration + 1, rng)
            new_val = x[:, ch].max().item() + self.delta
            x[start:start + duration, ch] = new_val
        return x, y, channels


class Chattering:
    """A discrete channel oscillates between two values."""
    name = "chattering"
    channel_scope = "discrete"

    def __init__(self, channel_frac: float = 0.1, *, max_duration_frac=0.5):
        self.ch_frac = _parse_channel_fraction_max(channel_frac)
        self.max_frac = max_duration_frac

    def __call__(self, x, y, severity, rng, _cont_channels, disc_channels):
        if severity == 0 or not disc_channels: return x, y, []
        max_duration = self.max_frac * (x.size(0) - 1)
        duration_float = lin_interp(severity, 0.0, max_duration)
        duration = int(np.ceil(duration_float))

        channels = select_channels(
            disc_channels,
            self.ch_frac,
            severity,
            rng,
            max_fraction_validated=True,
            severity_validated=True,
        )
        if not channels: return x, y, []
        x = x.clone()
        for ch in channels:
            start = randint_py(1, x.size(0) - duration + 1, rng)
            last = x[start - 1, ch].item()
            ch_min = x[:, ch].min().item()
            ch_max = x[:, ch].max().item()
            if last != ch_max:
                other_val = ch_max
            elif last != ch_min:
                other_val = ch_min
            else:  # constant channel
                other_val = last + 1
            pattern = torch.randint(0, 2, (duration,), generator=rng, device=x.device, dtype=x.dtype)
            new_vals = pattern * other_val + (1 - pattern) * last
            x[start:start + duration, ch] = new_vals
        return x, y, channels


##### all channels #####

class MissingData:
    """Remove a contiguous window inside the input and forward-fill. The target is untouched."""
    name = "missing_data"
    channel_scope = "all"

    def __init__(self, *, max_gap_frac: float = 0.5):
        self.max_frac = max_gap_frac

    def __call__(self, x, y, severity, rng, cont_channels, disc_channels):
        if severity == 0: return x, y, []

        input_len = x.size(0)
        max_gap = self.max_frac * (input_len - 1)
        gap_len_float = lin_interp(severity, 0.0, max_gap)
        gap_len = int(np.ceil(gap_len_float))
        gap_start = randint_py(1, input_len - gap_len + 1, rng)

        x_shifted = x.clone()
        # Forward-fill the gap
        fill_values = x_shifted[gap_start - 1, :]
        x_shifted[gap_start : gap_start + gap_len, :] = fill_values

        return x_shifted, y, list(range(x.shape[1]))


DEFAULT_PERTURBATIONS = (
    Drift,
    Attenuation,
    Noise,
    StuckSensor,
    MissingData,
    Spike,
    TimeStretch,
    TimeCompress,
    WrongState,
    Chattering,
)

EXTENDED_PERTURBATIONS = (
    LinearDrift,
    NonlinearDrift,
    Scaling,
    Outliers,
    TrimmingConstant,
    TrimmingVarying,
    TimeVaryingScaling,
    PacketLoss,
)

ALL_PERTURBATIONS = DEFAULT_PERTURBATIONS + EXTENDED_PERTURBATIONS

PERTURBATION_REGISTRY = {
    perturbation.name: perturbation
    for perturbation in ALL_PERTURBATIONS
}
for idx, perturbation in enumerate(PERTURBATION_REGISTRY.values()):
    perturbation.idx = idx
