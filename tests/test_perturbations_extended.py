import torch
import pytest

import data.perturbations as perturbations
from data.perturbations import (
    ALL_PERTURBATIONS,
    DEFAULT_PERTURBATIONS,
    Attenuation,
    Drift,
    LinearDrift,
    NonlinearDrift,
    Outliers,
    PacketLoss,
    Spike,
    WrongState,
    Scaling,
    TimeVaryingScaling,
    TrimmingConstant,
    TrimmingVarying,
    build_perturbation_scenario_params_signature,
    select_channels,
)


_EXTENDED_NAMES = {
    "linear_drift",
    "nonlinear_drift",
    "scaling",
    "outliers",
    "packet_loss",
    "trimming_constant",
    "trimming_varying",
    "time_varying_scaling",
}


@pytest.mark.parametrize(
    "pert_cls",
    [
        LinearDrift,
        NonlinearDrift,
        Outliers,
        Scaling,
        TimeVaryingScaling,
        TrimmingConstant,
        TrimmingVarying,
    ],
)
def test_extended_perturbations_zero_severity_no_change(pert_cls):
    rng = torch.Generator().manual_seed(7)
    x = torch.randn(8, 4, generator=rng)
    y = torch.randn(2, 1, generator=rng)
    cont_channels = [0, 1, 2, 3]
    disc_channels = []

    pert = pert_cls(channel_frac=1.0)
    x_pert, y_pert, channels = pert(
        x, y, 0.0, torch.Generator().manual_seed(9), cont_channels, disc_channels
    )

    assert torch.equal(x_pert, x)
    assert torch.equal(y_pert, y)
    assert channels == []


@pytest.mark.parametrize(
    "pert_cls",
    [
        LinearDrift,
        NonlinearDrift,
        Outliers,
        Scaling,
        TimeVaryingScaling,
        TrimmingConstant,
        TrimmingVarying,
    ],
)
def test_extended_perturbations_channel_fraction(pert_cls):
    rng = torch.Generator().manual_seed(11)
    x = torch.randn(6, 4, generator=rng)
    y = torch.zeros(1, 1)
    cont_channels = [0, 1, 2, 3]
    disc_channels = []

    pert = pert_cls(channel_frac=0.5)
    x_pert, _, channels = pert(
        x, y, 1.0, torch.Generator().manual_seed(13), cont_channels, disc_channels
    )

    assert len(channels) == 2
    untouched = [idx for idx in cont_channels if idx not in channels]
    if untouched:
        assert torch.equal(x_pert[:, untouched], x[:, untouched])


def test_linear_drift_values():
    x = torch.arange(5, dtype=torch.float32).unsqueeze(1)
    y = torch.zeros(1, 1)
    cont_channels = [0]

    pert = LinearDrift(channel_frac=1.0, min_drift=0.0, max_drift=1.0)
    x_pert, _, _ = pert(
        x, y, 1.0, torch.Generator().manual_seed(1), cont_channels, []
    )

    time_steps = torch.arange(x.size(0), dtype=x.dtype)
    drift = (1.0 / (x.size(0) - 1)) * time_steps
    expected = x + drift.unsqueeze(-1)

    assert torch.allclose(x_pert, expected)


def test_nonlinear_drift_values():
    x = torch.arange(5, dtype=torch.float32).unsqueeze(1)
    y = torch.zeros(1, 1)
    cont_channels = [0]

    pert = NonlinearDrift(
        channel_frac=1.0,
        min_linear=0.0,
        max_linear=1.0,
        min_quadratic=0.0,
        max_quadratic=0.5,
    )
    x_pert, _, _ = pert(
        x, y, 1.0, torch.Generator().manual_seed(2), cont_channels, []
    )

    time_steps = torch.arange(x.size(0), dtype=x.dtype)
    linear_coeff = 1.0 / (x.size(0) - 1)
    quad_coeff = 0.5 / ((x.size(0) - 1) ** 2)
    drift = linear_coeff * time_steps + quad_coeff * (time_steps ** 2)
    expected = x + drift.unsqueeze(-1)

    assert torch.allclose(x_pert, expected)


def test_default_endpoint_severities_match_core_fault_contract():
    rng = torch.Generator().manual_seed(20)
    y = torch.zeros(1, 1)
    cont_channels = [0]

    x_drift = torch.zeros(4, 1)
    drift = Drift(channel_frac=1.0)
    x_drift_pert, _, _ = drift(
        x_drift, y, 1.0, torch.Generator().manual_seed(21), cont_channels, []
    )
    assert torch.allclose(x_drift_pert, torch.full_like(x_drift, 0.75))

    x_attenuation = torch.ones(4, 1) * 4.0
    attenuation = Attenuation(channel_frac=1.0)
    x_attenuation_pert, _, _ = attenuation(
        x_attenuation, y, 1.0, torch.Generator().manual_seed(22), cont_channels, []
    )
    assert torch.allclose(x_attenuation_pert, torch.ones_like(x_attenuation))

    x_spike = torch.zeros(4, 1)
    spike = Spike(channel_frac=1.0)
    x_spike_pert, _, _ = spike(
        x_spike, y, 1.0, rng, cont_channels, []
    )
    assert x_spike_pert.max().item() == pytest.approx(7.5)
    assert torch.count_nonzero(x_spike_pert).item() == 1


def test_perturbation_scenario_params_signature_tracks_default_endpoints():
    baseline_signature = build_perturbation_scenario_params_signature(
        ("drift", "attenuation", "spike")
    )

    original_max_offset = Drift.__init__.__kwdefaults__["max_offset"]
    Drift.__init__.__kwdefaults__["max_offset"] = original_max_offset + 0.1
    try:
        changed_signature = build_perturbation_scenario_params_signature(
            ("drift", "attenuation", "spike")
        )
    finally:
        Drift.__init__.__kwdefaults__["max_offset"] = original_max_offset

    assert changed_signature != baseline_signature


def test_scaling_values():
    x = torch.arange(1, 6, dtype=torch.float32).unsqueeze(1)
    y = torch.zeros(1, 1)
    cont_channels = [0]

    pert = Scaling(channel_frac=1.0, min_factor=1.0, max_factor=2.0)
    x_pert, _, _ = pert(
        x, y, 1.0, torch.Generator().manual_seed(3), cont_channels, []
    )

    assert torch.allclose(x_pert, x * 2.0)


def test_time_varying_scaling_values():
    x = torch.ones(5, 1)
    y = torch.zeros(1, 1)
    cont_channels = [0]

    pert = TimeVaryingScaling(channel_frac=1.0, min_factor=1.0, max_factor=2.0)
    x_pert, _, _ = pert(
        x, y, 1.0, torch.Generator().manual_seed(4), cont_channels, []
    )

    time_steps = torch.arange(x.size(0), dtype=x.dtype)
    scale = 1.0 + (2.0 - 1.0) * (time_steps / (x.size(0) - 1))
    expected = x * scale.unsqueeze(-1)

    assert torch.allclose(x_pert, expected)


def test_trimming_constant_bounds():
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]).unsqueeze(1)
    y = torch.zeros(1, 1)
    cont_channels = [0]

    pert = TrimmingConstant(channel_frac=1.0, min_std_mult=1.0, max_std_mult=1.0)
    x_pert, _, _ = pert(
        x, y, 1.0, torch.Generator().manual_seed(5), cont_channels, []
    )

    expected = torch.clamp(x, -1.0, 1.0)

    assert torch.allclose(x_pert, expected)


def test_trimming_varying_bounds():
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]).unsqueeze(1)
    y = torch.zeros(1, 1)
    cont_channels = [0]

    pert = TrimmingVarying(
        channel_frac=1.0,
        min_std_mult=1.0,
        max_std_mult=1.0,
        min_damping=0.6,
        max_damping=0.6,
    )
    x_pert, _, _ = pert(
        x, y, 1.0, torch.Generator().manual_seed(6), cont_channels, []
    )

    values = x[:, 0]
    lower = -1.0
    upper = 1.0
    damping = 0.6
    below = values < lower
    above = values > upper
    dampened_below = lower + (values - lower) * damping
    dampened_above = upper + (values - upper) * damping
    expected = torch.where(
        below,
        dampened_below,
        torch.where(above, dampened_above, values),
    ).unsqueeze(-1)

    assert torch.allclose(x_pert, expected)


def test_packet_loss_forward_fills_bursty_outages():
    x = torch.arange(4, dtype=torch.float32).unsqueeze(1)
    y = torch.zeros(1, 1)
    cont_channels = [0]

    pert = PacketLoss(
        channel_frac=1.0,
        max_start_prob=1.0,
        max_continue_prob=1.0,
    )
    x_pert, _, channels = pert(
        x, y, 1.0, torch.Generator().manual_seed(8), cont_channels, []
    )

    assert channels == [0]
    changed = torch.nonzero(x_pert[:, 0] != x[:, 0], as_tuple=False).flatten()
    assert len(changed) >= 1
    first_changed = int(changed[0].item())
    assert first_changed >= 1
    assert torch.all(x_pert[first_changed:, 0] == x_pert[first_changed - 1, 0])


def test_packet_loss_guarantees_one_burst_per_selected_channel():
    x = torch.arange(6, dtype=torch.float32).unsqueeze(1)
    y = torch.zeros(1, 1)
    cont_channels = [0]

    pert = PacketLoss(
        channel_frac=1.0,
        max_start_prob=0.0,
        max_continue_prob=0.0,
    )
    x_pert, _, channels = pert(
        x, y, 1.0, torch.Generator().manual_seed(10), cont_channels, []
    )

    changed = torch.nonzero(x_pert[:, 0] != x[:, 0], as_tuple=False).flatten()
    assert channels == [0]
    assert len(changed) == 1
    changed_pos = int(changed[0].item())
    assert changed_pos >= 1
    assert x_pert[changed_pos, 0] == x_pert[changed_pos - 1, 0]


def test_packet_loss_respects_channel_fraction_and_leaves_other_channels_clean():
    x = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    y = torch.zeros(1, 1)
    cont_channels = [0, 1, 2, 3]

    pert = PacketLoss(
        channel_frac=0.5,
        max_start_prob=1.0,
        max_continue_prob=1.0,
    )
    x_pert, _, channels = pert(
        x, y, 1.0, torch.Generator().manual_seed(9), cont_channels, []
    )

    assert len(channels) == 2
    untouched = [idx for idx in cont_channels if idx not in channels]
    if untouched:
        assert torch.equal(x_pert[:, untouched], x[:, untouched])


def test_outliers_bounds_with_stats():
    x = torch.zeros(6, 1)
    y = torch.zeros(1, 1)
    cont_channels = [0]

    pert = Outliers(channel_frac=1.0, min_prob=1.0, max_prob=1.0, min_range=1.0, max_range=1.0)
    x_pert, _, _ = pert(
        x,
        y,
        1.0,
        torch.Generator().manual_seed(7),
        cont_channels,
        [],
    )

    assert torch.all(x_pert.abs() <= 1.0 + 1e-6)


def test_default_perturbations_exclude_extended():
    default_names = {pert.name for pert in DEFAULT_PERTURBATIONS}
    extended_names = {pert.name for pert in perturbations.EXTENDED_PERTURBATIONS}
    assert "drift" in default_names
    assert "noise" in default_names
    assert "packet_loss" not in default_names
    assert "packet_loss" in extended_names
    assert "time_compress" in default_names
    assert not default_names.intersection(_EXTENDED_NAMES)


def test_severity_coupled_selection_monotonic_counts():
    channel_pool = list(range(10))
    max_fraction = 0.5
    severities = [0.0, 0.01, 0.25, 0.5, 0.75, 1.0]
    expected_counts = [0, 1, 2, 3, 4, 5]

    observed = []
    for idx, severity in enumerate(severities):
        selected = select_channels(
            channel_pool,
            max_fraction,
            severity,
            torch.Generator().manual_seed(101 + idx),
        )
        observed.append(len(selected))

    assert observed == expected_counts


def test_severity_coupled_selection_low_nonzero_hits_one_channel():
    selected = select_channels(
        list(range(8)),
        0.5,
        1e-9,
        torch.Generator().manual_seed(321),
    )
    assert len(selected) == 1


def test_select_channels_prevalidated_fraction_matches_default_path():
    channel_pool = list(range(10))
    severity = 0.75
    seed = 123

    default_selected = select_channels(
        channel_pool,
        0.5,
        severity,
        torch.Generator().manual_seed(seed),
    )
    prevalidated_selected = select_channels(
        channel_pool,
        0.5,
        severity,
        torch.Generator().manual_seed(seed),
        max_fraction_validated=True,
    )

    assert prevalidated_selected == default_selected


def test_select_channels_prevalidated_fraction_skips_fraction_parser(monkeypatch):
    def _fail_parse(*_args, **_kwargs):
        raise AssertionError("fraction parser should not be called for pre-validated max_fraction")

    monkeypatch.setattr(perturbations, "_parse_channel_fraction_max", _fail_parse)
    selected = select_channels(
        list(range(6)),
        0.5,
        1.0,
        torch.Generator().manual_seed(88),
        max_fraction_validated=True,
    )
    assert len(selected) == 3


def test_select_channels_prevalidated_severity_skips_severity_parser(monkeypatch):
    def _fail_parse(*_args, **_kwargs):
        raise AssertionError("severity parser should not be called for pre-validated severity")

    monkeypatch.setattr(perturbations, "_parse_severity", _fail_parse)
    selected = select_channels(
        list(range(6)),
        0.5,
        1.0,
        torch.Generator().manual_seed(89),
        max_fraction_validated=True,
        severity_validated=True,
    )
    assert len(selected) == 3


@pytest.mark.parametrize("invalid_fraction", [0.0, -0.1, 1.2])
def test_select_channels_prevalidated_fraction_rejects_invalid_values(invalid_fraction):
    with pytest.raises(ValueError, match="max_fraction must satisfy 0 < max_fraction <= 1"):
        select_channels(
            list(range(5)),
            invalid_fraction,
            0.5,
            torch.Generator().manual_seed(90),
            max_fraction_validated=True,
        )


@pytest.mark.parametrize("invalid_severity", [-0.1, 1.1])
def test_select_channels_prevalidated_severity_rejects_out_of_range_values(invalid_severity):
    with pytest.raises(ValueError, match="severity must satisfy 0 <= severity <= 1"):
        select_channels(
            list(range(5)),
            0.5,
            invalid_severity,
            torch.Generator().manual_seed(92),
            max_fraction_validated=True,
            severity_validated=True,
        )


def test_select_channels_prevalidated_fraction_rejects_non_numeric_value():
    with pytest.raises(ValueError, match="max_fraction must be numeric"):
        select_channels(
            list(range(5)),
            "bad",
            0.5,
            torch.Generator().manual_seed(91),
            max_fraction_validated=True,
        )


def test_select_channels_prevalidated_severity_rejects_non_numeric_value():
    with pytest.raises(ValueError, match="severity must be numeric"):
        select_channels(
            list(range(5)),
            0.5,
            "bad",
            torch.Generator().manual_seed(93),
            max_fraction_validated=True,
            severity_validated=True,
        )


def test_discrete_channel_count_coupling_matches_rule():
    x = torch.zeros(12, 8)
    y = torch.zeros(1, 1)
    disc_channels = list(range(8))
    perturbation = WrongState(channel_frac=0.5, max_duration_frac=0.5, delta=3)

    _, _, low_channels = perturbation(
        x,
        y,
        0.01,
        torch.Generator().manual_seed(41),
        [],
        disc_channels,
    )
    _, _, high_channels = perturbation(
        x,
        y,
        1.0,
        torch.Generator().manual_seed(42),
        [],
        disc_channels,
    )

    assert len(low_channels) == 1
    assert len(high_channels) == 4


def test_all_perturbations_define_channel_scope():
    allowed_scopes = {"continuous", "discrete", "all"}
    scopes = {getattr(pert_cls, "channel_scope", None) for pert_cls in ALL_PERTURBATIONS}
    assert scopes.issubset(allowed_scopes)
    assert None not in scopes
