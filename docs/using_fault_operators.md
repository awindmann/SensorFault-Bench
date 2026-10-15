# Using the Fault Operators

The fault operators in `data/perturbations.py` also serve purposes outside the
benchmark. They can supply faulty input histories for pretraining or
fine-tuning, augment training data, or stress-test a model with faults
calibrated to a specific system. This guide describes the operators and their
calibration before it turns to evaluating, reporting, and citing work that uses
them.

## Operator Groups

`PERTURBATION_REGISTRY` in `data/perturbations.py` maps each operator key to
its class. Each group of operators has a fixed role within the benchmark,
whereas outside the benchmark every operator can serve any purpose.

<!-- operator-groups:start -->
| Group | Operators | Role in the benchmark |
| --- | --- | --- |
| Scored scenarios | `drift`, `attenuation`, `noise`, `spike`, `time_stretch`, `time_compress`, `stuck_sensor`, `missing_data` | Evaluation only |
| Transfer fault families | `linear_drift`, `nonlinear_drift`, `scaling`, `time_varying_scaling`, `trimming_constant`, `trimming_varying`, `packet_loss` | Training faults of fault augmentation |
| Sensitivity scenarios | `negative_drift`, `negative_spike`, `local_timing_warp` | Alternative scenarios of the sensitivity analyses |
| Discrete-state operators | `wrong_state`, `chattering` | Datasets with discrete channels, outside the scored set |
| Further operators | `outliers` | Not used |
<!-- operator-groups:end -->

The membership of the scored scenarios is defined by `PERTURBATION_SCENARIOS`
in [`configs/defaults.yaml`](../configs/defaults.yaml), that of the transfer
fault families by the `holdout` profile in
[`configs/pipelines/fault_augmentation.yaml`](../configs/pipelines/fault_augmentation.yaml),
and that of the sensitivity scenarios by
[`sensitivity_analyses.md`](sensitivity_analyses.md).

The scored scenarios and the transfer fault families have separate origins.
The scored scenarios are adapted from the disturbances of the CPS forecasting
benchmark of
[Windmann et al. (2025)](https://doi.org/10.1109/ETFA65518.2025.11205527).
Appendix B of the paper states this origin and links each scenario to
documented sensor faults.
The transfer fault families consist of six failure modes adapted from the
sensor failure taxonomy of Brandt et al. (2025) and of `packet_loss`, which
SensorFault-Bench adds.

## Interface

Each operator is a class whose constructor sets the fault endpoints and whose
call corrupts one input window. All operators share the interface that the
[benchmark protocol](benchmark_protocol.md#scenarios) defines for the scored
scenarios. A call takes the window as a tensor indexed by time step and
channel, the target, a severity in $[0, 1]$, a `torch.Generator` that supplies
all randomness, and the indices of the continuous and discrete channels. It
returns the corrupted window, the unchanged target, and the affected channels.

The operators expect inputs standardized with training statistics. Value
magnitudes such as offsets and spike heights are therefore in standard
deviations of the affected channel.

## Calibrating to Observed Faults

Four settings adapt the operators to the faults of a specific system.

- **Endpoints.** The keyword arguments of each constructor set the fault
  quantity at severity 0 and 1, for example `min_offset` and `max_offset` of
  `drift` or `max_duration_frac` of `stuck_sensor`. Observed magnitudes enter
  in standard deviations of the affected channel's training data.
- **Affected channels.** `channel_frac` caps the fraction of eligible channels
  that one call affects. The constructors default to 0.1, while the benchmark
  passes 0.5 from `PERTURBATION_CHANNEL_FRACTION_MAX`.
- **Channel count.** By default, the number of affected channels grows with
  severity up to the cap. A call with `channel_count_mode="fixed_fraction"`
  instead affects the fraction of channels given by `channel_count_value`, at
  most the cap.
- **Severity distribution.** The caller passes the severity, so it can follow
  any distribution, for example one fitted to how often faults of each
  magnitude occur. Besides the uniform distribution of the benchmark,
  `data/samplers.py` provides `beta_1_2_severity`, which draws from Beta(1, 2)
  and favors mild faults.

A calibrated variant is specific to its system and lies outside the benchmark's
default configuration. Robustness to the calibrated faults therefore describes
that system and is not comparable with the default configuration.

The snippet below calibrates two operators and builds pairs of clean and
corrupted windows, with severities that favor mild faults.

<!-- calibrated-pairs:start -->
```python
import torch

from data.perturbations import PERTURBATION_REGISTRY

operators = {
    "drift": PERTURBATION_REGISTRY["drift"](channel_frac=0.25, max_offset=1.5),
    "stuck_sensor": PERTURBATION_REGISTRY["stuck_sensor"](
        channel_frac=0.25, max_duration_frac=0.3,
    ),
}


def sample_severity(rng):
    """Favor mild faults with S = U^2 for U ~ U(0, 1)."""
    return torch.rand((), generator=rng).item() ** 2


generator = torch.Generator().manual_seed(0)
windows = torch.randn(8, 96, 7, generator=generator)
targets = torch.randn(8, 24, 7, generator=generator)
continuous_channels = list(range(windows.shape[-1]))

pairs = []
for clean, target in zip(windows, targets):
    for name, operator in operators.items():
        corrupted, _, channels = operator(
            clean, target, sample_severity(generator), generator, continuous_channels, [],
        )
        pairs.append(
            {"operator": name, "clean": clean, "corrupted": corrupted, "channels": channels}
        )
```
<!-- calibrated-pairs:end -->

## Pretraining and Augmentation

Since the caller keeps the clean window, each call yields a pair of clean and
corrupted histories together with the affected channels. Such pairs supervise
objectives such as reconstructing the clean history from the corrupted one,
detecting the affected channels, or keeping representations consistent between
both inputs. The
[paired transfer inputs](extending_methods.md#paired-transfer-inputs) build
such pairs from the transfer fault families.

The benchmark evaluates one training use of the operators, fault augmentation
on the transfer fault families. It does not measure whether pretraining with
the operators improves robustness.

A pretraining corpus can come from any data source. When it includes the
benchmark datasets, their terms apply in addition to the code license, as
listed in [`data/README.md`](../data/README.md#sources-and-terms).

## Evaluating After Fault Training

A model trained or pretrained with the operators can be evaluated like any
other model, as described in [Evaluating a new model or
method](../README.md#evaluating-a-new-model-or-method).
Its report should name the operators that the model saw during training,
together with their parameters. If these include scored scenarios, the results
measure robustness to faults seen in training rather than transfer to unseen
faults.
The protocol section
[Fault Exposure in Training](benchmark_protocol.md#fault-exposure-in-training)
states this rule.

## Reporting and Citation

Work that uses the operators is reproducible when it reports the following
settings:

- the SensorFault-Bench release or commit
- the operator keys
- every constructor argument that differs from its default
- the channel-count mode and the severity distribution

Please cite the SensorFault-Bench paper, listed under
[Citation](../README.md#citation), for work that uses the operators. The
operators `linear_drift`, `nonlinear_drift`, `scaling`, `time_varying_scaling`,
`trimming_constant`, `trimming_varying`, and `outliers` adapt failure modes
from Brandt et al. (2025), "From Faults to Features: Pretraining to Learn
Robust Representations against Sensor Failures", NeurIPS 2025,
<https://doi.org/10.52202/085713-5194>. Work that uses these operators should
cite that paper as well. [`CITATION.cff`](../CITATION.cff) lists both works.

## License

The operators are licensed under Apache-2.0 together with the rest of the
SensorFault-Bench code (see [`LICENSE`](../LICENSE)). The operators that adapt
failure modes from Brandt et al. (2025) were written by consulting the
implementation in the FaultsToFeatures repository, whose license is recorded in
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md). Dataset terms are
separate from the code license and are listed in
[`data/README.md`](../data/README.md).
