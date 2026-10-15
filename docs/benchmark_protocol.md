# Benchmark Protocol

This guide defines the parts of the SensorFault-Bench protocol and names the
code that implements each part. The paper sections "Protocol for structured
sensor faults" and "Robustness score and evaluation" give the formal
definitions and their rationale.

## Scenarios

A scenario is one fault operator from `data/perturbations.py`, applied to the
input history of a test window. The scored set $\mathcal{P}$ is the ordered list
`PERTURBATION_SCENARIOS` in `configs/defaults.yaml`. It contains four value
scenarios (Drift, Attenuation, Noise, Spike), two timing scenarios
(TimeStretch, TimeCompress), and two availability scenarios (StuckSensor,
MissingData). Operators never change the forecast target, so the recorded
future remains the correct forecast.

Each operator receives the input window, the target, a severity, a per-sample
`torch.Generator`, and the continuous and discrete channel indices. It returns
the perturbed input, the unchanged target, and the affected channels. All
randomness comes from the explicit generator, so an evaluation is reproducible
from its seeds.

## Severity and Channels

Severity $s \in [0, 1]$ interpolates linearly between the operator endpoints at
$s = 0$ and $s = 1$. The operator constructors in `data/perturbations.py` define
these endpoints, which the README scenario table lists.

Scenarios scoped to channels act on a subset of the continuous channels, drawn
from the sample generator. The subset grows with severity. No channel is
affected at $s = 0$, every severity above 0 affects at least one channel, and
the count rises to $\lceil 0.5 \cdot C \rceil$ channels at $s = 1$, where $C$ is
the number of eligible channels and 0.5 is `PERTURBATION_CHANNEL_FRACTION_MAX`.
MissingData is the exception and acts on all channels.

Testing draws one severity per test window and scenario from the severity
sampling distribution `PERTURBATION_SEVERITY_DISTRIBUTION`, which is uniform
on $[0, 1]$ in the benchmark. `data/samplers.py` implements the distributions.

## Reported Quantities

For a selected model $f$, clean and perturbed errors use one shared sample of
test windows. Fresh severity and perturbation randomness is drawn for each
scenario and sample.

- **Clean error** $\mathrm{MSE}_c(f)$: test MSE on unperturbed input histories.
- **Scenario degradation**
  $\mathcal{D}_p(f) = \mathrm{MSE}_p(f) / \mathrm{MSE}_c(f)$,
  where $\mathrm{MSE}_p(f)$ is the test MSE under scenario $p$, averaged over
  the severity sampling distribution. A value of 1 means no expected error
  increase.
- **Worst-scenario degradation**
  $\mathcal{D}_w(f) = \max_{p \in \mathcal{P}} \mathcal{D}_p(f)$,
  which measures relative robustness. Exact ties break in scenario order.
- **Worst-scenario fault-time error**
  $\mathrm{MSE}_w(f) = \max_{p \in \mathcal{P}} \mathrm{MSE}_p(f)$.
  Because $\mathrm{MSE}_c(f)$ does not depend on $p$, it shares the worst
  scenario with $\mathcal{D}_w(f)$.

Lower $\mathcal{D}_w$ means better relative robustness at the model's own worst
scenario. Lower $\mathrm{MSE}_w$ can reflect lower clean error rather than
lower fault sensitivity, so the three quantities are reported together.
`testing/evaluation.py` runs the evaluation loop and computes the quantities
with `utils/scoring.py`.

## Intervals

Intervals are 95% percentile bootstrap intervals with
`BOOTSTRAP_CI_RESAMPLES` draws. They are conditional on the selected models.

- For one model, each draw resamples the shared pool of test windows and reuses
  it for the clean error and every scenario before the quantities are
  recomputed.
- For a method mean on one dataset, each draw resamples the pairs of method and
  baseline forecasting model across architectures and recomputes the mean
  delta.

These intervals summarize variation over sampled test windows or architecture
pairs but not over retraining or a different selection, which the analysis of
training and search seeds in Appendix H.3 of the paper reports separately.

## Selection

Every combination of dataset, architecture, and method receives its own
hyperparameter search. The benchmark budget is up to 40 candidates from a
seeded random subgrid of `configs/baseline_hparams.yaml` or the method recipe,
trained for at most 200 epochs with early stopping. Models trained end to end
are selected by their clean validation objective, whereas wrapping methods are
selected by the clean validation MSE of the wrapper. Candidates are ranked by
the lowest validation score, with ties broken by the most recent end time and
then by the run ID. The validation score covers full batches only and omits
the windows of a final incomplete batch. Test data never enters selection.
`pipelines/selection.py` and `pipelines/ranking.py` implement these rules.

## Method Pairing

A robustness-improvement method is compared with the baseline forecasting
model of the same dataset and architecture, under the same data configuration
signature. For each reported quantity $Q$, the paired delta is
$\Delta Q = Q(f') - Q(f_b)$, where $f'$ is the selected robustness-improved
variant and $f_b$ the selected baseline forecasting model, so negative deltas
indicate improvement. Worst-scenario quantities are taken at each model's own
worst scenario, which can differ between the two sides of a pair. Method
results per dataset average the paired deltas over the architectures that the
method applies to. `testing/meta_analysis.py` matches the pairs and computes
the deltas.

## Transfer Split

Fault augmentation trains on the seven training-only transfer fault families
`linear_drift`, `nonlinear_drift`, `scaling`, `time_varying_scaling`,
`trimming_constant`, `trimming_varying`, and `packet_loss`. These families are
disjoint from the scored set, so the results of fault augmentation measure
transfer to unseen scenarios. The recipe
`configs/pipelines/fault_augmentation.yaml` defines the training profiles,
which `data/data_module.py` applies to training windows only.

## Fault Exposure in Training

The scored scenarios are reserved for evaluation. The baseline forecasting
models train on clean windows, and fault augmentation draws its training faults
from the transfer families, so in the default configuration no model saw a
scored scenario during training. A model trained or pretrained with the scored
operators, in their benchmark form or in a calibrated form, can still be
evaluated. Its results then measure robustness to faults seen in training and
are not comparable with the default configuration.

A report should therefore name the fault operators that the model saw during
training, including pretraining, together with their parameters. The benchmark
relies on this disclosure and does not inspect training data in code. Training
perturbations of a robustness-improvement method, such as the Gaussian input
noise of randomized training, can resemble a scored scenario. They belong to
the method's recipe under `configs/pipelines/` and are reported with the
method. [`using_fault_operators.md`](using_fault_operators.md) describes uses
of the operators outside the benchmark.

## Reporting Results

A result is comparable with the default configuration when it follows that
configuration on all four datasets. A report on a model or method
states:

- clean MSE, worst-scenario degradation, worst-scenario fault-time MSE, and
  the worst scenario
- for a robustness-improvement method, the paired deltas against the baseline
  forecasting model of the same architecture, with their intervals
- the `data_config_signature` of each dataset, which matches the entry under
  `data_config_signatures` in `paper_artifact/MANIFEST.json` for the default
  configuration
- the number of test windows and the `robustness_scoring_semantics` of the
  evaluation, which are 10,000 and `uniform_severity_degradation` in the
  benchmark
- the training and search seed, which is 42 in the default configuration, and
  the evaluation data seed when it differs from the default
- the SensorFault-Bench release or commit
- the fault operators that the model saw during training, including
  pretraining, as described in the previous section
- every deviation from the default configuration, such as fewer datasets,
  other scenarios, another severity sampling distribution, or a tuning budget
  that differs from the one described under Selection

A result with deviations is reported apart from the comparison with the
default configuration.

## Terms in Code and Results

The paper, the configuration files, and the result tables name some concepts
differently.

| Paper term | Name in code, configuration, or result tables |
| --- | --- |
| Scenario or fault operator | Perturbation class in `data/perturbations.py`, keyed in `PERTURBATION_REGISTRY` |
| Scored set $\mathcal{P}$ | `PERTURBATION_SCENARIOS` in `configs/defaults.yaml` |
| Transfer fault families $\mathcal{P}_{\mathrm{trans}}$ | `holdout` profile in `configs/pipelines/fault_augmentation.yaml` |
| Severity $s$ | `severity` in `tables/scenario_samples.csv`, drawn from `PERTURBATION_SEVERITY_DISTRIBUTION` |
| Clean error $\mathrm{MSE}_c$ | `MSE_test` |
| Scenario degradation $\mathcal{D}_p$ | `D` in `scenario_summary.csv` |
| Worst-scenario degradation $\mathcal{D}_w$ | `D_w` |
| Worst-scenario fault-time error $\mathrm{MSE}_w$ | `MSE_w`, also `err_pert_ws` |
| Robustness-improvement method | `pipeline_method` in recipes and run tags, `robustness_method` in result tables |
| Recipe variant | `pipeline_id` |
| Baseline forecasting model | Runs with `robustness_method` equal to `baseline` |
| Paired delta $\Delta Q$ | `delta_MSE_test`, `delta_D_w`, `delta_MSE_w` |
| Data configuration | `data_config_signature` |
