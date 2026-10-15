# Sensitivity Analyses

This guide reproduces the sensitivity analyses of the paper: the benchmark
design checks of Appendix E, the training seed analysis of Appendix H.3, and
the ETTh1 task ablation of Appendix A.2. Each section states what the paper
checks and where it reports the results, which commands reproduce them, and
which caveats apply.

All commands run from the repository root and extend the full benchmark
workflow in [`reproducing_benchmark.md`](reproducing_benchmark.md). Add your
usual `--logdir` and `--mlflow-experiment-prefix` options. To compare a
reproduction with the runs of the paper, use the bundled result tables that the
[paper artifact guide](../paper_artifact/README.md) lists for each check.

## Shared Rules

Most checks reevaluate the models of the default configuration, which were
selected on clean validation data with training and search seed 42. These checks
cover evaluation data seeds, the selector, channel counts, severity weighting,
fault direction, and timing. Train and test the default configuration before
running them.

Testing tags each selected model with its evaluation context, such as the
scenario list and the severity sampling distribution. An alternative evaluation
under the experiment prefix of the default configuration therefore replaces that
context on those models. Before such an evaluation, run the analysis of the
default configuration and keep its tables. Afterwards, rerun the same testing
command to restore the original context. Because analysis reads only runs whose
evaluation context matches its own arguments, results from different contexts
never mix.

The scored scenario vocabulary of the figures is defined in
`configs/reporting/core_figures.yaml`. When an analysis uses scenarios outside
it, run the analysis without `--full-coverage`. The analysis then writes all
result tables, omits the heatmap of method deltas per scenario, and leaves the
two delta tables per scenario family empty. With `--full-coverage`, it stops
and names the scenarios outside the vocabulary. An analysis with
`--fixed-channel-fraction` also stops for these scenarios, because its summary
compares the scenario classes of the worst scenarios.

Alternative scenarios take the position of the scenario they replace. Because
the random streams of each sample are indexed by scenario position, an
alternative scenario reuses the severity, channel, and window draws of the
scenario it replaces.

## Evaluation Data Seed, Selector, and Channel Count

The paper checks the evaluation of the default configuration against four
further evaluation data seeds, against a selector that chooses each
robustness-improved variant by its worst-scenario validation error under faults,
and against fixed fractions of affected channels (Appendices E.1 to E.3).

Test the default configuration once per setting:

```bash
uv run python run_testing.py --full-coverage --eval-data-seed 0
uv run python run_testing.py --full-coverage --eval-data-seed 1
uv run python run_testing.py --full-coverage --eval-data-seed 2
uv run python run_testing.py --full-coverage --eval-data-seed 3
uv run python run_testing.py --full-coverage --improvement-selection-mode perturbed_worst
uv run python run_testing.py --full-coverage --fixed-channel-fraction 0.5
uv run python run_testing.py --full-coverage --fixed-channel-fraction 0.25
```

Then run `run_analysis.py` with the same options, for example:

```bash
uv run python run_analysis.py --full-coverage --eval-data-seed 0
```

The selector changes which robustness-improved variants are selected, so rerun
the testing command of the default configuration afterwards, as for every other
check here.

## Training and Search Seeds

The paper repeats training and hyperparameter search with seeds 1001 and 1002
while the data splits and evaluation sampling stay fixed (Appendix H.3).
SeasonalNaive and Chronos-2 do not depend on the training seed and stay fixed.

Train each seed under its own experiment prefix and keep the seed-42 data
split:

```bash
uv run python run_training.py --seed 1001 --data-split-seed 42 --mlflow-experiment-prefix seed-1001
```

Test each dataset with the evaluation data seed of the default configuration:

```bash
uv run python run_testing.py --seed 1001 --data-split-seed 42 --mlflow-experiment-prefix seed-1001 --full-coverage --data-files BeijingAir_Tiantan --eval-data-seed 1806770612
uv run python run_testing.py --seed 1001 --data-split-seed 42 --mlflow-experiment-prefix seed-1001 --full-coverage --data-files Penmanshiel_Hourly_WT08 --eval-data-seed 1515776824
uv run python run_testing.py --seed 1001 --data-split-seed 42 --mlflow-experiment-prefix seed-1001 --full-coverage --data-files ETTh1 --eval-data-seed 341970080
uv run python run_testing.py --seed 1001 --data-split-seed 42 --mlflow-experiment-prefix seed-1001 --full-coverage --data-files traffic --eval-data-seed 74851880
```

Then run `run_analysis.py` with the same seed and prefix options. Repeat all
commands with `--seed 1002` and its own prefix, which again costs the full
training budget of the benchmark.

## Severity Weighting

The paper reevaluates the models of the default configuration with severity
sampled from Beta(1,2) instead of the uniform distribution (Appendix E.4).
While the support
and the severity endpoints are unchanged, the mean severity falls from 1/2 to
1/3. Because the StuckSensor and MissingData window lengths grow with severity,
those windows also become shorter on average. Clean errors are unchanged.

```bash
uv run python run_testing.py --full-coverage --perturbation-severity-distribution beta_1_2
uv run python run_analysis.py --full-coverage --perturbation-severity-distribution beta_1_2
```

The severity sampling distribution also applies to selection on perturbed
validation data and to evaluation with fixed channel fractions. It does not
change clean validation, clean test error, fault augmentation during training,
or the data configuration signature. Runs evaluated under Beta(1,2) carry the
scoring semantics `beta_1_2_severity_degradation`.

## Fault Direction

The paper replaces Drift and Spike with negative variants that subtract the
same magnitudes at the same sampled channels and times (Appendix E.5). The
other six scenarios are unchanged.

```bash
uv run python run_testing.py --full-coverage --perturbation-scenarios negative_drift attenuation noise negative_spike time_stretch time_compress stuck_sensor missing_data
uv run python run_analysis.py --perturbation-scenarios negative_drift attenuation noise negative_spike time_stretch time_compress stuck_sensor missing_data
```

The negative variants mirror the realized positive fault around the clean
input and draw no additional random numbers. The diagnostic covers one seed and
leaves the default configuration with positive Drift and Spike.

## Timing

The paper replaces TimeCompress with a local timing warp that compresses and
stretches the selected window while keeping both window endpoints
(Appendix E.6). It also recomputes the results with TimeCompress omitted from
the scored set.

```bash
uv run python run_testing.py --full-coverage --perturbation-scenarios drift attenuation noise spike time_stretch local_timing_warp stuck_sensor missing_data
uv run python run_analysis.py --perturbation-scenarios drift attenuation noise spike time_stretch local_timing_warp stuck_sensor missing_data
```

Omitting TimeCompress needs no new evaluation, because worst-scenario
degradation and worst-scenario fault-time MSE are maxima over the scored
scenarios and therefore follow from the tables per scenario. For the models of
the default configuration:

<!-- time-compress-omission:start -->
```python
import pandas as pd

scenarios = pd.read_csv(
    "paper_artifact/tables/default_configuration/scenario_summary.csv"
)
baselines = scenarios[scenarios["robustness_method"] == "baseline"]
without_time_compress = (
    baselines[baselines["scenario"] != "time_compress"]
    .groupby(["dataset", "model_architecture"], as_index=False)
    .agg(
        D_w_without_time_compress=("D", "max"),
        MSE_w_without_time_compress=("err_pert", "max"),
    )
)
print(without_time_compress)
```
<!-- time-compress-omission:end -->

Because the local timing warp both compresses and stretches, no check
isolates compression at a constant rate without held endpoints.

## ETTh1 Target Scope and Horizon

The paper evaluates ETTh1 with all seven channels or oil temperature (`OT`)
alone as forecast targets, at horizons of 96 and 6 steps (Appendix A.2). The
96-step input window and all seven input channels stay fixed.

Each of the four combinations of target scope and horizon needs its own
experiment prefix, because each has its own data configuration signature. For
example, oil temperature at horizon 6:

```bash
uv run python run_training.py --data-files ETTh1 --data-targets ot --input-len 96 --target-len 6 --mlflow-experiment-prefix etth1-ot-h6
uv run python run_testing.py --data-files ETTh1 --data-targets ot --input-len 96 --target-len 6 --mlflow-experiment-prefix etth1-ot-h6 --full-coverage
uv run python run_analysis.py --data-files ETTh1 --data-targets ot --input-len 96 --target-len 6 --mlflow-experiment-prefix etth1-ot-h6 --full-coverage
```

The other combinations use `--data-targets all` or `--target-len 96`. All
targets at horizon 96 form the ETTh1 task of the default configuration.
