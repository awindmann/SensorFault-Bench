# Extending Models

## Owner Paths

1. Add the implementation under `models/`.
2. Export the class from `models/__init__.py`.
3. Add the hyperparameter grid to `configs/baseline_hparams.yaml`.
4. Add the architecture to `configs/defaults.yaml` only if it should enter the
   default benchmark scope.
5. Add benchmark display and role metadata to `configs/benchmark_scope.yaml` only if
   the architecture is part of the benchmark scope.

## Benchmark Built-Ins

The benchmark currently ships these architecture keys:

- `DLinear`
- `GRU`
- `ModernTCN`
- `PatchTST`
- `TSMixer`
- `SeasonalNaive`
- `Chronos2`

`SeasonalNaive` and `Chronos2` are baseline-only reference architectures.
Method-comparison architectures are `DLinear`, `GRU`, `ModernTCN`, `PatchTST`,
and `TSMixer`.
