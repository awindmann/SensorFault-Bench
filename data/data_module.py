from typing import Any, Mapping, Optional, Tuple, List, Sequence
import os
import pandas as pd
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import warnings

from data.dataset import (
    TSDataset,
    PerturbedDataset,
    NoisyDataset,
    TrainPerturbedDataset,
    BatchSegment,
)
from data.samplers import PerturbationSeveritySampler
from data.perturbations import (
    PERTURBATION_REGISTRY,
    require_perturbation_channel_scope,
)
from utils.parsing import (
    validate_noise_channels,
    build_perturbation_scenarios_signature,
    parse_perturbation_channel_fraction_max,
    parse_perturbation_scenarios,
    parse_train_fault_profiles,
    parse_train_perturbation_probability,
    parse_train_perturbation_severity_max,
    resolve_train_perturbation_profile_config,
)


def validate_train_noise_config(train_noise_std, train_noise_channels) -> None:
    if train_noise_std is None:
        return
    try:
        noise_std = float(train_noise_std)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"train_noise_std must be numeric; got '{train_noise_std}'."
        ) from exc
    if noise_std > 0:
        if train_noise_channels is None or str(train_noise_channels).strip() == "":
            raise ValueError(
                "train_noise_channels is required when train_noise_std > 0."
            )
        validate_noise_channels(
            train_noise_channels,
            key="train_noise_channels",
        )


def _resolve_perturbation_classes(
    scenario_names: Sequence[str],
    *,
    registry: Mapping[str, Any],
    key: str,
) -> list[type]:
    names = parse_perturbation_scenarios(scenario_names, key=key)
    classes: list[type] = []
    for scenario_name in names:
        pert_class = registry.get(scenario_name)
        if pert_class is None:
            available = ", ".join(sorted(registry.keys()))
            raise ValueError(
                f"Unknown perturbation scenario '{scenario_name}'. "
                f"Known scenarios: {available}."
            )
        classes.append(pert_class)
    return classes


def _require_registered_perturbation_class(
    perturbation_class: type,
    *,
    registry: Mapping[str, Any],
) -> None:
    if not hasattr(perturbation_class, "name"):
        raise ValueError(
            f"Perturbation class '{perturbation_class.__name__}' is missing required name metadata."
        )
    name = getattr(perturbation_class, "name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(
            f"Perturbation class '{perturbation_class.__name__}' must declare a non-empty name."
        )
    registered_class = registry.get(name)
    if registered_class is None:
        raise ValueError(
            f"Perturbation class '{perturbation_class.__name__}' with name '{name}' "
            "is not registered in the provided perturbation registry."
        )
    if registered_class is not perturbation_class:
        raise ValueError(
            f"Perturbation class '{perturbation_class.__name__}' does not match "
            f"the registered class for name '{name}'."
        )


def _instantiate_perturbations(
    perturbation_classes: Sequence[type],
    *,
    registry: Mapping[str, Any],
    channel_fraction_max: float,
) -> list[Any]:
    perturbations: list[Any] = []
    for pert_class in perturbation_classes:
        _require_registered_perturbation_class(pert_class, registry=registry)
        scope = require_perturbation_channel_scope(
            pert_class,
            context=f"Perturbation class '{pert_class.__name__}'",
        )
        if scope in {"continuous", "discrete"}:
            perturbations.append(
                pert_class(channel_frac=channel_fraction_max)
            )
            continue
        if scope == "all":
            perturbations.append(pert_class())
            continue
        raise AssertionError(
            f"Unhandled channel scope '{scope}' for perturbation class '{pert_class.__name__}'."
        )
    return perturbations


def _build_probability_vector(
    p_perturbations: Any,
    *,
    n_perturbations: int,
) -> torch.Tensor:
    if isinstance(p_perturbations, str):
        mode = p_perturbations.strip()
        if mode != "uniform":
            raise ValueError(
                "p_perturbations must be 'uniform' or a numeric weight vector."
            )
        return torch.ones(n_perturbations, dtype=torch.float32)
    try:
        return torch.as_tensor(p_perturbations, dtype=torch.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "p_perturbations must be 'uniform' or a numeric weight vector."
        ) from exc


def _make_bounded_uniform_sampler(bound: float):
    """Return a severity law that samples uniformly in [0, bound]."""
    def _sample(rng: torch.Generator) -> float:
        return float(torch.rand((), generator=rng).item() * bound)
    return _sample


def _build_train_severity_laws(
    perturbation_names: Sequence[str],
    *,
    severity_max: float,
) -> dict[str, Any]:
    native_sampler = _make_bounded_uniform_sampler(severity_max)
    return {name: native_sampler for name in perturbation_names}


def _build_perturbation_sampler(
    *,
    scenario_names: Sequence[str],
    registry: Mapping[str, Any],
    channel_fraction_max: float,
    p_perturbations: Any,
    severity_laws: Optional[Mapping[str, Any]],
    key: str,
) -> tuple[PerturbationSeveritySampler, list[str], dict[int, str]]:
    perturbation_classes = _resolve_perturbation_classes(
        scenario_names,
        registry=registry,
        key=key,
    )
    perturbations = _instantiate_perturbations(
        perturbation_classes,
        registry=registry,
        channel_fraction_max=channel_fraction_max,
    )
    probs = _build_probability_vector(
        p_perturbations,
        n_perturbations=len(perturbations),
    )
    sampler = PerturbationSeveritySampler(
        perturbations,
        probs,
        severity_laws,
    )
    for local_idx, perturbation in enumerate(perturbations):
        perturbation.idx = int(local_idx)
    perturbation_names = [pert.name for pert in perturbations]
    perturbation_name_by_idx = {
        int(idx): pert.name for idx, pert in enumerate(perturbations)
    }
    return sampler, perturbation_names, perturbation_name_by_idx


def validate_perturbation_channel_support(
    *,
    dataset_key: str,
    perturbations: Sequence[Any],
    continuous_channels: Sequence[str],
    discrete_channels: Sequence[str],
    n_inputs: int,
) -> None:
    """Require every configured perturbation to have a non-empty eligible channel pool."""
    scope_pool_sizes = {
        "continuous": len(continuous_channels),
        "discrete": len(discrete_channels),
        "all": int(n_inputs),
    }
    unsupported: List[str] = []
    for perturbation in perturbations:
        scope = require_perturbation_channel_scope(
            perturbation,
            context=f"Perturbation '{perturbation.name}'",
        )
        if scope_pool_sizes[scope] <= 0:
            unsupported.append(f"{perturbation.name}({scope})")
    if unsupported:
        raise ValueError(
            f"Dataset '{dataset_key}' does not support configured perturbations: "
            f"{', '.join(unsupported)}. "
            f"continuous_channels={len(continuous_channels)}, "
            f"discrete_channels={len(discrete_channels)}, "
            f"n_inputs={n_inputs}."
        )


class TSDataModule(pl.LightningDataModule):
    """Data module for time series data.

    Args:
        file_path: Path to csv file containing the data.
        dataset_spec: ResolvedDatasetSpec with path, channels, batch column, and split mode.
        input_len: Length of input sequence.
        target_len: Length of target sequence.
        train_split: Fraction of training data.
        val_split: Fraction of validation data.
        purged_fraction: Fraction of data purged at start of val/test sets to avoid leakage.
        stride: Stride between samples (used when n_samples is None for sequential drawing).
        n_train_samples: Number of train samples (None = sequential).
        n_val_samples: Number of val samples (None = sequential).
        n_test_samples: Number of test samples (None = sequential).
        perturbation_channel_fraction_max: Maximum fraction of eligible sensors affected at severity 1.0.
        perturbation_scenarios: Ordered perturbation scenario names to sample from.
        p_perturbations: Perturbation probability mode.
        severity_laws: Severity distribution overrides.
        shuffle_batches_before_split: Whether to shuffle batches before splitting.
        batch_size: Batch size.
        num_workers: Number of dataloader workers.
        pin_memory: Pin memory for dataloader.
        persistent_workers: Persistent workers for dataloader.
        s3_endpoint: S3 storage URL.
        seed: Canonical data seed. Used for train sampling and as the default
            base for validation/test sampling when split-specific overrides are absent.
        val_seed: Optional override for validation-window sampling. When set,
            validation sampling uses ``val_seed + 1`` instead of ``seed + 1``.
        test_seed: Optional override for test evaluation sampling. When set,
            test-window sampling uses ``test_seed`` and perturbation sampling uses
            ``test_seed + 1`` instead of the default ``seed + 2`` / ``seed + 3``.
        strict_iid: Whether to enforce strict i.i.d. sampling.
        train_noise_std: Multiplicative noise std for randomized training.
        train_noise_channels: Channels to apply training noise to.
        train_noise_generator: Optional torch Generator for training noise.
    """
    def __init__(
        self,
        file_path=None,
        dataset_spec=None,
        input_len=100,
        target_len=20,
        train_split=0.7,
        val_split=0.15,
        purged_fraction=0.01,
        stride=1,
        n_train_samples=None,
        n_val_samples=None,
        n_test_samples=None,
        perturbation_channel_fraction_max=None,
        perturbation_scenarios=None,
        p_perturbations="uniform",
        severity_laws=None,
        shuffle_batches_before_split=False,
        batch_size=64,
        num_workers=None,
        pin_memory=True,
        persistent_workers=False,
        s3_endpoint="",
        seed=42,
        val_seed=None,
        test_seed=None,
        strict_iid=False,
        train_noise_std=0.0,
        train_noise_channels=None,
        train_noise_generator: torch.Generator | None = None,
        train_fault_profiles=None,
        train_perturbation_profile=None,
        train_perturbation_scenarios=None,
        train_perturbation_scenarios_signature=None,
        train_perturbation_probability=None,
        train_perturbation_severity_max=None,
        train_perturbation_channel_fraction_max=None,
        train_perturbation_generator: torch.Generator | None = None,
    ):
        super().__init__()

        if dataset_spec is not None:
            if file_path is not None and file_path != dataset_spec.path:
                raise ValueError("Provide either dataset_spec or file_path, not both.")
            file_path = dataset_spec.path
        if file_path is None:
            raise ValueError("A data file path or dataset_spec must be provided.")

        self.dataset_spec = dataset_spec
        self.dataset_key = dataset_spec.key if dataset_spec is not None else os.path.splitext(os.path.basename(file_path))[0]
        self.storage_options = {
            "key": os.environ.get('AWS_ACCESS_KEY_ID'),
            "secret": os.environ.get('AWS_SECRET_ACCESS_KEY'),
        }
        if s3_endpoint != "":
            self.storage_options["client_kwargs"] = {"endpoint_url": s3_endpoint}
        self.file_path = file_path
        if dataset_spec is None:
            raise ValueError("dataset_spec is required; cannot infer split_mode without it.")
        self.batch_column = dataset_spec.batch_column
        self.split_mode = dataset_spec.split_mode

        self.selected_inputs = tuple(dataset_spec.input_channels) if dataset_spec.input_channels else None
        self.selected_continuous_inputs = (
            tuple(dataset_spec.continuous_channels)
            if dataset_spec.continuous_channels is not None
            else None
        )
        self.selected_discrete_inputs = (
            tuple(dataset_spec.discrete_channels)
            if dataset_spec.discrete_channels is not None
            else None
        )
        self.selected_targets: Optional[Tuple[str, ...]] = (
            tuple(dataset_spec.target_channels)
            if dataset_spec.target_channels is not None
            else None
        )
        self.target_alias = dataset_spec.target_alias

        self.input_len = input_len
        self.target_len = target_len
        self.stride = stride

        self.train_split = train_split
        self.val_split = val_split
        self.purged_fraction = purged_fraction
        self.test_split = 1.0 - self.train_split - self.val_split
        if self.train_split <= 0 or self.val_split <= 0 or self.test_split <= 0:
            raise ValueError("train_split, val_split, and test_split must all be positive and sum to 1.")

        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers is not None else os.cpu_count()
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        if seed is None:
            raise ValueError("seed must be set for TSDataModule.")
        self.seed = int(seed)
        if val_seed is None:
            self.val_seed = None
        else:
            self.val_seed = int(val_seed)
        if test_seed is None:
            self.test_seed = None
        else:
            self.test_seed = int(test_seed)
        self._shuffle_generator = torch.Generator().manual_seed(self.seed)
        self.strict_iid = strict_iid
        if train_noise_std is None:
            train_noise_std = 0.0
        try:
            train_noise_std = float(train_noise_std)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"train_noise_std must be numeric; got '{train_noise_std}'."
            ) from exc
        if train_noise_std < 0:
            raise ValueError("train_noise_std must be >= 0.")
        self.train_noise_std = train_noise_std
        self.train_noise_channels = train_noise_channels
        self.train_noise_generator = train_noise_generator

        self.input_columns: Tuple[str, ...] = tuple()
        self.output_columns: Tuple[str, ...] = tuple()
        self.n_inputs = 0
        self.n_outputs = 0
        self._prepare_column_metadata()
        self.n_train_samples = n_train_samples
        self.n_val_samples = n_val_samples
        self.n_test_samples = n_test_samples
        self.perturbation_channel_fraction_max = parse_perturbation_channel_fraction_max(
            perturbation_channel_fraction_max,
            key="perturbation_channel_fraction_max",
        )
        self.perturbation_scenarios = parse_perturbation_scenarios(
            perturbation_scenarios,
            key="perturbation_scenarios",
        )
        self.perturbation_scenarios_signature = build_perturbation_scenarios_signature(
            self.perturbation_scenarios
        )
        self.shuffle_batches_before_split = shuffle_batches_before_split

        (
            self.pert_sampler,
            self.perturbation_names,
            self.perturbation_name_by_idx,
        ) = _build_perturbation_sampler(
            scenario_names=self.perturbation_scenarios,
            registry=PERTURBATION_REGISTRY,
            channel_fraction_max=self.perturbation_channel_fraction_max,
            p_perturbations=p_perturbations,
            severity_laws=severity_laws,
            key="perturbation_scenarios",
        )
        if tuple(self.perturbation_names) != tuple(self.perturbation_scenarios):
            raise ValueError(
                "Perturbation scenario resolution mismatch between configured names and "
                "instantiated perturbations."
            )

        self.train_perturbation_profile: Optional[str] = None
        self.train_perturbation_scenarios: tuple[str, ...] | None = None
        self.train_perturbation_scenarios_signature: Optional[str] = None
        self.train_perturbation_probability: Optional[float] = None
        self.train_perturbation_severity_max: Optional[float] = None
        self.train_perturbation_channel_fraction_max: Optional[float] = None
        self.train_perturbation_generator = train_perturbation_generator
        self.train_pert_sampler: Optional[PerturbationSeveritySampler] = None

        train_perturbation_inputs = (
            train_fault_profiles,
            train_perturbation_profile,
            train_perturbation_scenarios,
            train_perturbation_scenarios_signature,
            train_perturbation_probability,
            train_perturbation_severity_max,
            train_perturbation_channel_fraction_max,
        )
        train_perturbation_enabled = any(value is not None for value in train_perturbation_inputs)
        if train_perturbation_enabled:
            if self.train_noise_std > 0:
                raise ValueError(
                    "train_noise_* and train_perturbation_* cannot be enabled simultaneously."
                )
            if train_fault_profiles is None:
                raise ValueError(
                    "train_fault_profiles is required when train perturbation config is present."
                )
            if train_perturbation_profile is None:
                raise ValueError(
                    "train_perturbation_profile is required when train perturbation config is present."
                )
            if train_perturbation_scenarios is None:
                raise ValueError(
                    "train_perturbation_scenarios is required when train perturbation config is present."
                )
            if train_perturbation_generator is None:
                raise ValueError(
                    "train_perturbation_generator is required when train perturbation config is present."
                )
            known_train_profiles = parse_train_fault_profiles(
                train_fault_profiles,
                registry_names=tuple(PERTURBATION_REGISTRY.keys()),
            )
            (
                self.train_perturbation_profile,
                self.train_perturbation_scenarios,
                self.train_perturbation_scenarios_signature,
            ) = resolve_train_perturbation_profile_config(
                train_perturbation_profile,
                profiles=known_train_profiles,
                registry_names=tuple(PERTURBATION_REGISTRY.keys()),
                scenarios=train_perturbation_scenarios,
                scenarios_signature=train_perturbation_scenarios_signature,
                profile_key="train_perturbation_profile",
                profiles_key="train_fault_profiles",
                scenarios_key="train_perturbation_scenarios",
                signature_key="train_perturbation_scenarios_signature",
            )
            self.train_perturbation_probability = parse_train_perturbation_probability(
                train_perturbation_probability,
                key="train_perturbation_probability",
            )
            self.train_perturbation_severity_max = parse_train_perturbation_severity_max(
                train_perturbation_severity_max,
                key="train_perturbation_severity_max",
            )
            self.train_perturbation_channel_fraction_max = parse_perturbation_channel_fraction_max(
                train_perturbation_channel_fraction_max,
                key="train_perturbation_channel_fraction_max",
            )
            (
                self.train_pert_sampler,
                _train_pert_names,
                _train_pert_name_by_idx,
            ) = _build_perturbation_sampler(
                scenario_names=self.train_perturbation_scenarios,
                registry=PERTURBATION_REGISTRY,
                channel_fraction_max=self.train_perturbation_channel_fraction_max,
                p_perturbations="uniform",
                severity_laws=_build_train_severity_laws(
                    self.train_perturbation_scenarios,
                    severity_max=self.train_perturbation_severity_max,
                ),
                key="train_perturbation_scenarios",
            )

        # datasets, populated in setup()
        self.ds_train = None
        self.ds_val = None
        self.ds_test = None

    def _validate_perturbation_channel_support(self) -> None:
        validate_perturbation_channel_support(
            dataset_key=self.dataset_key,
            perturbations=self.pert_sampler.perturbations,
            continuous_channels=self.continuous_channels,
            discrete_channels=self.discrete_channels,
            n_inputs=self.n_inputs,
        )

    def _validate_train_perturbation_channel_support(self) -> None:
        if self.train_pert_sampler is None:
            return
        validate_perturbation_channel_support(
            dataset_key=self.dataset_key,
            perturbations=self.train_pert_sampler.perturbations,
            continuous_channels=self.continuous_channels,
            discrete_channels=self.discrete_channels,
            n_inputs=self.n_inputs,
        )


    def _missing_dataset_file_message(self) -> str:
        return (
            f"Dataset file for registered key '{self.dataset_key}' was not found at "
            f"'{self.file_path}'. Place the expected file under DATA_ROOT or set "
            "--data-root to the directory or s3:// URI containing it. See data/README.md "
            "for acquisition and validation instructions."
        )

    @staticmethod
    def _set_csv_datetime_index(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty or len(frame.columns) == 0:
            return frame
        try:
            datetime_index = pd.to_datetime(
                frame.iloc[:, 0],
                format="%Y-%m-%d %H:%M:%S",
                errors="raise",
            )
        except (ValueError, TypeError):
            return frame
        frame = frame.copy()
        frame.set_index(datetime_index, inplace=True)
        frame.drop(frame.columns[0], axis=1, inplace=True)
        return frame

    def _read_dataframe(self, *, nrows: int | None = None) -> pd.DataFrame:
        try:
            if self.file_path.endswith(".parquet"):
                if self.file_path.startswith('s3://'):
                    frame = pd.read_parquet(self.file_path, storage_options=self.storage_options)
                else:
                    frame = pd.read_parquet(self.file_path)
            elif self.file_path.endswith(".csv"):
                if self.file_path.startswith('s3://'):
                    frame = pd.read_csv(self.file_path, nrows=nrows, storage_options=self.storage_options)
                else:
                    frame = pd.read_csv(self.file_path, nrows=nrows)
                frame = self._set_csv_datetime_index(frame)
            else:
                raise ValueError("File format not supported.")
        except FileNotFoundError as exc:
            raise FileNotFoundError(self._missing_dataset_file_message()) from exc
        return frame

    def _prepare_column_metadata(self):
        df_sample = self._read_dataframe(nrows=3)

        available_columns = list(df_sample.columns)
        del df_sample

        batch_column = self.batch_column
        if batch_column is not None:
            if batch_column not in available_columns:
                raise ValueError(f"Batch column '{batch_column}' not found among dataset columns.")
            available_columns = [col for col in available_columns if col != batch_column]

        if self.selected_inputs is not None:
            if batch_column is not None and batch_column in self.selected_inputs:
                raise ValueError(f"Batch column '{batch_column}' cannot be part of the input channel selection.")
            missing_inputs = sorted(set(self.selected_inputs) - set(available_columns))
            if missing_inputs:
                raise ValueError(
                    f"Input channels {missing_inputs} not found in dataset columns {available_columns}."
                )
            input_columns = list(self.selected_inputs)
        else:
            input_columns = available_columns

        if self.selected_targets is None:
            output_columns = tuple(input_columns)
        else:
            if batch_column is not None and batch_column in self.selected_targets:
                raise ValueError(f"Batch column '{batch_column}' cannot be part of the target selection.")
            missing_targets = sorted(set(self.selected_targets) - set(available_columns))
            if missing_targets:
                raise ValueError(
                    f"Target channels {missing_targets} not present in dataset columns {available_columns}."
                )
            output_columns = self.selected_targets

        self.input_columns = tuple(input_columns)
        self.output_columns = tuple(output_columns)
        self.n_inputs = len(self.input_columns)
        self.n_outputs = len(self.output_columns)
        if all(col in self.input_columns for col in self.output_columns):
            self.target_column_indices = tuple(self.input_columns.index(col) for col in self.output_columns)
        else:
            self.target_column_indices = None
        self.batch_column = batch_column

    def _load_full_dataframe(self) -> pd.DataFrame:
        return self._read_dataframe(nrows=None)

    @staticmethod
    def _collect_segments(batch_ids: pd.Series) -> List[BatchSegment]:
        if batch_ids.empty:
            raise ValueError("Dataset is empty, cannot build segments.")
        if batch_ids.isna().any():
            raise ValueError("Batch identifiers contain missing values.")
        segments: List[BatchSegment] = []
        current_id = batch_ids.iloc[0]
        start_idx = 0
        completed_ids = set()
        repeated_ids = []
        for idx in range(1, len(batch_ids)):
            value = batch_ids.iloc[idx]
            if value != current_id:
                length = idx - start_idx
                if length > 0:
                    segments.append(BatchSegment(start=start_idx, length=length, batch_id=current_id))
                completed_ids.add(current_id)
                if value in completed_ids and value not in repeated_ids:
                    repeated_ids.append(value)
                start_idx = idx
                current_id = value
        final_length = len(batch_ids) - start_idx
        if final_length > 0:
            segments.append(BatchSegment(start=start_idx, length=final_length, batch_id=current_id))
        if repeated_ids:
            raise ValueError(
                "Batch identifiers must be contiguous; found non-contiguous repeated batch ids: "
                f"{repeated_ids}."
            )
        return segments

    @staticmethod
    def _slice_segments(segments: List[BatchSegment], start: int, end: int) -> List[BatchSegment]:
        sliced: List[BatchSegment] = []
        for seg in segments:
            seg_start = seg.start
            seg_end = seg.end
            if seg_end <= start:
                continue
            if seg_start >= end:
                break
            clip_start = max(seg_start, start)
            clip_end = min(seg_end, end)
            length = clip_end - clip_start
            if length <= 0:
                continue
            sliced.append(BatchSegment(start=clip_start, length=length, batch_id=seg.batch_id))
        return sliced

    @staticmethod
    def _purge_segments(segments: List[BatchSegment], fraction: float) -> List[BatchSegment]:
        if not segments or fraction <= 0:
            return segments
        total = sum(seg.length for seg in segments)
        purge = int(total * fraction)
        if purge <= 0:
            return segments
        trimmed: List[BatchSegment] = []
        remaining = purge
        for idx, seg in enumerate(segments):
            if remaining >= seg.length:
                remaining -= seg.length
                continue
            trimmed.append(BatchSegment(start=seg.start + remaining, length=seg.length - remaining, batch_id=seg.batch_id))
            remaining = 0
            trimmed.extend(segments[idx + 1:])
            break
        if remaining > 0 or not trimmed:
            raise ValueError("Purging removed all rows from the split; adjust purged_fraction or splits.")
        return trimmed

    @staticmethod
    def _reindex_segments(df: pd.DataFrame, segments: List[BatchSegment]) -> Tuple[pd.DataFrame, Tuple[BatchSegment, ...]]:
        if not segments:
            return pd.DataFrame(columns=df.columns), tuple()
        pieces: List[pd.DataFrame] = []
        new_segments: List[BatchSegment] = []
        cursor = 0
        for seg in segments:
            chunk = df.iloc[seg.start:seg.end]
            if chunk.empty:
                continue
            pieces.append(chunk)
            new_segments.append(BatchSegment(start=cursor, length=len(chunk), batch_id=seg.batch_id))
            cursor += len(chunk)
        if not new_segments:
            return pd.DataFrame(columns=df.columns), tuple()
        combined = pd.concat(pieces, axis=0).reset_index(drop=True)
        return combined, tuple(new_segments)

    def _partition_segments_by_batches(
        self,
        segments: Sequence[BatchSegment],
    ) -> Tuple[List[BatchSegment], List[BatchSegment], List[BatchSegment]]:
        total = sum(seg.length for seg in segments)
        train_threshold = total * self.train_split
        val_threshold = total * (self.train_split + self.val_split)
        train_segments: List[BatchSegment] = []
        val_segments: List[BatchSegment] = []
        test_segments: List[BatchSegment] = []
        acc = 0
        for seg in segments:
            acc += seg.length
            if acc <= train_threshold or not train_segments:
                train_segments.append(seg)
                continue
            if acc <= val_threshold or not val_segments:
                val_segments.append(seg)
                continue
            test_segments.append(seg)
        if not train_segments or not val_segments or not test_segments:
            raise ValueError("Not enough batches to create train/val/test splits without splitting batches.")
        return train_segments, val_segments, test_segments

    @staticmethod
    def _purge_each_segment(
        segments: Sequence[BatchSegment],
        fraction: float,
        *,
        min_length: int,
        split_name: str,
    ) -> List[BatchSegment]:
        if not segments or fraction <= 0:
            return list(segments)
        trimmed: List[BatchSegment] = []
        failures: List[str] = []
        for seg in segments:
            purge = int(seg.length * fraction)
            remaining = seg.length - purge
            if remaining < min_length:
                failures.append(
                    f"{seg.batch_id} (remaining_length={remaining}, required_length={min_length})"
                )
                continue
            trimmed.append(
                BatchSegment(
                    start=seg.start + purge,
                    length=remaining,
                    batch_id=seg.batch_id,
                )
            )
        if failures:
            raise ValueError(
                f"Purging removed usable {split_name} windows for batch segments: {failures}."
            )
        return trimmed

    def _compute_within_batch_split_bounds(
        self,
        *,
        seg: BatchSegment,
        seq_len: int,
    ) -> Tuple[int, int]:
        length = seg.length
        min_total_length = 3 * seq_len
        if length < min_total_length:
            raise ValueError(
                f"Batch '{seg.batch_id}' is too short for split_mode='within_batches': "
                f"length={length}, required_length>={min_total_length}."
            )
        raw_train_end = int(length * self.train_split)
        raw_val_end = int(length * (self.train_split + self.val_split))
        # Clamp to guarantee each split has at least seq_len rows.
        # The early length >= 3*seq_len check ensures these clamps are feasible.
        train_end = max(raw_train_end, seq_len)
        train_end = min(train_end, length - (2 * seq_len))
        val_end = max(raw_val_end, train_end + seq_len)
        val_end = min(val_end, length - seq_len)
        return train_end, val_end

    def _partition_segments_within_batches(
        self,
        segments: Sequence[BatchSegment],
        *,
        seq_len: int,
    ) -> Tuple[List[BatchSegment], List[BatchSegment], List[BatchSegment]]:
        train_segments: List[BatchSegment] = []
        val_segments: List[BatchSegment] = []
        test_segments: List[BatchSegment] = []
        for seg in segments:
            train_end, val_end = self._compute_within_batch_split_bounds(
                seg=seg,
                seq_len=seq_len,
            )
            train_segments.append(
                BatchSegment(start=seg.start, length=train_end, batch_id=seg.batch_id)
            )
            val_segments.append(
                BatchSegment(
                    start=seg.start + train_end,
                    length=val_end - train_end,
                    batch_id=seg.batch_id,
                )
            )
            test_segments.append(
                BatchSegment(
                    start=seg.start + val_end,
                    length=seg.length - val_end,
                    batch_id=seg.batch_id,
                )
            )
        return train_segments, val_segments, test_segments

    def _make_loader(self, ds, shuffle=True, drop_last=True):
        generator = self._shuffle_generator if shuffle else None
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=drop_last,
            generator=generator,
        )

    def _resolve_split_frames_and_segments(
        self,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        tuple[BatchSegment, ...],
        tuple[BatchSegment, ...],
        tuple[BatchSegment, ...],
    ]:
        df_full = self._load_full_dataframe()
        required_cols = list(dict.fromkeys(self.input_columns + tuple(self.output_columns)))
        missing_required = sorted(set(required_cols + ([self.batch_column] if self.batch_column else [])) - set(df_full.columns))
        if missing_required:
            raise ValueError(f"Columns missing from dataframe: {missing_required}")
        feature_df = df_full.loc[:, required_cols].copy()
        if self.batch_column is not None:
            batch_ids = df_full[self.batch_column].copy()
            if self.batch_column in feature_df.columns:
                feature_df.drop(columns=[self.batch_column], inplace=True)
        else:
            batch_ids = pd.Series(0, index=df_full.index, name="batch_id_placeholder")
        del df_full

        segments = self._collect_segments(batch_ids)
        seq_len = self.input_len + self.target_len
        segments = [seg for seg in segments if seg.length >= seq_len]
        if not segments:
            raise ValueError("No batch segments satisfy the required input/target window length.")

        base_df, base_segments = self._reindex_segments(feature_df, segments)
        if not base_segments:
            raise ValueError("Failed to assemble base dataset after filtering short segments.")
        total_length = base_df.shape[0]
        if self.split_mode == "across_batches":
            if self.shuffle_batches_before_split:
                # Reorder entire batch segments before partitioning (reproducible with seed)
                perm = torch.randperm(len(base_segments), generator=torch.Generator().manual_seed(self.seed)).tolist()
                base_segments = tuple(base_segments[i] for i in perm)
            train_slices, val_slices, test_slices = self._partition_segments_by_batches(base_segments)
        elif self.split_mode == "within_batches":
            if self.shuffle_batches_before_split:
                raise ValueError(
                    "shuffle_batches_before_split must be false when split_mode='within_batches'."
                )
            train_slices, val_slices, test_slices = self._partition_segments_within_batches(
                base_segments,
                seq_len=seq_len,
            )
        elif self.split_mode == "temporal":
            train_end = int(total_length * self.train_split)
            val_end = int(total_length * (self.train_split + self.val_split))
            train_end = max(train_end, seq_len)
            train_end = min(train_end, total_length - max(seq_len, 1))
            val_end = max(val_end, train_end)
            val_end = min(val_end, total_length - seq_len)
            if train_end <= 0 or train_end >= total_length:
                raise ValueError("Training split boundary invalid, adjust train_split or sequence lengths.")
            if val_end < train_end:
                raise ValueError("Validation split boundary precedes training split, adjust val_split.")
            base_segment_list = list(base_segments)
            train_slices = self._slice_segments(base_segment_list, 0, train_end)
            val_slices = self._slice_segments(base_segment_list, train_end, val_end)
            test_slices = self._slice_segments(base_segment_list, val_end, total_length)
        else:
            raise ValueError(f"Unsupported split_mode '{self.split_mode}'.")

        if self.split_mode == "temporal" and self.purged_fraction > 0:
            val_slices = self._purge_segments(val_slices, self.purged_fraction)
            test_slices = self._purge_segments(test_slices, self.purged_fraction)
        elif self.split_mode == "within_batches" and self.purged_fraction > 0:
            val_slices = self._purge_each_segment(
                val_slices,
                self.purged_fraction,
                min_length=seq_len,
                split_name="validation",
            )
            test_slices = self._purge_each_segment(
                test_slices,
                self.purged_fraction,
                min_length=seq_len,
                split_name="test",
            )
        elif self.split_mode == "across_batches" and self.purged_fraction > 0:
            warnings.warn(
                f"Purging disabled for batched dataset '{self.dataset_key}' because "
                f"split_mode='across_batches' and purged_fraction={self.purged_fraction}; "
                "set purged_fraction=0 to silence.",
                RuntimeWarning,
            )

        train_df, train_segments = self._reindex_segments(base_df, train_slices)
        val_df, val_segments = self._reindex_segments(base_df, val_slices)
        test_df, test_segments = self._reindex_segments(base_df, test_slices)

        if not train_segments:
            raise ValueError("Training split contains no usable windows; adjust configuration.")
        if not test_segments:
            raise ValueError("Test split contains no usable windows; adjust configuration.")
        if self.val_split > 0 and not val_segments:
            raise ValueError("Validation split contains no usable windows; adjust configuration or purged_fraction.")

        return (
            train_df,
            val_df,
            test_df,
            train_segments,
            val_segments,
            test_segments,
        )

    def setup(self, stage=None) -> None:
        verbose = self.trainer.testing and self.trainer.test_loop.verbose if self.trainer is not None else True
        if verbose:
            print("Setting up the dataset.")

        (
            train_df,
            val_df,
            test_df,
            train_segments,
            val_segments,
            test_segments,
        ) = self._resolve_split_frames_and_segments()

        # Ensure columns to be standardized have float dtype before writing scaled values back.
        input_cols = list(self.input_columns)
        target_only = [col for col in self.output_columns if col not in self.input_columns]
        float_columns = tuple(dict.fromkeys([*input_cols, *target_only]))
        if float_columns:
            float_cast = {col: "float32" for col in float_columns}
            train_df = train_df.astype(float_cast, copy=False)
            val_df = val_df.astype(float_cast, copy=False)
            test_df = test_df.astype(float_cast, copy=False)

        input_means = train_df.loc[:, self.input_columns].mean()
        input_stds = train_df.loc[:, self.input_columns].std().replace(0, 1.0)
        overlap_targets = [col for col in self.output_columns if col in self.input_columns]
        target_only = [col for col in self.output_columns if col not in self.input_columns]
        target_means = train_df.loc[:, self.output_columns].mean()
        target_stds = train_df.loc[:, self.output_columns].std().replace(0, 1.0)
        if overlap_targets:
            target_means.loc[overlap_targets] = input_means.loc[overlap_targets]
            target_stds.loc[overlap_targets] = input_stds.loc[overlap_targets]

        train_df.loc[:, self.input_columns] = (train_df.loc[:, self.input_columns] - input_means) / input_stds
        if not val_df.empty:
            val_df.loc[:, self.input_columns] = (val_df.loc[:, self.input_columns] - input_means) / input_stds
        test_df.loc[:, self.input_columns] = (test_df.loc[:, self.input_columns] - input_means) / input_stds

        if target_only:
            train_df.loc[:, target_only] = (train_df.loc[:, target_only] - target_means.loc[target_only]) / target_stds.loc[target_only]
            if not val_df.empty:
                val_df.loc[:, target_only] = (val_df.loc[:, target_only] - target_means.loc[target_only]) / target_stds.loc[target_only]
            test_df.loc[:, target_only] = (test_df.loc[:, target_only] - target_means.loc[target_only]) / target_stds.loc[target_only]

        input_series = input_means.loc[list(self.input_columns)]
        input_std_series = input_stds.loc[list(self.input_columns)]
        target_series = target_means.loc[list(self.output_columns)]
        target_std_series = target_stds.loc[list(self.output_columns)]

        self._input_means_torch = torch.tensor(input_series.to_numpy(dtype=float), dtype=torch.float32)
        self._input_stds_torch = torch.tensor(input_std_series.to_numpy(dtype=float), dtype=torch.float32)
        self._target_means_torch = torch.tensor(target_series.to_numpy(dtype=float), dtype=torch.float32)
        self._target_stds_torch = torch.tensor(target_std_series.to_numpy(dtype=float), dtype=torch.float32)
        self.input_feature_names = list(self.input_columns)
        self.target_feature_names = list(self.output_columns)

        self.ds_train = TSDataset(
            df=train_df,
            input_len=self.input_len,
            target_len=self.target_len,
            stride=self.stride,
            n_samples=self.n_train_samples,
            continuous_channels=self.selected_continuous_inputs,
            discrete_channels=self.selected_discrete_inputs,
            input_columns=self.input_columns,
            target_columns=self.output_columns,
            seed=self.seed,
            segments=train_segments,
            strict_iid=self.strict_iid,
        )
        self.continuous_channels = self.ds_train.continuous_channels
        self.discrete_channels = self.ds_train.discrete_channels
        self._validate_perturbation_channel_support()
        self._validate_train_perturbation_channel_support()

        if self.train_noise_std > 0:
            self.ds_train = NoisyDataset(
                self.ds_train,
                self.train_noise_std,
                noise_channels=self.train_noise_channels,
                input_means=self.input_means,
                input_stds=self.input_stds,
                noise_generator=self.train_noise_generator,
            )
        if self.train_pert_sampler is not None:
            self.ds_train = TrainPerturbedDataset(
                self.ds_train,
                self.train_pert_sampler,
                perturbation_probability=self.train_perturbation_probability,
                perturbation_generator=self.train_perturbation_generator,
            )

        if self.val_seed is None:
            val_sample_seed = self.seed + 1
        else:
            val_sample_seed = self.val_seed + 1

        self.ds_val = TSDataset(
            df=val_df,
            input_len=self.input_len,
            target_len=self.target_len,
            stride=self.stride,
            n_samples=self.n_val_samples,
            continuous_channels=self.continuous_channels,
            discrete_channels=self.discrete_channels,
            input_columns=self.input_columns,
            target_columns=self.output_columns,
            seed=val_sample_seed,
            segments=val_segments,
            strict_iid=self.strict_iid,
        )

        if self.test_seed is None:
            test_sample_seed = self.seed + 2
            test_perturbation_seed = self.seed + 3
        else:
            test_sample_seed = self.test_seed
            test_perturbation_seed = self.test_seed + 1

        base_test = TSDataset(
            df=test_df,
            input_len=self.input_len,
            target_len=self.target_len,
            stride=self.stride,
            n_samples=self.n_test_samples,
            continuous_channels=self.continuous_channels,
            discrete_channels=self.discrete_channels,
            input_columns=self.input_columns,
            target_columns=self.output_columns,
            seed=test_sample_seed,
            segments=test_segments,
            strict_iid=self.strict_iid,
        )
        self.ds_test = PerturbedDataset(
            base_test,
            self.pert_sampler,
            seed=test_perturbation_seed,
        )
        if verbose:
            print("Dataset setup complete.")

    def get_split_frames(self):
        train_df, val_df, test_df, *_ = self._resolve_split_frames_and_segments()
        return train_df, val_df, test_df, self.input_columns

    def train_dataloader(self):
        return self._make_loader(self.ds_train, shuffle=True)

    def val_dataloader(self):
        return self._make_loader(self.ds_val, shuffle=False)

    def test_dataloader(self):
        return self._make_loader(self.ds_test, shuffle=False, drop_last=False)

    def destandardize_inputs(self, tensor: torch.Tensor) -> torch.Tensor:
        means = self.input_means.to(tensor.device, dtype=tensor.dtype).view(1, 1, -1)
        stds = self.input_stds.to(tensor.device, dtype=tensor.dtype).view(1, 1, -1)
        return tensor * stds + means

    def destandardize_targets(self, tensor: torch.Tensor) -> torch.Tensor:
        means = self._target_means_torch.to(tensor.device, dtype=tensor.dtype).view(1, 1, -1)
        stds = self._target_stds_torch.to(tensor.device, dtype=tensor.dtype).view(1, 1, -1)
        return tensor * stds + means

    @property
    def input_means(self) -> torch.Tensor:
        if not hasattr(self, "_input_means_torch"):
            raise ValueError(
                "TSDataModule input means are unavailable before setup()."
            )
        return self._input_means_torch

    @property
    def input_stds(self) -> torch.Tensor:
        if not hasattr(self, "_input_stds_torch"):
            raise ValueError(
                "TSDataModule input stds are unavailable before setup()."
            )
        return self._input_stds_torch
