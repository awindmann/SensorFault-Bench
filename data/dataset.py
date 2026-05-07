import warnings

import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass
from torch.utils.data import Dataset
from typing import Iterable, Optional, Sequence, Tuple

from data.samplers import (
    PerturbationSeveritySampler,
    normalize_sampled_perturbation,
)
from utils.noise import apply_raw_space_noise
from utils.parsing import (
    build_noise_channel_mask,
    parse_train_perturbation_probability,
    validate_input_stats,
    validate_noise_channels,
)


@dataclass(frozen=True)
class BatchSegment:
    start: int
    length: int
    batch_id: object

    @property
    def end(self) -> int:
        return self.start + self.length


class TSDataset(Dataset):
    """Time Series Dataset
    A sample consists of a (random) time window + consecutive time horizon.
    Args:
        df (pd.DataFrame): dataframe containing the data
        file_path (str): path to csv file containing the data
        input_len (int): length of input sequence
        target_len (int): length of target sequence
        stride (int): stride between samples. Only used if n_samples is None and samples are thus drawn sequentially.
        n_samples (int): number of samples to draw. If None, all possible samples are drawn sequentially. Else, samples are drawn randomly.
        continuous_features (list): list of continuous features
        discrete_features (list): list of discrete features
        target_columns (list): optional subset of dataframe columns to use as prediction targets
        input_columns (list): optional subset (and order) of dataframe columns to expose as model inputs
        segments (Iterable[BatchSegment]): optional contiguous regions that constrain sampling (e.g., per batch)
        seed (int): seed for reproducibility
    """
    def __init__(
        self,
        df=None,
        file_path=None,
        input_len=90,
        target_len=30,
        stride=1,
        n_samples=None,
        continuous_channels=None,
        discrete_channels=None,
        target_columns=None,
        input_columns: Optional[Sequence[str]] = None,
        segments: Optional[Iterable[BatchSegment]] = None,
        seed=42,
        strict_iid=False,
    ):
        super().__init__()
        if df is not None:
            self.df = df
        elif file_path is not None:
            if file_path.endswith(".parquet"):
                self.df = pd.read_parquet(file_path)
            elif file_path.endswith(".csv"):
                self.df = pd.read_csv(file_path)
            else:
                raise ValueError("File format not supported.")
            try:  # Try to convert the first column to datetime. If not possible, ignore it.
                self.df.set_index(pd.to_datetime(self.df.iloc[:,0], format="%Y-%m-%d %H:%M:%S"), inplace=True)
                self.df.drop(self.df.columns[0], axis=1, inplace=True)
            except ValueError:
                pass
        else:
            raise ValueError("Either df or file_path must be given.")

        self.input_len = input_len
        self.target_len = target_len
        self.stride = stride  # only used if n_samples is None and samples are drawn sequentially

        self.random_sampling = n_samples is not None
        self.df_columns = self.df.columns
        if input_columns is not None:
            missing_inputs = sorted(set(input_columns) - set(self.df_columns))
            if missing_inputs:
                raise ValueError(f"Input channels {missing_inputs} not present in dataset columns.")
            self.input_columns: Tuple[str, ...] = tuple(input_columns)
        else:
            self.input_columns = tuple(self.df_columns)
        self.channel_names = pd.Index(self.input_columns)
        self._input_indices = [self.df_columns.get_loc(col) for col in self.input_columns]
        self.n_channels = len(self.input_columns)
        if target_columns is not None:
            missing_targets = sorted(set(target_columns) - set(self.df_columns))
            if missing_targets:
                raise ValueError(f"Target channels {missing_targets} not present in dataset columns.")
            self.target_columns: Tuple[str, ...] = tuple(target_columns)
            self._target_indices = [self.df_columns.get_loc(col) for col in self.target_columns]
        else:
            self.target_columns = tuple(self.df_columns)
            self._target_indices = None
        self.n_target_channels = len(self.target_columns)
        self.continuous_channels, self.discrete_channels = self.split_hybrid_data(continuous_channels, discrete_channels)

        self.rng = np.random.default_rng(seed)

        if segments is None:
            self.segments = (BatchSegment(start=0, length=len(self.df), batch_id=None),)
        else:
            self.segments = tuple(BatchSegment(seg.start, seg.length, seg.batch_id) for seg in segments)

        self.total_seq_len = self.input_len + self.target_len
        self.strict_iid = strict_iid
        self._all_start_positions = self._build_all_start_positions()

        if self.random_sampling:
            if n_samples is None:
                raise ValueError("n_samples must be provided when random sampling is enabled.")

            if self.strict_iid:
                self.sample_idxs = self._build_block_sampled_positions(n_samples)
                self.n_samples = len(self.sample_idxs)
            else:
                selected = self.rng.choice(len(self._all_start_positions), size=n_samples, replace=True)
                self.sample_idxs = self._all_start_positions[selected]
                self.n_samples = int(n_samples)
        else:
            sequential_starts = self._build_sequential_start_positions()
            self.sample_idxs = sequential_starts
            self.n_samples = int(self.sample_idxs.size)

        if self.sample_idxs.size == 0:
            raise ValueError("No valid windows available for the provided dataset configuration.")

    def _build_all_start_positions(self) -> np.ndarray:
        seq_len = self.input_len + self.target_len
        positions = []
        for seg in self.segments:
            max_start = seg.length - seq_len
            if max_start < 0:
                continue
            start_values = np.arange(seg.start, seg.start + max_start + 1, dtype=np.int64)
            positions.append(start_values)
        if not positions:
            return np.array([], dtype=np.int64)
        return np.concatenate(positions).astype(np.int64, copy=False)

    def _build_sequential_start_positions(self) -> np.ndarray:
        seq_len = self.input_len + self.target_len
        positions = []
        for seg in self.segments:
            max_start = seg.length - seq_len
            if max_start < 0:
                continue
            seg_positions = np.arange(seg.start, seg.start + max_start + 1, self.stride, dtype=np.int64)
            positions.append(seg_positions)
        if not positions:
            return np.array([], dtype=np.int64)
        return np.concatenate(positions).astype(np.int64, copy=False)

    def _build_block_sampled_positions(self, n_samples: int) -> np.ndarray:
        """Divide available data into n_samples non-overlapping blocks and sample one window per block.

        This ensures IID sampling by guaranteeing no temporal overlap between samples.
        Block size adapts dynamically to the requested n_samples.
        """
        seq_len = self.total_seq_len
        positions = []

        # Calculate total usable length and max non-overlapping samples across all segments
        segment_info = []  # (segment, usable_length, max_non_overlapping)
        total_usable = 0
        total_max_non_overlapping = 0

        for seg in self.segments:
            usable_length = seg.length - seq_len + 1  # Number of valid start positions
            if usable_length <= 0:
                continue
            max_non_overlapping = seg.length // seq_len
            segment_info.append((seg, usable_length, max_non_overlapping))
            total_usable += usable_length
            total_max_non_overlapping += max_non_overlapping

        if not segment_info:
            raise ValueError("No segments have sufficient length for sampling.")

        # Warning if requesting more than 50% of max non-overlapping capacity
        if n_samples > 0.5 * total_max_non_overlapping:
            if n_samples > total_max_non_overlapping:
                warnings.warn(
                    f"Requested {n_samples} IID samples but only {total_max_non_overlapping} "
                    f"non-overlapping windows available. Capping at {total_max_non_overlapping}.",
                    RuntimeWarning,
                )
                n_samples = total_max_non_overlapping
            else:
                warnings.warn(
                    f"Requested {n_samples} IID samples is more than 50% of the "
                    f"{total_max_non_overlapping} maximum non-overlapping windows. "
                    f"Consider reducing n_samples for better temporal coverage per block.",
                    RuntimeWarning,
                )

        # Allocate samples to segments proportionally to their usable length
        samples_per_segment = []
        remaining_samples = n_samples

        for i, (seg, usable_length, max_non_overlapping) in enumerate(segment_info):
            if i == len(segment_info) - 1:
                # Last segment gets remaining samples
                seg_samples = remaining_samples
            else:
                # Proportional allocation
                seg_samples = int(n_samples * usable_length / total_usable)

            # Cap at segment's max non-overlapping capacity
            seg_samples = min(seg_samples, max_non_overlapping)
            samples_per_segment.append(seg_samples)
            remaining_samples -= seg_samples

        # Distribute any remaining samples (due to rounding) to segments with capacity
        for i, (seg, usable_length, max_non_overlapping) in enumerate(segment_info):
            if remaining_samples <= 0:
                break
            additional = min(remaining_samples, max_non_overlapping - samples_per_segment[i])
            samples_per_segment[i] += additional
            remaining_samples -= additional

        # Sample from each segment using block-based approach
        for (seg, usable_length, _), seg_n_samples in zip(segment_info, samples_per_segment):
            if seg_n_samples <= 0:
                continue

            # Divide usable range into seg_n_samples blocks
            block_size = usable_length // seg_n_samples

            for i in range(seg_n_samples):
                block_start = seg.start + i * block_size
                # Last block extends to end of usable range
                if i == seg_n_samples - 1:
                    block_end = seg.start + usable_length
                else:
                    block_end = block_start + block_size

                # Random position within block
                pos = self.rng.integers(block_start, block_end)
                positions.append(pos)

        return np.array(positions, dtype=np.int64)

    def split_hybrid_data(self, continuous_features=None, discrete_features=None):
        """Resolve continuous/discrete channel typing without cardinality inference."""
        if continuous_features is None and discrete_features is None:
            return tuple(self.input_columns), tuple()
        if continuous_features is None or discrete_features is None:
            raise ValueError(
                "continuous_channels and discrete_channels must be provided together."
            )

        continuous_tuple = tuple(continuous_features)
        discrete_tuple = tuple(discrete_features)
        if len(set(continuous_tuple)) != len(continuous_tuple):
            raise ValueError("continuous_channels contains duplicate channel names.")
        if len(set(discrete_tuple)) != len(discrete_tuple):
            raise ValueError("discrete_channels contains duplicate channel names.")
        continuous_set = set(continuous_tuple)
        discrete_set = set(discrete_tuple)

        missing_cont = sorted(continuous_set - set(self.input_columns))
        missing_disc = sorted(discrete_set - set(self.input_columns))
        if missing_cont or missing_disc:
            raise ValueError("Continuous/discrete feature lists must reference input columns only.")

        overlap = sorted(continuous_set & discrete_set)
        if overlap:
            raise ValueError(
                f"Continuous/discrete feature lists overlap on channels: {overlap}."
            )

        assigned = continuous_set | discrete_set
        missing = [name for name in self.input_columns if name not in assigned]
        if missing:
            raise ValueError(
                f"Continuous/discrete feature lists must cover all input columns; missing {missing}."
            )
        return continuous_tuple, discrete_tuple

    def __len__(self):
        """Number of samples"""
        return int(self.sample_idxs.size)

    def __getitem__(self, index):
        """Get one sample.
        A sample consists of a time window of length input_len and a consecutive time horizon of length target_len.
        Returns:
            x (np.array): input sequence
            y (np.array): target sequence
        """
        start_idx = int(self.sample_idxs[index])
        end_idx = start_idx + self.input_len + self.target_len
        df_ = self.df.iloc[start_idx:end_idx]
        x = df_.iloc[:self.input_len, self._input_indices].to_numpy().astype(np.float32)
        y_frame = df_.iloc[self.input_len:]
        if self._target_indices is None:
            y = y_frame.to_numpy().astype(np.float32)
        else:
            y = y_frame.iloc[:, self._target_indices].to_numpy().astype(np.float32)
        del df_
        return x, y


class PerturbedDataset(Dataset):
    """
    Wrap a clean TSDataset and sample one perturbation per item.
    """

    def __init__(
        self,
        base_ds: "TSDataset",
        perturbation_sampler: PerturbationSeveritySampler,
        seed: int,
    ):
        super().__init__()
        self.base_ds = base_ds
        self.perturbation_sampler = perturbation_sampler
        self.seed = seed

        input_pos = {name: idx for idx, name in enumerate(base_ds.input_columns)}
        missing_cont = sorted(set(base_ds.continuous_channels) - set(input_pos))
        missing_disc = sorted(set(base_ds.discrete_channels) - set(input_pos))
        if missing_cont or missing_disc:
            raise ValueError("Perturbation channels must be part of the model inputs.")
        self.cont_idx = [input_pos[ch] for ch in base_ds.continuous_channels]
        self.disc_idx = [input_pos[ch] for ch in base_ds.discrete_channels]

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, index):
        x_np, y_np = self.base_ds[index]
        x = torch.as_tensor(x_np, dtype=torch.float32)
        y = torch.as_tensor(y_np, dtype=torch.float32)

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # single-process data loading, return the full instance
            rng = torch.Generator().manual_seed(self.seed + index)
        else:
            # in a worker process
            # configure each dataset to only process a portion of the original dataset.
            rng = torch.Generator().manual_seed(self.seed + worker_info.id + index)

        sampled = normalize_sampled_perturbation(
            self.perturbation_sampler(rng),
            context="PerturbedDataset perturbation_sampler",
        )
        perturbation = sampled.perturbation
        sev = sampled.severity
        severity_idx = -1 if sampled.severity_idx is None else int(sampled.severity_idx)
        x_pert, y_pert, affected_channels = perturbation(
            x,
            y,
            sev,
            rng,
            self.cont_idx,
            self.disc_idx,
        )

        affected_channels_tensor = torch.full((self.base_ds.n_channels,), -1, dtype=torch.long)
        if affected_channels is not None and len(affected_channels) > 0:
            affected_channels_tensor[:len(affected_channels)] = torch.tensor(affected_channels, dtype=torch.long)

        return {
            "x": x,
            "y": y,
            "x_pert": x_pert,
            "y_pert": y_pert,
            "pert_idx": torch.tensor(perturbation.idx, dtype=torch.long),
            "severity_idx": torch.tensor(severity_idx, dtype=torch.long),
            "severity": torch.tensor(sev, dtype=torch.float64),
            "affected_channels": affected_channels_tensor,
        }


class NoisyDataset(Dataset):
    """Wrap a TSDataset to add multiplicative Gaussian noise to inputs during training.

    Paper-aligned raw-space formula:
        x_raw = x_std * sigma + mu
        x_raw_noisy = x_raw * (1 + noise_std * eps)
        x_std_noisy = (x_raw_noisy - mu) / sigma

    Reference: Yoon et al., 2022
    Paper: https://proceedings.mlr.press/v151/yoon22a.html
    Repo: https://github.com/tetrzim/robust-probabilistic-forecasting
    """

    def __init__(
        self,
        base_ds: TSDataset,
        noise_std: float,
        noise_channels: str = "all",
        input_means: Optional[torch.Tensor] = None,
        input_stds: Optional[torch.Tensor] = None,
        noise_generator: Optional[torch.Generator] = None,
    ):
        self.base_ds = base_ds
        try:
            self.noise_std = float(noise_std)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"noise_std must be numeric; got '{noise_std}'.") from exc
        if self.noise_std < 0:
            raise ValueError("noise_std must be >= 0.")
        self.noise_channels = validate_noise_channels(
            noise_channels,
            key="train_noise_channels",
        )
        input_columns = getattr(self.base_ds, "input_columns", None)
        if input_columns is None:
            raise ValueError("NoisyDataset requires base dataset input_columns.")
        self._channel_mask = build_noise_channel_mask(
            input_columns,
            getattr(self.base_ds, "target_columns", None),
            getattr(self.base_ds, "continuous_channels", None),
            self.noise_channels,
            key="train_noise_channels",
        )
        self._input_means: torch.Tensor | None = None
        self._input_stds: torch.Tensor | None = None
        if self.noise_std > 0:
            self._input_means, self._input_stds = validate_input_stats(
                input_means,
                input_stds,
                len(input_columns),
                key="NoisyDataset",
            )
        self.noise_generator = noise_generator
        if self.noise_std > 0 and self.noise_generator is None:
            raise ValueError(
                "NoisyDataset requires an explicit noise_generator when noise_std > 0."
            )
        self._noise_seed = (
            int(self.noise_generator.initial_seed())
            if self.noise_generator is not None
            else None
        )
        self._worker_generators: dict[object, torch.Generator] = {}
        if self.noise_channels == "all":
            discrete = getattr(self.base_ds, "discrete_channels", None)
            if discrete:
                warnings.warn(
                    "train_noise_channels='all' will also perturb discrete channels; "
                    "use train_noise_channels='target_only' or 'continuous' to avoid corrupting them.",
                    UserWarning,
                )

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        x_np, y_np = self.base_ds[idx]
        x = torch.as_tensor(x_np, dtype=torch.float32)
        y = torch.as_tensor(y_np, dtype=torch.float32)
        if self.noise_std <= 0:
            return x, y
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        key = worker_id if worker_info is not None else "main"
        generator = self._worker_generators.get(key)
        if generator is None:
            seed = self._noise_seed + worker_id
            generator = torch.Generator().manual_seed(seed)
            self._worker_generators[key] = generator
        eps = torch.randn(
            x.shape,
            dtype=x.dtype,
            device=x.device,
            generator=generator,
        )
        x_noisy = apply_raw_space_noise(
            x, eps, self.noise_std,
            self._input_means, self._input_stds,
            self._channel_mask,
        )
        return x_noisy, y


class TrainPerturbedDataset(Dataset):
    """Wrap a clean TSDataset and sample fresh train-time perturbations per access."""

    def __init__(
        self,
        base_ds: TSDataset,
        perturbation_sampler: PerturbationSeveritySampler,
        perturbation_probability: float,
        perturbation_generator: Optional[torch.Generator] = None,
    ):
        self.base_ds = base_ds
        self.perturbation_sampler = perturbation_sampler
        self.perturbation_probability = parse_train_perturbation_probability(
            perturbation_probability,
            key="perturbation_probability",
        )
        self.perturbation_generator = perturbation_generator
        if self.perturbation_generator is None:
            raise ValueError(
                "TrainPerturbedDataset requires an explicit perturbation_generator."
            )
        self._perturbation_seed = int(self.perturbation_generator.initial_seed())
        self._worker_generators: dict[object, torch.Generator] = {}

        input_pos = {name: idx for idx, name in enumerate(base_ds.input_columns)}
        missing_cont = sorted(set(base_ds.continuous_channels) - set(input_pos))
        missing_disc = sorted(set(base_ds.discrete_channels) - set(input_pos))
        if missing_cont or missing_disc:
            raise ValueError("Perturbation channels must be part of the model inputs.")
        self.cont_idx = [input_pos[ch] for ch in base_ds.continuous_channels]
        self.disc_idx = [input_pos[ch] for ch in base_ds.discrete_channels]

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        x_np, y_np = self.base_ds[idx]
        x = torch.as_tensor(x_np, dtype=torch.float32)
        y = torch.as_tensor(y_np, dtype=torch.float32)

        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        key = worker_id if worker_info is not None else "main"
        generator = self._worker_generators.get(key)
        if generator is None:
            seed = self._perturbation_seed + worker_id
            generator = torch.Generator().manual_seed(seed)
            self._worker_generators[key] = generator

        draw = torch.rand((), generator=generator).item()
        if draw > self.perturbation_probability:
            return x, y

        sampled = normalize_sampled_perturbation(
            self.perturbation_sampler(generator),
            context="TrainPerturbedDataset perturbation_sampler",
        )
        perturbation = sampled.perturbation
        sev = sampled.severity
        x_pert, _y_pert, _affected_channels = perturbation(
            x,
            y,
            sev,
            generator,
            self.cont_idx,
            self.disc_idx,
        )
        return x_pert, y
