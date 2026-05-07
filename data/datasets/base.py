from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

from utils.parsing import (
    join_data_root_path,
    parse_dataset_split_mode,
    parse_relative_dataset_path,
    validate_channel_name_partition,
    validate_split_mode_batch_column,
)

TargetLike = Union[str, Sequence[str], None]


def _normalize_alias(alias: str) -> str:
    return alias.strip().lower()


def _normalize_path(path: str) -> str:
    path_str = str(path)
    if path_str.lower().startswith("s3:"):
        _, _, suffix = path_str.partition(":")
        return "s3://" + suffix.lstrip("/")
    return Path(path_str).as_posix()


@dataclass(frozen=True)
class ResolvedDatasetSpec:
    key: str
    path: str
    input_channels: Optional[Tuple[str, ...]]
    target_channels: Optional[Tuple[str, ...]]
    target_alias: Optional[str]
    split_mode: str
    description: Optional[str] = None
    batch_column: Optional[str] = None
    continuous_channels: Optional[Tuple[str, ...]] = None
    discrete_channels: Optional[Tuple[str, ...]] = None
    relative_path: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _normalize_path(self.path))
        if self.relative_path is not None:
            object.__setattr__(
                self,
                "relative_path",
                parse_relative_dataset_path(
                    self.relative_path,
                    key=f"ResolvedDatasetSpec[{self.key}].relative_path",
                ),
            )
        object.__setattr__(self, "split_mode", parse_dataset_split_mode(self.split_mode))
        input_channels, continuous_channels, discrete_channels = validate_channel_name_partition(
            self.input_channels,
            self.continuous_channels,
            self.discrete_channels,
            context=f"ResolvedDatasetSpec[{self.key}]",
        )
        if input_channels is not None and not input_channels:
            raise ValueError("input_channels cannot be empty; use None for 'all'.")
        if self.target_channels is not None and not self.target_channels:
            raise ValueError("target_channels cannot be empty; use None for 'all'.")
        if input_channels is not None and continuous_channels is None and discrete_channels is None:
            continuous_channels = input_channels
            discrete_channels = ()
        object.__setattr__(self, "input_channels", input_channels)
        object.__setattr__(self, "continuous_channels", continuous_channels)
        object.__setattr__(self, "discrete_channels", discrete_channels)
        validate_split_mode_batch_column(self.split_mode, self.batch_column)

    @property
    def path_posix(self) -> str:
        return _normalize_path(self.path)


@dataclass
class DatasetSpec:
    key: str
    path: str
    split_mode: str
    input_channels: Optional[Sequence[str]] = None
    channel_groups: Mapping[str, Sequence[str]] = field(default_factory=dict)
    default_target: TargetLike = None
    description: Optional[str] = None
    batch_column: Optional[str] = None
    continuous_channels: Optional[Sequence[str]] = None
    discrete_channels: Optional[Sequence[str]] = None
    normalized_path: str = field(init=False)

    def __post_init__(self) -> None:
        normalized_groups = {}
        alias_display = {}
        for alias, columns in (self.channel_groups or {}).items():
            if not alias:
                raise ValueError("Target group aliases must be non-empty strings.")
            norm_alias = _normalize_alias(alias)
            normalized_groups[norm_alias] = tuple(columns)
            alias_display[norm_alias] = alias
        self._target_groups_lookup = normalized_groups
        self._alias_display = alias_display
        normalized_path = parse_relative_dataset_path(
            self.path,
            key=f"DatasetSpec[{self.key}].path",
        )
        self.path = normalized_path
        self.normalized_path = normalized_path
        self.split_mode = parse_dataset_split_mode(self.split_mode)
        (
            self._input_channels_tuple,
            self._continuous_channels_tuple,
            self._discrete_channels_tuple,
        ) = validate_channel_name_partition(
            self.input_channels,
            self.continuous_channels,
            self.discrete_channels,
            context=f"DatasetSpec[{self.key}]",
        )
        if self._input_channels_tuple is not None and not self._input_channels_tuple:
            raise ValueError("input_channels cannot be empty; use None for 'all'.")
        if (
            self._input_channels_tuple is not None
            and self._continuous_channels_tuple is None
            and self._discrete_channels_tuple is None
        ):
            self._continuous_channels_tuple = self._input_channels_tuple
            self._discrete_channels_tuple = ()
        validate_split_mode_batch_column(self.split_mode, self.batch_column)

    def resolve(
        self,
        target: TargetLike = None,
        *,
        data_root: str | None = None,
    ) -> ResolvedDatasetSpec:
        alias = None
        resolved_target: Optional[Tuple[str, ...]]
        target_value = target if target is not None else self.default_target

        if isinstance(target_value, str):
            norm = _normalize_alias(target_value)
            alias = target_value
            if not norm:
                raise ValueError(
                    f"Target alias for dataset '{self.key}' must be non-empty."
                )
            if norm == "all":
                resolved_target = None
            else:
                try:
                    resolved_target = self._target_groups_lookup[norm]
                except KeyError as exc:
                    raise KeyError(
                        f"Unknown target alias '{target_value}' for dataset '{self.key}'. "
                        f"Available: {sorted(self._alias_display.values()) or ['all']}"
                    ) from exc
        elif target_value is None:
            alias = None if self.default_target not in (None, "all") else "all"
            resolved_target = None if self.default_target in (None, "all") else self._coerce_sequence(self.default_target)
        else:
            resolved_target = self._coerce_sequence(target_value)

        resolved_path = self.path
        if data_root is not None:
            resolved_path = join_data_root_path(
                data_root,
                self.path,
                key=f"DATA_ROOT for dataset '{self.key}'",
            )

        return ResolvedDatasetSpec(
            key=self.key,
            path=resolved_path,
            relative_path=self.normalized_path,
            input_channels=self._input_channels_tuple,
            target_channels=resolved_target,
            target_alias=alias,
            split_mode=self.split_mode,
            description=self.description,
            batch_column=self.batch_column,
            continuous_channels=self._continuous_channels_tuple,
            discrete_channels=self._discrete_channels_tuple,
        )

    @staticmethod
    def _coerce_sequence(values: Union[Sequence[str], None]) -> Tuple[str, ...]:
        if values is None:
            return ()
        if isinstance(values, str):
            return (values,)
        coerced = tuple(values)
        if not coerced:
            raise ValueError("Target channel list must not be empty; use None for 'all'.")
        blank_channels = [str(value) for value in coerced if not str(value).strip()]
        if blank_channels:
            raise ValueError("Target channel names must be non-empty.")
        return coerced

    @property
    def available_target_aliases(self) -> Tuple[str, ...]:
        aliases = tuple(self._alias_display.values())
        return aliases if aliases else ("all",)


class DatasetRegistry:
    def __init__(self, *specs: DatasetSpec):
        self._specs_by_key: Dict[str, DatasetSpec] = {}
        self._normalized_keys: set[str] = set()
        self._paths: set[str] = set()
        for spec in specs:
            self.register(spec)

    def register(self, spec: DatasetSpec) -> None:
        key_norm = _normalize_alias(spec.key)
        if key_norm in self._normalized_keys:
            raise ValueError(f"Duplicate dataset key '{spec.key}'.")
        if spec.normalized_path in self._paths:
            raise ValueError(f"Duplicate dataset path '{spec.path}'.")
        self._specs_by_key[spec.key] = spec
        self._normalized_keys.add(key_norm)
        self._paths.add(spec.normalized_path)

    def keys(self) -> Sequence[str]:
        return tuple(spec.key for spec in self._specs_by_key.values())

    def get(self, identifier: str) -> DatasetSpec:
        if identifier in self._specs_by_key:
            return self._specs_by_key[identifier]
        raise KeyError(f"Dataset '{identifier}' is not registered.")

    def resolve_many(
        self,
        identifiers: Iterable[str],
        targets: Optional[Sequence[TargetLike]] = None,
        *,
        data_root: str,
    ) -> Tuple[ResolvedDatasetSpec, ...]:
        resolved_specs = []
        identifiers = list(identifiers)
        overrides = list(targets) if targets is not None else None
        if overrides is not None and len(overrides) > len(identifiers):
            raise ValueError(
                "DATA_TARGETS contains more entries than DATA_FILES; provide at most "
                "one target override per dataset."
            )
        for idx, identifier in enumerate(identifiers):
            override = None
            if overrides is not None and idx < len(overrides):
                override = overrides[idx]
                if isinstance(override, str) and not override.strip():
                    raise ValueError(
                        "DATA_TARGETS entries must be non-empty, omit the entry to use "
                        "the dataset default target."
                    )
            spec = self.get(identifier)
            resolved_specs.append(spec.resolve(override, data_root=data_root))
        return tuple(resolved_specs)


def resolve_with_defaults(
    datasets: Optional[Union[str, Sequence[str]]] = None,
    targets: Optional[Union[TargetLike, Sequence[TargetLike]]] = None,
    *,
    data_root: str,
):
    if datasets is None:
        raise ValueError("DATA_FILES must contain at least one registered dataset key.")
    dataset_list = _ensure_str_list(datasets)
    if not dataset_list:
        raise ValueError("DATA_FILES must contain at least one registered dataset key.")
    target_list: Optional[Sequence[TargetLike]]
    if targets is not None:
        target_list = _ensure_target_list(targets)
    else:
        target_list = None
    from .specs import DATASET_REGISTRY  # local import to avoid cycles

    return DATASET_REGISTRY.resolve_many(
        dataset_list,
        target_list,
        data_root=data_root,
    )


def spec_to_tags(spec: ResolvedDatasetSpec, *, n_inputs: int, n_outputs: int) -> Mapping[str, str]:
    target_channels = spec.target_channels or ()
    alias = spec.target_alias or ("all" if n_outputs == n_inputs else "custom")
    input_channels = spec.input_channels or ()
    dataset_path = spec.relative_path if spec.relative_path is not None else spec.path
    tags = {
        "dataset_path": dataset_path,
        "split_mode": spec.split_mode,
        "target_alias": alias,
        "target_channels": ";".join(target_channels),
        "target_channel_count": str(n_outputs),
        "input_channel_count": str(n_inputs),
        "input_channels": ";".join(input_channels),
    }
    if spec.continuous_channels is not None:
        tags["continuous_input_channel_count"] = str(len(spec.continuous_channels))
        tags["continuous_input_channels"] = ";".join(spec.continuous_channels)
    if spec.discrete_channels is not None:
        tags["discrete_input_channel_count"] = str(len(spec.discrete_channels))
        tags["discrete_input_channels"] = ";".join(spec.discrete_channels)
    return tags


def filter_spec_tags(tags: Mapping[str, str]):
    keep = {
        "dataset_path",
        "split_mode",
        "target_alias",
        "target_channels",
        "target_channel_count",
        "input_channels",
        "input_channel_count",
        "continuous_input_channels",
        "continuous_input_channel_count",
        "discrete_input_channels",
        "discrete_input_channel_count",
    }
    return {k: tags[k] for k in keep if k in tags}


def _ensure_str_list(values: Union[str, Sequence[str]]) -> Sequence[str]:
    if isinstance(values, (list, tuple)):
        return [str(v) for v in values]
    return [str(values)]


def _ensure_target_list(values: Union[TargetLike, Sequence[TargetLike]]) -> Sequence[TargetLike]:
    if isinstance(values, (list, tuple)):
        return list(values)
    return [values]


__all__ = [
    "DatasetSpec",
    "ResolvedDatasetSpec",
    "DatasetRegistry",
    "TargetLike",
    "resolve_with_defaults",
    "spec_to_tags",
    "filter_spec_tags",
]
