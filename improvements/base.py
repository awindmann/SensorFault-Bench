import tempfile
from dataclasses import dataclass, field
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import mlflow
import yaml
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST

from data.datasets import filter_spec_tags
from pipelines.training import normalize_mlflow_run_name
from utils.artifacts import download_best_checkpoint
from utils.parsing import (
    optional_nonempty_tag_value,
    parse_backbone_run_ids,
    parse_required_nonempty_string,
    require_nonempty_tag_value,
)
from utils.rng import derive_component_seeds


@dataclass
class BackboneReference:
    """Structured source-run reference stored in improvement metadata."""

    run_id: str
    run_name: str
    dataset: str
    model_architecture: str
    pipeline_id: str
    pipeline_method: str
    pipeline_kind: str
    data_config_signature: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "run_id": self.run_id,
            "run_name": self.run_name,
            "dataset": self.dataset,
            "model_architecture": self.model_architecture,
            "pipeline_id": self.pipeline_id,
            "pipeline_method": self.pipeline_method,
            "pipeline_kind": self.pipeline_kind,
            "data_config_signature": self.data_config_signature,
        }

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "BackboneReference":
        if not isinstance(payload, dict):
            raise ValueError(
                "BackboneReference payload must be a dictionary."
            )
        required_keys = {
            "run_id",
            "run_name",
            "dataset",
            "model_architecture",
            "pipeline_id",
            "pipeline_method",
            "pipeline_kind",
            "data_config_signature",
        }
        unsupported_keys = sorted(set(payload) - required_keys)
        if unsupported_keys:
            raise ValueError(
                "BackboneReference payload has unsupported key(s): "
                f"{', '.join(unsupported_keys)}."
            )
        return BackboneReference(
            run_id=parse_required_nonempty_string(
                payload.get("run_id"),
                key="run_id",
                context="BackboneReference payload",
            ),
            run_name=parse_required_nonempty_string(
                payload.get("run_name"),
                key="run_name",
                context="BackboneReference payload",
            ),
            dataset=parse_required_nonempty_string(
                payload.get("dataset"),
                key="dataset",
                context="BackboneReference payload",
            ),
            model_architecture=parse_required_nonempty_string(
                payload.get("model_architecture"),
                key="model_architecture",
                context="BackboneReference payload",
            ),
            pipeline_id=parse_required_nonempty_string(
                payload.get("pipeline_id"),
                key="pipeline_id",
                context="BackboneReference payload",
            ),
            pipeline_method=parse_required_nonempty_string(
                payload.get("pipeline_method"),
                key="pipeline_method",
                context="BackboneReference payload",
            ),
            pipeline_kind=parse_required_nonempty_string(
                payload.get("pipeline_kind"),
                key="pipeline_kind",
                context="BackboneReference payload",
            ),
            data_config_signature=parse_required_nonempty_string(
                payload.get("data_config_signature"),
                key="data_config_signature",
                context="BackboneReference payload",
            ),
        )


@dataclass
class ImprovementSpec:
    """Structured metadata stored alongside an improvement run."""

    parameters: Dict[str, Any] = field(default_factory=dict)
    backbones: List[BackboneReference] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parameters": self.parameters,
            "backbones": [artifact.to_dict() for artifact in self.backbones],
        }

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "ImprovementSpec":
        if not isinstance(payload, dict):
            raise ValueError("ImprovementSpec payload must be a dictionary.")
        required_keys = {"parameters", "backbones"}
        unsupported_keys = sorted(set(payload) - required_keys)
        if unsupported_keys:
            raise ValueError(
                "ImprovementSpec payload has unsupported key(s): "
                f"{', '.join(unsupported_keys)}."
            )
        if "parameters" not in payload:
            raise ValueError("ImprovementSpec payload is missing required key 'parameters'.")
        if "backbones" not in payload:
            raise ValueError("ImprovementSpec payload is missing required key 'backbones'.")
        parameters = payload["parameters"]
        if not isinstance(parameters, dict):
            raise ValueError("ImprovementSpec 'parameters' must be a dictionary.")
        backbone_entries = payload["backbones"]
        if not isinstance(backbone_entries, list):
            raise ValueError("ImprovementSpec 'backbones' must be a list.")
        backbones = [_parse_backbone_entry(entry) for entry in backbone_entries]
        return ImprovementSpec(
            parameters=dict(parameters),
            backbones=backbones,
        )


ImprovementBuilder = Callable[[Any, Any], Tuple[Any, str]]


@dataclass(frozen=True)
class ImprovementRegistration:
    """Registry entry for a concrete improvement strategy."""

    name: str
    recipe_cls: Type["BaseImprovement"]
    builder: ImprovementBuilder


@dataclass(frozen=True)
class WrapExecutionOutcome:
    status: str
    reason: str
    run_id: Optional[str] = None

    def __post_init__(self) -> None:
        allowed = {"executed", "skipped_existing", "skipped_policy", "failed"}
        if self.status not in allowed:
            raise ValueError(
                f"Invalid wrap outcome status '{self.status}'. Allowed: {sorted(allowed)}."
            )
        if not str(self.reason).strip():
            raise ValueError("Wrap outcome reason must be a non-empty string.")
        if self.status == "executed":
            if self.run_id is None:
                raise ValueError(
                    "Wrap outcome with status='executed' requires a non-empty run_id."
                )
            if not str(self.run_id).strip():
                raise ValueError(
                    "Wrap outcome with status='executed' requires a non-empty run_id."
                )
        if self.status != "executed" and self.run_id is not None and not str(self.run_id).strip():
            raise ValueError("Wrap outcome run_id must be non-empty when provided.")


LOADER_KIND_TAG = "loader_kind"
_BACKBONE_REFERENCE_IDENTITY_FIELDS = (
    "dataset",
    "model_architecture",
    "pipeline_id",
    "pipeline_method",
    "pipeline_kind",
    "data_config_signature",
)


def is_missing_run_lookup_error(exc: Exception) -> bool:
    if isinstance(exc, KeyError):
        return True
    if not isinstance(exc, MlflowException):
        return False
    error_code = getattr(exc, "error_code", None)
    return error_code in {
        RESOURCE_DOES_NOT_EXIST,
        "RESOURCE_DOES_NOT_EXIST",
    }


def build_backbone_reference(run) -> BackboneReference:
    """Capture the required source-run identity stored in improvement metadata."""
    run_id = parse_required_nonempty_string(
        getattr(run.info, "run_id", None),
        key="run_id",
        context="Backbone source run",
    )
    tags = getattr(run.data, "tags", None)
    if tags is None:
        raise ValueError(f"Backbone run {run_id} is missing tags.")
    return BackboneReference(
        run_id=run_id,
        run_name=parse_required_nonempty_string(
            tags.get("mlflow.runName"),
            key="mlflow.runName",
            context=f"Backbone run {run_id}",
        ),
        dataset=parse_required_nonempty_string(
            tags.get("dataset"),
            key="dataset",
            context=f"Backbone run {run_id}",
        ),
        model_architecture=parse_required_nonempty_string(
            tags.get("model_architecture"),
            key="model_architecture",
            context=f"Backbone run {run_id}",
        ),
        pipeline_id=parse_required_nonempty_string(
            tags.get("pipeline_id"),
            key="pipeline_id",
            context=f"Backbone run {run_id}",
        ),
        pipeline_method=parse_required_nonempty_string(
            tags.get("pipeline_method"),
            key="pipeline_method",
            context=f"Backbone run {run_id}",
        ),
        pipeline_kind=parse_required_nonempty_string(
            tags.get("pipeline_kind"),
            key="pipeline_kind",
            context=f"Backbone run {run_id}",
        ),
        data_config_signature=parse_required_nonempty_string(
            tags.get("data_config_signature"),
            key="data_config_signature",
            context=f"Backbone run {run_id}",
        ),
    )


def _parse_backbone_entry(
    payload: Dict[str, Any],
) -> BackboneReference:
    if not isinstance(payload, dict):
        raise ValueError("Backbone entry payload must be a dictionary.")
    return BackboneReference.from_dict(payload)


def require_backbone_references(
    spec: ImprovementSpec,
    *,
    run_id: str,
) -> list[BackboneReference]:
    references: list[BackboneReference] = []
    for entry in spec.backbones:
        if isinstance(entry, BackboneReference):
            references.append(entry)
            continue
        raise ValueError(
            f"Wrap run {run_id} has unsupported backbone entry type "
            f"{type(entry).__name__}."
        )
    return references


def resolve_wrap_backbone_references(
    wrap_run,
    spec: ImprovementSpec,
) -> list[BackboneReference]:
    """Validate wrap-lineage tags against stored backbone references."""
    wrap_run_id = parse_required_nonempty_string(
        getattr(wrap_run.info, "run_id", None),
        key="run_id",
        context="Wrap run",
    )
    tags = getattr(wrap_run.data, "tags", None)
    if tags is None:
        raise ValueError(f"Wrap run {wrap_run_id} is missing tags.")
    base_method = require_nonempty_tag_value(
        tags,
        key="base_pipeline_method",
        run_id=wrap_run_id,
    )
    backbone_run_id = optional_nonempty_tag_value(tags, key="backbone_run_id")
    backbone_run_ids = parse_backbone_run_ids(
        tags.get("backbone_run_ids"),
        run_id=wrap_run_id,
    )
    if backbone_run_id is not None and len(backbone_run_ids) == 1:
        if backbone_run_id != backbone_run_ids[0]:
            raise ValueError(
                f"Wrap run {wrap_run_id} has inconsistent backbone_run_id and "
                "backbone_run_ids tags."
            )
    if backbone_run_id is not None and len(backbone_run_ids) > 1:
        raise ValueError(
            f"Wrap run {wrap_run_id} cannot define both backbone_run_id and "
            "multi-run backbone_run_ids."
        )
    expected_run_ids: list[str]
    if backbone_run_id is not None:
        expected_run_ids = [backbone_run_id]
    else:
        expected_run_ids = backbone_run_ids
    if not expected_run_ids:
        raise ValueError(
            f"Wrap run {wrap_run_id} is missing required backbone_run_id/backbone_run_ids "
            "lineage tags."
        )

    references = require_backbone_references(spec, run_id=wrap_run_id)
    metadata_run_ids = [reference.run_id for reference in references]
    if not metadata_run_ids:
        raise ValueError(
            f"Wrap run {wrap_run_id} has no backbone references in improvement/metadata.yaml."
        )
    if metadata_run_ids != expected_run_ids:
        raise ValueError(
            f"Wrap run {wrap_run_id} has lineage backbone run ids {expected_run_ids} "
            f"but improvement/metadata.yaml references {metadata_run_ids}."
        )
    for reference in references:
        if reference.pipeline_method != base_method:
            raise ValueError(
                f"Wrap run {wrap_run_id} expects base_pipeline_method='{base_method}' "
                f"but backbone reference {reference.run_id} stores "
                f"pipeline_method='{reference.pipeline_method}'."
            )
    return references


def _backbone_reference_details(
    reference: BackboneReference,
    *,
    run_name: str | None = None,
) -> str:
    resolved_run_name = reference.run_name
    if run_name is not None and str(run_name).strip():
        resolved_run_name = str(run_name).strip()
    return (
        f"run_name='{resolved_run_name}', dataset='{reference.dataset}', "
        f"model_architecture='{reference.model_architecture}', "
        f"pipeline_id='{reference.pipeline_id}', "
        f"pipeline_method='{reference.pipeline_method}'"
    )


def _optional_live_run_name(tags: Dict[str, Any] | None) -> str | None:
    if tags is None:
        return None
    raw_value = tags.get("mlflow.runName")
    if raw_value is None:
        return None
    normalized = str(raw_value).strip()
    if not normalized:
        return None
    return normalized


def _validate_backbone_reference_identity(
    source_run,
    *,
    wrap_run_id: str,
    reference: BackboneReference,
) -> str | None:
    tags = getattr(source_run.data, "tags", None)
    if tags is None:
        raise ValueError(
            f"Backbone run {reference.run_id} ({_backbone_reference_details(reference)}) "
            "is missing tags required to validate the stored wrap reference."
        )
    mismatches: list[str] = []
    for field_name in _BACKBONE_REFERENCE_IDENTITY_FIELDS:
        live_value = parse_required_nonempty_string(
            tags.get(field_name),
            key=field_name,
            context=f"Backbone run {reference.run_id}",
        )
        stored_value = str(getattr(reference, field_name))
        if live_value != stored_value:
            mismatches.append(
                f"{field_name} stored='{stored_value}' live='{live_value}'"
            )
    if mismatches:
        raise ValueError(
            f"Wrap run {wrap_run_id} references backbone run {reference.run_id} "
            f"({_backbone_reference_details(reference, run_name=_optional_live_run_name(tags))}) "
            "but live run metadata mismatches the stored reference: "
            + ", ".join(mismatches)
            + "."
        )
    return _optional_live_run_name(tags)


def resolve_backbone_source_run(
    client,
    wrap_run,
    reference: BackboneReference,
):
    """Fetch and validate the source run referenced by wrap metadata."""
    wrap_run_id = parse_required_nonempty_string(
        getattr(wrap_run.info, "run_id", None),
        key="run_id",
        context="Wrap run",
    )
    try:
        source_run = client.get_run(reference.run_id)
    except Exception as exc:
        if is_missing_run_lookup_error(exc):
            raise ValueError(
                f"Wrap run {wrap_run_id} references missing backbone run {reference.run_id} "
                f"({_backbone_reference_details(reference)})."
            ) from exc
        raise
    live_run_name = _validate_backbone_reference_identity(
        source_run,
        wrap_run_id=wrap_run_id,
        reference=reference,
    )
    return source_run, live_run_name


def download_backbone_reference_checkpoint(
    client,
    wrap_run,
    reference: BackboneReference,
    *,
    dst_path: str | None = None,
) -> str:
    """Download the authoritative checkpoint for one referenced backbone run."""
    source_run, live_run_name = resolve_backbone_source_run(
        client,
        wrap_run,
        reference,
    )
    try:
        return download_best_checkpoint(client, source_run.info.run_id, dst_path=dst_path)
    except ValueError as exc:
        raise ValueError(
            f"Backbone run {reference.run_id} "
            f"({_backbone_reference_details(reference, run_name=live_run_name)}) "
            "has no checkpoint available under the standard checkpoint contract."
        ) from exc


class BaseImprovement:
    """Common interface for MLflow robustness improvement pipelines."""

    registry_name: str = ""
    loader_kind_key: str = ""
    sort_metric: str = "best_val_loss"
    spec_artifact_path: str = "improvement/metadata.yaml"

    def __init__(self, args):
        self.args = args
        self.top_k = 1  # Number of backbones to select (default: best single model)

    @property
    def allow_rerun(self) -> bool:
        return bool(getattr(self.args, "rerun", False))

    @property
    def method_tag(self) -> str:
        return parse_required_nonempty_string(
            getattr(self, "tag", None),
            key="tag",
            context=self.__class__.__name__,
        )

    @property
    def loader_identifier(self) -> str:
        return parse_required_nonempty_string(
            self.loader_kind_key,
            key="loader_kind_key",
            context=self.__class__.__name__,
        )

    def _require_registry_name(self) -> str:
        if not self.registry_name:
            raise ValueError(
                f"{self.__class__.__name__} must define registry_name to set pipeline_method."
            )
        return self.registry_name

    def should_skip(self, base_runs: List) -> bool:
        has_runs = len(base_runs) >= self.top_k
        return not has_runs

    def filter_base_runs(self, base_runs: List) -> List:
        """Select top-K runs from pre-sorted list. Assumes runs are already sorted by sort_metric."""
        valid_runs = [run for run in base_runs if self.sort_metric in run.data.metrics]
        return valid_runs[: self.top_k]

    def existing_run(self, improvement_runs: List, base_selection: List) -> Optional[str]:
        return None

    def matching_runs(self, improvement_runs: List, base_selection: List) -> List:
        existing = self.existing_run(improvement_runs, base_selection)
        if not existing:
            return []
        return [
            run
            for run in improvement_runs
            if run.info.run_id == existing
            and getattr(run.info, "lifecycle_stage", "active") == "active"
        ]

    def build_run_name(self, architecture: str, base_selection: List) -> str:
        return self.format_run_name(architecture)

    def _lineage_tags(self, base_selection: List) -> Dict[str, str]:
        tags: Dict[str, str] = {}
        if not base_selection:
            raise ValueError("Improvement lineage tagging requires at least one base run.")
        run_ids = [run.info.run_id for run in base_selection]
        if not run_ids:
            raise ValueError("Improvement lineage tagging requires valid base run IDs.")
        if len(run_ids) == 1:
            tags["backbone_run_id"] = str(run_ids[0])
        else:
            tags["backbone_run_ids"] = ",".join(str(rid) for rid in run_ids if rid)
        base_method = getattr(self, "rs_backbone_method", None)
        if base_method is not None:
            if not str(base_method).strip():
                raise ValueError("rs_backbone_method must be a non-empty string.")
        else:
            base_tags = base_selection[0].data.tags
            if base_tags is None:
                raise ValueError(
                    f"Base run {base_selection[0].info.run_id} is missing tags."
                )
            base_method = base_tags.get("pipeline_method")
            if not base_method or not str(base_method).strip():
                raise ValueError(
                    f"Base run {base_selection[0].info.run_id} is missing pipeline_method tag."
                )
        tags["base_pipeline_method"] = str(base_method)
        return tags

    def extra_tags(self, dataset_name: str, architecture: str, base_selection: List) -> Optional[Dict[str, str]]:
        return self._lineage_tags(base_selection)

    @contextmanager
    def start_run(
        self,
        experiment,
        dataset_name: str,
        architecture: str,
        run_name: str,
        extra_tags=None,
        existing_run_id: Optional[str] = None,
    ):
        start_kwargs = {"experiment_id": experiment.experiment_id}
        if existing_run_id:
            start_kwargs["run_id"] = existing_run_id
        else:
            start_kwargs["run_name"] = normalize_mlflow_run_name(run_name)
        with mlflow.start_run(**start_kwargs) as active_run:
            loader_tag_value = self.loader_identifier
            pipeline_kind = "wrap"
            pipeline_id = self.method_tag
            pipeline_method = self._require_registry_name()
            tags = {
                "stage": "improve",
                "dataset": dataset_name,
                "model_architecture": architecture,
                "derived_model_architecture": loader_tag_value,
                LOADER_KIND_TAG: loader_tag_value,
                "robustness_method": pipeline_method,
                "best_model": "false",
                "pipeline_id": pipeline_id,
                "pipeline_method": pipeline_method,
                "pipeline_kind": pipeline_kind,
            }
            if extra_tags:
                tags.update(extra_tags)
            mlflow.set_tags(tags)
            yield active_run

    def run(
        self,
        client,
        experiment,
        dataset_name: str,
        architecture: str,
        base_runs: List,
        improvement_runs: List,
    ) -> WrapExecutionOutcome:
        if self.top_k <= 0:
            raise ValueError(
                f"Invalid top_k={self.top_k} for improvement '{self.method_tag}'. "
                "top_k must be >= 1."
            )
        if len(base_runs) < self.top_k:
            raise ValueError(
                f"Requested top_k={self.top_k} for improvement '{self.method_tag}' but only "
                f"{len(base_runs)} baseline runs available for {architecture} on {dataset_name}."
            )
        if self.should_skip(base_runs):
            return WrapExecutionOutcome(
                status="skipped_policy",
                reason="base_runs_do_not_meet_wrap_policy",
            )
        base_selection = self.filter_base_runs(base_runs)
        if not base_selection:
            print(
                f"No valid base runs with '{self.sort_metric}' metric found for {architecture} on {dataset_name}. "
                "This typically means training runs are incomplete or failed. Skipping."
            )
            return WrapExecutionOutcome(
                status="skipped_policy",
                reason=f"no_valid_base_runs_with_metric:{self.sort_metric}",
            )
        # Validate all selected runs have the required metric
        missing_metric = [run for run in base_selection if self.sort_metric not in run.data.metrics]
        if missing_metric:
            missing_ids = ",".join(run.info.run_id for run in missing_metric)
            return WrapExecutionOutcome(
                status="failed",
                reason=f"selected_base_runs_missing_metric:{self.sort_metric}:{missing_ids}",
            )
        matching = self.matching_runs(improvement_runs, base_selection)
        if matching and not self.allow_rerun:
            run_ids = ", ".join(run.info.run_id for run in matching)
            print(
                f"{self.loader_identifier} for {architecture} on {dataset_name} already logged "
                f"({len(matching)} run(s): {run_ids}). Skipping."
            )
            return WrapExecutionOutcome(
                status="skipped_existing",
                reason=f"matching_runs_exist:{run_ids}",
                run_id=matching[0].info.run_id,
            )
        if matching and self.allow_rerun:
            match_ids = {run.info.run_id for run in matching}
            for run in matching:
                if getattr(run.info, "lifecycle_stage", "active") != "active":
                    continue
                client.delete_run(run.info.run_id)
            if improvement_runs:
                improvement_runs[:] = [
                    run for run in improvement_runs if run.info.run_id not in match_ids
                ]
            print(
                f"Deleted {len(match_ids)} matching run(s) for {self.loader_identifier} "
                f"on {architecture}/{dataset_name}. Logging a fresh run."
            )
        run_name = self.build_run_name(architecture, base_selection)
        tags = self.extra_tags(dataset_name, architecture, base_selection) or {}
        if base_selection:
            exemplar_tags = filter_spec_tags(base_selection[0].data.tags)
            for key, value in exemplar_tags.items():
                tags.setdefault(key, value)
        if base_selection:
            base_tags = base_selection[0].data.tags or {}
            data_config_signature = base_tags.get("data_config_signature")
            if not data_config_signature:
                raise ValueError(
                    f"Base run {base_selection[0].info.run_id} is missing data_config_signature tag."
                )

            def _require_int_tag(key: str) -> int:
                raw = base_tags.get(key)
                if raw is None:
                    raise ValueError(
                        f"Base run {base_selection[0].info.run_id} is missing '{key}' tag."
                    )
                try:
                    return int(raw)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Base run {base_selection[0].info.run_id} has invalid '{key}' tag '{raw}'."
                    ) from exc

            seed_master = _require_int_tag("seed_master")
            seed_data = _require_int_tag("seed_data")
            seed_model = _require_int_tag("seed_model")
            seed_policy = base_tags.get("seed_policy")
            if not seed_policy:
                raise ValueError(
                    f"Base run {base_selection[0].info.run_id} is missing seed_policy tag."
                )

            pipeline_method = self._require_registry_name()
            pipeline_id = self.method_tag
            pipeline_kind = "wrap"
            eval_seed = derive_component_seeds(
                base_seed=seed_master,
                dataset_key=dataset_name,
                data_config_signature=data_config_signature,
                architecture=architecture,
                pipeline_id=pipeline_id,
                pipeline_method=pipeline_method,
                pipeline_kind=pipeline_kind,
            )["eval_seed"]
            tags.update(
                {
                    "data_config_signature": data_config_signature,
                    "seed_master": str(seed_master),
                    "seed_data": str(seed_data),
                    "seed_model": str(seed_model),
                    "seed_eval": str(eval_seed),
                    "seed_policy": str(seed_policy),
                }
            )
        if not matching:
            print(f"Logging {self.loader_identifier} '{run_name}' for dataset '{dataset_name}'.")
        result = self.execute(
            client,
            experiment,
            dataset_name,
            architecture,
            base_selection,
            run_name,
            tags,
            None,
        )
        if result is None:
            return WrapExecutionOutcome(
                status="failed",
                reason="execute_returned_none",
            )
        improvement_runs.append(result)
        return WrapExecutionOutcome(
            status="executed",
            reason="logged_new_wrap_run",
            run_id=result.info.run_id,
        )

    def execute(
        self,
        client,
        experiment,
        dataset_name: str,
        architecture: str,
        base_selection: List,
        run_name: str,
        extra_tags: Optional[Dict[str, str]],
        existing_run_id: Optional[str],
    ):
        with self.start_run(
            experiment,
            dataset_name,
            architecture,
            run_name,
            extra_tags=extra_tags,
            existing_run_id=existing_run_id,
        ) as active_run:
            spec = self.create_spec(
                client=client,
                dataset_name=dataset_name,
                architecture=architecture,
                base_selection=base_selection,
                active_run=active_run,
            )
            if not isinstance(spec, ImprovementSpec):
                raise TypeError(
                    f"Improvement '{self.method_tag}' returned unsupported spec type {type(spec)!r}."
                )
            self.log_spec(spec)
            run_id = active_run.info.run_id
        print(f"Logged {self.loader_identifier} run {run_id} for architecture {architecture}.")
        return client.get_run(run_id)

    def create_spec(
        self,
        client,
        dataset_name: str,
        architecture: str,
        base_selection: List,
        active_run,
    ) -> ImprovementSpec:
        raise NotImplementedError

    def log_spec(self, spec: ImprovementSpec) -> None:
        payload = yaml.safe_dump(spec.to_dict(), sort_keys=False)
        mlflow.log_text(payload, self.spec_artifact_path)

    @staticmethod
    def _sanitize_name_token(token: str) -> str:
        sanitized = (
            str(token)
            .replace(" ", "")
            .replace(".", "-")
            .replace("[", "(")
            .replace("]", ")")
            .replace(",", "-")
        )
        return sanitized

    def format_run_name(self, architecture: str, *suffixes: str) -> str:
        parts = [self.method_tag, architecture, *suffixes]
        cleaned = [self._sanitize_name_token(part) for part in parts if part]
        return "_".join(cleaned)

    def capture_backbone_reference(self, run) -> BackboneReference:
        return build_backbone_reference(run)


def load_improvement_spec(client, run, artifact_path: str = "improvement/metadata.yaml") -> ImprovementSpec:
    """Download and parse the structured spec logged for an improvement run."""

    with tempfile.TemporaryDirectory(prefix="robust-") as tmpdir:
        local_path = client.download_artifacts(run.info.run_id, artifact_path, dst_path=tmpdir)
        with open(local_path, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
    if payload is None:
        raise ValueError(
            f"Improvement run {run.info.run_id} has empty spec artifact '{artifact_path}'."
        )
    return ImprovementSpec.from_dict(payload)
