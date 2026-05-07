from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

from data.data_module import TSDataModule
from models.chronos2 import Chronos2
from utils.parsing import (
    get_tag_or_param_value,
    parse_feature_indices,
    parse_value,
    require_loss_metadata,
    require_mapping,
    require_nonempty_tag_value,
    require_typed_mapping_value,
    require_typed_tag_or_param_value,
)


def _require_target_indices(
    *,
    tags,
    params,
    n_inputs: int,
    n_outputs: int,
    run_id: str,
) -> tuple[int, ...] | None:
    raw_value = get_tag_or_param_value(tags, params, key="target_indices")
    if raw_value is None:
        raise ValueError(
            f"Run {run_id} is missing required 'target_indices' reconstruction metadata."
        )
    try:
        parsed_tuple = parse_value(raw_value, tuple, key="target_indices")
    except ValueError:
        # MLflow params are truncated at 500 chars.  For high-channel datasets
        # (e.g. electricity, 321 channels) the identity-mapping tuple is too
        # long.  If n_inputs == n_outputs the identity mapping is equivalent
        # to target_indices=None.
        if n_inputs == n_outputs:
            logger.warning(
                "Run %s: 'target_indices' param is truncated (MLflow 500-char "
                "limit). Since n_inputs == n_outputs == %d, using "
                "identity mapping (target_indices=None). This is equivalent to "
                "the full tuple (0, 1, ..., %d).",
                run_id,
                n_inputs,
                n_inputs - 1,
            )
            return None
        raise ValueError(
            f"Run {run_id} has truncated 'target_indices' param and "
            f"n_inputs ({n_inputs}) != n_outputs ({n_outputs}), "
            "so identity mapping is not valid."
        )
    return parse_feature_indices(
        parsed_tuple,
        n_features=n_inputs,
        key="target_indices",
        allow_none=True,
    )


def load_chronos2_model(
    client: Any,
    run: Any,
    args: Any,
    datamodule: TSDataModule,
):
    del args

    if not isinstance(datamodule, TSDataModule):
        raise ValueError(
            f"Run {run.info.run_id} requires a TSDataModule for pretrained model reconstruction."
        )

    run_id = run.info.run_id
    run_context = f"Run {run_id}"
    loader_context = f"{run_context} for pretrained loader"
    tags = require_mapping(run.data.tags, key="tags metadata", context=run_context)
    params = require_mapping(run.data.params, key="params metadata", context=run_context)

    loader_kind = require_nonempty_tag_value(tags, key="loader_kind", run_id=run_id)
    if loader_kind != "pretrained":
        raise ValueError(
            f"Run {run_id} has loader_kind='{loader_kind}', expected 'pretrained'."
        )
    architecture = require_nonempty_tag_value(
        tags,
        key="model_architecture",
        run_id=run_id,
    )
    if architecture != "Chronos2":
        raise ValueError(
            f"Chronos2 pretrained loader received unsupported model_architecture "
            f"'{architecture}' for run {run_id}."
        )

    try:
        snapshot_dir = client.download_artifacts(run_id, Chronos2.pretrained_artifact_path)
    except Exception as exc:
        raise ValueError(
            f"Run {run_id} is missing required pretrained snapshot artifact "
            f"'{Chronos2.pretrained_artifact_path}'."
        ) from exc
    if not os.path.isdir(snapshot_dir):
        raise ValueError(
            f"Downloaded pretrained snapshot for run {run_id} is not a directory: "
            f"'{snapshot_dir}'."
        )

    n_inputs = require_typed_mapping_value(
        tags,
        key="input_channel_count",
        expected_type=int,
        context=loader_context,
    )
    n_outputs = require_typed_mapping_value(
        tags,
        key="target_channel_count",
        expected_type=int,
        context=loader_context,
    )
    if n_inputs != datamodule.n_inputs:
        raise ValueError(
            f"Run {run_id} expects {n_inputs} input features but datamodule has "
            f"{datamodule.n_inputs}."
        )
    if n_outputs != datamodule.n_outputs:
        raise ValueError(
            f"Run {run_id} expects {n_outputs} target features but datamodule has "
            f"{datamodule.n_outputs}."
        )

    input_len = require_typed_tag_or_param_value(
        tags,
        params,
        key="d_seq_in",
        expected_type=int,
        context=loader_context,
    )
    target_len = require_typed_tag_or_param_value(
        tags,
        params,
        key="d_seq_out",
        expected_type=int,
        context=loader_context,
    )
    if input_len != datamodule.input_len:
        raise ValueError(
            f"Run {run_id} expects input_len={input_len} but datamodule has "
            f"{datamodule.input_len}."
        )
    if target_len != datamodule.target_len:
        raise ValueError(
            f"Run {run_id} expects target_len={target_len} but datamodule has "
            f"{datamodule.target_len}."
        )

    run_target_indices = _require_target_indices(
        tags=tags,
        params=params,
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        run_id=run_id,
    )
    dm_target_indices = datamodule.target_column_indices
    if dm_target_indices is None:
        normalized_dm_target_indices = None
    else:
        normalized_dm_target_indices = tuple(int(index) for index in dm_target_indices)
    # Normalize identity mapping to None for comparison: the explicit tuple
    # (0, 1, ..., n-1) is semantically identical to None when all input
    # channels are targets.  This matters when the run-side value was
    # normalized from a truncated param but the datamodule supplies the
    # explicit identity tuple.
    identity_tuple = tuple(range(n_inputs)) if n_inputs == n_outputs else None
    if run_target_indices == identity_tuple:
        run_target_indices = None
    if normalized_dm_target_indices == identity_tuple:
        normalized_dm_target_indices = None
    if run_target_indices != normalized_dm_target_indices:
        raise ValueError(
            f"Run {run_id} target_indices {run_target_indices} do not match current "
            f"datamodule target_indices {normalized_dm_target_indices}."
        )

    chronos_model_id = require_typed_tag_or_param_value(
        tags,
        params,
        key="chronos_model_id",
        expected_type=str,
        context=loader_context,
    )
    chronos_model_revision = get_tag_or_param_value(
        tags,
        params,
        key="chronos_model_revision",
    )
    if chronos_model_revision is not None:
        chronos_model_revision = str(chronos_model_revision)
    loss = require_loss_metadata(tags, params, run_id=run_id)

    model = Chronos2(
        chronos_model_id=str(chronos_model_id),
        chronos_model_revision=(
            None if chronos_model_revision is None else str(chronos_model_revision)
        ),
        chronos_snapshot_path=snapshot_dir,
        d_input_features=datamodule.n_inputs,
        d_target_features=datamodule.n_outputs,
        d_seq_in=datamodule.input_len,
        d_seq_out=datamodule.target_len,
        target_indices=normalized_dm_target_indices,
        loss=loss,
    )
    default_root_dir = os.path.dirname(os.path.normpath(snapshot_dir))
    return model, default_root_dir


PRETRAINED_MODEL_LOADERS = {
    "chronos2": load_chronos2_model,
}


__all__ = ["PRETRAINED_MODEL_LOADERS", "load_chronos2_model"]
