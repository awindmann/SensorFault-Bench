# Chronos-2 pretrained forecasting wrapper.
#
# Reference: Ansari et al., 2025
# Paper: https://arxiv.org/abs/2510.15821
# Repo: https://github.com/amazon-science/chronos-forecasting
# Model source: https://huggingface.co/amazon/chronos-2
# Upstream license: Apache-2.0.

from __future__ import annotations

import os
import tempfile

import torch

from models.base_module import BaseLitModule
from utils.parsing import (
    has_explicit_value,
    parse_required_nonempty_string,
    parse_value,
)


def _import_chronos2_pipeline_class():
    try:
        from chronos import Chronos2Pipeline
    except ImportError as exc:
        raise ImportError(
            "Chronos2 requires the 'chronos-forecasting' package and its "
            "transformers/accelerate dependencies to be installed."
        ) from exc
    return Chronos2Pipeline


_UNSUPPORTED_OPTIMIZER_KEYS = (
    "optimizer",
    "lr",
    "scheduler_type",
    "scheduler_factor",
    "scheduler_patience",
    "initial_lr",
    "peak_lr",
    "min_lr",
    "warmup_div",
    "beta1",
    "beta2",
    "weight_decay",
    "eps",
    "grad_clip",
    "grad_clip_after_warmup",
)

_POINT_FORECAST_QUANTILE_LEVELS = [0.5]


class Chronos2(BaseLitModule):
    supports_improvements = False
    uses_base_optimizer = False
    loader_kind = "pretrained"
    max_fit_epochs = 1
    pretrained_artifact_path = "model/pretrained/chronos2"

    def __init__(
        self,
        chronos_model_id: str,
        *,
        chronos_model_revision: str | None = None,
        chronos_snapshot_path: str | None = None,
        loss: str = "MSE",
        **kwargs,
    ) -> None:
        lr_scheduler_raw = kwargs.pop("lr_scheduler", False)
        lr_scheduler_value = parse_value(
            lr_scheduler_raw,
            bool,
            key="lr_scheduler",
        )
        if bool(lr_scheduler_value):
            raise ValueError("Chronos2 does not support lr_scheduler=True.")
        for key in _UNSUPPORTED_OPTIMIZER_KEYS:
            raw_value = kwargs.pop(key, None)
            if has_explicit_value(raw_value):
                raise ValueError(
                    f"Chronos2 does not support optimizer config '{key}'."
                )

        super().__init__(
            chronos_model_id=chronos_model_id,
            chronos_model_revision=chronos_model_revision,
            loss=loss,
            optimizer=None,
            lr=None,
            lr_scheduler=False,
            scheduler_type=None,
            scheduler_factor=None,
            scheduler_patience=None,
            initial_lr=None,
            peak_lr=None,
            min_lr=None,
            warmup_div=None,
            beta1=None,
            beta2=None,
            weight_decay=None,
            eps=None,
            grad_clip=None,
            grad_clip_after_warmup=None,
            **kwargs,
        )
        self.model_architecture = "Chronos2"
        self.automatic_optimization = False
        self.loss_fn = self._build_loss_fn(loss)
        self.chronos_model_id = parse_required_nonempty_string(
            chronos_model_id,
            key="chronos_model_id",
            context="Chronos2",
            disallow_none_token=True,
        )
        if chronos_model_revision is None:
            self.chronos_model_revision = None
        else:
            self.chronos_model_revision = parse_required_nonempty_string(
                chronos_model_revision,
                key="chronos_model_revision",
                context="Chronos2",
                disallow_none_token=True,
            )
        if chronos_snapshot_path is None:
            self._chronos_snapshot_path = None
        else:
            self._chronos_snapshot_path = parse_required_nonempty_string(
                chronos_snapshot_path,
                key="chronos_snapshot_path",
                context="Chronos2",
                disallow_none_token=True,
            )
        if self.target_indices is None and self.d_input_features != self.d_target_features:
            raise ValueError("Chronos2 requires target channels to be present in inputs.")

        # Disable Lightning ModelSummary forward pass — Chronos2's pipeline
        # creates an internal DataLoader with pin_memory=True which crashes
        # when the example input is already on CUDA.
        self.example_input_array = None

        self._chronos_pipeline = None
        self._bound_pipeline_device: str | None = None
        self._load_pipeline()
        self._validate_model_lengths()

    @staticmethod
    def _freeze_loaded_pipeline(pipeline) -> None:
        inner_model = getattr(pipeline, "model", None)
        # Freeze the innermost model if it exists, otherwise freeze the pipeline
        # directly. Avoid freezing both since pipeline.parameters() typically
        # recurses into inner_model.
        target = inner_model if inner_model is not None and inner_model is not pipeline else pipeline
        parameters = getattr(target, "parameters", None)
        if callable(parameters):
            for parameter in parameters():
                parameter.requires_grad_(False)
        if hasattr(target, "eval"):
            target.eval()

    def _load_pipeline(self) -> None:
        if self._chronos_pipeline is not None:
            return
        pipeline_cls = _import_chronos2_pipeline_class()
        if self._chronos_snapshot_path is None:
            source = self.chronos_model_id
            load_kwargs = {}
            if self.chronos_model_revision is not None:
                load_kwargs["revision"] = self.chronos_model_revision
        else:
            source = self._chronos_snapshot_path
            load_kwargs = {"local_files_only": True}
        try:
            pipeline = pipeline_cls.from_pretrained(source, **load_kwargs)
        except Exception as exc:
            raise ValueError(
                f"Failed to load Chronos2Pipeline from '{source}'."
            ) from exc
        if not isinstance(pipeline, pipeline_cls):
            raise ValueError(
                f"Configured source '{source}' did not resolve to a Chronos2Pipeline."
            )
        self._freeze_loaded_pipeline(pipeline)
        self._chronos_pipeline = pipeline

    def _require_pipeline(self):
        pipeline = self._chronos_pipeline
        if pipeline is None:
            raise ValueError("Chronos2 pipeline failed to initialize.")
        return pipeline

    def _validate_model_lengths(self) -> None:
        pipeline = self._require_pipeline()
        try:
            model_context_length = int(pipeline.model_context_length)
        except Exception as exc:
            raise ValueError(
                "Chronos2 pipeline is missing integer model_context_length."
            ) from exc
        try:
            model_prediction_length = int(pipeline.model_prediction_length)
        except Exception as exc:
            raise ValueError(
                "Chronos2 pipeline is missing integer model_prediction_length."
            ) from exc
        if self.d_seq_in > model_context_length:
            raise ValueError(
                f"Chronos2 input_len {self.d_seq_in} exceeds model_context_length "
                f"{model_context_length}."
            )
        if self.d_seq_out > model_prediction_length:
            raise ValueError(
                f"Chronos2 target_len {self.d_seq_out} exceeds model_prediction_length "
                f"{model_prediction_length}."
            )

    def _validate_single_runtime_device(self) -> None:
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return
        strategy = getattr(trainer, "strategy", None)
        if strategy is None:
            raise ValueError("Chronos2 requires trainer.strategy to resolve devices.")
        parallel_devices = getattr(strategy, "parallel_devices", None)
        if parallel_devices is not None:
            resolved = list(parallel_devices)
            if len(resolved) != 1:
                raise ValueError(
                    "Chronos2 supports exactly one runtime device. "
                    f"Resolved devices: {resolved}."
                )
            return
        if not hasattr(strategy, "root_device"):
            raise ValueError("Chronos2 requires trainer.strategy to expose a device.")

    def _bind_pipeline_to_device(self, device: torch.device):
        """Bind the pipeline to the given device and return it."""
        pipeline = self._require_pipeline()
        device_token = str(device)
        if self._bound_pipeline_device == device_token:
            return pipeline

        moved = False
        model = getattr(pipeline, "model", None)
        if model is not None and hasattr(model, "to"):
            model.to(device)
            moved = True
        elif hasattr(pipeline, "to"):
            pipeline.to(device)
            moved = True

        if not moved:
            raise ValueError(
                "Chronos2 pipeline does not expose a supported device binding method."
            )
        self._bound_pipeline_device = device_token
        return pipeline

    def on_fit_start(self) -> None:
        self._validate_single_runtime_device()

    def on_validation_start(self) -> None:
        self._validate_single_runtime_device()

    def training_step(self, batch, _) -> torch.Tensor | None:
        x, y = batch
        outputs = self._shared_step(x, y)
        loss = outputs["loss"]
        if loss is not None:
            self.log("train_loss", loss)
        return loss

    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()
        if self.test_mode:
            return
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return
        if getattr(trainer, "sanity_checking", False):
            return
        trainer.should_stop = True

    def configure_optimizers(self):
        return None

    def on_fit_end(self) -> None:
        if self.test_mode:
            return
        logger = getattr(self, "logger", None)
        if logger is None or not hasattr(logger, "experiment"):
            raise ValueError(
                "Chronos2 requires an MLflow logger to log pretrained snapshot artifacts."
            )
        pipeline = self._require_pipeline()
        with tempfile.TemporaryDirectory(prefix="robust-chronos2-pretrained-") as tmpdir:
            snapshot_dir = os.path.join(tmpdir, "chronos2")
            pipeline.save_pretrained(snapshot_dir)
            logger.experiment.log_artifacts(
                logger.run_id,
                snapshot_dir,
                artifact_path=self.pretrained_artifact_path,
            )

    def _stack_mean_forecasts(
        self,
        mean,
        *,
        batch_size: int,
        feature_count: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if not isinstance(mean, list):
            raise ValueError(
                "Chronos2 predict_quantiles must return mean forecasts as a list of tensors."
            )
        if len(mean) != batch_size:
            raise ValueError(
                f"Chronos2 returned {len(mean)} mean forecasts for batch_size={batch_size}."
            )
        expected_shape = (feature_count, self.d_seq_out)
        for index, item in enumerate(mean):
            if not isinstance(item, torch.Tensor):
                raise ValueError(
                    "Chronos2 mean forecasts must be tensors. "
                    f"Received {type(item).__name__} at batch index {index}."
                )
            if item.shape != expected_shape:
                raise ValueError(
                    f"Chronos2 mean forecast at batch index {index} has shape "
                    f"{tuple(item.shape)}; expected {expected_shape}."
                )
        return torch.stack(mean, dim=0).to(device=device, dtype=dtype)

    def _shared_step(self, x, y):
        pipeline = self._bind_pipeline_to_device(x.device)
        # Chronos2Pipeline.predict() creates an internal DataLoader with
        # pin_memory=True when the model is on CUDA.  Passing CUDA tensors
        # as context causes "cannot pin CUDA tensor" errors.  Move to CPU
        # so the pipeline's DataLoader can pin and transfer to GPU itself.
        context = x.transpose(1, 2).cpu()
        try:
            with torch.no_grad():
                _, mean = pipeline.predict_quantiles(
                    context,
                    prediction_length=self.d_seq_out,
                    quantile_levels=_POINT_FORECAST_QUANTILE_LEVELS,
                    context_length=self.d_seq_in,
                    limit_prediction_length=False,
                    cross_learning=False,
                )
        except Exception as exc:
            raise ValueError("Chronos2 predict_quantiles failed.") from exc

        mean_forecast = self._stack_mean_forecasts(
            mean,
            batch_size=x.size(0),
            feature_count=x.size(2),
            device=x.device,
            dtype=x.dtype,
        )
        y_pred_raw = mean_forecast.transpose(1, 2)
        y_pred = self.project_targets(y_pred_raw)
        loss = self.loss_fn(y_pred, y).mean() if y is not None else None
        return {
            "pred": y_pred,
            "target": y,
            "loss": loss,
        }
