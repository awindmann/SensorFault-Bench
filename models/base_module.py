from typing import Mapping, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.components.adversarial import (
    build_adversarial_channel_mask,
    generate_masked_linf_pgd_examples,
)
from models.components.revin import RevIN

from metrics.loss import (
    BASE_LOSS_NAMES,
    RLOSS_PARAM_KEYS,
    build_loss,
    resolve_stateless_loss,
)
from utils.parsing import (
    AdvtrainConfig,
    parse_advtrain_config,
    parse_feature_indices,
    parse_optimizer_name,
    parse_required_finite_float,
    parse_required_nonnegative_int,
    parse_revin_settings,
    parse_scheduler_type,
    parse_value,
)
from utils.rng import derive_seed


class BaseLitModule(pl.LightningModule):
    """pytorch lightning core module
    This module is the base class for all models.
    It implements shared training and validation behavior plus the common
    model-query interface.
    Canonical benchmark evaluation lives in ``testing.evaluation`` and calls
    ``forward()``, which delegates prediction-only inference to
    ``_shared_step(x, None)``.
    Args:
        d_input_features (int): number of input features per time-step
        d_target_features (int): number of target features per prediction step
        d_seq_in (int): input sequence length
        d_seq_out (int): output sequence length
        target_indices (Optional[Tuple[int, ...]]): indices of target features in input features
        lr (float): learning rate
        lr_scheduler (bool): use of learning rate scheduler
        scheduler_type (str): scheduler family, currently "plateau"
        beta1 (float): beta1 for Adam optimizer
        beta2 (float): beta2 for Adam optimizer
        weight_decay (float): weight decay for Adam optimizer
        eps (float): epsilon for Adam optimizer
        grad_clip (float): gradient clipping norm (optional)
        grad_clip_after_warmup (bool): retained checkpoint hparam. No benchmark scheduler uses warmup
        use_revin (bool): enable RevIN normalization/denormalization
        revin_affine (bool): enable RevIN affine parameters
        revin_denorm (bool): enable RevIN output denormalization
        revin_eps (float): numerical stability for RevIN
    """

    uses_base_optimizer = True

    def __init__(
        self,
        d_input_features=None, d_target_features=None, d_seq_in=None, d_seq_out=None,
        target_indices=None,
        lr=None, lr_scheduler=None,
        optimizer=None,
        scheduler_type=None,
        scheduler_factor=None, scheduler_patience=None,
        initial_lr=None, peak_lr=None, min_lr=None, warmup_div=None,
        beta1=None, beta2=None, weight_decay=None, eps=None,
        grad_clip=None, grad_clip_after_warmup=True,
        use_revin=False, revin_affine=True, revin_denorm=True, revin_eps=1e-5,
        save_hparams_ignore=(),
        **kwargs,
    ):
        super().__init__()

        if d_input_features is None:
            raise ValueError("d_input_features must be provided.")
        if d_target_features is None:
            raise ValueError("d_target_features must be provided.")
        if d_seq_in is None:
            raise ValueError("d_seq_in must be provided.")
        if d_seq_out is None:
            raise ValueError("d_seq_out must be provided.")

        # Keep subclass-specific kwargs so checkpoint reconstruction remains
        # compatible for models that persist extra constructor hparams.
        ignore_names = list(save_hparams_ignore)
        if "save_hparams_ignore" not in ignore_names:
            ignore_names.append("save_hparams_ignore")
        self.save_hyperparameters(
            ignore=ignore_names
        )  # stores hyperparameters in self.hparams and allows logging and checkpointing
        self.d_input_features = d_input_features
        self.d_target_features = d_target_features
        self.d_seq_in = d_seq_in
        self.d_seq_out = d_seq_out
        self.target_indices = parse_feature_indices(
            target_indices,
            n_features=self.d_input_features,
            key="target_indices",
            allow_none=True,
        )
        if self.target_indices is None:
            self.d_output_features = self.d_target_features
        else:
            self.d_output_features = self.d_input_features

        (
            use_revin,
            revin_affine,
            revin_denorm,
            revin_eps_value,
        ) = parse_revin_settings(
            use_revin=use_revin,
            revin_affine=revin_affine,
            revin_denorm=revin_denorm,
            revin_eps=revin_eps,
        )

        self.use_revin = use_revin
        self.revin_affine = revin_affine
        self.revin_denorm = revin_denorm
        self.revin_eps = revin_eps_value
        self.revin: Optional[RevIN] = None
        if self.use_revin:
            if self.target_indices is None:
                raise ValueError(
                    "RevIN requires target_indices to map denormalization to output targets."
                )
            self.revin = RevIN(
                num_features=self.d_input_features,
                eps=self.revin_eps,
                affine=self.revin_affine,
            )

        self.model_architecture = None
        self.example_input_array = torch.zeros(1, self.d_seq_in, self.d_input_features)  # 1 as example batch size

        self.loss_fn: Optional[nn.Module] = None  # set by subclass __init__
        self.test_mode = False  # concerns logging in validation step. Set via self.set_test_mode()
        self.test_metric = None
        self.test_metric_fn = None
        
        self._val_losses = list()
        self._best_val_loss = float("inf")
        self._warmup_steps = None
        self._total_steps = None
        self._scheduler_type = scheduler_type
        _adv_mode = self.hparams.get("adversarial_training_mode")
        if _adv_mode is not None and _adv_mode != "pgd_linf":
            raise ValueError(
                f"Unknown adversarial_training_mode '{_adv_mode}'. "
                "Must be 'pgd_linf'."
            )
        self._advtrain_enabled: bool = _adv_mode == "pgd_linf"
        self._model_seed: Optional[int] = None
        self._adv_attack_generator: Optional[torch.Generator] = None
        self._advtrain_cached_config: Optional[AdvtrainConfig] = None
        self._advtrain_cached_mask: Optional[torch.Tensor] = None

    def _build_loss_fn(self, loss_name: str):
        """Build a loss function from the loss spec and model hparams.

        For stateless losses (MSE, MAE, etc.), returns a callable.
        For AdaptiveRobustLoss, returns a trainable nn.Module with learned
        shape/scale parameters.  Assigning the result to ``self.loss_fn``
        registers trainable modules automatically via nn.Module.__setattr__.
        """
        rloss_kwargs = {}
        for key in RLOSS_PARAM_KEYS:
            val = getattr(self.hparams, key, None)
            if val is not None:
                rloss_kwargs[key] = val
        return build_loss(
            loss_name,
            d_seq_out=self.d_seq_out,
            d_target_features=self.d_target_features,
            **rloss_kwargs,
        )

    def apply_loss_overrides_from_kwargs(
        self,
        model_kwargs: Mapping[str, object],
    ) -> None:
        """Rebuild self.loss_fn from model_kwargs after checkpoint load."""
        loss_name = model_kwargs.get("loss")
        if loss_name is None:
            if any(key in model_kwargs for key in RLOSS_PARAM_KEYS):
                raise ValueError(
                    "rloss_* overrides require an explicit loss override."
                )
            return
        loss_name = str(loss_name)
        is_adaptive = loss_name.strip().upper() == "ADAPTIVEROBUSTLOSS"
        self.hparams.loss = loss_name
        for key in RLOSS_PARAM_KEYS:
            if key in model_kwargs:
                setattr(self.hparams, key, model_kwargs[key])
            elif not is_adaptive:
                setattr(self.hparams, key, None)
        self.loss_fn = self._build_loss_fn(loss_name)

    def _shared_step(self, x, y):
        """Shared model step used by training, validation, and testing.

        Canonical evaluation queries the model through ``forward()``, which
        returns ``_shared_step(x, None)["pred"]``. Implementations must
        therefore support prediction-only calls with ``y=None`` and return a
        dictionary containing at least a prediction ``"pred"`` and a loss
        entry ``"loss"`` (which may be ``None`` when no targets are given).
        """
        raise NotImplementedError("This should be implemented in the model that inherits from BaseLitModule.")

    @torch.no_grad()
    def forward(self, x):
        return self._shared_step(x, None)["pred"]

    def project_targets(self, y_pred: torch.Tensor) -> torch.Tensor:
        if self.target_indices:
            index = torch.as_tensor(self.target_indices, device=y_pred.device, dtype=torch.long)
            return torch.index_select(y_pred, dim=-1, index=index)
        return y_pred

    def _revin_norm_inputs(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_revin:
            return x
        if self.revin is None:
            raise RuntimeError("RevIN is enabled but not initialized.")
        return self.revin(x, mode="norm")

    def _revin_denorm_targets(self, y_pred: torch.Tensor) -> torch.Tensor:
        if not self.use_revin:
            return y_pred
        if not self.revin_denorm:
            return y_pred
        if self.revin is None:
            raise RuntimeError("RevIN is enabled but not initialized.")
        return self.revin(
            y_pred,
            mode="denorm",
            target_indices=self.target_indices,
        )

    def prepare_autoregressive_input(
        self,
        y_step: torch.Tensor,
        reference_step: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if y_step.size(-1) != self.d_target_features:
            raise ValueError(
                f"Autoregressive step received tensor with last dimension {y_step.size(-1)} but expected {self.d_target_features}."
            )
        if not self.target_indices:
            return y_step
        if reference_step is None:
            raise ValueError("reference_step must be provided when target_indices are defined.")
        if reference_step.size(-1) != self.d_input_features:
            raise ValueError(
                f"Reference tensor last dimension {reference_step.size(-1)} does not match expected {self.d_input_features}."
            )
        if reference_step.shape[:-1] != y_step.shape[:-1]:
            raise ValueError(
                "reference_step must match y_step shape on all but the last dimension."
            )
        mapped = reference_step.clone()
        index = torch.as_tensor(self.target_indices, device=y_step.device, dtype=torch.long)
        mapped_flat = mapped.reshape(-1, self.d_input_features)
        y_flat = y_step.reshape(-1, self.d_target_features)
        mapped_flat[:, index] = y_flat
        return mapped
    
    def _per_sample_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-sample loss without reduction.

        Returns a tensor of shape (batch_size,) with one loss value per sample.
        All loss functions in metrics/loss.py already reduce over (seq, feature)
        dimensions and return (batch_size,).
        """
        if self.loss_fn is None:
            raise ValueError(
                f"{self.__class__.__name__} has no loss_fn set. "
                "Subclasses must set self.loss_fn before calling _per_sample_loss."
            )
        result = self.loss_fn(pred, target)
        if result.ndim != 1 or result.shape[0] != pred.shape[0]:
            raise ValueError(
                f"_per_sample_loss expected shape ({pred.shape[0]},) but got "
                f"{tuple(result.shape)}. loss_fn must return one value per sample."
            )
        return result

    def _resolve_advtrain_config(self) -> AdvtrainConfig:
        if self._advtrain_cached_config is not None:
            return self._advtrain_cached_config
        if not self._advtrain_enabled:
            raise ValueError("Adversarial training is not enabled for this module.")
        self._advtrain_cached_config = parse_advtrain_config(
            dict(self.hparams),
            context=f"{self.__class__.__name__} adversarial training",
            loss_value=self.hparams.loss,
        )
        return self._advtrain_cached_config

    def set_model_seed(self, seed: int) -> None:
        if seed is None:
            raise ValueError("seed_model must be provided.")
        self._model_seed = int(seed)
        self._adv_attack_generator = None

    def _get_attack_generator(self) -> torch.Generator:
        if self._model_seed is None:
            raise RuntimeError(
                "Adversarial attack RNG requires an explicit seed_model. "
                "Call set_model_seed(seed_model) before adversarial training."
            )
        if self._adv_attack_generator is None:
            attack_seed = derive_seed(
                self._model_seed,
                f"{self.__class__.__name__}:adversarial_attack",
            )
            self._adv_attack_generator = torch.Generator(device="cpu")
            self._adv_attack_generator.manual_seed(attack_seed)
        return self._adv_attack_generator

    def _build_attack_channel_mask(
        self,
        x: torch.Tensor,
        *,
        attack_channels: str,
        context: str,
    ) -> torch.Tensor:
        trainer = self._trainer
        if trainer is None:
            raise RuntimeError(f"{context} requires a trainer to be attached.")
        dm = trainer.datamodule
        if dm is None:
            raise RuntimeError(f"{context} requires a datamodule to be attached.")
        return build_adversarial_channel_mask(
            n_features=int(x.size(-1)),
            attack_channels=attack_channels,
            input_feature_names=getattr(dm, "input_feature_names", None),
            continuous_channels=getattr(dm, "continuous_channels", None),
            device=x.device,
            dtype=torch.float32,
            context=context,
        )

    def _generate_advtrain_inputs(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        config = self._resolve_advtrain_config()
        if self._advtrain_cached_mask is None:
            self._advtrain_cached_mask = self._build_attack_channel_mask(
                x,
                attack_channels=config.attack_channels,
                context=f"{self.__class__.__name__} adversarial training",
            )
        return generate_masked_linf_pgd_examples(
            model=self,
            step_fn=self._shared_step,
            per_sample_loss_fn=self._per_sample_loss,
            x=x,
            y=y,
            attack_steps=config.attack_steps,
            epsilon=config.epsilon,
            step_size=config.step_size,
            random_start=config.random_start,
            mask=self._advtrain_cached_mask,
            attack_generator=self._get_attack_generator(),
        )

    def _loss_from_outputs(
        self,
        outputs: dict,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Extract or recompute the mean loss from a _shared_step result."""
        loss = outputs["loss"]
        if loss is None:
            loss = self.loss_fn(outputs["pred"], y)
        return loss.mean()

    def _run_adversarial_training_step(
        self,
        y: torch.Tensor,
        *,
        x_adv: torch.Tensor,
        adv_metric_name: str,
    ) -> torch.Tensor:
        loss = self._loss_from_outputs(self._shared_step(x_adv, y), y)
        self.log("train_loss", loss)
        self.log(adv_metric_name, loss)
        return loss

    def training_step(self, batch, _):
        x, y = batch
        if self._advtrain_enabled:
            return self._run_adversarial_training_step(
                y,
                x_adv=self._generate_advtrain_inputs(x, y),
                adv_metric_name="train_loss_adv",
            )
        loss = self._shared_step(x, y)["loss"]
        self.log("train_loss", loss.mean())
        return loss

    def validation_step(self, batch, batch_id):
        x, y = batch
        outputs = self._shared_step(x, y)
        if not self.test_mode:  # log custom loss
            loss = outputs["loss"]
            self.log("val_loss", loss.mean(), logger=True)
            # self.log(f"loss/val", loss.mean(), logger=True)
        else:  # log test metric
            loss = self.test_metric_fn(outputs["pred"], y)  # can differ from loss fn
        self._val_losses.append(loss)
        return loss

    def on_validation_epoch_end(self):
        trainer = getattr(self, "trainer", None)
        if trainer is not None and getattr(trainer, "sanity_checking", False):
            self._val_losses.clear()
            return
        mean_loss = torch.stack(self._val_losses).mean()
        self._val_losses.clear()  # free memory
        if not self.test_mode:  # log custom loss
            self.log("ep_val_loss", mean_loss, prog_bar=True, logger=True)
            # save if this is the best model so far
            if mean_loss < self._best_val_loss:
                self._best_val_loss = mean_loss
                # required for selecting the best run per pipeline variant (and for val-only candidate selection).
                self.log("best_val_loss", mean_loss, logger=True)
            self._log_adaptive_loss_metrics()
        else:  # log test metric
            self.log(f"{self.test_metric}_val", mean_loss, logger=True)

    def _log_adaptive_loss_metrics(self) -> None:
        """Log summary metrics for adaptive loss parameters (alpha, scale)."""
        from metrics.adaptive_robust_loss import AdaptiveRobustLoss

        loss_fn = getattr(self, "loss_fn", None)
        if not isinstance(loss_fn, AdaptiveRobustLoss):
            return
        with torch.no_grad():
            alpha = loss_fn.get_alpha().detach()
            scale = loss_fn.get_scale().detach()
        self.log("rloss_alpha_mean", alpha.mean(), logger=True)
        self.log("rloss_alpha_min", alpha.min(), logger=True)
        self.log("rloss_alpha_max", alpha.max(), logger=True)
        self.log("rloss_scale_mean", scale.mean(), logger=True)
        self.log("rloss_scale_min", scale.min(), logger=True)
        self.log("rloss_scale_max", scale.max(), logger=True)

    def set_test_mode(self, test_metric):
        """Bind validation to an explicit evaluation metric."""
        try:
            self.test_metric_fn = resolve_stateless_loss(test_metric)
        except (KeyError, ValueError) as exc:
            available = ", ".join(sorted(BASE_LOSS_NAMES))
            raise ValueError(
                f"test_metric {test_metric} not recognized. Available base metrics: {available}."
            ) from exc
        self.test_mode = True
        self.test_metric = test_metric

    def test_step(self, batch, batch_id, dataloader_idx=0):
        raise RuntimeError(
            "BaseLitModule.test_step() is unsupported. Use the degradation "
            "evaluator in testing.evaluation instead of trainer.test()."
        )

    def _require_hparam(self, key: str, *, context: str) -> object:
        if key not in self.hparams:
            raise ValueError(
                f"{context} requires hparam '{key}'. "
                "Populate optimizer defaults through configs/defaults.yaml."
            )
        value = self.hparams[key]
        if value is None:
            raise ValueError(
                f"{context} requires non-null hparam '{key}'. "
                "Populate optimizer defaults through configs/defaults.yaml."
            )
        return value

    def configure_optimizers(self):
        context = f"{self.__class__.__name__}.configure_optimizers"
        optimizer_name = parse_optimizer_name(
            self._require_hparam("optimizer", context=context),
            key="optimizer",
            context=context,
        )

        lr_value = parse_required_finite_float(
            self._require_hparam("lr", context=context),
            key="lr",
        )
        beta1 = parse_required_finite_float(
            self._require_hparam("beta1", context=context),
            key="beta1",
        )
        beta2 = parse_required_finite_float(
            self._require_hparam("beta2", context=context),
            key="beta2",
        )
        weight_decay = parse_required_finite_float(
            self._require_hparam("weight_decay", context=context),
            key="weight_decay",
        )
        eps = parse_required_finite_float(
            self._require_hparam("eps", context=context),
            key="eps",
        )
        if beta1 <= 0.0 or beta1 >= 1.0:
            raise ValueError(f"beta1 must satisfy 0 < beta1 < 1; got {beta1}.")
        if beta2 <= 0.0 or beta2 >= 1.0:
            raise ValueError(f"beta2 must satisfy 0 < beta2 < 1; got {beta2}.")
        if weight_decay < 0.0:
            raise ValueError(f"weight_decay must be >= 0; got {weight_decay}.")
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0; got {eps}.")
        self._warmup_steps = None
        self._total_steps = None
        use_scheduler = parse_value(
            self._require_hparam("lr_scheduler", context=context),
            bool,
            key="lr_scheduler",
        )
        optimizer = Adam(
            self.parameters(),
            lr=lr_value,
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=weight_decay,
        )
        if use_scheduler:
            scheduler_type = parse_scheduler_type(
                self._require_hparam("scheduler_type", context=context),
                key="scheduler_type",
                context=context,
            )
            self._scheduler_type = scheduler_type
            scheduler_factor = parse_required_finite_float(
                self._require_hparam("scheduler_factor", context=context),
                key="scheduler_factor",
            )
            scheduler_patience = parse_required_nonnegative_int(
                self._require_hparam("scheduler_patience", context=context),
                key="scheduler_patience",
                context=context,
            )
            scheduler_min_lr = parse_required_finite_float(
                self._require_hparam("min_lr", context=context),
                key="min_lr",
            )
            if scheduler_factor <= 0.0:
                raise ValueError(
                    f"scheduler_factor must be > 0; got {scheduler_factor}."
                )
            if scheduler_min_lr < 0.0:
                raise ValueError(f"min_lr must be >= 0; got {scheduler_min_lr}.")
            scheduler = ReduceLROnPlateau(
                optimizer,
                factor=scheduler_factor,
                patience=scheduler_patience,
                min_lr=scheduler_min_lr,
            )
            return [optimizer], [
                {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "monitor": "ep_val_loss",
                }
            ]
        return optimizer

    def _should_clip_gradients(self) -> bool:
        return True

    def configure_gradient_clipping(
        self,
        optimizer,
        gradient_clip_val=None,
        gradient_clip_algorithm=None,
    ):
        grad_clip = getattr(self.hparams, "grad_clip", None)
        if grad_clip is None:
            return
        grad_clip = float(grad_clip)
        if grad_clip <= 0:
            return
        if not self._should_clip_gradients():
            return
        algorithm = gradient_clip_algorithm or "norm"
        self.clip_gradients(
            optimizer,
            gradient_clip_val=grad_clip,
            gradient_clip_algorithm=algorithm,
        )
