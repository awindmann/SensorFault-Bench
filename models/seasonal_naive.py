import math
import os
import tempfile
import torch

from models.base_module import BaseLitModule


class SeasonalNaive(BaseLitModule):
    """Seasonal naive forecaster.

    Args:
        season_length: number of timesteps in the seasonal cycle.
        loss: loss function name.
    """

    supports_improvements = False
    uses_base_optimizer = False

    def __init__(self, season_length: int, loss: str = "MSE", **kwargs):
        if season_length is None:
            raise ValueError("season_length must be provided.")
        season_length = int(season_length)
        if season_length < 0:
            raise ValueError("season_length must be >= 0.")

        super().__init__(season_length=season_length, loss=loss, **kwargs)
        self.model_architecture = "SeasonalNaive"

        if self.target_indices is None and self.d_input_features != self.d_target_features:
            raise ValueError("SeasonalNaive requires target channels to be present in inputs.")

        self.season_length = season_length
        self.season_length_effective = min(self.season_length, int(self.d_seq_in))
        self.loss_fn = self._build_loss_fn(loss)
        self.automatic_optimization = False
        self._logged_season_lengths = False
        # Ensure checkpoints have a non-empty state dict and an optimizer exists.
        self._dummy_param = torch.nn.Parameter(torch.zeros(1))

    def _log_season_lengths(self) -> None:
        if self._logged_season_lengths:
            return
        if not (self.logger and hasattr(self.logger, "experiment")):
            return
        try:
            self.logger.experiment.log_param(
                self.logger.run_id,
                "season_length",
                int(self.season_length),
            )
            self.logger.experiment.log_param(
                self.logger.run_id,
                "season_length_effective",
                int(self.season_length_effective),
            )
        except Exception:
            pass
        self._logged_season_lengths = True

    def on_fit_start(self):
        self._log_season_lengths()

    def configure_optimizers(self):
        # Zero-lr optimizer on a dummy param keeps Lightning's checkpoint/log_model paths happy.
        return torch.optim.SGD([self._dummy_param], lr=0.0)

    def training_step(self, batch, _):
        x, y = batch
        outputs = self._shared_step(x, y)
        loss = outputs["loss"]
        if loss is not None:
            self.log("train_loss", loss.mean())
        return loss

    def on_fit_end(self):
        super().on_fit_end()
        trainer = getattr(self, "trainer", None)
        logger = getattr(self, "logger", None)
        if trainer is None or logger is None or not hasattr(logger, "experiment"):
            return
        if getattr(trainer, "checkpoint_callback", None) is None:
            return
        with tempfile.TemporaryDirectory(prefix="robust-snaive-ckpt-") as tmpdir:
            ckpt_path = os.path.join(tmpdir, "best.ckpt")
            trainer.save_checkpoint(ckpt_path)
            try:
                logger.experiment.log_artifact(
                    logger.run_id,
                    ckpt_path,
                    artifact_path="model/checkpoints",
                )
            except Exception as exc:
                print(f"SeasonalNaive: failed to log checkpoint artifact: {exc}")

    def _per_feature_mean(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0 or x.size(1) == 0:
            return torch.zeros(
                x.size(0),
                1,
                x.size(2),
                device=x.device,
                dtype=x.dtype,
            )
        return x.mean(dim=1, keepdim=True)

    def _shared_step(self, x, y):
        effective = min(self.season_length, int(self.d_seq_in), int(x.size(1)))
        if effective <= 0:
            template = self._per_feature_mean(x)
            y_pred_raw = template.repeat(1, self.d_seq_out, 1)
        else:
            template = x[:, -effective:, :]
            repeats = int(math.ceil(self.d_seq_out / effective)) if self.d_seq_out > 0 else 1
            y_pred_raw = template.repeat(1, repeats, 1)[:, : self.d_seq_out, :]

        y_pred = self.project_targets(y_pred_raw)
        loss = self.loss_fn(y_pred, y).mean() if y is not None else None
        return {
            "pred": y_pred,
            "target": y,
            "loss": loss,
        }
