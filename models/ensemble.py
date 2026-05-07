import torch
from typing import Iterable, List

from .base_module import BaseLitModule
from utils.parsing import validate_ensemble_combine_method


class Ensemble(BaseLitModule):
    """Post-hoc ensemble wrapper over fixed backbone forecasts."""

    uses_base_optimizer = False

    def __init__(self, backbones: Iterable[BaseLitModule], combine_method: str):
        backbones = list(backbones)
        if not backbones:
            raise ValueError("Ensemble requires at least one backbone model.")
        combine_method = validate_ensemble_combine_method(
            combine_method,
            key="ensemble_combine_method",
        )

        first_model = backbones[0]
        super().__init__(
            d_input_features=first_model.d_input_features,
            d_target_features=first_model.d_target_features,
            d_seq_in=first_model.d_seq_in,
            d_seq_out=first_model.d_seq_out,
            lr_scheduler=False,
            # Lightning deep-copies captured init args into hparams. Ignore
            # live backbone modules and persist only explicit ensemble metadata.
            save_hparams_ignore=("backbones",),
            target_indices=first_model.target_indices,
        )

        self.combine_method = combine_method
        self.backbones = torch.nn.ModuleList()
        architectures: List[str] = []
        self.loss_fn = getattr(first_model, "loss_fn", None)
        for module in backbones:
            module.freeze()
            module.eval()
            # detach from any stale trainer without triggering property access
            if hasattr(module, "_trainer"):
                module._trainer = None
            if hasattr(module, "_logger"):
                module._logger = None
            self.backbones.append(module)
            architectures.append(module.__class__.__name__)

        self.save_hyperparameters({
            "combine_method": combine_method,
            "backbone_count": len(backbones),
            "backbone_architectures": architectures,
        })

    def set_test_mode(self, test_metric):
        super().set_test_mode(test_metric=test_metric)
        for module in self.backbones:
            if hasattr(module, "set_test_mode"):
                module.set_test_mode(test_metric=test_metric)

    def _shared_step(self, x, y):
        preds = []
        losses = []
        for module in self.backbones:
            if hasattr(module, "_shared_step"):
                outputs = module._shared_step(x, y)
                preds.append(outputs["pred"])
                module_loss = outputs.get("loss")
                if module_loss is not None:
                    losses.append(module_loss)
            else:
                preds.append(module(x))

        stacked = torch.stack(preds, dim=0)
        if self.combine_method == "median":
            pred = torch.quantile(stacked, 0.5, dim=0, interpolation="midpoint")
        elif self.combine_method == "mean":
            pred = stacked.mean(dim=0)
        else:
            raise ValueError(
                f"Unsupported ensemble_combine_method '{self.combine_method}'."
            )

        result = {"pred": pred}

        if y is not None:
            if losses:
                result["loss"] = torch.stack(losses, dim=0).mean(dim=0)
            elif self.loss_fn is not None:
                result["loss"] = self.loss_fn(pred, y).mean()

        return result
