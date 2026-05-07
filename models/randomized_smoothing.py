import math
import torch

from .base_module import BaseLitModule
from utils.parsing import (
    coerce_int,
    parse_optional_nonempty_string,
    parse_required_nonnegative_float,
    parse_required_positive_int,
    validate_trim_alpha,
)
from utils.rng import derive_seed


class RandomizedSmoothing(BaseLitModule):
    """Wrap a trained backbone with test-time randomized smoothing.

    Reference: Rekavandi et al., 2024
    Paper: https://proceedings.neurips.cc/paper_files/paper/2024/hash/f21a76d688be0553c943a6b6c1d4bb1f-Abstract-Conference.html
    Repo: https://github.com/arekavandi/Certified_adv_RRegression
    """

    uses_base_optimizer = False

    def __init__(
        self,
        wrapped_backbone: BaseLitModule,
        noise_std: float,
        sample_count: int,
        trim_alpha: float,
        noise_generator: torch.Generator | None = None,
    ):
        super().__init__(
            d_input_features=wrapped_backbone.d_input_features,
            d_target_features=wrapped_backbone.d_target_features,
            d_seq_in=wrapped_backbone.d_seq_in,
            d_seq_out=wrapped_backbone.d_seq_out,
            lr_scheduler=False,
            save_hparams_ignore=("wrapped_backbone", "noise_generator"),
            target_indices=wrapped_backbone.target_indices,
        )
        noise_std = parse_required_nonnegative_float(
            noise_std,
            key="noise_std",
        )
        sample_count = parse_required_positive_int(
            sample_count,
            key="sample_count",
        )
        trim_alpha = validate_trim_alpha(trim_alpha, sample_count)
        self._backbone = wrapped_backbone.eval()
        self._backbone.freeze()
        if not hasattr(wrapped_backbone, "loss_fn") or wrapped_backbone.loss_fn is None:
            raise ValueError(
                "RandomizedSmoothing requires wrapped_backbone.loss_fn to "
                "recompute loss from the trimmed prediction."
            )
        self.loss_fn = wrapped_backbone.loss_fn
        self.noise_std = noise_std
        self.sample_count = sample_count
        self.trim_alpha = trim_alpha
        self.trim_count = math.floor(self.trim_alpha * self.sample_count)
        self.noise_generator = noise_generator
        self._noise_generators: dict[str, torch.Generator] = {}
        self._bound_noise_sample_ids: tuple[int, ...] | None = None
        self._bound_noise_context: str | None = None
        self.backbone_class = wrapped_backbone.__class__.__name__
        self.hparams["noise_std"] = self.noise_std
        self.hparams["sample_count"] = self.sample_count
        self.hparams["trim_alpha"] = self.trim_alpha

    def set_test_mode(self, test_metric):
        super().set_test_mode(test_metric=test_metric)
        if hasattr(self._backbone, "set_test_mode"):
            self._backbone.set_test_mode(test_metric=test_metric)

    def set_noise_generator(self, generator: torch.Generator | None) -> None:
        self.noise_generator = generator
        self._noise_generators.clear()

    def bind_noise_sample_ids(self, sample_ids, *, context_key: str) -> None:
        if sample_ids is None:
            raise ValueError("RandomizedSmoothing noise sample binding requires sample_ids.")
        context = parse_optional_nonempty_string(
            context_key,
            key="context_key",
            context="RandomizedSmoothing noise sample binding",
            disallow_none_token=True,
        )
        if context is None:
            raise ValueError(
                "RandomizedSmoothing noise sample binding requires non-empty context_key."
            )
        normalized_sample_ids: list[int] = []
        for position, raw_sample_id in enumerate(sample_ids):
            if isinstance(raw_sample_id, bool):
                raise ValueError(
                    "RandomizedSmoothing noise sample binding requires integer "
                    f"sample_ids; got {raw_sample_id!r} at position {position}."
                )
            parsed_sample_id = coerce_int(raw_sample_id)
            if parsed_sample_id is None or int(parsed_sample_id) < 0:
                raise ValueError(
                    "RandomizedSmoothing noise sample binding requires non-negative "
                    f"integer sample_ids; got {raw_sample_id!r} at position {position}."
                )
            normalized_sample_ids.append(int(parsed_sample_id))
        self._bound_noise_sample_ids = tuple(normalized_sample_ids)
        self._bound_noise_context = context

    def clear_noise_sample_ids(self) -> None:
        self._bound_noise_sample_ids = None
        self._bound_noise_context = None

    def bind_eval_context(
        self,
        *,
        input_columns,
        target_columns,
        continuous_channels,
        input_means: torch.Tensor,
        input_stds: torch.Tensor,
    ) -> None:
        # Evaluation always binds dataset metadata through this interface.
        # Rekavandi-style RS perturbs the full model input tensor directly, so
        # no channel mask or destandardization context is needed.
        return None

    def _resolve_noise_generator(self, device: torch.device) -> torch.Generator | None:
        if self.noise_generator is None:
            return None
        gen_device = getattr(self.noise_generator, "device", None)
        if gen_device is None or str(gen_device) == str(device):
            return self.noise_generator
        seed = self.noise_generator.initial_seed()
        key = str(device)
        cached = self._noise_generators.get(key)
        if cached is None or cached.initial_seed() != seed:
            cached = torch.Generator(device=device).manual_seed(seed)
            self._noise_generators[key] = cached
        return cached

    def _bound_row_noise_generators(
        self,
        device: torch.device,
        *,
        batch_size: int,
    ) -> list[torch.Generator] | None:
        if self._bound_noise_sample_ids is None:
            return None
        if self.noise_generator is None:
            raise ValueError(
                "RandomizedSmoothing requires an explicit noise_generator when "
                "noise sample IDs are bound."
            )
        if self._bound_noise_context is None:
            raise ValueError(
                "RandomizedSmoothing bound noise sample IDs are missing the required "
                "context_key."
            )
        if len(self._bound_noise_sample_ids) != int(batch_size):
            raise ValueError(
                "RandomizedSmoothing bound noise sample IDs length does not match the "
                f"current batch size: {len(self._bound_noise_sample_ids)} vs {batch_size}."
            )
        base_seed = int(self.noise_generator.initial_seed())
        return [
            torch.Generator(device=device).manual_seed(
                derive_seed(
                    base_seed,
                    f"randomized_smoothing:{self._bound_noise_context}:sample:{sample_id}",
                )
            )
            for sample_id in self._bound_noise_sample_ids
        ]

    def _shared_step(self, x, y):
        if self.noise_generator is None:
            raise ValueError(
                "RandomizedSmoothing requires an explicit noise_generator; "
                "call set_noise_generator() before evaluation."
            )
        preds = []
        self._backbone = self._backbone.to(x.device)
        bound_row_generators = self._bound_row_noise_generators(
            x.device,
            batch_size=int(x.shape[0]),
        )
        with torch.no_grad():
            for _ in range(self.sample_count):
                if bound_row_generators is None:
                    generator = self._resolve_noise_generator(x.device)
                    eps = torch.randn(
                        x.shape,
                        dtype=x.dtype,
                        device=x.device,
                        generator=generator,
                    )
                else:
                    eps = torch.cat(
                        [
                            torch.randn(
                                (1, *x.shape[1:]),
                                dtype=x.dtype,
                                device=x.device,
                                generator=row_generator,
                            )
                            for row_generator in bound_row_generators
                        ],
                        dim=0,
                    )
                perturbed_x = x + self.noise_std * eps
                outputs = self._backbone._shared_step(perturbed_x, y)
                preds.append(outputs["pred"])
        stacked_preds = torch.stack(preds, dim=0)
        # Coordinate-wise symmetric trimming mirrors the paper's trim-mean
        # reduction, but stays Torch-native on forecast tensors.
        sorted_preds = torch.sort(stacked_preds, dim=0).values
        center_preds = sorted_preds[
            self.trim_count : self.sample_count - self.trim_count
        ]
        pred = center_preds.mean(dim=0)
        result = {"pred": pred}
        if y is not None:
            result["loss"] = self.loss_fn(pred, y).mean()
        return result
