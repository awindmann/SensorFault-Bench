"""Masked PGD helpers for adversarial training.

Reference: Madry et al., 2018
Paper: https://openreview.net/forum?id=rJzIBfZAb
Repo: https://github.com/MadryLab/mnist_challenge
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Callable, Iterator, Mapping, Sequence

import torch


def _disabled_autocast(device_type: str):
    if device_type in {"cpu", "cuda", "mps"}:
        return torch.autocast(device_type=device_type, enabled=False)
    return nullcontext()


def _module_has_cuda_parameters(params: Sequence[torch.nn.Parameter]) -> bool:
    return any(param.device.type == "cuda" for param in params)


def _restore_cuda_rnn_training_modes(
    module: torch.nn.Module,
    original_training_modes: Mapping[torch.nn.Module, bool],
) -> None:
    # CuDNN-backed RNNs require training mode to backpropagate input gradients.
    for submodule in module.modules():
        if (
            isinstance(submodule, torch.nn.RNNBase)
            and original_training_modes.get(submodule, False)
        ):
            submodule.train(True)


@contextmanager
def freeze_module_for_input_grad(module: torch.nn.Module) -> Iterator[None]:
    params = tuple(module.parameters())
    original_requires_grad = [param.requires_grad for param in params]
    original_training_modes = {
        submodule: submodule.training for submodule in module.modules()
    }
    module.eval()
    if _module_has_cuda_parameters(params):
        _restore_cuda_rnn_training_modes(module, original_training_modes)
    try:
        for param in params:
            param.requires_grad_(False)
        yield
    finally:
        for param, rg in zip(params, original_requires_grad):
            param.requires_grad_(rg)
        for submodule, was_training in original_training_modes.items():
            submodule.train(was_training)


def build_adversarial_channel_mask(
    *,
    n_features: int,
    attack_channels: str,
    input_feature_names: Sequence[str] | None,
    continuous_channels: Sequence[str] | None,
    device: torch.device,
    dtype: torch.dtype,
    context: str,
) -> torch.Tensor:
    if n_features <= 0:
        raise ValueError(f"{context} requires n_features > 0; got {n_features}.")
    if attack_channels == "all":
        return torch.ones(1, 1, n_features, device=device, dtype=dtype)
    if attack_channels != "continuous":
        raise ValueError(
            f"{context} has unsupported attack_channels='{attack_channels}'."
        )

    def _duplicate_names(names: Sequence[str]) -> list[str]:
        seen: set[str] = set()
        duplicates: set[str] = set()
        for name in names:
            if name in seen:
                duplicates.add(name)
                continue
            seen.add(name)
        return sorted(duplicates)

    if input_feature_names is None:
        raise ValueError(
            f"{context} requires datamodule.input_feature_names for continuous-channel attacks."
        )
    if continuous_channels is None:
        raise ValueError(
            f"{context} requires datamodule.continuous_channels for continuous-channel attacks."
        )
    ordered_inputs = tuple(str(name).strip() for name in input_feature_names)
    ordered_continuous = tuple(str(name).strip() for name in continuous_channels)
    if len(ordered_inputs) != n_features:
        raise ValueError(
            f"{context} input_feature_names length {len(ordered_inputs)} does not match "
            f"n_features={n_features}."
        )
    blank_inputs = [idx for idx, name in enumerate(ordered_inputs) if not name]
    if blank_inputs:
        raise ValueError(
            f"{context} input_feature_names contains blank name(s) at index/indices "
            f"{blank_inputs}."
        )
    duplicate_inputs = _duplicate_names(ordered_inputs)
    if duplicate_inputs:
        raise ValueError(
            f"{context} input_feature_names contains duplicate name(s): "
            f"{duplicate_inputs}."
        )
    if len(ordered_continuous) == 0:
        raise ValueError(
            f"{context} requires at least one continuous channel for attack_channels='continuous'."
        )
    blank_continuous = [idx for idx, name in enumerate(ordered_continuous) if not name]
    if blank_continuous:
        raise ValueError(
            f"{context} continuous_channels contains blank name(s) at index/indices "
            f"{blank_continuous}."
        )
    duplicate_continuous = _duplicate_names(ordered_continuous)
    if duplicate_continuous:
        raise ValueError(
            f"{context} continuous_channels contains duplicate name(s): "
            f"{duplicate_continuous}."
        )
    input_positions = {name: idx for idx, name in enumerate(ordered_inputs)}
    missing = sorted(set(ordered_continuous) - set(input_positions))
    if missing:
        raise ValueError(
            f"{context} continuous channel metadata references unknown input channel(s): {missing}."
        )
    mask = torch.zeros(n_features, device=device, dtype=dtype)
    for name in ordered_continuous:
        mask[input_positions[name]] = 1.0
    return mask.view(1, 1, n_features)


def _require_pred_output(
    outputs: Mapping[str, torch.Tensor],
    *,
    context: str,
) -> torch.Tensor:
    if "pred" not in outputs:
        raise ValueError(f"{context} step_fn must return a mapping with key 'pred'.")
    return outputs["pred"]


# ---------------------------------------------------------------------------
# masked L_inf PGD
# ---------------------------------------------------------------------------

def project_masked_linf_delta(
    delta: torch.Tensor,
    *,
    epsilon: float,
    mask: torch.Tensor,
) -> torch.Tensor:
    projected = delta.clamp(min=-float(epsilon), max=float(epsilon))
    return projected * mask


def sample_masked_linf_random_start(
    shape: torch.Size,
    *,
    epsilon: float,
    mask: torch.Tensor,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    delta_cpu = torch.rand(
        shape,
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    )
    delta_cpu = (delta_cpu * 2.0 - 1.0) * float(epsilon)
    delta = delta_cpu.to(device=device, dtype=dtype)
    return delta * mask


def generate_masked_linf_pgd_examples(
    *,
    model: torch.nn.Module,
    step_fn: Callable[[torch.Tensor, torch.Tensor], Mapping[str, torch.Tensor]],
    per_sample_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    attack_steps: int,
    epsilon: float,
    step_size: float,
    random_start: bool,
    mask: torch.Tensor,
    attack_generator: torch.Generator,
) -> torch.Tensor:
    x_inner = x.detach().to(dtype=torch.float32)
    y_inner = y.detach().to(device=x.device, dtype=torch.float32)
    mask_inner = mask.to(device=x.device, dtype=torch.float32)
    if mask_inner.shape != (1, 1, x_inner.size(-1)):
        raise ValueError(
            "PGD mask must have shape (1, 1, n_features); "
            f"received {tuple(mask_inner.shape)}."
        )
    if random_start:
        delta = sample_masked_linf_random_start(
            x_inner.shape,
            epsilon=epsilon,
            mask=mask_inner,
            generator=attack_generator,
            device=x.device,
            dtype=torch.float32,
        )
    else:
        delta = torch.zeros_like(x_inner, dtype=torch.float32)

    with freeze_module_for_input_grad(model), _disabled_autocast(x.device.type):
        for _ in range(int(attack_steps)):
            delta = delta.detach()
            delta.requires_grad_(True)
            outputs = step_fn(x_inner + delta, y_inner)
            pred = _require_pred_output(outputs, context="PGD")
            inner_loss = per_sample_loss_fn(pred, y_inner).mean()
            (grad,) = torch.autograd.grad(inner_loss, delta)
            delta = delta.detach() + float(step_size) * grad.sign() * mask_inner
            delta = project_masked_linf_delta(
                delta,
                epsilon=epsilon,
                mask=mask_inner,
            )

    return (x_inner + delta.detach()).to(device=x.device, dtype=x.dtype)
