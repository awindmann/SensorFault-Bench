from typing import Dict, Optional
import hashlib

import torch


_TORCH_GENERATORS: Dict[str, torch.Generator] = {}
_GLOBAL_SEED: Optional[int] = None


def set_seed(seed: int) -> None:
    if seed is None:
        raise ValueError("seed must be set.")
    global _GLOBAL_SEED
    _GLOBAL_SEED = int(seed)
    _TORCH_GENERATORS.clear()


def _get_seed() -> int:
    if _GLOBAL_SEED is None:
        raise RuntimeError(
            "Seed is not set. Call set_seed(args.seed) before using RNG helpers."
        )
    return _GLOBAL_SEED


def torch_generator(device: torch.device) -> torch.Generator:
    key = str(device)
    gen = _TORCH_GENERATORS.get(key)
    if gen is None:
        gen = torch.Generator(device=device)
        gen.manual_seed(_get_seed())
        _TORCH_GENERATORS[key] = gen
    return gen


def derive_seed(base_seed: int, key: str) -> int:
    if base_seed is None:
        raise ValueError("seed must be set.")
    if not key:
        return int(base_seed)
    digest = hashlib.sha256(str(key).encode("utf-8")).hexdigest()
    offset = int(digest[:8], 16)
    return (int(base_seed) + offset) % (2**31 - 1)


def derive_component_seeds(
    *,
    base_seed: int,
    dataset_key: str,
    data_config_signature: str,
    architecture: Optional[str],
    pipeline_id: Optional[str],
    pipeline_method: Optional[str],
    pipeline_kind: Optional[str],
    data_base_seed: Optional[int] = None,
    model_stage: Optional[str] = None,
    eval_variant: Optional[str] = None,
) -> Dict[str, int]:
    if base_seed is None:
        raise ValueError("base_seed must be set.")
    if not dataset_key:
        raise ValueError("dataset_key must be set.")
    data_key = f"data:{dataset_key}:{data_config_signature}"
    model_key = (
        "model:"
        f"{dataset_key}:"
        f"{architecture or ''}:"
        f"{pipeline_id or ''}:"
        f"{pipeline_method or ''}:"
        f"{pipeline_kind or ''}"
    )
    if model_stage:
        model_key = f"{model_key}:{model_stage}"
    eval_key = f"eval:{dataset_key}:{data_config_signature}"
    if eval_variant:
        eval_key = f"{eval_key}:{eval_variant}"

    data_seed_base = base_seed if data_base_seed is None else data_base_seed
    return {
        "data_seed": derive_seed(data_seed_base, data_key),
        "model_seed": derive_seed(base_seed, model_key),
        "eval_seed": derive_seed(base_seed, eval_key),
    }


def derive_tuning_seed(
    *,
    base_seed: int,
    dataset_key: str,
    architecture: str,
    data_config_signature: str,
    pipeline_method: str,
    pipeline_kind: str,
    tuning_strategy: str,
) -> int:
    if base_seed is None:
        raise ValueError("base_seed must be set.")
    required = {
        "dataset_key": dataset_key,
        "architecture": architecture,
        "data_config_signature": data_config_signature,
        "pipeline_method": pipeline_method,
        "pipeline_kind": pipeline_kind,
        "tuning_strategy": tuning_strategy,
    }
    missing = [key for key, value in required.items() if not str(value).strip()]
    if missing:
        raise ValueError(
            "derive_tuning_seed missing required value(s): " + ", ".join(missing)
        )

    tuning_key = (
        "tuning:"
        f"{dataset_key}:"
        f"{architecture}:"
        f"{data_config_signature}:"
        f"{pipeline_method}:"
        f"{pipeline_kind}:"
        f"{tuning_strategy}"
    )
    return derive_seed(base_seed, tuning_key)
