from pathlib import Path

import pytest

from pipelines.specs import PipelineSpec


RECIPE_PATHS = (
    "configs/pipelines/randomized_training.yaml",
    "configs/pipelines/adversarial_training.yaml",
    "configs/pipelines/adaptive_robust_loss.yaml",
    "configs/pipelines/fault_augmentation.yaml",
    "configs/pipelines/revin.yaml",
    "configs/pipelines/randomized_smoothing.yaml",
)


def _load_spec(path: str) -> PipelineSpec:
    return PipelineSpec.from_yaml(Path(path))


def _recipe_defaults(spec: PipelineSpec) -> dict[str, object]:
    return {key: entry["default"] for key, entry in spec.recipe_params.items()}


def _recipe_entry_options(entry: dict[str, object]) -> list[object]:
    if "grid" not in entry or entry["grid"] is None:
        return [entry["default"]]
    grid = entry["grid"]
    if isinstance(grid, list):
        return grid or [entry["default"]]
    return [grid]


def _assert_recipe_structure(
    spec: PipelineSpec,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    params = spec.expand_params()
    defaults = _recipe_defaults(spec)

    assert params
    assert len(params) == spec.count_param_combinations()
    assert all(set(candidate) == set(spec.recipe_params) for candidate in params)

    for key, entry in spec.recipe_params.items():
        assert "default" in entry
        if "grid" in entry and isinstance(entry["grid"], list):
            assert entry["grid"]

    rendered_pipeline_id = spec.format_pipeline_id(defaults)
    assert rendered_pipeline_id
    assert "{" not in rendered_pipeline_id
    assert "}" not in rendered_pipeline_id
    return defaults, params


@pytest.mark.parametrize("recipe_path", RECIPE_PATHS)
def test_recipe_specs_keep_defaults_renderable(recipe_path: str):
    spec = _load_spec(recipe_path)

    _assert_recipe_structure(spec)


def test_randomized_training_recipe_renders_default_noise_kwargs():
    spec = _load_spec("configs/pipelines/randomized_training.yaml")
    defaults, _ = _assert_recipe_structure(spec)

    rendered = spec.render_kwargs(spec.datamodule_kwargs, defaults)

    assert rendered["train_noise_std"] == defaults["smoothing_noise_std"]
    assert rendered["train_noise_channels"] == defaults["smoothing_noise_channels"]


def test_adversarial_training_recipe_keeps_raw_step_size_surface():
    spec = _load_spec("configs/pipelines/adversarial_training.yaml")

    _assert_recipe_structure(spec)

    assert "advtrain_step_size" in spec.recipe_params
    assert "advtrain_step_multiplier" not in spec.recipe_params
    assert "{advtrain_step_size}" in spec.pipeline_id
    assert spec.model_kwargs["advtrain_step_size"] == "{advtrain_step_size}"


def test_adaptive_robust_loss_recipe_keeps_loss_runtime_kwargs():
    spec = _load_spec("configs/pipelines/adaptive_robust_loss.yaml")

    _assert_recipe_structure(spec)

    assert spec.model_kwargs["loss"] == "AdaptiveRobustLoss"
    for key in (
        "rloss_alpha_lo",
        "rloss_alpha_hi",
        "rloss_alpha_init",
        "rloss_scale_lo",
        "rloss_scale_init",
        "rloss_param_scope",
    ):
        assert key in spec.recipe_params
        assert key in spec.model_kwargs


def test_fault_augmentation_active_profiles_are_registry_backed():
    spec = _load_spec("configs/pipelines/fault_augmentation.yaml")

    _assert_recipe_structure(spec)

    assert spec.train_fault_profiles
    for profile_name in _recipe_entry_options(
        spec.recipe_params["train_perturbation_profile"]
    ):
        profile = spec.train_fault_profiles[profile_name]
        assert profile["scenarios"]
        assert "packet_loss" in profile["scenarios"]


def test_revin_recipe_contract():
    spec = _load_spec("configs/pipelines/revin.yaml")

    _assert_recipe_structure(spec)

    assert spec.pipeline_method == "revin"
    assert spec.model_hparams_mode == "inherit_baseline"
    assert spec.model_kwargs["use_revin"] is True
    assert "revin_denorm" not in spec.model_kwargs


def test_randomized_smoothing_recipe_wrap_contract():
    spec = _load_spec("configs/pipelines/randomized_smoothing.yaml")
    defaults, params = _assert_recipe_structure(spec)

    assert spec.pipeline_kind == "wrap"
    assert defaults["rs_backbone_method"] == "baseline"
    assert {param["rs_backbone_method"] for param in params} == {"baseline"}
