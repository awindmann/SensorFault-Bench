from pathlib import Path

import pytest

from pipelines.specs import PipelineSpec


def _write_recipe(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")


def test_from_yaml_rejects_unsupported_keys(tmp_path):
    recipe = tmp_path / "recipe.yaml"
    _write_recipe(
        recipe,
        """
pipeline_id: baseline
pipeline_method: baseline
pipeline_kind: train
recipe_params: {}
model_hparams:
  mode: baseline_grid
unexpected_key: true
""",
    )

    with pytest.raises(ValueError, match="unsupported key\\(s\\): unexpected_key"):
        PipelineSpec.from_yaml(recipe)


def test_from_yaml_requires_model_hparams_mapping(tmp_path):
    recipe = tmp_path / "recipe.yaml"
    _write_recipe(
        recipe,
        """
pipeline_id: baseline
pipeline_method: baseline
pipeline_kind: train
recipe_params: {}
model_hparams: baseline_grid
""",
    )

    with pytest.raises(ValueError, match="model_hparams must be a mapping"):
        PipelineSpec.from_yaml(recipe)


def test_from_yaml_rejects_train_fault_profiles_for_non_fault_augmentation(tmp_path):
    recipe = tmp_path / "recipe.yaml"
    _write_recipe(
        recipe,
        """
pipeline_id: baseline
pipeline_method: baseline
pipeline_kind: train
recipe_params: {}
model_hparams:
  mode: baseline_grid
train_fault_profiles:
  profile_a:
    scenarios: [noise]
""",
    )

    with pytest.raises(
        ValueError,
        match="train_fault_profiles is only supported",
    ):
        PipelineSpec.from_yaml(recipe)


def test_from_yaml_requires_fault_augmentation_profiles(tmp_path):
    recipe = tmp_path / "recipe.yaml"
    _write_recipe(
        recipe,
        """
pipeline_id: fa
pipeline_method: fault_augmentation
pipeline_kind: train
recipe_params: {}
model_hparams:
  mode: inherit_baseline
""",
    )

    with pytest.raises(
        ValueError,
        match="fault_augmentation recipes must define train_fault_profiles",
    ):
        PipelineSpec.from_yaml(recipe)


def test_empty_recipe_param_grid_raises_instead_of_using_default():
    spec = PipelineSpec(
        pipeline_id="baseline",
        pipeline_method="baseline",
        pipeline_kind="train",
        recipe_params={
            "example": {
                "default": 1,
                "grid": [],
            },
        },
        model_hparams_mode="baseline_grid",
    )

    with pytest.raises(ValueError, match="empty grid"):
        spec.expand_params()
    with pytest.raises(ValueError, match="empty grid"):
        spec.count_param_combinations()
