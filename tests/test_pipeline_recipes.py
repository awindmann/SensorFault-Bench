import pytest

from config_loader import load_defaults
from pipelines import recipes
from pipelines import selection as selection_module


def test_selection_re_exports_pipeline_configs_dir():
    assert selection_module.PIPELINE_CONFIGS_DIR == recipes.PIPELINE_CONFIGS_DIR


def test_require_pipeline_method_value_rejects_missing_and_blank():
    for value in (None, "", "   "):
        try:
            recipes.require_pipeline_method_value(
                value,
                context="test_require_pipeline_method_value",
            )
        except ValueError as exc:
            assert "pipeline_method must be a non-empty string" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("Expected ValueError for invalid pipeline_method.")


def test_load_recipe_specs_requires_explicit_recipe_order():
    with pytest.raises(ValueError, match="BENCHMARK_METHODS"):
        recipes.load_recipe_specs()


def test_benchmark_recipe_specs_use_yaml_method_order():
    methods = tuple(load_defaults()["BENCHMARK_METHODS"])
    specs = selection_module.load_benchmark_recipe_specs_for_scope(load_defaults())
    assert len(specs) == len(methods)
    assert [spec.pipeline_method for spec in specs[:3]] == [
        "baseline",
        "randomized_training",
        "adversarial_training",
    ]
    assert "revin_input_only" not in {
        str(spec.pipeline_method).strip() for spec in specs
    }


def test_load_pipeline_spec_for_method_resolves_revin():
    spec = recipes.load_pipeline_spec_for_method("revin")
    assert spec.pipeline_method == "revin"
    assert spec.pipeline_kind == "train"


def test_load_pipeline_spec_for_method_applies_allowed_methods_filter():
    spec = recipes.load_pipeline_spec_for_method(
        "revin",
        allowed_methods={"baseline", "revin"},
    )
    assert spec.pipeline_method == "revin"


def test_load_pipeline_spec_for_method_rejects_method_outside_allowed_methods():
    with pytest.raises(ValueError, match="Known methods: baseline, revin"):
        recipes.load_pipeline_spec_for_method(
            "revin_input_only",
            allowed_methods={"baseline", "revin"},
        )


def test_load_pipeline_spec_for_method_rejects_disabled_revin_input_only():
    with pytest.raises(ValueError, match="Unknown pipeline_method 'revin_input_only'"):
        recipes.load_pipeline_spec_for_method("revin_input_only")


def test_load_pipeline_spec_for_method_rejects_unknown_method():
    try:
        recipes.load_pipeline_spec_for_method("not_a_method")
    except ValueError as exc:
        assert "Unknown pipeline_method 'not_a_method'" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected ValueError for unknown pipeline_method.")
