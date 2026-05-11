import pytest
import numpy as np
import pandas as pd
import torch
from types import SimpleNamespace

from utils.parsing import (
    assert_no_duplicate_rows,
    BOOTSTRAP_CI_SEMANTICS,
    SHARED_ANCHOR_BOOTSTRAP_CI_SEMANTICS,
    METHOD_DELTA_PAIR_BOOTSTRAP_CI_SEMANTICS,
    DEGRADATION_SCORING_SEMANTICS,
    ROBUSTNESS_RESULTS_COMPLETE_TAG,
    build_noise_channel_mask,
    build_shared_anchor_bootstrap_ci_tag_payload,
    build_selection_perturbation_context_tag_payload,
    build_degradation_eval_context_tag_payload,
    build_winner_selection_provenance_tag_payload,
    build_export_manifest_eval_context,
    build_method_delta_pair_bootstrap_ci_seed_key,
    build_shared_anchor_bootstrap_ci_seed_key,
    build_seeded_eval_input_artifact_prefix,
    build_seeded_degradation_artifact_prefix,
    build_perturbation_idx_name_map,
    build_perturbation_scenarios_signature,
    coerce_int,
    coerce_hparam_value,
    extract_required_typed_hparams,
    shared_anchor_bootstrap_ci_context_matches,
    degradation_eval_context_matches,
    degradation_n_test_samples_meet_policy,
    get_tag_or_param_value,
    infer_hparam_expected_type,
    normalize_yaml_value,
    optional_nonempty_tag_value,
    parse_advtrain_config,
    parse_advtrain_epsilon,
    parse_eval_data_seed,
    parse_improvement_selection_mode,
    parse_robustness_results_complete,
    parse_max_hp_trials_per_model,
    parse_dataset_split_mode,
    parse_bootstrap_ci_confidence_level,
    parse_bootstrap_ci_resamples,
    parse_required_choice,
    parse_required_dropout,
    parse_required_finite_float,
    parse_required_nonnegative_float,
    parse_required_nonnegative_int,
    parse_required_positive_int,
    parse_runtime_precision,
    parse_semicolon_delimited_strings,
    parse_optional_nonempty_string,
    parse_core_figure_registry_config,
    parse_feature_indices,
    parse_perturbation_channel_fraction_max,
    parse_perturbation_idx_name_map,
    parse_perturbation_scenarios_from_signature,
    parse_perturbation_scenarios,
    parse_perturbation_scenarios_signature,
    parse_reference_normalization_anchor_model,
    parse_revin_settings,
    parse_optimizer_name,
    parse_scheduler_type,
    parse_train_fault_profiles,
    parse_train_perturbation_probability,
    parse_train_perturbation_profile,
    parse_train_perturbation_severity_max,
    require_dataframe_columns,
    require_integer_series,
    require_nonempty_string_series,
    resolve_train_perturbation_profile_config,
    parse_scenario_metric_key,
    parse_winner_candidate_tags,
    require_tsmixer_hparams,
    require_perturbation_coupling_params,
    require_perturbation_coupling_tags,
    require_namespace_bool,
    require_namespace_nonempty_string,
    require_namespace_value,
    require_eval_data_seed_tag,
    require_selection_perturbation_context_from_args,
    require_selection_perturbation_context_tags,
    require_shared_anchor_bootstrap_ci_context_from_args,
    require_shared_anchor_bootstrap_ci_context_tags,
    require_degradation_eval_context_from_args,
    require_degradation_eval_context_tags,
    require_winner_selection_provenance_tags,
    require_mapping,
    require_meta_analysis_eval_data_seed_scope_tags,
    require_order_sensitive_perturbation_idx_name_map,
    require_robustness_results_complete_tag,
    require_robustness_scoring_semantics_tag,
    require_stage_tag,
    require_typed_mapping_value,
    require_typed_tag_or_param_value,
    robustness_results_complete_tag_value,
    sample_dataframe_records,
    resolve_effective_eval_data_seed,
    resolve_meta_analysis_eval_data_seed_scope,
    tag_is_truthy,
    validate_noise_channels,
    validate_scoped_raw_display_id_values,
    validate_trim_alpha,
)


def test_require_stage_tag_normalizes_to_lowercase():
    assert require_stage_tag({"stage": " Train "}, run_id="run_1") == "train"


def test_require_stage_tag_raises_when_missing():
    with pytest.raises(ValueError, match="missing required stage tag"):
        require_stage_tag({}, run_id="run_2")


def test_optional_nonempty_tag_value_normalizes_and_handles_missing():
    assert optional_nonempty_tag_value({"k": " value "}, key="k") == "value"
    assert optional_nonempty_tag_value({"k": ""}, key="k") is None
    assert optional_nonempty_tag_value({}, key="k") is None


def test_coerce_int_accepts_numpy_integral_values():
    assert coerce_int(np.int64(7)) == 7


def test_require_namespace_value_preserves_explicit_none():
    args = SimpleNamespace(eval_data_seed=None)

    assert require_namespace_value(args, key="eval_data_seed") is None


def test_require_namespace_bool_parses_required_flag():
    args = SimpleNamespace(full_coverage="true")

    assert require_namespace_bool(args, key="full_coverage") is True


def test_parse_runtime_precision_normalizes_aliases_per_device():
    cpu_precision = parse_runtime_precision(
        16,
        device_type="cpu",
        context="Manual evaluation",
    )
    assert cpu_precision.precision == "bf16-mixed"
    assert cpu_precision.model_dtype is None
    assert cpu_precision.input_dtype is None
    assert cpu_precision.autocast_dtype == torch.bfloat16

    mps_precision = parse_runtime_precision(
        16,
        device_type="mps",
        context="Manual evaluation",
    )
    assert mps_precision.precision == "16-mixed"
    assert mps_precision.autocast_dtype == torch.float16


def test_parse_runtime_precision_normalizes_true_precision_modes():
    fp32_precision = parse_runtime_precision(
        32,
        device_type="cpu",
        context="Manual evaluation",
    )
    assert fp32_precision.precision == "32-true"
    assert fp32_precision.model_dtype == torch.float32
    assert fp32_precision.input_dtype == torch.float32
    assert fp32_precision.autocast_dtype is None

    fp64_precision = parse_runtime_precision(
        "64-true",
        device_type="cpu",
        context="Manual evaluation",
    )
    assert fp64_precision.precision == "64-true"
    assert fp64_precision.model_dtype == torch.float64
    assert fp64_precision.input_dtype == torch.float64
    assert fp64_precision.autocast_dtype is None


def test_require_namespace_nonempty_string_raises_when_missing():
    args = SimpleNamespace()

    with pytest.raises(ValueError, match=r"args\.model is required"):
        require_namespace_nonempty_string(args, key="model")


def test_require_mapping_accepts_mapping_and_rejects_invalid_inputs():
    values = {"k": "v"}
    assert require_mapping(values, key="tags", context="Run run_1") is values
    with pytest.raises(ValueError, match="missing tags"):
        require_mapping(None, key="tags", context="Run run_1")
    with pytest.raises(ValueError, match="non-mapping tags"):
        require_mapping([], key="tags", context="Run run_1")


def test_require_dataframe_columns_rejects_missing_columns():
    with pytest.raises(ValueError, match="missing columns"):
        require_dataframe_columns(
            pd.DataFrame([{"a": 1}]),
            {"a", "b"},
            context="test frame",
        )


def test_sample_dataframe_records_deduplicates_requested_columns():
    df = pd.DataFrame([{"dataset": "d1", "model": "m1"}])

    records = sample_dataframe_records(
        df,
        ["dataset", "dataset", "model"],
    )

    assert records == [{"dataset": "d1", "model": "m1"}]


def test_require_nonempty_string_series_strips_and_rejects_blank_values():
    df = pd.DataFrame(
        [
            {"dataset": " d1 ", "model": "GRU"},
            {"dataset": "   ", "model": "DLinear"},
        ]
    )

    with pytest.raises(ValueError, match="empty 'dataset'"):
        require_nonempty_string_series(
            df,
            "dataset",
            context="trajectory input",
            sample_cols=["dataset", "model"],
        )


def test_require_integer_series_rejects_fractional_values():
    df = pd.DataFrame(
        [
            {"dataset": "d1", "sample_id": 1.0},
            {"dataset": "d1", "sample_id": 1.5},
        ]
    )

    with pytest.raises(ValueError, match="non-integer 'sample_id'"):
        require_integer_series(
            df,
            "sample_id",
            context="sample rows",
            sample_cols=["dataset", "sample_id"],
            min_value=0,
        )


def test_assert_no_duplicate_rows_normalizes_string_keys():
    df = pd.DataFrame(
        [
            {"dataset": "ETTh1", "model": "GRU"},
            {"dataset": " ETTh1 ", "model": "GRU"},
        ]
    )

    with pytest.raises(ValueError, match="Examples"):
        assert_no_duplicate_rows(
            df,
            ["dataset", "model"],
            context="duplicate rows",
        )


def test_get_tag_or_param_value_prefers_tags_over_params():
    assert (
        get_tag_or_param_value({"input_len": "8"}, {"input_len": "4"}, key="input_len")
        == "8"
    )
    assert get_tag_or_param_value({}, {"target_len": "2"}, key="target_len") == "2"
    assert get_tag_or_param_value({}, {}, key="missing") is None


def test_require_typed_mapping_value_requires_present_typed_values():
    assert (
        require_typed_mapping_value(
            {"input_channel_count": "3"},
            key="input_channel_count",
            expected_type=int,
            context="Run run_2",
        )
        == 3
    )
    with pytest.raises(ValueError, match="missing required 'input_channel_count'"):
        require_typed_mapping_value(
            {},
            key="input_channel_count",
            expected_type=int,
            context="Run run_2",
        )


def test_require_typed_tag_or_param_value_prefers_tags_and_rejects_missing():
    assert (
        require_typed_tag_or_param_value(
            {"warmup_div": "10"},
            {"warmup_div": "4"},
            key="warmup_div",
            expected_type=int,
            context="Run run_3",
        )
        == 10
    )
    with pytest.raises(ValueError, match="missing required 'warmup_div'"):
        require_typed_tag_or_param_value(
            {},
            {},
            key="warmup_div",
            expected_type=int,
            context="Run run_3",
        )


def test_parse_core_figure_registry_config_derives_method_order_without_panel_alias():
    parsed = parse_core_figure_registry_config(
        {
            "BASELINE_RANK_PARETO_METRIC": "D_w",
            "CORE_IMPROVEMENT_TRAJECTORY_METHOD": "adversarial_training",
            "CORE_IMPROVEMENT_TRAJECTORY_METRIC": "D_mean",
            "CORE_FIGURE_DATASET_SPEC": [
                "BeijingAir_Tiantan=BeijingAir",
                "traffic=Traffic",
            ],
            "CORE_METHOD_DISPLAY": [
                "adversarial_training=PGD",
                "ensemble=Ens.",
            ],
            "CORE_SCENARIO_DISPLAY_ORDER": ["drift", "noise"],
            "CORE_SCENARIO_DISPLAY": ["drift=Drift", "noise=Noise"],
            "CORE_SCENARIO_GROUPS": ["Value=drift,noise"],
        },
        context="core figure registry",
    )

    assert parsed.dataset_spec == (
        ("BeijingAir_Tiantan", "BeijingAir"),
        ("traffic", "Traffic"),
    )
    assert parsed.method_order == (
        "adversarial_training",
        "ensemble",
    )
    assert parsed.baseline_rank_pareto_metric == "D_w"
    assert parsed.core_improvement_trajectory_method == "adversarial_training"
    assert parsed.core_improvement_trajectory_metric == "D_mean"
    assert not hasattr(parsed, "trajectory_panel_spec")


def test_parse_core_figure_registry_config_rejects_out_of_order_scenario_groups():
    with pytest.raises(
        ValueError,
        match="scenario groups must cover CORE_SCENARIO_DISPLAY_ORDER exactly once and in order",
    ):
        parse_core_figure_registry_config(
            {
                "BASELINE_RANK_PARETO_METRIC": "D_w",
                "CORE_IMPROVEMENT_TRAJECTORY_METHOD": "adversarial_training",
                "CORE_IMPROVEMENT_TRAJECTORY_METRIC": "D_w",
                "CORE_FIGURE_DATASET_SPEC": [
                    "BeijingAir_Tiantan=BeijingAir",
                    "traffic=Traffic",
                ],
                "CORE_METHOD_DISPLAY": [
                    "adversarial_training=PGD",
                    "ensemble=Ens.",
                ],
                "CORE_SCENARIO_DISPLAY_ORDER": ["drift", "noise", "spike"],
                "CORE_SCENARIO_DISPLAY": [
                    "drift=Drift",
                    "noise=Noise",
                    "spike=Spike",
                ],
                "CORE_SCENARIO_GROUPS": [
                    "Value=drift,spike",
                    "Timing=noise",
                ],
            },
            context="core figure registry",
        )


def test_parse_core_figure_registry_config_rejects_removed_policy_keys():
    with pytest.raises(
        ValueError,
        match="has unsupported key\\(s\\): CORE_REQUIRED_FIGURE_TYPES",
    ):
        parse_core_figure_registry_config(
            {
                "BASELINE_RANK_PARETO_METRIC": "D_w",
                "CORE_IMPROVEMENT_TRAJECTORY_METHOD": "adversarial_training",
                "CORE_IMPROVEMENT_TRAJECTORY_METRIC": "D_w",
                "CORE_FIGURE_DATASET_SPEC": [
                    "BeijingAir_Tiantan=BeijingAir",
                    "traffic=Traffic",
                ],
                "CORE_METHOD_DISPLAY": [
                    "adversarial_training=PGD",
                    "ensemble=Ens.",
                ],
                "CORE_SCENARIO_DISPLAY_ORDER": ["drift", "noise"],
                "CORE_SCENARIO_DISPLAY": ["drift=Drift", "noise=Noise"],
                "CORE_SCENARIO_GROUPS": ["Value=drift,noise"],
                "CORE_REQUIRED_FIGURE_TYPES": [
                    "core_baseline_architecture_pareto",
                ],
            },
            context="core figure registry",
        )


def test_parse_core_figure_registry_config_rejects_unknown_selection_method():
    with pytest.raises(
        ValueError,
        match="Unsupported CORE_IMPROVEMENT_TRAJECTORY_METHOD 'unknown_method'",
    ):
        parse_core_figure_registry_config(
            {
                "BASELINE_RANK_PARETO_METRIC": "D_w",
                "CORE_IMPROVEMENT_TRAJECTORY_METHOD": "unknown_method",
                "CORE_IMPROVEMENT_TRAJECTORY_METRIC": "D_w",
                "CORE_FIGURE_DATASET_SPEC": [
                    "BeijingAir_Tiantan=BeijingAir",
                    "traffic=Traffic",
                ],
                "CORE_METHOD_DISPLAY": [
                    "adversarial_training=PGD",
                    "ensemble=Ens.",
                ],
                "CORE_SCENARIO_DISPLAY_ORDER": ["drift", "noise"],
                "CORE_SCENARIO_DISPLAY": ["drift=Drift", "noise=Noise"],
                "CORE_SCENARIO_GROUPS": ["Value=drift,noise"],
            },
            context="core figure registry",
        )


def test_validate_scoped_raw_display_id_values_allows_known_out_of_scope_raw_ids():
    validate_scoped_raw_display_id_values(
        ["BeijingAir_Tiantan", "electricity"],
        raw_ids=("BeijingAir_Tiantan", "traffic"),
        display_mapping={
            "BeijingAir_Tiantan": "BeijingAir",
            "traffic": "Traffic",
        },
        known_raw_ids=("BeijingAir_Tiantan", "traffic", "electricity"),
        context="core figure dataset scope",
        id_label="dataset",
    )


def test_validate_scoped_raw_display_id_values_rejects_unknown_ids():
    with pytest.raises(ValueError, match="unexpected dataset ids"):
        validate_scoped_raw_display_id_values(
            ["BeijingAir_Tiantan", "not_registered"],
            raw_ids=("BeijingAir_Tiantan", "traffic"),
            display_mapping={
                "BeijingAir_Tiantan": "BeijingAir",
                "traffic": "Traffic",
            },
            known_raw_ids=("BeijingAir_Tiantan", "traffic", "electricity"),
            context="core figure dataset scope",
            id_label="dataset",
        )


def test_parse_optional_nonempty_string_normalizes_and_allows_none():
    assert (
        parse_optional_nonempty_string(
            " value ",
            key="pipeline_id",
            context="test context",
        )
        == "value"
    )
    assert (
        parse_optional_nonempty_string(
            None,
            key="pipeline_id",
            context="test context",
        )
        is None
    )


def test_parse_improvement_selection_mode_normalizes_and_rejects_unknown_values():
    assert parse_improvement_selection_mode(" perturbed_worst ") == "perturbed_worst"

    with pytest.raises(ValueError, match="Unsupported improvement_selection_mode"):
        parse_improvement_selection_mode("unknown_mode")


def test_parse_semicolon_delimited_strings_normalizes_and_rejects_empty_entries():
    assert parse_semicolon_delimited_strings(" x ; y ", key="input_channels") == (
        "x",
        "y",
    )
    assert parse_semicolon_delimited_strings("", key="input_channels") == ()
    with pytest.raises(ValueError, match="input_channels\\[1\\]"):
        parse_semicolon_delimited_strings(
            "x;;y",
            key="input_channels",
            context="Run run_4",
        )


def test_parse_optional_nonempty_string_rejects_empty_and_none_token_when_disallowed():
    with pytest.raises(ValueError, match="empty pipeline_id"):
        parse_optional_nonempty_string(
            "  ",
            key="pipeline_id",
            context="test context",
        )
    with pytest.raises(ValueError, match="invalid pipeline_id token 'none'"):
        parse_optional_nonempty_string(
            "none",
            key="pipeline_id",
            context="test context",
            disallow_none_token=True,
        )


def test_parse_reference_normalization_anchor_model_normalizes_whitespace():
    assert (
        parse_reference_normalization_anchor_model(
            "  SeasonalNaive  ",
            context="args",
        )
        == "SeasonalNaive"
    )


def test_parse_reference_normalization_anchor_model_rejects_blank_values():
    with pytest.raises(ValueError, match="empty reference_normalization_anchor_model"):
        parse_reference_normalization_anchor_model(
            "   ",
            context="args",
        )


def test_parse_reference_normalization_anchor_model_rejects_missing_values():
    with pytest.raises(ValueError, match="missing required reference_normalization_anchor_model"):
        parse_reference_normalization_anchor_model(
            None,
            context="args",
        )


def test_parse_reference_normalization_anchor_model_rejects_none_token():
    with pytest.raises(ValueError, match="invalid reference_normalization_anchor_model token 'none'"):
        parse_reference_normalization_anchor_model(
            "none",
            context="args",
        )


def test_parse_reference_normalization_anchor_model_rejects_non_seasonal_naive_values():
    with pytest.raises(ValueError, match="Expected 'SeasonalNaive'"):
        parse_reference_normalization_anchor_model(
            "DLinear",
            context="args",
        )


def test_parse_advtrain_epsilon_accepts_positive_float():
    assert parse_advtrain_epsilon("0.2") == 0.2


def test_parse_advtrain_config_accepts_valid_step_size():
    cfg = parse_advtrain_config(
        {
            "advtrain_epsilon": 0.2,
            "advtrain_step_size": 0.1,
            "advtrain_attack_steps": 5,
            "advtrain_random_start": True,
            "advtrain_attack_channels": "continuous",
        }
    )

    assert cfg.epsilon == 0.2
    assert cfg.step_size == 0.1
    assert cfg.attack_steps == 5
    assert cfg.attack_channels == "continuous"


def test_parse_advtrain_config_requires_step_size():
    with pytest.raises(ValueError, match="advtrain_step_size is required"):
        parse_advtrain_config(
            {
                "advtrain_epsilon": 0.2,
                "advtrain_attack_steps": 5,
                "advtrain_random_start": True,
                "advtrain_attack_channels": "all",
            }
        )


def test_parse_train_perturbation_profile_requires_underscore_safe_key():
    assert parse_train_perturbation_profile("holdout_simple") == "holdout_simple"
    with pytest.raises(ValueError, match="underscore-safe"):
        parse_train_perturbation_profile("mean-bias")


def test_parse_train_fault_profiles_canonicalizes_profile_membership():
    parsed = parse_train_fault_profiles(
        {
            "holdout": {
                "scenarios": ["trimming_varying", "linear_drift", "scaling"],
            },
        },
        registry_names=(
            "linear_drift",
            "nonlinear_drift",
            "scaling",
            "time_varying_scaling",
            "trimming_constant",
            "trimming_varying",
        ),
    )

    assert parsed["holdout"] == ("linear_drift", "scaling", "trimming_varying")


def test_parse_train_fault_profiles_rejects_unknown_scenarios():
    with pytest.raises(ValueError, match="unknown scenario"):
        parse_train_fault_profiles(
            {
                "bad_profile": {
                    "scenarios": ["linear_drift", "unknown_fault"],
                }
            },
            registry_names=("linear_drift", "scaling", "trimming_constant"),
        )


def test_resolve_train_perturbation_profile_config_canonicalizes_and_validates_signature():
    profile, scenarios, signature = resolve_train_perturbation_profile_config(
        "holdout_simple",
        profiles={
            "holdout_simple": (
                "linear_drift",
                "scaling",
                "trimming_constant",
            ),
        },
        registry_names=(
            "linear_drift",
            "nonlinear_drift",
            "scaling",
            "time_varying_scaling",
            "trimming_constant",
            "trimming_varying",
        ),
        scenarios=("trimming_constant", "linear_drift", "scaling"),
        scenarios_signature='["linear_drift","scaling","trimming_constant"]',
    )

    assert profile == "holdout_simple"
    assert scenarios == ("linear_drift", "scaling", "trimming_constant")
    assert signature == '["linear_drift","scaling","trimming_constant"]'


def test_train_fault_numeric_parsers_reject_non_finite_values():
    with pytest.raises(ValueError, match="must be finite"):
        parse_perturbation_channel_fraction_max(float("nan"))
    with pytest.raises(ValueError, match="must be finite"):
        parse_train_perturbation_probability(float("inf"))
    with pytest.raises(ValueError, match="must be finite"):
        parse_train_perturbation_severity_max(float("nan"))


def test_parse_revin_settings_accepts_valid_values():
    assert parse_revin_settings(
        use_revin=True,
        revin_affine=False,
        revin_denorm=False,
        revin_eps="1e-5",
    ) == (True, False, False, 1e-5)


@pytest.mark.parametrize(
    "kwargs, expected_message",
    [
        (
            {
                "use_revin": "true",
                "revin_affine": True,
                "revin_denorm": True,
                "revin_eps": 1e-5,
            },
            "use_revin must be a bool",
        ),
        (
            {
                "use_revin": True,
                "revin_affine": "true",
                "revin_denorm": True,
                "revin_eps": 1e-5,
            },
            "revin_affine must be a bool",
        ),
        (
            {
                "use_revin": True,
                "revin_affine": True,
                "revin_denorm": "false",
                "revin_eps": 1e-5,
            },
            "revin_denorm must be a bool",
        ),
        (
            {
                "use_revin": True,
                "revin_affine": True,
                "revin_denorm": True,
                "revin_eps": "bad",
            },
            "revin_eps must be a positive float",
        ),
        (
            {
                "use_revin": True,
                "revin_affine": True,
                "revin_denorm": True,
                "revin_eps": 0.0,
            },
            "revin_eps must be > 0",
        ),
    ],
)
def test_parse_revin_settings_rejects_invalid_values(kwargs, expected_message):
    with pytest.raises(ValueError, match=expected_message):
        parse_revin_settings(**kwargs)


def test_validate_noise_channels_accepts_all_supported_scopes():
    assert validate_noise_channels(" target_only ") == "target_only"
    assert validate_noise_channels("continuous") == "continuous"
    assert validate_noise_channels("all") == "all"


def test_validate_noise_channels_rejects_unknown_scope():
    with pytest.raises(ValueError, match="Unsupported noise_channels"):
        validate_noise_channels("targets")


def test_validate_trim_alpha_accepts_supported_range_with_real_trimming():
    assert validate_trim_alpha("0.35", 100) == pytest.approx(0.35)


@pytest.mark.parametrize(
    ("alpha", "sample_count", "expected_message"),
    [
        (0.0, 100, "must satisfy 0 < rs_trim_alpha < 0.5"),
        (0.5, 100, "must satisfy 0 < rs_trim_alpha < 0.5"),
        (float("nan"), 100, "must be finite"),
        (float("inf"), 100, "must be finite"),
        (0.1, 5, "trims zero samples per tail"),
    ],
)
def test_validate_trim_alpha_rejects_invalid_configurations(
    alpha,
    sample_count,
    expected_message,
):
    with pytest.raises(ValueError, match=expected_message):
        validate_trim_alpha(alpha, sample_count)


def test_parse_required_nonnegative_float_rejects_negative_values():
    with pytest.raises(ValueError, match="noise_std must be >= 0"):
        parse_required_nonnegative_float(-0.1, key="noise_std")


def test_build_noise_channel_mask_target_only_uses_input_target_intersection():
    mask = build_noise_channel_mask(
        input_columns=("a", "b", "c"),
        target_columns=("c", "z"),
        continuous_channels=("a", "b"),
        noise_channels="target_only",
    )

    assert torch.equal(mask, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32))


def test_build_noise_channel_mask_continuous_rejects_missing_channels():
    with pytest.raises(ValueError, match="not present in model inputs"):
        build_noise_channel_mask(
            input_columns=("a", "b"),
            target_columns=("a",),
            continuous_channels=("c",),
            noise_channels="continuous",
        )


def test_tag_is_truthy_accepts_only_true_tokens():
    assert tag_is_truthy({"flag": "true"}, key="flag") is True
    assert tag_is_truthy({"flag": " YeS "}, key="flag") is True
    assert tag_is_truthy({"flag": "false"}, key="flag") is False
    assert tag_is_truthy({"flag": "unexpected"}, key="flag") is False
    assert tag_is_truthy({}, key="flag") is False


def test_parse_winner_candidate_tags_returns_none_when_absent():
    assert parse_winner_candidate_tags({}, run_id="run_3") is None


def test_parse_winner_candidate_tags_raises_when_only_one_tag_present():
    with pytest.raises(ValueError, match="missing backbone_current"):
        parse_winner_candidate_tags({"best_model": "true"}, run_id="run_4")


def test_parse_winner_candidate_tags_raises_on_invalid_boolean_value():
    with pytest.raises(ValueError, match="invalid best_model tag"):
        parse_winner_candidate_tags(
            {"best_model": "maybe", "backbone_current": "true"},
            run_id="run_5",
        )


def test_parse_robustness_results_complete_normalizes_bool_values():
    assert parse_robustness_results_complete(" true ") is True
    assert parse_robustness_results_complete("false") is False
    with pytest.raises(ValueError, match="invalid robustness_results_complete"):
        parse_robustness_results_complete("stale")


def test_require_robustness_results_complete_tag_requires_present_tag():
    tags = {
        ROBUSTNESS_RESULTS_COMPLETE_TAG: robustness_results_complete_tag_value(
            complete=True
        )
    }
    assert require_robustness_results_complete_tag(tags, run_id="run_gc") is True
    assert (
        require_robustness_results_complete_tag(
            {
                ROBUSTNESS_RESULTS_COMPLETE_TAG: robustness_results_complete_tag_value(
                    complete=False
                )
            },
            run_id="run_gc_incomplete",
        )
        is False
    )
    with pytest.raises(ValueError, match="missing required robustness_results_complete tag"):
        require_robustness_results_complete_tag({}, run_id="run_gc_missing")


def test_require_selection_perturbation_context_from_args_builds_expected_context():
    args = SimpleNamespace(
        perturbation_channel_fraction_max=0.5,
        perturbation_scenarios=["drift", "noise"],
    )

    parsed = require_selection_perturbation_context_from_args(
        args,
        context="args",
    )

    assert parsed["selection_metric_semantics"] == "perturbed_validation_error"
    assert parsed["selection_perturbation_channel_fraction_max"] == pytest.approx(0.5)
    assert (
        parsed["selection_perturbation_scenarios_signature"]
        == "[\"drift\",\"noise\"]"
    )


def test_require_selection_perturbation_context_from_args_rejects_non_finite_fraction():
    args = SimpleNamespace(
        perturbation_channel_fraction_max=float("nan"),
        perturbation_scenarios=["noise"],
    )

    with pytest.raises(ValueError, match="perturbation_channel_fraction_max"):
        require_selection_perturbation_context_from_args(
            args,
            context="args",
        )


def test_require_selection_perturbation_context_tags_validates_expected_context():
    tags = build_selection_perturbation_context_tag_payload(
        {
            "selection_metric_semantics": "perturbed_validation_error",
            "selection_perturbation_channel_fraction_max": 0.5,
            "selection_perturbation_scenarios_signature": "[\"drift\",\"noise\"]",
        }
    )

    parsed = require_selection_perturbation_context_tags(
        tags,
        run_id="run_selection_context",
        expected_context={
            "selection_metric_semantics": "perturbed_validation_error",
            "selection_perturbation_channel_fraction_max": 0.5,
            "selection_perturbation_scenarios_signature": "[\"drift\",\"noise\"]",
        },
    )

    assert parsed["selection_metric_semantics"] == "perturbed_validation_error"
    assert parsed["selection_perturbation_channel_fraction_max"] == pytest.approx(0.5)


def test_require_selection_perturbation_context_tags_raises_when_signature_missing():
    tags = {
        "selection_metric_semantics": "perturbed_validation_error",
        "selection_perturbation_channel_fraction_max": "0.5",
    }

    with pytest.raises(ValueError, match="selection_perturbation_scenarios_signature"):
        require_selection_perturbation_context_tags(
            tags,
            run_id="run_selection_missing_signature",
        )


def test_require_selection_perturbation_context_tags_rejects_unknown_semantics():
    tags = {
        "selection_metric_semantics": "clean_validation_error",
        "selection_perturbation_channel_fraction_max": "0.5",
        "selection_perturbation_scenarios_signature": "[\"noise\"]",
    }

    with pytest.raises(ValueError, match="selection_metric_semantics"):
        require_selection_perturbation_context_tags(
            tags,
            run_id="run_selection_bad_semantics",
        )


def test_require_selection_perturbation_context_tags_rejects_unsupported_expected_key():
    tags = build_selection_perturbation_context_tag_payload(
        {
            "selection_metric_semantics": "perturbed_validation_error",
            "selection_perturbation_channel_fraction_max": 0.5,
            "selection_perturbation_scenarios_signature": "[\"noise\"]",
        }
    )

    with pytest.raises(ValueError, match="Unsupported selection perturbation context key"):
        require_selection_perturbation_context_tags(
            tags,
            run_id="run_selection_bad_expected_key",
            expected_context={"selection_unknown_key": "value"},
        )


def test_require_winner_selection_provenance_tags_accepts_clean_provenance():
    tags = build_winner_selection_provenance_tag_payload(
        {
            "winner_selection_mode": "clean",
            "winner_selection_metric_name": "best_val_loss",
        }
    )

    parsed = require_winner_selection_provenance_tags(
        tags,
        run_id="run_winner_clean",
        expected_context={
            "winner_selection_mode": "clean",
            "winner_selection_metric_name": "best_val_loss",
        },
    )

    assert parsed["winner_selection_mode"] == "clean"
    assert parsed["winner_selection_metric_name"] == "best_val_loss"


def test_require_winner_selection_provenance_tags_accepts_perturbed_provenance():
    tags = build_winner_selection_provenance_tag_payload(
        {
            "winner_selection_mode": "perturbed_worst",
            "winner_selection_metric_name": "MSE_pert_ws_val",
            "winner_selection_metric_semantics": "perturbed_validation_error",
            "winner_selection_perturbation_channel_fraction_max": 0.5,
            "winner_selection_perturbation_scenarios_signature": "[\"drift\",\"noise\"]",
        }
    )

    parsed = require_winner_selection_provenance_tags(
        tags,
        run_id="run_winner_perturbed",
        expected_context={
            "winner_selection_mode": "perturbed_worst",
            "winner_selection_metric_name": "MSE_pert_ws_val",
            "winner_selection_metric_semantics": "perturbed_validation_error",
            "winner_selection_perturbation_channel_fraction_max": 0.5,
            "winner_selection_perturbation_scenarios_signature": "[\"drift\",\"noise\"]",
        },
    )

    assert parsed["winner_selection_mode"] == "perturbed_worst"
    assert parsed["winner_selection_metric_name"] == "MSE_pert_ws_val"
    assert parsed["winner_selection_metric_semantics"] == "perturbed_validation_error"


def test_require_winner_selection_provenance_tags_rejects_missing_metric_name():
    with pytest.raises(ValueError, match="winner_selection_metric_name"):
        require_winner_selection_provenance_tags(
            {"winner_selection_mode": "clean"},
            run_id="run_winner_missing_metric",
        )


def test_require_winner_selection_provenance_tags_rejects_mismatched_expected_mode():
    tags = build_winner_selection_provenance_tag_payload(
        {
            "winner_selection_mode": "clean",
            "winner_selection_metric_name": "best_val_loss",
        }
    )

    with pytest.raises(ValueError, match="expected winner-selection provenance"):
        require_winner_selection_provenance_tags(
            tags,
            run_id="run_winner_mode_mismatch",
            expected_context={
                "winner_selection_mode": "perturbed_worst",
                "winner_selection_metric_name": "MSE_pert_ws_val",
                "winner_selection_metric_semantics": "perturbed_validation_error",
                "winner_selection_perturbation_channel_fraction_max": 0.5,
                "winner_selection_perturbation_scenarios_signature": "[\"noise\"]",
            },
        )


def test_require_degradation_eval_context_from_args_builds_expected_context():
    args = SimpleNamespace(
        test_metric="MSE",
        n_test_samples=10000,
        perturbation_channel_fraction_max=0.5,
        perturbation_scenarios=["drift", "noise"],
        strict_iid=False,
    )

    parsed = require_degradation_eval_context_from_args(
        args,
        eval_data_seed=17,
        context="args",
    )

    assert parsed["robustness_scoring_semantics"] == DEGRADATION_SCORING_SEMANTICS
    assert parsed["test_metric"] == "MSE"
    assert parsed["eval_data_seed"] == 17
    assert parsed["n_test_samples"] == 10000
    assert parsed["perturbation_channel_fraction_max"] == pytest.approx(0.5)
    assert parsed["perturbation_scenarios_signature"] == "[\"drift\",\"noise\"]"
    assert parsed["perturbation_scenarios_count"] == 2
    assert parsed["perturbation_idx_name_map"] == {0: "drift", 1: "noise"}


def test_build_degradation_eval_context_tag_payload_stringifies_normalized_values():
    payload = build_degradation_eval_context_tag_payload(
        {
            "robustness_scoring_semantics": DEGRADATION_SCORING_SEMANTICS,
            "test_metric": "MSE",
            "eval_data_seed": 17,
            "n_test_samples": 10000,
            "perturbation_channel_fraction_max": 0.5,
            "perturbation_scenarios_signature": "[\"drift\",\"noise\"]",
            "perturbation_scenarios_count": 2,
            "perturbation_idx_name_map": {0: "drift", 1: "noise"},
        },
        context_name="degradation_eval_context",
    )

    assert payload == {
        "robustness_scoring_semantics": DEGRADATION_SCORING_SEMANTICS,
        "test_metric": "MSE",
        "eval_data_seed": "17",
        "n_test_samples": "10000",
        "perturbation_channel_fraction_max": "0.5",
        "perturbation_scenarios_signature": "[\"drift\",\"noise\"]",
        "perturbation_scenarios_count": "2",
        "perturbation_idx_name_map": "{\"0\":\"drift\",\"1\":\"noise\"}",
    }


def test_build_degradation_eval_context_tag_payload_can_omit_eval_data_seed():
    payload = build_degradation_eval_context_tag_payload(
        {
            "robustness_scoring_semantics": DEGRADATION_SCORING_SEMANTICS,
            "test_metric": "MSE",
            "n_test_samples": 10000,
            "perturbation_channel_fraction_max": 0.5,
            "perturbation_scenarios_signature": "[\"drift\",\"noise\"]",
            "perturbation_scenarios_count": 2,
            "perturbation_idx_name_map": {0: "drift", 1: "noise"},
        },
        context_name="degradation_eval_context",
        include_eval_data_seed=False,
    )

    assert "eval_data_seed" not in payload
    assert payload["n_test_samples"] == "10000"


def test_build_degradation_eval_context_tag_payload_can_ignore_optional_eval_data_seed():
    payload = build_degradation_eval_context_tag_payload(
        {
            "robustness_scoring_semantics": DEGRADATION_SCORING_SEMANTICS,
            "test_metric": "MSE",
            "eval_data_seed": 23,
            "n_test_samples": 10000,
            "perturbation_channel_fraction_max": 0.5,
            "perturbation_scenarios_signature": "[\"drift\",\"noise\"]",
            "perturbation_scenarios_count": 2,
            "perturbation_idx_name_map": {0: "drift", 1: "noise"},
        },
        context_name="degradation_eval_context",
        include_eval_data_seed=False,
        validate_optional_eval_data_seed=False,
    )

    assert "eval_data_seed" not in payload


def test_require_degradation_eval_context_tags_validates_order_sensitive_idx_map():
    tags = {
        "robustness_scoring_semantics": DEGRADATION_SCORING_SEMANTICS,
        "test_metric": "MSE",
        "eval_data_seed": "17",
        "n_test_samples": "10000",
        "perturbation_channel_fraction_max": "0.5",
        "perturbation_scenarios_signature": "[\"drift\",\"noise\"]",
        "perturbation_scenarios_count": "2",
        "perturbation_idx_name_map": "{\"0\":\"drift\",\"1\":\"noise\"}",
    }

    parsed = require_degradation_eval_context_tags(
        tags,
        run_id="run_gc",
        expected_test_metric="MSE",
        expected_context={
            "eval_data_seed": 17,
            "n_test_samples": 10000,
            "perturbation_channel_fraction_max": 0.5,
            "perturbation_scenarios_signature": "[\"drift\",\"noise\"]",
            "perturbation_scenarios_count": 2,
            "perturbation_idx_name_map": {0: "drift", 1: "noise"},
        },
    )

    assert parsed["perturbation_idx_name_map"] == {0: "drift", 1: "noise"}


def test_require_degradation_eval_context_tags_can_skip_eval_data_seed_requirement():
    tags = {
        "robustness_scoring_semantics": DEGRADATION_SCORING_SEMANTICS,
        "test_metric": "MSE",
        "n_test_samples": "10000",
        "perturbation_channel_fraction_max": "0.5",
        "perturbation_scenarios_signature": "[\"drift\",\"noise\"]",
        "perturbation_scenarios_count": "2",
        "perturbation_idx_name_map": "{\"0\":\"drift\",\"1\":\"noise\"}",
    }

    parsed = require_degradation_eval_context_tags(
        tags,
        run_id="run_gc_canonical_meta",
        expected_test_metric="MSE",
        require_eval_data_seed=False,
    )

    assert "eval_data_seed" not in parsed
    assert parsed["perturbation_idx_name_map"] == {0: "drift", 1: "noise"}


def test_require_degradation_eval_context_tags_validates_optional_eval_data_seed_when_present():
    tags = {
        "robustness_scoring_semantics": DEGRADATION_SCORING_SEMANTICS,
        "test_metric": "MSE",
        "eval_data_seed": "not-an-int",
        "n_test_samples": "10000",
        "perturbation_channel_fraction_max": "0.5",
        "perturbation_scenarios_signature": "[\"drift\",\"noise\"]",
        "perturbation_scenarios_count": "2",
        "perturbation_idx_name_map": "{\"0\":\"drift\",\"1\":\"noise\"}",
    }

    with pytest.raises(ValueError, match="eval_data_seed"):
        require_degradation_eval_context_tags(
            tags,
            run_id="run_gc_optional_seed",
            expected_test_metric="MSE",
            require_eval_data_seed=False,
        )


def test_degradation_eval_context_matches_supports_partial_expected_context():
    context = {
        "robustness_scoring_semantics": DEGRADATION_SCORING_SEMANTICS,
        "test_metric": "MSE",
        "eval_data_seed": 17,
        "n_test_samples": 10000,
        "perturbation_channel_fraction_max": 0.5,
        "perturbation_scenarios_signature": "[\"drift\",\"noise\"]",
        "perturbation_scenarios_count": 2,
        "perturbation_idx_name_map": {0: "drift", 1: "noise"},
    }

    assert degradation_eval_context_matches(
        context,
        expected_context={
            "test_metric": "MSE",
            "perturbation_scenarios_signature": "[\"drift\",\"noise\"]",
            "perturbation_idx_name_map": {0: "drift", 1: "noise"},
        },
    )
    assert not degradation_eval_context_matches(
        context,
        expected_context={"eval_data_seed": 99},
    )


def test_degradation_n_test_samples_meet_policy_requires_exact_count():
    assert degradation_n_test_samples_meet_policy(
        10000,
        expected_n_test_samples=10000,
        full_coverage=True,
    )
    assert not degradation_n_test_samples_meet_policy(
        20000,
        expected_n_test_samples=10000,
        full_coverage=True,
    )
    assert not degradation_n_test_samples_meet_policy(
        20000,
        expected_n_test_samples=10000,
        full_coverage=False,
    )
    assert not degradation_n_test_samples_meet_policy(
        5000,
        expected_n_test_samples=10000,
        full_coverage=False,
    )


def test_parse_max_hp_trials_per_model_accepts_none_tokens():
    assert parse_max_hp_trials_per_model(None) is None
    assert parse_max_hp_trials_per_model("none") is None


def test_parse_max_hp_trials_per_model_rejects_non_positive_values():
    with pytest.raises(ValueError, match="must be positive"):
        parse_max_hp_trials_per_model(0)


def test_parse_eval_data_seed_accepts_null_and_integer_tokens():
    assert parse_eval_data_seed(None) is None
    assert parse_eval_data_seed("null") is None
    assert parse_eval_data_seed("17") == 17


def test_parse_eval_data_seed_rejects_non_integer_tokens():
    with pytest.raises(ValueError, match="Cannot parse eval_data_seed='abc' as int"):
        parse_eval_data_seed("abc")


def test_resolve_meta_analysis_eval_data_seed_scope_returns_mode_label():
    assert resolve_meta_analysis_eval_data_seed_scope(None) == (None, "canonical")
    assert resolve_meta_analysis_eval_data_seed_scope("17") == (17, "17")


def test_resolve_meta_analysis_eval_data_seed_scope_uses_default_seed_when_present():
    assert resolve_meta_analysis_eval_data_seed_scope(
        None,
        key="args.eval_data_seed",
        default_eval_data_seed="19",
        default_key="defaults.EVAL_DATA_SEED",
    ) == (19, "19")


def test_require_meta_analysis_eval_data_seed_scope_tags_accepts_canonical_mode_with_optional_seed():
    parsed_seed, parsed_mode = require_meta_analysis_eval_data_seed_scope_tags(
        {
            "analysis_scope": "meta_analysis",
            "eval_data_seed_mode": "canonical",
            "eval_data_seed": "29",
        },
        run_id="meta_canonical",
    )

    assert parsed_seed is None
    assert parsed_mode == "canonical"


def test_require_meta_analysis_eval_data_seed_scope_tags_rejects_mismatched_explicit_mode_and_seed():
    with pytest.raises(ValueError, match="eval_data_seed_mode='31' but eval_data_seed=29"):
        require_meta_analysis_eval_data_seed_scope_tags(
            {
                "analysis_scope": "meta_analysis",
                "eval_data_seed_mode": "31",
                "eval_data_seed": "29",
            },
            run_id="meta_explicit",
        )


def test_build_export_manifest_eval_context_accepts_explicit_null_eval_data_seed():
    assert build_export_manifest_eval_context(
        {
            "TEST_METRIC": "MSE",
            "N_TEST_SAMPLES": 10000,
            "EVAL_DATA_SEED": None,
            "PERTURBATION_CHANNEL_FRACTION_MAX": 0.5,
            "PERTURBATION_SCENARIOS": ["drift", "noise"],
            "BOOTSTRAP_CI_RESAMPLES": 1000,
            "BOOTSTRAP_CI_CONFIDENCE_LEVEL": 0.95,
        },
        n_test_samples=10000,
    ) == {
        "robustness_scoring_semantics": DEGRADATION_SCORING_SEMANTICS,
        "test_metric": "MSE",
        "n_test_samples": 10000,
        "eval_data_seed": None,
        "perturbation_channel_fraction_max": 0.5,
        "perturbation_scenarios": ["drift", "noise"],
        "bootstrap_ci_resamples": 1000,
        "bootstrap_ci_confidence_level": pytest.approx(0.95),
    }


def test_build_export_manifest_eval_context_requires_explicit_eval_data_seed_key():
    with pytest.raises(ValueError, match="missing required 'EVAL_DATA_SEED'"):
        build_export_manifest_eval_context(
            {
                "TEST_METRIC": "MSE",
                "N_TEST_SAMPLES": 10000,
                "PERTURBATION_CHANNEL_FRACTION_MAX": 0.5,
                "PERTURBATION_SCENARIOS": ["drift", "noise"],
                "BOOTSTRAP_CI_RESAMPLES": 1000,
                "BOOTSTRAP_CI_CONFIDENCE_LEVEL": 0.95,
            },
            n_test_samples=10000,
        )


def test_parse_bootstrap_ci_resamples_requires_positive_int():
    assert parse_bootstrap_ci_resamples("17") == 17
    with pytest.raises(ValueError, match="bootstrap_ci_resamples must be > 0"):
        parse_bootstrap_ci_resamples(0)


def test_parse_bootstrap_ci_confidence_level_requires_open_interval():
    assert parse_bootstrap_ci_confidence_level("0.95") == pytest.approx(0.95)
    with pytest.raises(
        ValueError,
        match="bootstrap_ci_confidence_level must satisfy 0 < bootstrap_ci_confidence_level < 1",
    ):
        parse_bootstrap_ci_confidence_level(1.0)


def test_resolve_effective_eval_data_seed_prefers_override():
    assert (
        resolve_effective_eval_data_seed(
            77,
            canonical_seed_data=11,
        )
        == 77
    )
    assert (
        resolve_effective_eval_data_seed(
            None,
            canonical_seed_data="11",
        )
        == 11
    )


def test_require_eval_data_seed_tag_requires_integer_tag():
    assert require_eval_data_seed_tag({"eval_data_seed": "19"}, run_id="run_1") == 19
    with pytest.raises(ValueError, match="missing required eval_data_seed tag"):
        require_eval_data_seed_tag({}, run_id="run_2")


def test_build_seeded_eval_input_artifact_prefix_uses_seed_specific_layout():
    assert (
        build_seeded_eval_input_artifact_prefix(
            test_metric="MSE",
            eval_data_seed=23,
        )
        == "robustness_inputs/MSE/seed_data_23"
    )


def test_build_seeded_degradation_artifact_prefix_uses_seed_specific_layout():
    assert (
        build_seeded_degradation_artifact_prefix(
            test_metric="MSE",
            eval_data_seed=23,
        )
        == "robustness/degradation/MSE/seed_data_23"
    )


def test_build_degradation_bootstrap_ci_seed_key_uses_degradation_namespace():
    assert (
        build_shared_anchor_bootstrap_ci_seed_key("MSE")
        == "bootstrap_ci:degradation:MSE:shared_anchor_percentile"
    )


def test_build_method_delta_pair_bootstrap_ci_seed_key_uses_method_delta_namespace():
    assert (
        build_method_delta_pair_bootstrap_ci_seed_key(
            "MSE",
            dataset="ETTh1",
            data_config_signature="sig_a",
            robustness_method="ensemble",
        )
        == (
            "bootstrap_ci:method_delta:MSE:ETTh1:sig_a:ensemble:"
            f"{METHOD_DELTA_PAIR_BOOTSTRAP_CI_SEMANTICS}"
        )
    )


def test_require_shared_anchor_bootstrap_ci_context_from_args_derives_seed_from_eval_identity():
    args = SimpleNamespace(
        bootstrap_ci_resamples=1000,
        bootstrap_ci_confidence_level=0.95,
    )

    context = require_shared_anchor_bootstrap_ci_context_from_args(
        args,
        eval_data_seed=17,
        test_metric="MSE",
        context="args",
    )

    assert context["bootstrap_ci_semantics"] == SHARED_ANCHOR_BOOTSTRAP_CI_SEMANTICS
    assert context["bootstrap_ci_resamples"] == 1000
    assert context["bootstrap_ci_confidence_level"] == pytest.approx(0.95)
    assert isinstance(context["bootstrap_ci_seed"], int)


def test_build_shared_anchor_bootstrap_ci_tag_payload_stringifies_normalized_values():
    payload = build_shared_anchor_bootstrap_ci_tag_payload(
        {
            "bootstrap_ci_semantics": SHARED_ANCHOR_BOOTSTRAP_CI_SEMANTICS,
            "bootstrap_ci_resamples": 1000,
            "bootstrap_ci_confidence_level": 0.95,
            "bootstrap_ci_seed": 123,
        },
        context_name="shared_anchor_bootstrap_ci_context",
    )

    assert payload == {
        "bootstrap_ci_semantics": SHARED_ANCHOR_BOOTSTRAP_CI_SEMANTICS,
        "bootstrap_ci_resamples": "1000",
        "bootstrap_ci_confidence_level": "0.95",
        "bootstrap_ci_seed": "123",
    }


def test_require_shared_anchor_bootstrap_ci_context_tags_rejects_unknown_semantics():
    tags = {
        "bootstrap_ci_semantics": BOOTSTRAP_CI_SEMANTICS,
        "bootstrap_ci_resamples": "1000",
        "bootstrap_ci_confidence_level": "0.95",
        "bootstrap_ci_seed": "123",
    }
    with pytest.raises(ValueError, match="unsupported bootstrap_ci_semantics"):
        require_shared_anchor_bootstrap_ci_context_tags(
            tags,
            run_id="run_bootstrap_previous",
            require_seed=True,
        )


def test_shared_anchor_bootstrap_ci_context_matches_supports_seed_optional_matching():
    context = {
        "bootstrap_ci_semantics": SHARED_ANCHOR_BOOTSTRAP_CI_SEMANTICS,
        "bootstrap_ci_resamples": 1000,
        "bootstrap_ci_confidence_level": 0.95,
    }

    assert shared_anchor_bootstrap_ci_context_matches(
        context,
        expected_context={
            "bootstrap_ci_semantics": SHARED_ANCHOR_BOOTSTRAP_CI_SEMANTICS,
            "bootstrap_ci_resamples": 1000,
        },
        require_seed=False,
    )
    assert not shared_anchor_bootstrap_ci_context_matches(
        context,
        expected_context={"bootstrap_ci_confidence_level": 0.9},
        require_seed=False,
    )


def test_parse_feature_indices_accepts_valid_values():
    assert parse_feature_indices((0, 2.0, "3"), n_features=5) == (0, 2, 3)


def test_parse_feature_indices_allows_none_when_configured():
    assert parse_feature_indices(None, n_features=3, allow_none=True) is None
    with pytest.raises(ValueError, match="target_indices is required"):
        parse_feature_indices(None, n_features=3)


@pytest.mark.parametrize(
    "value, expected_message",
    [
        ((), "non-empty"),
        ((0, 0), "unique"),
        ((0, 4), "range"),
        ((1.5,), "integer values"),
        ((True,), "not bools"),
    ],
)
def test_parse_feature_indices_rejects_invalid_values(value, expected_message):
    with pytest.raises(ValueError, match=expected_message):
        parse_feature_indices(value, n_features=4)


def test_parse_feature_indices_rejects_non_scalar_tensor():
    with pytest.raises(ValueError, match="scalar values"):
        parse_feature_indices((torch.tensor([1, 2]),), n_features=5)


def test_parse_required_positive_int_requires_present_positive_values():
    assert parse_required_positive_int("3", key="n_layers_enc") == 3


def test_parse_required_nonnegative_int_requires_integer_nonnegative_values():
    assert (
        parse_required_nonnegative_int(
            "3",
            key="scenario_idx",
            context="degradation metric key",
        )
        == 3
    )
    with pytest.raises(ValueError, match="requires integer scenario_idx"):
        parse_required_nonnegative_int(
            1.9,
            key="scenario_idx",
            context="degradation metric key",
        )
    with pytest.raises(ValueError, match="requires scenario_idx >= 0"):
        parse_required_nonnegative_int(
            -1,
            key="scenario_idx",
            context="degradation metric key",
        )


def test_parse_required_finite_float_requires_present_finite_values():
    assert parse_required_finite_float("0.25", key="scheduler_factor") == 0.25
    with pytest.raises(ValueError, match="scheduler_factor is required"):
        parse_required_finite_float(None, key="scheduler_factor")
    with pytest.raises(ValueError, match="scheduler_factor must be finite"):
        parse_required_finite_float(float("inf"), key="scheduler_factor")
    with pytest.raises(ValueError, match="scheduler_factor must be finite"):
        parse_required_finite_float(float("nan"), key="scheduler_factor")


def test_parse_optimizer_name_accepts_only_adam():
    assert parse_optimizer_name(" Adam ") == "Adam"
    assert parse_optimizer_name("adam") == "adam"
    with pytest.raises(ValueError, match="Unknown optimizer 'SGD'"):
        parse_optimizer_name("SGD")


def test_parse_scheduler_type_accepts_only_plateau():
    assert parse_scheduler_type("plateau") == "plateau"
    with pytest.raises(ValueError, match="Unknown scheduler_type 'cosine'"):
        parse_scheduler_type("cosine")


def test_parse_required_positive_int_rejects_missing_and_nonpositive_values():
    with pytest.raises(ValueError, match="n_layers_enc must be provided"):
        parse_required_positive_int(None, key="n_layers_enc")
    with pytest.raises(ValueError, match="n_layers_enc must be > 0"):
        parse_required_positive_int(0, key="n_layers_enc")


def test_parse_required_dropout_validates_bounds():
    assert parse_required_dropout("0.1") == pytest.approx(0.1)
    with pytest.raises(ValueError, match="dropout must satisfy 0 <= dropout < 1"):
        parse_required_dropout(1.0)
    with pytest.raises(ValueError, match="dropout must be a float in \\[0, 1\\)"):
        parse_required_dropout("not-a-float")


def test_parse_required_choice_normalizes_and_rejects_invalid_values():
    assert (
        parse_required_choice(
            " GeLU ",
            key="activation",
            allowed=("gelu", "relu"),
        )
        == "gelu"
    )
    with pytest.raises(ValueError, match="Unsupported activation"):
        parse_required_choice(
            "swish",
            key="activation",
            allowed=("gelu", "relu"),
        )


def test_parse_dataset_split_mode_normalizes_and_rejects_invalid_values():
    assert parse_dataset_split_mode(" Within_Batches ") == "within_batches"
    with pytest.raises(ValueError, match="Unsupported split_mode"):
        parse_dataset_split_mode("stationwise")


def test_require_tsmixer_hparams_accepts_valid_values():
    parsed = require_tsmixer_hparams(
        {
            "n_block": 2,
            "ff_dim": 64,
            "dropout": 0.1,
            "norm_type": "L",
            "activation": "gelu",
        }
    )
    assert parsed["n_block"] == 2
    assert parsed["ff_dim"] == 64
    assert parsed["dropout"] == pytest.approx(0.1)
    assert parsed["norm_type"] == "L"
    assert parsed["activation"] == "gelu"


def test_require_tsmixer_hparams_rejects_invalid_values():
    with pytest.raises(ValueError, match="missing n_block"):
        require_tsmixer_hparams(
            {
                "ff_dim": 64,
                "dropout": 0.1,
                "norm_type": "L",
                "activation": "relu",
            }
        )
    with pytest.raises(ValueError, match="norm_type must be exactly"):
        require_tsmixer_hparams(
            {
                "n_block": 2,
                "ff_dim": 64,
                "dropout": 0.1,
                "norm_type": "layer",
                "activation": "relu",
            }
        )


def test_parse_perturbation_channel_fraction_max_accepts_valid_values():
    assert parse_perturbation_channel_fraction_max(0.5) == pytest.approx(0.5)
    assert parse_perturbation_channel_fraction_max("1.0") == pytest.approx(1.0)


def test_parse_perturbation_channel_fraction_max_rejects_invalid_values():
    with pytest.raises(
        ValueError,
        match="must satisfy 0 < perturbation_channel_fraction_max <= 1",
    ):
        parse_perturbation_channel_fraction_max(0.0)
    with pytest.raises(
        ValueError,
        match="must satisfy 0 < perturbation_channel_fraction_max <= 1",
    ):
        parse_perturbation_channel_fraction_max(1.1)


def test_parse_perturbation_scenarios_accepts_valid_lists():
    assert parse_perturbation_scenarios(["drift", "noise"]) == ("drift", "noise")


def test_parse_perturbation_scenarios_rejects_empty_or_duplicate_names():
    with pytest.raises(ValueError, match="non-empty list of scenario names"):
        parse_perturbation_scenarios([])
    with pytest.raises(ValueError, match="duplicate scenario name"):
        parse_perturbation_scenarios(["drift", "drift"])
    with pytest.raises(ValueError, match="must be a string"):
        parse_perturbation_scenarios(["drift", 1])


def test_build_perturbation_scenarios_signature_is_deterministic():
    signature_one = build_perturbation_scenarios_signature(("drift", "noise"))
    signature_two = build_perturbation_scenarios_signature(("drift", "noise"))
    assert signature_one == signature_two
    assert signature_one == "[\"drift\",\"noise\"]"


def test_parse_perturbation_scenarios_signature_requires_canonical_form():
    canonical = "[\"drift\",\"noise\"]"
    assert parse_perturbation_scenarios_signature(canonical) == canonical
    assert parse_perturbation_scenarios_from_signature(canonical) == ("drift", "noise")
    with pytest.raises(ValueError, match="canonical JSON"):
        parse_perturbation_scenarios_signature("drift|noise")
    with pytest.raises(ValueError, match="canonical JSON"):
        parse_perturbation_scenarios_signature("[\"drift\", \"noise\"]")


def test_build_perturbation_idx_name_map_is_deterministic():
    canonical = build_perturbation_idx_name_map({2: "noise", 0: "drift"})
    assert canonical == "{\"0\":\"drift\",\"2\":\"noise\"}"


def test_parse_perturbation_idx_name_map_requires_canonical_form():
    canonical = "{\"0\":\"drift\",\"2\":\"noise\"}"
    parsed = parse_perturbation_idx_name_map(canonical)
    assert parsed == {0: "drift", 2: "noise"}
    with pytest.raises(ValueError, match="canonical JSON object"):
        parse_perturbation_idx_name_map("0:drift,2:noise")
    with pytest.raises(ValueError, match="must be canonical JSON with sorted integer-string keys"):
        parse_perturbation_idx_name_map("{\"2\":\"noise\",\"0\":\"drift\"}")


def test_build_perturbation_idx_name_map_rejects_invalid_mapping():
    with pytest.raises(ValueError, match="non-empty mapping"):
        build_perturbation_idx_name_map({})
    with pytest.raises(ValueError, match="negative perturbation index"):
        build_perturbation_idx_name_map({-1: "drift"})
    with pytest.raises(ValueError, match="duplicate scenario name"):
        build_perturbation_idx_name_map({0: "drift", 1: "drift"})


def test_require_order_sensitive_perturbation_idx_name_map_requires_contiguous_matching_order():
    signature = build_perturbation_scenarios_signature(("drift", "noise"))

    parsed = require_order_sensitive_perturbation_idx_name_map(
        {0: "drift", 1: "noise"},
        scenarios_signature=signature,
        scenarios_count=2,
    )

    assert parsed == {0: "drift", 1: "noise"}

    with pytest.raises(ValueError, match="contiguous zero-based indices"):
        require_order_sensitive_perturbation_idx_name_map(
            {0: "drift", 2: "noise"},
            scenarios_signature=signature,
            scenarios_count=2,
        )

    with pytest.raises(ValueError, match="does not match perturbation_scenarios_signature"):
        require_order_sensitive_perturbation_idx_name_map(
            {0: "noise", 1: "drift"},
            scenarios_signature=signature,
            scenarios_count=2,
        )


def test_require_perturbation_coupling_params_parses_required_values():
    signature = build_perturbation_scenarios_signature(("drift", "noise"))
    parsed = require_perturbation_coupling_params(
        {
            "perturbation_channel_fraction_max": "0.5",
            "perturbation_scenarios_signature": signature,
        },
        run_id="run_pert",
    )
    assert parsed["perturbation_channel_fraction_max"] == pytest.approx(0.5)
    assert parsed["perturbation_scenarios_signature"] == signature


def test_require_perturbation_coupling_params_rejects_missing_or_invalid_values():
    signature = build_perturbation_scenarios_signature(("drift", "noise"))
    with pytest.raises(ValueError, match="invalid perturbation coupling params"):
        require_perturbation_coupling_params(
            {},
            run_id="run_missing_max",
        )
    with pytest.raises(ValueError, match="invalid perturbation coupling params"):
        require_perturbation_coupling_params(
            {"perturbation_channel_fraction_max": "2.0"},
            run_id="run_bad_max",
        )
    with pytest.raises(ValueError, match="invalid perturbation coupling params"):
        require_perturbation_coupling_params(
            {
                "perturbation_channel_fraction_max": "0.5",
                "perturbation_scenarios_signature": "[\"drift\", \"noise\"]",
            },
            run_id="run_bad_signature",
        )
    with pytest.raises(
        ValueError,
        match="canonical JSON with deterministic ordering",
    ):
        require_perturbation_coupling_params(
            {
                "perturbation_channel_fraction_max": "0.5",
                "perturbation_scenarios_signature": signature,
            },
            run_id="run_missing_signature",
            expected_scenarios_signature="[\"drift\", \"noise\"]",
        )


def test_require_perturbation_coupling_params_rejects_expected_mismatches():
    signature = build_perturbation_scenarios_signature(("drift", "noise"))
    with pytest.raises(ValueError, match="expected 0.25"):
        require_perturbation_coupling_params(
            {
                "perturbation_channel_fraction_max": "0.5",
                "perturbation_scenarios_signature": signature,
            },
            run_id="run_mismatch_max",
            expected_max=0.25,
        )
    with pytest.raises(ValueError, match="expected \\[\"drift\",\"missing_data\"\\]"):
        require_perturbation_coupling_params(
            {
                "perturbation_channel_fraction_max": "0.5",
                "perturbation_scenarios_signature": signature,
            },
            run_id="run_mismatch_signature",
            expected_scenarios_signature=build_perturbation_scenarios_signature(
                ("drift", "missing_data")
            ),
        )


def test_require_perturbation_coupling_tags_parses_and_validates():
    signature = build_perturbation_scenarios_signature(("drift", "noise"))
    parsed = require_perturbation_coupling_tags(
        {
            "perturbation_channel_fraction_max": "0.5",
            "perturbation_scenarios_signature": signature,
        },
        run_id="run_tagged",
        expected_max=0.5,
        expected_scenarios_signature=signature,
    )
    assert parsed["perturbation_channel_fraction_max"] == pytest.approx(0.5)
    assert parsed["perturbation_scenarios_signature"] == signature


def test_require_perturbation_coupling_tags_rejects_missing_mapping():
    with pytest.raises(ValueError, match="missing tags required for perturbation coupling"):
        require_perturbation_coupling_tags(None, run_id="run_missing_tags")


def test_require_robustness_scoring_semantics_tag_defaults_to_degradation():
    assert (
        require_robustness_scoring_semantics_tag(
            {"robustness_scoring_semantics": "uniform_severity_degradation"},
            run_id="run_8",
        )
        == "uniform_severity_degradation"
    )
    with pytest.raises(ValueError, match="missing required robustness_scoring_semantics"):
        require_robustness_scoring_semantics_tag({}, run_id="run_9")
    with pytest.raises(ValueError, match="expected 'uniform_severity_degradation'"):
        require_robustness_scoring_semantics_tag(
            {"robustness_scoring_semantics": "unsupported_semantics"},
            run_id="run_10",
        )


def test_require_robustness_scoring_semantics_tag_rejects_bin_v1():
    with pytest.raises(ValueError, match="expected 'uniform_severity_degradation'"):
        require_robustness_scoring_semantics_tag(
            {"robustness_scoring_semantics": "bin_v1"},
            run_id="run_bin_v1",
        )


def test_parse_scenario_metric_key_parses_integer_scenario_tokens():
    parsed = parse_scenario_metric_key(
        "linear_r1_phi1/MSE/scenario/2/R_mean",
        scenario_prefix="linear_r1_phi1/MSE/scenario/",
        run_id="run_11",
    )
    assert parsed == (2, "R_mean")


def test_parse_scenario_metric_key_returns_none_when_prefix_does_not_match():
    assert (
        parse_scenario_metric_key(
            "linear_r1_phi1/MSE/R_mean",
            scenario_prefix="linear_r1_phi1/MSE/scenario/",
            run_id="run_11b",
        )
        is None
    )


def test_parse_scenario_metric_key_rejects_non_integer_scenario_tokens():
    with pytest.raises(ValueError, match="integer pert_idx"):
        parse_scenario_metric_key(
            "linear_r1_phi1/MSE/scenario/Noise/R_mean",
            scenario_prefix="linear_r1_phi1/MSE/scenario/",
            run_id="run_12",
        )


def test_parse_scenario_metric_key_rejects_negative_scenario_index():
    with pytest.raises(ValueError, match="non-negative pert_idx"):
        parse_scenario_metric_key(
            "linear_r1_phi1/MSE/scenario/-1/R_mean",
            scenario_prefix="linear_r1_phi1/MSE/scenario/",
            run_id="run_13",
        )


def test_infer_hparam_expected_type_handles_numeric_and_null_grids():
    assert infer_hparam_expected_type([1, 2, None], key="d_model") is int
    assert infer_hparam_expected_type([0.1, 0.2], key="dropout") is float
    assert infer_hparam_expected_type([1, 1.5], key="mixed_num") is float
    assert infer_hparam_expected_type([None], key="all_null") is type(None)


def test_coerce_hparam_value_parses_bool_and_list_tokens():
    assert coerce_hparam_value("true", bool, key="flag") is True
    assert coerce_hparam_value("False", bool, key="flag") is False
    assert coerce_hparam_value("[1, 2, 3]", list, key="shape") == [1, 2, 3]


def test_coerce_hparam_value_handles_null_token_when_allowed():
    assert coerce_hparam_value("none", int, key="d_model", allow_none=True) is None


def test_coerce_hparam_value_rejects_invalid_bool_token():
    with pytest.raises(ValueError, match="to bool"):
        coerce_hparam_value("maybe", bool, key="flag")


def test_coerce_hparam_value_rejects_null_token_when_disallowed():
    with pytest.raises(ValueError, match="does not allow null"):
        coerce_hparam_value("none", int, key="d_model")


def test_extract_required_typed_hparams_requires_all_keys_and_types():
    spec = {
        "lr": [0.001, 0.0005],
        "dropout": [0.0, 0.1],
        "autoregressive": [False, True],
        "d_hidden_layers": [[128], [256, 256]],
        "optional_width": [64, None],
    }
    typed = extract_required_typed_hparams(
        {
            "lr": "0.001",
            "dropout": "0.1",
            "autoregressive": "false",
            "d_hidden_layers": "[256, 256]",
            "optional_width": "none",
        },
        spec,
        context="hparam extraction test",
    )
    assert typed["lr"] == pytest.approx(0.001)
    assert typed["dropout"] == pytest.approx(0.1)
    assert typed["autoregressive"] is False
    assert typed["d_hidden_layers"] == [256, 256]
    assert typed["optional_width"] is None


def test_extract_required_typed_hparams_raises_on_missing_key():
    with pytest.raises(ValueError, match="missing required hyperparameter 'lr'"):
        extract_required_typed_hparams(
            {"dropout": "0.1"},
            {"lr": [0.001, 0.0005], "dropout": [0.0, 0.1]},
            context="missing key test",
        )


def test_extract_required_typed_hparams_rejects_disallowed_null_values():
    with pytest.raises(ValueError, match="does not allow null"):
        extract_required_typed_hparams(
            {"lr": "none"},
            {"lr": [0.001, 0.0005]},
            context="null disallowed test",
        )


def test_normalize_yaml_value_recursively_normalizes_numpy_and_collections():
    payload = {
        "b": {"x": [1, 2, 3], "y": ("a", "b")},
        "a": 1,
        "c": {np.int64(1), np.float32(2.5)},
        "d": np.float64(0.5),
    }

    normalized = normalize_yaml_value(payload)

    assert normalized["a"] == 1
    assert normalized["b"] == {"x": [1, 2, 3], "y": ["a", "b"]}
    assert sorted(normalized["c"]) == [1, 2.5]
    assert normalized["d"] == pytest.approx(0.5)
