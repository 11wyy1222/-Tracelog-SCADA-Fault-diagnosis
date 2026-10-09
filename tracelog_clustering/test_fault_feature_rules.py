from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fault_feature_rules import (
    FaultFeatureRuleError,
    compile_confirmation_frame,
    evaluate_approved_rules,
    load_approved_rules,
    load_description_config,
    write_confirmation_table,
)


HERE = Path(__file__).resolve().parent
EXAMPLE_CONFIG = HERE / "fault_feature_descriptions.example.yaml"


def test_example_rules_compile_and_evaluate(tmp_path):
    config = load_description_config(EXAMPLE_CONFIG)
    review = compile_confirmation_frame(config)
    assert len(review) == 8
    assert set(review["compile_status"]) == {"ok"}

    review["expert_decision"] = "通过"
    confirmation_path = tmp_path / "confirmed.csv"
    write_confirmation_table(review, confirmation_path)
    rules = load_approved_rules(confirmation_path)

    base = pd.DataFrame(
        {
            "sample_index": [0, 1],
            "slip_ring_jump_score": [5.0, 1.0],
            "proximity_jump_score": [1.0, 4.0],
            "generator_drop_score": [2.0, 0.5],
            "rotor_rise_score": [3.0, 0.5],
            "generator_jump_score": [1.5, 2.0],
            "sensor_stable_score": [0.5, 0.25],
            "converter_zero_score": [0.0, 2.0],
            "converter_drop_score": [1.0, 1.0],
            "converter_negative_ratio": [0.1, 0.0],
            "gen_converter_corr": [0.8, -0.2],
            "sensor_zero_score": [0.0, 0.4],
            "rotor_drop_score": [0.0, 2.0],
        }
    )
    evaluated, feature_columns = evaluate_approved_rules(base, rules)

    assert feature_columns == [f"configured_feature_{code}" for code in "ABCDEFGH"]
    np.testing.assert_allclose(evaluated["configured_feature_A"], [5.0, 1.0])
    np.testing.assert_allclose(evaluated["configured_feature_B"], [1.0, 4.0])
    np.testing.assert_allclose(evaluated["configured_feature_C"], [3.0, 0.0625])
    np.testing.assert_allclose(evaluated["configured_feature_F"], [0.5, 0.5])
    assert list(evaluated["rule_candidate_label"]) == ["A", "B"]


def test_pending_rules_are_blocked(tmp_path):
    review = compile_confirmation_frame(load_description_config(EXAMPLE_CONFIG))
    confirmation_path = tmp_path / "pending.xlsx"
    write_confirmation_table(review, confirmation_path)

    with pytest.raises(FaultFeatureRuleError, match="尚未人工确认"):
        load_approved_rules(confirmation_path)


def test_expert_expression_is_validated(tmp_path):
    review = compile_confirmation_frame(load_description_config(EXAMPLE_CONFIG)).iloc[:1].copy()
    review["expert_decision"] = "通过"
    review["expert_expression"] = "__import__('os').system('dir')"
    confirmation_path = tmp_path / "unsafe.csv"
    write_confirmation_table(review, confirmation_path)

    with pytest.raises(FaultFeatureRuleError, match="只允许调用"):
        load_approved_rules(confirmation_path)
