"""Compatibility entry point that injects reviewed fault rules into the existing clusterer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import cluster_tracelog_samples as legacy

from fault_feature_rules import evaluate_approved_rules, load_approved_rules


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--fault-feature-confirmation-path",
        required=True,
        help="已完成人工确认的故障特征规则表（.xlsx/.csv）",
    )
    wrapper_args, legacy_args = parser.parse_known_args()

    confirmation_path = Path(wrapper_args.fault_feature_confirmation_path).resolve()
    approved_rules = load_approved_rules(confirmation_path)
    original_build = legacy.build_fault_tree_feature_frame
    original_save = legacy.save_outputs

    def build_configured_feature_frame(samples: Sequence, main_sensor_channels: Sequence[str]):
        base_frame = original_build(samples, main_sensor_channels)
        configured_frame, configured_columns = evaluate_approved_rules(base_frame, approved_rules)

        # Mutate the existing list so default references held by the legacy module see the new columns.
        legacy.CORE_FAULT_TREE_FEATURES.clear()
        legacy.CORE_FAULT_TREE_FEATURES.extend(configured_columns)
        return configured_frame

    def save_outputs_with_rule_audit(*args, **kwargs):
        original_save(*args, **kwargs)
        output_dir = Path(kwargs["output_dir"] if "output_dir" in kwargs else args[0])
        metadata_path = output_dir / "run_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["fault_feature_confirmation_path"] = str(confirmation_path)
        metadata["approved_fault_features"] = [
            {
                "feature_id": rule.feature_id,
                "fault_description": rule.fault_description,
                "expression": rule.expression,
                "input_features": list(rule.input_features),
                "weight": rule.weight,
            }
            for rule in approved_rules
        ]
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    legacy.build_fault_tree_feature_frame = build_configured_feature_frame
    legacy.save_outputs = save_outputs_with_rule_audit
    sys.argv = [sys.argv[0], *legacy_args]
    legacy.main()


if __name__ == "__main__":
    main()

