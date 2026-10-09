import argparse
from pathlib import Path

from fault_feature_rules import prepare_confirmation_table


def main() -> None:
    parser = argparse.ArgumentParser(description="把故障自然语言规则编译为人工确认表")
    parser.add_argument("--description-config", required=True, help="自然语言规则 YAML 参数文件")
    parser.add_argument("--output", required=True, help="输出确认表（.xlsx 或 .csv）")
    args = parser.parse_args()

    frame = prepare_confirmation_table(Path(args.description_config), Path(args.output))
    errors = int((frame["compile_status"] != "ok").sum())
    print(f"已生成确认表: {Path(args.output).resolve()}")
    print(f"规则数: {len(frame)}，编译失败: {errors}")
    print("请填写 expert_decision（通过/拒绝）；必要时在 expert_expression 中修正公式。")


if __name__ == "__main__":
    main()

