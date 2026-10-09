"""Compile reviewable Chinese fault-feature rules and evaluate approved rules safely."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import yaml


APPROVED_DECISIONS = {"approved", "approve", "通过", "确认", "同意", "yes", "y"}
REJECTED_DECISIONS = {"rejected", "reject", "拒绝", "不通过", "否决", "no", "n"}
REVIEW_COLUMNS = [
    "feature_id",
    "fault_description",
    "judgment_method",
    "natural_rule",
    "generated_expression",
    "input_features",
    "weight",
    "enabled",
    "compile_status",
    "compile_message",
    "source_sequence",
    "source_file",
    "expert_decision",
    "expert_expression",
    "expert_comment",
    "config_version",
    "config_hash",
]


class FaultFeatureRuleError(ValueError):
    """Raised when a rule file is invalid or has not passed expert review."""


@dataclass(frozen=True)
class ApprovedRule:
    feature_id: str
    fault_description: str
    expression: str
    input_features: Tuple[str, ...]
    weight: float


def _as_bool(value: Any, default: bool = True) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "n", "否", "禁用"}


def _normalise_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def load_description_config(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict):
        raise FaultFeatureRuleError("故障特征描述参数文件的顶层必须是对象")
    if not isinstance(config.get("features"), list) or not config["features"]:
        raise FaultFeatureRuleError("参数文件必须包含非空 features 列表")
    glossary = config.get("glossary", {})
    if not isinstance(glossary, dict) or not glossary:
        raise FaultFeatureRuleError("参数文件必须包含 glossary（中文术语到原子特征名的映射）")
    return config


def _config_hash(config: Mapping[str, Any]) -> str:
    canonical = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def translate_natural_rule(natural_rule: str, glossary: Mapping[str, str]) -> str:
    """Translate deterministic, controlled Chinese wording into a safe expression."""
    expression = _normalise_text(natural_rule)
    if not expression:
        raise FaultFeatureRuleError("natural_rule 不能为空")

    expression = (
        expression.replace("（", "(")
        .replace("）", ")")
        .replace("，", ",")
        .replace("：", ":")
        .replace("×", "*")
        .replace("÷", "/")
        .replace("＋", "+")
        .replace("－", "-")
    )

    # Longest aliases first prevents a shorter term from corrupting a longer one.
    for alias, feature_name in sorted(glossary.items(), key=lambda item: len(str(item[0])), reverse=True):
        alias_text = _normalise_text(alias)
        feature_text = _normalise_text(feature_name)
        if not alias_text or not feature_text.isidentifier():
            raise FaultFeatureRuleError(f"glossary 映射无效: {alias!r} -> {feature_name!r}")
        expression = expression.replace(alias_text, feature_text)

    word_replacements = [
        ("大于等于", ">="),
        ("不小于", ">="),
        ("小于等于", "<="),
        ("不大于", "<="),
        ("不等于", "!="),
        ("较大值", "max"),
        ("最大值", "max"),
        ("较小值", "min"),
        ("最小值", "min"),
        ("绝对值", "abs"),
        ("并且", " and "),
        ("同时", " and "),
        ("或者", " or "),
        ("大于", ">"),
        ("小于", "<"),
        ("等于", "=="),
        ("乘以", "*"),
        ("除以", "/"),
        ("加上", "+"),
        ("减去", "-"),
        ("且", " and "),
        ("或", " or "),
    ]
    for source, target in word_replacements:
        expression = expression.replace(source, target)

    expression = re.sub(r"\s+", " ", expression).strip()
    return expression


class _ExpressionValidator(ast.NodeVisitor):
    ALLOWED_BINARY = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod)
    ALLOWED_UNARY = (ast.UAdd, ast.USub, ast.Not)
    ALLOWED_BOOLEAN = (ast.And, ast.Or)
    ALLOWED_COMPARE = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)
    ALLOWED_CALLS = {"min", "max", "abs", "clip"}

    def __init__(self) -> None:
        self.names: Set[str] = set()

    def generic_visit(self, node: ast.AST) -> None:
        allowed = (
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.BoolOp,
            ast.Compare,
            ast.Call,
            ast.Name,
            ast.Load,
            ast.Constant,
        )
        if not isinstance(node, allowed + self.ALLOWED_BINARY + self.ALLOWED_UNARY + self.ALLOWED_BOOLEAN + self.ALLOWED_COMPARE):
            raise FaultFeatureRuleError(f"表达式包含不允许的语法: {type(node).__name__}")
        super().generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id not in self.ALLOWED_CALLS:
            self.names.add(node.id)

    def visit_Call(self, node: ast.Call) -> None:
        if not isinstance(node.func, ast.Name) or node.func.id not in self.ALLOWED_CALLS:
            raise FaultFeatureRuleError("只允许调用 min、max、abs、clip")
        if node.keywords:
            raise FaultFeatureRuleError("函数调用不支持关键字参数")
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if not isinstance(node.op, self.ALLOWED_BINARY):
            raise FaultFeatureRuleError(f"不允许的二元运算: {type(node.op).__name__}")
        self.generic_visit(node)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        if not isinstance(node.op, self.ALLOWED_UNARY):
            raise FaultFeatureRuleError(f"不允许的一元运算: {type(node.op).__name__}")
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        if not isinstance(node.op, self.ALLOWED_BOOLEAN):
            raise FaultFeatureRuleError(f"不允许的逻辑运算: {type(node.op).__name__}")
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        if any(not isinstance(operator, self.ALLOWED_COMPARE) for operator in node.ops):
            raise FaultFeatureRuleError("表达式包含不允许的比较运算")
        self.generic_visit(node)


def validate_expression(expression: str) -> Tuple[ast.Expression, Tuple[str, ...]]:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise FaultFeatureRuleError(f"无法解析生成公式: {exc.msg}") from exc
    validator = _ExpressionValidator()
    validator.visit(tree)
    return tree, tuple(sorted(validator.names))


def compile_confirmation_frame(config: Mapping[str, Any]) -> pd.DataFrame:
    glossary = config["glossary"]
    version = _normalise_text(config.get("version", "unversioned"))
    digest = _config_hash(config)
    rows: List[Dict[str, Any]] = []
    seen_ids: Set[str] = set()

    for raw_feature in config["features"]:
        feature = raw_feature if isinstance(raw_feature, dict) else {}
        feature_id = _normalise_text(feature.get("feature_id"))
        natural_rule = _normalise_text(feature.get("natural_rule"))
        status = "ok"
        message = ""
        generated = ""
        inputs: Sequence[str] = ()
        if not feature_id:
            status, message = "error", "feature_id 不能为空"
        elif feature_id in seen_ids:
            status, message = "error", f"feature_id 重复: {feature_id}"
        else:
            seen_ids.add(feature_id)
            try:
                generated = translate_natural_rule(natural_rule, glossary)
                _, inputs = validate_expression(generated)
                known_features = set(map(str, glossary.values()))
                unknown = sorted(set(inputs) - known_features)
                if unknown:
                    raise FaultFeatureRuleError(f"公式引用 glossary 中不存在的原子特征: {', '.join(unknown)}")
            except FaultFeatureRuleError as exc:
                status, message = "error", str(exc)

        rows.append(
            {
                "feature_id": feature_id,
                "fault_description": _normalise_text(feature.get("fault_description")),
                "judgment_method": _normalise_text(feature.get("judgment_method")),
                "natural_rule": natural_rule,
                "generated_expression": generated,
                "input_features": ",".join(inputs),
                "weight": float(feature.get("weight", 1.0)),
                "enabled": _as_bool(feature.get("enabled"), default=True),
                "compile_status": status,
                "compile_message": message,
                "source_sequence": _normalise_text(feature.get("source_sequence")),
                "source_file": _normalise_text(config.get("source_file")),
                "expert_decision": "pending",
                "expert_expression": "",
                "expert_comment": "",
                "config_version": version,
                "config_hash": digest,
            }
        )
    return pd.DataFrame(rows, columns=REVIEW_COLUMNS)


def write_confirmation_table(frame: pd.DataFrame, output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".xlsx":
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            frame.to_excel(writer, index=False, sheet_name="故障特征确认")
            worksheet = writer.sheets["故障特征确认"]
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            widths = {
                "A": 14, "B": 30, "C": 60, "D": 46, "E": 46, "F": 40, "G": 10,
                "H": 10, "I": 14, "J": 34, "K": 12, "L": 28, "M": 16, "N": 46,
                "O": 28, "P": 18, "Q": 20,
            }
            for column, width in widths.items():
                worksheet.column_dimensions[column].width = width
    elif suffix == ".csv":
        frame.to_csv(output_path, index=False, encoding="utf-8-sig")
    else:
        raise FaultFeatureRuleError("确认表只支持 .xlsx 或 .csv")


def read_confirmation_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path, sheet_name="故障特征确认")
    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def load_approved_rules(path: Path) -> List[ApprovedRule]:
    frame = read_confirmation_table(path)
    required_columns = {
        "feature_id",
        "fault_description",
        "generated_expression",
        "weight",
        "enabled",
        "compile_status",
        "expert_decision",
    }
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise FaultFeatureRuleError(f"确认表缺少列: {', '.join(missing)}")

    enabled = frame[frame["enabled"].map(lambda value: _as_bool(value, default=True))].copy()
    compile_errors = enabled[enabled["compile_status"].astype(str).str.lower() != "ok"]
    if not compile_errors.empty:
        ids = ", ".join(compile_errors["feature_id"].astype(str))
        raise FaultFeatureRuleError(f"以下启用规则编译失败，不能进入聚类: {ids}")

    decisions = enabled["expert_decision"].fillna("").astype(str).str.strip().str.lower()
    pending_mask = ~decisions.isin(APPROVED_DECISIONS | REJECTED_DECISIONS)
    if pending_mask.any():
        ids = ", ".join(enabled.loc[pending_mask, "feature_id"].astype(str))
        raise FaultFeatureRuleError(f"以下规则尚未人工确认: {ids}")

    approved_frame = enabled[decisions.isin(APPROVED_DECISIONS)].copy()
    if approved_frame.empty:
        raise FaultFeatureRuleError("确认表中没有通过的故障特征")

    rules: List[ApprovedRule] = []
    for _, row in approved_frame.iterrows():
        expert_expression = _normalise_text(row.get("expert_expression"))
        expression = expert_expression or _normalise_text(row["generated_expression"])
        _, inputs = validate_expression(expression)
        rules.append(
            ApprovedRule(
                feature_id=_normalise_text(row["feature_id"]),
                fault_description=_normalise_text(row["fault_description"]),
                expression=expression,
                input_features=inputs,
                weight=float(row.get("weight", 1.0)),
            )
        )
    return rules


def _evaluate_node(node: ast.AST, values: Mapping[str, np.ndarray]) -> Any:
    if isinstance(node, ast.Expression):
        return _evaluate_node(node.body, values)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in values:
            raise FaultFeatureRuleError(f"样本特征表缺少原子特征: {node.id}")
        return values[node.id]
    if isinstance(node, ast.BinOp):
        left, right = _evaluate_node(node.left, values), _evaluate_node(node.right, values)
        operations = {
            ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply,
            ast.Div: np.divide, ast.Pow: np.power, ast.Mod: np.mod,
        }
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            return operations[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp):
        operand = _evaluate_node(node.operand, values)
        if isinstance(node.op, ast.USub):
            return np.negative(operand)
        if isinstance(node.op, ast.UAdd):
            return operand
        return np.logical_not(operand)
    if isinstance(node, ast.BoolOp):
        operands = [_evaluate_node(value, values) for value in node.values]
        operation = np.logical_and if isinstance(node.op, ast.And) else np.logical_or
        result = operands[0]
        for operand in operands[1:]:
            result = operation(result, operand)
        return result
    if isinstance(node, ast.Compare):
        compare_ops = {
            ast.Eq: np.equal, ast.NotEq: np.not_equal, ast.Lt: np.less,
            ast.LtE: np.less_equal, ast.Gt: np.greater, ast.GtE: np.greater_equal,
        }
        left = _evaluate_node(node.left, values)
        result: Any = True
        for operator, comparator in zip(node.ops, node.comparators):
            right = _evaluate_node(comparator, values)
            result = np.logical_and(result, compare_ops[type(operator)](left, right))
            left = right
        return result
    if isinstance(node, ast.Call):
        function_name = node.func.id  # validated by validate_expression
        args = [_evaluate_node(argument, values) for argument in node.args]
        if function_name == "abs" and len(args) == 1:
            return np.abs(args[0])
        if function_name in {"min", "max"} and len(args) >= 2:
            operation = np.minimum if function_name == "min" else np.maximum
            result = args[0]
            for argument in args[1:]:
                result = operation(result, argument)
            return result
        if function_name == "clip" and len(args) == 3:
            return np.clip(args[0], args[1], args[2])
        raise FaultFeatureRuleError(f"函数 {function_name} 的参数数量不正确")
    raise FaultFeatureRuleError(f"不支持的表达式节点: {type(node).__name__}")


def evaluate_approved_rules(
    base_feature_frame: pd.DataFrame,
    rules: Sequence[ApprovedRule],
) -> Tuple[pd.DataFrame, List[str]]:
    output = base_feature_frame.copy()
    values = {
        column: output[column].to_numpy(dtype=np.float64)
        for column in output.columns
        if pd.api.types.is_numeric_dtype(output[column])
    }
    configured_columns: List[str] = []
    score_columns: List[str] = []

    for rule in rules:
        missing = sorted(set(rule.input_features) - set(values))
        if missing:
            raise FaultFeatureRuleError(
                f"故障特征 {rule.feature_id} 缺少原子特征: {', '.join(missing)}"
            )
        tree, _ = validate_expression(rule.expression)
        evaluated = _evaluate_node(tree, values)
        vector = np.asarray(evaluated, dtype=np.float64)
        if vector.ndim == 0:
            vector = np.full(len(output), float(vector), dtype=np.float64)
        if vector.shape != (len(output),):
            raise FaultFeatureRuleError(f"故障特征 {rule.feature_id} 的结果维度不正确: {vector.shape}")
        vector = np.nan_to_num(vector * rule.weight, nan=0.0, posinf=0.0, neginf=0.0)
        feature_column = f"configured_feature_{rule.feature_id}"
        score_column = f"rule_score_{rule.feature_id}"
        output[feature_column] = vector
        output[score_column] = vector
        values[feature_column] = vector
        configured_columns.append(feature_column)
        score_columns.append(score_column)

    score_matrix = output[score_columns].to_numpy(dtype=np.float64)
    winner_indices = np.argmax(score_matrix, axis=1)
    output["rule_candidate_label"] = [rules[index].feature_id for index in winner_indices]
    output["rule_candidate_score"] = score_matrix[np.arange(len(output)), winner_indices]
    return output, configured_columns


def prepare_confirmation_table(description_path: Path, output_path: Path) -> pd.DataFrame:
    config = load_description_config(description_path)
    frame = compile_confirmation_frame(config)
    write_confirmation_table(frame, output_path)
    return frame
