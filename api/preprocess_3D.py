"""
数据预处理模块（C 语言接口版）

对外暴露两个核心函数:
    build_overrides(...)  →  构造自定义参数字典（供工程师按需填写）
    preprocess(config_file, overrides)  →  执行预处理

调用示例:
    # 1. 纯配置文件，不覆盖任何参数
    preprocess(config_file="preprocess.yaml")

    # 2. 配置文件 + 覆盖部分参数
    overrides = build_overrides(max_seq_len=6000, output_dir="/tmp/out")
    preprocess(config_file="preprocess.yaml", overrides=overrides)

    # 3. 完全不用配置文件，全靠 overrides
    overrides = build_overrides(
        label_file="/data/label.csv",
        output_dir="/data/out",
        max_seq_len=8000,
    )
    preprocess(overrides=overrides)

参数优先级（高 → 低）:
    1. overrides 中的参数（最高优先级，覆盖配置文件）
    2. config_file 指定的 YAML 配置文件
    3. 代码内置默认值（_DEFAULT_CONFIG，两者都缺失时兜底）

错误编码:
    1001 CONFIG_LOAD_ERROR      配置文件加载失败
    1002 CONFIG_OVERRIDE_ERROR  overrides 参数应用异常
    1003 PREPROCESS_INIT_ERROR  DataPreprocessor 初始化失败
    1004 PROCESS_ERROR          数据预处理过程出错
    1005 SPLIT_NORMALIZE_ERROR  划分或标准化数据时出错
    1006 SAVE_DATASET_ERROR     保存 .pt 文件失败
    1999 UNKNOWN_ERROR          未知错误
"""

import argparse
import yaml
from copy import deepcopy
from enum import Enum
from data_preprocessing.preprocessor import DataPreprocessor


# ================================================================
# 错误代码
# ================================================================
class ErrorCode(Enum):
    CONFIG_LOAD_ERROR     = 1001
    CONFIG_OVERRIDE_ERROR = 1002
    PREPROCESS_INIT_ERROR = 1003
    PROCESS_ERROR         = 1004
    SPLIT_NORMALIZE_ERROR = 1005
    SAVE_DATASET_ERROR    = 1006
    UNKNOWN_ERROR         = 1999


_error_descriptions = {
    ErrorCode.CONFIG_LOAD_ERROR:     "配置文件加载失败，例如路径不存在或格式错误",
    ErrorCode.CONFIG_OVERRIDE_ERROR: "overrides 参数应用异常",
    ErrorCode.PREPROCESS_INIT_ERROR: "创建 DataPreprocessor 实例时出错，可能配置缺失",
    ErrorCode.PROCESS_ERROR:         "预处理过程中发生错误（读取/转换/检查数据）",
    ErrorCode.SPLIT_NORMALIZE_ERROR: "划分或标准化数据时出错",
    ErrorCode.SAVE_DATASET_ERROR:    "保存数据集到磁盘失败（路径或权限问题）",
    ErrorCode.UNKNOWN_ERROR:         "未知错误，请检查日志",
}


def _log_error(code: ErrorCode, message: str = "") -> None:
    desc = _error_descriptions.get(code, "")
    print(f"错误代码: {code.value} ({code.name}) -> {desc}")
    if message:
        print(f"详细信息: {message}")


# ================================================================
# 内置默认配置（兜底）
# ================================================================
_DEFAULT_CONFIG: dict = {
    "paths": {
        "label_file":  "",
        "config_file": "",
        "output_dir":  "",
    },
    "features": {
        "target_tags": [
            "grRotorSpeedFormSpeedRelay1",
            "grRotorSpeedFormSpeedRelay2",
            "grRotorSpeedFromCounterModule1",
            "rotor_speed",
        ]
    },
    "processing": {
        "max_seq_len":          12000,
        "missing_value_method": "interpolate",
        "padding_mode":         "constant",
    },
    "normalization": {
        "method":         "zscore",
        "save_stats":     True,
        "stats_filename": "normalization_stats.npz",
    },
    "split": {
        "test_size":      0.4,
        "val_test_split": 0.5,
        "random_state":   42,
        "stratify":       True,
    },
    "output": {
        "train_file": "train.pt",
        "val_file":   "val.pt",
        "test_file":  "test.pt",
    },
    "logging": {
        "verbose":  True,
        "save_log": True,
        "log_file": "preprocessing.log",
    },
}


# ================================================================
# 扁平参数名 → 嵌套配置路径映射
# ================================================================
_PARAM_MAP: dict[str, tuple[str, str]] = {
    # paths
    "label_file":           ("paths",         "label_file"),
    "output_dir":           ("paths",         "output_dir"),
    # features
    "target_tags":          ("features",      "target_tags"),
    # processing
    "max_seq_len":          ("processing",    "max_seq_len"),
    "missing_value_method": ("processing",    "missing_value_method"),
    "padding_mode":         ("processing",    "padding_mode"),
    # normalization
    "norm_method":          ("normalization", "method"),
    "save_stats":           ("normalization", "save_stats"),
    "stats_filename":       ("normalization", "stats_filename"),
    # split
    "test_size":            ("split",         "test_size"),
    "val_test_split":       ("split",         "val_test_split"),
    "random_state":         ("split",         "random_state"),
    "stratify":             ("split",         "stratify"),
    # output
    "train_file":           ("output",        "train_file"),
    "val_file":             ("output",        "val_file"),
    "test_file":            ("output",        "test_file"),
    # logging
    "verbose":              ("logging",       "verbose"),
    "save_log":             ("logging",       "save_log"),
    "log_file":             ("logging",       "log_file"),
}


# ================================================================
# 工具函数（内部）
# ================================================================
def _merge_configs(base: dict, override: dict) -> dict:
    """深度合并，override 的值覆盖 base。"""
    result = deepcopy(base)
    for section, values in override.items():
        if section in result and isinstance(result[section], dict) and isinstance(values, dict):
            result[section].update(values)
        else:
            result[section] = deepcopy(values)
    return result


def _apply_overrides(config: dict, overrides: dict) -> dict:
    """将 overrides 字典中的扁平参数写入嵌套配置，仅处理非 None 的项。"""
    config = deepcopy(config)
    for flat_key, value in overrides.items():
        if value is None:
            continue
        if flat_key not in _PARAM_MAP:
            print(f"[警告] overrides 中存在未知参数 '{flat_key}'，已忽略。")
            continue
        section, key = _PARAM_MAP[flat_key]
        config.setdefault(section, {})[key] = value
    return config


# ================================================================
# 对外函数 1：build_overrides —— 帮助工程师构造参数字典
# ================================================================
def build_overrides(
    # ── 路径 ──────────────────────────────────────────────────────
    label_file:             str   = None,   # 标签文件路径
    output_dir:             str   = None,   # 输出目录
    # ── 特征 ──────────────────────────────────────────────────────
    target_tags:            list  = None,   # 目标特征标签列表
    # ── 数据处理 ──────────────────────────────────────────────────
    max_seq_len:            int   = None,   # 统一时间步数
    missing_value_method:   str   = None,   # 空值处理: interpolate/ffill/bfill/mean/zero
    padding_mode:           str   = None,   # 填充模式: constant/edge/reflect
    # ── 标准化 ────────────────────────────────────────────────────
    norm_method:            str   = None,   # 标准化方法: zscore/minmax/robust/none
    save_stats:             bool  = None,   # 是否保存标准化统计量
    stats_filename:         str   = None,   # 统计量保存文件名
    # ── 数据集划分 ────────────────────────────────────────────────
    test_size:              float = None,   # 测试+验证集比例
    val_test_split:         float = None,   # 验证集在(测试+验证)中的比例
    random_state:           int   = None,   # 随机种子
    stratify:               bool  = None,   # 是否按标签分层抽样
    # ── 输出文件名 ────────────────────────────────────────────────
    train_file:             str   = None,   # 训练集文件名
    val_file:               str   = None,   # 验证集文件名
    test_file:              str   = None,   # 测试集文件名
    # ── 日志 ──────────────────────────────────────────────────────
    verbose:                bool  = None,   # 是否详细输出
    save_log:               bool  = None,   # 是否保存日志
    log_file:               str   = None,   # 日志文件名
) -> tuple[int, dict]:
    """
    构造自定义参数字典，传入 preprocess() 的 overrides 参数。

    只需填写想要覆盖的参数，未填写的项保持为 None，
    preprocess() 会自动从配置文件或内置默认值中取值。

    Returns
    -------
    dict : 仅包含非 None 参数的字典，可直接传给 preprocess(overrides=...)。

    Examples
    --------
    >>> overrides = build_overrides(max_seq_len=6000, output_dir="/tmp/out")
    >>> preprocess(config_file="preprocess.yaml", overrides=overrides)
    """
    all_params = {
        "label_file":           label_file,
        "output_dir":           output_dir,
        "target_tags":          target_tags,
        "max_seq_len":          max_seq_len,
        "missing_value_method": missing_value_method,
        "padding_mode":         padding_mode,
        "norm_method":          norm_method,
        "save_stats":           save_stats,
        "stats_filename":       stats_filename,
        "test_size":            test_size,
        "val_test_split":       val_test_split,
        "random_state":         random_state,
        "stratify":             stratify,
        "train_file":           train_file,
        "val_file":             val_file,
        "test_file":            test_file,
        "verbose":              verbose,
        "save_log":             save_log,
        "log_file":             log_file,
    }
    # 类型检查，任何不匹配的参数都会被视为错误并返回错误码
    expected_types = {
        "label_file": str,
        "output_dir": str,
        "target_tags": list,
        "max_seq_len": int,
        "missing_value_method": str,
        "padding_mode": str,
        "norm_method": str,
        "save_stats": bool,
        "stats_filename": str,
        "test_size": float,
        "val_test_split": float,
        "random_state": int,
        "stratify": bool,
        "train_file": str,
        "val_file": str,
        "test_file": str,
        "verbose": bool,
        "save_log": bool,
        "log_file": str,
    }
    for k, v in all_params.items():
        if v is None:
            continue
        expected = expected_types.get(k)
        if expected and not isinstance(v, expected):
            print(f"[错误] 参数 '{k}' 类型应为 {expected.__name__}，但传入 {type(v).__name__}")
            return ErrorCode.CONFIG_OVERRIDE_ERROR.value, {}
    # overrides = {k: v for k, v in all_params.items() if v is not None}
    # 仅保留非 None 项，调用方可直观看到哪些参数被覆盖
    return 0, {k: v for k, v in all_params.items() if v is not None}


# ================================================================
# 对外函数 2：preprocess —— 预处理入口
# ================================================================
def preprocess(
    config_file: str  = "",
    overrides:   dict = None,
) -> int:
    """
    数据预处理主函数。

    Parameters
    ----------
    config_file : str
        YAML 配置文件路径。为空字符串时跳过文件加载，仅用内置默认值兜底。
    overrides : dict, optional
        自定义参数字典，由 build_overrides() 构造。
        其中的参数会覆盖 config_file 中的对应值。
        为 None 或空字典时表示完全使用配置文件。

    Returns
    -------
    int : 0 表示成功；非 0 为对应错误代码（见模块文档）。

    Examples
    --------
    # 纯配置文件
    preprocess(config_file="preprocess.yaml")

    # 配置文件 + 覆盖
    preprocess(config_file="preprocess.yaml", overrides=build_overrides(max_seq_len=6000))

    # 完全不用配置文件
    preprocess(overrides=build_overrides(label_file="/data/label.csv", output_dir="/data/out"))
    """
    if overrides is None:
        overrides = {}

    # ── Step 1: 内置默认值 ──────────────────────────────────────
    config = deepcopy(_DEFAULT_CONFIG)

    # ── Step 2: 配置文件覆盖（第二优先级）──────────────────────
    if config_file:
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                yaml_cfg = yaml.safe_load(f)
            config = _merge_configs(config, yaml_cfg)
        except Exception as e:
            _log_error(ErrorCode.CONFIG_LOAD_ERROR, str(e))
            return ErrorCode.CONFIG_LOAD_ERROR.value

    # ── Step 3: overrides 覆盖（最高优先级）────────────────────
    try:
        config = _apply_overrides(config, overrides)
    except Exception as e:
        _log_error(ErrorCode.CONFIG_OVERRIDE_ERROR, str(e))
        return ErrorCode.CONFIG_OVERRIDE_ERROR.value

    if config.get("logging", {}).get("verbose"):
        print("=" * 60)
        print("[信息] 最终生效配置:")
        for section, values in config.items():
            print(f"  [{section}]")
            if isinstance(values, dict):
                for k, v in values.items():
                    print(f"    {k}: {v}")
        print("=" * 60)

    # ── Step 4: 执行预处理 ──────────────────────────────────────
    try:
        preprocessor = DataPreprocessor(config)
    except Exception as e:
        _log_error(ErrorCode.PREPROCESS_INIT_ERROR, str(e))
        return ErrorCode.PREPROCESS_INIT_ERROR.value

    try:
        X, y = preprocessor.process()
    except Exception as e:
        _log_error(ErrorCode.PROCESS_ERROR, str(e))
        return ErrorCode.PROCESS_ERROR.value

    try:
        data_splits = preprocessor.split_and_normalize(X, y)
    except Exception as e:
        _log_error(ErrorCode.SPLIT_NORMALIZE_ERROR, str(e))
        return ErrorCode.SPLIT_NORMALIZE_ERROR.value

    try:
        preprocessor.save_datasets(data_splits)
    except Exception as e:
        _log_error(ErrorCode.SAVE_DATASET_ERROR, str(e))
        return ErrorCode.SAVE_DATASET_ERROR.value

    print("\n" + "=" * 60)
    print("预处理完成!")
    print("=" * 60)
    return 0


# ================================================================
# 命令行入口（C 工程师也可用 subprocess 调用）
# ================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TS-TCC 数据预处理",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config",               type=str,   default="",   help="YAML 配置文件路径")
    parser.add_argument("--label_file",           type=str,   default=None)
    parser.add_argument("--output_dir",           type=str,   default=None)
    parser.add_argument("--target_tags",          type=str,   nargs="+",    default=None)
    parser.add_argument("--max_seq_len",          type=int,   default=None)
    parser.add_argument("--missing_value_method", type=str,   default=None)
    parser.add_argument("--padding_mode",         type=str,   default=None)
    parser.add_argument("--norm_method",          type=str,   default=None)
    parser.add_argument("--save_stats",           type=lambda x: x.lower() == "true", default=None)
    parser.add_argument("--stats_filename",       type=str,   default=None)
    parser.add_argument("--test_size",            type=float, default=None)
    parser.add_argument("--val_test_split",       type=float, default=None)
    parser.add_argument("--random_state",         type=int,   default=None)
    parser.add_argument("--stratify",             type=lambda x: x.lower() == "true", default=None)
    parser.add_argument("--train_file",           type=str,   default=None)
    parser.add_argument("--val_file",             type=str,   default=None)
    parser.add_argument("--test_file",            type=str,   default=None)
    parser.add_argument("--verbose",              type=lambda x: x.lower() == "true", default=None)
    parser.add_argument("--save_log",             type=lambda x: x.lower() == "true", default=None)
    parser.add_argument("--log_file",             type=str,   default=None)

    args = parser.parse_args()
    overrides = build_overrides(
        label_file=args.label_file,
        output_dir=args.output_dir,
        target_tags=args.target_tags,
        max_seq_len=args.max_seq_len,
        missing_value_method=args.missing_value_method,
        padding_mode=args.padding_mode,
        norm_method=args.norm_method,
        save_stats=args.save_stats,
        stats_filename=args.stats_filename,
        test_size=args.test_size,
        val_test_split=args.val_test_split,
        random_state=args.random_state,
        stratify=args.stratify,
        train_file=args.train_file,
        val_file=args.val_file,
        test_file=args.test_file,
        verbose=args.verbose,
        save_log=args.save_log,
        log_file=args.log_file,
    )
    raise SystemExit(preprocess(config_file=args.config, overrides=overrides))
