# configs/config_loader.py (predict_standalone 精简版)
import yaml
import os
import pandas as pd
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict


def _load_yaml() -> Dict[str, Any]:
    yaml_path = Path(__file__).parent / 'preprocess_2D.yaml'
    if not yaml_path.is_file():
        raise FileNotFoundError(f"找不到配置文件: {yaml_path}")
    with yaml_path.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _resolve_placeholders(raw_cfg: Dict[str, Any]) -> Dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]

    def _replace(value):
        if isinstance(value, str):
            return value.replace("__PROJECT_ROOT__", str(project_root))
        elif isinstance(value, list):
            return [_replace(v) for v in value]
        elif isinstance(value, dict):
            return {k: _replace(v) for k, v in value.items()}
        return value

    return _replace(raw_cfg)


def _dict_to_namespace(d: Dict[str, Any]) -> SimpleNamespace:
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = _dict_to_namespace(v)
    return SimpleNamespace(**d)


def _namespace_to_dict(ns):
    if isinstance(ns, SimpleNamespace):
        return {k: _namespace_to_dict(v) for k, v in vars(ns).items()}
    elif isinstance(ns, list):
        return [_namespace_to_dict(item) for item in ns]
    return ns


# ---------- 加载配置 ----------
_raw_cfg = _load_yaml()
_resolved_cfg = _resolve_placeholders(_raw_cfg)
CONFIG = _dict_to_namespace(_resolved_cfg)


# ---------- 列名映射 ----------
def _load_column_mapping_from_excel(excel_path: str) -> Dict[str, list]:
    try:
        df = pd.read_excel(excel_path)
        mapping = {}
        for _, row in df.iterrows():
            std_name = str(row.iloc[0]).strip()
            if not std_name or std_name == 'nan':
                continue
            candidates = []
            for val in row.iloc[2:]:
                val_str = str(val).strip()
                if val_str and val_str != '1' and val_str != 'nan':
                    candidates.append(val_str)
            if candidates:
                mapping[std_name] = candidates
        return mapping
    except Exception as e:
        print(f"列名映射加载失败: {e}")
        return None


if hasattr(CONFIG, 'column_mapping_file'):
    COLUMN_MAPPING = _load_column_mapping_from_excel(CONFIG.column_mapping_file)
else:
    COLUMN_MAPPING = None


# ---------- 导出变量（仅保留 diagnose.py 依赖链所需） ----------
def __getattr__(name: str) -> Any:
    mapping = {
        "TIMESTAMP_WINDOW": tuple(CONFIG.constants.timestamp_window),
        "DEFAULT_FAULT_TIME": CONFIG.constants.default_fault_time,
        "MAIN_SENSOR_CHANNELS": CONFIG.channels.main_sensor_channels,
        "MULTI_ANALYSIS_CHANNELS": CONFIG.channels.multi_analysis_channels,
        "KEY_ROTOR_CHANNEL": CONFIG.channels.key_rotor_channel,
        "FAULT_KNOWLEDGE_FILE": '',
        "KNOWLEDGE_LABEL_COLUMN": '',
        "KNOWLEDGE_ID_COLUMN": '',
        "FEATURE_CONFIG": _namespace_to_dict(CONFIG.feature_extraction),
    }

    if name in mapping:
        return mapping[name]

    raise AttributeError(f"module 'configs.config_loader' has no attribute '{name}'")
