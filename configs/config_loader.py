# src/config_loader.py
import yaml
import os
import pandas as pd
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict


# ---------- 读取 YAML ----------
def _load_yaml() -> Dict[str, Any]:
    yaml_path = Path(__file__).parent / 'preprocess_2D.yaml'
    print(yaml_path)
    if not yaml_path.is_file():
        raise FileNotFoundError(f"找不到配置文件: {yaml_path}")

    with yaml_path.open('r', encoding='utf-8') as f:
        raw_cfg = yaml.safe_load(f)
    return raw_cfg


def _resolve_placeholders(raw_cfg: Dict[str, Any]) -> Dict[str, Any]:
    # 项目根目录：src/../（即 repo 根）
    project_root = Path(__file__).resolve().parents[1]

    def _replace(value):
        if isinstance(value, str):
            # 1) 项目根占位符
            value = value.replace("__PROJECT_ROOT__", str(project_root))
            # 2) Windows UNC 路径处理
            return value
        elif isinstance(value, list):
            return [_replace(v) for v in value]
        elif isinstance(value, dict):
            return {k: _replace(v) for k, v in value.items()}
        else:
            return value

    return _replace(raw_cfg)


# ---------- 将 dict 包装为对象 ----------
def _dict_to_namespace(d: Dict[str, Any]) -> SimpleNamespace:
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = _dict_to_namespace(v)
    return SimpleNamespace(**d)


# ---------- 对外提供的单例 ----------
_raw_cfg = _load_yaml()
_resolved_cfg = _resolve_placeholders(_raw_cfg)
CONFIG = _dict_to_namespace(_resolved_cfg)

# ---------- 导出列名映射 ----------
def _namespace_to_dict(ns):
    """将 SimpleNamespace 转换为 dict"""
    if isinstance(ns, SimpleNamespace):
        return {k: _namespace_to_dict(v) for k, v in vars(ns).items()}
    elif isinstance(ns, list):
        return [_namespace_to_dict(item) for item in ns]
    else:
        return ns


def _safe_getattr(obj, attr, default=None):
    """安全获取属性，缺失时返回默认值。"""
    return getattr(obj, attr, default) if obj is not None else default


def _load_column_mapping_from_excel(excel_path: str) -> Dict[str, list]:
    """
    从Excel文件加载列名映射表
    
    Excel格式：
        tags | Chinese_tags | config1 | config2 | ... | config14
        第一列为标准列名，后续列为候选列名，值为'1'表示占位（忽略）
    
    Returns:
        dict: {标准列名: [候选列名列表]}
    """
    try:
        df = pd.read_excel(excel_path)
        mapping = {}
        
        for _, row in df.iterrows():
            std_name = str(row.iloc[0]).strip()  # tags列
            if not std_name or std_name == 'nan':
                continue
            
            # 从config1~config14收集候选列名，跳过值为'1'或空的
            candidates = []
            for val in row.iloc[2:]:  # 跳过tags和Chinese_tags
                val_str = str(val).strip()
                if val_str and val_str != '1' and val_str != 'nan':
                    candidates.append(val_str)
            
            if candidates:
                mapping[std_name] = candidates
        
        print(f"✅ 从Excel加载列名映射: {len(mapping)} 个标准列名")
        return mapping
        
    except Exception as e:
        print(f"⚠️ 从Excel加载列名映射失败: {e}")
        print(f"   文件路径: {excel_path}")
        return None


# 优先从Excel加载，否则从yaml加载
if hasattr(CONFIG, 'column_mapping_file'):
    COLUMN_MAPPING = _load_column_mapping_from_excel(CONFIG.column_mapping_file)
elif hasattr(CONFIG, 'column_mapping'):
    COLUMN_MAPPING = _namespace_to_dict(CONFIG.column_mapping)
else:
    COLUMN_MAPPING = None


# ---------- 兼容性映射 (__getattr__) ----------
def __getattr__(name: str) -> Any:


    # 动态获取当前的项目根路径（以防 config.yaml 里没用占位符）
    project_root = Path(__file__).resolve().parents[1]

    channels_cfg = _safe_getattr(CONFIG, 'channels')
    excel_cfg = _safe_getattr(CONFIG, 'excel')
    knowledge_cfg = _safe_getattr(CONFIG, 'knowledge')
    data_split_cfg = _safe_getattr(CONFIG, 'data_split')
    data_proc_cfg = _safe_getattr(CONFIG, 'data_processing')
    feature_cfg = _safe_getattr(CONFIG, 'feature_extraction')
    training_cfg = _safe_getattr(CONFIG, 'training')
    inference_cfg = _safe_getattr(CONFIG, 'inference')
    constants_cfg = _safe_getattr(CONFIG, 'constants')

    mapping = {
        # === 基础路径 ===
        "PROJECT_ROOT": project_root,

        # === 数据源 ===
        "DATA_SOURCE_EXCEL": CONFIG.data.source_excel,
        "DATA_SOURCE_TEST_EXCEL": CONFIG.data.source_test_excel,
        "PROCESSED_DATA_DIR": CONFIG.data.processed_dir,
        "REPORTS_DIR": CONFIG.data.reports_dir,

        # === 输出结果路径 ===
        "PREDICTION_RESULT_CSV": Path(CONFIG.output.prediction_csv),
        "PREDICTION_REPORT_IMG": Path(CONFIG.output.prediction_report_img),
        "CV_DATASET_FILENAME": "cv_dataset.pt",  # 保存的训练集名称
        "TEST_SET_FILENAME": "test_set.pt",      # 保存的测试集名称

        # === 模型路径 ===
        # 如果配置文件里写的是相对路径 "./models"，则这里拼接成绝对路径
        "MODEL_SAVE_DIR": (project_root / CONFIG.model.save_dir).resolve(),

        "MODEL_FILENAME": CONFIG.model.filename,

        # 为了防止旧代码报错，可以设为 None
        "SCALER_FILENAME": None,
        "ENCODER_FILENAME": None,

        # === 向量数据库 ===
        "QDRANT_HOST": CONFIG.db.host,
        "QDRANT_PORT": CONFIG.db.port,
        "COLLECTION_NAME": CONFIG.db.collection_name,
        "BATCH_UPLOAD_SIZE": CONFIG.db.batch_upload_size,

        # === 常量 ===
        "TIMESTAMP_WINDOW": tuple(_safe_getattr(constants_cfg, 'timestamp_window', (-2000, 2000))),
        "DEFAULT_FAULT_TIME": _safe_getattr(constants_cfg, 'default_fault_time', 0),

        # === 传感器通道 ===
        "MAIN_SENSOR_CHANNELS": _safe_getattr(channels_cfg, 'main_sensor_channels', []),
        "MULTI_ANALYSIS_CHANNELS": _safe_getattr(channels_cfg, 'multi_analysis_channels', []),
        "KEY_ROTOR_CHANNEL": _safe_getattr(channels_cfg, 'key_rotor_channel', None),

        # === Excel 列名 ===
        "COL_TRACELOG_PATH": _safe_getattr(excel_cfg, 'trace_path_column', None),
        "COL_LABEL": _safe_getattr(excel_cfg, 'label_column', None),
        "COL_FAULT_REASON": _safe_getattr(excel_cfg, 'fault_reason_column', None),
        "COL_FAULT_TIME": _safe_getattr(excel_cfg, 'fault_time_column', None),

        # === 预测使用的配置表 ===
        "FAULT_KNOWLEDGE_FILE": _safe_getattr(knowledge_cfg, 'fault_knowledge_file', None),
        "KNOWLEDGE_LABEL_COLUMN": _safe_getattr(knowledge_cfg, 'label_column', None),
        "KNOWLEDGE_ID_COLUMN": _safe_getattr(knowledge_cfg, 'id_column', None),

        # === 数据划分配置 ===
        "TEST_SIZE": _safe_getattr(data_split_cfg, 'test_size', None),
        "VAL_SIZE": _safe_getattr(data_split_cfg, 'val_size', None),
        "RANDOM_SEED": _safe_getattr(data_split_cfg, 'random_seed', 42),

        # === 数据处理参数 ===
        "MIN_ROWS_AFTER_WINDOW": _safe_getattr(data_proc_cfg, 'min_rows_after_window', None),
        "TIME_OFFSET_THRESHOLD": _safe_getattr(data_proc_cfg, 'time_offset_threshold', None),
        "COLUMN_OFFSET_CHECK_RANGE": _safe_getattr(data_proc_cfg, 'column_offset_check_range', None),
        "TIME_VALUE_THRESHOLD": _safe_getattr(data_proc_cfg, 'time_value_threshold', None),

        # === 特征提取配置 ===
        "FEATURE_CONFIG": _namespace_to_dict(feature_cfg) if feature_cfg is not None else {},

        # === 模型训练配置 ===
        "TRAINING_CONFIG": _namespace_to_dict(training_cfg) if training_cfg is not None else {},

        # === 模型推理配置 ===
        "INFERENCE_DEVICE": _safe_getattr(inference_cfg, 'device', 'cpu'),
        "INFERENCE_WEIGHTS_ONLY": _safe_getattr(inference_cfg, 'weights_only', False),

        # === 兼容旧代码 ===
        "GENERATE_PT_ONLY": True,  # 是否仅生成PT文件
        "N_FOLDS": 5,  # 交叉验证折数（已废弃，使用train/val/test划分）
    }

    if name in mapping:
        return mapping[name]

    raise AttributeError(f"module 'src.config_loader' has no attribute '{name}'")
