"""
一键故障诊断脚本

输入：故障名称 + 预测数据路径
输出：预测结果JSON

整合了 prepare_ml_data（特征提取）和 predict（模型预测），
跳过中间 .pt 文件生成，直接在内存中完成 特征提取 → 预测。

使用方法:
    python diagnose.py --fault 变桨心跳 --data_path "D:\\test_data"
    python diagnose.py --fault 变桨心跳 --data_path "D:\\test.xlsx" --col_path "Tracelog路径"
    python diagnose.py --fault 变桨轴X驱动器输出错误 --data_type scada --data_path "D:\\test.xlsx" --col_path "scada_path" --col_fault_time "fault_time"
    python diagnose.py --fault 变桨轴X驱动器输出错误 --data_type scada --data_path "D:\\60009085_20260312.csv" --fault_time "2026-03-12 07:49:13"

参数说明:
    --fault      : 故障名称（必须在 configs/model_registry.yaml 中注册）
    --data_path  : 预测数据路径（CSV目录 或 Excel索引文件）
    --data_type  : 数据类型，tracelog 或 scada
    --col_path   : Excel中路径列名（目录模式下不需要）
    --col_fault_time : SCADA Excel/CSV索引表中的故障时间列名
    --fault_time : SCADA单文件或目录模式下的故障发生时刻
    --output_dir : 结果输出目录（默认 ./output）
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import yaml
import json
import torch
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime

from data_preprocessing.pipeline import FaultDiagnosisPipeline
from data_preprocessing.data_processor import DataProcessor
from models.model_factory import ModelFactory, get_model_config
from utils.fault_diagnosis_mapper import FaultDiagnosisMapper
from error_codes import ErrorCode, log_error as _log_error
import models  # 触发自动注册

project_root = os.path.dirname(os.path.abspath(__file__))

# ML模型列表
ML_MODELS_2D = ['RandomForest', 'ExtraTrees', 'SVM', 'XGBoost', 'LightGBM',
                'KNN', 'LogisticRegression']
NON_ITERATIVE_MODELS = ['RC'] + ML_MODELS_2D
SUPPORTED_DATA_TYPES = ('tracelog', 'scada')


def _load_registry(registry_path='configs/model_registry.yaml'):
    """加载故障模型注册表"""
    with open(registry_path, 'r', encoding='utf-8') as f:
        reg = yaml.safe_load(f)
    return reg.get('faults', {})


def _load_paths(data_source, col_tracelog_path):
    """从Excel或目录加载文件路径列表"""
    data_source = Path(data_source)
    if data_source.is_dir():
        paths = sorted(data_source.glob("**/*.csv")) + sorted(data_source.glob("**/*.txt"))
        return paths

    if not data_source.exists():
        return []

    # 单个 CSV/TXT 文件直接返回
    if data_source.suffix.lower() in ('.csv', '.txt'):
        return [data_source]

    df = pd.read_excel(data_source)
    if col_tracelog_path not in df.columns:
        return []

    paths = []
    for _, row in df.iterrows():
        path_str = row.get(col_tracelog_path)
        if pd.notna(path_str):
            p = Path(str(path_str).strip())
            if p.exists():
                paths.append(p)
    return paths


def _load_table(path):
    """读取 Excel/CSV 索引表。"""
    suffix = Path(path).suffix.lower()
    if suffix in ('.xlsx', '.xls'):
        return pd.read_excel(path)
    if suffix == '.csv':
        return pd.read_csv(path)
    return None


def _load_scada_samples(data_source, col_scada_path, col_fault_time, default_fault_time, fault_time=None):
    """加载 SCADA 样本，返回 [(path, fault_time), ...]。"""
    data_source = Path(data_source)
    fallback_fault_time = fault_time if fault_time not in (None, '') else default_fault_time

    if data_source.is_dir():
        suffixes = ('.csv', '.txt', '.gz', '.zip', '.rar', '.7z')
        return [(p, fallback_fault_time) for p in sorted(data_source.glob("**/*")) if p.suffix.lower() in suffixes]

    if not data_source.exists():
        return []

    table = _load_table(data_source)
    if table is not None and col_scada_path in table.columns:
        if col_fault_time not in table.columns:
            return []
        samples = []
        for _, row in table.iterrows():
            path_str = row.get(col_scada_path)
            fault_time = row.get(col_fault_time)
            if pd.notna(path_str) and pd.notna(fault_time):
                p = Path(str(path_str).strip())
                if p.exists():
                    samples.append((p, fault_time))
        return samples

    return [(data_source, fallback_fault_time)]


def _extract_tracelog_features(paths, main_channels, multi_channels, timestamp_window, default_fault_time):
    """从 tracelog 文件提取模型特征。"""
    data_processor = DataProcessor()
    pipeline = FaultDiagnosisPipeline(
        main_sensor_channels=main_channels,
        multi_analysis_channels=multi_channels if multi_channels else None,
        timestamp_window=timestamp_window,
        default_fault_time=default_fault_time,
        missing_channel_strategy='zero'
    )

    X_list = []
    valid_filenames = []

    for i, p in enumerate(paths):
        if not p.exists():
            continue
        try:
            result = data_processor.extract_columns_with_fallback(
                p,
                fault_time=default_fault_time,
                timestamp_range=timestamp_window
            )
            if isinstance(result, tuple):
                df, err_code, err_msg = result
                if err_code:
                    continue
            else:
                df = result

            if df is None or df.empty:
                continue

            feat_result = pipeline.run(df)
            if feat_result and feat_result.get('status') == 'success':
                X_list.append(feat_result['features'])
                valid_filenames.append(p.name)
        except Exception:
            continue

        if (i + 1) % 10 == 0:
            print(f"  特征提取: {i + 1}/{len(paths)}")

    return X_list, valid_filenames


def _extract_scada_features(samples, main_channels, scada_time_window):
    """从 SCADA 文件提取模型特征。"""
    from data_preprocessing.scada_pipeline import SCADA_Pipeline

    config_path = os.path.join(project_root, 'configs', 'preprocess_scada.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"SCADA配置不存在: {config_path}")

    pipeline = SCADA_Pipeline(
        config_path=config_path,
        missing_channel_strategy='zero',
        log_enable=False
    )

    time_window = tuple(scada_time_window) if scada_time_window else None
    tag_points = main_channels if main_channels else None
    X_list = []
    valid_filenames = []

    for i, (p, fault_time) in enumerate(samples):
        if not p.exists():
            continue
        try:
            feat_result = pipeline.process_single(
                scada_path=str(p),
                fault_timestamp=fault_time,
                tag_points=tag_points,
                time_window=time_window
            )
            if feat_result and feat_result.get('status') == 'success':
                X_list.append(feat_result['features'])
                valid_filenames.append(p.name)
        except Exception:
            continue

        if (i + 1) % 10 == 0:
            print(f"  特征提取: {i + 1}/{len(samples)}")

    return X_list, valid_filenames


def diagnose(
    fault_name,
    data_path,
    result_dict,
    col_tracelog_path='',
    output_dir='./output',
    data_type='tracelog',
    col_fault_time='fault_time',
    fault_time=None
):
    """
    一键故障诊断入口。

    Parameters
    ----------
    fault_name : str
        故障名称（必须在 model_registry.yaml 中注册）
    data_path : str
        预测数据路径（CSV目录 或 Excel索引文件）
    result_dict : dict
        输出容器
    col_tracelog_path : str
        Excel/CSV索引表中路径列名（目录模式下传空字符串；SCADA默认使用 scada_path）
    output_dir : str
        结果输出目录
    data_type : str
        数据类型，支持 'tracelog' 或 'scada'
    col_fault_time : str
        SCADA索引表中的故障时间列名
    fault_time : str
        SCADA单文件或目录模式下的故障发生时刻，例如 '2026-03-12 07:49:13'

    Returns
    -------
    int : 0 表示成功，非0为错误码
    """
    data_type = str(data_type).lower().strip()
    if data_type not in SUPPORTED_DATA_TYPES:
        _log_error(ErrorCode.PREDICT_PARAM_ERROR, f"data_type 必须是 {SUPPORTED_DATA_TYPES}，当前: {data_type}")
        return ErrorCode.PREDICT_PARAM_ERROR.value

    # === 1. 加载注册表 ===
    registry_path = os.path.join(project_root, 'configs', 'model_registry.yaml')
    if not os.path.exists(registry_path):
        _log_error(ErrorCode.CONFIG_MISSING, f"注册表不存在: {registry_path}")
        return ErrorCode.CONFIG_MISSING.value

    registry = _load_registry(registry_path)
    if fault_name not in registry:
        available = list(registry.keys())
        _log_error(ErrorCode.PREDICT_PARAM_ERROR, f"故障 '{fault_name}' 未注册，可选: {available}")
        return ErrorCode.PREDICT_PARAM_ERROR.value

    fault_cfg = registry[fault_name]
    model_type = fault_cfg['model_type']
    checkpoint_path = fault_cfg['checkpoint']
    mapping_path = fault_cfg.get('mapping', '')
    main_channels = fault_cfg.get('main_sensor_channels', [])
    multi_channels = fault_cfg.get('multi_analysis_channels', [])
    timestamp_window = fault_cfg.get('timestamp_window', [-2000, 2000])
    scada_time_window = fault_cfg.get('scada_time_window')
    default_fault_time = fault_cfg.get('default_fault_time', 0)
    num_classes = fault_cfg.get('num_classes', 2)

    print(f"故障: {fault_name} | 模型: {model_type} | 数据类型: {data_type} | 通道数: {len(main_channels)}")

    # === 2. 加载数据路径 ===
    if data_type == 'tracelog':
        samples = _load_paths(data_path, col_tracelog_path)
    else:
        scada_path_col = col_tracelog_path or 'scada_path'
        samples = _load_scada_samples(data_path, scada_path_col, col_fault_time, default_fault_time, fault_time)

    if not samples:
        _log_error(ErrorCode.DATA_EMPTY, f"数据源中无有效文件: {data_path}")
        return ErrorCode.DATA_EMPTY.value

    print(f"数据源: {data_path} ({len(samples)} 个文件)")

    # === 3. 特征提取（内存中，不生成pt文件） ===
    try:
        if data_type == 'tracelog':
            X_list, valid_filenames = _extract_tracelog_features(
                samples,
                main_channels,
                multi_channels,
                timestamp_window,
                default_fault_time
            )
        else:
            X_list, valid_filenames = _extract_scada_features(
                samples,
                main_channels,
                scada_time_window
            )
    except Exception as e:
        _log_error(ErrorCode.TRACELOG_PROCESS_ERROR, str(e))
        return ErrorCode.TRACELOG_PROCESS_ERROR.value

    if not X_list:
        _log_error(ErrorCode.NO_VALID_SAMPLES, "没有样本被成功处理")
        return ErrorCode.NO_VALID_SAMPLES.value

    X_array = np.array(X_list, dtype=np.float32)
    print(f"特征提取完成: {len(valid_filenames)} 样本, 维度 {X_array.shape}")

    # === 4. 加载模型 ===
    if not os.path.exists(checkpoint_path):
        _log_error(ErrorCode.CHECKPOINT_MISSING, checkpoint_path)
        return ErrorCode.CHECKPOINT_MISSING.value

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        checkpoint_config = checkpoint.get('config', None)
    except Exception as e:
        _log_error(ErrorCode.MODEL_LOAD_ERROR, str(e))
        return ErrorCode.MODEL_LOAD_ERROR.value

    # 构建模型配置
    if checkpoint_config:
        model_config = checkpoint_config
    else:
        model_config = {'num_classes': num_classes, 'input_channels': 1, 'seq_len': X_array.shape[1], 'num_channels': 1}

    model_config['batch_size'] = 128

    try:
        model_instance, _ = ModelFactory.create_model(model_type, model_config, 'cpu')
        if model_type in NON_ITERATIVE_MODELS:
            model_instance.load_checkpoint(checkpoint_path)
        else:
            model_instance.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        _log_error(ErrorCode.MODEL_LOAD_ERROR, str(e))
        return ErrorCode.MODEL_LOAD_ERROR.value

    # === 5. 执行预测 ===
    model_instance.eval()
    x_tensor = torch.from_numpy(X_array).float()

    if model_type in ML_MODELS_2D:
        # ML模型需要展平
        if len(x_tensor.shape) == 2:
            x_input = x_tensor
        else:
            x_input = x_tensor.reshape(x_tensor.shape[0], -1)

        if hasattr(model_instance, 'predict_with_proba'):
            preds, probs = model_instance.predict_with_proba(x_input)
            predictions = preds.numpy()
            probabilities = probs.numpy()
        else:
            predictions = model_instance(x_input).numpy()
            probabilities = np.eye(num_classes)[predictions]
    else:
        # 深度学习模型
        with torch.no_grad():
            output = model_instance(x_tensor)
            logits = output[0] if isinstance(output, tuple) else output
            probabilities = torch.softmax(logits, dim=-1).numpy()
            predictions = np.argmax(probabilities, axis=1)

    max_probs = probabilities.max(axis=1)
    print(f"预测完成: {len(predictions)} 样本")

    # === 6. 故障诊断映射 ===
    diagnosis_mapper = None
    if mapping_path and os.path.exists(mapping_path):
        try:
            diagnosis_mapper = FaultDiagnosisMapper(mapping_path, fault_name=fault_name)
        except Exception:
            pass

    # === 7. 组装结果 ===
    results = {
        'fault_name': fault_name,
        'model_name': model_type,
        'checkpoint': checkpoint_path,
        'data_type': data_type,
        'data_source': str(data_path),
        'num_samples': len(predictions),
        'num_classes': int(num_classes),
        'timestamp': datetime.now().isoformat(),
        'predictions': predictions.tolist(),
        'probabilities': probabilities.tolist(),
        'filenames': valid_filenames,
        'confidence_stats': {
            'mean': float(max_probs.mean()),
            'min': float(max_probs.min()),
            'max': float(max_probs.max()),
            'std': float(max_probs.std()),
        },
    }

    # 逐样本诊断
    if diagnosis_mapper and diagnosis_mapper.has_mapping():
        diag_list = {}
        for i, pred_label in enumerate(predictions):
            diagnosis = diagnosis_mapper.get_diagnosis(int(pred_label))
            entry = {
                '文件名': valid_filenames[i],
                '预测标签': int(pred_label),
                '置信度': float(probabilities[i].max()),
            }
            if diagnosis:
                entry['故障原因'] = diagnosis['故障原因']
                entry['处理方案'] = diagnosis['处理方案']
            diag_list[str(i)] = entry
        results['diagnosis'] = diag_list

    # === 8. 保存结果 ===
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"diagnosis_{fault_name}_{timestamp}.json")

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    except Exception as e:
        _log_error(ErrorCode.SAVE_RESULT_ERROR, str(e))
        return ErrorCode.SAVE_RESULT_ERROR.value

    result_dict['output_path'] = output_path
    result_dict['num_samples'] = len(predictions)
    result_dict['predictions'] = predictions.tolist()

    print(f"[OK] 诊断完成，结果保存至: {output_path}")
    return 0


# ================================================================
# 命令行入口
# ================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="一键故障诊断")
    parser.add_argument("--fault", type=str, required=True, help="故障名称")
    parser.add_argument("--data_path", type=str, required=True, help="预测数据路径（目录或Excel）")
    parser.add_argument("--data_type", type=str, default='tracelog', choices=SUPPORTED_DATA_TYPES,
                        help="数据类型: tracelog 或 scada")
    parser.add_argument("--col_path", type=str, default='', help="Excel/CSV索引表中路径列名")
    parser.add_argument("--col_fault_time", type=str, default='fault_time', help="SCADA索引表中故障时间列名")
    parser.add_argument("--fault_time", type=str, default=None, help="SCADA单文件或目录模式下的故障发生时刻")
    parser.add_argument("--output_dir", type=str, default='./output', help="结果输出目录")

    args = parser.parse_args()

    result_dict = {}
    code = diagnose(
        fault_name=args.fault,
        data_path=args.data_path,
        result_dict=result_dict,
        col_tracelog_path=args.col_path,
        output_dir=args.output_dir,
        data_type=args.data_type,
        col_fault_time=args.col_fault_time,
        fault_time=args.fault_time,
    )

    if code == 0:
        print(f"结果: {result_dict['output_path']}")
    raise SystemExit(code)
