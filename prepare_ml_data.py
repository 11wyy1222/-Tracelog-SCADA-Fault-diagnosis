import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import pandas as pd
import numpy as np
import warnings
import logging
import torch
from pathlib import Path
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from data_preprocessing.pipeline import FaultDiagnosisPipeline
from data_preprocessing.data_processor import DataProcessor
from error_codes import ErrorCode
from utils.logger import preprocess_logger

warnings.filterwarnings('ignore')


def prepare_ml_data(dict_para, result_dict) -> ErrorCode:
    """
    接口名称: prepare_ml_data

    功能说明:
    预测数据预处理主流程，从Excel索引或目录读取原始数据路径，
    批量加载CSV/TXT并提取传感器特征，将特征矩阵保存为.pt格式文件，
    同时生成文件名映射CSV用于结果追溯。

    Args:
        dict_para: 参数字典，包含所有输入配置（详见API文档）
        result_dict: 输出结果字典（由函数填充）

    Returns:
        ErrorCode: 错误码
    """
    log_dir = os.path.join(project_root, 'logs', datetime.now().strftime('%Y%m%d_%H%M%S'), 'dataprocessing')
    logger = preprocess_logger(log_dir, log_name="prepare_ml_data")

    # === 参数解析 ===
    # 数据源
    data_source = Path(dict_para.get('source_test_excel', ''))
    col_tracelog_path = dict_para.get('col_tracelog_path', '')
    # 输出
    output_dir = Path(dict_para.get('output_dir', './external_data/ml_data'))
    output_filename = dict_para.get('output_filename', 'prediction_features.pt')
    output_dir.mkdir(parents=True, exist_ok=True)

    save_pt_path = output_dir / output_filename
    mapping_path = output_dir / output_filename.replace('.pt', '_filenames.csv')

    # Pipeline 参数
    main_channels = dict_para.get('main_sensor_channels', None)
    timestamp_window = dict_para.get('timestamp_window', None)
    default_fault_time = dict_para.get('default_fault_time', None)
    missing_strategy = dict_para.get('missing_channel_strategy', 'zero')
    column_mapping_file = dict_para.get('column_mapping_file', None)

    logger.info(f"开始批量数据处理流程")
    logger.info(f"数据源: {data_source}")
    logger.info(f"输出路径: {save_pt_path}")

    # === 步骤1：加载数据源路径 ===
    try:
        paths = _load_paths(data_source, col_tracelog_path, logger)
    except Exception as e:
        logger.error(f"数据源读取失败: {e}")
        return ErrorCode.EXCEL_READ_ERROR

    if not paths:
        logger.error("数据源中无有效路径")
        return ErrorCode.DATA_EMPTY

    # === 步骤2：逐个读取CSV并提取特征 ===
    data_processor = DataProcessor()
    pipeline = FaultDiagnosisPipeline(
        main_sensor_channels=main_channels,
        timestamp_window=timestamp_window,
        default_fault_time=default_fault_time,
        missing_channel_strategy=missing_strategy
    )
    X_list = []
    valid_filenames = []

    logger.info(f"开始加载数据并提取特征 (共 {len(paths)} 个样本)...")

    for i, p in enumerate(paths):
        if not p.exists():
            logger.warning(f"文件不存在: {p}")
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
                    logger.warning(f"读取跳过 {p.name}: [{err_code}] {err_msg}")
                    continue
            else:
                df = result

            if df is None or df.empty:
                continue

            feat_result = pipeline.run(df)
            if feat_result is not None and feat_result.get('status') == 'success':
                X_list.append(feat_result['features'])
                valid_filenames.append(p.name)
            else:
                logger.warning(f"特征提取失败: {p.name}")

        except Exception as e:
            logger.warning(f"处理异常 {p.name}: {e}")

        if (i + 1) % 10 == 0:
            logger.info(f"  进度: {i + 1}/{len(paths)}")

    if not X_list:
        logger.error("没有样本被成功处理")
        return ErrorCode.NO_VALID_SAMPLES

    X_array = np.array(X_list)
    logger.info(f"特征提取完成: {len(valid_filenames)} 个样本, 维度 {X_array.shape}")

    # === 步骤3：保存结果 ===
    try:
        samples_tensor = torch.from_numpy(X_array.astype(np.float32))
        torch.save({'samples': samples_tensor}, save_pt_path)
        logger.info(f"预测特征已保存至: {save_pt_path}")

        df_mapping = pd.DataFrame({
            'index': range(len(valid_filenames)),
            'filename': valid_filenames
        })
        df_mapping.to_csv(mapping_path, index=False)
        logger.info(f"文件名映射已保存至: {mapping_path}")

    except Exception as e:
        logger.error(f"保存文件失败: {e}")
        return ErrorCode.PROCESS_ERROR

    # === 填充结果字典 ===
    result_dict['pt_path'] = str(save_pt_path)
    result_dict['mapping_path'] = str(mapping_path)
    result_dict['sample_count'] = len(valid_filenames)
    result_dict['feature_dim'] = X_array.shape[1] if X_array.ndim > 1 else 0

    logger.info("数据处理完成")
    return ErrorCode.SUCCESS


def _load_paths(data_source, col_tracelog_path, logger):
    """从Excel或目录加载CSV文件路径列表"""
    if data_source.is_dir():
        logger.info(f"输入是目录，扫描所有CSV文件: {data_source}")
        paths = list(data_source.glob("**/*.csv"))
        logger.info(f"扫描到 {len(paths)} 个CSV文件")
        return paths

    if not data_source.exists():
        logger.error(f"找不到数据源文件: {data_source}")
        return []

    df = pd.read_excel(data_source)

    if col_tracelog_path not in df.columns:
        logger.error(f"Excel中未找到列名: '{col_tracelog_path}'")
        logger.error(f"当前可用列: {list(df.columns)}")
        return []

    paths = []
    for _, row in df.iterrows():
        path_str = row.get(col_tracelog_path)
        if pd.notna(path_str):
            path_obj = Path(str(path_str).strip())
            if path_obj.exists():
                paths.append(path_obj)
            else:
                logger.warning(f"文件路径无效: {path_str}")

    logger.info(f"Excel读取完成: {len(paths)} 个有效路径")
    return paths


if __name__ == "__main__":
    from configs.config_loader import (
        DATA_SOURCE_TEST_EXCEL, MODEL_SAVE_DIR,
        COL_TRACELOG_PATH, MAIN_SENSOR_CHANNELS,
        TIMESTAMP_WINDOW, DEFAULT_FAULT_TIME
    )

    dict_para = {
        'source_test_excel': DATA_SOURCE_TEST_EXCEL,
        'col_tracelog_path': COL_TRACELOG_PATH,
        'output_dir': MODEL_SAVE_DIR,
        'output_filename': 'prediction_features.pt',
        'main_sensor_channels': MAIN_SENSOR_CHANNELS,
        'timestamp_window': TIMESTAMP_WINDOW,
        'default_fault_time': DEFAULT_FAULT_TIME,
        'missing_channel_strategy': 'zero',
    }

    result_dict = {}
    error_code = prepare_ml_data(dict_para, result_dict)
    print(f"返回码: {error_code}")
    print(f"结果: {result_dict}")
