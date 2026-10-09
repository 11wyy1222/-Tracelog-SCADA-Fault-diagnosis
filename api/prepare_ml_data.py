"""
2D数据预处理模块（预测数据）

对外暴露核心函数:
    prepare_ml_data(dict_para, result_dict, progress_callback=None) → int

调用示例:
    result_dict = {}
    error_code = prepare_ml_data(
        dict_para={
            'source_test_excel': r'\\server\data\test.xlsx',
            'col_tracelog_path': 'Tracelog路径',
            'output_dir': './external_data/ml_data',
            'main_sensor_channels': ['rotor_speed', 'generator_speed'],
        },
        result_dict=result_dict,
    )

参数说明:
    dict_para       : dict — 所有输入参数（平铺）
    result_dict     : dict — 输出容器
    progress_callback : callable, optional — 进度回调函数
        签名: progress_callback(current, total, phase, message)

错误编码 (1100-1199):
    1100 SUCCESS           成功
    1102 DATA_EMPTY        数据为空
    1004 PROCESS_ERROR     处理过程出错
    1150 NO_VALID_SAMPLES  没有样本被成功处理
    1170 EXCEL_READ_ERROR  读取数据源失败
"""

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

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data_preprocessing.pipeline import FaultDiagnosisPipeline
from data_preprocessing.data_processor import DataProcessor
from error_codes import ErrorCode
from utils.logger import preprocess_logger

warnings.filterwarnings('ignore')


def _noop_callback(current, total, phase, message):
    pass


class _CallbackLogHandler(logging.Handler):
    """将 logging 日志桥接到 progress_callback"""
    
    LEVEL_MAP = {
        logging.DEBUG: 'debug',
        logging.INFO: 'log_info',
        logging.WARNING: 'log_warn',
        logging.ERROR: 'log_error',
        logging.CRITICAL: 'log_critical',
    }
    
    def __init__(self, callback):
        super().__init__()
        self.callback = callback
    
    def emit(self, record):
        try:
            phase = self.LEVEL_MAP.get(record.levelno, 'log_info')
            msg = self.format(record)
            self.callback(0, 0, phase, msg)
        except Exception:
            pass


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
        return []

    paths = []
    for _, row in df.iterrows():
        path_str = row.get(col_tracelog_path)
        if pd.notna(path_str):
            path_obj = Path(str(path_str).strip())
            if path_obj.exists():
                paths.append(path_obj)
    logger.info(f"Excel读取完成: {len(paths)} 个有效路径")
    return paths


def prepare_ml_data(dict_para, result_dict, progress_callback=None) -> int:
    """
    预测数据预处理入口。

    Parameters
    ----------
    dict_para : dict
        所有输入参数（平铺），详见API文档。
    result_dict : dict
        输出容器。成功后写入 pt_path/mapping_path/sample_count/feature_dim。
    progress_callback : callable, optional
        进度回调函数，签名: (current, total, phase, message)

    Returns
    -------
    int : 1100 (SUCCESS) 表示成功；其他为错误代码。
    """
    cb = progress_callback or _noop_callback

    log_dir = os.path.join(_PROJECT_ROOT, 'logs', datetime.now().strftime('%Y%m%d_%H%M%S'), 'dataprocessing')
    logger = preprocess_logger(log_dir, log_name="prepare_ml_data")

    # 将日志桥接到回调
    _cb_handler = _CallbackLogHandler(cb)
    _cb_handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    logging.getLogger().addHandler(_cb_handler)

    # === 参数解析 ===
    data_source = Path(dict_para.get('source_test_excel', ''))
    col_tracelog_path = dict_para.get('col_tracelog_path', '')
    output_dir = Path(dict_para.get('output_dir', './external_data/ml_data'))
    output_filename = dict_para.get('output_filename', 'prediction_features.pt')
    output_dir.mkdir(parents=True, exist_ok=True)

    save_pt_path = output_dir / output_filename
    mapping_path = output_dir / output_filename.replace('.pt', '_filenames.csv')

    main_channels = dict_para.get('main_sensor_channels', None)
    timestamp_window = dict_para.get('timestamp_window', None)
    default_fault_time = dict_para.get('default_fault_time', None)
    missing_strategy = dict_para.get('missing_channel_strategy', 'zero')

    cb(0, 0, 'load', f'数据源: {data_source}')
    logger.info(f"数据源: {data_source}")

    # === 步骤1：加载路径 ===
    try:
        paths = _load_paths(data_source, col_tracelog_path, logger)
    except Exception as e:
        msg = f"数据源读取失败: {e}"
        logger.error(msg)
        cb(0, 0, 'error', msg)
        logging.getLogger().removeHandler(_cb_handler)
        return ErrorCode.EXCEL_READ_ERROR.value

    if not paths:
        msg = "数据源中无有效路径"
        logger.error(msg)
        cb(0, 0, 'error', msg)
        logging.getLogger().removeHandler(_cb_handler)
        return ErrorCode.DATA_EMPTY.value

    total = len(paths)
    cb(0, total, 'extract', f'开始提取特征 (共 {total} 个样本)...')

    # === 步骤2：逐个提取特征 ===
    data_processor = DataProcessor()
    pipeline = FaultDiagnosisPipeline(
        main_sensor_channels=main_channels,
        timestamp_window=timestamp_window,
        default_fault_time=default_fault_time,
        missing_channel_strategy=missing_strategy
    )
    X_list = []
    valid_filenames = []

    for i, p in enumerate(paths):
        if not p.exists():
            cb(i + 1, total, 'extract', f'文件不存在: {p.name}')
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
                    cb(i + 1, total, 'extract', f'跳过 {p.name}: {err_msg}')
                    continue
            else:
                df = result

            if df is None or df.empty:
                continue

            feat_result = pipeline.run(df)
            if feat_result is not None and feat_result.get('status') == 'success':
                X_list.append(feat_result['features'])
                valid_filenames.append(p.name)
        except Exception as e:
            cb(i + 1, total, 'extract', f'处理异常 {p.name}: {e}')

        if (i + 1) % 5 == 0 or (i + 1) == total:
            cb(i + 1, total, 'extract', f'进度: {i + 1}/{total}')

    if not X_list:
        msg = "没有样本被成功处理"
        logger.error(msg)
        cb(0, total, 'error', msg)
        logging.getLogger().removeHandler(_cb_handler)
        return ErrorCode.NO_VALID_SAMPLES.value

    X_array = np.array(X_list)
    cb(total, total, 'save', f'特征提取完成: {len(valid_filenames)} 样本, 维度 {X_array.shape}')

    # === 步骤3：保存 ===
    try:
        samples_tensor = torch.from_numpy(X_array.astype(np.float32))
        torch.save({'samples': samples_tensor}, save_pt_path)

        df_mapping = pd.DataFrame({
            'index': range(len(valid_filenames)),
            'filename': valid_filenames
        })
        df_mapping.to_csv(mapping_path, index=False)
    except Exception as e:
        msg = f"保存文件失败: {e}"
        logger.error(msg)
        cb(0, 0, 'error', msg)
        logging.getLogger().removeHandler(_cb_handler)
        return ErrorCode.PROCESS_ERROR.value

    # === 填充结果 ===
    result_dict['pt_path'] = str(save_pt_path)
    result_dict['mapping_path'] = str(mapping_path)
    result_dict['sample_count'] = len(valid_filenames)
    result_dict['feature_dim'] = X_array.shape[1] if X_array.ndim > 1 else 0

    cb(total, total, 'done', '数据处理完成')
    logger.info("数据处理完成")
    logging.getLogger().removeHandler(_cb_handler)
    return ErrorCode.SUCCESS.value
