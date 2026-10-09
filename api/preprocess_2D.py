"""
2D数据预处理模块（训练数据）

对外暴露核心函数:
    process_tagged_data(dict_para, result_dict, progress_callback=None) → int

调用示例:
    result_dict = {}
    error_code = process_tagged_data(
        dict_para={
            'source_excel': r'\\server\data\record.xlsx',
            'col_tracelog_path': 'Tracelog路径',
            'col_label': '标签',
            'processed_dir': './data/output',
            'main_sensor_channels': ['rotor_speed', 'generator_speed'],
            'fault_knowledge_file': './configs/class_mapping.xlsx',
            'timestamp_window': [-2000, 2000],
        },
        result_dict=result_dict,
    )

参数说明:
    dict_para       : dict — 所有输入参数（平铺）
    result_dict     : dict — 输出容器
    progress_callback : callable, optional — 进度回调函数
        签名: progress_callback(current, total, phase, message)
        - current: 当前已处理数量
        - total: 总数量
        - phase: 阶段名 ('split'/'train'/'val'/'test'/'done'/'error')
        - message: 日志消息

错误编码 (1100-1199):
    1100 SUCCESS           成功
    1101 FILE_NOT_FOUND    文件未找到
    1102 DATA_EMPTY        数据为空
    1004 PROCESS_ERROR     处理过程出错
"""

import sys
import os
import warnings
import logging
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data_preprocessing.pipeline import FaultDiagnosisPipeline
from utils.logger import preprocess_logger
from error_codes import ErrorCode

warnings.filterwarnings('ignore')


def _noop_callback(current, total, phase, message):
    """默认空回调"""
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


def process_tagged_data(dict_para, result_dict, progress_callback=None) -> int:
    """
    训练数据预处理入口。

    Parameters
    ----------
    dict_para : dict
        所有输入参数（平铺），详见API文档。
    result_dict : dict
        输出容器。成功后写入 train_path/val_path/test_path 等。
    progress_callback : callable, optional
        进度回调函数，签名: (current, total, phase, message)

    Returns
    -------
    int : 0 (SUCCESS.value=1100) 表示成功；非 0 为错误代码。
    """
    cb = progress_callback or _noop_callback

    log_dir = os.path.join(_PROJECT_ROOT, 'logs', datetime.now().strftime('%Y%m%d_%H%M%S'), 'dataprocessing')
    logger = preprocess_logger(log_dir, log_name="data_processing")

    # 将日志也桥接到回调，让前端能实时看到 WARNING/ERROR 等
    _cb_handler = _CallbackLogHandler(cb)
    _cb_handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    logging.getLogger().addHandler(_cb_handler)

    # === 参数解析 ===
    excel_path = Path(dict_para.get('source_excel', ''))
    col_tracelog_path = dict_para.get('col_tracelog_path', '')
    col_label = dict_para.get('col_label', '')
    output_dir = Path(dict_para.get('processed_dir', './data/output'))
    random_seed = dict_para.get('random_seed', 42)
    test_size = dict_para.get('test_size', 0.1)
    val_ratio = dict_para.get('val_ratio', 1/9)
    use_stratify = dict_para.get('stratify', True)

    missing_strategy = dict_para.get('missing_channel_strategy', 'zero')
    add_indicator = dict_para.get('add_missing_indicator', False)
    main_channels = dict_para.get('main_sensor_channels', None)
    knowledge_file = dict_para.get('fault_knowledge_file', None)
    timestamp_window = dict_para.get('timestamp_window', None)
    default_fault_time = dict_para.get('default_fault_time', None)

    output_dir.mkdir(parents=True, exist_ok=True)

    # === 参数校验 ===
    if not excel_path or not Path(excel_path).exists():
        msg = f"找不到数据清单: {excel_path}"
        logger.error(msg)
        cb(0, 0, 'error', msg)
        return ErrorCode.FILE_NOT_FOUND.value

    if not col_tracelog_path or not col_label:
        msg = "col_tracelog_path 和 col_label 不能为空"
        logger.error(msg)
        cb(0, 0, 'error', msg)
        return ErrorCode.PROCESS_ERROR.value

    try:
        # === 读取 Excel 索引 ===
        cb(0, 0, 'split', '正在读取Excel索引...')
        df = pd.read_excel(excel_path)
        df = df.dropna(subset=[col_tracelog_path, col_label])
        raw_paths = df[col_tracelog_path].tolist()
        raw_labels = df[col_label].tolist()

        if len(raw_paths) == 0:
            msg = "Excel中无有效数据"
            logger.error(msg)
            cb(0, 0, 'error', msg)
            return ErrorCode.DATA_EMPTY.value

        total_samples = len(raw_labels)
        cb(0, total_samples, 'split', f'读取到 {total_samples} 个样本，开始划分数据集...')

        # === 索引划分 ===
        indices = np.arange(total_samples)
        stratify_labels = raw_labels if use_stratify else None
        try:
            train_val_idx, test_idx = train_test_split(
                indices, test_size=test_size, random_state=random_seed,
                shuffle=True, stratify=stratify_labels
            )
            train_val_labels = [raw_labels[i] for i in train_val_idx] if use_stratify else None
            train_idx, val_idx = train_test_split(
                train_val_idx, test_size=val_ratio, random_state=random_seed,
                shuffle=True, stratify=train_val_labels
            )
        except ValueError:
            train_val_idx, test_idx = train_test_split(
                indices, test_size=test_size, random_state=random_seed, shuffle=True
            )
            train_idx, val_idx = train_test_split(
                train_val_idx, test_size=val_ratio, random_state=random_seed, shuffle=True
            )

        split_msg = f"数据集划分: 训练集={len(train_idx)}, 验证集={len(val_idx)}, 测试集={len(test_idx)}"
        logger.info(split_msg)
        cb(0, total_samples, 'split', split_msg)

        # === 构建 Pipeline ===
        pipeline = FaultDiagnosisPipeline(
            missing_channel_strategy=missing_strategy,
            add_missing_indicator=add_indicator,
            main_sensor_channels=main_channels,
            fault_knowledge_file=knowledge_file,
            timestamp_window=timestamp_window,
            default_fault_time=default_fault_time
        )

        full_dataset_path = output_dir / "full_dataset.pt"
        train_path = output_dir / "train.pt"
        val_path = output_dir / "val.pt"
        test_path = output_dir / "test.pt"

        cb(0, total_samples, 'split', '正在处理完整数据集...')
        logger.info("正在处理完整数据集...")
        full_result = pipeline.process_batch_to_pt(
            data_list=raw_paths,
            label_list=raw_labels,
            output_path=None
        )

        torch.save({
            'samples': full_result['samples'],
            'labels': full_result['labels']
        }, full_dataset_path)

        full_msg = (
            f"完整数据集已保存: {full_result['samples'].shape[0]} 样本 "
            f"-> {full_dataset_path}"
        )
        logger.info(full_msg)
        cb(0, total_samples, 'split', full_msg)

        # === 分批处理 ===
        # 计算总处理量用于进度条
        total_to_process = len(train_idx) + len(val_idx) + len(test_idx)
        processed_count = 0

        for split_name, phase, split_idx, split_path in [
            ("训练集", "train", train_idx, train_path),
            ("验证集", "val", val_idx, val_path),
            ("测试集", "test", test_idx, test_path),
        ]:
            cb(processed_count, total_to_process, phase, f'正在处理{split_name}...')
            logger.info(f"正在处理{split_name}...")

            split_paths = [raw_paths[i] for i in split_idx]
            split_labels = [raw_labels[i] for i in split_idx]

            # 桥接pipeline内部的逐样本回调到API层回调
            def _pipeline_cb(cur, tot, msg, _phase=phase, _offset=processed_count):
                cb(_offset + cur, total_to_process, _phase, msg)

            result = pipeline.process_batch_to_pt(
                data_list=split_paths,
                label_list=split_labels,
                output_path=None,
                progress_callback=_pipeline_cb
            )

            torch.save({
                'samples': result['samples'],
                'labels': result['labels']
            }, split_path)

            processed_count += len(split_idx)
            done_msg = f"{split_name}已保存: {len(split_idx)} 样本"
            logger.info(done_msg)
            cb(processed_count, total_to_process, phase, done_msg)

        # === 填充结果 ===
        result_dict['full_dataset_path'] = str(full_dataset_path)
        result_dict['full_count'] = int(full_result['samples'].shape[0])
        result_dict['train_path'] = str(train_path)
        result_dict['val_path'] = str(val_path)
        result_dict['test_path'] = str(test_path)
        result_dict['train_count'] = len(train_idx)
        result_dict['val_count'] = len(val_idx)
        result_dict['test_count'] = len(test_idx)

        cb(total_to_process, total_to_process, 'done', '所有数据集生成完成')
        logger.info("所有数据集生成完成")
        logging.getLogger().removeHandler(_cb_handler)
        return ErrorCode.SUCCESS.value

    except Exception as e:
        msg = f"数据处理失败: {e}"
        logger.critical(msg, exc_info=True)
        cb(0, 0, 'error', msg)
        logging.getLogger().removeHandler(_cb_handler)
        return ErrorCode.PROCESS_ERROR.value
