import sys
import os
import warnings
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from data_preprocessing.pipeline import FaultDiagnosisPipeline
from utils.logger import preprocess_logger
from error_codes import ErrorCode

warnings.filterwarnings('ignore')


def process_tagged_data(dict_para, result_dict) -> ErrorCode:
    """
    接口名称: process_tagged_data

    功能说明:
    数据预处理主流程，从Excel索引读取原始数据路径，批量提取特征，
    进行数据清洗和标签编码，按8:1:1比例生成训练集、验证集和测试集，
    并保存为.pt格式文件。

    Args:
        dict_para: 参数字典，包含所有输入配置（详见API文档）
        result_dict: 输出结果字典（由函数填充）

    Returns:
        ErrorCode: 错误码
    """
    log_dir = os.path.join(project_root, 'logs', datetime.now().strftime('%Y%m%d_%H%M%S'), 'dataprocessing')
    logger = preprocess_logger(log_dir, log_name="data_processing")

    # === 参数解析 ===
    # 数据源
    excel_path = Path(dict_para.get('source_excel', ''))
    col_tracelog_path = dict_para.get('col_tracelog_path', '')
    col_label = dict_para.get('col_label', '')
    output_dir = Path(dict_para.get('processed_dir', './data/output'))

    # 数据划分
    random_seed = dict_para.get('random_seed', 42)
    test_size = dict_para.get('test_size', 0.1)
    val_ratio = dict_para.get('val_ratio', 1/9)
    use_stratify = dict_para.get('stratify', True)

    # Pipeline 参数
    missing_strategy = dict_para.get('missing_channel_strategy', 'zero')
    add_indicator = dict_para.get('add_missing_indicator', False)
    main_channels = dict_para.get('main_sensor_channels', None)
    knowledge_file = dict_para.get('fault_knowledge_file', None)
    timestamp_window = dict_para.get('timestamp_window', None)
    default_fault_time = dict_para.get('default_fault_time', None)
    column_mapping_file = dict_para.get('column_mapping_file', None)

    output_dir.mkdir(parents=True, exist_ok=True)

    # === 参数校验 ===
    if not excel_path or not Path(excel_path).exists():
        logger.error(f"找不到数据清单: {excel_path}")
        return ErrorCode.FILE_NOT_FOUND

    if not col_tracelog_path or not col_label:
        logger.error("col_tracelog_path 和 col_label 不能为空")
        return ErrorCode.PROCESS_ERROR

    logger.info("开始生成 full/train/val/test 数据集")

    try:
        # === 读取 Excel 索引 ===
        df = pd.read_excel(excel_path)
        df = df.dropna(subset=[col_tracelog_path, col_label])
        raw_paths = df[col_tracelog_path].tolist()
        raw_labels = df[col_label].tolist()

        if len(raw_paths) == 0:
            logger.error("Excel中无有效数据")
            return ErrorCode.DATA_EMPTY

        # === 索引划分（分层抽样） ===
        total_samples = len(raw_labels)
        indices = np.arange(total_samples)

        stratify_labels = raw_labels if use_stratify else None
        try:
            train_val_idx, test_idx = train_test_split(
                indices, test_size=test_size, random_state=random_seed, shuffle=True,
                stratify=stratify_labels
            )
            train_val_labels = [raw_labels[i] for i in train_val_idx] if use_stratify else None
            train_idx, val_idx = train_test_split(
                train_val_idx, test_size=val_ratio, random_state=random_seed, shuffle=True,
                stratify=train_val_labels
            )
        except ValueError as e:
            logger.warning(f"分层抽样失败({e})，回退到随机划分")
            train_val_idx, test_idx = train_test_split(
                indices, test_size=test_size, random_state=random_seed, shuffle=True
            )
            train_idx, val_idx = train_test_split(
                train_val_idx, test_size=val_ratio, random_state=random_seed, shuffle=True
            )

        logger.info(f"数据集划分: 训练集={len(train_idx)}, 验证集={len(val_idx)}, 测试集={len(test_idx)}")

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

        logger.info(
            f"完整数据集已保存: {full_result['samples'].shape[0]} 样本 "
            f"-> {full_dataset_path}"
        )

        # === 分批处理并保存 ===
        for split_name, split_idx, split_path in [
            ("训练集", train_idx, train_path),
            ("验证集", val_idx, val_path),
            ("测试集", test_idx, test_path)
        ]:
            logger.info(f"正在处理{split_name}...")
            split_paths = [raw_paths[i] for i in split_idx]
            split_labels = [raw_labels[i] for i in split_idx]

            result = pipeline.process_batch_to_pt(
                data_list=split_paths,
                label_list=split_labels,
                output_path=None
            )

            torch.save({
                'samples': result['samples'],
                'labels': result['labels']
            }, split_path)

            logger.info(f"{split_name}已保存: {len(split_idx)} 样本 ({len(split_idx)/total_samples*100:.1f}%)")

        # === 填充结果字典 ===
        result_dict['full_dataset_path'] = str(full_dataset_path)
        result_dict['full_count'] = int(full_result['samples'].shape[0])
        result_dict['train_path'] = str(train_path)
        result_dict['val_path'] = str(val_path)
        result_dict['test_path'] = str(test_path)
        result_dict['train_count'] = len(train_idx)
        result_dict['val_count'] = len(val_idx)
        result_dict['test_count'] = len(test_idx)

        logger.info("所有数据集生成完成")
        return ErrorCode.SUCCESS

    except Exception as e:
        logger.critical(f"数据处理失败: {e}", exc_info=True)
        return ErrorCode.PROCESS_ERROR


if __name__ == "__main__":
    from configs.config_loader import (
        DATA_SOURCE_EXCEL, PROCESSED_DATA_DIR,
        COL_TRACELOG_PATH, COL_LABEL, RANDOM_SEED,
        MAIN_SENSOR_CHANNELS, FAULT_KNOWLEDGE_FILE,
        TIMESTAMP_WINDOW, DEFAULT_FAULT_TIME, COLUMN_MAPPING
    )

    dict_para = {
        # 数据源
        'source_excel': DATA_SOURCE_EXCEL,
        'col_tracelog_path': COL_TRACELOG_PATH,
        'col_label': COL_LABEL,
        'processed_dir': PROCESSED_DATA_DIR,
        # 数据划分
        'random_seed': RANDOM_SEED,
        'test_size': 0.1,
        'val_ratio': 1/9,
        # 特征提取
        'main_sensor_channels': MAIN_SENSOR_CHANNELS,
        'timestamp_window': TIMESTAMP_WINDOW,
        'default_fault_time': DEFAULT_FAULT_TIME,
        'missing_channel_strategy': 'zero',
        'add_missing_indicator': False,
        # 标签映射
        'fault_knowledge_file': FAULT_KNOWLEDGE_FILE,
    }

    result_dict = {}
    error_code = process_tagged_data(dict_para, result_dict)
    print(f"返回码: {error_code}")
    print(f"结果: {result_dict}")
