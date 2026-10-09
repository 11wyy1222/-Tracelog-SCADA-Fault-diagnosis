# -*- coding: utf-8 -*-
"""
SCADA数据读取模块（统一接口）

提供统一的SCADA数据读取接口，支持多种压缩格式。
本模块整合了底层文件读取功能，提供更高层的封装。

支持格式: csv, gz, zip, rar, 7z

主要类:
    - ScadaDataProcessor: SCADA数据处理器，提供标签映射和数据标准化功能
"""

import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from .scada_file_readers import read_data
from .scada_tag_processor import (
    build_tag_mapping,
    get_standard_tags,
    standardize_scada_data
)
from .scada_data_cleaner import force_numeric_columns, clean_time_column

warnings.filterwarnings("ignore")


# ============================================================================
# SCADA数据处理器类
# ============================================================================

class ScadaDataProcessor:
    """
    SCADA数据处理器

    功能:
        1. 读取各种格式的SCADA文件
        2. 标签映射和标准化
        3. 数据清洗和格式转换
    """

    def __init__(self, tag_info_path: str = None):
        """
        初始化SCADA数据处理器

        参数:
            tag_info_path: 标签映射配置文件路径（Excel或CSV）
                          None时使用默认路径
        """
        if tag_info_path is None:
            # 默认路径：在项目根目录下查找
            project_root = Path(__file__).resolve().parents[1]
            default_paths = [
                project_root / "configs" / "scada_tag_info.xlsx",
                project_root / "configs" / "scada_tag_info.csv",
                project_root / "data_preprocessing" / "tracelog标签点配置表.csv"
            ]

            for path in default_paths:
                if path.exists():
                    tag_info_path = str(path)
                    break

            if tag_info_path is None:
                raise FileNotFoundError(
                    "未找到标签映射配置文件，请指定 tag_info_path 参数"
                )

        # 读取标签配置
        if tag_info_path.endswith('.csv'):
            self.tag_info = pd.read_csv(tag_info_path, encoding='utf-8')
        else:
            self.tag_info = pd.read_excel(tag_info_path)

    def get_useful_tags(self):
        """
        获取所有有效的模型标签和时间标签

        返回:
            (model_tags, time_tags): 模型标签列表和时间标签列表
        """
        # 提取所有标签（从第3列开始）
        tag_data = self.tag_info.iloc[:, 2:].values.tolist()
        tag_model = []

        for row in tag_data:
            tags = np.unique(row).tolist()
            tag_model.extend(tags)

        # 去重并移除占位符
        tag_model_all = [t for t in np.unique(tag_model).tolist() if t not in ['1', 1, '1.0', np.nan]]

        # 提取时间标签
        tag_time_rows = self.tag_info[self.tag_info['Chinese_tags'] == '时间标签']
        if len(tag_time_rows) > 0:
            tag_time_all = np.unique(tag_time_rows.iloc[:, 2:].values.tolist()).tolist()
            tag_time_all = [t for t in tag_time_all if t not in ['1', 1, '1.0', np.nan]]
        else:
            # 默认时间标签
            tag_time_all = ['real_time', 'TimeStamp', 'sequence_time', '时间']

        return tag_model_all, tag_time_all

    def standard_tags(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        标准化数据列名（将别名映射为标准标签）

        参数:
            data: 原始数据DataFrame

        返回:
            标准化后的DataFrame
        """
        return standardize_scada_data(
            data,
            self.tag_info,
            keep_only_matched=True,
            keep_order=True
        )

    def mapping_tags(self) -> dict:
        """
        获取标签映射字典

        返回:
            {别名: 标准标签} 的映射字典
        """
        return build_tag_mapping(self.tag_info)

    def get_data_by_scadapath(
        self,
        scada_path: str,
        need_tags: Optional[List[str]] = None,
        log_enable: bool = False
    ) -> pd.DataFrame:
        """
        读取并处理SCADA数据

        参数:
            scada_path: SCADA文件路径
            need_tags: 需要的标签列表，None则返回所有标签
            log_enable: 是否启用日志

        返回:
            DataFrame，包含 real_time 列和各标签列

        异常:
            ValueError: 数据中缺少 real_time 列
            FileNotFoundError: 文件不存在
        """
        if not os.path.exists(scada_path):
            raise FileNotFoundError(f"SCADA文件不存在: {scada_path}")

        # 获取所有标签
        tag_model_all, tag_time_all = self.get_useful_tags()

        # 读取原始数据
        alltag_data = read_data(
            scada_path,
            tag_model_all,
            tag_time_all,
            log_enable=log_enable
        )

        if alltag_data.empty:
            raise ValueError(f"无法读取SCADA数据: {scada_path}")

        # 标准化标签
        alltag_data = self.standard_tags(alltag_data)

        # 检查 real_time 列
        if 'real_time' not in alltag_data.columns:
            raise ValueError(
                f"数据中缺少 'real_time' 列，请检查SCADA数据路径: {scada_path}"
            )

        # 筛选需要的标签
        if need_tags is not None:
            tag_mapping = self.mapping_tags()
            need_tags_mapped = [tag_mapping.get(tag, tag) for tag in need_tags]

            # real_time 单独保留
            cols_needed = list(set(need_tags_mapped) - {'real_time'})
            available = alltag_data.columns.intersection(cols_needed).tolist()

            if not available:
                raise ValueError(
                    f"未找到任何需要的标签。\n"
                    f"需要的标签: {need_tags}\n"
                    f"映射后标签: {need_tags_mapped}\n"
                    f"数据中可用列: {alltag_data.columns.tolist()}"
                )

            alltag_data = alltag_data[['real_time'] + available]

        return alltag_data


# ============================================================================
# 向后兼容的���数接口
# ============================================================================

def read_scada_data(
    scada_path: str,
    tag_info_path: str = None,
    need_tags: Optional[List[str]] = None,
    log_enable: bool = False
) -> pd.DataFrame:
    """
    读取SCADA数据的便捷函数

    参数:
        scada_path: SCADA文件路径
        tag_info_path: 标签映射配置文件路径
        need_tags: 需要的标签列表
        log_enable: 是否启用日志

    返回:
        DataFrame
    """
    processor = ScadaDataProcessor(tag_info_path)
    return processor.get_data_by_scadapath(scada_path, need_tags, log_enable)


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    from datetime import datetime

    t1 = datetime.now()

    # 测试路径（根据实际情况修改）
    test_paths = [
        r'\\192.168.100.21\风场数据\海上工程\S1-20200066 上海海湾新能奉贤海上风电项目（Ⅱ标段）\SCADA文件\csv\60025008_20230320.csv.7z',
        r'\\192.168.100.21\风场数据\海上工程\S1-20190061 中广核广东汕尾甲子二海上风电项目\SCADA文件\csv\60038001_20240502.csv.gz',
    ]

    need_tags = ['OperationMode', 'grPitchAngle1B']

    # 使用类接口
    processor = ScadaDataProcessor()

    for scada_path in test_paths:
        if os.path.exists(scada_path):
            print(f"\n测试文件: {scada_path}")
            try:
                data = processor.get_data_by_scadapath(scada_path, need_tags)
                print(f"数据形状: {data.shape}")
                print(f"列名: {data.columns.tolist()}")
                print(data.head())
            except Exception as e:
                print(f"读取失败: {e}")
            break

    t2 = datetime.now()
    print(f"\n耗时: {t2 - t1}")
