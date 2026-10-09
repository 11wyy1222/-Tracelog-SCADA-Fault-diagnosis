# -*- coding: utf-8 -*-
"""
SCADA标签处理模块

提供标签映射、标准化、别名处理等功能。

主要功能:
    - 标签有效性验证
    - 标签映射构建
    - 标签标准化
    - 别名查询
    - 数据列名标准化
"""

import pandas as pd
from typing import List, Dict, Set


# ============================================================================
# 标签验证和清洗
# ============================================================================

def valid_tag_value(x) -> bool:
    """
    判断标签配置值是否有效

    无效值包括: NaN, 空字符串, '1', 'nan', 'None', 'null' 等占位符

    参数:
        x: 标签值

    返回:
        bool: 是否有效
    """
    if pd.isna(x):
        return False

    s = str(x).strip()

    # 常见占位符
    invalid_values = {
        "", "1", "1.0",
        "nan", "NaN", "NAN",
        "None", "NONE", "none",
        "null", "NULL", "Null"
    }

    return s not in invalid_values


def clean_tag_value(x) -> str:
    """
    清洗标签名（去除前后空格）

    参数:
        x: 标签值

    返回:
        清洗后的标签字符串
    """
    return str(x).strip()


def set_by_order(input_list: List) -> List:
    """
    按顺序去重（保持原始顺序）

    参数:
        input_list: 输入列表

    返回:
        去重后的列表
    """
    result = []
    for item in input_list:
        if item not in result:
            result.append(item)
    return result


# ============================================================================
# 标签映射构建
# ============================================================================

def build_tag_mapping(tag_info: pd.DataFrame) -> Dict[str, str]:
    """
    构建 别名 -> 标准标签 映射

    tag_info 表结构要求:
        tags | Chinese_tags | config1 | config2 | ...

    第一列(tags)是标准标签名，第三列及之后是各种别名配置

    参数:
        tag_info: 标签配置DataFrame

    返回:
        {别名: 标准标签} 的映射字典

    异常:
        ValueError: 标签配置表缺少 tags 列
    """
    if "tags" not in tag_info.columns:
        raise ValueError("标签配置表缺少 tags 列")

    tag_mapping = {}

    for _, row in tag_info.iterrows():
        standard_tag = row["tags"]

        if not valid_tag_value(standard_tag):
            continue

        standard_tag = clean_tag_value(standard_tag)

        # 标准标签自身也映射到自身
        tag_mapping[standard_tag] = standard_tag

        # config1/config2/config3... (从第3列开始)
        for item in row.iloc[2:]:
            if not valid_tag_value(item):
                continue

            alias = clean_tag_value(item)
            tag_mapping[alias] = standard_tag

    return tag_mapping


def get_standard_tags(tag_info: pd.DataFrame) -> List[str]:
    """
    获取所有标准标签列表

    参数:
        tag_info: 标签配置DataFrame

    返回:
        标准标签列表（按配置文件顺序）

    异常:
        ValueError: 标签配置表缺少 tags 列
    """
    if "tags" not in tag_info.columns:
        raise ValueError("标签配置表缺少 tags 列")

    tags = [
        clean_tag_value(x)
        for x in tag_info["tags"]
        if valid_tag_value(x)
    ]

    return set_by_order(tags)


# ============================================================================
# 标签转换和查询
# ============================================================================

def normalize_tag_points(
    tag_points: List[str],
    tag_info: pd.DataFrame
) -> List[str]:
    """
    将输入标签统一转换为标准标签

    例如:
        TimeStamp -> real_time
        sequence_time -> real_time
        grPitchBackupVoltage1 -> grPitchBackupVoltage1

    参数:
        tag_points: 输入标签列表
        tag_info: 标签配置DataFrame

    返回:
        标准标签列表
    """
    if not tag_points:
        return []

    tag_mapping = build_tag_mapping(tag_info)

    result = []

    for tag in tag_points:
        if not valid_tag_value(tag):
            continue

        tag = clean_tag_value(tag)
        result.append(tag_mapping.get(tag, tag))

    return set_by_order(result)


def get_aliases_for_standard_tags(
    standard_tags: List[str],
    tag_info: pd.DataFrame
) -> List[str]:
    """
    根据标准标签获取所有可匹配别名，包括标准标签自身

    例如输入:
        ["real_time", "grPitchBackupVoltage1"]

    返回:
        [
            "real_time", "sequence_time", "TimeStamp",
            "grPitchBackupVoltage1", "xxx_alias1", ...
        ]

    参数:
        standard_tags: 标准标签列表
        tag_info: 标签配置DataFrame

    返回:
        包含所有别名的列表
    """
    if not standard_tags:
        return []

    standard_set = set(clean_tag_value(x) for x in standard_tags)

    result = []

    for _, row in tag_info.iterrows():
        standard_tag = row["tags"]

        if not valid_tag_value(standard_tag):
            continue

        standard_tag = clean_tag_value(standard_tag)

        if standard_tag not in standard_set:
            continue

        # 标准标签自身
        result.append(standard_tag)

        # 后面的别名列
        for item in row.iloc[2:]:
            if not valid_tag_value(item):
                continue

            result.append(clean_tag_value(item))

    return set_by_order(result)


# ============================================================================
# 数据标准化
# ============================================================================

def standardize_scada_data(
    raw_data: pd.DataFrame,
    tag_info: pd.DataFrame,
    keep_only_matched: bool = True,
    keep_order: bool = True
) -> pd.DataFrame:
    """
    标准化 SCADA 数据列名

    功能:
      1. 根据 tag_info 将别名列改为标准标签列
      2. real_time 也作为普通标准标签处理
      3. 可选择只保留匹配到的列
      4. 重复列名合并，取每行第一个非空值
      5. 按 tag_info 中 tags 顺序排序

    参数:
        raw_data: 原始数据DataFrame
        tag_info: 标签配置DataFrame
        keep_only_matched: 是否只保留匹配到的列
        keep_order: 是否按tag_info顺序排列列

    返回:
        标准化后的DataFrame
    """
    if raw_data is None or raw_data.empty:
        return pd.DataFrame()

    data = raw_data.copy()

    # 清洗列名
    data.columns = [clean_tag_value(c) for c in data.columns]

    tag_mapping = build_tag_mapping(tag_info)

    matched_columns = []
    rename_dict = {}

    for col in data.columns:
        if col in tag_mapping:
            matched_columns.append(col)
            rename_dict[col] = tag_mapping[col]

    if keep_only_matched:
        data = data[matched_columns]

    data = data.rename(columns=rename_dict)

    # 合并重复列名
    if data.columns.duplicated().any():
        merged = pd.DataFrame(index=data.index)

        for col in set_by_order(list(data.columns)):
            same_cols = data.loc[:, data.columns == col]

            if same_cols.shape[1] == 1:
                merged[col] = same_cols.iloc[:, 0]
            else:
                # 取第一个非空值
                merged[col] = same_cols.bfill(axis=1).iloc[:, 0]

        data = merged

    # 按标准顺序排列
    if keep_order:
        standard_order = get_standard_tags(tag_info)

        ordered_cols = [
            col for col in standard_order
            if col in data.columns
        ]

        extra_cols = [
            col for col in data.columns
            if col not in ordered_cols
        ]

        data = data[ordered_cols + extra_cols]

    return data

