# -*- coding: utf-8 -*-
"""
SCADA数据清洗模块

提供数据清洗、数值转换、缺失值处理等功能。

主要功能:
    - 无效值识别和转换
    - 数值类型转换
    - 重复列合并
    - 时间列清洗
    - 插值和填充
"""

import numpy as np
import pandas as pd
from typing import Union, List


# ============================================================================
# 无效值映射表
# ============================================================================

INVALID_VALUE_MAP = {
    "": np.nan,
    " ": np.nan,
    "nan": np.nan,
    "NaN": np.nan,
    "NAN": np.nan,
    "none": np.nan,
    "None": np.nan,
    "NONE": np.nan,
    "null": np.nan,
    "Null": np.nan,
    "NULL": np.nan,
    "--": np.nan,
    "---": np.nan,
    "----": np.nan,
    "无": np.nan,
    "无数据": np.nan,
    "NA": np.nan,
    "N/A": np.nan,
    "#N/A": np.nan,
    "inf": np.nan,
    "Inf": np.nan,
    "INF": np.nan,
    "-inf": np.nan,
    "-Inf": np.nan,
    "-INF": np.nan,
}


# ============================================================================
# 数据清洗函数
# ============================================================================

def to_numeric_series(series: Union[pd.Series, pd.DataFrame]) -> pd.Series:
    """
    将单列数据安全转换成 float64

    处理步骤:
        1. 去除字符串前后空格
        2. 替换无效值为 NaN
        3. 移除千分位逗号
        4. 转换为数值类型
        5. 替换无穷值为 NaN

    参数:
        series: pandas Series 或 DataFrame（单列）

    返回:
        转换后的 float64 Series
    """
    s = series.copy()

    # 如果是DataFrame，递归处理并合并
    if isinstance(s, pd.DataFrame):
        s = s.apply(lambda x: to_numeric_series(x))
        s = s.bfill(axis=1).iloc[:, 0]
        return s.astype("float64")

    # 字符串清洗
    if s.dtype == "object":
        s = s.astype(str).str.strip()
        s = s.replace(INVALID_VALUE_MAP)
        s = s.str.replace(",", "", regex=False)

    # 数值转换
    s = pd.to_numeric(s, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)

    return s.astype("float64")


def merge_duplicate_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    合并重复列名

    标准化标签后，可能出现多个原始列都映射成同一个标准列名。
    此函数会对重复列取每行第一个非空值。

    参数:
        data: DataFrame

    返回:
        合并后的 DataFrame
    """
    if data is None or len(data) == 0:
        return pd.DataFrame()

    if not data.columns.duplicated().any():
        return data

    dup_cols = data.columns[data.columns.duplicated()].unique().tolist()
    print(f"[INFO] 检测到重复列，开始合并: {dup_cols}")

    result = pd.DataFrame(index=data.index)

    for col in data.columns.unique():
        same_cols = data.loc[:, data.columns == col]

        if same_cols.shape[1] == 1:
            result[col] = same_cols.iloc[:, 0]
            continue

        if col == "real_time":
            # 时间列保留第一个非空
            result[col] = same_cols.bfill(axis=1).iloc[:, 0]
        else:
            # 数值列先全部转数值，再取第一个非空
            numeric_same_cols = same_cols.apply(lambda x: to_numeric_series(x))
            result[col] = numeric_same_cols.bfill(axis=1).iloc[:, 0]

    return result


def force_numeric_columns(
    data: pd.DataFrame,
    time_col: str = "real_time",
    exclude_cols: List[str] = None
) -> pd.DataFrame:
    """
    将除时间列外的所有列强制转换为 float64

    参数:
        data: DataFrame
        time_col: 时间列名
        exclude_cols: 额外需要排除的列名列表

    返回:
        转换后的 DataFrame
    """
    if data is None or len(data) == 0:
        return pd.DataFrame()

    data = data.copy()

    # 如果还有重复列，先合并
    if data.columns.duplicated().any():
        data = merge_duplicate_columns(data)

    # 排除列集合
    exclude_set = {time_col}
    if exclude_cols:
        exclude_set.update(exclude_cols)

    for col in data.columns:
        if col in exclude_set:
            continue

        data[col] = to_numeric_series(data[col])

    return data


def clean_time_column(
    data: pd.DataFrame,
    time_col: str = "real_time",
    drop_invalid: bool = True
) -> pd.DataFrame:
    """
    清洗时间列

    参数:
        data: DataFrame
        time_col: 时间列名
        drop_invalid: 是否删除时间为空的行

    返回:
        清洗后的 DataFrame

    异常:
        ValueError: 数据中缺少时间列
    """
    if time_col not in data.columns:
        raise ValueError(f"数据中缺少 {time_col} 时间列")

    data = data.copy()

    # 转换为datetime
    data[time_col] = pd.to_datetime(data[time_col], errors="coerce")

    # 删除时间为空的行
    if drop_invalid:
        before_len = len(data)
        data = data.dropna(subset=[time_col])
        after_len = len(data)

        if before_len > after_len:
            print(f"[INFO] 删除了 {before_len - after_len} 行无效时间数据")

    return data


def interpolate_and_fill(
    data: pd.DataFrame,
    method: str = "linear",
    fill_value: float = 0.0
) -> pd.DataFrame:
    """
    插值并填充缺失值

    参数:
        data: DataFrame
        method: 插值方法 ('linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic')
        fill_value: 最终填充值

    返回:
        填充后的 DataFrame
    """
    data = data.copy()

    # 替换无穷值
    data = data.replace([np.inf, -np.inf], np.nan)

    # 插值 + 前后填充 + 最终补填充值
    data = data.interpolate(method=method, limit_direction="both")
    data = data.ffill().bfill().fillna(fill_value)

    # 最终强制 float64
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = data[numeric_cols].astype("float64")

    return data


def remove_outliers(
    data: pd.DataFrame,
    columns: List[str] = None,
    method: str = "iqr",
    threshold: float = 3.0
) -> pd.DataFrame:
    """
    移除异常值

    参数:
        data: DataFrame
        columns: 需要处理的列名列表，None则处理所有数值列
        method: 异常值检测方法 ('iqr', 'zscore')
        threshold: 阈值（IQR倍数或Z-score标准差倍数）

    返回:
        处理后的 DataFrame
    """
    data = data.copy()

    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()

    for col in columns:
        if col not in data.columns:
            continue

        if method == "iqr":
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)

        elif method == "zscore":
            mean = data[col].mean()
            std = data[col].std()
            lower_bound = mean - threshold * std
            upper_bound = mean + threshold * std
            data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)

    return data

