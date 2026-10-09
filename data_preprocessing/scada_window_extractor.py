# -*- coding: utf-8 -*-
"""
SCADA时间窗口提取模块

提供时间戳解析、时间窗口提取、边界延伸等功能。

修改说明：
    1. 边界延伸时对 real_time 列做时间外推，避免时间戳重复
    2. 用 argmin() 替代 idxmin()，确保使用位置索引而非 label 索引
    3. timestamp 改为相对点数（故障时刻=0），不再依赖真实秒数
    4. ideal_end_idx 改为 fault_pos + end_offset + 1，确保包含 end_offset 点
    5. 新增 convert_to_relative_point_timestamp，替代原 convert_to_relative_timestamp
"""

from datetime import datetime
from typing import Tuple, Union

import numpy as np
import pandas as pd


# ============================================================================
# 时间戳解析
# ============================================================================

def parse_timestamp(timestamp: Union[str, int, float, datetime]) -> datetime:
    """
    解析多种格式的时间戳

    参数:
        timestamp: 时间戳，支持多种格式
            - 字符串: 'YYYY/MM/DD HH:MM:SS' 或 'YYYY-MM-DD HH:MM:SS'
            - 整数: Unix时间戳(秒)
            - 浮点数: Unix时间戳(秒或毫秒)
            - datetime对象

    返回:
        datetime对象

    异常:
        ValueError: 时间戳格式无法识别
    """
    if isinstance(timestamp, datetime):
        return timestamp

    if isinstance(timestamp, str):
        formats = [
            '%Y/%m/%d %H:%M:%S',
            '%Y-%m-%d %H:%M:%S',
            '%Y/%m/%d %H:%M',
            '%Y-%m-%d %H:%M',
        ]
        for fmt in formats:
            try:
                return datetime.strptime(timestamp, fmt)
            except ValueError:
                continue
        raise ValueError(f"无法识别的时间戳格式: {timestamp}")

    if isinstance(timestamp, (int, float)):
        if timestamp > 1e10:  # 毫秒级时间戳
            return datetime.fromtimestamp(timestamp / 1000)
        else:                 # 秒级时间戳
            return datetime.fromtimestamp(timestamp)

    raise ValueError(f"不支持的时间戳类型: {type(timestamp)}")


# ============================================================================
# 采样间隔推算
# ============================================================================

def _infer_sample_interval(data: pd.DataFrame) -> pd.Timedelta:
    """
    从数据中推算采样间隔

    取前若干行中位数间隔，比只取前两行更稳健。

    参数:
        data: 含 real_time 列的 DataFrame

    返回:
        pd.Timedelta，推算出的采样间隔；若无法推算则返回 1 秒
    """
    if 'real_time' not in data.columns or len(data) < 2:
        return pd.Timedelta(seconds=1)

    # 取前 min(20, len) 行计算间隔中位数
    sample = data['real_time'].iloc[:min(20, len(data))]
    diffs = sample.diff().dropna()

    if len(diffs) == 0:
        return pd.Timedelta(seconds=1)

    median_dt = diffs.median()

    # 间隔异常（<=0 或 >1小时）时兜底
    if median_dt <= pd.Timedelta(0) or median_dt > pd.Timedelta(hours=1):
        return pd.Timedelta(seconds=1)

    return median_dt


# ============================================================================
# 时间窗口提取
# ============================================================================

def extract_time_window(
    data: pd.DataFrame,
    fault_timestamp: Union[str, datetime],
    time_window: Tuple[int, int]
) -> Tuple[pd.DataFrame, bool, str]:
    """
    在故障时刻周围提取数据点，按数据点数量截取，支持边界延伸。

    截取规则：
        time_window = (start_offset, end_offset)
        start_offset 为负数，表示故障前 N 个点
        end_offset   为正数，表示故障后 N 个点
        最终窗口共包含 (end_offset - start_offset + 1) 个点
        故障点对应 timestamp = 0

    边界延伸规则：
        - 前面不够：复制第一个点的值向前延伸，real_time 向前外推
        - 后面不够：复制最后一个点的值向后延伸，real_time 向后外推
        - 前后都不够：同时向前后延伸

    参数:
        data          : SCADA数据DataFrame，含 real_time 列（datetime 类型）
        fault_timestamp: 故障时刻
        time_window   : 数据点窗口 (起始偏移点数, 结束偏移点数)
                        例如 (-600, 600) 表示故障前 600 个点、故障点本身、故障后 600 个点，
                        共 1201 个点，timestamp 范围 [-600, 600]

    返回:
        (windowed_data, is_warning, warning_message)
        - windowed_data : 提取的数据窗口，real_time 列为 datetime，
                          时间连续无重复（延伸部分已外推）
        - is_warning    : 是否触发了边界延伸
        - warning_message: 警告消息
    """
    if time_window is None:
        raise ValueError("time_window 参数必须提供")

    # 解析故障时刻
    fault_time = parse_timestamp(fault_timestamp)

    # 确保 real_time 列是 datetime 类型
    if 'real_time' in data.columns:
        if not pd.api.types.is_datetime64_any_dtype(data['real_time']):
            try:
                data = data.copy()
                data['real_time'] = pd.to_datetime(data['real_time'])
            except Exception as e:
                raise ValueError(f"无法将 real_time 列转换为时间格式: {e}")

    # 确保数据按时间排序，并使用连续的整数位置索引
    if 'real_time' in data.columns:
        data = data.sort_values('real_time').reset_index(drop=True)
    else:
        data = data.reset_index(drop=True)

    total_points = len(data)
    start_offset = time_window[0]   # 负数，故障前 N 个点
    end_offset   = time_window[1]   # 正数，故障后 N 个点

    # ----------------------------------------------------------------
    # 找到故障时刻最近点的位置索引（positional index，不是 label）
    # 使用 argmin() 而非 idxmin()，避免非连续 index 时偏移计算错误
    # ----------------------------------------------------------------
    if 'real_time' in data.columns:
        time_diff = (data['real_time'] - fault_time).abs()
        fault_pos = int(time_diff.values.argmin())
    else:
        fault_pos = total_points // 2

    # ----------------------------------------------------------------
    # 理想截取范围（位置索引）
    # end_offset + 1 确保 iloc 切片包含 fault_pos + end_offset 这一行
    # 例如 fault_pos=100, end_offset=600 → ideal_end_idx=701
    # iloc[100:701] 包含位置 100~700，共 601 行（故障点 + 后 600 行）
    # ----------------------------------------------------------------
    ideal_start_idx = fault_pos + start_offset
    ideal_end_idx   = fault_pos + end_offset + 1

    extend_front = ideal_start_idx < 0
    extend_back  = ideal_end_idx > total_points

    # 实际可截取范围
    actual_start_idx = max(0, ideal_start_idx)
    actual_end_idx   = min(total_points, ideal_end_idx)

    windowed_data = data.iloc[actual_start_idx:actual_end_idx].copy()
    windowed_data = windowed_data.reset_index(drop=True)

    # 推算采样间隔（用原始 data，样本更多更准）
    dt = _infer_sample_interval(data)

    warning_message = ""

    # ----------------------------------------------------------------
    # 向前延伸：复制第一行的值，real_time 向前外推
    # ----------------------------------------------------------------
    if extend_front:
        front_extend_count = -ideal_start_idx  # 需要补的点数

        # 复制第一行的传感器值
        first_row = windowed_data.iloc[0:1].copy()
        front_rows = pd.concat(
            [first_row] * front_extend_count,
            ignore_index=True
        )

        # 时间外推：first_time - N*dt, ..., first_time - 1*dt
        first_time = windowed_data['real_time'].iloc[0]
        for i in range(front_extend_count):
            offset_steps = front_extend_count - i  # 距第一行的步数
            front_rows.at[i, 'real_time'] = first_time - offset_steps * dt

        windowed_data = pd.concat(
            [front_rows, windowed_data],
            ignore_index=True
        )
        warning_message += f"向前延伸{front_extend_count}个点 "

    # ----------------------------------------------------------------
    # 向后延伸：复制最后一行的值，real_time 向后外推
    # ----------------------------------------------------------------
    if extend_back:
        # ideal_end_idx 已经是 fault_pos + end_offset + 1
        # 超出部分就是需要补的点数
        back_extend_count = ideal_end_idx - total_points

        # 复制最后一行的传感器值
        last_row = windowed_data.iloc[-1:].copy()
        back_rows = pd.concat(
            [last_row] * back_extend_count,
            ignore_index=True
        )

        # 时间外推：last_time + 1*dt, ..., last_time + N*dt
        last_time = windowed_data['real_time'].iloc[-1]
        for i in range(back_extend_count):
            back_rows.at[i, 'real_time'] = last_time + (i + 1) * dt

        windowed_data = pd.concat(
            [windowed_data, back_rows],
            ignore_index=True
        )
        warning_message += f"向后延伸{back_extend_count}个点"

    is_warning = extend_front or extend_back

    return windowed_data, is_warning, warning_message.strip()


# ============================================================================
# real_time → 相对点数 timestamp（故障时刻 = 0）
# ============================================================================

def convert_to_relative_point_timestamp(
    data: pd.DataFrame,
    time_window: Tuple[int, int],
    output_col: str = 'timestamp'
) -> pd.DataFrame:
    """
    按相对点数生成 timestamp，故障时刻对应点 = 0。

    不依赖真实时间秒数，因此不受采样间隔影响，
    timestamp 范围严格等于 [start_offset, end_offset]。

    例如：
        time_window = (-600, 600)
        生成 timestamp: -600, -599, ..., -1, 0, 1, ..., 599, 600
        共 1201 个点

    参数:
        data      : extract_time_window 返回的 DataFrame
        time_window: 与 extract_time_window 传入的相同
                    (start_offset, end_offset)
        output_col: 输出列名，默认 'timestamp'

    返回:
        新增 timestamp 列的 DataFrame（float，单位：点数偏移）

    异常:
        ValueError: 数据长度与 time_window 不匹配
    """
    if data is None or len(data) == 0:
        return data

    data = data.copy()

    start_offset, end_offset = time_window
    expected_len = end_offset - start_offset + 1

    if len(data) != expected_len:
        raise ValueError(
            f"数据长度与 time_window 不匹配:\n"
            f"  当前数据长度 = {len(data)}\n"
            f"  time_window  = {time_window}\n"
            f"  期望长度     = {expected_len}\n"
            f"请确认 extract_time_window 是否正常执行（包含边界延伸）。"
        )

    data[output_col] = np.arange(start_offset, end_offset + 1, dtype=float)

    return data


# ============================================================================
# 滑动窗口计算（备用）
# ============================================================================

def calculate_sliding_window(
    data_start: datetime,
    data_end: datetime,
    fault_time: datetime,
    window_duration: int
) -> Tuple[datetime, datetime, bool]:
    """
    计算带滑动行为的实际窗口边界

    参数:
        data_start     : 数据起始时间
        data_end       : 数据结束时间
        fault_time     : 故障时刻
        window_duration: 窗口时长(秒)

    返回:
        (actual_start, actual_end, is_warning)
        - actual_start: 实际窗口起始时间
        - actual_end  : 实际窗口结束时间
        - is_warning  : 是否有警告(数据不足)
    """
    half_window = window_duration // 2
    ideal_start = datetime.fromtimestamp(fault_time.timestamp() - half_window)
    ideal_end   = datetime.fromtimestamp(fault_time.timestamp() + half_window)

    is_warning = False

    if fault_time < data_start or fault_time > data_end:
        return None, None, True

    if ideal_start < data_start:
        actual_start = data_start
        actual_end   = datetime.fromtimestamp(data_start.timestamp() + window_duration)
        if actual_end > data_end:
            actual_end = data_end
            is_warning = True
    elif ideal_end > data_end:
        actual_end   = data_end
        actual_start = datetime.fromtimestamp(data_end.timestamp() - window_duration)
        if actual_start < data_start:
            actual_start = data_start
            is_warning   = True
    else:
        actual_start = ideal_start
        actual_end   = ideal_end

    return actual_start, actual_end, is_warning
