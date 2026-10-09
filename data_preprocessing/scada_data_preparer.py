# -*- coding: utf-8 -*-
"""
SCADA数据预处理模块

负责将原始SCADA文件处理为适合特征提取的中间格式：
  - 按标签点筛选列
  - 截取时间窗口
  - 将 real_time 转换为相对时间数值列（故障点=0，单位：数据点偏移量）
  - 保留原始 real_time 列供排查使用
  - 落盘到 processed_dir

输出文件格式（CSV）：
  real_time       | timestamp  | tag1   | tag2   | ...
  2024-01-10 07:  | -600       | 123.4  | 456.7  | ...
  ...             | 0          | ...    | ...    | ...
  ...             | 600        | ...    | ...    | ...

配置文件字段映射：
  tag_mapping_path        ← tags.mapping_file
  processed_dir           ← data.processed_dir
  time_window             ← time_window.start_offset / end_offset
  default_tags            ← tags.default_tags
  missing_channel_strategy← data_processing.missing_channel_strategy
  min_rows_after_window   ← data_processing.min_rows_after_window
  log_enable              ← logging.enable
"""

import os
import hashlib
import traceback
from datetime import datetime
from pathlib import Path
from typing import Union, Tuple, List, Optional
import numpy as np
import pandas as pd
import yaml

from .scada_errors import SCADA_ERROR_CODES, SCADA_Error
from .scada_window_extractor import parse_timestamp, extract_time_window
from .scada_read import ScadaDataProcessor


# ============================================================================
# 模块级工具函数
# ============================================================================

def _expand_path(path_str: str) -> str:
    """替换 __PROJECT_ROOT__ 占位符为实际项目根路径"""
    if '__PROJECT_ROOT__' in path_str:
        project_root = Path(__file__).resolve().parents[2]
        return path_str.replace('__PROJECT_ROOT__', str(project_root))
    return path_str


def _make_output_filename(
    scada_path: str,
    fault_timestamp: Union[str, datetime]
) -> str:
    """
    根据原始文件路径和故障时刻生成唯一的输出文件名

    格式: {原始文件名（去扩展名）}_{故障时刻}_{hash}.csv
    例如: 60009085_20260312_20260312T074913_a3f2c1.csv

    参数:
        scada_path     : 原始SCADA文件路径
        fault_timestamp: 故障时刻

    返回:
        输出文件名（不含目录）
    """
    # 去掉所有扩展名（如 .csv.gz → 去两层）
    basename = os.path.basename(scada_path)
    for _ in range(3):
        root, ext = os.path.splitext(basename)
        if not ext:
            break
        basename = root

    # 故障时刻格式化
    if isinstance(fault_timestamp, datetime):
        ts_str = fault_timestamp.strftime('%Y%m%dT%H%M%S')
    else:
        try:
            dt = parse_timestamp(fault_timestamp)
            ts_str = dt.strftime('%Y%m%dT%H%M%S')
        except Exception:
            ts_str = (
                str(fault_timestamp)
                .replace(' ', 'T')
                .replace('/', '')
                .replace(':', '')
            )

    # 短 hash，防止同文件名不同路径冲突
    hash_src   = f"{scada_path}_{fault_timestamp}"
    short_hash = hashlib.md5(hash_src.encode()).hexdigest()[:6]

    return f"{basename}_{ts_str}_{short_hash}.csv"


def _clean_numeric_column(series: pd.Series) -> np.ndarray:
    """
    将一列数据清洗为 float64 数组

    处理常见占位符：'----', 'N/A', 'null' 等

    参数:
        series: 原始列数据

    返回:
        float64 numpy 数组
    """
    _PLACEHOLDER_MAP = {
        '': '0', 'nan': '0', 'NaN': '0', 'NAN': '0',
        'NULL': '0', 'null': '0', 'None': '0', 'none': '0',
        'N/A': '0', 'n/a': '0', 'NA': '0',
        '----': '0', '---': '0', '--': '0', '-': '0',
        'inf': '0', 'Inf': '0', 'INF': '0',
        '-inf': '0', '-Inf': '0',
    }
    cleaned = (
        series.astype(str)
              .str.strip()
              .replace(_PLACEHOLDER_MAP)
    )
    return pd.to_numeric(cleaned, errors='coerce').fillna(0.0).values.astype(np.float64)


# ============================================================================
# SCADA_Data_Preparer 类
# ============================================================================

class SCADA_Data_Preparer:
    """
    SCADA数据预处理器

    将原始SCADA文件处理为适合特征提取的中间CSV文件：
      - 筛选目标标签点列
      - 截取时间窗口
      - real_time 保留原始值（供排查）
      - 新增 timestamp 列（相对故障时刻的点数偏移，故障点=0）
      - 各标签列强制转为 float64，清洗占位符
      - 标签缺失时按 missing_channel_strategy 处理

    支持两种初始化方式（可混用，关键字参数优先级最高）：
      1. config_path: 传入 YAML 配置文件路径，自动读取所有参数
      2. config    : 传入已解析的配置字典
      关键字参数（tag_mapping_path 等）显式传入时覆盖配置文件中的对应值
    """

    # missing_channel_strategy 合法值
    _VALID_STRATEGIES = ('zero', 'nan', 'skip')

    def __init__(
        self,
        config_path: str = None,
        config: dict = None,
        # 以下关键字参数优先级高于配置文件
        tag_mapping_path: str = None,
        processed_dir: str = None,
        time_window: Tuple[int, int] = None,
        default_tags: List[str] = None,
        missing_channel_strategy: str = None,
        log_enable: bool = None
    ):
        """
        初始化预处理器

        参数:
            config_path             : YAML 配置文件路径，与 config 二选一
            config                  : 已解析的配置字典，与 config_path 二选一
            tag_mapping_path        : 标签映射Excel文件路径
                                      覆盖 tags.mapping_file
            processed_dir           : 处理后数据输出目录
                                      覆盖 data.processed_dir
            time_window             : 时间窗口 (start_offset, end_offset)，单位：数据点数
                                      覆盖 time_window.start_offset / end_offset
            default_tags            : 目标标签点列表
                                      覆盖 tags.default_tags
            missing_channel_strategy: 标签列缺失时的处理策略
                                      'zero' 填零 | 'nan' 填NaN | 'skip' 跳过该标签列
                                      覆盖 data_processing.missing_channel_strategy
            log_enable              : 是否打印日志
                                      覆盖 logging.enable
        """
        # ── 1. 加载配置字典 ──
        cfg = {}
        if config is not None:
            cfg = config
        elif config_path is not None:
            cfg = self._load_yaml(config_path)

        # ── 2. 按优先级解析各参数（关键字参数 > 配置文件 > 默认值）──

        # tags.mapping_file
        self.tag_mapping_path = (
            tag_mapping_path
            or cfg.get('tags', {}).get('mapping_file')
        )

        # data.processed_dir
        raw_dir = (
            processed_dir
            or cfg.get('data', {}).get('processed_dir', './scada_processed')
        )
        self.processed_dir = _expand_path(raw_dir)

        # time_window.start_offset / end_offset
        if time_window is not None:
            self.time_window = time_window
        else:
            tw_cfg = cfg.get('time_window', {})
            self.time_window = (
                tw_cfg.get('start_offset', -600),
                tw_cfg.get('end_offset',    600)
            )

        # tags.default_tags
        self.default_tags = (
            default_tags
            if default_tags is not None
            else cfg.get('tags', {}).get('default_tags', [])
        )

        # data_processing.missing_channel_strategy
        raw_strategy = (
            missing_channel_strategy
            or cfg.get('data_processing', {}).get('missing_channel_strategy', 'zero')
        )
        if raw_strategy not in self._VALID_STRATEGIES:
            raise ValueError(
                f"missing_channel_strategy 非法值: '{raw_strategy}'，"
                f"合法值: {self._VALID_STRATEGIES}"
            )
        self.missing_channel_strategy = raw_strategy

        # data_processing.min_rows_after_window
        self.min_rows_after_window = (
            cfg.get('data_processing', {}).get('min_rows_after_window', 5)
        )

        # logging.enable
        if log_enable is not None:
            self.log_enable = log_enable
        else:
            self.log_enable = cfg.get('logging', {}).get('enable', True)

        # ── 3. 初始化底层读取器 ──
        self.scada_processor = ScadaDataProcessor(
            tag_info_path=self.tag_mapping_path
        )

    # ------------------------------------------------------------------
    # 配置加载
    # ------------------------------------------------------------------

    @staticmethod
    def _load_yaml(config_path: str) -> dict:
        """读取 YAML 配置文件"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        return cfg or {}

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def prepare(
        self,
        scada_path: str,
        fault_timestamp: Union[str, datetime],
        tag_points: List[str] = None,
        time_window: Tuple[int, int] = None,
        overwrite: bool = False
    ) -> dict:
        """
        处理单个SCADA文件并落盘

        参数:
            scada_path     : 原始SCADA文件路径
            fault_timestamp: 故障时刻
            tag_points     : 目标标签点，None 则使用 default_tags
            time_window    : 时间窗口，None 则使用初始化时的配置
            overwrite      : 是否覆盖已存在的输出文件

        返回:
            {
                'status'      : 'success' | 'skipped' | 'failed',
                'output_path' : str,        # 落盘文件路径（success/skipped 时有效）
                'rows'        : int | None, # 数据行数
                'warning'     : str | None, # 时间窗口警告
                'error_type'  : str,        # 失败时的错误类型
                'message'     : str         # 失败时的错误信息
            }
        """
        tags = tag_points if tag_points is not None else self.default_tags
        tw   = time_window if time_window is not None else self.time_window

        output_path = self._resolve_output_path(scada_path, fault_timestamp)

        # 已存在且不覆盖 → 跳过
        if os.path.exists(output_path) and not overwrite:
            if self.log_enable:
                print(f"  ⏭ 已存在，跳过: {os.path.basename(output_path)}")
            return {
                'status': 'skipped',
                'output_path': output_path,
                'rows': None,
                'warning': None
            }

        try:
            # 1. 读取原始文件
            raw_data = self._read_raw(scada_path, tags)

            # 2. 截取时间窗口
            windowed, is_warning, warning_msg = extract_time_window(
                raw_data, fault_timestamp, tw
            )

            if windowed.empty or len(windowed) < self.min_rows_after_window:
                msg = (
                    warning_msg
                    or f"时间窗口截取后行数不足 {self.min_rows_after_window} 行"
                )
                return {
                    'status': 'failed',
                    'output_path': None,
                    'rows': len(windowed),
                    'warning': msg,
                    'error_type': 'INSUFFICIENT_DATA',
                    'message': msg
                }

            # 3. 构建输出 DataFrame（传入 tw，用于生成 timestamp 列）
            processed = self._build_processed_df(windowed, fault_timestamp, tags, tw)

            # 4. 落盘
            os.makedirs(self.processed_dir, exist_ok=True)
            processed.to_csv(output_path, index=False, encoding='utf-8-sig')

            if self.log_enable:
                print(f"  💾 已保存: {os.path.basename(output_path)} "
                      f"({len(processed)} 行, {len(processed.columns)} 列)")

            return {
                'status': 'success',
                'output_path': output_path,
                'rows': len(processed),
                'warning': warning_msg if is_warning else None
            }

        except FileNotFoundError as e:
            return {
                'status': 'failed',
                'output_path': None,
                'rows': 0,
                'warning': None,
                'error_type': 'FILE_NOT_FOUND',
                'message': str(e)
            }
        except Exception as e:
            return {
                'status': 'failed',
                'output_path': None,
                'rows': 0,
                'warning': None,
                'error_type': 'UNEXPECTED',
                'message': f"{e}\n{traceback.format_exc()}"
            }

    def load_processed(self, output_path: str) -> pd.DataFrame:
        """
        读取已落盘的预处理文件

        参数:
            output_path: 落盘文件路径

        返回:
            DataFrame，包含：
              - real_time : datetime，仅供排查，不参与特征提取
              - timestamp : float64，相对故障时刻的点数偏移
              - 各标签列 : float64
        """
        df = pd.read_csv(output_path, encoding='utf-8-sig')

        # real_time 恢复为 datetime
        if 'real_time' in df.columns:
            df['real_time'] = pd.to_datetime(df['real_time'], errors='coerce')

        # timestamp 确保为 float64
        if 'timestamp' in df.columns:
            df['timestamp'] = df['timestamp'].astype(np.float64)

        # 标签列确保为 float64
        for col in df.columns:
            if col not in ('real_time', 'timestamp'):
                df[col] = (
                    pd.to_numeric(df[col], errors='coerce')
                    .fillna(0.0)
                    .astype(np.float64)
                )

        return df

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _resolve_output_path(
        self,
        scada_path: str,
        fault_timestamp: Union[str, datetime]
    ) -> str:
        """计算落盘文件的完整路径"""
        filename = _make_output_filename(scada_path, fault_timestamp)
        return os.path.join(self.processed_dir, filename)

    def _read_raw(self, scada_path: str, tag_points: List[str]) -> pd.DataFrame:
        """
        读取原始SCADA文件，按标签点过滤

        参数:
            scada_path: 文件路径
            tag_points: 目标标签点

        返回:
            包含 real_time 列和目标标签列的 DataFrame
        """
        if not os.path.exists(scada_path):
            raise FileNotFoundError(f"SCADA文件不存在: {scada_path}")

        data = self.scada_processor.get_data_by_scadapath(
            scada_path,
            need_tags=tag_points if tag_points else None
        )

        if data.empty:
            raise ValueError(f"文件读取结果为空: {scada_path}")

        if 'real_time' not in data.columns:
            raise ValueError(f"数据中缺少 real_time 列: {scada_path}")

        return data

    def _build_processed_df(
        self,
        windowed: pd.DataFrame,
        fault_timestamp: Union[str, datetime],
        tags: List[str],
        time_window: Tuple[int, int]
    ) -> pd.DataFrame:
        """
        构建落盘用的 DataFrame

        列顺序：real_time | timestamp | tag1 | tag2 | ...

        - real_time : 原始时间字符串，保留供排查
        - timestamp : 相对故障时刻的点数偏移（float64，故障点=0）
                      范围严格等于 [start_offset, end_offset]
        - 各标签列 : float64，占位符已清洗
                     缺失时按 self.missing_channel_strategy 处理：
                       'zero' → 填零
                       'nan'  → 填 NaN
                       'skip' → 不写入该列

        参数:
            windowed       : 时间窗口截取后的 DataFrame
            fault_timestamp: 故障时刻（保留参数，供子类扩展使用）
            tags           : 目标标签点列表
            time_window    : (start_offset, end_offset)，与 extract_time_window 一致

        返回:
            处理后的 DataFrame
        """
        df = windowed.reset_index(drop=True)
        n  = len(df)

        start_offset, end_offset = time_window
        expected_len = end_offset - start_offset + 1

        # 长度校验：windowed 行数必须与时间窗口期望长度一致
        if n != expected_len:
            raise ValueError(
                f"windowed 行数 ({n}) 与 time_window {time_window} "
                f"期望长度 ({expected_len}) 不匹配，"
                f"请检查 extract_time_window 是否正常执行。"
            )

        result = pd.DataFrame()

        # ── real_time：保留原始值（字符串形式，方便CSV查看）──
        result['real_time'] = df['real_time'].astype(str)

        # ── timestamp：相对故障点的点数偏移，范围严格为 [start_offset, end_offset] ──
        result['timestamp'] = np.arange(
            start_offset,
            end_offset + 1,
            dtype=np.float64
        )

        # ── 各标签列：float64，按策略处理缺失 ──
        for tag in tags:
            if tag in df.columns:
                result[tag] = _clean_numeric_column(df[tag])
            else:
                if self.missing_channel_strategy == 'zero':
                    result[tag] = np.zeros(n, dtype=np.float64)
                    if self.log_enable:
                        print(f"  ⚠ 标签 {tag} 不存在，按策略 'zero' 填零")
                elif self.missing_channel_strategy == 'nan':
                    result[tag] = np.full(n, np.nan, dtype=np.float64)
                    if self.log_enable:
                        print(f"  ⚠ 标签 {tag} 不存在，按策略 'nan' 填NaN")
                else:  # 'skip'
                    if self.log_enable:
                        print(f"  ⚠ 标签 {tag} 不存在，按策略 'skip' 跳过")

        return result
