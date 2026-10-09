import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import traceback
from configs.config_loader import TIMESTAMP_WINDOW, DEFAULT_FAULT_TIME, COLUMN_MAPPING
from error_codes import ErrorCode

warnings.filterwarnings('ignore', category=RuntimeWarning)

# 获取日志记录器
logger = logging.getLogger(__name__)


class DataProcessor:
    """
    DataProcessor - 工业数据清洗
    规范化日志：使用 [阶段标签]
    异常处理：返回 (Result, Error_Code, Message)
    """

    def __init__(self, custom_mapping=None):
        self.COLUMN_MAPPING = custom_mapping or COLUMN_MAPPING
        if not self.COLUMN_MAPPING:
            raise ValueError("未提供列名映射表，请检查 column_mapping_file 配置或 config_loader.py")
        self.logger = logger
        logger.info("[INIT] DataProcessor 映射配置加载完成")

    def load_excel_data(self, excel_path):
        """读取任务索引表"""
        logger.info(f"[EXCEL] 尝试读取任务文件: {excel_path}")
        try:
            df = pd.read_excel(excel_path)
            logger.info(f"✅ Excel 读取成功, 样本总数: {len(df)}")
            return df
        except Exception as e:
            logger.error(f"❌ [{ErrorCode.EXCEL_READ_ERROR.name}({ErrorCode.EXCEL_READ_ERROR.value})] 读取Excel失败: {str(e)}")
            raise  # 索引表失败通常是毁灭性的，直接抛出

    def _get_error_index(self, data):
        """根据运行模式跳变推断故障索引。"""
        mode_columns = [
            'iOperationMode', 'giWindTurbineOperationMode',
            'ioperation_mode', 'iTurbineOperationMode',
            'giTurbineOperationMode', 'GVL_OP_gS_OP_giOperationMode',
            'rOperationMode', 'grOperationMode', 'giOperationMode'
        ]

        mode_column = None
        for column in mode_columns:
            if column in data.columns:
                mode_column = column
                break

        if mode_column is None:
            return None, None

        mode_series = pd.to_numeric(data[mode_column], errors='coerce')
        error_indexes = []

        for i in range(1, len(mode_series)):
            prev_mode = mode_series.iloc[i - 1]
            curr_mode = mode_series.iloc[i]

            if pd.isna(prev_mode) or pd.isna(curr_mode):
                continue

            prev_mode = int(prev_mode)
            curr_mode = int(curr_mode)

            if curr_mode in range(1, 8) and prev_mode in range(8, 16):
                error_indexes.append(i - 1)

        if error_indexes:
            return error_indexes[0], mode_column, error_indexes
        return None, mode_column, error_indexes

    def extract_columns_with_fallback(self, csv_file, fault_time=DEFAULT_FAULT_TIME, timestamp_range=TIMESTAMP_WINDOW):
        """
        核心方法：带溯源的列提取与时间清洗
        返回: (DataFrame or None, ErrorCode or None, Message)
        """
        file_name = Path(csv_file).name

        try:
            # --- 1. 读取文件 ---
            df = None
            for encoding in ['gbk', 'utf-8', 'utf-8-sig', 'latin-1']:
                try:
                    # 先读前几KB检测分隔符（用二进制模式避免换行符问题）
                    with open(csv_file, 'rb') as f:
                        raw_head = f.read(4096).decode(encoding, errors='ignore')
                    
                    # 按出现次数判断分隔符
                    sep_counts = {
                        ';': raw_head.count(';'),
                        ',': raw_head.count(','),
                        '\t': raw_head.count('\t')
                    }
                    sep = max(sep_counts, key=sep_counts.get)
                    if sep_counts[sep] == 0:
                        sep = ','  # 兜底
                    
                    logger.debug(f"[{file_name}] 编码={encoding}, 分隔符计数={sep_counts}, 选择='{sep}'")
                    
                    # 优先用c引擎（更快更鲁棒），失败再用python引擎
                    try:
                        df = pd.read_csv(csv_file, sep=sep, index_col=False, encoding=encoding, engine='c')
                    except Exception:
                        df = pd.read_csv(csv_file, sep=sep, index_col=False, encoding=encoding, engine='python')
                    
                    # 如果只读到1列，可能是文件头部有多余行，尝试跳过前几行
                    if len(df.columns) <= 1:
                        for skip in range(1, 6):
                            try:
                                df_retry = pd.read_csv(csv_file, sep=sep, index_col=False, encoding=encoding,
                                                       engine='c', skiprows=skip)
                            except Exception:
                                try:
                                    df_retry = pd.read_csv(csv_file, sep=sep, index_col=False, encoding=encoding,
                                                           engine='python', skiprows=skip)
                                except Exception:
                                    continue
                            if len(df_retry.columns) > 1:
                                logger.warning(f"[{file_name}] 跳过前 {skip} 行后成功读取 (列数: {len(df_retry.columns)})")
                                df = df_retry
                                break

                    # 自动检测并修复列名偏移
                    actual_time_idx = -1
                    for i in range(min(5, len(df.columns))):
                        col_data = pd.to_numeric(df.iloc[:, i], errors='coerce')
                        if col_data.min() < -100:
                            actual_time_idx = i
                            break

                    if actual_time_idx > 0:
                        self.logger.warning(
                            f"⚠️ 检测到列位移！真正的时间轴在物理第 {actual_time_idx} 列。正在重新对齐...")
                        original_headers = df.columns.tolist()
                        new_headers = ["unnamed_offset_" + str(i) for i in range(actual_time_idx)] + original_headers
                        df.columns = new_headers[:len(df.columns)]

                    if len(df.columns) > 1: break
                except:
                    continue
            
            # 如果所有编码都只读到1列，强制用分号分隔符重试
            if df is not None and len(df.columns) <= 1:
                for encoding in ['gbk', 'utf-8', 'utf-8-sig', 'latin-1']:
                    try:
                        df = pd.read_csv(csv_file, sep=';', index_col=False, encoding=encoding, engine='python')
                        if len(df.columns) > 1:
                            logger.debug(f"[{file_name}] 强制分号分隔重试成功 (encoding={encoding}, cols={len(df.columns)})")
                            break
                    except:
                        continue

            if df is None:
                return None, ErrorCode.CSV_PARSE_ERROR, f"无法解析CSV编码或文件格式错误: {file_name}"

            # --- 调试：打印读取结果 ---
            logger.debug(f"[{file_name}] 读取列数: {len(df.columns)}, 行数: {len(df)}, 前3列名: {list(df.columns[:3])}")

            # --- 1.5 修复第一列列名被元数据前缀污染 ---
            # 某些txt文件第一行格式如: "13D05.20_NS01_008_P01_20240921grTime;iOperationMode;..."
            # 导致第一列名变成 "13D05.20_NS01_008_P01_20240921grTime"
            first_col = df.columns[0]
            # 只有当第一列名比候选名长很多时（前缀至少10个字符），才认为是前缀污染
            all_candidates = []
            for candidates in self.COLUMN_MAPPING.values():
                if isinstance(candidates, list):
                    all_candidates.extend(candidates)
            # 按长度降序排列，优先匹配最长的候选名
            all_candidates.sort(key=len, reverse=True)
            for candidate in all_candidates:
                prefix_len = len(first_col) - len(candidate)
                if prefix_len >= 10 and first_col.endswith(candidate):
                    logger.warning(f"[{file_name}] 修复第一列名前缀污染: '{first_col}' → '{candidate}'")
                    df.rename(columns={first_col: candidate}, inplace=True)
                    break

            # --- 2. 时间维度挖掘 (Index Recovery) ---
            index_vals = df.index.to_numpy()
            if index_vals.min() < 0:
                df.reset_index(inplace=True)
                logger.debug(f"[{file_name}] 时间数据已从 Index 恢复至列")

            # --- 3. 智能列选 PK 逻辑 ---
            # 兼容 'timestamp' 和 'time_stamp' 两种标准名
            time_key = None
            for key in ['timestamp', 'time_stamp']:
                if key in self.COLUMN_MAPPING:
                    time_key = key
                    break
            
            potential_cols = [c for c in self.COLUMN_MAPPING.get(time_key, []) if c in df.columns] if time_key else []
            best_time_col, max_std = None, -1

            for col in potential_cols:
                s = pd.to_numeric(df[col], errors='coerce').fillna(0)
                curr_std = s.std()
                if curr_std > max_std:
                    max_std = curr_std
                    best_time_col = col
            
            # 如果数值匹配失败，检查是否有日期时间格式的列
            if best_time_col is None or max_std <= 0:
                for col in potential_cols:
                    try:
                        # 尝试多种日期格式
                        ts = None
                        sample_val = str(df[col].dropna().iloc[0])
                        
                        # 格式: 2025-05-30-10-07-58-659 (年-月-日-时-分-秒-毫秒)
                        if len(sample_val.split('-')) >= 6:
                            ts = pd.to_datetime(df[col], format='%Y-%m-%d-%H-%M-%S-%f', errors='coerce')
                        
                        # 通用解析
                        if ts is None or ts.notna().sum() < len(df) * 0.5:
                            ts = pd.to_datetime(df[col], errors='coerce')
                        
                        if ts.notna().sum() > len(df) * 0.5:
                            ref_time = ts.dropna().iloc[len(ts.dropna()) // 2]
                            df[col] = (ts - ref_time).dt.total_seconds()
                            best_time_col = col
                            max_std = 1
                            logger.info(f"[{file_name}] 检测到日期时间列: {col}，已转换为相对秒数")
                            break
                    except Exception:
                        continue

            error_index, mode_column, error_indexes = self._get_error_index(df)

            # --- 4. 构建映射 ---
            mapped_cols = {}
            if error_index is not None:
                if error_index <= 0:
                    logger.debug(
                        f"[{file_name}] 运行模式候选故障索引(前5个): "
                        f"{error_indexes[:5]}"
                    )
                    logger.debug(
                        f"[{file_name}] '{mode_column}' 在故障索引附近 [0:{min(len(df), 4)}] 的取值: "
                        f"{pd.to_numeric(df[mode_column], errors='coerce').iloc[:min(len(df), 4)].to_dict()}"
                    )
                    return None, ErrorCode.TIME_WINDOW_EMPTY, (
                        f"运行模式列 '{mode_column}' 识别到的故障索引为 {error_index}，"
                        f"无法提供故障前窗口，跳过样本"
                    )

                df['fault_index_timestamp'] = (
                    np.arange(len(df), dtype=np.float64) - float(error_index)
                )
                mapped_cols['fault_index_timestamp'] = 'timestamp'
                debug_start = max(0, error_index - 3)
                debug_end = min(len(df), error_index + 4)
                mode_debug = pd.to_numeric(
                    df[mode_column], errors='coerce'
                ).iloc[debug_start:debug_end]
                logger.info(
                    f"[{file_name}] 通过运行模式列 '{mode_column}' "
                    f"识别故障索引 {error_index}，使用相对索引时间轴"
                )
                logger.debug(
                    f"[{file_name}] 运行模式候选故障索引(前5个): "
                    f"{error_indexes[:5]}"
                )
                logger.debug(
                    f"[{file_name}] '{mode_column}' 在故障索引附近 "
                    f"[{debug_start}:{debug_end}] 的取值: "
                    f"{mode_debug.to_dict()}"
                )
            else:
                if mode_column is not None:
                    logger.debug(
                        f"[{file_name}] 运行模式列 '{mode_column}' 未识别到 "
                        f"8~15 -> 1~7 跳变"
                    )
                    return None, ErrorCode.TIME_WINDOW_EMPTY, (
                        f"运行模式列 '{mode_column}' 未识别到 8~15 -> 1~7 跳变，跳过样本"
                    )

                if best_time_col and max_std > 0:
                    mapped_cols[best_time_col] = 'timestamp'
                else:
                    logger.warning(f"[{file_name}] 未找到天然时间列，执行强制时间重构")
                    row_cnt = len(df)
                    df['forced_time'] = np.linspace(-row_cnt / 2, row_cnt / 2, row_cnt)
                    mapped_cols['forced_time'] = 'timestamp'

            for std_col, candidates in self.COLUMN_MAPPING.items():
                if std_col in ('timestamp', 'time_stamp'): continue
                for cand in candidates:
                    if cand in df.columns:
                        mapped_cols[cand] = std_col
                        break

            if 'timestamp' not in mapped_cols.values():
                return None, ErrorCode.TIMESTAMP_MISSING, "无法构造 timestamp 列"

            # --- 5. 数据提取与窗口对齐 ---
            df_processed = df[list(mapped_cols.keys())].rename(columns=mapped_cols).copy()
            # 强制将除了 timestamp 以外的所有列也转换为数值类型
            for col in df_processed.columns:
                # errors='coerce' 会把无法转换的脏字符变成 NaN，随后会被填充为 0
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')

            # 智能偏置校正
            raw_mean = df_processed['timestamp'].mean()
            if abs(raw_mean) > 10000 and fault_time:
                df_processed['timestamp'] = df_processed['timestamp'] - fault_time

            # 窗口截取
            if timestamp_range:
                mask = (df_processed['timestamp'] >= timestamp_range[0]) & \
                       (df_processed['timestamp'] <= timestamp_range[1])
                df_processed = df_processed[mask].copy()

            if df_processed.empty:
                return None, ErrorCode.TIME_WINDOW_EMPTY, f"时间窗口截取后无有效数据: {timestamp_range}"

            return df_processed, None, "Success"

        except Exception as e:
            logger.error(f"❌ [{file_name}] 处理异常: {str(e)}")
            return None, ErrorCode.DATA_PROCESS_CRASH, f"内部处理崩溃: {str(e)}"

    def clean_features(self, features):
        """清洗特征中的无效值"""
        if features is None:
            return None
        return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    def save_processed_data(self, df, output_path):
        """保存清洗后的 CSV"""
        try:
            path_obj = Path(output_path)
            path_obj.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path_obj, index=False, encoding='utf-8-sig')
            logger.info(f"💾 中间数据已存至: {output_path}")
        except Exception as e:
            logger.error(f"❌ 保存失败 {output_path}: {e}")

    def validate_data_quality(self, df):
        """数据质量快速校验"""
        if df is None or df.empty:
            return False, ErrorCode.DATA_IS_EMPTY, "数据为空"
        if 'timestamp' not in df.columns:
            return False, ErrorCode.TIMESTAMP_MISSING, "缺失 timestamp 关键列"
        return True, None, "Pass"
