"""
数据准备模块（C 语言接口版）

从 tracelog CSV 文件生成用于预测的 .pt 文件，
使用与训练数据预处理相同的逻辑，确保数据格式一致。

对外暴露核心函数:
    prepare_data(config, tracelog, result_dict) → int

调用示例:
    result_dict = {}
    error_code = prepare_data(
        config="configs/preprocess.yaml",
        tracelog="path/to/tracelog.csv",
        result_dict=result_dict,
    )
    if error_code == 0:
        print(result_dict["result_dir"])

参数说明:
    config      : str  — 预处理 YAML 配置文件路径（与训练时相同）
    tracelog    : str  — tracelog CSV 文件路径
    result_dict : dict — 输出容器，函数执行后写入 {"result_dir": "<输出.pt文件路径>"}

注意:
    output 路径从 config 中 paths.output_dir 读取（默认 ./external_data）。

错误编码:
    4101 CONFIG_LOAD_ERROR        配置文件加载失败
    4102 TRACELOG_NOT_FOUND       tracelog 文件不存在
    4103 PROCESS_ERROR            tracelog 处理过程出错
    4104 SAVE_ERROR               保存 pt 文件失败
    4105 PARAM_ERROR              参数校验失败
    4199 UNKNOWN_ERROR            未知错误
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
import torch
import logging
from pathlib import Path
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Tuple

# 将项目根目录加入 sys.path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ================================================================
# 错误代码（统一定义在 api/error_codes.py）
# ================================================================
from api.error_codes import ErrorCode, log_error as _log_error


# ================================================================
# Tracelog 预处理器（从原 prepare_dp_data.py 提取）
# ================================================================
class _TracelogPreprocessor:
    """Tracelog 文件预处理器（用于预测）"""

    def __init__(self, config: Dict):
        self.config = config
        self.logger = self._setup_logger()
        self.features = config['features']
        self.processing = config['processing']
        self.normalization = config.get('normalization', {})
        self.norm_stats = None
        if self.normalization.get('save_stats', False):
            self._load_normalization_stats()

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger('TracelogPreprocessor')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            logger.addHandler(handler)
        return logger

    def _load_normalization_stats(self):
        if 'paths' not in self.config:
            return
        output_dir = self.config['paths'].get('output_dir', '')
        stats_filename = self.normalization.get(
            'stats_filename', 'normalization_stats.npz')
        stats_path = os.path.join(output_dir, stats_filename)
        if os.path.exists(stats_path):
            self.norm_stats = np.load(stats_path)
            self.logger.info(f"已加载标准化统计量: {stats_path}")

    def _load_synonym_groups(self) -> Tuple[List[List[str]], List[str]]:
        target_tags = self.features['target_tags']
        if 'paths' not in self.config or 'config_file' not in self.config['paths']:
            return [[t] for t in target_tags], target_tags

        config_file = self.config['paths']['config_file']
        if not os.path.exists(config_file):
            return [[t] for t in target_tags], target_tags

        config_df = pd.read_csv(config_file)
        synonym_groups, tag_names = [], []
        for tag in target_tags:
            row = config_df[config_df['tags'] == tag]
            if len(row) == 0:
                synonym_groups.append([tag])
                tag_names.append(tag)
                continue
            config_cols = [c for c in config_df.columns if c.startswith('config')]
            synonyms = []
            for col in config_cols:
                val = row[col].values[0]
                if pd.notna(val) and str(val).strip() not in ['1', '']:
                    synonyms.append(str(val).strip())
            synonym_groups.append(synonyms if synonyms else [tag])
            tag_names.append(tag)
        return synonym_groups, tag_names

    def _find_matching_column(self, columns: List[str],
                              synonym_group: List[str]) -> Optional[str]:
        for synonym in synonym_group:
            for col in columns:
                if synonym.lower() in col.lower() or col.lower() in synonym.lower():
                    return col
        return None

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        method = self.processing.get('missing_value_method', 'interpolate')
        if method == 'interpolate':
            df = df.interpolate(method='linear', limit_direction='both')
        elif method == 'ffill':
            df = df.fillna(method='ffill')
        elif method == 'bfill':
            df = df.fillna(method='bfill')
        elif method == 'mean':
            df = df.fillna(df.mean())
        elif method == 'zero':
            df = df.fillna(0)
        if df.isna().sum().sum() > 0:
            df.fillna(0, inplace=True)
        return df

    def _adjust_sequence_length(self, data: np.ndarray) -> np.ndarray:
        max_len = self.processing['max_seq_len']
        padding_mode = self.processing.get('padding_mode', 'constant')
        if data.shape[0] < max_len:
            pad_width = max_len - data.shape[0]
            data = np.pad(data, ((0, pad_width), (0, 0)), mode=padding_mode)
        elif data.shape[0] > max_len:
            data = data[:max_len, :]
        return data

    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        method = self.normalization.get('method', 'none')
        if method is None or method == 'none' or self.norm_stats is None:
            return data
        n_features, seq_len = data.shape
        data_r = data.T
        if method == 'zscore':
            data_r = (data_r - self.norm_stats['mean']) / self.norm_stats['std']
        elif method == 'minmax':
            mn, mx = self.norm_stats['min'], self.norm_stats['max']
            rng = np.where((mx - mn) == 0, 1, mx - mn)
            data_r = (data_r - mn) / rng
        elif method == 'robust':
            data_r = (data_r - self.norm_stats['median']) / self.norm_stats['iqr']
        else:
            return data
        return data_r.T

    def process_tracelog(self, tracelog_path: str) -> Optional[np.ndarray]:
        """处理单个 tracelog，返回 (n_features, seq_len) 数组。"""
        if not os.path.exists(tracelog_path):
            return None

        synonym_groups, tag_names = self._load_synonym_groups()
        df = pd.read_csv(tracelog_path)

        selected_columns = []
        for i, sg in enumerate(synonym_groups):
            matched = self._find_matching_column(df.columns, sg)
            if matched:
                selected_columns.append(matched)

        if not selected_columns:
            self.logger.error("未找到任何匹配的列")
            return None

        df_sel = df[selected_columns].copy()

        if df_sel.isna().sum().sum() > 0:
            df_sel = self._handle_missing_values(df_sel)

        df_sel.replace([np.inf, -np.inf], 0, inplace=True)

        data = self._adjust_sequence_length(df_sel.values)  # (seq_len, feat)
        data = data.T  # (feat, seq_len)
        data = self._normalize_data(data)
        return data


# ================================================================
# 对外函数：prepare_data
# ================================================================
def prepare_data(
    config: str,
    tracelog: str,
    result_dict: dict,
) -> int:
    """
    数据准备入口（C 语言接口版）。

    从 tracelog CSV 生成预测用 .pt 文件。

    Parameters
    ----------
    config : str
        预处理 YAML 配置文件路径（与训练时使用的配置相同）。
    tracelog : str
        tracelog CSV 文件路径。
    result_dict : dict
        输出容器。执行成功后写入:
            result_dict["result_dir"] = "<输出 .pt 文件路径>"

    Returns
    -------
    int
        0 表示成功；非 0 为错误代码。
    """
    # ── 参数校验 ──
    if not isinstance(result_dict, dict):
        _log_error(ErrorCode.PREPARE_PARAM_ERROR, "result_dict 必须是 dict 类型")
        return ErrorCode.PREPARE_PARAM_ERROR.value

    if not os.path.exists(config):
        _log_error(ErrorCode.PREPARE_CONFIG_ERROR, f"配置文件不存在: {config}")
        return ErrorCode.PREPARE_CONFIG_ERROR.value

    if not os.path.exists(tracelog):
        _log_error(ErrorCode.TRACELOG_NOT_FOUND, f"文件不存在: {tracelog}")
        return ErrorCode.TRACELOG_NOT_FOUND.value

    # ── 加载配置 ──
    try:
        with open(config, 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f)
    except Exception as e:
        _log_error(ErrorCode.PREPARE_CONFIG_ERROR, str(e))
        return ErrorCode.PREPARE_CONFIG_ERROR.value

    # ── 创建预处理器 ──
    try:
        preprocessor = _TracelogPreprocessor(yaml_config)
    except Exception as e:
        _log_error(ErrorCode.PREPARE_CONFIG_ERROR, str(e))
        return ErrorCode.PREPARE_CONFIG_ERROR.value

    # ── 处理 tracelog ──
    try:
        data = preprocessor.process_tracelog(tracelog)
    except Exception as e:
        _log_error(ErrorCode.TRACELOG_PROCESS_ERROR, str(e))
        return ErrorCode.TRACELOG_PROCESS_ERROR.value

    if data is None:
        _log_error(ErrorCode.TRACELOG_PROCESS_ERROR, "处理结果为空，请检查 tracelog 内容和配置")
        return ErrorCode.TRACELOG_PROCESS_ERROR.value

    # ── 确定输出路径 ──
    output_dir = yaml_config.get('paths', {}).get('output_dir', './external_data')
    tracelog_stem = Path(tracelog).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{tracelog_stem}_{timestamp}.pt"
    output_path = os.path.join(output_dir, output_filename)

    # ── 保存 .pt 文件 ──
    try:
        data_tensor = torch.from_numpy(data).float().unsqueeze(0)  # (1, feat, seq)
        data_dict = {'samples': data_tensor}

        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        torch.save(data_dict, output_path, _use_new_zipfile_serialization=True)
    except Exception as e:
        _log_error(ErrorCode.PREPARE_SAVE_ERROR, str(e))
        return ErrorCode.PREPARE_SAVE_ERROR.value

    # ── 写入输出字典 ──
    result_dict["result_dir"] = output_path

    print(f"✓ 数据准备完成: {output_path} (shape: {data_tensor.shape})")
    return 0


# ================================================================
# 命令行入口
# ================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="从 tracelog CSV 生成预测用 .pt 文件（C 语言接口版）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, required=True,
                        help="预处理 YAML 配置文件路径")
    parser.add_argument("--tracelog", type=str, required=True,
                        help="tracelog CSV 文件路径")

    args = parser.parse_args()

    result_dict = {}
    code = prepare_data(
        config=args.config,
        tracelog=args.tracelog,
        result_dict=result_dict,
    )

    if code == 0:
        print(f"输出路径: {result_dict['result_dir']}")
    raise SystemExit(code)
