"""
从 tracelog CSV 文件生成用于预测的 .pt 文件
使用与训练数据预处理相同的逻辑，确保数据格式一致

用法示例:
    python prepare_external_data.py \
        --tracelog path/to/tracelog.csv \
        --config configs/preprocess.yaml \
        --output ./external_data/predict_data.pt

此脚本使用错误类型编码以便快速诊断问题。

错误编码说明:
    4101 CONFIG_LOAD_ERROR        配置文件加载失败
    4102 TRACELOG_NOT_FOUND        tracelog 文件不存在
    4103 PROCESS_ERROR             tracelog 处理过程出错
    4104 SAVE_ERROR                保存 pt 文件失败
    4199 UNKNOWN_ERROR             未知错误
"""
    
import argparse
import yaml
import os
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import logging
from enum import Enum


class ErrorCode(Enum):
    CONFIG_LOAD_ERROR = 4101
    TRACELOG_NOT_FOUND = 4102
    PROCESS_ERROR = 4103
    SAVE_ERROR = 4104
    UNKNOWN_ERROR = 4199


_error_descriptions = {
    ErrorCode.CONFIG_LOAD_ERROR: "配置文件加载失败，可能路径不存在或 YAML 语法错误",
    ErrorCode.TRACELOG_NOT_FOUND: "指定的 tracelog 文件不存在",
    ErrorCode.PROCESS_ERROR: "在处理 tracelog 时发生异常",
    ErrorCode.SAVE_ERROR: "保存输出 .pt 文件失败",
    ErrorCode.UNKNOWN_ERROR: "未知错误，请查看日志"
}


def log_error(code: ErrorCode, message: str = ""):
    desc = _error_descriptions.get(code, "")
    print(f"错误代码: {code.value} ({code.name}) -> {desc}")
    if message:
        print(f"详细信息: {message}")



class TracelogPreprocessor:
    """Tracelog 文件预处理器（用于预测）"""
    
    def __init__(self, config: Dict):
        """
        初始化预处理器
        
        Args:
            config: 配置字典（与训练时相同的配置）
        """
        self.config = config
        self.logger = self._setup_logger()
        
        # 提取配置参数
        self.features = config['features']
        self.processing = config['processing']
        self.normalization = config.get('normalization', {})
        
        # 加载标准化统计量（如果存在）
        self.norm_stats = None
        if self.normalization.get('save_stats', False):
            self._load_normalization_stats()
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger('TracelogPreprocessor')
        logger.setLevel(logging.INFO)
        
        # 避免重复添加 handler
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def _load_normalization_stats(self):
        """加载训练时保存的标准化统计量"""
        if 'paths' not in self.config:
            self.logger.warning("配置中未找到 paths，跳过加载标准化统计量")
            return
        
        output_dir = self.config['paths'].get('output_dir', '')
        stats_filename = self.normalization.get('stats_filename', 'normalization_stats.npz')
        stats_path = os.path.join(output_dir, stats_filename)
        
        if os.path.exists(stats_path):
            self.norm_stats = np.load(stats_path)
            self.logger.info(f"已加载标准化统计量: {stats_path}")
            self.logger.info(f"  包含的统计量: {list(self.norm_stats.keys())}")
        else:
            self.logger.warning(f"未找到标准化统计量文件: {stats_path}")
            self.logger.warning("预测数据将不进行标准化！")
    
    def _load_synonym_groups(self) -> Tuple[List[List[str]], List[str]]:
        """加载同义词组（与训练时相同的逻辑）"""
        self.logger.info("正在加载特征配置...")
        
        # 检查是否有配置文件路径
        if 'paths' not in self.config or 'config_file' not in self.config['paths']:
            # 如果没有配置文件，直接使用 target_tags
            self.logger.info("未找到配置文件，直接使用 target_tags 作为列名")
            target_tags = self.features['target_tags']
            synonym_groups = [[tag] for tag in target_tags]
            tag_names = target_tags
            return synonym_groups, tag_names
        
        config_file = self.config['paths']['config_file']
        if not os.path.exists(config_file):
            self.logger.warning(f"配置文件不存在: {config_file}")
            self.logger.info("将直接使用 target_tags 作为列名")
            target_tags = self.features['target_tags']
            synonym_groups = [[tag] for tag in target_tags]
            tag_names = target_tags
            return synonym_groups, tag_names
        
        # 读取配置文件
        config_df = pd.read_csv(config_file)
        target_tags = self.features['target_tags']
        
        synonym_groups = []
        tag_names = []
        
        for tag in target_tags:
            row = config_df[config_df['tags'] == tag]
            
            if len(row) == 0:
                self.logger.warning(f"配置表中未找到标签 '{tag}'，将直接使用该标签名")
                synonym_groups.append([tag])
                tag_names.append(tag)
                continue
            
            config_cols = [col for col in config_df.columns 
                          if col.startswith('config')]
            synonyms = []
            
            for col in config_cols:
                value = row[col].values[0]
                if pd.notna(value) and str(value).strip() not in ['1', '']:
                    synonyms.append(str(value).strip())
            
            if len(synonyms) > 0:
                synonym_groups.append(synonyms)
                tag_names.append(tag)
                self.logger.info(
                    f"加载标签 '{tag}': 找到 {len(synonyms)} 个同义列名"
                )
            else:
                # 如果没有找到同义词，使用标签名本身
                synonym_groups.append([tag])
                tag_names.append(tag)
        
        self.logger.info(f"成功加载 {len(synonym_groups)} 个特征组")
        return synonym_groups, tag_names
    
    def _find_matching_column(
        self,
        columns: List[str],
        synonym_group: List[str]
    ) -> Optional[str]:
        """在列名中查找匹配的同义词"""
        for synonym in synonym_group:
            for col in columns:
                if synonym.lower() in col.lower() or col.lower() in synonym.lower():
                    return col
        return None
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理缺失值（与训练时相同的逻辑）"""
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
        
        # 最后检查
        remaining_nan = df.isna().sum().sum()
        if remaining_nan > 0:
            self.logger.warning(
                f"填充后仍有 {remaining_nan} 个空值，将用0填充"
            )
            df.fillna(0, inplace=True)
        
        return df
    
    def _adjust_sequence_length(self, data: np.ndarray) -> np.ndarray:
        """调整序列长度（与训练时相同的逻辑）"""
        max_len = self.processing['max_seq_len']
        padding_mode = self.processing.get('padding_mode', 'constant')
        
        if data.shape[0] < max_len:
            # 填充
            pad_width = max_len - data.shape[0]
            data = np.pad(
                data, ((0, pad_width), (0, 0)), mode=padding_mode
            )
            self.logger.info(f"填充: {data.shape[0] - pad_width} -> {max_len}")
            
        elif data.shape[0] > max_len:
            # 截断
            original_len = data.shape[0]
            data = data[:max_len, :]
            self.logger.info(f"截断: {original_len} -> {max_len}")
        
        return data
    
    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """标准化数据（使用训练时的统计量）"""
        method = self.normalization.get('method', 'none')
        
        if method is None or method == 'none':
            self.logger.info("跳过标准化")
            return data
        
        if self.norm_stats is None:
            self.logger.warning("未找到标准化统计量，跳过标准化！")
            return data
        
        self.logger.info(f"使用训练时的统计量进行标准化 (方法: {method})...")
        
        # 重塑为 (seq_len, n_features)
        n_features, seq_len = data.shape
        data_reshaped = data.T  # (seq_len, n_features)
        
        if method == 'zscore':
            mean = self.norm_stats['mean']
            std = self.norm_stats['std']
            data_norm = (data_reshaped - mean) / std
            
        elif method == 'minmax':
            min_val = self.norm_stats['min']
            max_val = self.norm_stats['max']
            range_val = max_val - min_val
            range_val = np.where(range_val == 0, 1, range_val)
            data_norm = (data_reshaped - min_val) / range_val
            
        elif method == 'robust':
            median = self.norm_stats['median']
            iqr = self.norm_stats['iqr']
            data_norm = (data_reshaped - median) / iqr
        
        else:
            self.logger.warning(f"未知的标准化方法: {method}，跳过标准化")
            return data
        
        # 转回 (n_features, seq_len)
        data_norm = data_norm.T
        
        self.logger.info("标准化完成")
        self.logger.info(f"  数据范围: [{data_norm.min():.4f}, {data_norm.max():.4f}]")
        self.logger.info(f"  均值: {data_norm.mean():.4f}")
        self.logger.info(f"  标准差: {data_norm.std():.4f}")
        
        return data_norm
    
    def process_tracelog(self, tracelog_path: str) -> Optional[np.ndarray]:
        """
        处理单个 tracelog 文件
        
        Args:
            tracelog_path: tracelog CSV 文件路径
            
        Returns:
            处理后的数据数组，形状为 (n_features, seq_len)
        """
        self.logger.info("=" * 60)
        self.logger.info(f"处理 tracelog 文件: {tracelog_path}")
        self.logger.info("=" * 60)
        
        if not os.path.exists(tracelog_path):
            self.logger.error(f"文件不存在: {tracelog_path}")
            log_error(ErrorCode.TRACELOG_NOT_FOUND, tracelog_path)
            return None
        
        # 加载同义词组
        synonym_groups, tag_names = self._load_synonym_groups()
        
        # 读取 CSV
        self.logger.info("读取 CSV 文件...")
        df = pd.read_csv(tracelog_path)
        self.logger.info(f"  原始数据形状: {df.shape}")
        self.logger.info(f"  列名: {list(df.columns)}")
        
        # 查找匹配的列
        selected_columns = []
        for i, synonym_group in enumerate(synonym_groups):
            matched_col = self._find_matching_column(
                df.columns, synonym_group
            )
            if matched_col:
                selected_columns.append(matched_col)
                self.logger.info(
                    f"特征 '{tag_names[i]}' 匹配到列: {matched_col}"
                )
            else:
                self.logger.warning(
                    f"特征 '{tag_names[i]}' 未找到匹配列"
                )
        
        if len(selected_columns) == 0:
            self.logger.error("未找到任何匹配的列！")
            return None
        
        # 选择相关列
        df_selected = df[selected_columns].copy()
        self.logger.info(f"选择了 {len(selected_columns)} 个特征列")
        
        # 处理缺失值
        total_missing = df_selected.isna().sum().sum()
        if total_missing > 0:
            self.logger.info(f"检测到 {total_missing} 个空值，开始处理...")
            df_selected = self._handle_missing_values(df_selected)
            self.logger.info("✓ 空值处理完成")
        
        # 处理无穷值
        inf_count = np.isinf(df_selected.values).sum()
        if inf_count > 0:
            self.logger.info(f"检测到 {inf_count} 个无穷值，替换为0")
            df_selected.replace([np.inf, -np.inf], 0, inplace=True)
        
        # 转换为 numpy 数组
        data = df_selected.values  # (seq_len, n_features)
        
        # 调整序列长度
        data = self._adjust_sequence_length(data)
        
        # 转置为 (n_features, seq_len)
        data = data.T
        
        self.logger.info(f"处理后数据形状: {data.shape}")
        
        # 标准化
        data = self._normalize_data(data)
        
        return data
    
    def save_for_prediction(
        self,
        data: np.ndarray,
        output_path: str,
        include_dummy_label: bool = False
    ):
        """
        保存为预测用的 .pt 文件
        
        Args:
            data: 处理后的数据，形状为 (n_features, seq_len)
            output_path: 输出文件路径
            include_dummy_label: 是否包含虚拟标签（某些预测脚本可能需要）
        """
        # 添加 batch 维度: (1, n_features, seq_len)
        data_tensor = torch.from_numpy(data).float().unsqueeze(0)
        
        # 构建数据字典
        data_dict = {
            'samples': data_tensor
        }
        
        if include_dummy_label:
            # 添加虚拟标签（预测时会被忽略）
            dummy_label = torch.tensor([0], dtype=torch.long)
            data_dict['labels'] = dummy_label
        
        # 保存文件
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        torch.save(
            data_dict,
            output_path,
            _use_new_zipfile_serialization=True
        )
        
        file_size = os.path.getsize(output_path) / 1024
        self.logger.info("=" * 60)
        self.logger.info("数据保存完成！")
        self.logger.info(f"  输出路径: {output_path}")
        self.logger.info(f"  文件大小: {file_size:.1f} KB")
        self.logger.info(f"  数据形状: {data_tensor.shape}")
        self.logger.info(f"  包含标签: {include_dummy_label}")
        self.logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='从 tracelog CSV 文件生成预测用的 .pt 文件'
    )
    
    parser.add_argument(
        '--tracelog',
        type=str,
        required=True,
        help='tracelog CSV 文件路径'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='预处理配置文件路径（与训练时使用的配置相同）'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='./external_data/predict_data.pt',
        help='输出 .pt 文件路径（默认: ./external_data/predict_data.pt）'
    )
    
    parser.add_argument(
        '--include-label',
        action='store_true',
        help='是否包含虚拟标签（某些预测脚本可能需要）'
    )
    
    args = parser.parse_args()
    
    # 加载配置
    try:
        print(f"\n加载配置文件: {args.config}")
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        log_error(ErrorCode.CONFIG_LOAD_ERROR, str(e))
        return 1
    
    # 创建预处理器
    preprocessor = TracelogPreprocessor(config)
    
    # 处理 tracelog
    try:
        data = preprocessor.process_tracelog(args.tracelog)
    except Exception as e:
        log_error(ErrorCode.PROCESS_ERROR, str(e))
        return 1
    
    if data is None:
        log_error(ErrorCode.TRACELOG_NOT_FOUND, args.tracelog)
        return 1
    
    # 保存为预测用的文件
    try:
        preprocessor.save_for_prediction(
            data,
            args.output,
            include_dummy_label=args.include_label
        )
    except Exception as e:
        log_error(ErrorCode.SAVE_ERROR, str(e))
        return 1
    
    print("\n✅ 成功！现在可以使用以下命令进行预测:")
    print(f"   python predict.py --data_file {args.output} ...")
    
    return 0


if __name__ == '__main__':
    exit(main())