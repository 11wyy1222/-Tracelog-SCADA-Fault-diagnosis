"""
数据预处理核心模块
负责从原始CSV文件生成训练所需的.pt文件
"""

import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional
import logging


class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self, config: Dict):
        """
        初始化预处理器
        
        Args:
            config: 配置字典
        """
        self.config = config
        
        # 先提取配置参数
        self.paths = config['paths']
        self.features = config['features']
        self.processing = config['processing']
        self.normalization = config['normalization']
        self.split = config['split']
        self.output = config['output']
        
        # 再设置 logger（依赖 self.paths）
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger('DataPreprocessor')
        logger.setLevel(logging.INFO)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 文件处理器(如果配置要求)
        if self.config.get('logging', {}).get('save_log', False):
            log_file = os.path.join(
                self.paths['output_dir'],
                self.config['logging']['log_file']
            )
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def process(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        执行完整的预处理流程
        
        Returns:
            处理后的数据和标签
        """
        self.logger.info("=" * 60)
        self.logger.info("开始数据预处理...")
        self.logger.info("=" * 60)
        
        # 1. 加载配置和标签
        synonym_groups, tag_names = self._load_synonym_groups()
        labels_df = self._load_labels()
        
        # 缓存所有同义词名称（供 _robust_read_file 前缀修复使用）
        self._all_synonym_names = []
        for group in synonym_groups:
            self._all_synonym_names.extend(group)
        self._all_synonym_names.sort(key=len, reverse=True)
        
        # 2. 处理所有文件
        all_samples, all_labels = self._process_all_files(
            labels_df, synonym_groups, tag_names
        )
        
        if len(all_samples) == 0:
            raise ValueError("没有成功处理任何文件!")
        
        # 3. 转换为数组
        X = np.stack(all_samples, axis=0)
        y = np.array(all_labels)
        
        self.logger.info(f"\n最终数据形状:")
        self.logger.info(f"  X: {X.shape}")
        self.logger.info(f"  y: {y.shape}")
        
        return X, y
    
    def _load_synonym_groups(self) -> Tuple[List[List[str]], List[str]]:
        """加载同义词组"""
        self.logger.info("正在加载配置表...")
        
        config_path = self.paths['config_file']
        if config_path.endswith('.xlsx') or config_path.endswith('.xls'):
            config_df = pd.read_excel(config_path)
        else:
            config_df = pd.read_csv(config_path)
        target_tags = self.features['target_tags']
        
        synonym_groups = []
        tag_names = []
        
        for tag in target_tags:
            row = config_df[config_df['tags'] == tag]
            
            if len(row) == 0:
                self.logger.warning(f"配置表中未找到标签 '{tag}'")
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
        
        self.logger.info(f"成功加载 {len(synonym_groups)} 个特征组")
        return synonym_groups, tag_names
    
    def _load_labels(self) -> pd.DataFrame:
        """加载标签文件"""
        label_path = self.paths['label_file']
        if label_path.endswith('.xlsx') or label_path.endswith('.xls'):
            labels_df = pd.read_excel(label_path)
        else:
            labels_df = pd.read_csv(label_path)
        labels_df['fault_label'] = (
            labels_df['fault_type'].astype('category').cat.codes
        )
        return labels_df
    
    def _process_all_files(
        self,
        labels_df: pd.DataFrame,
        synonym_groups: List[List[str]],
        tag_names: List[str]
    ) -> Tuple[List[np.ndarray], List[int]]:
        """处理所有文件"""
        all_samples = []
        all_labels = []
        skipped_files = []
        
        self.logger.info("\n开始处理数据文件...")
        
        for idx, row in labels_df.iterrows():
            file_path = row['filename']
            
            if not os.path.exists(file_path):
                self.logger.warning(f"文件不存在: {file_path}")
                skipped_files.append(os.path.basename(file_path))
                continue
            
            self.logger.info(
                f"\n处理文件 [{idx+1}/{len(labels_df)}]: "
                f"{os.path.basename(file_path)}"
            )
            
            try:
                # 处理单个文件
                data = self._process_single_file(
                    file_path, synonym_groups, tag_names
                )
                
                if data is not None:
                    all_samples.append(data)
                    all_labels.append(row['fault_label'])
                    self.logger.info(f"  ✓ 处理成功，数据形状: {data.shape}")
                else:
                    skipped_files.append(os.path.basename(file_path))
                    
            except Exception as e:
                self.logger.error(f"  错误: {str(e)}")
                skipped_files.append(os.path.basename(file_path))
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info(f"数据处理完成!")
        self.logger.info(f"成功处理: {len(all_samples)} 个文件")
        self.logger.info(f"跳过文件: {len(skipped_files)} 个")
        if skipped_files:
            self.logger.info("跳过的文件列表:")
            for file in skipped_files:
                self.logger.info(f"  - {file}")
        
        return all_samples, all_labels
    
    def _robust_read_file(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        鲁棒的文件读取，支持CSV/TXT，自动检测分隔符和编码。
        处理：分号/逗号/tab分隔、多种编码、头部多余行、列名前缀污染。
        """
        file_name = os.path.basename(file_path)
        df = None

        for encoding in ['gbk', 'utf-8', 'utf-8-sig', 'latin-1']:
            try:
                # 用二进制读前4KB检测分隔符
                with open(file_path, 'rb') as f:
                    raw_head = f.read(4096).decode(encoding, errors='ignore')

                sep_counts = {';': raw_head.count(';'), ',': raw_head.count(','), '\t': raw_head.count('\t')}
                sep = max(sep_counts, key=sep_counts.get)
                if sep_counts[sep] == 0:
                    sep = ','

                # 优先c引擎，失败用python引擎
                try:
                    df = pd.read_csv(file_path, sep=sep, index_col=False, encoding=encoding, engine='c')
                except Exception:
                    df = pd.read_csv(file_path, sep=sep, index_col=False, encoding=encoding, engine='python')

                # 只读到1列，尝试跳过头部多余行
                if len(df.columns) <= 1:
                    for skip in range(1, 6):
                        try:
                            df_retry = pd.read_csv(file_path, sep=sep, index_col=False,
                                                   encoding=encoding, engine='c', skiprows=skip)
                        except Exception:
                            try:
                                df_retry = pd.read_csv(file_path, sep=sep, index_col=False,
                                                       encoding=encoding, engine='python', skiprows=skip)
                            except Exception:
                                continue
                        if len(df_retry.columns) > 1:
                            self.logger.info(f"  [{file_name}] 跳过前 {skip} 行后成功读取")
                            df = df_retry
                            break

                if len(df.columns) > 1:
                    break
            except Exception:
                continue

        if df is None or len(df.columns) <= 1:
            self.logger.warning(f"  [{file_name}] 无法解析文件，跳过")
            return None

        # 修复第一列名前缀污染（如 "13D05.20_NS01_008_P01_20240921grTime"）
        first_col = df.columns[0]
        if hasattr(self, '_all_synonym_names') and self._all_synonym_names:
            for candidate in self._all_synonym_names:
                prefix_len = len(first_col) - len(candidate)
                if prefix_len >= 10 and first_col.endswith(candidate):
                    self.logger.info(f"  [{file_name}] 修复列名前缀: '{first_col}' → '{candidate}'")
                    df.rename(columns={first_col: candidate}, inplace=True)
                    break

        # 强制数值转换
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    def _process_single_file(
        self,
        file_path: str,
        synonym_groups: List[List[str]],
        tag_names: List[str]
    ) -> Optional[np.ndarray]:
        """处理单个文件"""
        # 鲁棒读取（支持CSV/TXT，自动检测分隔符和编码）
        df = self._robust_read_file(file_path)
        if df is None:
            return None
        
        # 查找匹配的列，缺失通道用零填充保证维度一致
        n_features = len(synonym_groups)
        selected_data = []
        
        for i, synonym_group in enumerate(synonym_groups):
            matched_col = self._find_matching_column(
                df.columns, synonym_group
            )
            if matched_col:
                selected_data.append(df[matched_col].to_numpy())
                self.logger.info(
                    f"  特征 '{tag_names[i]}' 匹配到列: {matched_col}"
                )
            else:
                # 缺失通道用零填充
                selected_data.append(np.zeros(len(df)))
                self.logger.warning(
                    f"  特征 '{tag_names[i]}' 未找到匹配列，使用零填充"
                )
        
        # 组装为 DataFrame
        df_selected = pd.DataFrame(
            np.column_stack(selected_data),
            columns=tag_names
        )
        
        # 处理缺失值
        total_missing = df_selected.isna().sum().sum()
        if total_missing > 0:
            self.logger.info(f"  检测到 {total_missing} 个空值，开始处理...")
            df_selected = self._handle_missing_values(df_selected)
            self.logger.info("  ✓ 空值处理完成")
        
        # 处理无穷值
        inf_count = np.isinf(df_selected.values).sum()
        if inf_count > 0:
            self.logger.info(f"  检测到 {inf_count} 个无穷值，替换为0")
            df_selected.replace([np.inf, -np.inf], 0, inplace=True)
        
        # 转换为numpy数组
        data = df_selected.to_numpy()
        
        # 验证数据质量
        if np.isnan(data).any() or np.isinf(data).any():
            self.logger.error("  处理后仍有 NaN 或 Inf，跳过此文件")
            return None
        
        # 长度处理
        data = self._adjust_sequence_length(data)
        
        # 转置为 (n_features, seq_len)
        data = data.T
        
        return data
    
    def _find_matching_column(
        self,
        df_columns: pd.Index,
        synonym_group: List[str]
    ) -> Optional[str]:
        """从数据框的列名中找到与同义词组匹配的列"""
        for col in df_columns:
            if col in synonym_group:
                return col
        return None
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """处理缺失值"""
        df = data.copy()
        method = self.processing['missing_value_method']
        
        for col in df.columns:
            missing_count = df[col].isna().sum()
            
            if missing_count > 0:
                if method == 'interpolate':
                    df[col] = df[col].interpolate(
                        method='linear', limit_direction='both'
                    )
                    if df[col].isna().any():
                        df[col] = df[col].ffill().bfill()
                        
                elif method == 'ffill':
                    df[col] = df[col].ffill().bfill()
                    
                elif method == 'bfill':
                    df[col] = df[col].bfill().ffill()
                    
                elif method == 'mean':
                    df[col].fillna(df[col].mean(), inplace=True)
                    
                elif method == 'zero':
                    df[col].fillna(0, inplace=True)
        
        # 最后检查
        remaining_nan = df.isna().sum().sum()
        if remaining_nan > 0:
            self.logger.warning(
                f"  填充后仍有 {remaining_nan} 个空值，将用0填充"
            )
            df.fillna(0, inplace=True)
        
        return df
    
    def _adjust_sequence_length(self, data: np.ndarray) -> np.ndarray:
        """调整序列长度"""
        max_len = self.processing['max_seq_len']
        padding_mode = self.processing['padding_mode']
        
        if data.shape[0] < max_len:
            # 填充
            pad_width = max_len - data.shape[0]
            data = np.pad(
                data, ((0, pad_width), (0, 0)), mode=padding_mode
            )
            self.logger.info(f"  填充: {data.shape[0] - pad_width} -> {max_len}")
            
        elif data.shape[0] > max_len:
            # 截断
            original_len = data.shape[0]
            data = data[:max_len, :]
            self.logger.info(f"  截断: {original_len} -> {max_len}")
        
        return data
    
    def split_and_normalize(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """划分数据集并标准化"""
        self.logger.info("\n开始划分数据集...")
        
        # 划分训练/验证/测试集
        stratify_param = y if self.split['stratify'] else None
        
        X_train, X_tmp, y_train, y_tmp = train_test_split(
            X, y,
            test_size=self.split['test_size'],
            random_state=self.split['random_state'],
            stratify=stratify_param
        )
        
        stratify_tmp = y_tmp if self.split['stratify'] else None
        X_val, X_test, y_val, y_test = train_test_split(
            X_tmp, y_tmp,
            test_size=self.split['val_test_split'],
            random_state=self.split['random_state'],
            stratify=stratify_tmp
        )
        
        self.logger.info("数据集划分完成:")
        self.logger.info(
            f"  训练集: {X_train.shape}, "
            f"标签分布: {np.bincount(y_train)}"
        )
        self.logger.info(
            f"  验证集: {X_val.shape}, "
            f"标签分布: {np.bincount(y_val)}"
        )
        self.logger.info(
            f"  测试集: {X_test.shape}, "
            f"标签分布: {np.bincount(y_test)}"
        )
        
        # 数据标准化
        X_train, X_val, X_test, norm_stats = self._normalize_data(
            X_train, X_val, X_test
        )
        
        # 保存标准化统计量
        if self.normalization['save_stats']:
            self._save_normalization_stats(norm_stats)
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }
    
    def _normalize_data(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """标准化数据"""
        method = self.normalization['method']
        
        if method is None or method == 'none':
            self.logger.info("\n跳过标准化")
            return X_train, X_val, X_test, {}
        
        self.logger.info(f"\n开始数据标准化 (方法: {method})...")
        
        # 重塑为 (n_samples * seq_len, n_features)
        n_train, n_features, seq_len = X_train.shape
        n_val = X_val.shape[0]
        n_test = X_test.shape[0]
        
        X_train_reshaped = X_train.transpose(0, 2, 1).reshape(-1, n_features)
        X_val_reshaped = X_val.transpose(0, 2, 1).reshape(-1, n_features)
        X_test_reshaped = X_test.transpose(0, 2, 1).reshape(-1, n_features)
        
        stats = {}
        
        if method == 'zscore':
            mean = np.mean(X_train_reshaped, axis=0, keepdims=True)
            std = np.std(X_train_reshaped, axis=0, keepdims=True)
            std = np.where(std == 0, 1, std)
            
            X_train_norm = (X_train_reshaped - mean) / std
            X_val_norm = (X_val_reshaped - mean) / std
            X_test_norm = (X_test_reshaped - mean) / std
            
            stats = {'mean': mean, 'std': std}
            self.logger.info("Z-score标准化完成")
            
        elif method == 'minmax':
            min_val = np.min(X_train_reshaped, axis=0, keepdims=True)
            max_val = np.max(X_train_reshaped, axis=0, keepdims=True)
            range_val = max_val - min_val
            range_val = np.where(range_val == 0, 1, range_val)
            
            X_train_norm = (X_train_reshaped - min_val) / range_val
            X_val_norm = (X_val_reshaped - min_val) / range_val
            X_test_norm = (X_test_reshaped - min_val) / range_val
            
            stats = {'min': min_val, 'max': max_val}
            self.logger.info("Min-Max标准化完成")
            
        elif method == 'robust':
            median = np.median(X_train_reshaped, axis=0, keepdims=True)
            q1 = np.percentile(X_train_reshaped, 25, axis=0, keepdims=True)
            q3 = np.percentile(X_train_reshaped, 75, axis=0, keepdims=True)
            iqr = q3 - q1
            iqr = np.where(iqr == 0, 1, iqr)
            
            X_train_norm = (X_train_reshaped - median) / iqr
            X_val_norm = (X_val_reshaped - median) / iqr
            X_test_norm = (X_test_reshaped - median) / iqr
            
            stats = {'median': median, 'q1': q1, 'q3': q3, 'iqr': iqr}
            self.logger.info("鲁棒标准化完成")
        
        else:
            raise ValueError(f"未知的标准化方法: {method}")
        
        # 重塑回原始形状
        X_train_norm = X_train_norm.reshape(
            n_train, seq_len, n_features
        ).transpose(0, 2, 1)
        X_val_norm = X_val_norm.reshape(
            n_val, seq_len, n_features
        ).transpose(0, 2, 1)
        X_test_norm = X_test_norm.reshape(
            n_test, seq_len, n_features
        ).transpose(0, 2, 1)
        
        self.logger.info(f"\n标准化后的数据统计:")
        self.logger.info(
            f"  训练集范围: [{X_train_norm.min():.4f}, "
            f"{X_train_norm.max():.4f}]"
        )
        self.logger.info(f"  训练集均值: {X_train_norm.mean():.4f}")
        self.logger.info(f"  训练集标准差: {X_train_norm.std():.4f}")
        
        return X_train_norm, X_val_norm, X_test_norm, stats
    
    def _save_normalization_stats(self, stats: Dict):
        """保存标准化统计量"""
        if not stats:
            return
        
        output_dir = self.paths['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        stats_path = os.path.join(
            output_dir,
            self.normalization['stats_filename']
        )
        np.savez(stats_path, **stats)
        self.logger.info(f"\n标准化统计量已保存到: {stats_path}")
    
    def save_datasets(self, data_splits: Dict):
        """保存数据集"""
        output_dir = self.paths['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存训练集
        self._save_pt(
            data_splits['X_train'],
            data_splits['y_train'],
            output_dir,
            self.output['train_file']
        )
        
        # 保存验证集
        self._save_pt(
            data_splits['X_val'],
            data_splits['y_val'],
            output_dir,
            self.output['val_file']
        )
        
        # 保存测试集
        self._save_pt(
            data_splits['X_test'],
            data_splits['y_test'],
            output_dir,
            self.output['test_file']
        )
        
        self.logger.info(f"\n数据集保存完成!")
        self.logger.info(f"  训练集: {self.output['train_file']}")
        self.logger.info(f"  验证集: {self.output['val_file']}")
        self.logger.info(f"  测试集: {self.output['test_file']}")
        self.logger.info(f"\n文件保存路径: {output_dir}")
    
    def _save_pt(
        self,
        X: np.ndarray,
        y: np.ndarray,
        output_dir: str,
        filename: str
    ):
        """保存为 pt 文件"""
        dat_dict = {
            'samples': torch.from_numpy(X).float(),
            'labels': torch.from_numpy(y).long()
        }
        torch.save(dat_dict, os.path.join(output_dir, filename))