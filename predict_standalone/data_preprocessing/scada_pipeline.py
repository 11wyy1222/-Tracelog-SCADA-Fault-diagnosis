# -*- coding: utf-8 -*-
"""
SCADA数据处理管道模块

处理流程:
    原始文件 → [SCADA_Data_Preparer] → processed CSV → [特征提取] → .pt 文件

主要类:
    - SCADA_Pipeline: SCADA数据处理管道类
"""

import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Union, Callable
import warnings

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split

from .scada_errors import SCADA_ERROR_CODES, SCADA_Error
from .scada_data_preparer import SCADA_Data_Preparer
from data_preprocessing.feature_extractor import AdvancedFeatureExtractor, DEFAULT_FEATURE_CONFIG

warnings.filterwarnings("ignore")


class SCADA_Pipeline:
    """
    SCADA数据处理管道类

    两阶段处理：
      阶段1 (prepare) : 原始文件 → 预处理CSV落盘（时间转数值、筛选标签）
      阶段2 (extract) : 预处理CSV → 特征提取 → .pt 文件
    """

    @staticmethod
    def _deep_merge(default: dict, override: dict) -> dict:
        result = default.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = SCADA_Pipeline._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    @staticmethod
    def _resolve_placeholders(config: Any) -> Any:
        project_root = Path(__file__).resolve().parents[1]

        if isinstance(config, str):
            return config.replace('__PROJECT_ROOT__', str(project_root))
        if isinstance(config, list):
            return [SCADA_Pipeline._resolve_placeholders(item) for item in config]
        if isinstance(config, dict):
            return {key: SCADA_Pipeline._resolve_placeholders(value) for key, value in config.items()}
        return config

    def __init__(
        self,
        config_path: str = None,
        missing_channel_strategy: str = 'zero',
        log_enable: bool = True
    ):
        self.log_enable = log_enable
        self.missing_channel_strategy = missing_channel_strategy

        # 加载配置
        if not config_path or not os.path.exists(config_path):
            raise ValueError(f"配置文件不存在或未指定: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f) or {}
        self.config = self._resolve_placeholders(self.config)

        # 预处理器（missing_channel_strategy 统一在此处理）
        tw_cfg = self.config.get('time_window', {})
        self.preparer = SCADA_Data_Preparer(
            config=self.config,
            missing_channel_strategy=missing_channel_strategy,
            log_enable=log_enable
        )

        # 特征提取器
        yaml_config = {
            **self.config.get('feature_extraction', {}),
            **self.config.get('features', {})
        }
        merged_config = SCADA_Pipeline._deep_merge(DEFAULT_FEATURE_CONFIG, yaml_config)
        self.feature_extractor = AdvancedFeatureExtractor(config=merged_config)

        self.default_tags = self.config.get('tags', {}).get('default_tags', [])

        excel_defaults = {
            'scada_path_column': 'scada_path',
            'fault_time_column': 'fault_time',
            'label_column':      'label'
        }
        excel_defaults.update(self.config.get('excel', {}))
        self.excel_columns = excel_defaults

        # 加载标签映射（如果配置了知识库文件）
        self.label_map = {}
        knowledge_cfg = self.config.get('knowledge', {})
        mapping_file = knowledge_cfg.get('fault_knowledge_file')
        if mapping_file:
            mapping_path = self._expand_path(mapping_file)
            if os.path.exists(mapping_path):
                try:
                    df_map = pd.read_excel(mapping_path)
                    label_col = knowledge_cfg.get('label_column', 'label')
                    id_col = knowledge_cfg.get('id_column', 'id')
                    if label_col in df_map.columns and id_col in df_map.columns:
                        df_map = df_map.dropna(subset=[label_col, id_col])
                        self.label_map = dict(zip(
                            df_map[label_col].astype(str).str.strip(),   # ← 加 str.strip()
                            df_map[id_col].astype(int)
                        ))
                        if self.log_enable:
                            print(f"📖 已加载标签映射: {len(self.label_map)} 个类别")
                except Exception as e:
                    if self.log_enable:
                        print(f"⚠ 标签映射文件加载失败: {e}")

    def _expand_path(self, path_str: str) -> str:
        if '__PROJECT_ROOT__' in path_str:
            project_root = Path(__file__).resolve().parents[1]
            return path_str.replace('__PROJECT_ROOT__', str(project_root))
        return path_str

    # ------------------------------------------------------------------
    # 特征提取
    # ------------------------------------------------------------------

    def _extract_features(self, processed_df: pd.DataFrame) -> np.ndarray:
        """
        从预处理后的 DataFrame 提取特征

        参数:
            processed_df: real_time 已剔除，包含 timestamp + 各标签列（float64）

        返回:
            特征数组 (n_features,)
        """
        channels = self.default_tags or [
            col for col in processed_df.columns if col != 'timestamp'
        ]
        # timestamp 列已是相对时间，故障点=0
        self.feature_extractor.fault_time_target = 0.0

        feature_list = []
        for channel in channels:
            if channel not in processed_df.columns:
                # 缺失列已由 preparer 按策略处理（zero/nan/skip）
                # 走到这里说明策略是 skip，直接跳过
                if self.log_enable:
                    print(f"  ⚠ 通道 {channel} 不在数据中，跳过")
                continue
            try:
                features = self.feature_extractor.extract_single_channel_features(
                    processed_df, channel
                )
                feature_list.append(features)
            except Exception as e:
                if self.log_enable:
                    print(f"  ⚠ 通道 {channel} 特征提取失败: {e}")
                feature_list.append(np.zeros(80))

        return np.concatenate(feature_list) if feature_list else np.array([])

    def _extract_from_path(self, output_path: str) -> np.ndarray:
        """
        读取落盘文件并提取特征，供 process_single 和 process_batch 复用

        参数:
            output_path: 落盘CSV路径

        返回:
            特征数组 (n_features,)

        异常:
            ValueError: 特征提取结果为空
        """
        processed_df = self.preparer.load_processed(output_path)
        feature_df   = processed_df.drop(columns=['real_time'], errors='ignore')
        features     = self._extract_features(feature_df)
        if features.size == 0:
            raise ValueError("特征提取结果为空")
        return features

    # ------------------------------------------------------------------
    # 单样本处理
    # ------------------------------------------------------------------

    def process_single(
        self,
        scada_path: str,
        fault_timestamp: Union[str, datetime],
        tag_points: List[str] = None,
        time_window: Tuple[int, int] = None
    ) -> Dict[str, Any]:
        """
        处理单个SCADA样本

        内部走两阶段：prepare 落盘 → 读取落盘文件 → 提取特征

        返回:
            {
                'status'        : 'success' | 'failed',
                'features'      : np.ndarray,   # status=success 时有效
                'processed_path': str,
                'warning'       : str | None,
                'error_type'    : str,           # status=failed 时有效
                'message'       : str
            }
        """
        prep = self.preparer.prepare(
            scada_path=scada_path,
            fault_timestamp=fault_timestamp,
            tag_points=tag_points,
            time_window=time_window,
            overwrite=False
        )

        if prep['status'] == 'failed':
            return {
                'status':     'failed',
                'error_type': prep['error_type'],
                'message':    prep['message']
            }

        try:
            features = self._extract_from_path(prep['output_path'])
            return {
                'status':         'success',
                'features':       features,
                'processed_path': prep['output_path'],
                'warning':        prep.get('warning')
            }
        except Exception as e:
            return {
                'status':     'failed',
                'error_type': 'FEATURE_EXTRACTION_ERROR',
                'message':    f"特征提取失败: {e}\n{traceback.format_exc()}"
            }

    def process_batch(
        self,
        source_excel: str = None,
        progress_callback: Callable = None
    ) -> Dict[str, Any]:
        """
        批量处理SCADA样本并划分数据集

        两阶段：
          阶段1: 所有样本 prepare 落盘（可断点续跑，已存在自动跳过）
          阶段2: 从落盘文件提取特征，划分并保存 train/val/test.pt

        参数:
            source_excel     : 输入文件路径，None 则读配置文件
            progress_callback: 进度回调 callback(current, total, message)

        返回:
            处理统计信息字典
        """
        # ── 读取输入文件 ──
        if source_excel is None:
            source_excel = self.config.get('data', {}).get('source_excel')
            if source_excel is None:
                raise ValueError("未指定输入文件路径")

        if self.log_enable:
            print(f"📖 读取样本列表: {source_excel}")

        if source_excel.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(source_excel)
        elif source_excel.endswith('.csv'):
            df = pd.read_csv(source_excel)
        else:
            raise ValueError(f"不支持的输入文件格式: {source_excel}")

        scada_col = self.excel_columns['scada_path_column']
        time_col  = self.excel_columns['fault_time_column']
        label_col = self.excel_columns['label_column']

        for col in (scada_col, time_col, label_col):
            if col not in df.columns:
                raise ValueError(f"Excel中缺少列: {col}")

        total = len(df)
        if self.log_enable:
            print(f"🏁 共 {total} 个样本待处理\n")

        # ── 阶段1：预处理落盘 ──
        if self.log_enable:
            print("=" * 60)
            print("阶段1: 数据预处理（原始文件 → processed CSV）")
            print("=" * 60)

        prepare_success = []
        error_records   = []

        for idx, row in df.iterrows():
            scada_path = row[scada_col]
            fault_time = row[time_col]
            label      = row[label_col]

            if self.log_enable:
                print(f"[{idx+1}/{total}] {os.path.basename(str(scada_path))}")

            prep = self.preparer.prepare(
                scada_path=str(scada_path),
                fault_timestamp=fault_time,
                overwrite=False
            )

            # 附加元信息，供阶段2使用
            prep.update({
                'label':      label,
                'scada_path': scada_path,
                'fault_time': fault_time,
                'excel_row':  idx + 1
            })

            if prep['status'] == 'failed':
                error_records.append({
                    'excel_row':  idx + 1,
                    'scada_path': scada_path,
                    'fault_time': fault_time,
                    'label':      label,
                    'stage':      'prepare',
                    'error_type': prep.get('error_type', 'UNKNOWN'),
                    'message':    prep.get('message', '')
                })
                if self.log_enable:
                    print(f"  ❌ 预处理失败: {prep.get('error_type')} - "
                          f"{prep.get('message', '')[:80]}")
            else:
                prepare_success.append(prep)
                if prep.get('warning') and self.log_enable:
                    print(f"  ⚠ 警告: {prep['warning']}")

            if progress_callback:
                progress_callback(
                    idx + 1, total,
                    f"预处理: {os.path.basename(str(scada_path))}"
                )

        if self.log_enable:
            print(f"\n阶段1完成: 成功/跳过 {len(prepare_success)} | "
                  f"失败 {len(error_records)}\n")

        # ── 阶段2：特征提取 ──
        print(f"🔍 [EXTRACT] label_map = {self.label_map}")
        print(f"🔍 [EXTRACT] 第一个文件的标签原始值 = '{label}'")

        if self.log_enable:
            print("=" * 60)
            print("阶段2: 特征提取（processed CSV → 特征向量）")
            print("=" * 60)
        if self.log_enable and self.label_map:
            print(f"📖 当前标签映射表: {self.label_map}")


        feature_list = []
        label_list   = []

        for i, prep in enumerate(prepare_success):
            output_path = prep['output_path']
            label       = prep['label']

            if self.log_enable:
                print(f"[{i+1}/{len(prepare_success)}] "
                      f"提取特征: {os.path.basename(output_path)}")

            try:
                # ── 先做标签转换，转换失败则跳过，不污染 feature_list ──
                label_str = str(label).strip()
                if self.label_map:
                    if label_str not in self.label_map:
                        if self.log_enable:
                            print(f"  ⚠ 标签 '{label}' 不在映射表中，跳过")
                        error_records.append({
                            'excel_row':  prep['excel_row'],
                            'scada_path': prep['scada_path'],
                            'fault_time': prep['fault_time'],
                            'label':      label,
                            'stage':      'extract',
                            'error_type': 'LABEL_NOT_IN_MAP',
                            'message':    f"标签 '{label}' 不在映射表中"
                        })
                        continue
                    label_encoded = self.label_map[label_str]
                else:
                    try:
                        label_encoded = int(float(label_str))
                    except (ValueError, TypeError):
                        if self.log_enable:
                            print(f"  ⚠ 无法转换标签 '{label}' 为整数，跳过")
                        error_records.append({
                            'excel_row':  prep['excel_row'],
                            'scada_path': prep['scada_path'],
                            'fault_time': prep['fault_time'],
                            'label':      label,
                            'stage':      'extract',
                            'error_type': 'LABEL_CONVERT_ERROR',
                            'message':    f"无法将 '{label}' 转换为整数"
                        })
                        continue

                # ── 标签确认有效后，再提取特征 ──
                features = self._extract_from_path(output_path)
                feature_list.append(features)
                label_list.append(label_encoded)

                if self.log_enable:
                    print(f"  ✅ 成功 | 特征维度: ({features.shape[0]},) "
                          f"| 标签: {label_encoded}")

            except Exception as e:
                error_records.append({
                    'excel_row':  prep['excel_row'],
                    'scada_path': prep['scada_path'],
                    'fault_time': prep['fault_time'],
                    'label':      label,
                    'stage':      'extract',
                    'error_type': 'FEATURE_EXTRACTION_ERROR',
                    'message':    f"{e}\n{traceback.format_exc()}"
                })
                if self.log_enable:
                    print(f"  ❌ 特征提取失败: {e}")

            # ── progress_callback 在 try/except 外，每次循环必定执行 ──
            if progress_callback:
                progress_callback(
                    i + 1, len(prepare_success),
                    f"提取特征: {os.path.basename(output_path)}"
                )

        if self.log_enable:
            print(f"\n阶段2完成: 成功 {len(feature_list)} | "
                  f"失败 {len(error_records)}\n")

        self._save_error_log(error_records)

        if not feature_list:
            return {
                'status':  'failed',
                'total':   total,
                'success': 0,
                'failed':  len(error_records),
                'message': '所有样本处理失败，无法生成数据集'
            }

        # ── 构建样本矩阵 ──
        samples = np.stack(feature_list, axis=0)
        labels  = np.array(label_list, dtype=np.int64)

        if self.log_enable:
            print(f"📊 样本形状: {samples.shape}")
            unique, counts = np.unique(labels, return_counts=True)
            print(f"📊 标签分布: { {int(k): int(v) for k, v in zip(unique, counts)} }")

        # ── 数据集划分 ──
        split_cfg   = self.config.get('data_split', {})
        test_size   = split_cfg.get('test_size',   0.1)
        val_size    = split_cfg.get('val_size',     0.111)
        random_seed = split_cfg.get('random_seed',  42)
        stratify    = split_cfg.get('stratify',     True)

        if self.log_enable:
            print(f"\n📊 数据划分配置: test={test_size} val={val_size} "
                  f"seed={random_seed} stratify={stratify}")

        def _split(strat_labels):
            idx_tv, idx_test = train_test_split(
                np.arange(len(samples)),
                test_size=test_size,
                random_state=random_seed,
                stratify=strat_labels
            )
            idx_train, idx_val = train_test_split(
                idx_tv,
                test_size=val_size,
                random_state=random_seed,
                stratify=labels[idx_tv] if strat_labels is not None else None
            )
            return idx_train, idx_val, idx_test

        try:
            idx_train, idx_val, idx_test = _split(labels if stratify else None)
        except ValueError as e:
            if self.log_enable:
                print(f"  ⚠ 分层采样失败({e})，改用随机划分")
            idx_train, idx_val, idx_test = _split(None)

        splits = {
            'train': (samples[idx_train], labels[idx_train]),
            'val':   (samples[idx_val],   labels[idx_val]),
            'test':  (samples[idx_test],  labels[idx_test])
        }

        if self.log_enable:
            for name, (s, _) in splits.items():
                print(f"  - {name}: {len(s)}")

        # ── 保存 .pt 文件 ──
        output_cfg    = self.config.get('output', {})
        feature_dtype = getattr(torch, output_cfg.get('feature_dtype', 'float32'), torch.float32)
        label_dtype   = getattr(torch, output_cfg.get('label_dtype',   'int64'),   torch.int64)

        full_file = self._expand_path(output_cfg.get('full_file', './data/full_dataset.pt'))

        file_paths = {
            'train': self._expand_path(output_cfg.get('train_file', './data/train.pt')),
            'val':   self._expand_path(output_cfg.get('val_file',   './data/val.pt')),
            'test':  self._expand_path(output_cfg.get('test_file',  './data/test.pt'))
        }

        # 1) 保存完整数据集
        full_dir = os.path.dirname(full_file)
        if full_dir:
            os.makedirs(full_dir, exist_ok=True)

        full_data = {
            'samples': torch.tensor(samples, dtype=feature_dtype),
            'labels':  torch.tensor(labels, dtype=label_dtype)
        }
        torch.save(full_data, full_file)

        if self.log_enable:
            print(f"  💾 已保存完整数据集: {full_file}")
            print(f"     samples shape: {full_data['samples'].shape}")
            print(f"     labels shape : {full_data['labels'].shape}")

        # 2) 保存 train / val / test
        for name, path in file_paths.items():
            s, l = splits[name]

            out_dir = os.path.dirname(path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)

            torch.save(
                {
                    'samples': torch.tensor(s, dtype=feature_dtype),
                    'labels':  torch.tensor(l, dtype=label_dtype)
                },
                path
            )
            if self.log_enable:
                print(f"  💾 已保存{name}集: {path}")

        return {
            'status':     'success',
            'total':      total,
            'success':    len(feature_list),
            'failed':     len(error_records),

            'full_size':  len(samples),
            'full_file':  full_file,

            'train_size': len(splits['train'][0]),
            'val_size':   len(splits['val'][0]),
            'test_size':  len(splits['test'][0]),
            'train_file': file_paths['train'],
            'val_file':   file_paths['val'],
            'test_file':  file_paths['test']
        }



    # ------------------------------------------------------------------
    # 错误日志
    # ------------------------------------------------------------------

    def _save_error_log(self, error_records: List[Dict]) -> None:
        """保存错误日志（CSV + TXT 两份）"""
        if not error_records:
            return

        log_cfg         = self.config.get('logging', {})
        ts              = datetime.now().strftime(
                              log_cfg.get('log_timestamp_format', '%Y%m%d_%H%M%S'))
        error_file_base = os.path.splitext(
                              log_cfg.get('error_file', 'SCADA数据文件读取异常记录'))[0]
        sep             = (log_cfg.get('debug_separator', '=')
                           * log_cfg.get('debug_separator_length', 60))

        csv_path = f"{error_file_base}_{ts}.csv"
        pd.DataFrame(error_records).to_csv(csv_path, index=False, encoding='utf-8-sig')

        txt_path = f"{error_file_base}_{ts}.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"SCADA处理错误日志 - {ts}\n共 {len(error_records)} 条错误\n{sep}\n")
            for rec in error_records:
                f.write(
                    f"行号     : {rec['excel_row']}\n"
                    f"阶段     : {rec['stage']}\n"
                    f"文件     : {rec['scada_path']}\n"
                    f"故障时刻 : {rec['fault_time']}\n"
                    f"标签     : {rec['label']}\n"
                    f"错误类型 : {rec['error_type']}\n"
                    f"错误信息 : {rec['message']}\n"
                    f"{sep}\n"
                )

        if self.log_enable:
            print(f"\n💾 错误日志(CSV): {csv_path}")
            print(f"💾 错误日志(TXT): {txt_path}")
