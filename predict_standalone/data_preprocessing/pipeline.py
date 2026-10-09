import numpy as np
import pandas as pd
import logging
import traceback
import torch
from typing import List, Dict, Union, Any, Optional

from data_preprocessing.feature_extractor import AdvancedFeatureExtractor
from data_preprocessing.data_processor import DataProcessor
from error_codes import ErrorCode

from configs.config_loader import (
    DEFAULT_FAULT_TIME,
    TIMESTAMP_WINDOW,
    MAIN_SENSOR_CHANNELS,
    MULTI_ANALYSIS_CHANNELS,
    KEY_ROTOR_CHANNEL,
    FAULT_KNOWLEDGE_FILE,
    KNOWLEDGE_LABEL_COLUMN,
    KNOWLEDGE_ID_COLUMN
)

logger = logging.getLogger(__name__)


class FaultDiagnosisPipeline:
    def __init__(self, missing_channel_strategy='zero', add_missing_indicator=False,
                 main_sensor_channels=None, fault_knowledge_file=None,
                 timestamp_window=None, default_fault_time=None,
                 multi_analysis_channels=None):
        self.feature_extractor = AdvancedFeatureExtractor()
        self.data_processor = DataProcessor()
        self.main_sensor_channels = main_sensor_channels or MAIN_SENSOR_CHANNELS
        self.multi_analysis_channels = multi_analysis_channels if multi_analysis_channels is not None else MULTI_ANALYSIS_CHANNELS
        self.missing_channel_strategy = missing_channel_strategy
        self.add_missing_indicator = add_missing_indicator
        self.missing_flags = []
        self.fault_knowledge_file = fault_knowledge_file or FAULT_KNOWLEDGE_FILE
        self.timestamp_window = tuple(timestamp_window) if timestamp_window else TIMESTAMP_WINDOW
        self.default_fault_time = default_fault_time if default_fault_time is not None else DEFAULT_FAULT_TIME

        logger.info(f"[INIT] FaultDiagnosisPipeline 初始化完成 (通道数: {len(self.main_sensor_channels)}, 缺失策略: {missing_channel_strategy})")

    def _fail_response(self, error_code: ErrorCode, message: str, sample_id: str = None) -> Dict[str, Any]:
        err_msg = f"[{error_code.name}({error_code.value})] {message}"
        if sample_id:
            err_msg = f"Sample: {sample_id} | {err_msg}"
        logger.error(err_msg)
        return {
            "status": "failed",
            "error_code": error_code,
            "message": message,
            "features": None
        }

    # =========================================================================
    # 情况 1: 单样本处理
    # =========================================================================
    def run(self, csv_data_or_path: Union[str, pd.DataFrame],
            fault_time=None,
            timestamp_range=None) -> Dict[str, Any]:

        fault_time = fault_time if fault_time is not None else self.default_fault_time
        timestamp_range = timestamp_range or self.timestamp_window

        sample_id = csv_data_or_path if isinstance(csv_data_or_path, str) else "DataFrame_Input"
        sensor_data = None

        # --- 阶段 1: 数据加载与校验 ---
        try:
            if isinstance(csv_data_or_path, str):
                try:
                    sensor_data, err_code, err_msg = self.data_processor.extract_columns_with_fallback(
                        str(csv_data_or_path),
                        fault_time=fault_time,
                        timestamp_range=timestamp_range
                    )
                    if err_code:
                        return self._fail_response(err_code, err_msg, sample_id)
                except Exception as e:
                    return self._fail_response(ErrorCode.CSV_PARSE_ERROR, f"无法解析CSV编码或文件格式错误: {sample_id}", sample_id)
            else:
                sensor_data = csv_data_or_path

            if sensor_data is None or (isinstance(sensor_data, pd.DataFrame) and sensor_data.empty):
                return self._fail_response(ErrorCode.DATA_IS_EMPTY, "数据为空", sample_id)

            if 'timestamp' not in sensor_data.columns:
                return self._fail_response(ErrorCode.TIMESTAMP_MISSING, "缺失 timestamp 关键列", sample_id)

            try:
                mask = (sensor_data['timestamp'] >= timestamp_range[0]) & \
                       (sensor_data['timestamp'] <= timestamp_range[1])
                sensor_data = sensor_data[mask].copy()
            except Exception as e:
                return self._fail_response(ErrorCode.DATA_PROCESS_CRASH, f"内部处理崩溃(时间截取): {str(e)}", sample_id)

            if len(sensor_data) < 5:
                return self._fail_response(ErrorCode.TIME_WINDOW_EMPTY,
                                           f"时间窗口截取后无有效数据: {timestamp_range} (Rows={len(sensor_data)})",
                                           sample_id)

        except Exception as e:
            return self._fail_response(ErrorCode.DATA_PROCESS_CRASH, f"内部处理崩溃: {str(e)}", sample_id)

        # --- 阶段 2: 特征提取 ---
        try:
            features_raw = self._extract_enhanced_features(sensor_data)

        except Exception as e:
            error_msg = str(e)

            if "[STAGE:FEAT]" in error_msg:
                return self._fail_response(ErrorCode.FEAT_ABORTED, f"特征提取被中止: {error_msg}", sample_id)
            else:
                logger.error(traceback.format_exc())
                return self._fail_response(ErrorCode.FEAT_UNCAUGHT_ERROR, f"计算特征时发生未捕获异常: {error_msg}", sample_id)

        # --- 阶段 3 & 4: 清洗与分析 ---
        features_clean = self.data_processor.clean_features(features_raw)
        pattern_analysis = self._analyze_fault_pattern(features_clean, sensor_data)
        feature_names = self._generate_feature_names()

        return {
            "status": "success",
            "features": features_clean,
            "pattern": pattern_analysis,
            "processed_data": sensor_data,
            "feature_names": feature_names,
            "channels": self.main_sensor_channels
        }

    # =========================================================================
    # 情况 2: 批量处理
    # =========================================================================
    def process_batch_to_pt(self, data_list: List[str], label_list: List[Any],
                            output_path: Optional[str] = None,
                            fault_time=None,
                            timestamp_range=None,
                            progress_callback=None) -> Dict[str, Any]:

        fault_time = fault_time if fault_time is not None else self.default_fault_time
        timestamp_range = timestamp_range or self.timestamp_window
        cb = progress_callback or (lambda cur, tot, msg: None)

        if len(data_list) != len(label_list):
            raise ValueError(f"数据列表长度 ({len(data_list)}) 与标签列表长度 ({len(label_list)}) 不一致")

        # --- 步骤 A: 加载外部标签映射表 ---
        label_map = {}
        try:
            mapping_path = str(self.fault_knowledge_file)
            logger.info(f"📖 正在加载标签映射文件: {mapping_path}")

            df_map = pd.read_excel(mapping_path)

            label_col = KNOWLEDGE_LABEL_COLUMN
            id_col = KNOWLEDGE_ID_COLUMN
            if label_col not in df_map.columns or id_col not in df_map.columns:
                raise ValueError(f"映射文件缺少必要列: 必须包含 '{label_col}' 和 '{id_col}'。当前列: {df_map.columns.tolist()}")

            df_map = df_map.dropna(subset=[label_col, id_col])
            label_map = dict(zip(df_map[label_col].astype(str), df_map[id_col].astype(int)))

            logger.info(f"✅ 映射表加载成功，共 {len(label_map)} 个类别。")

        except Exception as e:
            logger.critical(f"❌ 无法加载标签映射文件: {e}")
            raise RuntimeError("必须提供有效的 class_mapping.xlsx 才能进行训练数据封装")

        # --- 步骤 B: 开始批量处理数据 ---
        valid_features = []
        valid_labels_encoded = []
        total_samples = len(data_list)
        log_interval = max(1, min(50, total_samples // 10))

        logger.info(f"🏁 [START] 开始批量处理 {total_samples} 个样本...")

        for idx, (data_input, raw_label) in enumerate(zip(data_list, label_list)):
            if (idx + 1) % log_interval == 0 or (idx + 1) == total_samples:
                progress_msg = f"正在处理: {idx + 1}/{total_samples} ({(idx + 1) / total_samples:.1%})"
                logger.info(f"⏳ [PROGRESS] {progress_msg}")
                cb(idx + 1, total_samples, progress_msg)

            raw_label_str = str(raw_label).strip()
            if raw_label_str not in label_map:
                logger.error(f"❌ [{ErrorCode.LABEL_NOT_IN_MAP.name}({ErrorCode.LABEL_NOT_IN_MAP.value})] 样本 {idx} 的标签 '{raw_label}' 未在映射表中定义，跳过。")
                continue

            target_id = label_map[raw_label_str]

            result = self.run(data_input, fault_time=fault_time, timestamp_range=timestamp_range)

            if result['status'] == 'success':
                valid_features.append(result['features'])
                valid_labels_encoded.append(target_id)
            else:
                pass

        if not valid_features:
            raise RuntimeError(f"❌ [{ErrorCode.NO_VALID_SAMPLES.name}({ErrorCode.NO_VALID_SAMPLES.value})] 没有样本被成功处理，无法生成 PT 文件")

        # --- 步骤 C: 封装为 Tensor ---
        features_np = np.array(valid_features, dtype=np.float32)
        labels_np = np.array(valid_labels_encoded, dtype=np.int64)
        unique_vals, counts = np.unique(labels_np, return_counts=True)
        dist_info = dict(zip(unique_vals, counts))
        logger.info(f"📊 最终生成的标签分布: {dist_info}")

        if len(unique_vals) < 2:
            logger.critical("🚨 [CRITICAL] 最终数据集中只有一个类别！请检查输入 Excel 的标签列和映射表。")

        samples_tensor = torch.from_numpy(features_np)
        labels_tensor = torch.from_numpy(labels_np)

        n_channels = len(self.main_sensor_channels)
        single_channel_dim = 80
        multi_channel_dim = 9
        expected_dim = n_channels * single_channel_dim + multi_channel_dim
        if samples_tensor.shape[1] != expected_dim:
            logger.warning(f"⚠️ [WARN] 特征维度为 {samples_tensor.shape[1]} (预期 {expected_dim}: {n_channels}通道×{single_channel_dim} + {multi_channel_dim})")

        # --- 步骤 D: 结果封装 ---
        output_dict = {
            'samples': samples_tensor,
            'labels': labels_tensor
        }

        if output_path:
            try:
                torch.save(output_dict, output_path)
                logger.info(f"💾 文件已保存至: {output_path}")
            except Exception as e:
                logger.error(f"❌ 保存失败: {e}")

        return output_dict

    # =========================================================================
    # 特征提取逻辑
    # =========================================================================
    def _extract_enhanced_features(self, sensor_data: pd.DataFrame):
        all_features = []
        self.missing_flags = []

        # ==========================
        # 1. 单通道特征提取
        # ==========================
        for channel in self.main_sensor_channels:
            channel_missing = False
            try:
                if channel not in sensor_data.columns:
                    channel_missing = True
                    if self.missing_channel_strategy == 'skip':
                        msg = f"[STAGE:FEAT] [{ErrorCode.CHANNEL_MISSING.name}({ErrorCode.CHANNEL_MISSING.value})] 通道缺失: {channel}"
                        logger.error(msg)
                        raise ValueError(msg)
                    elif self.missing_channel_strategy == 'zero':
                        msg = f"[STAGE:FEAT] [WARN] 通道缺失: {channel}，使用零向量填充特征"
                        logger.warning(msg)
                        ch_feats = np.zeros(80)
                        all_features.extend(ch_feats)
                        self.missing_flags.append(1)
                        continue
                    elif self.missing_channel_strategy == 'nan':
                        msg = f"[STAGE:FEAT] [WARN] 通道缺失: {channel}，使用NaN填充特征（后续由scaler处理）"
                        logger.warning(msg)
                        ch_feats = np.full(80, np.nan)
                        all_features.extend(ch_feats)
                        self.missing_flags.append(1)
                        continue

                if sensor_data[channel].isnull().all() or len(sensor_data[channel]) == 0:
                    channel_missing = True
                    if self.missing_channel_strategy == 'skip':
                        msg = f"[STAGE:FEAT] [{ErrorCode.CHANNEL_ALL_NAN.name}({ErrorCode.CHANNEL_ALL_NAN.value})] 通道 {channel} 数据全空或全为 NaN"
                        logger.error(msg)
                        raise ValueError(msg)
                    elif self.missing_channel_strategy == 'zero':
                        msg = f"[STAGE:FEAT] [WARN] 通道 {channel} 数据全空，使用零向量填充特征"
                        logger.warning(msg)
                        ch_feats = np.zeros(80)
                        all_features.extend(ch_feats)
                        self.missing_flags.append(1)
                        continue
                    elif self.missing_channel_strategy == 'nan':
                        msg = f"[STAGE:FEAT] [WARN] 通道 {channel} 数据全空，使用NaN填充特征"
                        logger.warning(msg)
                        ch_feats = np.full(80, np.nan)
                        all_features.extend(ch_feats)
                        self.missing_flags.append(1)
                        continue

                ch_feats = self.feature_extractor.extract_single_channel_features(sensor_data, channel)
                all_features.extend(ch_feats)
                self.missing_flags.append(0)

            except Exception as e:
                if "[STAGE:FEAT]" in str(e):
                    raise e

                msg = f"[STAGE:FEAT] [{ErrorCode.CHANNEL_EXTRACT_CRASH.name}({ErrorCode.CHANNEL_EXTRACT_CRASH.value})] 通道 {channel} 特征提取崩溃: {str(e)}"
                logger.error(msg)
                raise ValueError(msg)

        # ==========================
        # 2. 多通道特征提取
        # ==========================
        try:
            if self.multi_analysis_channels:
                multi_feats = self.feature_extractor.extract_multi_channel_features(
                    sensor_data, self.multi_analysis_channels
                )
                if multi_feats is not None:
                    all_features.extend(multi_feats)
                else:
                    all_features.extend(np.zeros(9))
            else:
                all_features.extend(np.zeros(9))

        except Exception as e:
            logger.warning(f"[STAGE:FEAT] [WARN] 多通道特征提取异常: {str(e)}，使用零向量填充")
            all_features.extend(np.zeros(9))

        # ==========================
        # 3. 添加缺失指示器特征（可选）
        # ==========================
        if self.add_missing_indicator:
            all_features.extend(self.missing_flags)
            logger.info(f"[STAGE:FEAT] 添加了 {len(self.missing_flags)} 个缺失指示器特征")

        return np.array(all_features, dtype=np.float64)

    def _analyze_fault_pattern(self, features, sensor_data):
        analysis_result = {
            'likely_pattern': 'unknown',
            'confidence': 0.0,
            'key_indicators': {}
        }

        try:
            business_features = self.feature_extractor.extract_business_rule_features(
                sensor_data, self.main_sensor_channels
            )

            if len(business_features) >= 5:
                avg_rotor_speed = business_features[0]
                speed_above_1 = business_features[1]
                regularity_score = business_features[2]
                drop_score = business_features[3]

                if KEY_ROTOR_CHANNEL in sensor_data.columns:
                    p_feats = self.feature_extractor.extract_periodicity_features(sensor_data, KEY_ROTOR_CHANNEL)
                    if len(p_feats) >= 8:
                        peaks_ge_3 = p_feats[6]
                        peaks_le_2 = p_feats[7]

                        if regularity_score > 0.5 and peaks_ge_3 > 0:
                            analysis_result.update({
                                'likely_pattern': '规律波动',
                                'confidence': float(min(regularity_score, 1.0)),
                                'key_indicators': {'规律性得分': f"{regularity_score:.3f}"}
                            })
                        elif drop_score > 0.5 and speed_above_1 > 0:
                            analysis_result.update({
                                'likely_pattern': '跌落',
                                'confidence': float(min(drop_score, 1.0)),
                                'key_indicators': {'跌落得分': f"{drop_score:.3f}"}
                            })

        except Exception as e:
            logger.warning(f"[{ErrorCode.BIZ_ANALYSIS_WARN.name}({ErrorCode.BIZ_ANALYSIS_WARN.value})] 故障模式分析非关键性出错: {e}")

        return analysis_result

    def _generate_feature_names(self):
        names = []
        for channel in self.main_sensor_channels:
            names.extend([f"{channel}_basic_{i:02d}" for i in range(20)])
            names.extend([f"{channel}_dynamic_{i:02d}" for i in range(25)])
            names.extend([f"{channel}_freq_{i:02d}" for i in range(12)])
            names.extend([f"{channel}_stability_{i:02d}" for i in range(15)])
            names.extend([f"{channel}_periodicity_{i:02d}" for i in range(8)])

        names.extend([f"multi_sensor_sync_{i:02d}" for i in range(3)])
        names.extend([f"business_rule_{i:02d}" for i in range(6)])
        return names
