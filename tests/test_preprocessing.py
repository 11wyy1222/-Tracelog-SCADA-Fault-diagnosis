"""
数据预处理模块单元测试

验证需求:
- 2.1: 有效CSV数据的特征提取，返回status为"success"且特征维度为329或569
- 2.4: 批量处理后保存为包含samples和labels张量的.pt文件
- 2.5: 无效CSV数据的错误处理（跳过并继续）
- 2.6: 标签映射缺失时的跳过逻辑
- 2.7: 所有样本失败时抛出RuntimeError
"""
import os
# 解决 OpenMP 多次初始化导致的 abort 问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pytest
import numpy as np
import pandas as pd
import sys
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data_preprocessing.pipeline import FaultDiagnosisPipeline


# ---------------------------------------------------------------------------
# Helpers: 构造测试用 DataFrame
# ---------------------------------------------------------------------------
def _make_valid_sensor_df(n_rows=200, include_all_channels=True):
    """
    构造一个包含所有必要列的有效传感器 DataFrame。
    时间戳范围覆盖默认窗口 [-2000, 2000]。
    """
    np.random.seed(42)
    timestamps = np.linspace(-1500, 1500, n_rows)

    data = {"timestamp": timestamps}

    channels = [
        "rotor_speed",
        "generator_speed",
        "rotor_speed_relay1",
        "rotor_speed_relay2",
        "rotor_speed_counter1",
        "breaker_feedback1",
        "breaker_feedback2",
    ]

    if not include_all_channels:
        channels = channels[:4]  # 只保留前4个通道

    for ch in channels:
        # 生成带有一定变化的信号（非常量），避免特征提取时全零
        data[ch] = np.sin(np.linspace(0, 4 * np.pi, n_rows)) * 10 + np.random.randn(n_rows) * 0.5

    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Requirement 2.1 – 特征提取成功场景
# ---------------------------------------------------------------------------
class TestFeatureExtractionSuccess:
    """验证有效数据输入时，pipeline.run() 返回 status='success' 且特征维度正确。"""

    def test_run_returns_success_status(self):
        """有效 DataFrame 输入应返回 status='success'。"""
        pipeline = FaultDiagnosisPipeline()
        df = _make_valid_sensor_df()
        result = pipeline.run(df)
        assert result["status"] == "success"

    def test_features_dimension_is_329_or_569(self):
        """成功提取的特征向量维度应为 329 或 569。"""
        pipeline = FaultDiagnosisPipeline()
        df = _make_valid_sensor_df()
        result = pipeline.run(df)
        assert result["status"] == "success"
        features = result["features"]
        assert isinstance(features, np.ndarray)
        assert features.shape[0] in (329, 569), (
            f"特征维度应为 329 或 569，实际为 {features.shape[0]}"
        )

    def test_features_are_finite(self):
        """提取的特征值应全部为有限数（无 NaN/Inf）。"""
        pipeline = FaultDiagnosisPipeline()
        df = _make_valid_sensor_df()
        result = pipeline.run(df)
        assert result["status"] == "success"
        assert np.all(np.isfinite(result["features"]))

    def test_result_contains_expected_keys(self):
        """成功结果应包含 features、pattern、processed_data 等键。"""
        pipeline = FaultDiagnosisPipeline()
        df = _make_valid_sensor_df()
        result = pipeline.run(df)
        assert "features" in result
        assert "pattern" in result
        assert "processed_data" in result
        assert "feature_names" in result


# ---------------------------------------------------------------------------
# Requirement 2.5 – 无效CSV数据的错误处理
# ---------------------------------------------------------------------------
class TestInvalidDataHandling:
    """验证无效数据输入时，pipeline.run() 返回 status='failed' 并包含错误信息。"""

    def test_empty_dataframe_returns_failed(self):
        """空 DataFrame 应返回 status='failed'。"""
        pipeline = FaultDiagnosisPipeline()
        empty_df = pd.DataFrame()
        result = pipeline.run(empty_df)
        assert result["status"] == "failed"

    def test_missing_timestamp_column_returns_failed(self):
        """缺少 timestamp 列的 DataFrame 应返回 status='failed'。"""
        pipeline = FaultDiagnosisPipeline()
        df = pd.DataFrame({"rotor_speed": [1.0, 2.0, 3.0]})
        result = pipeline.run(df)
        assert result["status"] == "failed"
        assert result["error_code"] == "ERR_DAT_105"

    def test_insufficient_rows_after_window_returns_failed(self):
        """时间窗口截取后数据不足应返回 status='failed'。"""
        pipeline = FaultDiagnosisPipeline()
        # 时间戳全部在窗口外
        df = pd.DataFrame({
            "timestamp": [99999, 99999.1, 99999.2],
            "rotor_speed": [1.0, 2.0, 3.0],
        })
        result = pipeline.run(df)
        assert result["status"] == "failed"

    def test_none_input_returns_failed(self):
        """None 输入应返回 status='failed'。"""
        pipeline = FaultDiagnosisPipeline()
        result = pipeline.run(None)
        assert result["status"] == "failed"

    def test_failed_result_contains_error_code(self):
        """失败结果应包含 error_code 和 message 字段。"""
        pipeline = FaultDiagnosisPipeline()
        result = pipeline.run(pd.DataFrame())
        assert result["status"] == "failed"
        assert "error_code" in result
        assert "message" in result


# ---------------------------------------------------------------------------
# Requirement 2.6 – 标签映射缺失时的跳过逻辑
# ---------------------------------------------------------------------------
class TestLabelMappingSkip:
    """验证批量处理时，标签不在映射表中的样本被跳过。"""

    @patch("data_preprocessing.pipeline.FAULT_KNOWLEDGE_FILE", new="__mock__")
    def test_unknown_label_is_skipped(self, tmp_path):
        """标签不在映射表中的样本应被跳过，不影响其他样本处理。"""
        pipeline = FaultDiagnosisPipeline()

        valid_df = _make_valid_sensor_df()
        known_label = "已知标签"
        unknown_label = "未知标签_XYZ"

        # Mock Excel 读取，只包含 known_label 的映射
        mock_map_df = pd.DataFrame({"标签": [known_label], "数字": [0]})

        # Mock pipeline.run 使其总是成功
        fake_features = np.random.randn(569)
        mock_run_result = {
            "status": "success",
            "features": fake_features,
        }

        with patch("pandas.read_excel", return_value=mock_map_df), \
             patch.object(pipeline, "run", return_value=mock_run_result):
            result = pipeline.process_batch_to_pt(
                data_list=[valid_df, valid_df],
                label_list=[known_label, unknown_label],
                output_path=None,
            )

        # 只有 known_label 的样本被保留
        assert result["samples"].shape[0] == 1
        assert result["labels"].shape[0] == 1
        assert result["labels"][0].item() == 0


# ---------------------------------------------------------------------------
# Requirement 2.7 – 所有样本失败时抛出 RuntimeError
# ---------------------------------------------------------------------------
class TestAllSamplesFailRaisesError:
    """验证批量处理中所有样本均失败时抛出 RuntimeError。"""

    @patch("data_preprocessing.pipeline.FAULT_KNOWLEDGE_FILE", new="__mock__")
    def test_all_samples_fail_raises_runtime_error(self):
        """当所有样本的 run() 均返回 failed 时，process_batch_to_pt 应抛出 RuntimeError。"""
        pipeline = FaultDiagnosisPipeline()

        label = "测试标签"
        mock_map_df = pd.DataFrame({"标签": [label], "数字": [0]})

        fail_result = {
            "status": "failed",
            "error_code": "ERR_DAT_104",
            "message": "数据为空",
            "features": None,
        }

        with patch("pandas.read_excel", return_value=mock_map_df), \
             patch.object(pipeline, "run", return_value=fail_result):
            with pytest.raises(RuntimeError, match="没有样本被成功处理"):
                pipeline.process_batch_to_pt(
                    data_list=["fake_path_1.csv", "fake_path_2.csv"],
                    label_list=[label, label],
                    output_path=None,
                )

    @patch("data_preprocessing.pipeline.FAULT_KNOWLEDGE_FILE", new="__mock__")
    def test_all_labels_unknown_raises_runtime_error(self):
        """当所有标签都不在映射表中时，应抛出 RuntimeError。"""
        pipeline = FaultDiagnosisPipeline()

        mock_map_df = pd.DataFrame({"标签": ["已知标签"], "数字": [0]})

        with patch("pandas.read_excel", return_value=mock_map_df):
            with pytest.raises(RuntimeError, match="没有样本被成功处理"):
                pipeline.process_batch_to_pt(
                    data_list=["path1.csv", "path2.csv"],
                    label_list=["未知A", "未知B"],
                    output_path=None,
                )
