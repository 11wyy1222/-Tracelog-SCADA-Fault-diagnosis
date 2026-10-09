"""
标签编码和故障诊断映射单元测试

验证需求:
- 8.1: 标签编码映射是双射函数，不同原始标签映射到不同数字编码
- 8.2: 故障诊断映射器能将数字标签正确还原为故障原因描述
"""
import pytest
import os
import sys
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.fault_diagnosis_mapper import FaultDiagnosisMapper


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_mapping_excel(tmp_path):
    """创建一个包含测试映射数据的临时Excel文件。"""
    data = {
        '标签': [0, 1, 2, 3],
        '故障原因': [
            '滑环编码器转速跳变',
            '滑环编码器转速规律波动',
            '超速继电器转速异常',
            '转速未超限',
        ],
        '处理方案': [
            '检查梅花联轴器有无松动或损坏',
            '检查滑环编码器屏蔽线有无异常',
            '检查超速继电器1有无掉电或损坏',
            '检查安全链回路接线有无松动',
        ],
    }
    filepath = tmp_path / "test_mapping.xlsx"
    pd.DataFrame(data).to_excel(filepath, index=False, engine='openpyxl')
    return str(filepath)


@pytest.fixture
def mapper(sample_mapping_excel):
    """返回一个已加载测试映射数据的 FaultDiagnosisMapper 实例。"""
    return FaultDiagnosisMapper(sample_mapping_excel)


@pytest.fixture
def label_encoding_excel(tmp_path):
    """创建一个模拟 class_mapping1.xlsx 格式的标签编码Excel文件（含 '标签' 和 '数字' 列）。"""
    data = {
        '故障原因': ['故障A', '故障B', '故障C', '故障D', '故障E'],
        '标签': ['A1', 'A2', 'B1', 'B2', 'C1'],
        '处理方案': ['方案A', '方案B', '方案C', '方案D', '方案E'],
        '数字': [0, 1, 2, 3, 4],
    }
    filepath = tmp_path / "test_label_encoding.xlsx"
    pd.DataFrame(data).to_excel(filepath, index=False, engine='openpyxl')
    return str(filepath)


# ---------------------------------------------------------------------------
# Requirement 8.1 – 标签编码唯一性（双射性）
# ---------------------------------------------------------------------------
class TestLabelEncodingUniqueness:
    """验证标签编码映射的唯一性：不同原始标签映射到不同数字编码。"""

    def test_label_encoding_is_injective(self, label_encoding_excel):
        """不同的原始标签（字符串）必须映射到不同的数字编码。"""
        df = pd.read_excel(label_encoding_excel)
        label_map = dict(zip(df['标签'].astype(str), df['数字'].astype(int)))

        # 原始标签数量应等于编码后数字的去重数量
        assert len(label_map) == len(set(label_map.values())), \
            "标签编码不是单射：存在不同标签映射到相同数字编码"

    def test_real_class_mapping_is_injective(self):
        """验证项目实际的 class_mapping.xlsx 标签编码唯一性。"""
        mapping_path = PROJECT_ROOT / "configs" / "class_mapping.xlsx"
        if not mapping_path.exists():
            pytest.skip("class_mapping.xlsx 不存在")

        df = pd.read_excel(mapping_path)
        labels = df['标签'].tolist()
        assert len(labels) == len(set(labels)), \
            "class_mapping.xlsx 中存在重复标签"

    def test_real_class_mapping1_is_injective(self):
        """验证项目实际的 class_mapping1.xlsx 标签→数字编码唯一性。"""
        mapping_path = PROJECT_ROOT / "configs" / "class_mapping1.xlsx"
        if not mapping_path.exists():
            pytest.skip("class_mapping1.xlsx 不存在")

        df = pd.read_excel(mapping_path)
        label_map = dict(zip(df['标签'].astype(str), df['数字'].astype(int)))

        assert len(label_map) == len(set(label_map.values())), \
            "class_mapping1.xlsx 中存在不同标签映射到相同数字编码"

    def test_mapper_labels_are_unique(self, mapper):
        """FaultDiagnosisMapper 内部 mapping_dict 的键（标签）应唯一。"""
        all_labels = mapper.get_all_labels()
        assert len(all_labels) == len(set(all_labels))


# ---------------------------------------------------------------------------
# Requirement 8.2 – 故障诊断映射器：标签→故障原因还原
# ---------------------------------------------------------------------------
class TestFaultDiagnosisMapper:
    """验证 FaultDiagnosisMapper 能将数字标签正确还原为故障原因描述。"""

    def test_get_diagnosis_returns_correct_info(self, mapper):
        """get_diagnosis 应返回包含 '故障原因' 和 '处理方案' 的字典。"""
        diag = mapper.get_diagnosis(0)
        assert diag is not None
        assert '故障原因' in diag
        assert '处理方案' in diag
        assert diag['故障原因'] == '滑环编码器转速跳变'

    def test_get_diagnosis_all_labels(self, mapper):
        """所有已定义标签都应能通过 get_diagnosis 还原。"""
        expected = {
            0: '滑环编码器转速跳变',
            1: '滑环编码器转速规律波动',
            2: '超速继电器转速异常',
            3: '转速未超限',
        }
        for label, expected_reason in expected.items():
            diag = mapper.get_diagnosis(label)
            assert diag is not None, f"标签 {label} 未找到诊断信息"
            assert diag['故障原因'] == expected_reason

    def test_get_diagnosis_returns_none_for_invalid_label(self, mapper):
        """对于未定义的标签，get_diagnosis 应返回 None。"""
        assert mapper.get_diagnosis(999) is None
        assert mapper.get_diagnosis(-1) is None

    def test_has_mapping_returns_true(self, mapper):
        """加载有效映射文件后，has_mapping 应返回 True。"""
        assert mapper.has_mapping() is True

    def test_has_mapping_returns_false_for_missing_file(self, tmp_path):
        """映射文件不存在时，has_mapping 应返回 False。"""
        m = FaultDiagnosisMapper(str(tmp_path / "nonexistent.xlsx"))
        assert m.has_mapping() is False

    def test_get_all_labels(self, mapper):
        """get_all_labels 应返回所有已定义标签的有序列表。"""
        labels = mapper.get_all_labels()
        assert labels == [0, 1, 2, 3]

    def test_format_diagnosis_valid_label(self, mapper):
        """format_diagnosis 应为有效标签生成包含故障原因的可读字符串。"""
        output = mapper.format_diagnosis(0, confidence=0.95)
        assert '滑环编码器转速跳变' in output
        assert '0.95' in output or '95.00%' in output

    def test_format_diagnosis_invalid_label(self, mapper):
        """format_diagnosis 应为无效标签返回提示信息。"""
        output = mapper.format_diagnosis(999)
        assert '未找到' in output

    def test_format_diagnosis_without_confidence(self, mapper):
        """format_diagnosis 在不提供置信度时也应正常工作。"""
        output = mapper.format_diagnosis(1)
        assert '滑环编码器转速规律波动' in output


# ---------------------------------------------------------------------------
# 边界条件测试
# ---------------------------------------------------------------------------
class TestEdgeCases:
    """测试边界条件和异常场景。"""

    def test_missing_columns_in_excel(self, tmp_path):
        """Excel文件缺少必需列时，映射器应优雅处理。"""
        data = {'列A': [1, 2], '列B': ['x', 'y']}
        filepath = tmp_path / "bad_mapping.xlsx"
        pd.DataFrame(data).to_excel(filepath, index=False, engine='openpyxl')

        m = FaultDiagnosisMapper(str(filepath))
        assert m.has_mapping() is False
        assert m.get_diagnosis(0) is None

    def test_empty_excel(self, tmp_path):
        """空Excel文件（有列名但无数据行）时，映射器应优雅处理。"""
        data = {'标签': [], '故障原因': [], '处理方案': []}
        filepath = tmp_path / "empty_mapping.xlsx"
        pd.DataFrame(data).to_excel(filepath, index=False, engine='openpyxl')

        m = FaultDiagnosisMapper(str(filepath))
        assert m.has_mapping() is False
        assert m.get_all_labels() == []
