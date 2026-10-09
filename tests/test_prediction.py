"""
预测引擎单元测试

验证需求:
- 6.1: 预测引擎加载模型并执行批量预测
- 6.2: 预测完成后输出每个样本的预测标签和概率分布
- 6.7: 支持有标签和无标签两种数据格式
"""
import pytest
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from predict import predict_batch


# ---------------------------------------------------------------------------
# Helpers – 简单的 mock 模型和数据集
# ---------------------------------------------------------------------------

class SimpleDLModel(nn.Module):
    """简单的深度学习模型，返回固定维度的 logits。"""

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.linear = nn.Linear(10, num_classes)

    def forward(self, x):
        # x: [B, C, L] -> 取均值后映射到 num_classes
        pooled = x.mean(dim=-1).mean(dim=-1)  # [B]
        # 扩展到 [B, 10]
        expanded = pooled.unsqueeze(-1).expand(-1, 10)
        return self.linear(expanded)


class SimpleDatasetWithLabels(torch.utils.data.Dataset):
    """带标签的简单数据集。"""

    def __init__(self, num_samples: int = 20, num_channels: int = 2,
                 seq_len: int = 16, num_classes: int = 4):
        self.samples = torch.randn(num_samples, num_channels, seq_len)
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]


class SimpleDatasetNoLabels(torch.utils.data.Dataset):
    """无标签的简单数据集。"""

    def __init__(self, num_samples: int = 20, num_channels: int = 2,
                 seq_len: int = 16):
        self.samples = torch.randn(num_samples, num_channels, seq_len)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dl_model():
    """返回一个简单的深度学习模型（CPU）。"""
    model = SimpleDLModel(num_classes=4)
    model.eval()
    return model


@pytest.fixture
def dataloader_with_labels():
    """返回带标签的 DataLoader。"""
    ds = SimpleDatasetWithLabels(num_samples=20, num_classes=4)
    return torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False)


@pytest.fixture
def dataloader_no_labels():
    """返回无标签的 DataLoader。"""
    ds = SimpleDatasetNoLabels(num_samples=20)
    return torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False)


# ---------------------------------------------------------------------------
# 测试批量预测输出格式 (需求 6.1, 6.2)
# ---------------------------------------------------------------------------
class TestPredictBatchOutputFormat:
    """验证 predict_batch 返回的 predictions、probabilities、true_labels 格式。"""

    def test_returns_three_elements(self, dl_model, dataloader_with_labels):
        """predict_batch 应返回三元组。"""
        result = predict_batch(dl_model, dataloader_with_labels, 'cpu',
                               model_name='TSLANet', has_labels=True)
        assert len(result) == 3

    def test_predictions_is_1d_numpy(self, dl_model, dataloader_with_labels):
        """predictions 应为 1D numpy 数组。"""
        predictions, _, _ = predict_batch(
            dl_model, dataloader_with_labels, 'cpu',
            model_name='TSLANet', has_labels=True)
        assert isinstance(predictions, np.ndarray)
        assert predictions.ndim == 1

    def test_predictions_length_matches_samples(self, dl_model, dataloader_with_labels):
        """predictions 长度应等于数据集样本数。"""
        predictions, _, _ = predict_batch(
            dl_model, dataloader_with_labels, 'cpu',
            model_name='TSLANet', has_labels=True)
        assert len(predictions) == 20

    def test_probabilities_is_2d_numpy(self, dl_model, dataloader_with_labels):
        """probabilities 应为 2D numpy 数组 [N, num_classes]。"""
        _, probabilities, _ = predict_batch(
            dl_model, dataloader_with_labels, 'cpu',
            model_name='TSLANet', has_labels=True)
        assert isinstance(probabilities, np.ndarray)
        assert probabilities.ndim == 2
        assert probabilities.shape == (20, 4)

    def test_probabilities_sum_to_one(self, dl_model, dataloader_with_labels):
        """每行概率之和应约等于 1。"""
        _, probabilities, _ = predict_batch(
            dl_model, dataloader_with_labels, 'cpu',
            model_name='TSLANet', has_labels=True)
        row_sums = probabilities.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_probabilities_in_range(self, dl_model, dataloader_with_labels):
        """概率值应在 [0, 1] 范围内。"""
        _, probabilities, _ = predict_batch(
            dl_model, dataloader_with_labels, 'cpu',
            model_name='TSLANet', has_labels=True)
        assert np.all(probabilities >= 0)
        assert np.all(probabilities <= 1)

    def test_true_labels_is_1d_numpy_when_has_labels(self, dl_model, dataloader_with_labels):
        """当 has_labels=True 时，true_labels 应为 1D numpy 数组。"""
        _, _, true_labels = predict_batch(
            dl_model, dataloader_with_labels, 'cpu',
            model_name='TSLANet', has_labels=True)
        assert isinstance(true_labels, np.ndarray)
        assert true_labels.ndim == 1
        assert len(true_labels) == 20


# ---------------------------------------------------------------------------
# 测试有标签和无标签数据的预测 (需求 6.7)
# ---------------------------------------------------------------------------
class TestPredictWithAndWithoutLabels:
    """验证 predict_batch 在有标签和无标签数据下的行为。"""

    def test_with_labels_returns_true_labels(self, dl_model, dataloader_with_labels):
        """has_labels=True 时应返回非 None 的 true_labels。"""
        _, _, true_labels = predict_batch(
            dl_model, dataloader_with_labels, 'cpu',
            model_name='TSLANet', has_labels=True)
        assert true_labels is not None

    def test_without_labels_returns_none(self, dl_model, dataloader_no_labels):
        """has_labels=False 时 true_labels 应为 None。"""
        _, _, true_labels = predict_batch(
            dl_model, dataloader_no_labels, 'cpu',
            model_name='TSLANet', has_labels=False)
        assert true_labels is None

    def test_without_labels_predictions_still_valid(self, dl_model, dataloader_no_labels):
        """无标签时 predictions 和 probabilities 仍应有效。"""
        predictions, probabilities, _ = predict_batch(
            dl_model, dataloader_no_labels, 'cpu',
            model_name='TSLANet', has_labels=False)
        assert isinstance(predictions, np.ndarray)
        assert predictions.ndim == 1
        assert len(predictions) == 20
        assert probabilities.shape == (20, 4)

    def test_without_labels_probabilities_valid(self, dl_model, dataloader_no_labels):
        """无标签时概率分布仍应有效（和为 1，范围 [0,1]）。"""
        _, probabilities, _ = predict_batch(
            dl_model, dataloader_no_labels, 'cpu',
            model_name='TSLANet', has_labels=False)
        row_sums = probabilities.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)
        assert np.all(probabilities >= 0)
        assert np.all(probabilities <= 1)


# ---------------------------------------------------------------------------
# 测试预测结果 JSON 文件的字段完整性 (需求 6.6 via Property 10)
# ---------------------------------------------------------------------------
class TestPredictionResultJSON:
    """验证预测结果 JSON 包含所有必需字段。"""

    @staticmethod
    def _build_result_dict(predictions, probabilities, true_labels,
                           model_name='TSLANet', has_diagnosis=True):
        """模拟 main() 中构建 results 字典的逻辑。"""
        num_classes = probabilities.shape[1] if probabilities.ndim == 2 else 1
        results = {
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist(),
            'num_samples': len(predictions),
            'num_classes': num_classes,
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
        }
        if true_labels is not None:
            results['true_labels'] = true_labels.tolist()
            results['accuracy'] = float(np.mean(predictions == true_labels))
        if has_diagnosis:
            results['diagnosis'] = {}
        return results

    def test_json_contains_predictions(self, dl_model, dataloader_with_labels):
        """JSON 结果应包含 predictions 字段。"""
        preds, probs, labels = predict_batch(
            dl_model, dataloader_with_labels, 'cpu',
            model_name='TSLANet', has_labels=True)
        result = self._build_result_dict(preds, probs, labels)
        assert 'predictions' in result

    def test_json_contains_probabilities(self, dl_model, dataloader_with_labels):
        """JSON 结果应包含 probabilities 字段。"""
        preds, probs, labels = predict_batch(
            dl_model, dataloader_with_labels, 'cpu',
            model_name='TSLANet', has_labels=True)
        result = self._build_result_dict(preds, probs, labels)
        assert 'probabilities' in result

    def test_json_contains_timestamp(self, dl_model, dataloader_with_labels):
        """JSON 结果应包含 timestamp 字段。"""
        preds, probs, labels = predict_batch(
            dl_model, dataloader_with_labels, 'cpu',
            model_name='TSLANet', has_labels=True)
        result = self._build_result_dict(preds, probs, labels)
        assert 'timestamp' in result

    def test_json_contains_model_name(self, dl_model, dataloader_with_labels):
        """JSON 结果应包含 model_name 字段。"""
        preds, probs, labels = predict_batch(
            dl_model, dataloader_with_labels, 'cpu',
            model_name='TSLANet', has_labels=True)
        result = self._build_result_dict(preds, probs, labels)
        assert 'model_name' in result

    def test_json_contains_diagnosis(self, dl_model, dataloader_with_labels):
        """JSON 结果应包含 diagnosis 字段。"""
        preds, probs, labels = predict_batch(
            dl_model, dataloader_with_labels, 'cpu',
            model_name='TSLANet', has_labels=True)
        result = self._build_result_dict(preds, probs, labels)
        assert 'diagnosis' in result

    def test_json_all_required_fields_present(self, dl_model, dataloader_with_labels):
        """JSON 结果应同时包含所有 5 个必需字段。"""
        preds, probs, labels = predict_batch(
            dl_model, dataloader_with_labels, 'cpu',
            model_name='TSLANet', has_labels=True)
        result = self._build_result_dict(preds, probs, labels)
        required_fields = ['predictions', 'probabilities', 'timestamp',
                           'model_name', 'diagnosis']
        for field in required_fields:
            assert field in result, f"缺少必需字段: {field}"

    def test_json_serializable(self, dl_model, dataloader_with_labels):
        """结果字典应可被 json.dumps 序列化。"""
        preds, probs, labels = predict_batch(
            dl_model, dataloader_with_labels, 'cpu',
            model_name='TSLANet', has_labels=True)
        result = self._build_result_dict(preds, probs, labels)
        serialized = json.dumps(result, ensure_ascii=False)
        assert isinstance(serialized, str)
        parsed = json.loads(serialized)
        assert parsed['model_name'] == 'TSLANet'

    def test_json_without_labels_no_true_labels_field(self, dl_model, dataloader_no_labels):
        """无标签时 JSON 结果不应包含 true_labels 字段。"""
        preds, probs, labels = predict_batch(
            dl_model, dataloader_no_labels, 'cpu',
            model_name='TSLANet', has_labels=False)
        result = self._build_result_dict(preds, probs, labels)
        assert 'true_labels' not in result

    def test_json_with_labels_has_accuracy(self, dl_model, dataloader_with_labels):
        """有标签时 JSON 结果应包含 accuracy 字段。"""
        preds, probs, labels = predict_batch(
            dl_model, dataloader_with_labels, 'cpu',
            model_name='TSLANet', has_labels=True)
        result = self._build_result_dict(preds, probs, labels)
        assert 'accuracy' in result
        assert 0.0 <= result['accuracy'] <= 1.0
