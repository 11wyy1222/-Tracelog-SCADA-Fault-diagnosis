"""
数据加载器单元测试

验证需求:
- 3.1: 加载.pt格式数据文件，返回包含训练集、验证集和测试集的DataLoader元组
- 3.3: TS-TCC自监督模式下对数据应用增强策略
- 3.4: 输出张量的通道维度位于第二维（形状为[N, C, L]）
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pytest
import torch
import numpy as np
import sys
from pathlib import Path
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.unified_dataloader import (
    UnifiedDataset,
    calculate_padding,
    zero_pad_sequence,
    get_dataloader,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_pt_data(n_samples=32, n_channels=4, seq_len=100):
    """构造模拟的 .pt 数据字典，形状为 [N, C, L]。"""
    samples = torch.randn(n_samples, n_channels, seq_len)
    labels = torch.randint(0, 4, (n_samples,))
    return {"samples": samples, "labels": labels}


def _save_splits(tmp_path, n_samples=32, n_channels=4, seq_len=100):
    """在 tmp_path 下保存 train.pt / val.pt / test.pt。"""
    for name in ("train", "val", "test"):
        data = _make_pt_data(n_samples, n_channels, seq_len)
        torch.save(data, tmp_path / f"{name}.pt")


def _base_config():
    """返回一个最小可用的配置字典。"""
    return {
        "batch_size": 8,
        "drop_last": False,
        "patch_size": 8,
        "TS-TCC": {
            "augmentation": {
                "jitter_scale_ratio": 1.0,
                "jitter_ratio": 0.5,
                "max_seg": 5,
            }
        },
    }


# ---------------------------------------------------------------------------
# Requirement 3.1 – .pt 文件加载和 DataLoader 创建
# ---------------------------------------------------------------------------
class TestGetDataloader:
    """验证 get_dataloader 能正确加载 .pt 文件并返回 DataLoader 元组。"""

    def test_returns_three_dataloaders(self, tmp_path):
        """get_dataloader 应返回 (train, val, test) 三个 DataLoader。"""
        _save_splits(tmp_path)
        config = _base_config()
        train_dl, val_dl, test_dl = get_dataloader(
            str(tmp_path), config, "supervised", "TSLANet"
        )
        assert isinstance(train_dl, DataLoader)
        assert isinstance(val_dl, DataLoader)
        assert isinstance(test_dl, DataLoader)

    def test_dataloaders_yield_batches(self, tmp_path):
        """每个 DataLoader 应能迭代出至少一个批次。"""
        _save_splits(tmp_path, n_samples=16)
        config = _base_config()
        config["batch_size"] = 8
        train_dl, val_dl, test_dl = get_dataloader(
            str(tmp_path), config, "supervised", "TSLANet"
        )
        for dl in (train_dl, val_dl, test_dl):
            batch = next(iter(dl))
            # 每个批次应返回 (x, y, aug1, aug2) 四元组
            assert len(batch) == 4

    def test_batch_labels_are_long_tensors(self, tmp_path):
        """标签张量应为 long 类型。"""
        _save_splits(tmp_path)
        config = _base_config()
        train_dl, _, _ = get_dataloader(
            str(tmp_path), config, "supervised", "TSLANet"
        )
        batch = next(iter(train_dl))
        _, y, _, _ = batch
        assert y.dtype == torch.long


# ---------------------------------------------------------------------------
# Requirement 3.4 – 通道维度位于第二维 [N, C, L]
# ---------------------------------------------------------------------------
class TestChannelDimension:
    """验证 UnifiedDataset 确保通道维度在 dim=1。"""

    def test_channel_at_dim1_when_already_correct(self):
        """输入已经是 [N, C, L] 时，输出形状不变。"""
        data = _make_pt_data(n_samples=10, n_channels=4, seq_len=200)
        ds = UnifiedDataset(data, _base_config(), "supervised", "TSLANet")
        x, _, _, _ = ds[0]
        # x 应为 [C, L]，C=4, L=200
        assert x.shape[0] == 4
        assert x.shape[1] == 200

    def test_channel_at_dim1_when_transposed(self):
        """输入为 [N, L, C]（通道在最后）时，应自动转置为 [N, C, L]。"""
        n, c, l = 10, 2, 300
        # 构造 [N, L, C] 形状，其中 C < L，min 维度在最后
        samples = torch.randn(n, l, c)
        labels = torch.randint(0, 3, (n,))
        data = {"samples": samples, "labels": labels}
        ds = UnifiedDataset(data, _base_config(), "supervised", "TSLANet")
        x, _, _, _ = ds[0]
        # 转置后应为 [C, L]
        assert x.shape[0] == c
        assert x.shape[1] == l

    def test_2d_input_gets_unsqueezed(self):
        """输入为 [N, L]（2D）时，应自动扩展为 [N, 1, L]。"""
        n, l = 10, 500
        samples = torch.randn(n, l)
        labels = torch.randint(0, 2, (n,))
        data = {"samples": samples, "labels": labels}
        ds = UnifiedDataset(data, _base_config(), "supervised", "TSLANet")
        x, _, _, _ = ds[0]
        # 应为 [1, L]
        assert x.dim() == 2
        assert x.shape[0] == 1
        assert x.shape[1] == l

    def test_dataloader_batch_shape_is_NCL(self, tmp_path):
        """通过 get_dataloader 获取的批次形状应为 [B, C, L]。"""
        n, c, l = 20, 4, 100
        _save_splits(tmp_path, n_samples=n, n_channels=c, seq_len=l)
        config = _base_config()
        config["batch_size"] = 8
        train_dl, _, _ = get_dataloader(
            str(tmp_path), config, "supervised", "TSLANet"
        )
        batch_x, _, _, _ = next(iter(train_dl))
        assert batch_x.dim() == 3
        assert batch_x.shape[1] == c  # 通道在 dim=1


# ---------------------------------------------------------------------------
# Requirement 3.3 – TS-TCC 自监督模式数据增强
# ---------------------------------------------------------------------------
class TestTSTCCAugmentation:
    """验证 TS-TCC 自监督模式下产生增强数据。"""

    def test_self_supervised_returns_augmented_data(self):
        """self_supervised + TS-TCC 模式下，aug1 和 aug2 不应为 None。"""
        data = _make_pt_data(n_samples=16, n_channels=4, seq_len=100)
        config = _base_config()
        ds = UnifiedDataset(data, config, "self_supervised", "TS-TCC")
        assert ds.aug1 is not None
        assert ds.aug2 is not None

    def test_augmented_data_shape_matches_original(self):
        """增强数据的形状应与原始数据一致。"""
        n, c, l = 16, 4, 100
        data = _make_pt_data(n_samples=n, n_channels=c, seq_len=l)
        config = _base_config()
        ds = UnifiedDataset(data, config, "self_supervised", "TS-TCC")
        x, y, aug1, aug2 = ds[0]
        assert aug1.shape == x.shape
        assert aug2.shape == x.shape

    def test_augmented_data_differs_from_original(self):
        """增强数据应与原始数据不完全相同（概率极高）。"""
        data = _make_pt_data(n_samples=16, n_channels=4, seq_len=100)
        config = _base_config()
        ds = UnifiedDataset(data, config, "self_supervised", "TS-TCC")
        x, _, aug1, aug2 = ds[0]
        # 增强后的数据不应与原始完全相同
        x_np = x.numpy() if isinstance(x, torch.Tensor) else np.array(x)
        aug1_np = aug1.numpy() if isinstance(aug1, torch.Tensor) else np.array(aug1)
        assert not np.allclose(x_np, aug1_np, atol=1e-6)

    def test_non_self_supervised_no_augmentation(self):
        """非 self_supervised 模式下，aug1/aug2 应为 None。"""
        data = _make_pt_data(n_samples=16, n_channels=4, seq_len=100)
        config = _base_config()
        ds = UnifiedDataset(data, config, "supervised", "TS-TCC")
        assert ds.aug1 is None
        assert ds.aug2 is None

    def test_non_tstcc_model_no_augmentation(self):
        """非 TS-TCC 模型即使在 self_supervised 模式下也不应增强。"""
        data = _make_pt_data(n_samples=16, n_channels=4, seq_len=100)
        config = _base_config()
        ds = UnifiedDataset(data, config, "self_supervised", "TSLANet")
        assert ds.aug1 is None
        assert ds.aug2 is None


# ---------------------------------------------------------------------------
# 工具函数测试 – calculate_padding / zero_pad_sequence
# ---------------------------------------------------------------------------
class TestCalculatePadding:
    """验证 calculate_padding 工具函数。"""

    def test_no_padding_needed(self):
        """序列长度已是 patch_size 整数倍时，填充为 0。"""
        assert calculate_padding(16, 8) == 0
        assert calculate_padding(100, 10) == 0

    def test_padding_needed(self):
        """序列长度不是 patch_size 整数倍时，返回正确的填充量。"""
        assert calculate_padding(15, 8) == 1   # 15 + 1 = 16
        assert calculate_padding(17, 8) == 7   # 17 + 7 = 24
        assert calculate_padding(101, 10) == 9  # 101 + 9 = 110

    def test_padding_result_divisible(self):
        """填充后的长度应能被 patch_size 整除。"""
        for seq_len in (7, 13, 29, 100, 329):
            for patch_size in (4, 8, 16):
                pad = calculate_padding(seq_len, patch_size)
                assert (seq_len + pad) % patch_size == 0


class TestZeroPadSequence:
    """验证 zero_pad_sequence 工具函数。"""

    def test_pad_increases_last_dim(self):
        """填充后最后一维长度应增加 pad_length。"""
        t = torch.randn(4, 2, 100)
        padded = zero_pad_sequence(t, 10)
        assert padded.shape == (4, 2, 110)

    def test_pad_zero_no_change(self):
        """pad_length=0 时，张量不变。"""
        t = torch.randn(4, 2, 100)
        padded = zero_pad_sequence(t, 0)
        assert torch.equal(t, padded)

    def test_padded_region_is_zero(self):
        """填充区域的值应全为 0。"""
        t = torch.ones(2, 1, 50)
        padded = zero_pad_sequence(t, 10)
        assert torch.all(padded[:, :, 50:] == 0)
        assert torch.all(padded[:, :, :50] == 1)
