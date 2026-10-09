import torch
import random

# ── 配置 ──────────────────────────────────────────────
N = 254          # 总样本数
SEED = 42        # 随机种子（可修改）
RATIOS = (0.6, 0.2, 0.2)   # train / val / test

# ── 生成随机数据 ──────────────────────────────────────
torch.manual_seed(SEED)
random.seed(SEED)

samples = torch.randn(N, 80, dtype=torch.float32)
labels  = torch.randint(0, 10, (N,), dtype=torch.int64)   # 假设 10 个类别

# ── 划分索引 ──────────────────────────────────────────
indices = list(range(N))
random.shuffle(indices)

n_train = int(N * RATIOS[0])          # 152
n_val   = int(N * RATIOS[1])          # 50
n_test  = N - n_train - n_val         # 52

splits = {
    "train": indices[:n_train],
    "val":   indices[n_train : n_train + n_val],
    "test":  indices[n_train + n_val :],
}

# ── 保存 .pt 文件 ─────────────────────────────────────
for name, idx in splits.items():
    data = {
        "samples": samples[idx],   # torch.float32
        "labels":  labels[idx],    # torch.int64
    }
    torch.save(data, f"{name}.pt")
    print(f"[{name}]  samples: {data['samples'].shape}  labels: {data['labels'].shape}")

print("\n✅ 生成完毕：train.pt / val.pt / test.pt")