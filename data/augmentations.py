import numpy as np
import torch


def DataTransform(sample, config):
    """
    数据增强函数
    
    Args:
        sample: 输入数据 shape: (batch_size, channels, seq_len)
        config: 配置字典，需要包含 TS-TCC 的 augmentation 配置
    
    Returns:
        weak_aug, strong_aug: 两种增强后的数据
    """
    # 从配置中获取增强参数
    if isinstance(config, dict):
        # config 是字典形式
        ts_tcc_config = config.get('TS-TCC', {})
        aug_config = ts_tcc_config.get('augmentation', {})
        jitter_scale_ratio = aug_config.get('jitter_scale_ratio', 1.0)
        jitter_ratio = aug_config.get('jitter_ratio', 0.5)
        max_seg = aug_config.get('max_seg', 10)
    else:
        # 兼容对象形式访问
        jitter_scale_ratio = config.TS-TCC.augmentation.jitter_scale_ratio
        jitter_ratio = config.TS-TCC.augmentation.jitter_ratio
        max_seg = config.TS-TCC.augmentation.max_seg

    weak_aug = scaling(sample, jitter_scale_ratio)
    strong_aug = jitter(permutation(sample, max_segments=max_seg), jitter_ratio)

    return weak_aug, strong_aug


def jitter(x, sigma=0.8):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def scaling(x, sigma=1.1):
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[0], x.shape[2]))
    ai = []
    for i in range(x.shape[1]):
        xi = x[:, i, :]
        ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
    return np.concatenate((ai), axis=1)


def permutation(x, max_segments=5, seg_mode="random"):
    orig_steps = np.arange(x.shape[2])

    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            # 随机排列 splits 列表中的元素，然后连接
            split_indices = np.arange(len(splits))
            np.random.shuffle(split_indices)
            warp = np.concatenate([splits[j] for j in split_indices]).ravel()
            ret[i] = pat[0, warp]
        else:
            ret[i] = pat
    return torch.from_numpy(ret)

