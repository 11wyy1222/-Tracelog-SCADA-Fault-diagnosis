"""
掩码工具模块
用于TSLANet等模型的预训练
"""
import torch
import numpy as np


def random_masking_3D(xb, mask_ratio):
    """
    3D输入的随机掩码 (改编自PatchTST)
    用于时间序列的掩码重构预训练
    
    Args:
        xb: 输入张量 [batch_size x num_patches x dim]
        mask_ratio: 要掩码的补丁比例 (0-1之间)
        
    Returns:
        x_masked: 掩码后的输入 [batch_size x num_patches x dim]
        x_kept: 仅保留的补丁 [batch_size x len_keep x dim]
        mask: 二进制掩码 (0=保留, 1=移除) [batch_size x num_patches]
        ids_restore: 恢复原始顺序的索引 [batch_size x num_patches]
    """
    bs, L, D = xb.shape
    x = xb.clone()

    # 计算保留的补丁数量
    len_keep = int(L * (1 - mask_ratio))

    # 生成随机噪声用于打乱
    noise = torch.rand(bs, L, device=xb.device)  # 噪声范围 [0, 1]

    # 对每个样本的噪声进行排序
    ids_shuffle = torch.argsort(noise, dim=1)  # 升序: 小的保留, 大的移除
    ids_restore = torch.argsort(ids_shuffle, dim=1)  # 恢复索引

    # 保留前面的子集
    ids_keep = ids_shuffle[:, :len_keep]  # [bs x len_keep]
    x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    # 移除的补丁 (用零填充)
    x_removed = torch.zeros(bs, L - len_keep, D, device=xb.device)
    x_ = torch.cat([x_kept, x_removed], dim=1)  # [bs x L x dim]

    # 恢复原始顺序
    x_masked = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D))

    # 生成二进制掩码: 0表示保留, 1表示移除
    mask = torch.ones([bs, L], device=x.device)
    mask[:, :len_keep] = 0
    # 打乱掩码以匹配原始顺序
    mask = torch.gather(mask, dim=1, index=ids_restore)
    
    return x_masked, x_kept, mask, ids_restore


def block_masking(xb, block_size, mask_ratio):
    """
    块状掩码 - 连续掩盖一段时间序列
    
    Args:
        xb: 输入张量 [batch_size x num_patches x dim]
        block_size: 块大小
        mask_ratio: 掩码比例
        
    Returns:
        x_masked: 掩码后的输入
        mask: 二进制掩码
    """
    bs, L, D = xb.shape
    x = xb.clone()
    
    # 计算需要掩盖的块数
    num_blocks = int((L / block_size) * mask_ratio)
    
    mask = torch.zeros([bs, L], device=xb.device)
    
    for b in range(bs):
        # 随机选择起始位置
        start_positions = torch.randint(0, L - block_size + 1, (num_blocks,), device=xb.device)
        
        for start in start_positions:
            # 掩盖连续的块
            mask[b, start:start + block_size] = 1
            x[b, start:start + block_size, :] = 0
    
    return x, mask


def random_masking_2D(xb, mask_ratio):
    """
    2D输入的随机掩码
    
    Args:
        xb: 输入张量 [batch_size x seq_len]
        mask_ratio: 掩码比例
        
    Returns:
        x_masked: 掩码后的输入
        mask: 二进制掩码
    """
    bs, L = xb.shape
    x = xb.clone()
    
    len_keep = int(L * (1 - mask_ratio))
    
    noise = torch.rand(bs, L, device=xb.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    
    ids_keep = ids_shuffle[:, :len_keep]
    x_kept = torch.gather(x, dim=1, index=ids_keep)
    
    x_removed = torch.zeros(bs, L - len_keep, device=xb.device)
    x_ = torch.cat([x_kept, x_removed], dim=1)
    
    x_masked = torch.gather(x_, dim=1, index=ids_restore)
    
    mask = torch.ones([bs, L], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    
    return x_masked, mask


def geometric_masking(xb, mask_ratio, temperature=1.0):
    """
    几何分布掩码 - 更倾向于掩盖某些区域
    
    Args:
        xb: 输入张量 [batch_size x num_patches x dim]
        mask_ratio: 掩码比例
        temperature: 温度参数,控制分布的集中程度
        
    Returns:
        x_masked: 掩码后的输入
        mask: 二进制掩码
    """
    bs, L, D = xb.shape
    x = xb.clone()
    
    len_keep = int(L * (1 - mask_ratio))
    
    # 使用几何分布生成概率
    positions = torch.arange(L, device=xb.device).float()
    probs = torch.exp(-positions / (L * temperature))
    probs = probs.unsqueeze(0).repeat(bs, 1)
    
    # 添加随机噪声
    noise = torch.rand(bs, L, device=xb.device)
    combined_scores = probs * noise
    
    # 排序选择
    ids_shuffle = torch.argsort(combined_scores, dim=1, descending=True)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    
    ids_keep = ids_shuffle[:, :len_keep]
    x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
    
    x_removed = torch.zeros(bs, L - len_keep, D, device=xb.device)
    x_ = torch.cat([x_kept, x_removed], dim=1)
    
    x_masked = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D))
    
    mask = torch.ones([bs, L], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    
    return x_masked, mask


def channel_masking(xb, mask_ratio):
    """
    通道掩码 - 掩盖整个通道
    
    Args:
        xb: 输入张量 [batch_size x num_channels x seq_len]
        mask_ratio: 掩码比例
        
    Returns:
        x_masked: 掩码后的输入
        mask: 二进制掩码 [batch_size x num_channels]
    """
    bs, C, L = xb.shape
    x = xb.clone()
    
    num_keep = int(C * (1 - mask_ratio))
    
    mask = torch.zeros([bs, C], device=xb.device)
    
    for b in range(bs):
        # 随机选择要掩盖的通道
        masked_channels = torch.randperm(C, device=xb.device)[num_keep:]
        mask[b, masked_channels] = 1
        x[b, masked_channels, :] = 0
    
    return x, mask


def adaptive_masking(xb, mask_ratio, importance_scores=None):
    """
    自适应掩码 - 根据重要性分数掩盖
    
    Args:
        xb: 输入张量 [batch_size x num_patches x dim]
        mask_ratio: 掩码比例
        importance_scores: 重要性分数 [batch_size x num_patches]
                          如果为None,则使用方差作为重要性
        
    Returns:
        x_masked: 掩码后的输入
        mask: 二进制掩码
    """
    bs, L, D = xb.shape
    x = xb.clone()
    
    len_keep = int(L * (1 - mask_ratio))
    
    # 如果没有提供重要性分数,使用方差
    if importance_scores is None:
        importance_scores = torch.var(xb, dim=-1)  # [bs x L]
    
    # 根据重要性排序(重要的保留,不重要的掩盖)
    ids_shuffle = torch.argsort(importance_scores, dim=1, descending=True)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    
    ids_keep = ids_shuffle[:, :len_keep]
    x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
    
    x_removed = torch.zeros(bs, L - len_keep, D, device=xb.device)
    x_ = torch.cat([x_kept, x_removed], dim=1)
    
    x_masked = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D))
    
    mask = torch.ones([bs, L], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    
    return x_masked, mask


def get_masking_function(masking_type='random'):
    """
    获取掩码函数
    
    Args:
        masking_type: 掩码类型
            - 'random': 随机掩码
            - 'block': 块状掩码
            - 'geometric': 几何分布掩码
            - 'channel': 通道掩码
            - 'adaptive': 自适应掩码
            
    Returns:
        masking_fn: 掩码函数
    """
    masking_functions = {
        'random': random_masking_3D,
        'block': block_masking,
        'geometric': geometric_masking,
        'channel': channel_masking,
        'adaptive': adaptive_masking
    }
    
    if masking_type not in masking_functions:
        raise ValueError(
            f"未知的掩码类型: {masking_type}\n"
            f"可用选项: {', '.join(masking_functions.keys())}"
        )
    
    return masking_functions[masking_type]