"""
损失函数模块
包含各种时间序列分类的损失函数
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NTXentLoss(nn.Module):
    """
    归一化温度缩放交叉熵损失 (Normalized Temperature-scaled Cross Entropy Loss)
    用于对比学习
    """

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        """
        初始化NTXent损失
        
        Args:
            device: 运行设备
            batch_size: 批次大小
            temperature: 温度参数
            use_cosine_similarity: 是否使用余弦相似度
        """
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        """获取相似度函数"""
        if use_cosine_similarity:
            self._cosine_similarity = nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        """
        生成相关性掩码
        用于屏蔽正样本对
        """
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        """点积相似度"""
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        """余弦相似度"""
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        """
        前向传播
        
        Args:
            zis: 第一组表示 [batch_size, dim]
            zjs: 第二组表示 [batch_size, dim]
            
        Returns:
            loss: NTXent损失
        """
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # 过滤正样本对的分数
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)


class FocalLoss(nn.Module):
    """
    Focal Loss - 用于处理类别不平衡
    """
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        """
        初始化Focal Loss
        
        Args:
            alpha: 平衡因子
            gamma: 聚焦参数
            reduction: 损失缩减方式
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        前向传播
        
        Args:
            inputs: 预测logits [batch_size, num_classes]
            targets: 真实标签 [batch_size]
            
        Returns:
            loss: Focal损失
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    标签平滑交叉熵损失
    """
    
    def __init__(self, smoothing=0.1):
        """
        初始化标签平滑损失
        
        Args:
            smoothing: 平滑系数
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
    
    def forward(self, inputs, targets):
        """
        前向传播
        
        Args:
            inputs: 预测logits [batch_size, num_classes]
            targets: 真实标签 [batch_size]
            
        Returns:
            loss: 标签平滑损失
        """
        num_classes = inputs.size(-1)
        log_probs = F.log_softmax(inputs, dim=-1)
        
        # 创建平滑标签
        targets = targets.unsqueeze(1)
        smooth_labels = torch.zeros_like(log_probs).scatter_(1, targets, 1)
        smooth_labels = smooth_labels * (1 - self.smoothing) + self.smoothing / num_classes
        
        loss = (-smooth_labels * log_probs).sum(dim=-1).mean()
        
        return loss


class SupConLoss(nn.Module):
    """
    监督对比损失 (Supervised Contrastive Loss)
    """
    
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        """
        初始化监督对比损失
        
        Args:
            temperature: 温度参数
            contrast_mode: 对比模式
            base_temperature: 基础温度
        """
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
    
    def forward(self, features, labels=None, mask=None):
        """
        前向传播
        
        Args:
            features: 特征向量 [batch_size, n_views, feature_dim]
            labels: 标签 [batch_size]
            mask: 对比掩码
            
        Returns:
            loss: 监督对比损失
        """
        device = features.device
        
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                           'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        
        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
        
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))
        
        # 计算相似度
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature
        )
        
        # 数值稳定性
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # 创建掩码
        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # 计算log概率
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # 计算平均log似然
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        
        # 损失
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        
        return loss


class TripletLoss(nn.Module):
    """
    三元组损失 (Triplet Loss)
    用于度量学习
    """
    
    def __init__(self, margin=1.0):
        """
        初始化三元组损失
        
        Args:
            margin: 间隔参数
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        """
        前向传播
        
        Args:
            anchor: 锚点样本
            positive: 正样本
            negative: 负样本
            
        Returns:
            loss: 三元组损失
        """
        pos_distance = F.pairwise_distance(anchor, positive)
        neg_distance = F.pairwise_distance(anchor, negative)
        
        loss = F.relu(pos_distance - neg_distance + self.margin)
        
        return loss.mean()


def get_loss_function(loss_name, **kwargs):
    """
    获取损失函数
    
    Args:
        loss_name: 损失函数名称
        **kwargs: 损失函数参数
        
    Returns:
        loss_fn: 损失函数
    """
    loss_functions = {
        'cross_entropy': nn.CrossEntropyLoss,
        'focal_loss': FocalLoss,
        'label_smoothing': LabelSmoothingCrossEntropy,
        'ntxent': NTXentLoss,
        'supcon': SupConLoss,
        'triplet': TripletLoss,
    }
    
    if loss_name not in loss_functions:
        raise ValueError(
            f"未知的损失函数: {loss_name}\n"
            f"可用选项: {', '.join(loss_functions.keys())}"
        )
    
    return loss_functions[loss_name](**kwargs)