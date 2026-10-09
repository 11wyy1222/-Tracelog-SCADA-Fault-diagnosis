"""
Utils模块 - 工具函数集合
"""

# 日志工具
from .logger import (
    _logger,
    set_requires_grad,
    count_parameters,
    print_model_info,
    LogFormatter,
    log_epoch_info,
    log_training_start,
    log_training_complete
)

# 损失函数
from .loss import (
    NTXentLoss,
    FocalLoss,
    LabelSmoothingCrossEntropy,
    SupConLoss,
    TripletLoss,
    get_loss_function
)

# 评估指标
from .metrics import (
    calculate_metrics,
    save_confusion_matrix,
    calculate_per_class_metrics,
    calculate_top_k_accuracy,
    calculate_macro_micro_metrics,
    plot_training_history,
    MetricsTracker,
    _calc_metrics  # 兼容旧版本
)

# 掩码工具
from .masking import (
    random_masking_3D,
    block_masking,
    random_masking_2D,
    geometric_masking,
    channel_masking,
    adaptive_masking,
    get_masking_function
)


__all__ = [
    # 日志
    '_logger',
    'set_requires_grad',
    'count_parameters',
    'print_model_info',
    'LogFormatter',
    'log_epoch_info',
    'log_training_start',
    'log_training_complete',
    
    # 损失
    'NTXentLoss',
    'FocalLoss',
    'LabelSmoothingCrossEntropy',
    'SupConLoss',
    'TripletLoss',
    'get_loss_function',
    
    # 指标
    'calculate_metrics',
    'save_confusion_matrix',
    'calculate_per_class_metrics',
    'calculate_top_k_accuracy',
    'calculate_macro_micro_metrics',
    'plot_training_history',
    'MetricsTracker',
    '_calc_metrics',
    
    # 掩码
    'random_masking_3D',
    'block_masking',
    'random_masking_2D',
    'geometric_masking',
    'channel_masking',
    'adaptive_masking',
    'get_masking_function',
]