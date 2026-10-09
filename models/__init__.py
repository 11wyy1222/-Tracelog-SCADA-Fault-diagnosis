"""
模型模块初始化
自动注册所有模型到ModelFactory
"""

# 首先导入ModelFactory
from .model_factory import ModelFactory

# 导入并注册TSLANet模型
from .tslanet import TSLANet, ICB, PatchEmbed, AdaptiveSpectralBlock, TSLANetLayer

# 注册TSLANet到工厂
ModelFactory.register('TSLANet')(TSLANet)

# 导入并注册TS-TCC模型
try:
    from .ts_tcc import base_Model, TC
    # TS-TCC已经在ts_tcc.py中使用装饰器注册了
except ImportError:
    print("警告: 无法导入TS-TCC模型")

# 导入注意力模块
try:
    from .attention import (
        Residual, PreNorm, FeedForward, 
        Attention, Transformer, Seq_Transformer
    )
except ImportError:
    print("警告: 无法导入attention模块")

# 打印已注册的模型
print(f"已注册的模型: {', '.join(ModelFactory.list_models())}")


__all__ = [
    'ModelFactory',
    'TSLANet',
    'ICB',
    'PatchEmbed',
    'AdaptiveSpectralBlock',
    'TSLANetLayer',
]