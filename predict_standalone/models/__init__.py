"""
模型包初始化 - 自动注册所有模型
"""
try:
    from models.tslanet import TSLANet
except ImportError:
    pass

try:
    from models.ts_tcc import base_Model
except ImportError:
    pass
