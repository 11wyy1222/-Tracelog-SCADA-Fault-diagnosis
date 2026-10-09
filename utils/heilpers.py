"""
辅助工具函数
包含各种常用的辅助功能
"""
import torch
import numpy as np
import random
import os
import shutil
import json
from pathlib import Path


def set_seed(seed=42):
    """
    设置所有随机种子以确保可复现性
    
    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 确保CUDA操作的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"✓ 随机种子已设置为: {seed}")


def get_device(prefer_cuda=True):
    """
    获取可用的计算设备
    
    Args:
        prefer_cuda: 是否优先使用CUDA
        
    Returns:
        device: torch设备对象
    """
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ 使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("✓ 使用CPU")
    
    return device


def save_checkpoint(model, optimizer, epoch, loss, save_path, **kwargs):
    """
    保存模型检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前轮次
        loss: 当前损失
        save_path: 保存路径
        **kwargs: 其他要保存的信息
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        **kwargs
    }
    
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    torch.save(checkpoint, save_path)
    print(f"✓ 检查点已保存至: {save_path}")


def load_checkpoint(model, checkpoint_path, optimizer=None, device='cpu'):
    """
    加载模型检查点
    
    Args:
        model: 模型
        checkpoint_path: 检查点路径
        optimizer: 优化器(可选)
        device: 设备
        
    Returns:
        epoch: 轮次
        loss: 损失
        additional_info: 其他信息
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', None)
    
    # 提取其他信息
    additional_info = {k: v for k, v in checkpoint.items() 
                      if k not in ['epoch', 'model_state_dict', 'optimizer_state_dict', 'loss']}
    
    print(f"✓ 检查点已加载: epoch={epoch}, loss={loss}")
    
    return epoch, loss, additional_info


def copy_files(source_files, destination_dir):
    """
    复制文件到目标目录
    
    Args:
        source_files: 源文件列表
        destination_dir: 目标目录
    """
    os.makedirs(destination_dir, exist_ok=True)
    
    if isinstance(source_files, str):
        source_files = [source_files]
    
    for source_file in source_files:
        if os.path.exists(source_file):
            shutil.copy(source_file, destination_dir)
            print(f"✓ 已复制: {source_file} -> {destination_dir}")
        else:
            print(f"⚠ 文件不存在,跳过: {source_file}")


def create_experiment_dir(base_dir, experiment_name, run_name=None):
    """
    创建实验目录结构
    
    Args:
        base_dir: 基础目录
        experiment_name: 实验名称
        run_name: 运行名称
        
    Returns:
        experiment_dir: 实验目录路径
    """
    if run_name:
        experiment_dir = os.path.join(base_dir, experiment_name, run_name)
    else:
        experiment_dir = os.path.join(base_dir, experiment_name)
    
    # 创建子目录
    subdirs = ['checkpoints', 'logs', 'figures', 'results']
    for subdir in subdirs:
        os.makedirs(os.path.join(experiment_dir, subdir), exist_ok=True)
    
    print(f"✓ 实验目录已创建: {experiment_dir}")
    
    return experiment_dir


def save_config(config, save_path):
    """
    保存配置到JSON文件
    
    Args:
        config: 配置字典
        save_path: 保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 配置已保存至: {save_path}")


def load_config(config_path):
    """
    从JSON文件加载配置
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        config: 配置字典
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print(f"✓ 配置已加载: {config_path}")
    
    return config


def count_model_parameters(model, trainable_only=False):
    """
    计算模型参数数量
    
    Args:
        model: 模型
        trainable_only: 是否只计算可训练参数
        
    Returns:
        num_params: 参数数量
    """
    if trainable_only:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        num_params = sum(p.numel() for p in model.parameters())
    
    return num_params


def format_time(seconds):
    """
    格式化时间
    
    Args:
        seconds: 秒数
        
    Returns:
        formatted_time: 格式化的时间字符串
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def format_number(number):
    """
    格式化数字(添加千位分隔符)
    
    Args:
        number: 数字
        
    Returns:
        formatted_number: 格式化的数字字符串
    """
    return f"{number:,}"


class EarlyStopping:
    """早停工具类"""
    
    def __init__(self, patience=10, min_delta=0, mode='min'):
        """
        初始化早停
        
        Args:
            patience: 容忍轮数
            min_delta: 最小改善量
            mode: 模式 ('min'表示越小越好, 'max'表示越大越好)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            raise ValueError(f"模式必须是 'min' 或 'max', 得到: {mode}")
    
    def __call__(self, score):
        """
        检查是否应该早停
        
        Args:
            score: 当前分数
            
        Returns:
            early_stop: 是否早停
        """
        if self.best_score is None:
            self.best_score = score
        elif self.monitor_op(score - self.min_delta, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def reset(self):
        """重置早停状态"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False


class AverageMeter:
    """平均值计算器"""
    
    def __init__(self):
        """初始化"""
        self.reset()
    
    def reset(self):
        """重置"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        """
        更新
        
        Args:
            val: 值
            n: 数量
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_lr(optimizer):
    """
    获取当前学习率
    
    Args:
        optimizer: 优化器
        
    Returns:
        lr: 学习率
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


def adjust_learning_rate(optimizer, epoch, initial_lr, decay_rate=0.1, decay_epochs=30):
    """
    调整学习率
    
    Args:
        optimizer: 优化器
        epoch: 当前轮次
        initial_lr: 初始学习率
        decay_rate: 衰减率
        decay_epochs: 衰减轮数
        
    Returns:
        lr: 新的学习率
    """
    lr = initial_lr * (decay_rate ** (epoch // decay_epochs))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return lr


# 兼容旧版本的函数别名
copy_Files = copy_files