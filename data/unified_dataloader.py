"""
统一数据加载器 - 支持TSLANet和TS-TCC
"""
import torch
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np


class UnifiedDataset(Dataset):
    """统一的数据集类,支持不同模型的需求"""
    
    def __init__(self, data_file, config, training_mode, model_name):
        """
        初始化数据集
        
        Args:
            data_file: 数据文件(包含'samples'和'labels')
            config: 配置字典或对象
            training_mode: 训练模式
            model_name: 模型名称
        """
        super(UnifiedDataset, self).__init__()
        self.training_mode = training_mode
        self.model_name = model_name
        
        # 加载数据
        X_train = data_file["samples"]
        y_train = data_file["labels"]
        
        # 处理维度
        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)
        
        # 确保通道在第二维
        if X_train.shape.index(min(X_train.shape)) != 1:
            X_train = X_train.permute(0, 2, 1)
        
        # 转换为tensor
        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = X_train
            self.y_data = y_train
        
        self.len = X_train.shape[0]
        
        # 数据增强(仅TS-TCC的自监督模式需要)
        self.aug1 = None
        self.aug2 = None
        
        if training_mode == "self_supervised" and model_name == "TS-TCC":
            from data.augmentations import DataTransform
            self.aug1, self.aug2 = DataTransform(self.x_data, config)
    
    def __getitem__(self, index):
        """获取单个样本"""
        if self.training_mode == "self_supervised" and self.model_name == "TS-TCC":
            # TS-TCC自监督需要增强数据
            return (
                self.x_data[index], 
                self.y_data[index], 
                self.aug1[index], 
                self.aug2[index]
            )
        else:
            # 其他情况返回原始数据(TSLANet会在模型内部处理增强)
            return (
                self.x_data[index], 
                self.y_data[index], 
                self.x_data[index], 
                self.x_data[index]
            )
    
    def __len__(self):
        return self.len


def calculate_padding(seq_len, patch_size):
    """计算所需的填充长度"""
    padding = patch_size - (seq_len % patch_size) if seq_len % patch_size != 0 else 0
    return padding


def zero_pad_sequence(input_tensor, pad_length):
    """零填充序列"""
    return torch.nn.functional.pad(input_tensor, (0, pad_length))


def get_dataloader(data_path, config, training_mode, model_name):
    """
    获取数据加载器
    
    Args:
        data_path: 数据集路径
        config: 配置字典
        training_mode: 训练模式
        model_name: 模型名称
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # 加载数据文件
    train_file = torch.load(os.path.join(data_path, "train.pt"))
    val_file = torch.load(os.path.join(data_path, "val.pt"))
    test_file = torch.load(os.path.join(data_path, "test.pt"))
    
    # TSLANet需要填充到patch_size的倍数
    if model_name == "TSLANet":
        patch_size = config.get('patch_size', 8)
        seq_len = train_file["samples"].shape[-1]
        required_padding = calculate_padding(seq_len, patch_size)
        
        if required_padding != 0:
            train_file["samples"] = zero_pad_sequence(train_file["samples"], required_padding)
            val_file["samples"] = zero_pad_sequence(val_file["samples"], required_padding)
            test_file["samples"] = zero_pad_sequence(test_file["samples"], required_padding)
    
    # 创建数据集
    train_dataset = UnifiedDataset(train_file, config, training_mode, model_name)
    val_dataset = UnifiedDataset(val_file, config, training_mode, model_name)
    test_dataset = UnifiedDataset(test_file, config, training_mode, model_name)
    
    # 获取批次大小和drop_last设置
    batch_size = config.get('batch_size', 16)
    drop_last = config.get('drop_last', True)
    
    # 调整批次大小(如果数据集太小)
    num_samples = train_dataset.x_data.shape[0]
    if num_samples < batch_size:
        batch_size = num_samples // 4
    
    # 创建数据加载器
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        num_workers=0
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=drop_last,
        num_workers=0
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )
    
    return train_loader, val_loader, test_loader