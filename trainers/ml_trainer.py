"""
传统机器学习模型训练器
使用 ModelFactory 创建的 MLBaseWrapper 模型
"""
import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score


class MLTrainer:
    """
    传统机器学习模型训练器
    
    使用 ModelFactory 创建的 MLBaseWrapper 模型进行训练
    MLBaseWrapper 已包含：StandardScaler、标签重映射、NaN处理
    """
    
    def __init__(self, model, auxiliary_model, model_optimizer, aux_optimizer,
                 device, logger, config, experiment_log_dir, training_mode, model_name):
        """
        Args:
            model: MLBaseWrapper 实例（由 ModelFactory 创建）
            auxiliary_model: 辅助模型（ML模型不使用）
            model_optimizer: 优化器（ML模型不使用）
            aux_optimizer: 辅助优化器（ML模型不使用）
            device: 设备
            logger: 日志记录器
            config: 模型配置
            experiment_log_dir: 实验日志目录
            training_mode: 训练模式
            model_name: 模型名称
        """
        self.model = model  # MLBaseWrapper 实例
        self.model_name = model_name
        self.logger = logger
        self.config = config
        self.experiment_log_dir = experiment_log_dir
        self.device = device
        self.training_mode = training_mode
    
    @staticmethod
    def _unpack_batch(batch):
        """通用的 batch 解包函数"""
        if isinstance(batch, (list, tuple)):
            if len(batch) >= 2:
                return batch[0], batch[1]
        raise ValueError(f"不支持的 batch 格式: {type(batch)}")
    
    def _collect_data(self, data_loader) -> tuple:
        """从 DataLoader 收集所有数据"""
        all_x, all_y = [], []
        
        for batch in data_loader:
            x, y = self._unpack_batch(batch)
            
            # 转换为 tensor
            if not isinstance(x, torch.Tensor):
                x = torch.from_numpy(x).float()
            if not isinstance(y, torch.Tensor):
                y = torch.from_numpy(y).long()
            
            all_x.append(x)
            all_y.append(y)
        
        X = torch.cat(all_x, dim=0)
        y = torch.cat(all_y, dim=0)
        
        return X, y
    
    def train(self, train_loader, val_loader, test_loader):
        """训练模型"""
        self.logger.info(f"开始训练 {self.model_name}...")
        
        # 收集训练数据
        X_train, y_train = self._collect_data(train_loader)
        self.logger.info(f"训练数据: X={X_train.shape}, y={y_train.shape}")
        
        # 打印类别分布
        y_np = y_train.numpy()
        unique, counts = np.unique(y_np, return_counts=True)
        self.logger.info(f"类别分布: {dict(zip(unique, counts))}")
        
        # 使用 MLBaseWrapper 的 fit 方法训练
        # MLBaseWrapper.fit() 已包含：展平、NaN处理、标准化、标签重映射
        self.model.fit(X_train, y_train)
        
        self.logger.info(f"✓ {self.model_name} 训练完成")
        
        # 保存模型
        self._save_model()
    
    def train_epoch(self, train_loader):
        """兼容接口：训练一个 epoch（ML模型实际只训练一次）"""
        X_train, y_train = self._collect_data(train_loader)
        self.logger.debug(f"训练数据形状: X={X_train.shape}, Y={y_train.shape}")
        
        # 使用 MLBaseWrapper 的 fit 方法
        self.model.fit(X_train, y_train)
        
        return 0.0, 0.0  # ML模型无迭代loss
    
    def evaluate(self, data_loader) -> tuple:
        """评估模型"""
        X, y_true = self._collect_data(data_loader)
        
        # 使用 MLBaseWrapper 的 forward 方法预测
        # forward() 已包含：展平、NaN处理、标准化、标签还原
        y_pred = self.model(X)
        
        # 转换为 numpy
        y_true_np = y_true.numpy()
        y_pred_np = y_pred.numpy()
        
        # 计算指标
        acc = accuracy_score(y_true_np, y_pred_np)
        
        self.logger.debug(f"[DEBUG] true_labels: {y_true_np[:10].tolist()}...")
        self.logger.debug(f"[DEBUG] pred_labels: {y_pred_np[:10].tolist()}...")
        
        # 伪损失
        loss = 1.0 - acc
        
        return loss, acc, y_pred_np, y_true_np
    
    def val_epoch(self, val_loader):
        """验证一个 epoch"""
        loss, acc, _, _ = self.evaluate(val_loader)
        return loss, acc
    
    def test_epoch(self, test_loader):
        """测试一个 epoch"""
        return self.evaluate(test_loader)
    
    def _save_model(self):
        """保存模型"""
        save_dir = os.path.join(self.experiment_log_dir, 'saved_models')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'ckp_last.pt')
        
        # 使用 MLBaseWrapper 的 save_checkpoint 方法
        self.model.save_checkpoint(save_path)
        self.logger.info(f"✓ 模型已保存: {save_path}")
