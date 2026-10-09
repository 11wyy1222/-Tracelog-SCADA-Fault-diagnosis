"""
统一训练器 - 支持TSLANet和TS-TCC
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.loss import NTXentLoss


class UnifiedTrainer:
    """统一的训练器,支持多种模型和训练模式"""
    
    def __init__(
        self, 
        model, 
        auxiliary_model,
        model_optimizer,
        aux_optimizer,
        device,
        logger,
        config,
        experiment_log_dir,
        training_mode,
        model_name
    ):
        """
        初始化统一训练器
        
        Args:
            model: 主模型
            auxiliary_model: 辅助模型(TSLANet为None, TS-TCC为TC)
            model_optimizer: 模型优化器
            aux_optimizer: 辅助模型优化器
            device: 运行设备
            logger: 日志记录器
            config: 配置对象/字典
            experiment_log_dir: 实验日志目录
            training_mode: 训练模式
            model_name: 模型名称
        """
        self.model = model
        self.auxiliary_model = auxiliary_model
        self.model_optimizer = model_optimizer
        self.aux_optimizer = aux_optimizer
        self.device = device
        self.logger = logger
        self.config = config
        self.experiment_log_dir = experiment_log_dir
        self.training_mode = training_mode
        self.model_name = model_name
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
       # 学习率调度器（仅当optimizer存在时创建）
        if model_optimizer is not None:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                model_optimizer, 'min'
            )
        else:
            self.scheduler = None
    
    def _get_config(self, key, default):
        """从config中获取参数"""
        if isinstance(self.config, dict):
            return self.config.get(key, default)
        else:
            return getattr(self.config, key, default)
    
    def train(self, train_loader, val_loader, test_loader):
        """训练流程"""
        
        num_epochs = self.config['num_epoch']
        best_acc = 0
        best_epoch = 0
        
        for epoch in range(num_epochs):
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_acc, _, _ = self.evaluate(val_loader)
            
            # 学习率调度
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
            
            # 打印日志
            self.logger.debug(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
            )
            
            # 保存最佳模型
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                self.save_checkpoint(epoch=epoch, is_best=True)
        
        # 保存最后一个检查点
        self.save_checkpoint(epoch=epoch)
        
        self.logger.debug(f"\n最佳模型: Epoch {best_epoch+1}, 准确率: {best_acc:.4f}")
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        if self.auxiliary_model is not None:
            self.auxiliary_model.train()
        
        total_loss = []
        total_acc = []
        
        for batch_idx, batch in enumerate(train_loader):
            # 解析批次数据
            data, labels, aug1, aug2 = self._parse_batch(batch)
            
            # 发送到设备
            data = data.float().to(self.device)
            labels = labels.long().to(self.device)
            aug1 = aug1.float().to(self.device)
            aug2 = aug2.float().to(self.device)
            
            # 检查数据有效性
            if torch.isnan(data).any() or torch.isinf(data).any():
                self.logger.debug(f"警告: 批次 {batch_idx} 包含NaN/Inf,跳过")
                continue
            
            # 梯度清零
            self.model_optimizer.zero_grad()
            if self.aux_optimizer is not None:
                self.aux_optimizer.zero_grad()
            
            # 根据模型类型和训练模式计算损失
            loss = self._compute_loss(
                data, labels, aug1, aug2, batch_idx
            )
            
            if loss is None:
                continue
            
            # 反向传播
            total_loss.append(loss.item())
            loss.backward()
            self.model_optimizer.step()
            if self.aux_optimizer is not None:
                self.aux_optimizer.step()
            
            # 计算准确率(监督学习)
            if self.training_mode != "self_supervised":
                output = self.model(data)
                predictions = output[0] if isinstance(output, tuple) else output
                acc = labels.eq(predictions.detach().argmax(dim=1)).float().mean()
                total_acc.append(acc.item())
        
        # 计算平均损失和准确率
        avg_loss = np.mean(total_loss) if total_loss else 0
        avg_acc = np.mean(total_acc) if total_acc else 0
        
        return avg_loss, avg_acc
    
    def _compute_loss(self, data, labels, aug1, aug2, batch_idx):
        """根据模型类型和训练模式计算损失"""
        
        if self.training_mode == "self_supervised":
            # 自监督训练
            if self.model_name == 'TSLANet':
                return self._compute_tslanet_pretrain_loss(aug1)
            elif self.model_name == 'TS-TCC':
                return self._compute_tstcc_selfsup_loss(aug1, aug2)
        else:
            # 监督训练
            output = self.model(data)
            predictions = output[0] if isinstance(output, tuple) else output
            loss = self.criterion(predictions, labels)
            
            # 检查损失有效性
            if torch.isnan(loss) or torch.isinf(loss):
                self.logger.debug(f"警告: 批次 {batch_idx} 损失为NaN/Inf,跳过")
                return None
            
            return loss
    
    def _compute_tslanet_pretrain_loss(self, data):
        """计算TSLANet预训练损失(掩码重构)"""
        masking_ratio = self._get_config('masking_ratio', 0.4)
        
        # 预训练前向传播
        preds, target, mask = self.model.pretrain_forward(data, masking_ratio)
        
        # 计算重构损失
        loss = (preds - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        
        return loss
    
    def _compute_tstcc_selfsup_loss(self, aug1, aug2):
        """计算TS-TCC自监督损失"""
        # 前向传播
        predictions1, features1 = self.model(aug1)
        predictions2, features2 = self.model(aug2)
        
        # 归一化特征
        features1 = F.normalize(features1, dim=1)
        features2 = F.normalize(features2, dim=1)
        
        # 时间对比损失
        temp_cont_loss1, temp_cont_lstm_feat1 = self.auxiliary_model(features1, features2)
        temp_cont_loss2, temp_cont_lstm_feat2 = self.auxiliary_model(features2, features1)
        
        # 归一化投影特征
        zis = temp_cont_lstm_feat1
        zjs = temp_cont_lstm_feat2
        
        # Context对比损失
        lambda1 = 1
        lambda2 = 0.7
        
        batch_size = self._get_config('batch_size', 16)
        
        # 获取Context_Cont配置
        if isinstance(self.config, dict):
            context_config = self.config.get('Context_Cont', {})
            if isinstance(context_config, dict):
                temperature = context_config.get('temperature', 0.2)
                use_cosine = context_config.get('use_cosine_similarity', True)
            else:
                temperature = getattr(context_config, 'temperature', 0.2)
                use_cosine = getattr(context_config, 'use_cosine_similarity', True)
        else:
            temperature = getattr(self.config.Context_Cont, 'temperature', 0.2)
            use_cosine = getattr(self.config.Context_Cont, 'use_cosine_similarity', True)
        
        nt_xent_criterion = NTXentLoss(
            self.device, batch_size, temperature, use_cosine
        )
        
        loss = (temp_cont_loss1 + temp_cont_loss2) * lambda1 + \
               nt_xent_criterion(zis, zjs) * lambda2
        
        return loss
    
    def evaluate(self, data_loader):
        """评估模型"""
        self.model.eval()
        if self.auxiliary_model is not None:
            self.auxiliary_model.eval()
        
        total_loss = []
        total_acc = []
        outs = np.array([])
        trgs = np.array([])
        
        with torch.no_grad():
            for batch in data_loader:
                data, labels, _, _ = self._parse_batch(batch)
                data = data.float().to(self.device)
                labels = labels.long().to(self.device)
                
                if self.training_mode == "self_supervised":
                    continue
                
                output = self.model(data)
                predictions = output[0] if isinstance(output, tuple) else output
                
                # 计算损失和准确率
                loss = self.criterion(predictions, labels)
                acc = labels.eq(predictions.detach().argmax(dim=1)).float().mean()
                
                total_loss.append(loss.item())
                total_acc.append(acc.item())
                
                # 收集预测结果
                pred = predictions.max(1, keepdim=True)[1]
                outs = np.append(outs, pred.cpu().numpy())
                trgs = np.append(trgs, labels.cpu().numpy())
        
        if self.training_mode == "self_supervised":
            return 0, 0, [], []
        
        avg_loss = np.mean(total_loss) if total_loss else 0
        avg_acc = np.mean(total_acc) if total_acc else 0
        
        return avg_loss, avg_acc, outs, trgs
    
    def _parse_batch(self, batch):
        """解析批次数据"""
        if len(batch) == 4:
            return batch[0], batch[1], batch[2], batch[3]
        elif len(batch) == 2:
            # 只有数据和标签
            return batch[0], batch[1], batch[0], batch[0]
        else:
            raise ValueError(f"不支持的批次格式,长度为 {len(batch)}")
    
    def save_checkpoint(self, save_dir='saved_models', epoch=None, is_best=False):
        """
        保存检查点
        
        Args:
            save_dir: 保存目录
            epoch: epoch号（可选）
            is_best: 是否为最佳模型
        """
        checkpoint_dir = os.path.join(self.experiment_log_dir, save_dir)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 对于RC模型使用特殊保存方式
        if self.model_name == 'RC':
            if epoch is not None:
                checkpoint_path = os.path.join(checkpoint_dir, f'ckp_epoch_{epoch}.pt')
            else:
                checkpoint_path = os.path.join(checkpoint_dir, 'ckp_last.pt')
            self.model.save_checkpoint(checkpoint_path)
            
            if is_best:
                best_path = os.path.join(checkpoint_dir, 'ckp_best.pt')
                self.model.save_checkpoint(best_path)
        else:
            # 标准PyTorch模型
            if self.model_optimizer is None:
                # 某些模型没有optimizer
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                }
            else:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.model_optimizer.state_dict(),
                }
            
            if epoch is not None:
                checkpoint_path = os.path.join(checkpoint_dir, f'ckp_epoch_{epoch}.pt')
            else:
                checkpoint_path = os.path.join(checkpoint_dir, 'ckp_last.pt')
            
            torch.save(checkpoint, checkpoint_path)
            
            if is_best:
                best_path = os.path.join(checkpoint_dir, 'ckp_best.pt')
                torch.save(checkpoint, best_path)
        
        self.logger.debug(f"检查点已保存: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"检查点不存在: {checkpoint_path}")
        
        # 对于RC模型使用特殊加载方式
        if self.model_name == 'RC':
            self.model.load_checkpoint(checkpoint_path)
        else:
            # 标准PyTorch模型
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            if self.model_optimizer and 'optimizer_state_dict' in checkpoint:
                self.model_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.logger.debug(f"检查点已加载: {checkpoint_path}")