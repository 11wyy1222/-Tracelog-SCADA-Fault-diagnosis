from trainers.unified_trainer import UnifiedTrainer
import torch
import numpy as np

class RCTrainer(UnifiedTrainer):
    """RC模型专用训练器"""
    
    @staticmethod
    def _unpack_batch(batch):
        """通用的 batch 解包函数"""
        if isinstance(batch, (list, tuple)):
            if len(batch) == 2:
                x, y = batch
            elif len(batch) == 3:
                x, y, _ = batch
            elif len(batch) == 4:
                x, y, _, _ = batch
            else:
                raise ValueError(f"不支持的 batch 格式，长度为 {len(batch)}")
        else:
            raise ValueError(f"不支持的 batch 类型: {type(batch)}")
        
        return x, y
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        all_x = []
        all_y = []
        
        for batch in train_loader:
            x, y = self._unpack_batch(batch)
            
            if isinstance(x, torch.Tensor):
                x = x.cpu()
            if isinstance(y, torch.Tensor):
                y = y.cpu()
            
            all_x.append(x)
            all_y.append(y)
        
        x_all = torch.cat(all_x, dim=0) if isinstance(all_x[0], torch.Tensor) else np.concatenate(all_x, axis=0)
        y_all = torch.cat(all_y, dim=0) if isinstance(all_y[0], torch.Tensor) else np.concatenate(all_y, axis=0)
        
        if not isinstance(x_all, torch.Tensor):
            x_all = torch.from_numpy(x_all).float()
        if not isinstance(y_all, torch.Tensor):
            y_all = torch.from_numpy(y_all).float()
        
        self.logger.debug(f"训练数据形状: X={x_all.shape}, Y={y_all.shape}")
        
        self.model.fit(x_all, y_all)
        
        return 0.0, 0.0
    
    def evaluate(self, val_loader):
        """
        验证/测试一个epoch - 重写以处理RC模型的特殊情况
        
        Returns:
            loss, acc, predictions, labels
        """
        self.model.eval()
        all_x = []
        all_y = []
        all_pred = []
        
        with torch.no_grad():
            for batch in val_loader:
                x, y = self._unpack_batch(batch)
                
                if isinstance(x, torch.Tensor):
                    x = x.cpu()
                if isinstance(y, torch.Tensor):
                    y = y.cpu()
                
                all_x.append(x)
                all_y.append(y)
                
                # 预测
                pred = self.model(x)
                all_pred.append(pred)
        
        # 合并数据
        x_all = torch.cat(all_x, dim=0) if isinstance(all_x[0], torch.Tensor) else np.concatenate(all_x, axis=0)
        y_all = torch.cat(all_y, dim=0) if isinstance(all_y[0], torch.Tensor) else np.concatenate(all_y, axis=0)
        pred_all = torch.cat(all_pred, dim=0)
        
        # 转换为tensor
        if not isinstance(x_all, torch.Tensor):
            x_all = torch.from_numpy(x_all).float()
        if not isinstance(y_all, torch.Tensor):
            y_all = torch.from_numpy(y_all).float()
        
        self.logger.debug(f"[DEBUG] y_all shape: {y_all.shape}, dtype: {y_all.dtype}")
        self.logger.debug(f"[DEBUG] pred_all shape: {pred_all.shape}, dtype: {pred_all.dtype}")
        
        # 处理标签格式 - 转换为类别索引
        if len(y_all.shape) > 1 and y_all.shape[1] > 1:
            # one-hot 编码 -> 类别索引
            true_labels = torch.argmax(y_all, dim=1).long()
        else:
            # 已经是类别索引或单列
            true_labels = y_all.long().squeeze()
        
        self.logger.debug(f"[DEBUG] true_labels shape: {true_labels.shape}, dtype: {true_labels.dtype}")
        self.logger.debug(f"[DEBUG] true_labels range: [{true_labels.min()}, {true_labels.max()}]")
        
        # 处理预测格式
        if isinstance(pred_all, np.ndarray):
            pred_all = torch.from_numpy(pred_all).float()
        
        pred_all = pred_all.float()
        
        self.logger.debug(f"[DEBUG] pred_all after conversion: shape={pred_all.shape}, dtype={pred_all.dtype}")
        
        # 获取预测的类别
        if len(pred_all.shape) > 1 and pred_all.shape[1] > 1:
            # 多类分类 - logits 或概率
            pred_labels = torch.argmax(pred_all, dim=1).long()
        else:
            # 一维输出 - 转换为类别索引
            if len(pred_all.shape) == 1:
                pred_labels = (pred_all > 0.5).long()
            else:
                pred_labels = pred_all.long().squeeze()
        
        self.logger.debug(f"[DEBUG] pred_labels shape: {pred_labels.shape}, range: [{pred_labels.min()}, {pred_labels.max()}]")
        
        # 计算准确率
        acc = (pred_labels == true_labels).float().mean().item()
        
        # 计算损失
        try:
            # 确保 pred_all 是 2D 的 logits，true_labels 是 1D 的类别索引
            if len(pred_all.shape) == 1:
                # 如果是一维的，转换为二维
                num_classes = true_labels.max().item() + 1
                pred_all_2d = torch.zeros(len(pred_all), num_classes, dtype=torch.float32)
                pred_all_2d.scatter_(1, pred_labels.unsqueeze(1), 1.0)
                pred_all = pred_all_2d
            
            loss = self.criterion(pred_all, true_labels)
        except RuntimeError as e:
            self.logger.warning(f"损失计算失败: {e}")
            self.logger.warning(f"pred_all shape: {pred_all.shape}, true_labels shape: {true_labels.shape}")
            loss = torch.tensor(1.0 - acc)
        
        return loss.item(), acc, pred_labels.numpy(), true_labels.numpy()
    
    def val_epoch(self, val_loader):
        """验证一个epoch"""
        loss, acc, _, _ = self.evaluate(val_loader)
        return loss, acc
    
    def test_epoch(self, test_loader):
        """测试一个epoch"""
        loss, acc, predictions, labels = self.evaluate(test_loader)
        return loss, acc, predictions, labels