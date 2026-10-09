"""
评估指标模块
提供各种分类和回归指标的计算
"""
import numpy as np
import os
import pandas as pd
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score
)
import matplotlib
matplotlib.use('Agg')  # 非GUI后端
import matplotlib.pyplot as plt
import seaborn as sns
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def calculate_metrics(predictions, targets, experiment_log_dir, home_path):
    """
    计算并保存分类指标
    
    Args:
        predictions: 预测标签
        targets: 真实标签
        experiment_log_dir: 实验日志目录
        home_path: 项目根目录
    """
    # 确保是numpy数组
    if not isinstance(predictions, np.ndarray):
        predictions = np.array(predictions)
    if not isinstance(targets, np.ndarray):
        targets = np.array(targets)
    
    # 基本指标
    accuracy = accuracy_score(targets, predictions)
    
    # 多分类指标
    num_classes = len(np.unique(targets))
    
    if num_classes == 2:
        # 二分类
        precision = precision_score(targets, predictions, average='binary')
        recall = recall_score(targets, predictions, average='binary')
        f1 = f1_score(targets, predictions, average='binary')
    else:
        # 多分类
        precision = precision_score(targets, predictions, average='macro')
        recall = recall_score(targets, predictions, average='macro')
        f1 = f1_score(targets, predictions, average='macro')
    
    # 打印基本指标
    print("\n" + "="*60)
    print("评估指标")
    print("="*60)
    print(f"准确率 (Accuracy):  {accuracy:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall):    {recall:.4f}")
    print(f"F1分数 (F1-Score):  {f1:.4f}")
    print("="*60)
    
    # 生成详细报告
    report = classification_report(
        targets, 
        predictions, 
        output_dict=True,
        zero_division=0
    )
    
    # 保存为DataFrame
    df_report = pd.DataFrame(report).transpose()
    
    # 保存到Excel
    os.makedirs(experiment_log_dir, exist_ok=True)
    report_path = os.path.join(experiment_log_dir, 'classification_report.xlsx')
    df_report.to_excel(report_path)
    print(f"\n✓ 分类报告已保存至: {report_path}")
    
    # 保存混淆矩阵
    cm = confusion_matrix(targets, predictions)
    save_confusion_matrix(cm, experiment_log_dir)
    
    # 保存指标到CSV
    metrics_dict = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    metrics_df = pd.DataFrame([metrics_dict])
    metrics_path = os.path.join(experiment_log_dir, 'metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"✓ 指标已保存至: {metrics_path}")
    
    return metrics_dict


def save_confusion_matrix(cm, save_dir, class_names=None):
    """
    保存混淆矩阵图像
    
    Args:
        cm: 混淆矩阵
        save_dir: 保存目录
        class_names: 类别名称列表
    """
    plt.figure(figsize=(10, 8))
    
    # 归一化混淆矩阵
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # 绘制热图
    sns.heatmap(
        cm_normalized, 
        annot=True, 
        fmt='.2f', 
        cmap='Blues',
        xticklabels=class_names if class_names else range(cm.shape[0]),
        yticklabels=class_names if class_names else range(cm.shape[0])
    )
    
    plt.title('归一化混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    
    # 保存
    save_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 混淆矩阵已保存至: {save_path}")


def calculate_per_class_metrics(predictions, targets, class_names=None):
    """
    计算每个类别的详细指标
    
    Args:
        predictions: 预测标签
        targets: 真实标签
        class_names: 类别名称列表
        
    Returns:
        per_class_metrics: 每个类别的指标字典
    """
    num_classes = len(np.unique(targets))
    
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(num_classes)]
    
    per_class_metrics = {}
    
    for i in range(num_classes):
        # 创建二分类掩码
        binary_targets = (targets == i).astype(int)
        binary_predictions = (predictions == i).astype(int)
        
        # 计算指标
        precision = precision_score(binary_targets, binary_predictions, zero_division=0)
        recall = recall_score(binary_targets, binary_predictions, zero_division=0)
        f1 = f1_score(binary_targets, binary_predictions, zero_division=0)
        
        # 支持度(样本数)
        support = np.sum(binary_targets)
        
        per_class_metrics[class_names[i]] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': support
        }
    
    return per_class_metrics


def calculate_top_k_accuracy(predictions_probs, targets, k=5):
    """
    计算Top-K准确率
    
    Args:
        predictions_probs: 预测概率 [N, num_classes]
        targets: 真实标签 [N]
        k: Top-K的K值
        
    Returns:
        top_k_acc: Top-K准确率
    """
    # 获取top-k预测
    top_k_predictions = np.argsort(predictions_probs, axis=1)[:, -k:]
    
    # 检查真实标签是否在top-k中
    correct = np.array([targets[i] in top_k_predictions[i] for i in range(len(targets))])
    
    top_k_acc = np.mean(correct)
    
    return top_k_acc


def calculate_macro_micro_metrics(predictions, targets):
    """
    计算宏平均和微平均指标
    
    Args:
        predictions: 预测标签
        targets: 真实标签
        
    Returns:
        metrics: 包含宏平均和微平均指标的字典
    """
    metrics = {
        'macro': {
            'precision': precision_score(targets, predictions, average='macro', zero_division=0),
            'recall': recall_score(targets, predictions, average='macro', zero_division=0),
            'f1_score': f1_score(targets, predictions, average='macro', zero_division=0)
        },
        'micro': {
            'precision': precision_score(targets, predictions, average='micro', zero_division=0),
            'recall': recall_score(targets, predictions, average='micro', zero_division=0),
            'f1_score': f1_score(targets, predictions, average='micro', zero_division=0)
        },
        'weighted': {
            'precision': precision_score(targets, predictions, average='weighted', zero_division=0),
            'recall': recall_score(targets, predictions, average='weighted', zero_division=0),
            'f1_score': f1_score(targets, predictions, average='weighted', zero_division=0)
        }
    }
    
    return metrics


def plot_training_history(history, save_dir):
    """
    绘制训练历史曲线
    
    Args:
        history: 训练历史字典,包含loss和acc
        save_dir: 保存目录
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # 损失曲线
    if 'train_loss' in history and 'val_loss' in history:
        axes[0].plot(history['train_loss'], label='训练损失')
        axes[0].plot(history['val_loss'], label='验证损失')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('损失')
        axes[0].set_title('训练和验证损失')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    # 准确率曲线
    if 'train_acc' in history and 'val_acc' in history:
        axes[1].plot(history['train_acc'], label='训练准确率')
        axes[1].plot(history['val_acc'], label='验证准确率')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('准确率')
        axes[1].set_title('训练和验证准确率')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存
    save_path = os.path.join(save_dir, 'training_history.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 训练历史曲线已保存至: {save_path}")


class MetricsTracker:
    """训练指标跟踪器"""
    
    def __init__(self):
        """初始化指标跟踪器"""
        self.metrics = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
        self.best_val_acc = 0.0
        self.best_epoch = 0
    
    def update(self, epoch, train_loss, train_acc, val_loss, val_acc, lr=None):
        """
        更新指标
        
        Args:
            epoch: 当前轮次
            train_loss: 训练损失
            train_acc: 训练准确率
            val_loss: 验证损失
            val_acc: 验证准确率
            lr: 学习率
        """
        self.metrics['train_loss'].append(train_loss)
        self.metrics['train_acc'].append(train_acc)
        self.metrics['val_loss'].append(val_loss)
        self.metrics['val_acc'].append(val_acc)
        
        if lr is not None:
            self.metrics['learning_rate'].append(lr)
        
        # 更新最佳指标
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_epoch = epoch
    
    def get_best_metrics(self):
        """获取最佳指标"""
        return {
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch
        }
    
    def save(self, save_dir):
        """
        保存指标到文件
        
        Args:
            save_dir: 保存目录
        """
        # 保存为CSV
        df = pd.DataFrame(self.metrics)
        csv_path = os.path.join(save_dir, 'training_metrics.csv')
        df.to_csv(csv_path, index=False)
        
        # 绘制曲线
        plot_training_history(self.metrics, save_dir)
        
        print(f"✓ 训练指标已保存至: {save_dir}")
    
    def __repr__(self):
        """打印指标信息"""
        return (
            f"MetricsTracker(\n"
            f"  最佳验证准确率: {self.best_val_acc:.4f} (Epoch {self.best_epoch})\n"
            f"  总轮次: {len(self.metrics['train_loss'])}\n"
            f")"
        )


# 兼容旧版本的函数别名
_calc_metrics = calculate_metrics