"""
批量训练传统机器学习模型脚本
同时训练多个ML模型并比较结果

使用方法:
    python train_all_ml.py --config configs/config.yaml
    python train_all_ml.py --config configs/config.yaml --models RandomForest XGBoost LightGBM
    python train_all_ml.py --config configs/config.yaml --cv 5
"""
import torch
import os
import numpy as np
from datetime import datetime
import argparse
import yaml
import pandas as pd
from tabulate import tabulate

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score

from utils.logger import _logger
from data.unified_dataloader import get_dataloader
from trainers.ml_trainer import MLTrainer
from models.model_factory import ModelFactory, get_model_config
from utils.metrics import calculate_metrics


# 支持的传统ML模型列表
ML_MODELS = ['RandomForest', 'ExtraTrees', 'XGBoost', 'LightGBM', 'SVM', 'KNN', 'LogisticRegression']


def parse_args():
    parser = argparse.ArgumentParser(description='批量训练传统ML模型')
    parser.add_argument('--config', default='configs/config.yaml', type=str,
                        help='YAML配置文件路径')
    parser.add_argument('--models', nargs='+', default=ML_MODELS,
                        help=f'要训练的模型列表，可选: {ML_MODELS}')
    parser.add_argument('--dataset', default=None, type=str,
                        help='数据集名称')
    parser.add_argument('--seed', default=None, type=int,
                        help='随机种子')
    parser.add_argument('--cv', default=5, type=int,
                        help='交叉验证折数')
    return parser.parse_args()


def get_param(cmd_arg, config_path, config_dict, default=None):
    """参数优先级: 命令行 > 配置文件 > 默认值"""
    if cmd_arg is not None:
        return cmd_arg
    keys = config_path.split('.')
    value = config_dict
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value if value is not None else default


def extract_xy_from_loader(data_loader, loader_name="unknown", logger=None):
    """
    从 DataLoader 中提取全部特征和标签
    兼容 MLTrainer._unpack_batch 的输入格式：(x, y)
    """
    X_list = []
    y_list = []
    batch_count = 0

    if data_loader is None:
        if logger:
            logger.warning(f"{loader_name} loader is None")
        return None, None

    for batch in data_loader:
        batch_count += 1

        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            x, y = batch[0], batch[1]
        else:
            raise ValueError(f"{loader_name} 不支持的 batch 格式: {type(batch)}")

        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        else:
            x = np.array(x)

        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        else:
            y = np.array(y)

        X_list.append(x)
        y_list.append(y)

    if logger:
        logger.info(f"{loader_name} loader 实际遍历到 batch 数: {batch_count}")

    if len(X_list) == 0:
        if logger:
            logger.warning(f"{loader_name} loader 可迭代结果为空")
        return None, None

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0).reshape(-1)

    if logger:
        logger.info(f"{loader_name} 提取后数据大小: X={X.shape}, y={y.shape}")

    return X, y


def load_xy_from_full_dataset(data_path, logger=None):
    """
    优先从 full_dataset.pt 读取完整数据集
    支持字段:
        - samples / labels
        - X / y
    """
    full_dataset_path = os.path.join(data_path, 'full_dataset.pt')

    if not os.path.exists(full_dataset_path):
        if logger:
            logger.info(f"未找到完整数据集文件: {full_dataset_path}")
        return None, None

    if logger:
        logger.info(f"优先读取完整数据集: {full_dataset_path}")

    data = torch.load(full_dataset_path, map_location='cpu')

    if not isinstance(data, dict):
        raise ValueError(f"full_dataset.pt 格式错误，期望 dict，实际: {type(data)}")

    if 'samples' in data and 'labels' in data:
        X = data['samples']
        y = data['labels']
    elif 'X' in data and 'y' in data:
        X = data['X']
        y = data['y']
    else:
        raise ValueError("full_dataset.pt 缺少 samples/labels 或 X/y 字段")

    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    else:
        X = np.array(X)

    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    else:
        y = np.array(y)

    y = y.reshape(-1)

    if logger:
        logger.info(f"完整数据集读取成功: X={X.shape}, y={y.shape}")

    return X, y


def load_xy_from_original_loaders(data_path, model_config, model_name, logger=None):
    """
    回退方案：从原始 train/val/test loader 中提取并合并
    为避免一次性 loader 被消费，分三次重新获取
    """
    if logger:
        logger.info("回退到原有方式：从 train/val/test loader 提取并合并完整数据集")

    train_loader, _, _ = get_dataloader(
        data_path, model_config, 'supervised', model_name
    )
    X_train, y_train = extract_xy_from_loader(train_loader, "train", logger)

    _, val_loader, _ = get_dataloader(
        data_path, model_config, 'supervised', model_name
    )
    X_val, y_val = extract_xy_from_loader(val_loader, "val", logger)

    _, _, test_loader = get_dataloader(
        data_path, model_config, 'supervised', model_name
    )
    X_test, y_test = extract_xy_from_loader(test_loader, "test", logger)

    X_parts = []
    y_parts = []

    for loader_name, X_part, y_part in [
        ('train', X_train, y_train),
        ('val', X_val, y_val),
        ('test', X_test, y_test)
    ]:
        if X_part is not None and y_part is not None:
            X_parts.append(X_part)
            y_parts.append(y_part)
        else:
            if logger:
                logger.warning(f"{loader_name} 数据为空，已跳过")

    if len(X_parts) == 0:
        raise ValueError("train/val/test 全部为空，无法构建完整数据集")

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)

    if logger:
        logger.info(f"由原始 loader 合并得到完整数据集: X={X.shape}, y={y.shape}")

    return X, y


class SimpleDataLoader:
    """
    为 MLTrainer 准备的轻量 DataLoader
    只需满足：
        for batch in data_loader:
            x, y = batch
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __iter__(self):
        yield self.X, self.y

    def __len__(self):
        return 1


def train_single_model_cv(model_name, yaml_config, X, y, experiment_log_dir, logger, device, cv=5, seed=42):
    """对单个模型执行分层K折交叉验证"""
    dataset_name = yaml_config.get('dataset', {}).get('selected', 'chaosu')
    model_config = get_model_config(yaml_config, model_name, dataset_name)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"开始训练模型（交叉验证）: {model_name}")
    logger.info(f"{'=' * 60}")

    start_time = datetime.now()
    model_log_dir = os.path.join(experiment_log_dir, model_name)
    os.makedirs(model_log_dir, exist_ok=True)

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)

    fold_results = []
    all_pred_labels = []
    all_true_labels = []

    try:
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
            logger.info(f"\n--- {model_name} | Fold {fold_idx}/{cv} ---")

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            train_loader = SimpleDataLoader(X_train, y_train)
            val_loader = SimpleDataLoader(X_val, y_val)

            model, auxiliary_model = ModelFactory.create_model(model_name, model_config, device)

            fold_log_dir = os.path.join(model_log_dir, f'fold_{fold_idx}')
            os.makedirs(fold_log_dir, exist_ok=True)

            trainer = MLTrainer(
                model=model,
                auxiliary_model=auxiliary_model,
                model_optimizer=None,
                aux_optimizer=None,
                device=device,
                logger=logger,
                config=model_config,
                experiment_log_dir=fold_log_dir,
                training_mode='supervised',
                model_name=model_name
            )

            trainer.train_epoch(train_loader)

            val_loss, val_acc, pred_labels, true_labels = trainer.evaluate(val_loader)

            precision = precision_score(true_labels, pred_labels, average='weighted', zero_division=0)
            recall = recall_score(true_labels, pred_labels, average='weighted', zero_division=0)
            f1 = f1_score(true_labels, pred_labels, average='weighted', zero_division=0)

            fold_result = {
                'fold': fold_idx,
                'accuracy': val_acc,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            fold_results.append(fold_result)

            all_pred_labels.extend(pred_labels.tolist())
            all_true_labels.extend(true_labels.tolist())

            logger.info(
                f"Fold {fold_idx} 结果 - "
                f"Accuracy: {val_acc:.4f}, "
                f"Precision: {precision:.4f}, "
                f"Recall: {recall:.4f}, "
                f"F1: {f1:.4f}"
            )

            calculate_metrics(pred_labels, true_labels, fold_log_dir, os.getcwd())
            trainer._save_model()

        fold_df = pd.DataFrame(fold_results)
        fold_csv_path = os.path.join(model_log_dir, 'cv_fold_results.csv')
        fold_df.to_csv(fold_csv_path, index=False)

        all_pred_labels = np.array(all_pred_labels)
        all_true_labels = np.array(all_true_labels)

        acc_mean = fold_df['accuracy'].mean()
        acc_std = fold_df['accuracy'].std(ddof=1) if len(fold_df) > 1 else 0.0
        precision_mean = fold_df['precision'].mean()
        precision_std = fold_df['precision'].std(ddof=1) if len(fold_df) > 1 else 0.0
        recall_mean = fold_df['recall'].mean()
        recall_std = fold_df['recall'].std(ddof=1) if len(fold_df) > 1 else 0.0
        f1_mean = fold_df['f1_score'].mean()
        f1_std = fold_df['f1_score'].std(ddof=1) if len(fold_df) > 1 else 0.0

        calculate_metrics(all_pred_labels, all_true_labels, model_log_dir, os.getcwd())

        train_time = (datetime.now() - start_time).total_seconds()

        result = {
            'model': model_name,
            'accuracy': acc_mean,
            'accuracy_std': acc_std,
            'precision': precision_mean,
            'precision_std': precision_std,
            'recall': recall_mean,
            'recall_std': recall_std,
            'f1_score': f1_mean,
            'f1_std': f1_std,
            'train_time': train_time,
            'status': 'success'
        }

        logger.info(f"\n✓ {model_name} 交叉验证完成")
        logger.info(f"Accuracy:  {acc_mean:.4f} ± {acc_std:.4f}")
        logger.info(f"Precision: {precision_mean:.4f} ± {precision_std:.4f}")
        logger.info(f"Recall:    {recall_mean:.4f} ± {recall_std:.4f}")
        logger.info(f"F1-Score:  {f1_mean:.4f} ± {f1_std:.4f}")
        logger.info(f"耗时: {train_time:.2f}s")

    except Exception as e:
        logger.error(f"✗ {model_name} 训练失败: {str(e)}")
        result = {
            'model': model_name,
            'accuracy': 0,
            'accuracy_std': 0,
            'precision': 0,
            'precision_std': 0,
            'recall': 0,
            'recall_std': 0,
            'f1_score': 0,
            'f1_std': 0,
            'train_time': 0,
            'status': f'failed: {str(e)}'
        }

    return result

def train_single_model_split(model_name, yaml_config, data_path, experiment_log_dir, logger, device):
    """对单个模型执行一次 train/val/test 训练评估。"""
    dataset_name = yaml_config.get('dataset', {}).get('selected', 'chaosu')
    model_config = get_model_config(yaml_config, model_name, dataset_name)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"开始训练模型（train/val/test）: {model_name}")
    logger.info(f"{'=' * 60}")

    start_time = datetime.now()
    model_log_dir = os.path.join(experiment_log_dir, model_name)
    os.makedirs(model_log_dir, exist_ok=True)

    try:
        train_loader, val_loader, test_loader = get_dataloader(
            data_path, model_config, 'supervised', model_name
        )

        model, auxiliary_model = ModelFactory.create_model(
            model_name, model_config, device
        )

        trainer = MLTrainer(
            model=model,
            auxiliary_model=auxiliary_model,
            model_optimizer=None,
            aux_optimizer=None,
            device=device,
            logger=logger,
            config=model_config,
            experiment_log_dir=model_log_dir,
            training_mode='supervised',
            model_name=model_name
        )

        trainer.train(train_loader, val_loader, test_loader)

        _, val_acc, val_pred, val_true = trainer.evaluate(val_loader)
        _, test_acc, test_pred, test_true = trainer.evaluate(test_loader)

        val_precision = precision_score(val_true, val_pred, average='weighted', zero_division=0)
        val_recall = recall_score(val_true, val_pred, average='weighted', zero_division=0)
        val_f1 = f1_score(val_true, val_pred, average='weighted', zero_division=0)

        test_precision = precision_score(test_true, test_pred, average='weighted', zero_division=0)
        test_recall = recall_score(test_true, test_pred, average='weighted', zero_division=0)
        test_f1 = f1_score(test_true, test_pred, average='weighted', zero_division=0)

        calculate_metrics(test_pred, test_true, model_log_dir, os.getcwd())

        train_time = (datetime.now() - start_time).total_seconds()
        result = {
            'model': model_name,
            'accuracy': test_acc,
            'accuracy_std': 0.0,
            'precision': test_precision,
            'precision_std': 0.0,
            'recall': test_recall,
            'recall_std': 0.0,
            'f1_score': test_f1,
            'f1_std': 0.0,
            'train_time': train_time,
            'status': 'success',
            'mode': 'split'
        }

        logger.info(f"\n✓ {model_name} train/val/test 训练完成")
        logger.info(
            f"Val  - Accuracy: {val_acc:.4f}, Precision: {val_precision:.4f}, "
            f"Recall: {val_recall:.4f}, F1: {val_f1:.4f}"
        )
        logger.info(
            f"Test - Accuracy: {test_acc:.4f}, Precision: {test_precision:.4f}, "
            f"Recall: {test_recall:.4f}, F1: {test_f1:.4f}"
        )
        logger.info(f"耗时: {train_time:.2f}s")

    except Exception as e:
        logger.error(f"✗ {model_name} train/val/test 训练失败: {str(e)}")
        result = {
            'model': model_name,
            'accuracy': 0,
            'accuracy_std': 0,
            'precision': 0,
            'precision_std': 0,
            'recall': 0,
            'recall_std': 0,
            'f1_score': 0,
            'f1_std': 0,
            'train_time': 0,
            'status': f'failed: {str(e)}',
            'mode': 'split'
        }

    return result



def main():
    args = parse_args()
    start_time = datetime.now()

    print(f"加载配置文件: {args.config}")
    with open(args.config, 'r', encoding='utf-8') as f:
        yaml_config = yaml.safe_load(f)

    SEED = get_param(args.seed, 'experiment.seed', yaml_config, 123)
    dataset_name = get_param(args.dataset, 'dataset.selected', yaml_config, 'chaosu')
    base_data_path = get_param(None, 'dataset.data_path', yaml_config, './data')
    logs_save_dir = get_param(None, 'paths.logs_save_dir', yaml_config, 'experiments_logs')

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device('cpu')

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_log_dir = os.path.join('logs', timestamp, 'train')
    os.makedirs(experiment_log_dir, exist_ok=True)

    log_file = os.path.join(experiment_log_dir, 'training.log')
    logger = _logger(log_file)

    logger.info("=" * 60)
    logger.info("批量训练传统机器学习模型（交叉验证）")
    logger.info("=" * 60)
    logger.info(f"数据集: {dataset_name}")
    logger.info(f"模型列表: {args.models}")
    logger.info(f"随机种子: {SEED}")
    logger.info(f"交叉验证折数: {args.cv}")

    valid_models = [m for m in args.models if m in ML_MODELS]
    invalid_models = [m for m in args.models if m not in ML_MODELS]
    if invalid_models:
        logger.warning(f"忽略无效模型: {invalid_models}")

    if not valid_models:
        logger.error(f"没有有效的模型，可选: {ML_MODELS}")
        return

    data_path = os.path.join(base_data_path, dataset_name)
    logger.info(f"\n加载数据集目录: {data_path}")

    model_config = get_model_config(yaml_config, valid_models[0], dataset_name)

    # 优先读取 full_dataset.pt；如果不存在，则回退到原有 loader 方式
    try:
        X, y = load_xy_from_full_dataset(data_path, logger)
        if X is None or y is None:
            X, y = load_xy_from_original_loaders(data_path, model_config, valid_models[0], logger)
    except Exception as e:
        logger.warning(f"读取 full_dataset.pt 失败，将回退到原有方式。原因: {e}")
        X, y = load_xy_from_original_loaders(data_path, model_config, valid_models[0], logger)

    logger.info(f"完整数据集大小: X={X.shape}, y={y.shape}")

    unique, counts = np.unique(y, return_counts=True)
    class_dist = {int(k): int(v) for k, v in zip(unique, counts)}
    logger.info(f"标签分布: {class_dist}")

    min_class_count = counts.min()
    if min_class_count < args.cv:
        logger.warning(
            f"最小类别样本数只有 {min_class_count}，小于 cv={args.cv}，将自动把 cv 调整为 {min_class_count}"
        )
        args.cv = int(min_class_count)

    use_cv = args.cv >= 2
    if not use_cv:
        logger.warning("无法进行交叉验证：某个类别样本数小于2，回退到 train/val/test 训练")

    results = []
    for model_name in valid_models:
        if use_cv:
            result = train_single_model_cv(
                model_name=model_name,
                yaml_config=yaml_config,
                X=X,
                y=y,
                experiment_log_dir=experiment_log_dir,
                logger=logger,
                device=device,
                cv=args.cv,
                seed=SEED
            )
        else:
            result = train_single_model_split(
                model_name=model_name,
                yaml_config=yaml_config,
                data_path=data_path,
                experiment_log_dir=experiment_log_dir,
                logger=logger,
                device=device
            )
        results.append(result)

    logger.info("\n" + "=" * 60)
    logger.info("训练结果汇总")
    logger.info("=" * 60)

    df = pd.DataFrame(results)
    df_success = df[df['status'] == 'success'].copy()

    if not df_success.empty:
        df_success = df_success.sort_values('f1_score', ascending=False)

        display_cols = [
            'model', 'accuracy', 'accuracy_std',
            'precision', 'recall', 'f1_score', 'f1_std', 'train_time'
        ]
        df_display = df_success[display_cols].copy()
        df_display['accuracy'] = df_display['accuracy'].apply(lambda x: f"{x:.4f}")
        df_display['accuracy_std'] = df_display['accuracy_std'].apply(lambda x: f"{x:.4f}")
        df_display['precision'] = df_display['precision'].apply(lambda x: f"{x:.4f}")
        df_display['recall'] = df_display['recall'].apply(lambda x: f"{x:.4f}")
        df_display['f1_score'] = df_display['f1_score'].apply(lambda x: f"{x:.4f}")
        df_display['f1_std'] = df_display['f1_std'].apply(lambda x: f"{x:.4f}")
        df_display['train_time'] = df_display['train_time'].apply(lambda x: f"{x:.2f}s")

        table = tabulate(df_display, headers='keys', tablefmt='grid', showindex=False)
        logger.info(f"\n{table}")

        best_model = df_success.iloc[0]
        logger.info(
            f"\n🏆 最佳模型: {best_model['model']} "
            f"(F1: {best_model['f1_score']:.4f} ± {best_model['f1_std']:.4f})"
        )

    results_filename = 'comparison_results_cv.csv' if use_cv else 'comparison_results_split.csv'
    results_file = os.path.join(experiment_log_dir, results_filename)
    df.to_csv(results_file, index=False)
    logger.info(f"\n结果已保存至: {results_file}")

    total_time = datetime.now() - start_time
    logger.info(f"\n总耗时: {total_time}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
