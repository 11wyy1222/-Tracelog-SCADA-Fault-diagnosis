"""
模型工厂 - 统一的模型创建接口
支持TSLANet, TS-TCC, RC以及传统机器学习模型
"""
import torch.nn as nn
from typing import Dict, Any


class ModelFactory:
    """模型工厂类,根据配置创建不同的模型"""

    _registry = {}  # 模型注册表

    @classmethod
    def register(cls, model_name: str):
        """装饰器:注册模型类"""
        def decorator(model_class):
            cls._registry[model_name] = model_class
            return model_class
        return decorator

    @classmethod
    def create_model(cls, model_name: str, config: Dict[str, Any], device='cuda'):
        """
        创建模型实例

        Args:
            model_name: 模型名称 (TSLANet, TS-TCC, RC, RandomForest等)
            config: 配置字典
            device: 运行设备

        Returns:
            model: 模型实例
            auxiliary_model: 辅助模型(如TS-TCC的TC模型,其他为None)
        """
        if model_name not in cls._registry:
            cls._auto_register_model(model_name)

        if model_name not in cls._registry:
            available = ', '.join(cls._registry.keys()) if cls._registry else '无'
            raise ValueError(
                f"模型 '{model_name}' 未注册!\n"
                f"可用模型: {available}"
            )

        model_class = cls._registry[model_name]
        ml_models = {
            'RandomForest', 'ExtraTrees', 'SVM',
            'XGBoost', 'LightGBM', 'KNN', 'LogisticRegression'
        }

        if model_name == 'TS-TCC':
            from models.ts_tcc import base_Model, TC
            main_model = base_Model(config).to(device)
            auxiliary_model = TC(config, device).to(device)
            return main_model, auxiliary_model

        elif model_name == 'TSLANet':
            main_model = model_class(
                seq_len=config['seq_len'],
                num_channels=config['num_channels'],
                num_classes=config['num_classes'],
                patch_size=config.get('patch_size', 8),
                emb_dim=config.get('emb_dim', 128),
                depth=config.get('depth', 2),
                dropout_rate=config.get('dropout_rate', 0.15),
                use_icb=config.get('use_icb', True),
                use_asb=config.get('use_asb', True),
                adaptive_filter=config.get('adaptive_filter', True)
            ).to(device)
            return main_model, None

        elif model_name == 'RC':
            from models.rc_model import RC
            main_model = RC(config).to(device)
            return main_model, None

        elif model_name in ml_models:
            main_model = model_class(config)
            return main_model, None

        else:
            main_model = model_class(config).to(device)
            return main_model, None

    @classmethod
    def _auto_register_model(cls, model_name: str):
        """
        自动注册模型

        Args:
            model_name: 模型名称
        """
        try:
            if model_name == 'TSLANet':
                from models.tslanet import TSLANet
                cls._registry['TSLANet'] = TSLANet
                print(f"✓ 自动注册模型: TSLANet")
            elif model_name == 'TS-TCC':
                from models.ts_tcc import base_Model
                cls._registry['TS-TCC'] = base_Model
                print(f"✓ 自动注册模型: TS-TCC")
            elif model_name == 'RC':
                from models.rc_model import RC
                cls._registry['RC'] = RC
                print(f"✓ 自动注册模型: RC")
            elif model_name == 'RandomForest':
                from models.ml_models import RandomForestWrapper
                cls._registry['RandomForest'] = RandomForestWrapper
                print(f"✓ 自动注册模型: RandomForest")
            elif model_name == 'ExtraTrees':
                from models.ml_models import ExtraTreesWrapper
                cls._registry['ExtraTrees'] = ExtraTreesWrapper
                print(f"✓ 自动注册模型: ExtraTrees")
            elif model_name == 'SVM':
                from models.ml_models import SVMWrapper
                cls._registry['SVM'] = SVMWrapper
                print(f"✓ 自动注册模型: SVM")
            elif model_name == 'XGBoost':
                from models.ml_models import XGBoostWrapper
                cls._registry['XGBoost'] = XGBoostWrapper
                print(f"✓ 自动注册模型: XGBoost")
            elif model_name == 'LightGBM':
                from models.ml_models import LightGBMWrapper
                cls._registry['LightGBM'] = LightGBMWrapper
                print(f"✓ 自动注册模型: LightGBM")
            elif model_name == 'KNN':
                from models.ml_models import KNNWrapper
                cls._registry['KNN'] = KNNWrapper
                print(f"✓ 自动注册模型: KNN")
            elif model_name == 'LogisticRegression':
                from models.ml_models import LogisticRegressionWrapper
                cls._registry['LogisticRegression'] = LogisticRegressionWrapper
                print(f"✓ 自动注册模型: LogisticRegression")
        except ImportError as e:
            print(f"⚠ 无法自动注册模型 {model_name}: {e}")

    @classmethod
    def list_models(cls):
        """列出所有已注册的模型"""
        return list(cls._registry.keys())


def get_model_config(yaml_config: Dict, model_name: str, dataset_name: str) -> Dict:
    """
    获取模型配置

    Args:
        yaml_config: YAML配置字典
        model_name: 模型名称
        dataset_name: 数据集名称

    Returns:
        合并后的配置字典
    """
    # 获取数据集配置
    dataset_config = yaml_config.get('dataset_configs', {}).get(dataset_name, {})

    # 获取模型特定配置
    model_config = yaml_config.get(model_name, {})

    # 获取训练配置
    training_config = yaml_config.get('training', {})
    pretraining_config = yaml_config.get('pretraining', {})

    # 合并配置
    config = {
        # 数据集配置
        'input_channels': dataset_config.get('input_channels', 1),
        'num_channels': dataset_config.get('num_channels', 1),
        'seq_len': dataset_config.get('seq_len', 128),
        'features_len': dataset_config.get('features_len', 127),
        'num_classes': dataset_config.get('num_classes', 2),

        # 训练配置
        'num_epoch': training_config.get('num_epochs', 100),
        'batch_size': training_config.get('batch_size', 16),
        'lr': training_config.get('learning_rate', 0.001),
        'optimizer': training_config.get('optimizer', 'adam'),
        'beta1': training_config.get('beta1', 0.9),
        'beta2': training_config.get('beta2', 0.99),
        'dropout': model_config.get('dropout', 0.35),
        'drop_last': yaml_config.get('dataset', {}).get('drop_last', True),

        # 预训练配置
        'pretrain_epochs': pretraining_config.get('pretrain_epochs', 50),
        'pretrain_lr': pretraining_config.get('pretrain_lr', 0.001),
        'masking_ratio': pretraining_config.get('masking_ratio', 0.4),
    }

    # 添加模型特定配置
    if model_name == 'TSLANet':
        config.update({
            'emb_dim': model_config.get('emb_dim', 128),
            'depth': model_config.get('depth', 2),
            'patch_size': model_config.get('patch_size', 8),
            'dropout_rate': model_config.get('dropout_rate', 0.15),
            'use_icb': model_config.get('use_icb', True),
            'use_asb': model_config.get('use_asb', True),
            'adaptive_filter': model_config.get('adaptive_filter', True),
        })

    elif model_name == 'TS-TCC':
        config.update({
            'kernel_size': model_config.get('kernel_size', 32),
            'stride': model_config.get('stride', 30),
            'final_out_channels': model_config.get('final_out_channels', 128),
        })

        # Context Contrast配置
        context_config = model_config.get('context_contrast', {})
        config['Context_Cont'] = type('obj', (object,), {
            'temperature': context_config.get('temperature', 0.2),
            'use_cosine_similarity': context_config.get('use_cosine_similarity', True)
        })()

        # Temporal Contrast配置
        tc_config = model_config.get('temporal_contrast', {})
        config['TC'] = type('obj', (object,), {
            'hidden_dim': tc_config.get('hidden_dim', 64),
            'timesteps': tc_config.get('timesteps', 15)
        })()

        # 数据增强配置
        aug_config = model_config.get('augmentation', {})
        config['augmentation'] = type('obj', (object,), {
            'jitter_scale_ratio': aug_config.get('jitter_scale_ratio', 1.0),
            'jitter_ratio': aug_config.get('jitter_ratio', 0.5),
            'max_seg': aug_config.get('max_seg', 10)
        })()

    elif model_name == 'base_CNN':
        config.update({
            'kernel_size': model_config.get('kernel_size', 25),
            'stride': model_config.get('stride', 6),
            'final_out_channels': model_config.get('final_out_channels', 128),
        })

    elif model_name == 'RC':
        config.update({
            'n_internal_units': model_config.get('n_internal_units', 100),
            'spectral_radius': model_config.get('spectral_radius', 0.99),
            'connectivity': model_config.get('connectivity', 0.3),
            'input_scaling': model_config.get('input_scaling', 0.2),
            'noise_level': model_config.get('noise_level', 0.0),
            'bidir': model_config.get('bidir', False),
            'circle': model_config.get('circle', False),
            'mts_rep': model_config.get('mts_rep', 'mean'),
            'w_ridge': model_config.get('w_ridge', 1.0),
        })

    elif model_name == 'RandomForest':
        config.update({
            'n_estimators': model_config.get('n_estimators', 100),
            'max_depth': model_config.get('max_depth', None),
            'min_samples_split': model_config.get('min_samples_split', 2),
            'min_samples_leaf': model_config.get('min_samples_leaf', 1),
            'class_weight': model_config.get('class_weight', 'balanced'),
            'random_state': model_config.get('random_state', 42),
        })

    elif model_name == 'ExtraTrees':
        config.update({
            'n_estimators': model_config.get('n_estimators', 100),
            'max_depth': model_config.get('max_depth', None),
            'min_samples_split': model_config.get('min_samples_split', 2),
            'min_samples_leaf': model_config.get('min_samples_leaf', 1),
            'class_weight': model_config.get('class_weight', 'balanced'),
            'random_state': model_config.get('random_state', 42),
        })

    elif model_name == 'SVM':
        config.update({
            'C': model_config.get('C', 1.0),
            'kernel': model_config.get('kernel', 'rbf'),
            'gamma': model_config.get('gamma', 'scale'),
            'class_weight': model_config.get('class_weight', 'balanced'),
            'random_state': model_config.get('random_state', 42),
        })

    elif model_name == 'XGBoost':
        config.update({
            'n_estimators': model_config.get('n_estimators', 100),
            'max_depth': model_config.get('max_depth', 6),
            'learning_rate': model_config.get('learning_rate', 0.1),
            'subsample': model_config.get('subsample', 0.8),
            'colsample_bytree': model_config.get('colsample_bytree', 0.8),
            'random_state': model_config.get('random_state', 42),
        })

    elif model_name == 'LightGBM':
        config.update({
            'n_estimators': model_config.get('n_estimators', 100),
            'max_depth': model_config.get('max_depth', -1),
            'learning_rate': model_config.get('learning_rate', 0.1),
            'num_leaves': model_config.get('num_leaves', 31),
            'subsample': model_config.get('subsample', 0.8),
            'colsample_bytree': model_config.get('colsample_bytree', 0.8),
            'class_weight': model_config.get('class_weight', 'balanced'),
            'random_state': model_config.get('random_state', 42),
        })

    elif model_name == 'KNN':
        config.update({
            'n_neighbors': model_config.get('n_neighbors', 5),
            'weights': model_config.get('weights', 'uniform'),
            'algorithm': model_config.get('algorithm', 'auto'),
            'metric': model_config.get('metric', 'minkowski'),
        })

    elif model_name == 'LogisticRegression':
        config.update({
            'C': model_config.get('C', 1.0),
            'penalty': model_config.get('penalty', 'l2'),
            'solver': model_config.get('solver', 'lbfgs'),
            'class_weight': model_config.get('class_weight', 'balanced'),
            'max_iter': model_config.get('max_iter', 1000),
            'random_state': model_config.get('random_state', 42),
        })

    return config
