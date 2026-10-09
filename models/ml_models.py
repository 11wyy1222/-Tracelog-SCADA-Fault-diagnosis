"""
传统机器学习模型包装器
将sklearn/XGBoost/LightGBM等模型包装成PyTorch风格的接口
"""
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


class MLBaseWrapper(nn.Module):
    """
    机器学习模型基类包装器
    提供统一的PyTorch风格接口
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_channels = config.get('input_channels', 1)
        self.seq_len = config.get('seq_len', 128)
        self.num_classes = config.get('num_classes', 2)

        self.model = None  # 子类需要初始化
        self.scaler = StandardScaler()  # 数据标准化器
        self.is_fitted = False
        self._output_dim = None
        self._classes = None
        self._unique_classes = None
        self._label_map = {}
        self._label_map_inv = {}

    def _flatten_input(self, x):
        """
        展平输入数据

        Args:
            x: [batch_size, seq_len, channels] 或 [batch_size, features]

        Returns:
            x_flat: [batch_size, features]
        """
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()

        if len(x.shape) == 3:
            # [batch, seq_len, channels] -> [batch, seq_len*channels]
            x = x.reshape(x.shape[0], -1)

        return x

    def fit(self, X, y):
        """
        训练模型

        Args:
            X: [batch_size, seq_len, channels] 或 [batch_size, features]
            y: [batch_size] 或 [batch_size, num_classes]
        """
        # 转换为numpy
        X = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else X
        y = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else y

        # 展平输入
        X = self._flatten_input(X)

        # 处理NaN值
        X = np.nan_to_num(X, nan=0.0)

        # 数据标准化
        X = self.scaler.fit_transform(X)

        # 处理标签格式
        if len(y.shape) > 1 and y.shape[1] > 1:
            # one-hot -> 类别索引
            y = np.argmax(y, axis=1)

        y = np.asarray(y).reshape(-1)

        # 重映射标签为连续值（XGBoost要求）
        self._unique_classes = np.unique(y)
        self._label_map = {orig: idx for idx, orig in enumerate(self._unique_classes)}
        self._label_map_inv = {idx: orig for orig, idx in self._label_map.items()}
        y_mapped = np.array([self._label_map[label] for label in y])

        # 训练模型
        self.model.fit(X, y_mapped)
        self.is_fitted = True
        self._output_dim = X.shape[1]
        self._classes = self._unique_classes

    def forward(self, x):
        """
        推理

        Args:
            x: [batch_size, seq_len, channels] 或 [batch_size, features]

        Returns:
            predictions: [batch_size] 类别标签（原始标签值）
        """
        if not self.is_fitted:
            raise RuntimeError("模型必须先调用fit()进行训练")

        # 转换并展平
        x_np = self._flatten_input(x)

        # 处理NaN值
        x_np = np.nan_to_num(x_np, nan=0.0)

        # 数据标准化（使用训练时的scaler）
        x_np = self.scaler.transform(x_np)

        # 预测（返回映射后的标签）
        pred_mapped = self.model.predict(x_np)

        # 还原为原始标签
        pred = np.array([self._label_map_inv[p] for p in pred_mapped])

        return torch.from_numpy(pred).long()

    def predict_with_proba(self, x):
        """
        预测并返回概率分布

        Returns:
            (predictions, probabilities): 类别标签和概率矩阵
        """
        if not self.is_fitted:
            raise RuntimeError("模型必须先调用fit()进行训练")

        x_np = self._flatten_input(x)
        x_np = np.nan_to_num(x_np, nan=0.0)
        x_np = self.scaler.transform(x_np)

        pred_mapped = self.model.predict(x_np)
        pred = np.array([self._label_map_inv[p] for p in pred_mapped])

        # 获取概率分布
        if hasattr(self.model, 'predict_proba'):
            probs = self.model.predict_proba(x_np)
        else:
            # 不支持概率的模型，用one-hot
            num_classes = len(self._unique_classes)
            probs = np.zeros((len(pred_mapped), num_classes), dtype=np.float32)
            for i, p in enumerate(pred_mapped):
                probs[i, p] = 1.0

        return torch.from_numpy(pred).long(), torch.from_numpy(probs).float()

    def save_checkpoint(self, path):
        """保存检查点"""
        dirpath = os.path.dirname(path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)

        checkpoint = {
            'model_state_dict': {
                'sklearn_model': pickle.dumps(self.model),
                'scaler': pickle.dumps(self.scaler),
                'is_fitted': self.is_fitted,
                'output_dim': self._output_dim,
                'classes': self._classes,
                'unique_classes': self._unique_classes,
                'label_map': self._label_map,
                'label_map_inv': self._label_map_inv,
            },
            'config': self.config
        }

        torch.save(checkpoint, path)
        print(f"✓ {self.__class__.__name__} 检查点已保存: {path}")

    def load_checkpoint(self, path):
        """加载检查点"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"检查点文件不存在: {path}")

        checkpoint = torch.load(path, map_location='cpu', weights_only=False)

        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
        else:
            model_state = checkpoint

        self.model = pickle.loads(model_state['sklearn_model'])
        self.scaler = pickle.loads(model_state.get('scaler', pickle.dumps(StandardScaler())))
        self.is_fitted = model_state.get('is_fitted', True)
        self._output_dim = model_state.get('output_dim', None)
        self._classes = model_state.get('classes', None)
        self._unique_classes = model_state.get('unique_classes', self._classes)
        self._label_map = model_state.get('label_map', {})
        self._label_map_inv = model_state.get('label_map_inv', {})

        print(f"✓ {self.__class__.__name__} 检查点已加载: {path}")

    def state_dict(self):
        """返回模型状态字典"""
        return {
            'sklearn_model': pickle.dumps(self.model),
            'scaler': pickle.dumps(self.scaler),
            'is_fitted': self.is_fitted,
            'output_dim': self._output_dim,
            'classes': self._classes,
            'unique_classes': self._unique_classes,
            'label_map': self._label_map,
            'label_map_inv': self._label_map_inv,
        }

    def load_state_dict(self, state_dict, strict=True):
        """加载模型状态字典"""
        self.model = pickle.loads(state_dict['sklearn_model'])
        self.scaler = pickle.loads(state_dict.get('scaler', pickle.dumps(StandardScaler())))
        self.is_fitted = state_dict['is_fitted']
        self._output_dim = state_dict['output_dim']
        self._classes = state_dict.get('classes', None)
        self._unique_classes = state_dict.get('unique_classes', self._classes)
        self._label_map = state_dict.get('label_map', {})
        self._label_map_inv = state_dict.get('label_map_inv', {})
        return self


class RandomForestWrapper(MLBaseWrapper):
    """随机森林分类器包装器"""

    def __init__(self, config):
        super().__init__(config)
        from sklearn.ensemble import RandomForestClassifier

        self.model = RandomForestClassifier(
            n_estimators=config.get('n_estimators', 300),
            max_depth=config.get('max_depth', 12),
            min_samples_split=config.get('min_samples_split', 2),
            min_samples_leaf=config.get('min_samples_leaf', 2),
            class_weight=config.get('class_weight', 'balanced'),
            random_state=config.get('random_state', 42),
            n_jobs=-1
        )


class ExtraTreesWrapper(MLBaseWrapper):
    """ExtraTrees分类器包装器"""

    def __init__(self, config):
        super().__init__(config)
        from sklearn.ensemble import ExtraTreesClassifier

        self.model = ExtraTreesClassifier(
            n_estimators=config.get('n_estimators', 300),
            max_depth=config.get('max_depth', 15),
            min_samples_split=config.get('min_samples_split', 5),
            min_samples_leaf=config.get('min_samples_leaf', 2),
            class_weight=config.get('class_weight', 'balanced'),
            random_state=config.get('random_state', 42),
            n_jobs=-1
        )


class SVMWrapper(MLBaseWrapper):
    """支持向量机分类器包装器"""

    def __init__(self, config):
        super().__init__(config)
        from sklearn.svm import SVC

        # 注意：MLBaseWrapper.fit() 已经做了标准化，这里不需要Pipeline
        self.model = SVC(
            C=config.get('C', 10.0),
            kernel=config.get('kernel', 'rbf'),
            gamma=config.get('gamma', 'scale'),
            class_weight=config.get('class_weight', 'balanced'),
            probability=True,
            random_state=config.get('random_state', 42)
        )


class XGBoostWrapper(MLBaseWrapper):
    """XGBoost分类器包装器"""

    def __init__(self, config):
        super().__init__(config)
        try:
            import xgboost as xgb

            self.model = xgb.XGBClassifier(
                n_estimators=config.get('n_estimators', 200),
                max_depth=config.get('max_depth', 6),
                learning_rate=config.get('learning_rate', 0.05),
                tree_method=config.get('tree_method', 'hist'),
                subsample=config.get('subsample', 0.8),
                colsample_bytree=config.get('colsample_bytree', 0.7),
                random_state=config.get('random_state', 42),
                n_jobs=-1
            )
        except ImportError:
            raise ImportError("请安装XGBoost: pip install xgboost")


class LightGBMWrapper(MLBaseWrapper):
    """LightGBM分类器包装器"""

    def __init__(self, config):
        super().__init__(config)
        try:
            import lightgbm as lgb

            self.model = lgb.LGBMClassifier(
                n_estimators=config.get('n_estimators', 200),
                max_depth=config.get('max_depth', -1),
                learning_rate=config.get('learning_rate', 0.05),
                num_leaves=config.get('num_leaves', 31),
                min_child_samples=config.get('min_child_samples', 10),
                class_weight=config.get('class_weight', 'balanced'),
                random_state=config.get('random_state', 42),
                verbose=-1,
                n_jobs=-1
            )
        except ImportError:
            raise ImportError("请安装LightGBM: pip install lightgbm")


class KNNWrapper(MLBaseWrapper):
    """K近邻分类器包装器"""

    def __init__(self, config):
        super().__init__(config)
        from sklearn.neighbors import KNeighborsClassifier

        self.model = KNeighborsClassifier(
            n_neighbors=config.get('n_neighbors', 5),
            weights=config.get('weights', 'distance'),
            algorithm=config.get('algorithm', 'auto'),
            metric=config.get('metric', 'minkowski'),
            n_jobs=-1
        )


class LogisticRegressionWrapper(MLBaseWrapper):
    """逻辑回归分类器包装器"""

    def __init__(self, config):
        super().__init__(config)
        from sklearn.linear_model import LogisticRegression

        self.model = LogisticRegression(
            C=config.get('C', 1.0),
            penalty=config.get('penalty', 'l2'),
            solver=config.get('solver', 'lbfgs'),
            max_iter=config.get('max_iter', 1000),
            class_weight=config.get('class_weight', 'balanced'),
            random_state=config.get('random_state', 42),
            n_jobs=-1
        )
