import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
import logging
import warnings

warnings.filterwarnings('ignore')


class FeatureSelector:
    """
    特征选择与降维工具
    职责：
    1. 基于树模型的特征重要性筛选 (Feature Selection)
    2. PCA 降维 (Dimensionality Reduction)
    """

    def __init__(self, method='tree', n_features=30):
        self.method = method  # 'tree' or 'pca'
        self.n_features = n_features
        self.selector = None
        self.selected_indices = None
        self.feature_importance = None

        # 初始化日志
        self.logger = logging.getLogger("FeatureSelector")
        if not self.logger.handlers:
            logging.basicConfig(level=logging.INFO)

    def fit(self, X, y, feature_names=None):
        """训练特征选择器"""
        self.logger.info(f"开始特征筛选 (方法={self.method}, 目标维度={self.n_features})...")

        if self.method == 'tree':
            # 使用随机森林计算特征重要性
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X, y)

            # 记录特征重要性
            self.feature_importance = rf.feature_importances_

            # 选择最重要的特征
            # threshold设置为 -np.inf 配合 max_features 使用
            self.selector = SelectFromModel(rf, max_features=self.n_features, threshold=-np.inf, prefit=True)
            self.selected_indices = self.selector.get_support(indices=True)

            if feature_names is not None:
                selected_names = [feature_names[i] for i in self.selected_indices]
                self.logger.info(f"✅ 选出的 Top {len(selected_names)} 特征: {selected_names[:5]}...")

        elif self.method == 'pca':
            self.selector = PCA(n_components=self.n_features)
            self.selector.fit(X)
            self.logger.info(f"✅ PCA 解释方差比: {sum(self.selector.explained_variance_ratio_):.4f}")

        return self

    def transform(self, X):
        """应用特征转换"""
        if self.selector is None:
            raise ValueError("Selector not fitted. Call fit() first.")

        if self.method == 'tree':
            return self.selector.transform(X)
        elif self.method == 'pca':
            return self.selector.transform(X)
        else:
            return X

    def fit_transform(self, X, y, feature_names=None):
        self.fit(X, y, feature_names)
        return self.transform(X)

    def get_support(self):
        """返回布尔掩码，指示哪些特征被选中 (仅适用于 Tree 方法)"""
        if self.method == 'tree' and self.selector:
            return self.selector.get_support()
        return None