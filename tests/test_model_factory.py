"""
模型工厂单元测试

验证需求:
- 4.1: 有效模型名称和配置创建已初始化的模型实例和可选辅助模型
- 4.2: 创建TS-TCC模型时同时返回TC辅助模块
- 4.5: 支持所有模型类型（TSLANet、TS-TCC、RandomForest、SVM等）
"""
import pytest
import sys
import torch
import torch.nn as nn
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.model_factory import ModelFactory, get_model_config


# ---------------------------------------------------------------------------
# Fixtures – 各模型所需的配置字典
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_registry():
    """每个测试前清空注册表，避免测试间干扰。"""
    original = ModelFactory._registry.copy()
    ModelFactory._registry.clear()
    yield
    ModelFactory._registry = original


@pytest.fixture
def tslanet_config():
    """TSLANet 模型所需的最小配置。"""
    return {
        'seq_len': 128,
        'num_channels': 1,
        'num_classes': 3,
        'patch_size': 8,
        'emb_dim': 32,
        'depth': 1,
        'dropout_rate': 0.1,
        'use_icb': True,
        'use_asb': True,
        'adaptive_filter': True,
    }


@pytest.fixture
def tstcc_config():
    """TS-TCC 模型所需的最小配置。"""
    return {
        'input_channels': 1,
        'kernel_size': 8,
        'stride': 4,
        'final_out_channels': 32,
        'features_len': 18,
        'num_classes': 3,
        'dropout': 0.1,
        'TC': {
            'hidden_dim': 32,
            'timesteps': 4,
        },
    }


@pytest.fixture
def ml_config():
    """传统机器学习模型所需的最小配置。"""
    return {
        'input_channels': 1,
        'seq_len': 128,
        'num_classes': 3,
        'random_state': 42,
    }


# ---------------------------------------------------------------------------
# Requirement 4.1, 4.5 – 测试所有支持模型的创建
# ---------------------------------------------------------------------------
class TestCreateTSLANet:
    """测试 TSLANet 模型的创建。"""

    def test_create_tslanet_returns_model_and_none(self, tslanet_config):
        """create_model('TSLANet') 应返回 (model, None)。"""
        model, aux = ModelFactory.create_model('TSLANet', tslanet_config, device='cpu')
        assert isinstance(model, nn.Module)
        assert aux is None

    def test_tslanet_on_cpu(self, tslanet_config):
        """TSLANet 模型参数应位于 CPU 上。"""
        model, _ = ModelFactory.create_model('TSLANet', tslanet_config, device='cpu')
        for param in model.parameters():
            assert param.device.type == 'cpu'


class TestCreateTSTCC:
    """测试 TS-TCC 模型的创建。"""

    def test_create_tstcc_returns_model_and_tc(self, tstcc_config):
        """create_model('TS-TCC') 应返回 (base_Model, TC) 两个模型。

        **验证: 需求 4.2** – TS-TCC 返回 TC 辅助模块
        """
        model, aux = ModelFactory.create_model('TS-TCC', tstcc_config, device='cpu')
        assert isinstance(model, nn.Module)
        assert aux is not None
        assert isinstance(aux, nn.Module)

    def test_tstcc_auxiliary_is_tc(self, tstcc_config):
        """辅助模型应为 TC 类的实例。"""
        from models.ts_tcc import TC
        _, aux = ModelFactory.create_model('TS-TCC', tstcc_config, device='cpu')
        assert isinstance(aux, TC)

    def test_tstcc_on_cpu(self, tstcc_config):
        """TS-TCC 模型参数应位于 CPU 上。"""
        model, aux = ModelFactory.create_model('TS-TCC', tstcc_config, device='cpu')
        for param in model.parameters():
            assert param.device.type == 'cpu'
        for param in aux.parameters():
            assert param.device.type == 'cpu'


class TestCreateMLModels:
    """测试传统机器学习模型的创建。"""

    @pytest.mark.parametrize("model_name", [
        'RandomForest', 'SVM', 'KNN', 'LogisticRegression',
    ])
    def test_create_ml_model_returns_wrapper_and_none(self, model_name, ml_config):
        """传统 ML 模型应返回 (wrapper, None)。"""
        model, aux = ModelFactory.create_model(model_name, ml_config, device='cpu')
        assert isinstance(model, nn.Module)
        assert aux is None

    @pytest.mark.parametrize("model_name", [
        'RandomForest', 'SVM', 'KNN', 'LogisticRegression',
    ])
    def test_ml_model_always_on_cpu(self, model_name, ml_config):
        """传统 ML 模型即使指定 cuda 也应在 CPU 上。"""
        model, _ = ModelFactory.create_model(model_name, ml_config, device='cuda')
        for param in model.parameters():
            assert param.device.type == 'cpu'


class TestCreateXGBoostLightGBM:
    """测试 XGBoost 和 LightGBM 模型的创建（可能缺少依赖）。"""

    def test_create_xgboost(self, ml_config):
        """XGBoost 模型应能成功创建或因缺少依赖而抛出 ImportError。"""
        try:
            model, aux = ModelFactory.create_model('XGBoost', ml_config, device='cpu')
            assert isinstance(model, nn.Module)
            assert aux is None
        except ImportError:
            pytest.skip("XGBoost 未安装")

    def test_create_lightgbm(self, ml_config):
        """LightGBM 模型应能成功创建或因缺少依赖而抛出 ImportError。"""
        try:
            model, aux = ModelFactory.create_model('LightGBM', ml_config, device='cpu')
            assert isinstance(model, nn.Module)
            assert aux is None
        except ImportError:
            pytest.skip("LightGBM 未安装")


# ---------------------------------------------------------------------------
# Requirement 4.5 – 测试不支持的模型名称的错误处理
# ---------------------------------------------------------------------------
class TestUnsupportedModel:
    """测试不支持的模型名称应抛出 ValueError。"""

    def test_unknown_model_raises_value_error(self):
        """传入未注册的模型名称应抛出 ValueError。"""
        with pytest.raises(ValueError, match="未注册"):
            ModelFactory.create_model('NonExistentModel', {}, device='cpu')

    def test_error_message_contains_model_name(self):
        """ValueError 消息应包含请求的模型名称。"""
        with pytest.raises(ValueError) as exc_info:
            ModelFactory.create_model('FakeModel', {}, device='cpu')
        assert 'FakeModel' in str(exc_info.value)


# ---------------------------------------------------------------------------
# list_models / register 辅助方法测试
# ---------------------------------------------------------------------------
class TestListAndRegister:
    """测试 list_models 和 register 方法。"""

    def test_list_models_returns_list(self):
        """list_models 应返回列表类型。"""
        result = ModelFactory.list_models()
        assert isinstance(result, list)

    def test_register_adds_model(self):
        """通过 register 装饰器注册的模型应出现在 list_models 中。"""

        @ModelFactory.register('DummyModel')
        class DummyModel(nn.Module):
            def __init__(self, config):
                super().__init__()

        assert 'DummyModel' in ModelFactory.list_models()


# ---------------------------------------------------------------------------
# get_model_config 辅助函数测试
# ---------------------------------------------------------------------------
class TestGetModelConfig:
    """测试 get_model_config 函数的配置合并逻辑。"""

    def test_returns_dict(self):
        """get_model_config 应返回字典。"""
        yaml_config = {
            'dataset_configs': {'test_ds': {'input_channels': 2, 'seq_len': 64, 'num_classes': 4}},
            'training': {},
            'pretraining': {},
            'dataset': {},
        }
        result = get_model_config(yaml_config, 'TSLANet', 'test_ds')
        assert isinstance(result, dict)

    def test_contains_dataset_fields(self):
        """返回的配置应包含数据集相关字段。"""
        yaml_config = {
            'dataset_configs': {'ds': {'input_channels': 3, 'seq_len': 256, 'num_classes': 5, 'num_channels': 3}},
            'training': {},
            'pretraining': {},
            'dataset': {},
        }
        result = get_model_config(yaml_config, 'TSLANet', 'ds')
        assert result['input_channels'] == 3
        assert result['seq_len'] == 256
        assert result['num_classes'] == 5

    def test_tslanet_specific_fields(self):
        """TSLANet 配置应包含模型特定字段。"""
        yaml_config = {
            'dataset_configs': {'ds': {}},
            'TSLANet': {'emb_dim': 64, 'depth': 3},
            'training': {},
            'pretraining': {},
            'dataset': {},
        }
        result = get_model_config(yaml_config, 'TSLANet', 'ds')
        assert result['emb_dim'] == 64
        assert result['depth'] == 3

    def test_random_forest_specific_fields(self):
        """RandomForest 配置应包含 n_estimators 等字段。"""
        yaml_config = {
            'dataset_configs': {'ds': {}},
            'RandomForest': {'n_estimators': 200},
            'training': {},
            'pretraining': {},
            'dataset': {},
        }
        result = get_model_config(yaml_config, 'RandomForest', 'ds')
        assert result['n_estimators'] == 200
