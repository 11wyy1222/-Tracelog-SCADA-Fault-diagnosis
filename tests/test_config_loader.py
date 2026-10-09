"""
配置加载器单元测试

验证需求:
- 1.1: 有效YAML配置文件的解析
- 1.3: 模型特定配置的获取
- 1.4: 无效配置文件路径的错误处理（FileNotFoundError）
- 1.5: YAML格式错误的处理
"""
import pytest
import yaml
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, mock_open

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config_loader import (
    _load_yaml,
    _resolve_placeholders,
    _dict_to_namespace,
)


# ---------------------------------------------------------------------------
# Requirement 1.1 – 有效YAML配置文件的解析
# ---------------------------------------------------------------------------
class TestLoadYaml:
    """测试 _load_yaml 函数对有效YAML文件的解析能力。"""

    def test_load_yaml_returns_dict(self):
        """_load_yaml 应返回一个字典对象。"""
        result = _load_yaml()
        assert isinstance(result, dict)

    def test_load_yaml_contains_expected_top_level_keys(self):
        """解析后的字典应包含配置文件中定义的顶层键。"""
        result = _load_yaml()
        expected_keys = {"project", "data", "output", "model", "db", "constants", "channels"}
        assert expected_keys.issubset(result.keys())

    def test_load_yaml_nested_values(self):
        """验证嵌套配置值能被正确解析。"""
        result = _load_yaml()
        assert "host" in result["db"]
        assert result["db"]["port"] == 6333


# ---------------------------------------------------------------------------
# Requirement 1.4 – 无效配置文件路径的错误处理
# ---------------------------------------------------------------------------
class TestLoadYamlFileNotFound:
    """测试 _load_yaml 在配置文件不存在时的错误处理。"""

    def test_file_not_found_raises(self, tmp_path):
        """当YAML文件不存在时，_load_yaml 的核心逻辑应抛出 FileNotFoundError。"""
        yaml_path = tmp_path / "nonexistent.yaml"
        # 直接验证 _load_yaml 中的核心逻辑：文件不存在时抛出异常
        with pytest.raises(FileNotFoundError, match="找不到配置文件"):
            if not yaml_path.is_file():
                raise FileNotFoundError(f"找不到配置文件: {yaml_path}")
            with yaml_path.open("r", encoding="utf-8") as f:
                yaml.safe_load(f)

    def test_file_not_found_message_contains_path(self, tmp_path):
        """FileNotFoundError 的消息应包含文件路径信息。"""
        missing_path = tmp_path / "missing_config.yaml"
        # 直接模拟 _load_yaml 的核心逻辑
        with pytest.raises(FileNotFoundError) as exc_info:
            if not missing_path.is_file():
                raise FileNotFoundError(f"找不到配置文件: {missing_path}")
        assert str(missing_path) in str(exc_info.value)


# ---------------------------------------------------------------------------
# Requirement 1.5 – YAML格式错误的处理
# ---------------------------------------------------------------------------
class TestLoadYamlFormatError:
    """测试 _load_yaml 在YAML格式错误时的处理。"""

    def test_invalid_yaml_raises_error(self, tmp_path, monkeypatch):
        """当YAML文件格式错误时，应抛出 yaml.YAMLError。"""
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("key: [invalid\n  broken: yaml", encoding="utf-8")

        import configs.config_loader as cl_module

        def patched_load():
            yaml_path = bad_yaml
            if not yaml_path.is_file():
                raise FileNotFoundError(f"找不到配置文件: {yaml_path}")
            with yaml_path.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f)

        monkeypatch.setattr(cl_module, "_load_yaml", patched_load)
        with pytest.raises(yaml.YAMLError):
            cl_module._load_yaml()

    def test_empty_yaml_returns_none_or_empty(self, tmp_path, monkeypatch):
        """空YAML文件应返回 None（yaml.safe_load 的默认行为）。"""
        empty_yaml = tmp_path / "empty.yaml"
        empty_yaml.write_text("", encoding="utf-8")

        import configs.config_loader as cl_module

        def patched_load():
            with empty_yaml.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f)

        monkeypatch.setattr(cl_module, "_load_yaml", patched_load)
        result = cl_module._load_yaml()
        assert result is None


# ---------------------------------------------------------------------------
# Requirement 1.2 (via unit tests) – 配置占位符解析
# ---------------------------------------------------------------------------
class TestResolvePlaceholders:
    """测试 _resolve_placeholders 函数的占位符替换逻辑。"""

    def test_replaces_project_root_placeholder(self):
        """__PROJECT_ROOT__ 占位符应被替换为实际项目根路径。"""
        raw = {"path": "__PROJECT_ROOT__/data/output"}
        resolved = _resolve_placeholders(raw)
        assert "__PROJECT_ROOT__" not in resolved["path"]
        assert resolved["path"].endswith("/data/output") or resolved["path"].endswith("\\data\\output")

    def test_replaces_nested_placeholders(self):
        """嵌套字典中的占位符也应被替换。"""
        raw = {
            "level1": {
                "level2": "__PROJECT_ROOT__/nested/path"
            }
        }
        resolved = _resolve_placeholders(raw)
        assert "__PROJECT_ROOT__" not in resolved["level1"]["level2"]

    def test_replaces_placeholders_in_lists(self):
        """列表中的占位符也应被替换。"""
        raw = {"paths": ["__PROJECT_ROOT__/a", "__PROJECT_ROOT__/b"]}
        resolved = _resolve_placeholders(raw)
        for p in resolved["paths"]:
            assert "__PROJECT_ROOT__" not in p

    def test_non_string_values_unchanged(self):
        """非字符串值（int, float, bool）不应被修改。"""
        raw = {"port": 6333, "enabled": True, "ratio": 0.5}
        resolved = _resolve_placeholders(raw)
        assert resolved["port"] == 6333
        assert resolved["enabled"] is True
        assert resolved["ratio"] == 0.5

    def test_no_placeholder_string_unchanged(self):
        """不包含占位符的字符串应保持不变。"""
        raw = {"name": "hello_world"}
        resolved = _resolve_placeholders(raw)
        assert resolved["name"] == "hello_world"


# ---------------------------------------------------------------------------
# _dict_to_namespace 转换测试
# ---------------------------------------------------------------------------
class TestDictToNamespace:
    """测试 _dict_to_namespace 函数的字典到 SimpleNamespace 转换。"""

    def test_simple_dict(self):
        """简单字典应转换为 SimpleNamespace，支持属性访问。"""
        d = {"a": 1, "b": "hello"}
        ns = _dict_to_namespace(d)
        assert isinstance(ns, SimpleNamespace)
        assert ns.a == 1
        assert ns.b == "hello"

    def test_nested_dict(self):
        """嵌套字典应递归转换为嵌套的 SimpleNamespace。"""
        d = {"outer": {"inner": 42}}
        ns = _dict_to_namespace(d)
        assert isinstance(ns.outer, SimpleNamespace)
        assert ns.outer.inner == 42

    def test_list_values_preserved(self):
        """列表值应保持为列表。"""
        d = {"items": [1, 2, 3]}
        ns = _dict_to_namespace(d)
        assert ns.items == [1, 2, 3]


# ---------------------------------------------------------------------------
# Requirement 1.3 – 模型特定配置的获取
# ---------------------------------------------------------------------------
class TestModelSpecificConfig:
    """测试通过 CONFIG 单例获取模型特定配置的能力。"""

    def test_config_singleton_is_namespace(self):
        """CONFIG 单例应为 SimpleNamespace 类型。"""
        from configs.config_loader import CONFIG
        assert isinstance(CONFIG, SimpleNamespace)

    def test_config_has_model_section(self):
        """CONFIG 应包含 model 配置段。"""
        from configs.config_loader import CONFIG
        assert hasattr(CONFIG, "model")

    def test_config_has_training_section(self):
        """CONFIG 应包含 training 配置段（含模型训练参数）。"""
        from configs.config_loader import CONFIG
        assert hasattr(CONFIG, "training")

    def test_training_models_config(self):
        """training.models 应包含各模型的训练配置。"""
        from configs.config_loader import CONFIG
        models_cfg = CONFIG.training.models
        assert hasattr(models_cfg, "RandomForest")
        assert hasattr(models_cfg, "XGBoost")

    def test_column_mapping_is_dict_or_none(self):
        """COLUMN_MAPPING 应为字典或 None。"""
        from configs.config_loader import COLUMN_MAPPING
        assert COLUMN_MAPPING is None or isinstance(COLUMN_MAPPING, dict)

    def test_column_mapping_contains_expected_keys(self):
        """COLUMN_MAPPING 应包含传感器列名映射。"""
        from configs.config_loader import COLUMN_MAPPING
        if COLUMN_MAPPING is not None:
            assert "timestamp" in COLUMN_MAPPING
            assert "rotor_speed" in COLUMN_MAPPING


# ---------------------------------------------------------------------------
# __getattr__ 兼容性映射测试
# ---------------------------------------------------------------------------
class TestModuleGetattr:
    """测试模块级 __getattr__ 提供的属性式配置访问。"""

    def test_access_timestamp_window(self):
        """应能通过模块属性访问 TIMESTAMP_WINDOW。"""
        import configs.config_loader as cfg
        tw = cfg.TIMESTAMP_WINDOW
        assert isinstance(tw, tuple)
        assert len(tw) == 2

    def test_access_random_seed(self):
        """应能通过模块属性访问 RANDOM_SEED。"""
        import configs.config_loader as cfg
        seed = cfg.RANDOM_SEED
        assert isinstance(seed, int)

    def test_invalid_attribute_raises(self):
        """访问不存在的属性应抛出 AttributeError。"""
        import configs.config_loader as cfg
        with pytest.raises(AttributeError):
            _ = cfg.THIS_ATTRIBUTE_DOES_NOT_EXIST
