"""
统一训练模块（C 语言接口版）

对外暴露核心函数:
    train(config, model, dataset, result_dict) → int

调用示例:
    result_dict = {}
    error_code = train(
        config="configs/config.yaml",
        model="TSLANet",
        dataset="chaosu",
        result_dict=result_dict,
    )
    if error_code == 0:
        print(result_dict["result_dir"])

参数说明:
    config      : str  — YAML 配置文件路径
    model       : str  — 模型名称 (TSLANet, TS-TCC, base_CNN, RC,
                         RandomForest, SVM, XGBoost, LightGBM, KNN, LogisticRegression)
    dataset     : str  — 数据集名称（如 chaosu）
    result_dict : dict — 输出容器，函数执行后写入 {"result_dir": "<实验日志目录>"}

注意:
    device / batch_size / num_epochs / seed 等参数均从 config YAML 文件中读取。

错误编码:
    2101 CONFIG_LOAD_ERROR            加载YAML配置失败
    2102 DATA_LOADER_ERROR            数据加载或生成dataloader失败
    2103 MODEL_CREATION_ERROR         模型或配置获取失败
    2104 PRETRAIN_LOAD_ERROR          预训练模型加载失败
    2105 TRAINING_ERROR               训练过程出错
    2106 EVALUATION_ERROR             测试/评估出错
    2107 CONFIG_SAVE_ERROR            保存配置到磁盘失败
    2108 PARAM_ERROR                  参数校验失败
    2999 UNKNOWN_ERROR               未知错误
"""

import torch
import os
import sys
import numpy as np
import yaml
from datetime import datetime
from pathlib import Path

# 将项目根目录加入 sys.path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.logger import _logger, set_requires_grad
from data.unified_dataloader import get_dataloader
from trainers.unified_trainer import UnifiedTrainer
import models  # 自动注册所有模型
from models.model_factory import ModelFactory, get_model_config
from utils.metrics import calculate_metrics


# ================================================================
# 错误代码（统一定义在 api/error_codes.py）
# ================================================================
from api.error_codes import ErrorCode, log_error as _log_error


# ================================================================
# 模型类别常量
# ================================================================
DEEP_LEARNING_MODELS = ['TSLANet', 'TS-TCC', 'base_CNN']
NON_ITERATIVE_MODELS = ['RC', 'RandomForest', 'SVM', 'XGBoost',
                        'LightGBM', 'KNN', 'LogisticRegression']
ML_MODELS_2D = ['RandomForest', 'SVM', 'XGBoost', 'LightGBM',
                'KNN', 'LogisticRegression']
ALL_MODELS = DEEP_LEARNING_MODELS + NON_ITERATIVE_MODELS


# ================================================================
# 内部工具
# ================================================================
def _get_param(config_dict: dict, config_path: str, default=None):
    """从嵌套字典中按点分路径取值。"""
    keys = config_path.split('.')
    value = config_dict
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value if value is not None else default


# ================================================================
# 对外函数：train
# ================================================================
def train(
    config: str,
    model: str,
    dataset: str,
    result_dict: dict,
) -> int:
    """
    统一训练入口（C 语言接口版）。

    Parameters
    ----------
    config : str
        YAML 配置文件路径。
    model : str
        模型名称，如 TSLANet / TS-TCC / RandomForest 等。
    dataset : str
        数据集名称（如 chaosu），对应 data/<dataset>/ 目录。
    result_dict : dict
        输出容器。执行成功后写入:
            result_dict["result_dir"] = "<实验日志目录路径>"

    Returns
    -------
    int
        0 表示成功；非 0 为错误代码。
    """
    start_time = datetime.now()

    # ── 参数校验 ──
    if not isinstance(result_dict, dict):
        _log_error(ErrorCode.TRAIN_PARAM_ERROR, "result_dict 必须是 dict 类型")
        return ErrorCode.TRAIN_PARAM_ERROR.value

    if model not in ALL_MODELS:
        _log_error(ErrorCode.TRAIN_PARAM_ERROR,
                   f"不支持的模型 '{model}'，可选: {ALL_MODELS}")
        return ErrorCode.TRAIN_PARAM_ERROR.value

    if not os.path.exists(config):
        _log_error(ErrorCode.TRAIN_CONFIG_ERROR, f"配置文件不存在: {config}")
        return ErrorCode.TRAIN_CONFIG_ERROR.value

    # ── 加载 YAML 配置 ──
    try:
        with open(config, 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f)
    except Exception as e:
        _log_error(ErrorCode.TRAIN_CONFIG_ERROR, str(e))
        return ErrorCode.TRAIN_CONFIG_ERROR.value

    # ── 从配置读取参数 ──
    home_dir = os.getcwd()
    experiment_description = _get_param(yaml_config, 'experiment.description', 'Exp1')
    run_description = _get_param(yaml_config, 'experiment.run_description', 'run1')
    SEED = _get_param(yaml_config, 'experiment.seed', 123)
    training_mode = _get_param(yaml_config, 'training.mode', 'supervised')
    num_epochs = _get_param(yaml_config, 'training.num_epochs', 100)
    batch_size = _get_param(yaml_config, 'training.batch_size', 16)
    weight_decay = _get_param(yaml_config, 'training.weight_decay', 3e-4)
    base_data_path = _get_param(yaml_config, 'dataset.data_path', './data')
    logs_save_dir = _get_param(yaml_config, 'paths.logs_save_dir', 'experiments_logs')
    home_path = _get_param(yaml_config, 'paths.home_path', home_dir)
    device_type = yaml_config.get('device', 'cuda')

    # 预训练相关
    load_pretrained = _get_param(yaml_config, 'pretraining.load_pretrained', False)
    pretrained_path = _get_param(yaml_config, 'pretraining.pretrained_path', None)
    delete_layers = _get_param(yaml_config, 'pretraining.delete_layers', ['logits'])
    freeze_backbone = _get_param(yaml_config, 'pretraining.freeze_backbone', False)

    # 随机种子
    torch_deterministic = _get_param(yaml_config, 'random_seed.torch_deterministic', False)
    torch_benchmark = _get_param(yaml_config, 'random_seed.torch_benchmark', False)

    # ── 设备 ──
    device = torch.device(device_type if torch.cuda.is_available() else 'cpu')

    # ── 设置随机种子 ──
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = torch_deterministic
    torch.backends.cudnn.benchmark = torch_benchmark
    np.random.seed(SEED)

    # ── 创建日志目录 ──
    experiment_log_dir = os.path.join(
        logs_save_dir, experiment_description, run_description,
        f"{training_mode}_seed_{SEED}"
    )
    os.makedirs(experiment_log_dir, exist_ok=True)

    # ── 初始化 Logger ──
    log_fmt = _get_param(yaml_config, 'logging.log_timestamp_format', '%Y%m%d_%H%M%S')
    log_file = os.path.join(
        experiment_log_dir,
        f"logs_{datetime.now().strftime(log_fmt)}.log"
    )
    logger = _logger(log_file)

    logger.debug(f"模型: {model} | 数据集: {dataset} | 模式: {training_mode} | 设备: {device}")

    # ── 获取模型配置 ──
    try:
        model_config = get_model_config(yaml_config, model, dataset)
    except Exception as e:
        _log_error(ErrorCode.MODEL_CREATION_ERROR, str(e))
        return ErrorCode.MODEL_CREATION_ERROR.value

    model_config['batch_size'] = batch_size
    model_config['num_epoch'] = num_epochs

    # ── 加载数据 ──
    data_path = os.path.join(base_data_path, dataset)
    try:
        train_loader, val_loader, test_loader = get_dataloader(
            data_path, model_config, training_mode, model
        )
    except Exception as e:
        _log_error(ErrorCode.DATA_LOADER_ERROR, str(e))
        return ErrorCode.DATA_LOADER_ERROR.value

    # ── 创建模型 ──
    try:
        model_instance, auxiliary_model = ModelFactory.create_model(
            model, model_config, device
        )
    except Exception as e:
        _log_error(ErrorCode.MODEL_CREATION_ERROR, str(e))
        return ErrorCode.MODEL_CREATION_ERROR.value

    # ── 加载预训练模型 ──
    if pretrained_path:
        load_from = pretrained_path
    else:
        load_from = os.path.join(
            logs_save_dir, experiment_description, run_description,
            f"self_supervised_seed_{SEED}", "saved_models"
        )

    if training_mode == "fine_tune":
        checkpoint_path = os.path.join(load_from, "ckp_last.pt")
        if os.path.exists(checkpoint_path):
            try:
                logger.debug(f"加载预训练模型: {checkpoint_path}")
                ckpt = torch.load(checkpoint_path, map_location=device)
                pretrained_dict = ckpt["model_state_dict"]
                model_dict = model_instance.state_dict()
                for key in list(pretrained_dict.keys()):
                    for layer in delete_layers:
                        if layer in key:
                            del pretrained_dict[key]
                model_dict.update(pretrained_dict)
                model_instance.load_state_dict(model_dict)
            except Exception as e:
                _log_error(ErrorCode.PRETRAIN_LOAD_ERROR, str(e))
                return ErrorCode.PRETRAIN_LOAD_ERROR.value

    if training_mode == "train_linear" or "tl" in training_mode:
        checkpoint_path = os.path.join(load_from, "ckp_last.pt")
        if os.path.exists(checkpoint_path):
            try:
                logger.debug(f"加载预训练模型: {checkpoint_path}")
                ckpt = torch.load(checkpoint_path, map_location=device)
                pretrained_dict = ckpt["model_state_dict"]
                model_dict = model_instance.state_dict()
                pretrained_dict = {k: v for k, v in pretrained_dict.items()
                                   if k in model_dict}
                for key in list(pretrained_dict.keys()):
                    for layer in delete_layers:
                        if layer in key:
                            del pretrained_dict[key]
                model_dict.update(pretrained_dict)
                model_instance.load_state_dict(model_dict)
                if freeze_backbone or training_mode == "train_linear":
                    set_requires_grad(model_instance, pretrained_dict,
                                      requires_grad=False)
            except Exception as e:
                _log_error(ErrorCode.PRETRAIN_LOAD_ERROR, str(e))
                return ErrorCode.PRETRAIN_LOAD_ERROR.value

    if training_mode == "random_init":
        model_dict = model_instance.state_dict()
        for key in list(model_dict.keys()):
            for layer in delete_layers:
                if layer in key:
                    del model_dict[key]
        set_requires_grad(model_instance, model_dict, requires_grad=False)

    # ── 创建优化器 ──
    def _params_list(m):
        try:
            return [p for p in m.parameters() if p is not None]
        except Exception:
            return []

    model_params = _params_list(model_instance)
    if len(model_params) == 0:
        model_optimizer = None
    else:
        model_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model_params),
            lr=model_config['lr'],
            betas=(model_config['beta1'], model_config['beta2']),
            weight_decay=weight_decay,
        )

    aux_optimizer = None
    if auxiliary_model is not None:
        aux_params = _params_list(auxiliary_model)
        if len(aux_params) > 0:
            aux_optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, aux_params),
                lr=model_config['lr'],
                betas=(model_config['beta1'], model_config['beta2']),
                weight_decay=weight_decay,
            )

    # ── 选择训练器 ──
    trainer_kwargs = dict(
        model=model_instance,
        auxiliary_model=auxiliary_model,
        model_optimizer=model_optimizer,
        aux_optimizer=aux_optimizer,
        device=device,
        logger=logger,
        config=model_config,
        experiment_log_dir=experiment_log_dir,
        training_mode=training_mode,
        model_name=model,
    )

    if model == "RC":
        from trainers.rc_trainer import RCTrainer
        trainer = RCTrainer(**trainer_kwargs)
    elif model in ML_MODELS_2D or model_optimizer is None:
        from trainers.ml_trainer import MLTrainer
        trainer = MLTrainer(**trainer_kwargs)
    else:
        trainer = UnifiedTrainer(**trainer_kwargs)

    # ── 开始训练 ──
    try:
        trainer.train(train_loader, val_loader, test_loader)
    except Exception as e:
        _log_error(ErrorCode.TRAINING_ERROR, str(e))
        return ErrorCode.TRAINING_ERROR.value

    # ── 测试评估 ──
    if training_mode != "self_supervised":
        try:
            test_loss, test_acc, pred_labels, true_labels = trainer.evaluate(
                test_loader
            )
            logger.debug(f"测试损失: {test_loss:.4f} | 测试准确率: {test_acc:.4f}")
            if len(pred_labels) > 0:
                calculate_metrics(pred_labels, true_labels,
                                  experiment_log_dir, home_path)
        except Exception as e:
            _log_error(ErrorCode.EVALUATION_ERROR, str(e))
            return ErrorCode.EVALUATION_ERROR.value

    # ── 保存配置副本 ──
    config_save_path = os.path.join(experiment_log_dir, 'config.yaml')
    try:
        with open(config_save_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_config, f, default_flow_style=False,
                      allow_unicode=True)
    except Exception as e:
        _log_error(ErrorCode.CONFIG_SAVE_ERROR, str(e))
        return ErrorCode.CONFIG_SAVE_ERROR.value

    # ── 写入输出字典 ──
    result_dict["result_dir"] = experiment_log_dir

    logger.debug(f"训练总耗时: {datetime.now() - start_time}")
    print(f"✓ 训练完成，结果目录: {experiment_log_dir}")
    return 0


# ================================================================
# 命令行入口
# ================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="统一训练接口（C 语言接口版）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, required=True,
                        help="YAML 配置文件路径")
    parser.add_argument("--model", type=str, required=True,
                        help="模型名称")
    parser.add_argument("--dataset", type=str, required=True,
                        help="数据集名称")

    args = parser.parse_args()

    result_dict = {}
    code = train(
        config=args.config,
        model=args.model,
        dataset=args.dataset,
        result_dict=result_dict,
    )

    if code == 0:
        print(f"结果目录: {result_dict['result_dir']}")
    raise SystemExit(code)
