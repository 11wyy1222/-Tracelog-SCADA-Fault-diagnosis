"""
统一预测模块（C 语言接口版）

对外暴露核心函数:
    predict(config, model, data_path, result_dict) → int

调用示例:
    result_dict = {}
    error_code = predict(
        config="configs/config.yaml",
        model="TSLANet",
        data_path="data/chaosu/test.pt",
        result_dict=result_dict,
    )
    if error_code == 0:
        print(result_dict["result_dir"])

参数说明:
    config      : str  — YAML 配置文件路径
    model       : str  — 模型名称 (TSLANet, TS-TCC, base_CNN, RC,
                         RandomForest, SVM, XGBoost, LightGBM, KNN, LogisticRegression)
    data_path   : str  — 外部数据文件路径(.pt 文件，包含 samples 和可选 labels)
    result_dict : dict — 输出容器，函数执行后写入 {"result_dir": "<结果JSON路径>"}

注意:
    batch_size 和 device 从 config YAML 文件中读取:
        - training.batch_size（默认 128）
        - device（默认 "cuda"）

返回值:
    int : 0 表示成功；非 0 为对应错误代码

错误编码:
    3101 CHECKPOINT_MISSING     检查点文件不存在或无法访问
    3102 CONFIG_MISSING         配置文件不存在或载入失败
    3103 MAPPING_LOAD_ERROR     故障映射加载失败
    3104 DATA_FILE_ERROR        外部数据文件缺失或格式错误
    3105 DATALOADER_ERROR       内置数据集加载失败
    3106 MODEL_CREATION_ERROR   模型实例化或权重加载失败
    3107 PREDICTION_ERROR       预测过程出错
    3108 SAVE_RESULT_ERROR      结果保存失败（文件IO）
    3109 PARAM_ERROR            参数校验失败
    3999 UNKNOWN_ERROR          未知错误
"""

import torch
import numpy as np
import yaml
import os
import sys
import json
from pathlib import Path
from datetime import datetime
from collections import Counter

# 将项目根目录加入 sys.path，确保子模块可导入
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.fault_diagnosis_mapper import FaultDiagnosisMapper
from models.model_factory import ModelFactory, get_model_config


# ================================================================
# 错误代码（统一定义在 api/error_codes.py）
# ================================================================
from error_codes import ErrorCode, log_error as _log_error


# ================================================================
# 模型类别常量
# ================================================================
DEEP_LEARNING_MODELS = ['TSLANet', 'TS-TCC', 'base_CNN']
NON_ITERATIVE_MODELS = ['RC', 'RandomForest', 'ExtraTrees', 'SVM', 'XGBoost',
                        'LightGBM', 'KNN', 'LogisticRegression']
ML_MODELS_2D = ['RandomForest', 'ExtraTrees', 'SVM', 'XGBoost', 'LightGBM',
                'KNN', 'LogisticRegression']
ALL_MODELS = DEEP_LEARNING_MODELS + NON_ITERATIVE_MODELS


# ================================================================
# 内部工具函数
# ================================================================
def _build_dataloader(data_path: str, batch_size: int = 128):
    """
    从 .pt 文件构建 DataLoader。

    Returns:
        (dataloader, has_labels)
    """
    data = torch.load(data_path, weights_only=False)

    if 'samples' not in data:
        raise ValueError("数据文件必须包含 'samples' 字段")

    samples = data['samples']
    labels = data.get('labels', None)
    has_labels = labels is not None

    class _SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, samples, labels=None):
            self.samples = (samples if isinstance(samples, torch.Tensor)
                            else torch.from_numpy(samples))
            self.labels = labels
            self.has_labels = labels is not None

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            sample = self.samples[idx].float()
            if self.has_labels:
                lbl = (self.labels if isinstance(self.labels, torch.Tensor)
                       else torch.from_numpy(self.labels))
                return sample, lbl[idx].long()
            return sample

    loader = torch.utils.data.DataLoader(
        _SimpleDataset(samples, labels),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return loader, has_labels


@torch.no_grad()
def _predict_batch(model, dataloader, device, model_name, has_labels):
    """
    执行批量预测，返回 (predictions, probabilities, true_labels)。
    """
    model.eval()

    # ---- 非迭代模型（RC + 传统ML）：一次性收集再预测 ----
    if model_name in NON_ITERATIVE_MODELS:
        all_data, all_labels = [], []
        for batch in dataloader:
            if has_labels and isinstance(batch, (list, tuple)) and len(batch) >= 2:
                data, labels = batch[0].float(), batch[1].long()
            else:
                data = batch[0].float() if isinstance(batch, (list, tuple)) else batch.float()
                labels = None

            if model_name in ML_MODELS_2D and len(data.shape) == 3:
                data = data.reshape(data.shape[0], -1)

            all_data.append(data.cpu())
            if labels is not None:
                all_labels.append(labels.cpu())

        x_all = torch.cat(all_data, dim=0)
        y_all = torch.cat(all_labels, dim=0) if all_labels else None

        device_for_model = device if model_name == 'RC' else 'cpu'
        logits = model(x_all.to(device_for_model))

        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits).float()
        elif not isinstance(logits, torch.Tensor):
            raise TypeError(f"不支持的 logits 类型: {type(logits)}")
        logits = logits.cpu()

        # ML模型用predict_with_proba获取真实概率
        if model_name in ML_MODELS_2D and hasattr(model, 'predict_with_proba'):
            preds, probs = model.predict_with_proba(x_all)
            preds = preds.cpu()
            probs = probs.cpu()
        else:
            probs = _logits_to_probs(logits, model_name)
            preds = torch.argmax(probs, dim=-1)

        return (preds.numpy(), probs.numpy(),
                y_all.numpy() if y_all is not None else None)

    # ---- 深度学习模型：迭代预测 ----
    all_preds, all_probs, all_labels = [], [], []
    for batch in dataloader:
        if has_labels and isinstance(batch, (list, tuple)) and len(batch) >= 2:
            data, labels = batch[0].float().to(device), batch[1].long().to(device)
        else:
            data = batch[0].float().to(device) if isinstance(batch, (list, tuple)) else batch.float().to(device)
            labels = None

        output = model(data)
        logits = output[0] if isinstance(output, tuple) else output
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)

        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        if labels is not None:
            all_labels.append(labels.cpu().numpy())

    return (np.concatenate(all_preds),
            np.concatenate(all_probs),
            np.concatenate(all_labels) if all_labels else None)


def _logits_to_probs(logits: torch.Tensor, model_name: str) -> torch.Tensor:
    """将模型输出转换为概率分布。"""
    if len(logits.shape) == 1:
        logits = logits.unsqueeze(1)
        return torch.softmax(torch.cat([1 - logits, logits], dim=1), dim=1)

    if len(logits.shape) == 2 and logits.shape[1] == 1:
        logits = logits.squeeze(1).unsqueeze(1)
        return torch.softmax(torch.cat([1 - logits, logits], dim=1), dim=1)

    if len(logits.shape) == 2:
        if model_name in ML_MODELS_2D:
            row_sums = logits.sum(dim=1)
            if torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4):
                return logits
        return torch.softmax(logits, dim=1)

    raise ValueError(f"不支持的 logits 形状: {logits.shape}")


# ================================================================
# 对外函数：predict
# ================================================================
def predict(
    config: str,
    model: str,
    data_path: str,
    result_dict: dict,
    *,
    mapping: str = "configs/class_mapping.xlsx",
    no_diagnosis: bool = False,
) -> int:
    """
    统一预测入口（C 语言接口版）。

    Parameters
    ----------
    config : str
        YAML 配置文件路径。
    model : str
        模型名称，如 TSLANet / TS-TCC / RandomForest 等。
    data_path : str
        外部数据文件路径（.pt 文件，包含 samples 和可选 labels）。
    result_dict : dict
        输出容器。执行成功后写入:
            result_dict["result_dir"] = "<结果JSON文件路径>"
    mapping : str
        故障诊断映射 Excel 文件路径，默认 configs/class_mapping.xlsx。
    no_diagnosis : bool
        True 时禁用故障诊断映射。

    Returns
    -------
    int
        0 表示成功；非 0 为错误代码。

    Notes
    -----
    batch_size 从 config 中 training.batch_size 读取（默认 128）。
    device 从 config 中顶层 device 字段读取（默认 "cuda"）。
    """
    # ── 参数校验 ──
    if not isinstance(result_dict, dict):
        _log_error(ErrorCode.PREDICT_PARAM_ERROR, "result_dict 必须是 dict 类型")
        return ErrorCode.PREDICT_PARAM_ERROR.value

    if model not in ALL_MODELS:
        _log_error(ErrorCode.PREDICT_PARAM_ERROR,
                   f"不支持的模型 '{model}'，可选: {ALL_MODELS}")
        return ErrorCode.PREDICT_PARAM_ERROR.value

    if not os.path.exists(config):
        _log_error(ErrorCode.CONFIG_MISSING, config)
        return ErrorCode.CONFIG_MISSING.value

    if not os.path.exists(data_path):
        _log_error(ErrorCode.DATA_FILE_ERROR, f"文件不存在: {data_path}")
        return ErrorCode.DATA_FILE_ERROR.value

    # ── 加载 YAML 配置 ──
    try:
        with open(config, 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f)
    except Exception as e:
        _log_error(ErrorCode.CONFIG_MISSING, str(e))
        return ErrorCode.CONFIG_MISSING.value

    # ── 从配置读取 device / batch_size ──
    device = yaml_config.get('device', 'cuda')
    batch_size = yaml_config.get('training', {}).get('batch_size', 128)

    # ── 故障诊断映射 ──
    diagnosis_mapper = None
    if not no_diagnosis and os.path.exists(mapping):
        try:
            diagnosis_mapper = FaultDiagnosisMapper(mapping)
        except Exception as e:
            _log_error(ErrorCode.MAPPING_LOAD_ERROR, str(e))
            # 映射加载失败不阻断预测，继续执行

    # ── 确定 checkpoint 路径（从配置推断） ──
    # 约定: data_path 同级目录或 checkpoint 字段
    # 这里从 YAML 中读取 checkpoint，若无则尝试同目录下 ckp_last.pt
    checkpoint_path = yaml_config.get('checkpoint', None)
    if checkpoint_path is None:
        # 尝试从实验日志目录推断
        exp_desc = yaml_config.get('experiment', {}).get('description', 'Exp1')
        run_desc = yaml_config.get('experiment', {}).get('run_description', 'run1')
        seed = yaml_config.get('experiment', {}).get('seed', 123)
        training_mode = yaml_config.get('training', {}).get('mode', 'supervised')
        logs_dir = yaml_config.get('paths', {}).get('logs_save_dir', 'experiments_logs')
        candidate = os.path.join(
            logs_dir, exp_desc, run_desc,
            f"{training_mode}_seed_{seed}", "saved_models", "ckp_last.pt"
        )
        if os.path.exists(candidate):
            checkpoint_path = candidate
        else:
            _log_error(ErrorCode.CHECKPOINT_MISSING,
                       f"配置中未指定 checkpoint，且默认路径不存在: {candidate}")
            return ErrorCode.CHECKPOINT_MISSING.value

    if not os.path.exists(checkpoint_path):
        _log_error(ErrorCode.CHECKPOINT_MISSING, checkpoint_path)
        return ErrorCode.CHECKPOINT_MISSING.value

    # ── 设备 ──
    if model in ML_MODELS_2D:
        run_device = torch.device('cpu')
    else:
        run_device = torch.device(
            device if device == 'cuda' and torch.cuda.is_available() else 'cpu'
        )

    # ── 加载 checkpoint 获取模型配置 ──
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu',
                                weights_only=False)
        checkpoint_config = checkpoint.get('config', None)
    except Exception as e:
        _log_error(ErrorCode.MODEL_LOAD_ERROR, str(e))
        return ErrorCode.MODEL_LOAD_ERROR.value

    dataset_name = yaml_config.get('dataset', {}).get('selected', 'chaosu')
    if checkpoint_config is not None:
        model_config = checkpoint_config
    else:
        model_config = get_model_config(yaml_config, model, dataset_name)

    model_config['batch_size'] = batch_size

    # ── 创建模型并加载权重 ──
    try:
        model_instance, _ = ModelFactory.create_model(model, model_config,
                                                      run_device)
    except Exception as e:
        _log_error(ErrorCode.MODEL_LOAD_ERROR, str(e))
        return ErrorCode.MODEL_LOAD_ERROR.value

    try:
        if model in NON_ITERATIVE_MODELS:
            model_instance.load_checkpoint(checkpoint_path)
        else:
            model_instance.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        _log_error(ErrorCode.MODEL_LOAD_ERROR, str(e))
        return ErrorCode.MODEL_LOAD_ERROR.value

    model_instance = model_instance.to(run_device)

    # ── 加载数据 ──
    try:
        dataloader, has_labels = _build_dataloader(data_path, batch_size)
    except Exception as e:
        _log_error(ErrorCode.DATA_FILE_ERROR, str(e))
        return ErrorCode.DATA_FILE_ERROR.value

    # ── 执行预测 ──
    try:
        predictions, probabilities, true_labels = _predict_batch(
            model_instance, dataloader, run_device, model, has_labels
        )
    except Exception as e:
        _log_error(ErrorCode.PREDICTION_ERROR, str(e))
        return ErrorCode.PREDICTION_ERROR.value

    # ── 组装结果 ──
    num_classes = model_config.get('num_classes', len(np.unique(predictions)))
    max_probs = probabilities.max(axis=1)

    results = {
        'model_name': model,
        'checkpoint': checkpoint_path,
        'data_source': data_path,
        'num_samples': len(predictions),
        'num_classes': int(num_classes),
        'batch_size': batch_size,
        'timestamp': datetime.now().isoformat(),
        'predictions': predictions.tolist(),
        'probabilities': probabilities.tolist(),
        'confidence_stats': {
            'mean': float(max_probs.mean()),
            'min': float(max_probs.min()),
            'max': float(max_probs.max()),
            'std': float(max_probs.std()),
        },
    }

    if true_labels is not None:
        accuracy = float(np.mean(predictions == true_labels))
        results['true_labels'] = true_labels.tolist()
        results['accuracy'] = accuracy
        try:
            from sklearn.metrics import confusion_matrix
            results['confusion_matrix'] = confusion_matrix(
                true_labels, predictions
            ).tolist()
        except ImportError:
            pass

    # 故障诊断
    if diagnosis_mapper and diagnosis_mapper.has_mapping():
        diag_list = {}
        for i, pred_label in enumerate(predictions):
            diagnosis = diagnosis_mapper.get_diagnosis(int(pred_label))
            if diagnosis:
                entry = {
                    '预测标签': int(pred_label),
                    '置信度': float(probabilities[i, pred_label]),
                    '故障原因': diagnosis['故障原因'],
                    '处理方案': diagnosis['处理方案'],
                }
                if true_labels is not None:
                    entry['真实标签'] = int(true_labels[i])
                    entry['预测正确'] = bool(pred_label == true_labels[i])
                diag_list[str(i)] = entry
        results['diagnosis'] = diag_list

    # ── 保存结果 ──
    checkpoint_dir = os.path.dirname(checkpoint_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"predictions_{model}_{timestamp}.json"
    output_path = os.path.join(checkpoint_dir, output_filename)

    try:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    except Exception as e:
        _log_error(ErrorCode.SAVE_RESULT_ERROR, str(e))
        return ErrorCode.SAVE_RESULT_ERROR.value

    # ── 写入输出字典 ──
    result_dict["result_dir"] = output_path

    print(f"✓ 预测完成: {len(predictions)} 个样本, 结果已保存至 {output_path}")
    return 0


# ================================================================
# 命令行入口（也可用 subprocess 调用）
# ================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="统一预测接口（C 语言接口版）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, required=True,
                        help="YAML 配置文件路径")
    parser.add_argument("--model", type=str, required=True,
                        help="模型名称")
    parser.add_argument("--data_path", type=str, required=True,
                        help="外部数据文件路径(.pt)")
    parser.add_argument("--mapping", type=str,
                        default="configs/class_mapping.xlsx",
                        help="故障诊断映射 Excel 文件路径")
    parser.add_argument("--no_diagnosis", action="store_true",
                        help="禁用故障诊断")

    args = parser.parse_args()

    result_dict = {}
    code = predict(
        config=args.config,
        model=args.model,
        data_path=args.data_path,
        result_dict=result_dict,
        mapping=args.mapping,
        no_diagnosis=args.no_diagnosis,
    )

    if code == 0:
        print(f"结果路径: {result_dict['result_dir']}")
    raise SystemExit(code)
