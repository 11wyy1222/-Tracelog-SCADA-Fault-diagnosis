# 操作手册

## 简介

本操作手册为故障诊断机器学习平台的使用指南，涵盖数据预处理、模型训练和模型预测的完整工作流程。系统支持深度学习模型（TSLANet、TS-TCC等）和传统机器学习模型（RandomForest、SVM、XGBoost等），通过YAML配置文件实现灵活的参数管理。

本章节详细介绍数据预处理相关的四个核心脚本，帮助用户将原始CSV传感器数据转换为模型可用的张量数据。

---

## 1. 数据预处理脚本

### 1.1 preprocess_3D.py — 3D数据预处理（C语言接口版）

#### 功能说明

`preprocess_3D.py` 是3D数据预处理的核心入口脚本，位于 `api/` 目录下。该脚本提供C语言风格的接口设计，支持三种调用方式：纯配置文件、配置文件+参数覆盖、纯参数覆盖。

主要功能包括：
- 从原始CSV传感器数据中提取目标特征（如转子转速等）
- 处理缺失值（支持插值、前向填充、后向填充、均值填充、零值填充）
- 统一序列长度（截断或填充至指定时间步数）
- 数据标准化（支持zscore、minmax、robust方法）
- 按比例划分训练集、验证集和测试集
- 保存为 `.pt` 格式的张量文件

参数优先级（高→低）：
1. `overrides` 中的参数（最高优先级）
2. `config_file` 指定的YAML配置文件
3. 代码内置默认值（兜底）

#### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--config` | str | `""` | YAML配置文件路径（如 `configs/preprocess.yaml`） |
| `--label_file` | str | None | 标签文件路径 |
| `--output_dir` | str | None | 输出目录路径 |
| `--target_tags` | str (多个) | None | 目标特征标签列表 |
| `--max_seq_len` | int | None | 统一的时间步数（默认12000） |
| `--missing_value_method` | str | None | 空值处理方法：`interpolate`/`ffill`/`bfill`/`mean`/`zero` |
| `--padding_mode` | str | None | 填充模式：`constant`/`edge`/`reflect` |
| `--norm_method` | str | None | 标准化方法：`zscore`/`minmax`/`robust`/`none` |
| `--save_stats` | bool | None | 是否保存标准化统计量 |
| `--stats_filename` | str | None | 统计量保存文件名 |
| `--test_size` | float | None | 测试+验证集比例 |
| `--val_test_split` | float | None | 验证集在(测试+验证)中的比例 |
| `--random_state` | int | None | 随机种子 |
| `--stratify` | bool | None | 是否按标签分层抽样 |
| `--train_file` | str | None | 训练集输出文件名 |
| `--val_file` | str | None | 验证集输出文件名 |
| `--test_file` | str | None | 测试集输出文件名 |
| `--verbose` | bool | None | 是否详细输出 |
| `--save_log` | bool | None | 是否保存日志 |
| `--log_file` | str | None | 日志文件名 |

#### 输入输出格式

**输入：**
- YAML配置文件（如 `configs/preprocess.yaml`），包含路径、特征、处理、标准化、划分等配置
- 原始CSV传感器数据文件（通过标签文件索引）

**输出：**
- `train.pt` — 训练集张量文件
- `val.pt` — 验证集张量文件
- `test.pt` — 测试集张量文件

每个 `.pt` 文件包含：
```python
{
    'samples': torch.Tensor,  # 样本特征张量
    'labels': torch.Tensor    # 标签张量
}
```

**错误编码：**

| 错误码 | 名称 | 说明 |
|--------|------|------|
| 1001 | CONFIG_LOAD_ERROR | 配置文件加载失败 |
| 1002 | CONFIG_OVERRIDE_ERROR | overrides参数应用异常 |
| 1003 | PREPROCESS_INIT_ERROR | DataPreprocessor初始化失败 |
| 1004 | PROCESS_ERROR | 数据预处理过程出错 |
| 1005 | SPLIT_NORMALIZE_ERROR | 划分或标准化数据时出错 |
| 1006 | SAVE_DATASET_ERROR | 保存.pt文件失败 |
| 1999 | UNKNOWN_ERROR | 未知错误 |

#### 使用示例

```bash
# 示例1: 使用配置文件执行预处理
python api/preprocess_3D.py --config configs/preprocess.yaml

# 示例2: 配置文件 + 覆盖部分参数
python api/preprocess_3D.py \
    --config configs/preprocess.yaml \
    --max_seq_len 6000 \
    --output_dir /tmp/output

# 示例3: 覆盖多个参数
python api/preprocess_3D.py \
    --config configs/preprocess.yaml \
    --norm_method minmax \
    --test_size 0.3 \
    --random_state 123

# 示例4: 指定目标特征标签
python api/preprocess_3D.py \
    --config configs/preprocess.yaml \
    --target_tags grRotorSpeedFormSpeedRelay1 grRotorSpeedFormSpeedRelay2
```

**Python API调用方式：**

```python
from api.preprocess_3D import preprocess, build_overrides

# 方式1: 纯配置文件
preprocess(config_file="configs/preprocess.yaml")

# 方式2: 配置文件 + 覆盖参数
err, overrides = build_overrides(max_seq_len=6000, output_dir="/tmp/out")
if err == 0:
    preprocess(config_file="configs/preprocess.yaml", overrides=overrides)

# 方式3: 完全不用配置文件，全靠overrides
err, overrides = build_overrides(
    label_file="/data/label.csv",
    output_dir="/data/out",
    max_seq_len=8000,
)
if err == 0:
    preprocess(overrides=overrides)
```

---

### 1.2 preprocess_2D.py — 2D数据预处理

#### 功能说明

`preprocess_2D.py` 位于项目根目录，是2D数据预处理的入口脚本。该脚本从Excel索引文件读取原始数据路径，通过 `FaultDiagnosisPipeline` 批量提取特征，进行数据清洗和标签编码，按 **8:1:1** 的比例生成训练集、验证集和测试集，并保存为 `.pt` 格式文件。

主要功能包括：
- 从Excel数据清单读取tracelog文件路径和标签
- 使用 `FaultDiagnosisPipeline` 进行批量特征提取（特征维度为329或569）
- 按8:1:1比例划分数据集（先分出10%测试集，再从剩余中分出1/9验证集）
- 保存为包含 `samples` 和 `labels` 张量的 `.pt` 文件

#### 命令行参数

该脚本不接受命令行参数，所有配置通过 `configs/config_loader.py` 中的常量导入：

| 配置项 | 来源 | 说明 |
|--------|------|------|
| `DATA_SOURCE_EXCEL` | config_loader.py | 数据源Excel文件路径 |
| `PROCESSED_DATA_DIR` | config_loader.py | 处理后数据输出目录 |
| `COL_TRACELOG_PATH` | config_loader.py | Excel中tracelog路径列名 |
| `COL_LABEL` | config_loader.py | Excel中标签列名 |
| `RANDOM_SEED` | config_loader.py | 随机种子（默认42） |

#### 输入输出格式

**输入：**
- Excel数据清单文件（包含tracelog路径列和标签列）
- 原始CSV传感器数据文件（tracelog文件）

**输出：**
- `train.pt` — 训练集（约80%样本）
- `val.pt` — 验证集（约10%样本）
- `test.pt` — 测试集（约10%样本）

每个 `.pt` 文件包含：
```python
{
    'samples': torch.Tensor,  # 样本特征张量
    'labels': torch.Tensor    # 标签张量
}
```

#### 使用示例

```bash
# 直接运行（使用配置文件中的默认路径）
python preprocess_2D.py
```

> **注意：** 运行前请确保 `configs/preprocess_2D.yaml` 中的 `data.source_excel` 和 `data.processed_dir` 路径配置正确。

---

### 1.3 prepare_dp_data.py — 深度学习预测数据准备

#### 功能说明

`prepare_dp_data.py` 位于项目根目录，用于将单个tracelog CSV文件转换为深度学习模型可用的 `.pt` 预测数据文件。该脚本使用与训练数据预处理相同的逻辑，确保预测数据与训练数据格式一致。

主要功能包括：
- 加载训练时使用的预处理配置（`preprocess.yaml`）
- 根据同义词组配置匹配CSV中的特征列
- 处理缺失值和无穷值
- 调整序列长度（截断或填充至 `max_seq_len`）
- 使用训练时保存的标准化统计量进行数据标准化
- 保存为 `.pt` 格式文件，可直接用于 `predict.py` 预测

#### 命令行参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--tracelog` | str | 是 | — | tracelog CSV文件路径 |
| `--config` | str | 是 | — | 预处理配置文件路径（与训练时使用的配置相同） |
| `--output` | str | 否 | `./external_data/predict_data.pt` | 输出 `.pt` 文件路径 |
| `--include-label` | flag | 否 | False | 是否包含虚拟标签（某些预测脚本可能需要） |

#### 输入输出格式

**输入：**
- tracelog CSV文件：包含传感器时序数据的CSV文件
- 预处理配置文件（YAML格式）：与训练时使用的配置相同

**输出：**
- `.pt` 文件，包含：
```python
{
    'samples': torch.Tensor  # 形状为 (1, n_features, seq_len) 的特征张量
}
# 如果指定 --include-label：
{
    'samples': torch.Tensor,  # 形状为 (1, n_features, seq_len) 的特征张量
    'labels': torch.Tensor    # 虚拟标签张量 [0]
}
```

**错误编码：**

| 错误码 | 名称 | 说明 |
|--------|------|------|
| 4101 | CONFIG_LOAD_ERROR | 配置文件加载失败 |
| 4102 | TRACELOG_NOT_FOUND | tracelog文件不存在 |
| 4103 | PROCESS_ERROR | tracelog处理过程出错 |
| 4104 | SAVE_ERROR | 保存.pt文件失败 |
| 4199 | UNKNOWN_ERROR | 未知错误 |

#### 使用示例

```bash
# 示例1: 基本用法 — 处理单个tracelog文件
python prepare_dp_data.py \
    --tracelog data_preprocessing/Tracelog/24-01-10-07_49_13_830_01010_SafetyRotorOv.csv \
    --config configs/preprocess.yaml \
    --output ./external_data/predict_data.pt

# 示例2: 包含虚拟标签（某些预测脚本需要labels字段）
python prepare_dp_data.py \
    --tracelog path/to/tracelog.csv \
    --config configs/preprocess.yaml \
    --output ./external_data/predict_data.pt \
    --include-label

# 示例3: 处理完成后进行预测
python prepare_dp_data.py \
    --tracelog path/to/tracelog.csv \
    --config configs/preprocess.yaml \
    --output ./external_data/predict_data.pt
# 然后使用predict.py进行预测：
# python predict.py --data_file ./external_data/predict_data.pt ...
```

---

### 1.4 prepare_ml_data.py — 传统机器学习预测数据准备

#### 功能说明

`prepare_ml_data.py` 位于项目根目录，用于批量处理传感器数据并为传统机器学习模型（如RandomForest、SVM、XGBoost等）准备预测数据。该脚本从Excel数据清单或目录中读取CSV文件，通过 `FaultDiagnosisPipeline` 提取特征向量，并保存为 `.pt` 格式文件。

主要功能包括：
- 从Excel文件或目录批量加载tracelog CSV数据
- 使用 `FaultDiagnosisPipeline` 逐个提取特征向量
- 将特征矩阵保存为 `.pt` 文件（仅包含 `samples` 张量）
- 同时保存文件名映射CSV（用于结果追溯）
- 支持加载已训练的ML模型进行批量预测和可视化

核心组件：
- `TestDataLoader`：负责从Excel或目录加载测试数据路径，逐个读取原始传感器数据
- `ModelTester`：负责加载ML模型（`.pt`格式），执行预测并生成可视化报告
- `run_pipeline_inference()`：使用Pipeline批量提取特征并可选保存为 `.pt` 文件

#### 命令行参数

该脚本不接受命令行参数，所有配置通过 `configs/config_loader.py` 中的常量导入：

| 配置项 | 来源 | 说明 |
|--------|------|------|
| `DATA_SOURCE_TEST_EXCEL` | config_loader.py | 测试数据源Excel文件路径或目录路径 |
| `MODEL_SAVE_DIR` | config_loader.py | 模型保存目录 |
| `MODEL_FILENAME` | config_loader.py | 模型文件名 |
| `PREDICTION_RESULT_CSV` | config_loader.py | 预测结果CSV输出路径 |
| `PREDICTION_REPORT_IMG` | config_loader.py | 预测报告图片输出路径 |
| `DEFAULT_FAULT_TIME` | config_loader.py | 默认故障时间点 |
| `TIMESTAMP_WINDOW` | config_loader.py | 时间戳截取窗口 |
| `COL_TRACELOG_PATH` | config_loader.py | Excel中tracelog路径列名 |

#### 输入输出格式

**输入：**
- Excel数据清单文件（包含tracelog路径列）或包含CSV文件的目录
- 原始CSV传感器数据文件

**输出：**
- `prediction_features.pt` — 特征张量文件：
```python
{
    'samples': torch.Tensor  # 形状为 (N, feature_dim) 的特征矩阵
}
```
- `prediction_features_filenames.csv` — 文件名映射表：
```
index,filename
0,sample_001.csv
1,sample_002.csv
...
```

#### 使用示例

```bash
# 直接运行（使用配置文件中的默认路径）
python prepare_ml_data.py
```

> **注意：** 运行前请确保 `configs/preprocess_2D.yaml` 中的以下配置正确：
> - `data.source_test_excel`：测试数据源路径（Excel文件或CSV目录）
> - `model.save_dir`：模型和输出文件保存目录
> - `model.filename`：ML模型文件名

**Python API调用方式：**

```python
from prepare_ml_data import TestDataLoader, run_pipeline_inference

# 步骤1: 加载数据
loader = TestDataLoader("path/to/test_data.xlsx")
batch_data = loader.load_batch()

# 步骤2: 提取特征并保存为PT文件
X, filenames = run_pipeline_inference(
    batch_data,
    save_pt_path="./output/prediction_features.pt"
)

print(f"成功处理 {len(filenames)} 个样本，特征维度: {X.shape}")
```


---

## 2. 模型训练脚本

### 2.1 train.py — 统一训练脚本

#### 功能说明

`train.py` 位于项目根目录，是模型训练的统一入口脚本。该脚本支持深度学习模型（TSLANet、TS-TCC、base_CNN）、储层计算模型（RC）和传统机器学习模型（RandomForest、SVM、XGBoost、LightGBM、KNN、LogisticRegression），通过YAML配置文件和命令行参数灵活控制训练流程。

主要功能包括：
- 从YAML配置文件加载实验参数，支持命令行参数覆盖
- 支持五种训练模式：随机初始化、监督学习、自监督预训练、微调、线性探测
- 根据模型类型自动选择训练器（UnifiedTrainer / MLTrainer / RCTrainer）
- 训练完成后在测试集上评估模型（自监督模式除外）
- 自动保存模型检查点（`ckp_best.pt`、`ckp_last.pt`）、训练日志和配置文件副本

参数优先级（高→低）：
1. 命令行参数（最高优先级）
2. YAML配置文件中的值
3. 代码内置默认值（兜底）

#### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--config` | str | `config.yaml` | YAML配置文件路径（如 `configs/config.yaml`） |
| `--model` | str | None | 模型名称，覆盖配置文件中的 `experiment.model` |
| `--dataset` | str | None | 数据集名称，覆盖配置文件中的 `dataset.selected` |
| `--training_mode` | str | None | 训练模式，覆盖配置文件中的 `training.mode` |
| `--device` | str | None | 运行设备（`cuda` 或 `cpu`），覆盖配置文件中的 `device` |
| `--seed` | int | None | 随机种子，覆盖配置文件中的 `experiment.seed` |
| `--batch_size` | int | None | 批次大小，覆盖配置文件中的 `training.batch_size` |
| `--num_epochs` | int | None | 训练轮数，覆盖配置文件中的 `training.num_epochs` |

#### 支持的训练模式

| 训练模式 | 说明 | 适用场景 |
|----------|------|----------|
| `random_init` | 随机初始化模型权重，冻结指定层后训练 | 基线对比实验 |
| `supervised` | 标准监督学习，使用带标签数据端到端训练 | 常规模型训练 |
| `self_supervised` | 自监督预训练，无需标签学习数据表征 | 预训练阶段（如TS-TCC对比学习） |
| `fine_tune` | 加载预训练权重后全参数微调 | 自监督预训练后的下游任务微调 |
| `train_linear` | 加载预训练权重，冻结主干网络，仅训练分类头 | 线性探测评估预训练质量 |

#### 支持的模型

| 模型类别 | 模型名称 | 训练器 | 输入格式 | 说明 |
|----------|----------|--------|----------|------|
| 深度学习 | `TSLANet` | UnifiedTrainer | 3D `[N, C, L]` | 时间序列自适应轻量网络 |
| 深度学习 | `TS-TCC` | UnifiedTrainer | 3D `[N, C, L]` | 时间序列对比学习模型 |
| 深度学习 | `base_CNN` | UnifiedTrainer | 3D `[N, C, L]` | 基础卷积神经网络 |
| 储层计算 | `RC` | RCTrainer | 3D `[N, C, L]` | 储层计算模型 |
| 传统ML | `RandomForest` | MLTrainer | 2D（自动展平） | 随机森林 |
| 传统ML | `SVM` | MLTrainer | 2D（自动展平） | 支持向量机 |
| 传统ML | `XGBoost` | MLTrainer | 2D（自动展平） | 梯度提升树 |
| 传统ML | `LightGBM` | MLTrainer | 2D（自动展平） | 轻量梯度提升 |
| 传统ML | `KNN` | MLTrainer | 2D（自动展平） | K近邻分类器 |
| 传统ML | `LogisticRegression` | MLTrainer | 2D（自动展平） | 逻辑回归 |

> **注意：** 传统ML模型自动在CPU上运行，输入数据会自动从3D展平为2D格式。

#### 输出目录结构

训练完成后，输出文件保存在以下目录结构中：

```
experiments_logs/
└── {experiment_description}/
    └── {run_description}/
        └── {training_mode}_seed_{seed}/
            ├── config.yaml                  # 配置文件副本
            ├── logs_{timestamp}.log         # 训练日志
            ├── metrics.csv                  # 训练指标（监督模式）
            ├── classification_report.xlsx   # 分类报告（监督模式）
            ├── confusion_matrix.png         # 混淆矩阵图（监督模式）
            └── saved_models/
                ├── ckp_best.pt              # 最佳模型检查点
                └── ckp_last.pt              # 最后一轮检查点
```

#### 错误编码

| 错误码 | 名称 | 说明 |
|--------|------|------|
| 2101 | CONFIG_LOAD_ERROR | 加载YAML配置失败，可能文件不存在或格式有误 |
| 2102 | DATA_LOADER_ERROR | 数据集加载或DataLoader创建失败 |
| 2103 | MODEL_CREATION_ERROR | 模型实例化或模型配置获取出错 |
| 2104 | PRETRAIN_LOAD_ERROR | 加载预训练模型失败，检查路径和格式 |
| 2105 | TRAINING_ERROR | 训练过程中发生异常 |
| 2106 | EVALUATION_ERROR | 测试/评估阶段出错 |
| 2107 | CONFIG_SAVE_ERROR | 配置文件保存失败 |
| 2999 | UNKNOWN_ERROR | 未知错误，请检查日志 |

#### 使用示例

**示例1: TSLANet 监督学习训练**

```bash
# 使用配置文件（确保config.yaml中设置了model和training.mode）
python train.py --config configs/config.yaml

# 使用命令行参数覆盖配置
python train.py --config configs/config.yaml \
    --model TSLANet \
    --training_mode supervised \
    --dataset chaosu \
    --num_epochs 100 \
    --batch_size 16

# 指定设备和随机种子
python train.py --config configs/config.yaml \
    --model TSLANet \
    --training_mode supervised \
    --device cuda \
    --seed 42
```

**示例2: 自监督预训练 + 微调（两阶段训练）**

```bash
# 阶段1: 自监督预训练
# 模型通过对比学习等方式学习数据表征，不需要标签
python train.py --config configs/config.yaml \
    --model TSLANet \
    --training_mode self_supervised \
    --num_epochs 50

# 预训练完成后，检查点保存在:
# experiments_logs/Exp1/run1/self_supervised_seed_123/saved_models/ckp_last.pt

# 阶段2: 微调
# 加载预训练权重，使用带标签数据进行全参数微调
python train.py --config configs/config.yaml \
    --model TSLANet \
    --training_mode fine_tune \
    --num_epochs 100

# 微调时会自动从以下路径加载预训练模型:
# experiments_logs/{experiment}/{run}/self_supervised_seed_{seed}/saved_models/ckp_last.pt
# 也可以在config.yaml中通过 pretraining.pretrained_path 指定自定义路径
```

> **提示：** 如果需要指定预训练模型路径，在 `configs/config.yaml` 中设置：
> ```yaml
> pretraining:
>   load_pretrained: true
>   pretrained_path: 'experiments_logs/Exp1/run1/self_supervised_seed_123/saved_models'
> ```

**示例3: 线性探测（评估预训练质量）**

```bash
# 冻结主干网络，仅训练分类头
python train.py --config configs/config.yaml \
    --model TSLANet \
    --training_mode train_linear \
    --num_epochs 50
```

**示例4: 传统机器学习模型训练**

```bash
# 随机森林
python train.py --config configs/config.yaml \
    --model RandomForest \
    --training_mode supervised

# SVM
python train.py --config configs/config.yaml \
    --model SVM \
    --training_mode supervised

# XGBoost
python train.py --config configs/config.yaml \
    --model XGBoost \
    --training_mode supervised

# LightGBM
python train.py --config configs/config.yaml \
    --model LightGBM \
    --training_mode supervised

# KNN
python train.py --config configs/config.yaml \
    --model KNN \
    --training_mode supervised

# 逻辑回归
python train.py --config configs/config.yaml \
    --model LogisticRegression \
    --training_mode supervised
```

> **注意：** 传统ML模型使用 `MLTrainer`，自动将3D输入数据展平为2D格式，且始终在CPU上运行。模型超参数通过 `configs/config.yaml` 中对应模型名称的配置节设置，例如：
> ```yaml
> RandomForest:
>   n_estimators: 100
>   max_depth: null
>   min_samples_split: 2
>   random_state: 42
> ```

**示例5: 随机初始化基线实验**

```bash
# 随机初始化模型权重，冻结指定层后训练（用于基线对比）
python train.py --config configs/config.yaml \
    --model TSLANet \
    --training_mode random_init \
    --num_epochs 100
```


---

## 3. 模型预测脚本

### 3.1 predict.py — 统一预测脚本（支持故障诊断）

#### 功能说明

`predict.py` 位于项目根目录，是模型预测的统一入口脚本。该脚本支持深度学习模型（TSLANet、TS-TCC、base_CNN）、储层计算模型（RC）和传统机器学习模型（RandomForest、SVM、XGBoost、LightGBM、KNN、LogisticRegression），并集成了故障诊断原因和处理方案的输出功能。

主要功能包括：
- 加载训练好的模型检查点，执行批量预测
- 支持两种数据输入模式：内置数据集预测和外部数据文件预测
- 自动识别模型类型，选择对应的预测策略（迭代预测 / 批量预测）
- 集成故障诊断映射，将预测标签映射为故障原因和处理方案
- 计算准确率、混淆矩阵、置信度统计等评估指标（有标签时）
- 将预测结果保存为结构化JSON文件

参数优先级（高→低）：
1. 命令行参数（最高优先级）
2. 检查点中保存的模型配置
3. YAML配置文件中的值（兜底）

#### 命令行参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--checkpoint` | str | 是 | — | 模型检查点路径（`.pt` 文件） |
| `--config` | str | 是 | — | 配置文件路径（`.yaml` 文件） |
| `--model` | str | 否 | None | 模型名称（如不指定则从检查点中读取） |
| `--mapping` | str | 否 | `class_mapping.xlsx` | 故障诊断映射Excel文件路径 |
| `--no-diagnosis` | flag | 否 | False | 禁用故障诊断功能 |
| `--data_file` | str | 否 | None | 外部数据文件路径（`.pt` 文件），与内置数据集二选一 |
| `--dataset` | str | 否 | `chaosu` | 数据集名称（仅在不使用 `--data_file` 时有效） |
| `--data_path` | str | 否 | `./data` | 数据根目录（仅在不使用 `--data_file` 时有效） |
| `--mode` | str | 否 | `test` | 预测哪个数据集：`train`/`val`/`test`（仅在不使用 `--data_file` 时有效） |
| `--device` | str | 否 | `cuda` | 运行设备：`cuda`/`cpu` |
| `--output` | str | 否 | None | 输出文件路径（默认保存在检查点同目录下） |
| `--batch_size` | int | 否 | `128` | 批处理大小 |
| `--show_samples` | int | 否 | `5` | 在终端显示前N个样本的预测结果 |
| `--show_diagnosis` | flag | 否 | False | 为每个样本显示详细的故障诊断信息 |

#### 两种预测模式

##### 模式1: 内置数据集预测

使用项目中已预处理好的数据集（如 `data/chaosu/`）进行预测。通过 `--dataset` 和 `--mode` 参数指定数据集和数据划分。此模式下数据包含真实标签，可计算准确率和混淆矩阵。

```bash
# 使用内置数据集的测试集进行预测
python predict.py \
    --checkpoint experiments_logs/Exp1/run1/supervised_seed_123/saved_models/ckp_best.pt \
    --config configs/config.yaml \
    --dataset chaosu \
    --mode test

# 使用验证集进行预测
python predict.py \
    --checkpoint experiments_logs/Exp1/run1/supervised_seed_123/saved_models/ckp_best.pt \
    --config configs/config.yaml \
    --dataset chaosu \
    --mode val
```

##### 模式2: 外部数据文件预测

使用 `--data_file` 参数指定外部 `.pt` 数据文件进行预测。外部数据文件通常由 `prepare_dp_data.py` 或 `prepare_ml_data.py` 生成。支持有标签和无标签两种数据格式：

- 有标签格式：`.pt` 文件包含 `samples` 和 `labels` 字段
- 无标签格式：`.pt` 文件仅包含 `samples` 字段（纯预测模式，不计算准确率）

```bash
# 使用外部数据文件进行预测（无标签，纯预测模式）
python predict.py \
    --checkpoint experiments_logs/Exp1/run1/supervised_seed_123/saved_models/ckp_best.pt \
    --config configs/config.yaml \
    --data_file ./external_data/predict_data.pt

# 使用带标签的外部数据文件（可计算准确率）
python predict.py \
    --checkpoint experiments_logs/Exp1/run1/supervised_seed_123/saved_models/ckp_best.pt \
    --config configs/config.yaml \
    --data_file ./external_data/labeled_data.pt
```

> **提示：** 使用 `prepare_dp_data.py` 生成深度学习模型的预测数据，使用 `prepare_ml_data.py` 生成传统ML模型的预测数据。如需在外部数据中包含虚拟标签，可在 `prepare_dp_data.py` 中使用 `--include-label` 参数。

#### 故障诊断映射功能

预测脚本集成了 `FaultDiagnosisMapper` 模块，可将数字预测标签映射为可读的故障原因和处理方案。

**映射文件格式：**

映射关系通过Excel文件（如 `configs/class_mapping.xlsx`）定义，文件必须包含以下三列：

| 列名 | 类型 | 说明 |
|------|------|------|
| `标签` | int | 数字标签（与模型输出的预测标签对应） |
| `故障原因` | str | 故障原因描述 |
| `处理方案` | str | 处理方案说明（支持 `\n` 换行） |

**使用方式：**

- 默认启用故障诊断，映射文件路径为 `class_mapping.xlsx`
- 通过 `--mapping` 参数指定自定义映射文件路径
- 通过 `--no-diagnosis` 参数禁用故障诊断功能
- 通过 `--show_diagnosis` 参数在终端为每个样本显示详细诊断信息

```bash
# 启用故障诊断并显示详细信息
python predict.py \
    --checkpoint experiments_logs/Exp1/run1/supervised_seed_123/saved_models/ckp_best.pt \
    --config configs/config.yaml \
    --dataset chaosu \
    --mode test \
    --mapping configs/class_mapping.xlsx \
    --show_samples 10 \
    --show_diagnosis

# 禁用故障诊断
python predict.py \
    --checkpoint experiments_logs/Exp1/run1/supervised_seed_123/saved_models/ckp_best.pt \
    --config configs/config.yaml \
    --dataset chaosu \
    --no-diagnosis
```

#### 输出JSON格式说明

预测完成后，结果保存为JSON文件。默认保存路径为检查点所在目录，文件名格式为 `predictions_{模型名}_{模式}_{时间戳}.json`。也可通过 `--output` 参数指定自定义输出路径。

**JSON文件结构：**

```json
{
  "predictions": [0, 1, 2, 0, 3],
  "probabilities": [[0.95, 0.03, 0.01, 0.01], ...],
  "num_samples": 50,
  "num_classes": 4,
  "dataset": "chaosu",
  "data_source": "内置数据集 (test): ./data/chaosu",
  "mode": "test",
  "timestamp": "2025-01-19T15:30:00.000000",
  "model_name": "TSLANet",
  "checkpoint": "experiments_logs/.../ckp_best.pt",
  "batch_size": 128,
  "true_labels": [0, 1, 2, 0, 3],
  "accuracy": 0.96,
  "confusion_matrix": [[12, 1, 0, 0], ...],
  "diagnosis": {
    "0": {
      "预测标签": 0,
      "置信度": 0.9523,
      "故障原因": "滑环编码器转速跳变",
      "处理方案": "1、检查梅花联轴器有无松动或损坏\n2、检查滑环编码器线束固定有无松动",
      "真实标签": 0,
      "预测正确": true
    },
    "1": {
      "预测标签": 1,
      "置信度": 0.8871,
      "故障原因": "滑环编码器转速规律波动",
      "处理方案": "1、检查梅花联轴器有无松动或损坏\n2、检查滑环编码器屏蔽线有无异常",
      "真实标签": 1,
      "预测正确": true
    }
  }
}
```

**字段说明：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `predictions` | list[int] | 每个样本的预测标签列表 |
| `probabilities` | list[list[float]] | 每个样本的概率分布（每行之和为1） |
| `num_samples` | int | 总样本数 |
| `num_classes` | int | 类别数 |
| `dataset` | str | 数据集名称（外部数据时为 `"external"`） |
| `data_source` | str | 数据来源描述 |
| `mode` | str | 数据划分模式（`train`/`val`/`test`） |
| `timestamp` | str | 预测执行时间（ISO 8601格式） |
| `model_name` | str | 模型名称 |
| `checkpoint` | str | 使用的检查点路径 |
| `batch_size` | int | 批处理大小 |
| `true_labels` | list[int] | 真实标签列表（仅有标签数据时存在） |
| `accuracy` | float | 预测准确率（仅有标签数据时存在） |
| `confusion_matrix` | list[list[int]] | 混淆矩阵（仅有标签数据时存在） |
| `diagnosis` | dict | 故障诊断信息（仅启用诊断时存在），键为样本索引 |

> **注意：** `true_labels`、`accuracy`、`confusion_matrix` 字段仅在数据包含真实标签时出现。`diagnosis` 字段仅在启用故障诊断且映射文件有效时出现。

#### 错误编码

| 错误码 | 名称 | 说明 |
|--------|------|------|
| 3101 | CHECKPOINT_MISSING | 检查点文件不存在或无法访问 |
| 3102 | CONFIG_MISSING | 配置文件缺失或格式错误，加载失败 |
| 3103 | MAPPING_LOAD_ERROR | 故障诊断映射文件读取失败 |
| 3104 | DATA_FILE_ERROR | 外部数据文件缺失或内容不符合预期 |
| 3105 | DATALOADER_ERROR | 内置数据集或DataLoader创建错误 |
| 3106 | MODEL_CREATION_ERROR | 模型实例化或权重加载发生错误 |
| 3107 | PREDICTION_ERROR | 预测过程中出现异常 |
| 3108 | SAVE_RESULT_ERROR | 保存预测结果失败（文件写入出错） |
| 3999 | UNKNOWN_ERROR | 未知错误，请查看详细日志 |

#### 使用示例

**示例1: 内置数据集预测（带故障诊断）**

```bash
python predict.py \
    --checkpoint experiments_logs/Exp1/run1/supervised_seed_123/saved_models/ckp_best.pt \
    --config configs/config.yaml \
    --dataset chaosu \
    --mode test \
    --mapping configs/class_mapping.xlsx \
    --show_samples 10 \
    --show_diagnosis
```

**示例2: 外部数据文件预测**

```bash
# 步骤1: 准备预测数据（深度学习模型）
python prepare_dp_data.py \
    --tracelog path/to/tracelog.csv \
    --config configs/preprocess.yaml \
    --output ./external_data/predict_data.pt

# 步骤2: 执行预测
python predict.py \
    --checkpoint experiments_logs/Exp1/run1/supervised_seed_123/saved_models/ckp_best.pt \
    --config configs/config.yaml \
    --data_file ./external_data/predict_data.pt \
    --mapping configs/class_mapping.xlsx \
    --show_diagnosis
```

**示例3: 传统机器学习模型预测**

```bash
python predict.py \
    --checkpoint experiments_logs/Exp1/run1/supervised_seed_123/saved_models/ckp_best.pt \
    --config configs/config.yaml \
    --model RandomForest \
    --dataset chaosu \
    --mode test
```

> **注意：** 传统ML模型自动在CPU上运行，输入数据会自动从3D展平为2D格式。

**示例4: 自定义输出路径和批处理大小**

```bash
python predict.py \
    --checkpoint experiments_logs/Exp1/run1/supervised_seed_123/saved_models/ckp_best.pt \
    --config configs/config.yaml \
    --dataset chaosu \
    --output ./results/my_predictions.json \
    --batch_size 64 \
    --show_samples 20
```

**示例5: 禁用故障诊断，仅获取预测结果**

```bash
python predict.py \
    --checkpoint experiments_logs/Exp1/run1/supervised_seed_123/saved_models/ckp_best.pt \
    --config configs/config.yaml \
    --dataset chaosu \
    --no-diagnosis
```


---

## 4. 典型工作流程

本节介绍系统的三种典型工作流程，帮助用户快速上手完整的故障诊断建模流程。

### 4.1 标准工作流程：数据预处理 → 监督训练 → 模型预测

这是最常用的工作流程，适用于已有标注数据的场景。

**步骤1: 数据预处理**

根据模型类型选择对应的预处理脚本：

```bash
# 深度学习模型（3D数据）：使用 preprocess_3D.py
python api/preprocess_3D.py --config configs/preprocess.yaml

# 传统机器学习模型（2D数据）：使用 preprocess_2D.py
python preprocess_2D.py
```

预处理完成后，输出目录下会生成 `train.pt`、`val.pt`、`test.pt` 三个文件。

**步骤2: 模型训练**

```bash
# 使用TSLANet进行监督学习训练
python train.py --config configs/config.yaml \
    --model TSLANet \
    --training_mode supervised \
    --num_epochs 100 \
    --batch_size 16
```

训练完成后，模型检查点保存在 `experiments_logs/{实验名}/{运行名}/supervised_seed_{种子}/saved_models/` 目录下。

**步骤3: 模型预测**

```bash
# 使用内置测试集进行预测评估
python predict.py \
    --checkpoint experiments_logs/Exp1/run1/supervised_seed_123/saved_models/ckp_best.pt \
    --config configs/config.yaml \
    --dataset chaosu \
    --mode test \
    --mapping configs/class_mapping.xlsx \
    --show_diagnosis
```

如需对新数据进行预测，先准备预测数据，再执行预测：

```bash
# 准备深度学习模型的预测数据
python prepare_dp_data.py \
    --tracelog path/to/new_tracelog.csv \
    --config configs/preprocess.yaml \
    --output ./external_data/predict_data.pt

# 执行预测
python predict.py \
    --checkpoint experiments_logs/Exp1/run1/supervised_seed_123/saved_models/ckp_best.pt \
    --config configs/config.yaml \
    --data_file ./external_data/predict_data.pt \
    --mapping configs/class_mapping.xlsx \
    --show_diagnosis
```

---

### 4.2 高级工作流程：自监督预训练 → 微调

当标注数据有限但无标签数据充足时，可先通过自监督学习让模型学习数据表征，再用少量标注数据微调。

**步骤1: 数据预处理**

```bash
python api/preprocess_3D.py --config configs/preprocess.yaml
```

**步骤2: 自监督预训练**

模型通过对比学习等方式从无标签数据中学习特征表示，不需要标签信息。

```bash
python train.py --config configs/config.yaml \
    --model TSLANet \
    --training_mode self_supervised \
    --num_epochs 50
```

预训练检查点保存在：
`experiments_logs/{实验名}/{运行名}/self_supervised_seed_{种子}/saved_models/ckp_last.pt`

**步骤3: 微调**

加载预训练权重，使用带标签数据进行全参数微调。

```bash
python train.py --config configs/config.yaml \
    --model TSLANet \
    --training_mode fine_tune \
    --num_epochs 100
```

> **提示：** 微调时会自动从对应的自监督预训练目录加载权重。如需指定自定义路径，在 `configs/config.yaml` 中设置：
> ```yaml
> pretraining:
>   load_pretrained: true
>   pretrained_path: 'experiments_logs/Exp1/run1/self_supervised_seed_123/saved_models'
> ```

**步骤4: 模型预测**

```bash
python predict.py \
    --checkpoint experiments_logs/Exp1/run1/fine_tune_seed_123/saved_models/ckp_best.pt \
    --config configs/config.yaml \
    --dataset chaosu \
    --mode test \
    --mapping configs/class_mapping.xlsx \
    --show_diagnosis
```

**可选：线性探测评估**

冻结预训练主干网络，仅训练分类头，用于评估预训练质量：

```bash
python train.py --config configs/config.yaml \
    --model TSLANet \
    --training_mode train_linear \
    --num_epochs 50
```

---

### 4.3 传统机器学习模型工作流程

适用于RandomForest、SVM、XGBoost、LightGBM、KNN、LogisticRegression等传统ML模型。

**步骤1: 数据预处理**

传统ML模型使用2D特征向量，通过 `preprocess_2D.py` 进行预处理：

```bash
python preprocess_2D.py
```

**步骤2: 模型训练**

传统ML模型仅支持 `supervised` 训练模式，自动在CPU上运行，输入数据会自动从3D展平为2D格式。

```bash
# 随机森林
python train.py --config configs/config.yaml \
    --model RandomForest \
    --training_mode supervised

# SVM
python train.py --config configs/config.yaml \
    --model SVM \
    --training_mode supervised

# XGBoost
python train.py --config configs/config.yaml \
    --model XGBoost \
    --training_mode supervised
```

> **注意：** 模型超参数通过 `configs/config.yaml` 中对应模型名称的配置节设置，例如：
> ```yaml
> RandomForest:
>   n_estimators: 100
>   max_depth: null
>   random_state: 42
> ```

**步骤3: 模型预测**

```bash
python predict.py \
    --checkpoint experiments_logs/Exp1/run1/supervised_seed_123/saved_models/ckp_best.pt \
    --config configs/config.yaml \
    --model RandomForest \
    --dataset chaosu \
    --mode test
```

如需对新数据进行批量预测，使用 `prepare_ml_data.py` 准备数据：

```bash
# 准备ML模型的预测数据（从Excel或目录批量加载）
python prepare_ml_data.py

# 执行预测
python predict.py \
    --checkpoint experiments_logs/Exp1/run1/supervised_seed_123/saved_models/ckp_best.pt \
    --config configs/config.yaml \
    --model RandomForest \
    --data_file ./output/prediction_features.pt
```


---

## 5. 配置文件参数说明

本系统通过YAML配置文件管理所有运行参数。主要配置文件包括：

- `configs/config.yaml` — 模型训练与预测的核心配置
- `configs/preprocess.yaml` — 3D数据预处理配置
- `configs/preprocess_2D.yaml` — 2D数据预处理及传统ML模型配置

参数优先级（高→低）：命令行参数 > YAML配置文件 > 代码内置默认值。

---

### 5.1 config.yaml — 模型训练与预测配置

`configs/config.yaml` 是模型训练和预测的核心配置文件，包含实验管理、训练参数、模型超参数、数据集配置等。

#### 5.1.1 实验配置（experiment）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `description` | str | `'Exp4'` | 实验描述，用于生成实验日志目录名 |
| `run_description` | str | `'LogisticRegression'` | 运行描述，用于生成运行子目录名 |
| `seed` | int | `123` | 随机种子，确保实验可复现 |
| `model` | str | `'TSLANet'` | 模型选择，可选值：`TSLANet`、`TS-TCC`、`base_CNN`、`RC`、`RandomForest`、`SVM`、`XGBoost`、`LightGBM`、`KNN`、`LogisticRegression` |

配置示例：

```yaml
experiment:
  description: 'Exp1'
  run_description: 'TSLANet_supervised'
  seed: 42
  model: 'TSLANet'
```

#### 5.1.2 训练配置（training）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `mode` | str | `'self_supervised'` | 训练模式，可选值：`random_init`、`supervised`、`self_supervised`、`fine_tune`、`train_linear` |
| `num_epochs` | int | `10` | 训练轮数 |
| `batch_size` | int | `16` | 批次大小 |
| `learning_rate` | float | `0.001` | 学习率 |
| `weight_decay` | float | `0.0003` | 权重衰减（L2正则化系数） |
| `optimizer` | str | `'adam'` | 优化器类型，可选值：`adam`、`adamw`、`sgd` |
| `beta1` | float | `0.9` | Adam优化器的β1参数 |
| `beta2` | float | `0.99` | Adam优化器的β2参数 |
| `use_scheduler` | bool | `true` | 是否使用学习率调度器 |
| `scheduler_type` | str | `'reduce_on_plateau'` | 学习率调度器类型 |

训练模式说明：

| 模式 | 说明 |
|------|------|
| `random_init` | 随机初始化模型权重，冻结指定层后训练，用于基线对比 |
| `supervised` | 标准监督学习，使用带标签数据端到端训练 |
| `self_supervised` | 自监督预训练，无需标签学习数据表征 |
| `fine_tune` | 加载预训练权重后全参数微调 |
| `train_linear` | 冻结主干网络，仅训练分类头（线性探测） |

#### 5.1.3 预训练配置（pretraining）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | bool | `true` | 是否启用预训练 |
| `pretrain_epochs` | int | `5` | 预训练轮数 |
| `pretrain_lr` | float | `0.001` | 预训练学习率 |
| `masking_ratio` | float | `0.4` | TSLANet掩码比例（自监督学习中随机遮蔽的数据比例） |
| `load_pretrained` | bool | `false` | 是否加载已有的预训练模型 |
| `pretrained_path` | str/null | `null` | 预训练模型路径，`null` 表示自动从实验目录查找 |
| `delete_layers` | list | `['logits']` | 加载预训练模型时需要删除的层名称列表 |
| `freeze_backbone` | bool | `false` | 是否冻结主干网络参数 |

#### 5.1.4 数据集配置（dataset）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `selected` | str | `'chaosu'` | 当前使用的数据集名称，需与 `dataset_configs` 中的键匹配 |
| `data_path` | str | `'./data'` | 数据根目录，数据集文件位于 `{data_path}/{selected}/` 下 |
| `normalize` | bool | `false` | 是否对数据进行归一化 |
| `drop_last` | bool | `true` | 是否丢弃最后一个不完整的批次 |

#### 5.1.5 路径配置（paths）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `home_path` | str/null | `null` | 项目根目录，`null` 表示使用当前工作目录 |
| `logs_save_dir` | str | `'experiments_logs'` | 实验日志保存目录 |
| `config_files_dir` | str | `'config_files'` | 配置文件副本保存目录 |

#### 5.1.6 设备配置（device）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `device` | str | `'cuda'` | 运行设备，可选值：`cuda`（GPU）、`cpu`。传统ML模型自动使用CPU |

#### 5.1.7 日志配置（logging）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `log_timestamp_format` | str | `'%Y%m%d_%H%M%S'` | 日志文件时间戳格式 |
| `debug_separator` | str | `'='` | 调试日志分隔符字符 |
| `debug_separator_length` | int | `60` | 调试日志分隔符长度 |
| `checkpoint_save_top_k` | int | `1` | 保存最优检查点的数量 |
| `monitor` | str | `'val_loss'` | 监控指标名称 |
| `mode` | str | `'min'` | 监控模式，`min` 表示指标越小越好 |

#### 5.1.8 随机种子配置（random_seed）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `torch_deterministic` | bool | `false` | 是否启用PyTorch确定性模式（可能降低性能） |
| `torch_benchmark` | bool | `false` | 是否启用cuDNN benchmark（启用可加速但结果不确定） |

#### 5.1.9 模型特定配置

每个模型在配置文件中有独立的配置节，以模型名称为键。

**TSLANet 配置：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `emb_dim` | int | `128` | 嵌入维度 |
| `depth` | int | `2` | Transformer层数 |
| `patch_size` | int | `8` | 时间序列分块大小 |
| `dropout_rate` | float | `0.15` | Dropout比率 |
| `use_icb` | bool | `true` | 是否使用ICB（交互式卷积块） |
| `use_asb` | bool | `true` | 是否使用ASB（自适应频谱块） |
| `adaptive_filter` | bool | `true` | 是否启用自适应滤波 |

**TS-TCC 配置：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `kernel_size` | int | `32` | 卷积核大小 |
| `stride` | int | `30` | 卷积步长 |
| `final_out_channels` | int | `128` | 最终输出通道数 |
| `dropout` | float | `0.35` | Dropout比率 |

TS-TCC 子模块配置：

| 子模块 | 参数 | 类型 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `context_contrast` | `temperature` | float | `0.2` | 对比学习温度参数 |
| `context_contrast` | `use_cosine_similarity` | bool | `true` | 是否使用余弦相似度 |
| `temporal_contrast` | `hidden_dim` | int | `64` | 时间对比隐藏层维度 |
| `temporal_contrast` | `timesteps` | int | `15` | 时间对比预测步数 |
| `augmentation` | `jitter_scale_ratio` | float | `1.0` | 抖动缩放比例 |
| `augmentation` | `jitter_ratio` | float | `0.5` | 抖动比例 |
| `augmentation` | `max_seg` | int | `10` | 最大分段数 |

**base_CNN 配置：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `kernel_size` | int | `25` | 卷积核大小 |
| `stride` | int | `6` | 卷积步长 |
| `final_out_channels` | int | `128` | 最终输出通道数 |
| `dropout` | float | `0.35` | Dropout比率 |

**RC（储层计算）配置：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `n_internal_units` | int | `100` | 内部单元数 |
| `spectral_radius` | float | `0.99` | 谱半径 |
| `connectivity` | float | `0.3` | 连接率 |
| `input_scaling` | float | `0.2` | 输入缩放系数 |
| `noise_level` | float | `0.0` | 噪声水平 |
| `bidir` | bool | `false` | 是否使用双向储层 |
| `circle` | bool | `false` | 是否使用环形拓扑 |
| `mts_rep` | str | `'mean'` | 多变量时间序列表示方法 |
| `w_ridge` | float | `1.0` | 岭回归正则化系数 |

**RandomForest 配置：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `n_estimators` | int | `100` | 决策树数量 |
| `max_depth` | int/null | `null` | 最大树深度，`null` 表示不限制 |
| `min_samples_split` | int | `2` | 内部节点分裂所需最小样本数 |
| `min_samples_leaf` | int | `1` | 叶节点最小样本数 |
| `random_state` | int | `42` | 随机种子 |

**SVM 配置：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `C` | float | `1.0` | 正则化参数 |
| `kernel` | str | `'rbf'` | 核函数类型，可选值：`linear`、`poly`、`rbf`、`sigmoid` |
| `gamma` | str | `'scale'` | 核函数系数 |
| `random_state` | int | `42` | 随机种子 |

**XGBoost 配置：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `n_estimators` | int | `100` | 提升轮数 |
| `max_depth` | int | `6` | 最大树深度 |
| `learning_rate` | float | `0.1` | 学习率 |
| `subsample` | float | `0.8` | 每棵树的样本采样比例 |
| `colsample_bytree` | float | `0.8` | 每棵树的特征采样比例 |
| `random_state` | int | `42` | 随机种子 |

**LightGBM 配置：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `n_estimators` | int | `100` | 提升轮数 |
| `max_depth` | int | `-1` | 最大树深度，`-1` 表示不限制 |
| `learning_rate` | float | `0.1` | 学习率 |
| `num_leaves` | int | `31` | 最大叶子节点数 |
| `subsample` | float | `0.8` | 样本采样比例 |
| `colsample_bytree` | float | `0.8` | 特征采样比例 |
| `random_state` | int | `42` | 随机种子 |

**KNN 配置：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `n_neighbors` | int | `5` | 近邻数量 |
| `weights` | str | `'uniform'` | 权重方式，可选值：`uniform`（等权重）、`distance`（距离加权） |
| `algorithm` | str | `'auto'` | 搜索算法 |
| `metric` | str | `'minkowski'` | 距离度量 |

**LogisticRegression 配置：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `C` | float | `1.0` | 正则化强度的倒数（值越小正则化越强） |
| `penalty` | str | `'l2'` | 正则化类型，可选值：`l1`、`l2`、`elasticnet`、`none` |
| `solver` | str | `'lbfgs'` | 优化算法，可选值：`lbfgs`、`liblinear`、`sag`、`saga` |
| `max_iter` | int | `1000` | 最大迭代次数 |
| `random_state` | int | `42` | 随机种子 |

#### 5.1.10 数据集特定配置（dataset_configs）

每个数据集在 `dataset_configs` 下有独立的配置节，定义该数据集的维度信息。系统根据 `dataset.selected` 的值自动选择对应的数据集配置。

| 参数 | 类型 | 说明 |
|------|------|------|
| `input_channels` | int | 输入通道数（特征维度） |
| `features_len` | int | 特征长度（用于模型内部计算） |
| `seq_len` | int | 序列长度（时间步数） |
| `num_classes` | int | 分类类别数 |
| `num_channels` | int | 数据通道数 |

当前支持的数据集配置：

| 数据集 | input_channels | seq_len | num_classes | 说明 |
|--------|---------------|---------|-------------|------|
| `chaosu` | 4 | 12000 | 4 | 叶轮超速故障数据集 |
| `Epilepsy` | 1 | 178 | 2 | 癫痫检测数据集 |
| `HAR` | 9 | 128 | 6 | 人体活动识别数据集 |
| `sleepEDF` | 1 | 3000 | 5 | 睡眠阶段分类数据集 |
| `pFD` | 1 | 144 | 3 | 预测性故障检测数据集 |
| `hhar` | 3 | 128 | 6 | 异构人体活动识别数据集 |

配置示例（添加自定义数据集）：

```yaml
dataset:
  selected: 'my_dataset'

dataset_configs:
  my_dataset:
    input_channels: 6
    features_len: 64
    seq_len: 5000
    num_classes: 3
    num_channels: 6
```

---

### 5.2 preprocess.yaml — 3D数据预处理配置

`configs/preprocess.yaml` 用于控制3D数据预处理流程（`api/preprocess_3D.py`），定义数据路径、特征选择、处理方式、标准化方法和数据集划分策略。

#### 5.2.1 文件路径配置（paths）

| 参数 | 类型 | 说明 |
|------|------|------|
| `label_file` | str | 标签文件路径（CSV格式），包含样本文件路径和对应标签 |
| `config_file` | str | tracelog标签点配置表路径（CSV格式），定义传感器通道映射 |
| `output_dir` | str | 预处理输出目录，生成的 `.pt` 文件保存在此目录 |

#### 5.2.2 特征选择配置（features）

| 参数 | 类型 | 说明 |
|------|------|------|
| `target_tags` | list[str] | 目标特征标签列表，指定从原始数据中提取哪些传感器通道 |

默认提取的特征通道：

| 标签名 | 说明 |
|--------|------|
| `grRotorSpeedFormSpeedRelay1` | 转子转速（超速继电器1） |
| `grRotorSpeedFormSpeedRelay2` | 转子转速（超速继电器2） |
| `grRotorSpeedFromCounterModule1` | 转子转速（计数器1） |
| `rotor_speed` | 叶轮转速 |

#### 5.2.3 数据处理配置（processing）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_seq_len` | int | `12000` | 统一的时间步数，短序列填充、长序列截断至此长度 |
| `missing_value_method` | str | `'interpolate'` | 缺失值处理方法，可选值：`interpolate`（插值）、`ffill`（前向填充）、`bfill`（后向填充）、`mean`（均值填充）、`zero`（零值填充） |
| `padding_mode` | str | `'constant'` | 序列填充模式，可选值：`constant`（常数填充）、`edge`（边缘值填充）、`reflect`（反射填充） |

#### 5.2.4 标准化配置（normalization）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `method` | str | `'zscore'` | 标准化方法，可选值：`zscore`（Z-score标准化）、`minmax`（最小-最大归一化）、`robust`（鲁棒标准化）、`none`（不标准化） |
| `save_stats` | bool | `true` | 是否保存标准化统计量（均值、标准差等），预测时需要使用相同的统计量 |
| `stats_filename` | str | `'normalization_stats.npz'` | 标准化统计量保存文件名 |

#### 5.2.5 数据集划分配置（split）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `test_size` | float | `0.4` | 测试集+验证集占总数据的比例 |
| `val_test_split` | float | `0.5` | 验证集在（测试+验证）中的比例 |
| `random_state` | int | `42` | 随机种子，确保划分可复现 |
| `stratify` | bool | `true` | 是否按标签分层抽样，确保各类别比例一致 |

> **说明：** 默认配置下，数据划分比例为 训练集60% : 验证集20% : 测试集20%。如需8:1:1比例，设置 `test_size: 0.2`，`val_test_split: 0.5`。

#### 5.2.6 输出文件名配置（output）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `train_file` | str | `'train.pt'` | 训练集输出文件名 |
| `val_file` | str | `'val.pt'` | 验证集输出文件名 |
| `test_file` | str | `'test.pt'` | 测试集输出文件名 |

#### 5.2.7 日志配置（logging）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `verbose` | bool | `true` | 是否输出详细处理日志 |
| `save_log` | bool | `true` | 是否将日志保存到文件 |
| `log_file` | str | `'preprocessing.log'` | 日志文件名 |

完整配置示例：

```yaml
paths:
  label_file: "./data_preprocessing/label.csv"
  config_file: "./data_preprocessing/tracelog标签点配置表.csv"
  output_dir: "./data/chaosu"

features:
  target_tags:
    - grRotorSpeedFormSpeedRelay1
    - grRotorSpeedFormSpeedRelay2
    - grRotorSpeedFromCounterModule1
    - rotor_speed

processing:
  max_seq_len: 12000
  missing_value_method: "interpolate"
  padding_mode: "constant"

normalization:
  method: "zscore"
  save_stats: true
  stats_filename: "normalization_stats.npz"

split:
  test_size: 0.2
  val_test_split: 0.5
  random_state: 42
  stratify: true

output:
  train_file: "train.pt"
  val_file: "val.pt"
  test_file: "test.pt"

logging:
  verbose: true
  save_log: true
  log_file: "preprocessing.log"
```

---

### 5.3 preprocess_2D.yaml — 2D数据预处理及传统ML配置

`configs/preprocess_2D.yaml` 用于控制2D数据预处理流程（`preprocess_2D.py`、`prepare_ml_data.py`），同时包含传统机器学习模型的训练和推理配置。该配置文件通过 `configs/config_loader.py` 加载，支持 `__PROJECT_ROOT__` 占位符自动替换为项目根目录。

#### 5.3.1 项目根路径配置（project）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `root` | str | `'__PROJECT_ROOT__'` | 项目根目录占位符，运行时自动替换为实际路径 |

#### 5.3.2 数据源配置（data）

| 参数 | 类型 | 说明 |
|------|------|------|
| `source_excel` | str | 训练数据源Excel文件路径，包含tracelog路径和标签信息 |
| `source_test_excel` | str | 测试数据源路径（Excel文件或包含CSV文件的目录） |
| `processed_dir` | str | 预处理后数据的输出目录 |
| `reports_dir` | str | 报告输出目录 |

#### 5.3.3 批量预测输出配置（output）

| 参数 | 类型 | 说明 |
|------|------|------|
| `prediction_csv` | str | 批量预测结果CSV文件输出路径 |
| `prediction_report_img` | str | 批量预测报告图片输出路径 |

#### 5.3.4 模型配置（model）

| 参数 | 类型 | 说明 |
|------|------|------|
| `save_dir` | str | 模型保存目录（相对路径会自动拼接项目根目录） |
| `filename` | str | 模型文件名（`.pt` 格式） |

#### 5.3.5 向量数据库配置（db）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `host` | str | `'localhost'` | Qdrant数据库主机地址 |
| `port` | int | `6333` | Qdrant数据库端口 |
| `collection_name` | str | `'wind_turbine_fault_diagnosis'` | 向量集合名称 |
| `batch_upload_size` | int | `50` | 批量上传大小 |

#### 5.3.6 全局常量配置（constants）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `timestamp_window` | list[int] | `[-2000, 2000]` | 故障时间窗口（秒），用于截取故障前后的数据段 |
| `default_fault_time` | int | `0` | 无标签预测时使用的默认故障时间点 |

#### 5.3.7 传感器通道配置（channels）

| 参数 | 类型 | 说明 |
|------|------|------|
| `main_sensor_channels` | list[str] | 单通道特征提取的传感器列表（每通道产生80维特征） |
| `multi_analysis_channels` | list[str] | 多通道同步性分析的传感器列表（产生9维特征） |
| `key_rotor_channel` | str | 业务逻辑中的关键转子通道名称 |

默认传感器通道：

| 通道类型 | 通道名称 | 说明 |
|----------|----------|------|
| 主传感器 | `rotor_speed` | 叶轮转速 |
| 主传感器 | `generator_speed` | 发电机转速 |
| 主传感器 | `rotor_speed_relay1` | 超速继电器1转速 |
| 主传感器 | `rotor_speed_relay2` | 超速继电器2转速 |
| 主传感器 | `rotor_speed_counter1` | 计数器1转速 |
| 主传感器 | `breaker_feedback1` | 断路器反馈1 |
| 主传感器 | `breaker_feedback2` | 断路器反馈2 |
| 多通道分析 | `rotor_speed` | 叶轮转速 |
| 多通道分析 | `rotor_speed_relay1` | 超速继电器1转速 |
| 多通道分析 | `rotor_speed_counter1` | 计数器1转速 |

#### 5.3.8 Excel列名映射（excel）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `trace_path_column` | str | `'Tracelog路径'` | Excel中tracelog文件路径所在列名 |
| `label_column` | str | `'标签'` | Excel中标签所在列名 |
| `fault_reason_column` | str | `'故障原因'` | Excel中故障原因所在列名（可选） |
| `fault_time_column` | str | `'故障时间'` | Excel中故障时间所在列名（可选） |

#### 5.3.9 知识库配置（knowledge）

| 参数 | 类型 | 说明 |
|------|------|------|
| `fault_knowledge_file` | str | 故障知识库Excel文件路径，包含标签到故障原因的映射关系 |

#### 5.3.10 数据划分配置（data_split）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `test_size` | float | `0.1` | 测试集比例（10%） |
| `val_size` | float | `0.111` | 验证集比例（约1/9 ≈ 11.1%，从剩余数据中划分） |
| `random_seed` | int | `42` | 随机种子 |

> **说明：** 先从总数据中分出10%作为测试集，再从剩余90%中分出约11.1%（即总数据的10%）作为验证集，最终比例约为 8:1:1。

#### 5.3.11 数据处理参数（data_processing）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `min_rows_after_window` | int | `5` | 时间窗口截取后的最小行数，低于此值则跳过该样本 |
| `time_offset_threshold` | int | `10000` | 时间偏置校正阈值 |
| `column_offset_check_range` | int | `5` | 列偏移检测范围 |
| `time_value_threshold` | int | `-100` | 判断是否为时间列的最小值阈值 |

#### 5.3.12 特征提取配置（feature_extraction）

特征提取包含多个子模块，每个子模块负责提取不同类型的特征：

**通用配置（general）：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `fault_timestamp` | int | `0` | 故障时间戳 |
| `multi_channels` | list | `[]` | 多通道列表 |

**统计特征配置（stat）：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `epsilon` | float | `1e-8` | 防除零常数 |
| `stability_segments` | int | `4` | 稳定性分析的分段数 |
| `mutation_std_ratio` | float | `1.5` | 突变检测的标准差倍数 |
| `min_len_mutation` | int | `10` | 突变检测的最小数据长度 |

**动态特征配置（dynamic）：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `window_pre_fault` | int | `2000` | 故障前窗口大小（数据点数） |
| `window_transient` | int | `500` | 瞬态过程窗口大小 |
| `window_post_fault` | int | `2000` | 故障后窗口大小 |
| `response_thresh_ratio` | float | `0.5` | 响应阈值比例 |
| `rise_rate_idx_min` | int | `10` | 上升率计算的最小索引 |

**频域特征配置（freq）：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `bands` | list[list] | `[[0, 0.1], [0.1, 0.3], [0.3, 0.5]]` | 频带划分区间 |
| `high_freq_filter.order` | int | `3` | 高通滤波器阶数 |
| `high_freq_filter.cutoff` | float | `0.1` | 高通滤波截止频率 |
| `high_freq_filter.type` | str | `'high'` | 滤波器类型 |

**周期性特征配置（period）：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `peak_height_std_ratio` | float | `0.5` | 峰值高度标准差倍数 |
| `business_dist` | int | `10` | 业务规则中的距离参数 |
| `min_len` | int | `50` | 最小数据长度 |

**业务规则特征配置（business）：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `jump_window` | int | `5` | 跳跃检测窗口大小 |
| `jump_ratio` | float | `2.0` | 跳跃比例阈值 |
| `sync_tolerance` | int | `10` | 多通道同步容差 |
| `fluctuation_w_ratio` | float | `0.1` | 波动窗口比例 |
| `pattern_thresh_high` | float | `0.7` | 模式识别高阈值 |
| `pattern_thresh_low` | float | `0.3` | 模式识别低阈值 |

#### 5.3.13 模型训练配置（training）

该配置节用于2D预处理流程中的传统ML模型训练：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `random_seed` | int | `42` | 随机种子 |
| `n_jobs` | int | `-1` | 并行任务数，`-1` 表示使用所有CPU核心 |

`training.models` 下包含各模型的超参数配置：

| 模型 | 关键参数 | 说明 |
|------|----------|------|
| `RandomForest` | `n_estimators: 300, max_depth: 12, class_weight: 'balanced'` | 随机森林（带类别平衡） |
| `ExtraTrees` | `n_estimators: 300, max_depth: 15, class_weight: 'balanced'` | 极端随机树 |
| `XGBoost` | `n_estimators: 200, max_depth: 6, learning_rate: 0.05` | XGBoost梯度提升 |
| `LightGBM` | `n_estimators: 200, learning_rate: 0.05, num_leaves: 31` | LightGBM轻量梯度提升 |

#### 5.3.14 模型推理配置（inference）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `device` | str | `'cpu'` | 推理设备 |
| `weights_only` | bool | `false` | 是否只加载模型权重（`true` 更安全但可能不兼容旧模型） |

#### 5.3.15 CSV列名映射（column_mapping）

`column_mapping` 定义了标准通道名称到原始CSV列名的同义词映射。系统在读取CSV文件时，会根据此映射自动匹配列名。每个通道可配置多个同义词：

| 标准通道名 | 同义词列表 | 说明 |
|------------|-----------|------|
| `timestamp` | `index`, `timestamp`, `iIndex`, `Time`, `time`, `时间` | 时间戳列 |
| `rotor_speed` | `rRotorSpeed`, `grRotorSpeed`, `叶轮转速` | 叶轮转速 |
| `generator_speed` | `rGeneratorSpeed`, `grGeneratorSpeed`, `发电机转速` | 发电机转速 |
| `rotor_speed_relay1` | `rRotorSpeedFormSpeedRelay1`, `grRotorSpeedFormSpeedRelay1` | 超速继电器1转速 |
| `rotor_speed_relay2` | `rRotorSpeedFormSpeedRelay2`, `grRotorSpeedFormSpeedRelay2` | 超速继电器2转速 |
| `rotor_speed_counter1` | `grRotorSpeedFromCounterModule1`, `rRotorSpeedFromCounterModule1` | 计数器1转速 |
| `breaker_feedback1` | `bNacelleBreakerCabinetBreakerOnFeedback1`, `gbNacelleBreakerCabinetBreakerOnFeedback1` | 断路器反馈1 |
| `breaker_feedback2` | `bNacelleBreakerCabinetBreakerOnFeedback2`, `gbNacelleBreakerCabinetBreakerOnFeedback2` | 断路器反馈2 |

> **提示：** 如果原始CSV文件的列名不在同义词列表中，可在 `column_mapping` 中添加新的同义词。例如，如果CSV中叶轮转速列名为 `RotorRPM`，添加到 `rotor_speed` 的同义词列表即可：
> ```yaml
> column_mapping:
>   rotor_speed:
>     - rRotorSpeed
>     - grRotorSpeed
>     - 叶轮转速
>     - RotorRPM    # 新增同义词
> ```

---

### 5.4 配置文件使用建议

1. **首次使用**：复制配置文件模板，修改路径参数（`paths`、`data`）指向实际数据位置
2. **切换模型**：修改 `experiment.model` 和 `training.mode`，模型超参数在对应配置节中调整
3. **调整数据集**：修改 `dataset.selected` 并确保 `dataset_configs` 中有对应配置
4. **命令行覆盖**：对于临时调整（如修改epoch数、batch_size），优先使用命令行参数而非修改配置文件
5. **实验管理**：每次实验修改 `experiment.description` 和 `run_description`，系统会自动创建独立的日志目录
6. **占位符**：`preprocess_2D.yaml` 中的 `__PROJECT_ROOT__` 会在运行时自动替换为项目根目录，无需手动修改


---

## 6. 常见错误处理指南

本节列出系统使用过程中常见的错误场景，包括错误现象、原因分析和解决方案，帮助用户快速定位和解决问题。

---

### 6.1 配置文件缺失或格式错误

**错误现象：**
- 终端输出 `❌ 错误: 配置文件不存在: <路径>`
- 终端输出 `❌ 错误: 配置文件格式错误: <YAML解析错误详情>`
- 程序以错误码 1 退出

**原因分析：**
- `--config` 参数指定的文件路径不存在或拼写错误
- YAML文件存在语法错误（如缩进不一致、冒号后缺少空格、使用了Tab而非空格）
- 配置文件编码不是UTF-8

**解决方案：**
1. 检查文件路径是否正确：`ls configs/config.yaml`
2. 使用YAML校验工具检查语法，例如在线工具或 `python -c "import yaml; yaml.safe_load(open('configs/config.yaml'))"`
3. 确保YAML文件使用空格缩进（不要使用Tab）
4. 参考项目自带的配置文件模板 `configs/config.yaml` 进行修正


---

### 6.2 数据文件损坏或格式不正确

**错误现象：**
- 日志中出现 `样本 X 处理失败: ERR_DAT_101` 或 `ERR_DAT_105` 等错误码
- 日志中出现 `样本 X 异常: <异常信息>`
- 批处理结束时报告大量失败样本
- 极端情况下抛出 `RuntimeError: 没有样本被成功处理`

**原因分析：**
- CSV文件编码错误或文件损坏（如传输中断导致不完整）
- CSV缺少必需的特征列（如转速列名与配置中的 `target_tags` 不匹配）
- 数据中存在大量空值或非数值内容
- 文件路径在标签文件/Excel清单中配置错误

**解决方案：**
1. 用文本编辑器或 `head -n 5 <csv文件>` 检查CSV文件是否可正常读取
2. 确认CSV列名与 `preprocess.yaml` 中 `column_mapping` 的同义词列表匹配
3. 检查标签文件中的文件路径是否指向实际存在的CSV文件
4. 如果个别样本损坏，系统会自动跳过并继续处理其他样本，无需干预


---

### 6.3 检查点加载失败

**错误现象：**
- 终端输出 `❌ 错误: 检查点文件不存在: <路径>`
- 终端输出 `❌ 错误: 模型权重加载失败: <详情>`
- 提示 `请确保检查点与当前模型架构匹配`
- 程序以错误码 1 退出（错误码 3101 CHECKPOINT_MISSING 或 2104 PRETRAIN_LOAD_ERROR）

**原因分析：**
- `--checkpoint` 参数指定的 `.pt` 文件路径不存在
- 检查点文件是用不同模型架构保存的（如用TSLANet的检查点加载TS-TCC模型）
- 检查点文件损坏（如训练中断导致保存不完整）
- 微调模式下，自监督预训练的检查点路径配置错误

**解决方案：**
1. 确认检查点文件存在：`ls <检查点路径>`
2. 确保 `--model` 参数与训练时使用的模型一致
3. 如果检查点损坏，使用 `ckp_last.pt` 替代 `ckp_best.pt`（或反之）
4. 微调时检查 `config.yaml` 中 `pretraining.pretrained_path` 是否指向正确的预训练目录


---

### 6.4 GPU内存不足

**错误现象：**
- 终端输出 `CUDA out of memory`
- 日志中出现 `GPU内存不足`
- 日志提示 `建议: 减小batch_size或使用--device cpu`
- 训练或预测过程中程序崩溃

**原因分析：**
- `batch_size` 设置过大，超出GPU显存容量
- 模型参数量过大（如序列长度 `max_seq_len` 设置过长）
- GPU被其他进程占用，可用显存不足
- 训练过程中梯度累积导致显存逐渐增长

**解决方案：**
1. 减小 `batch_size`：`python train.py --config configs/config.yaml --batch_size 8`
2. 切换到CPU运行：`python train.py --config configs/config.yaml --device cpu`
3. 检查GPU占用情况：`nvidia-smi`，关闭其他占用GPU的进程
4. 减小输入序列长度（在 `preprocess.yaml` 中调整 `max_seq_len`）
5. 如果系统支持自动降级，程序会尝试切换到CPU继续运行


---

### 6.5 标签映射缺失

**错误现象：**
- 日志中出现 `样本 X 的标签 'xxx' 未在映射表中定义，跳过`
- 处理结束时报告 `处理完成: 成功 N, 跳过 M`，跳过数量较多
- 预测时故障诊断信息缺失或显示为空

**原因分析：**
- 数据中存在新的故障类型标签，但标签映射表未更新
- 标签文件中的标签值格式与映射表不一致（如多余空格、大小写差异）
- `--mapping` 参数指定的 `class_mapping.xlsx` 文件缺失或格式不正确
- Excel映射文件中缺少 `标签`、`故障原因`、`处理方案` 必需列

**解决方案：**
1. 检查数据中的标签值，确认是否有新增的故障类型
2. 更新 `configs/class_mapping.xlsx`，添加缺失的标签映射
3. 确保标签值格式一致（去除前后空格，统一大小写）
4. 验证映射文件格式：确保包含 `标签`、`故障原因`、`处理方案` 三列


---

### 6.6 训练过程中出现NaN损失

**错误现象：**
- 日志中出现 `批次 X 损失为NaN/Inf，跳过`
- 日志中出现 `连续多个批次出现NaN，停止训练`
- 训练指标（metrics.csv）中损失值突然变为NaN
- 模型准确率不再提升或急剧下降

**原因分析：**
- 学习率设置过高，导致梯度爆炸
- 训练数据中存在异常值（极大值或极小值）
- 数据未经标准化处理，数值范围差异过大
- 模型权重初始化不当

**解决方案：**
1. 降低学习率：在 `config.yaml` 中将 `training.learning_rate` 减小（如从 `0.001` 降至 `0.0001`）
2. 启用数据标准化：在 `preprocess.yaml` 中设置 `norm_method: zscore` 或 `minmax`
3. 检查数据质量：确认预处理后的数据中无异常值或无穷值
4. 使用梯度裁剪：限制梯度范数防止梯度爆炸
5. 如果偶尔出现NaN，系统会自动跳过该批次继续训练；如果连续出现，需要调整参数后重新训练