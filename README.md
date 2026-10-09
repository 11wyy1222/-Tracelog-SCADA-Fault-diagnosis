# 故障诊断训练与预测平台

本项目用于风电设备 Tracelog/SCADA 数据预处理、故障分类模型训练、模型评估与预测。项目同时支持深度学习模型和传统机器学习模型。

> 项目根目录为 `D:\Platform\Platform`。`D:\Platform\.venv` 是当前开发虚拟环境；`predict_standalone` 是独立的预测部署代码，请按独立交付物维护。

## 主要功能

- Tracelog/SCADA 数据读取、清洗、特征提取和数据集划分
- TSLANet、TS-TCC、基础 CNN 和 RC 模型训练
- RandomForest、ExtraTrees、SVM、XGBoost、LightGBM、KNN、LogisticRegression 训练
- 使用 `.pt` 数据和已训练检查点执行批量预测
- 根据 Excel 类别映射输出故障诊断结果
- Tracelog 样本聚类和可视化分析

## 目录说明

```text
Platform/
├── api/                    # 对外预处理接口
├── configs/                # 训练、预测和预处理配置
├── data/                   # 训练/验证/测试数据
├── data_preprocessing/     # 数据读取、清洗及特征工程
├── docs/                   # 操作手册、接口和架构文档
├── experiments_logs/       # 历史实验结果与模型检查点
├── external_data/          # 外部预测数据示例
├── models/                 # 模型定义与模型工厂
├── predict_standalone/     # 独立预测部署代码
├── tests/                  # 自动化测试
├── tracelog_clustering/    # Tracelog 聚类工具
├── trainers/               # 统一训练器
├── utils/                  # 日志、指标、损失函数等工具
├── train.py                # 统一训练入口
├── predict.py              # 统一预测入口
├── environment.yml         # Conda 环境快照
└── requirements.txt        # pip 依赖
```

## 环境准备

推荐使用 Python 3.13。仓库外层已有虚拟环境时，在 PowerShell 中执行：

```powershell
cd D:\Platform\Platform
..\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

如需重新创建环境：

```powershell
cd D:\Platform
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
cd .\Platform
python -m pip install -r requirements.txt
```

也可以使用 Conda 环境快照：

```powershell
conda env create -f environment.yml
conda activate pythonProject
```

PyTorch 是否支持 CUDA 取决于安装的 wheel、显卡驱动和 CUDA 环境。若需要 GPU，请按 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/) 选择对应安装命令，再安装其余依赖。

## 配置与数据

训练和预测的主配置是 `configs/config.yaml`，重点检查以下字段：

- `checkpoint`：预测所用模型检查点
- `experiment`：实验名称、模型和随机种子
- `training`：训练模式、轮数、批次大小和学习率
- `dataset`：数据集名称和数据根目录
- `device`：`cpu` 或 `cuda`
- `dataset_configs`：各数据集的通道数、序列长度和类别数

训练数据通常位于 `data/<数据集名称>/`，并包含：

```text
train.pt
val.pt
test.pt
```

`.pt` 文件的基本结构为：

```python
{
    "samples": Tensor,       # 深度学习通常为 [N, C, L]，传统 ML 通常为 [N, F]
    "labels": Tensor | None  # 纯预测数据可不提供 labels
}
```

## 更换机器学习模型：训练与部署

更换模型分为两种情况：

- **同一份已处理数据只更换算法**：不需要重新执行 2D 预处理，修改训练配置并重新训练即可。
- **更换故障、通道、源数据或标签体系**：必须重新检查预处理配置、生成数据集、修改训练配置，然后同步更新部署配置。

下面是传统机器学习模型的完整操作顺序。不要直接复用旧数据集或只替换模型文件，否则容易产生特征维度、通道顺序或标签含义不一致。

### 1. 修改 2D 数据预处理配置

编辑 `configs/preprocess_2D.yaml`，至少检查以下项目：

1. **源数据和输出目录**

   ```yaml
   data:
     source_excel: 'D:\data\新故障样本索引.xlsx'
     processed_dir: './data/新数据集名称'
   ```

   `source_excel` 是训练样本索引表，不是单个 Tracelog 文件。每一行通常描述一个样本，并给出 Tracelog 路径和标签。

2. **传感器通道及其顺序**

   ```yaml
   channels:
     main_sensor_channels:
       - channel_a
       - channel_b
     multi_analysis_channels: []
     key_rotor_channel:
   ```

   通道名称、数量和排列顺序共同决定特征含义。换模型后，训练和部署必须使用完全相同的通道列表及顺序。当前单通道特征提取通常为每个通道 80 维；启用多通道特征后还会增加额外维度，应以生成的 `.pt` 文件实际形状为准。

3. **索引 Excel 的列名映射**

   ```yaml
   excel:
     trace_path_column: 'tr_path'
     label_column: 'label'
     fault_reason_column: '故障原因'
     fault_time_column: '故障时间'
   ```

   这些字段必须与 `source_excel` 的表头完全一致。其中路径列和标签列是训练数据生成的关键字段。

4. **训练标签知识库**

   ```yaml
   knowledge:
     fault_knowledge_file: '__PROJECT_ROOT__/configs/新故障知识库.xlsx'
     label_column: '标签'
     id_column: '序号'
   ```

   知识库负责把 `source_excel` 中的文本标签转换成数字类别。知识库中的标签文本必须能与源数据标签匹配；建议类别 ID 使用从 `0` 开始的连续整数。

5. **原始 Tracelog 列名映射（taginfo）**

   ```yaml
   column_mapping_file: '__PROJECT_ROOT__/configs/tracelog_tag_info.xlsx'
   ```

   `tracelog_tag_info.xlsx` 用于把不同机型或数据版本中的标签点别名转换成标准通道名。它不是上面的索引 Excel 列名配置。新增通道时，要确认其标准名存在于 taginfo 中，并补齐实际源文件可能出现的别名。

配置完成后执行：

```powershell
python preprocess_2D.py
```

正常情况下会在 `data.processed_dir` 下生成：

```text
full_dataset.pt
train.pt
val.pt
test.pt
```

在开始训练前，至少确认样本数、特征宽度、标签集合和预期一致：

```powershell
python -c "import torch; d=torch.load(r'data/新数据集名称/train.pt', weights_only=False); print(d['samples'].shape); print(sorted(d['labels'].unique().tolist()))"
```

### 2. 修改训练配置并训练

编辑 `configs/config.yaml`：

1. 将 `dataset.selected` 和 `dataset.data_path` 指向新数据集。
2. 在 `dataset_configs` 中新增或修改同名配置。
3. 根据实际数据填写通道数、特征维度和类别数。
4. 选择模型，并按需修改对应模型的超参数。

传统机器学习数据集配置示例：

```yaml
dataset:
  selected: '新数据集名称'
  data_path: './data'

dataset_configs:
  新数据集名称:
    input_channels: 6       # main_sensor_channels 的数量
    features_len: 80        # 每通道特征数
    seq_len: 480            # 实际总特征宽度，以 .pt shape 为准
    num_classes: 3          # 实际标签类别数
    num_channels: 6
```

`dataset_configs` 的键必须与命令行的 `--dataset` 完全一致。`seq_len` 不要只按配置推算，应与生成数据中 `samples.shape[-1]` 核对；`num_classes` 应与训练标签集合和知识库类别数一致。

训练示例：

```powershell
python train.py --config configs/config.yaml --model RandomForest --dataset 新数据集名称
```

也可以将 `RandomForest` 替换为 `ExtraTrees`、`SVM`、`XGBoost`、`LightGBM`、`KNN` 或 `LogisticRegression`。命令行的 `--model` 和 `--dataset` 是本次训练的明确输入，建议同时更新 `config.yaml` 中的 `experiment.model` 与 `dataset.selected`，避免实验记录产生歧义。

也可以使用 `train_all_ml.py` 批量训练和比较多个传统机器学习模型。项目中实际文件名是 `train_all_ml.py`，不是 `train_all.py`。

使用配置中的数据集批量训练全部支持的传统机器学习模型：

```powershell
python train_all_ml.py --config configs/config.yaml
```

明确指定数据集和交叉验证折数：

```powershell
python train_all_ml.py --config configs/config.yaml --dataset 新数据集名称 --cv 5
```

只比较指定模型：

```powershell
python train_all_ml.py --config configs/config.yaml --dataset 新数据集名称 --models RandomForest ExtraTrees XGBoost LightGBM --cv 5
```

`train_all_ml.py` 支持的模型包括 `RandomForest`、`ExtraTrees`、`XGBoost`、`LightGBM`、`SVM`、`KNN` 和 `LogisticRegression`。参数优先级为“命令行参数 > `config.yaml` > 脚本默认值”。批量训练前仍须完成前述 2D 数据预处理，并确保 `dataset_configs` 与生成的数据维度一致。

训练完成后，记录最终检查点、使用的 `preprocess_2D.yaml`、`config.yaml`、知识库和 taginfo 版本。这些文件共同定义了模型，不能只保存 `.pt` 检查点。

### 3. 同步到 `predict_standalone`

将训练完成的检查点复制到 `predict_standalone/saved_models/`，然后在 `predict_standalone/configs/model_registry.yaml` 中新增或修改对应故障条目：

```yaml
faults:
  新故障名称:
    model_type: 'RandomForest'
    checkpoint: 'saved_models/新故障（RandomForest）.pt'
    mapping: 'configs/class_mapping/all_faults_mapping.xlsx'
    main_sensor_channels:
      - channel_a
      - channel_b
    multi_analysis_channels: []
    timestamp_window: [-2000, 500]
    default_fault_time: 0
    num_classes: 3
```

部署时逐项确认：

- `model_type` 必须与训练算法一致。
- `checkpoint` 必须指向本次训练得到的检查点。
- `main_sensor_channels`、`multi_analysis_channels` 及其顺序必须与训练预处理完全一致。
- `timestamp_window`、默认故障时间和特征提取规则必须与训练时一致。
- `num_classes` 必须与训练标签数一致。
- Tracelog 预测要同步检查 `predict_standalone/configs/tracelog_tag_info.xlsx`；必要时直接复制训练时确认过的 taginfo。
- SCADA 预测要同步检查 `predict_standalone/configs/scada_tag_info.xlsx` 以及 `preprocess_scada.yaml` 中的映射、通道和时间窗口。
- `predict_standalone/configs/preprocess_2D.yaml` 中的 `column_mapping_file` 必须指向正确的 Tracelog taginfo。

### 4. 核对训练与预测的标签序号

这是换模型时最容易被忽略、同时影响最大的检查项。模型输出的是数字类别，预测端再用该数字查找故障原因和处理方案，因此数字相同但含义不同会产生“预测成功但诊断错误”的隐蔽问题。

必须保证以下映射完全一致：

```text
训练 source_excel 中的标签
        ↓
训练知识库 fault_knowledge_file：标签文本 → 序号
        ↓
模型输出的数字标签
        ↓
部署 mapping 文件：数字标签 → 故障原因/处理方案
```

部署映射文件通常为 `predict_standalone/configs/class_mapping/all_faults_mapping.xlsx`。对于统一映射表：

- `fault_name` 必须与 `model_registry.yaml` 中的故障名称完全一致。
- 数字键列可以是 `id`、`序号` 或 `标签`，但其中的值必须与训练知识库的 `id_column` 一一对应。
- 不要用 Excel 行号或行顺序代替类别 ID。
- 新增、删除或重排类别后必须重新核对全部 ID；不能只修改故障原因文本。
- 建议使用 `0..num_classes-1` 的连续序号，并确认训练集实际包含预期类别。

部署预测示例：

```powershell
cd predict_standalone
python diagnose.py --fault 新故障名称 --data_path 'D:\data\待预测样本目录'
```

使用 Excel 索引表预测：

```powershell
python diagnose.py --fault 新故障名称 --data_path 'D:\data\待预测样本.xlsx' --col_path 'tr_path'
```

上线前应选取每个已知类别的样本做一次冒烟测试，核对输出 JSON 中的 `预测标签`、`故障原因` 和 `处理方案`，确认它们与训练知识库中的类别定义一致。

## 常用命令

所有命令均应在项目根目录 `D:\Platform\Platform` 下执行。

查看命令参数：

```powershell
python train.py --help
python train_all_ml.py --help
python predict.py --help
python prepare_data.py --help
python prepare_dp_data.py --help
```

训练单个模型：

```powershell
python train.py --config configs/config.yaml --model TSLANet --dataset chaosu
```

传统机器学习示例：

```powershell
python train.py --config configs/config.yaml --model RandomForest --dataset chaosu
```

预测外部 `.pt` 数据：

```powershell
python predict.py --config configs/config.yaml --model TSLANet --data_path external_data/sample_data_without_labels.pt
```

禁用故障诊断映射：

```powershell
python predict.py --config configs/config.yaml --model TSLANet --data_path external_data/sample_data_without_labels.pt --no_diagnosis
```

运行测试：

```powershell
python -m pytest tests -q
```

## 输出位置

- 训练实验、指标和检查点：通常写入 `experiments_logs/`
- SCADA 管道日志：通常写入 `logs/`
- 预测结果：由 `predict.py` 根据配置和运行时间生成

这些目录可能包含大量模型和中间文件，归档或清理前请确认仍在使用的检查点。

## 详细文档

- [操作手册](docs/operation_manual.md)
- [脚本接口说明](docs/脚本接口说明文档.md)
- [系统框架说明](docs/system_framework.md)

部分历史文档中的命令可能对应旧版接口；实际参数请以各脚本的 `--help` 输出为准。

## 常见问题

- **找不到项目模块**：确认当前目录是 `D:\Platform\Platform`，不要在外层目录直接运行入口脚本。
- **检查点不存在**：检查 `configs/config.yaml` 中的 `checkpoint` 路径。
- **数据维度不匹配**：核对 `dataset_configs` 中的 `input_channels`、`seq_len` 和 `num_classes`。
- **CUDA 不可用**：将配置中的 `device` 改为 `cpu`，或重新安装与本机 CUDA 匹配的 PyTorch。
- **RAR 文件无法读取**：Windows 下需要可用的 UnRAR DLL；代码也支持通过 `UNRAR_LIB_PATH` 指定 DLL 路径。
