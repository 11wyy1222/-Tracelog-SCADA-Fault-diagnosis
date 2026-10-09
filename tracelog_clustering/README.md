# Tracelog 故障特征与聚类

本目录提供完整的 tracelog 聚类流程，支持：

- 原始时序聚类，并用故障树特征解释聚类结果。
- 使用故障树特征直接聚类。
- 从受控自然语言 YAML 生成故障特征规则确认表。
- 专家确认规则后，使用确认通过的故障特征生成聚类结果。
- 聚类后按“簇 + 规则候选”生成代表样本和专家批量确认表。
- 输出聚类质量、特征分布和代表样本可视化报告。

自然语言规则流程不会修改原有预处理、训练和基础聚类脚本。`cluster_with_confirmed_features.py`
在运行时向现有聚类流程注入已确认规则，便于故障树变化时只更新 YAML 和确认表。

## 目录结构

```text
tracelog_clustering/
├─ cluster_tracelog_samples.py
│  └─ 基础入口：tracelog 预处理、内置 A-H 特征、聚类及专家复核输出
├─ cluster_with_confirmed_features.py
│  └─ 推荐入口：把人工确认的配置特征注入基础聚类流程
├─ fault_feature_rules.py
│  └─ 受控自然语言编译、确认门禁、安全表达式计算
├─ prepare_fault_feature_confirmation.py
│  └─ 根据 YAML 生成 Excel/CSV 故障特征确认表
├─ fault_feature_descriptions.example.yaml
│  └─ 根据 configs/14003.xlsx 更新的 A-H 标签、故障原因和判断方式示例
├─ fault_feature_confirmation.example.xlsx
│  └─ 确认表格式示例；YAML 更新后应重新生成正式确认表
├─ NATURAL_LANGUAGE_FAULT_FEATURES.md
│  └─ 自然语言规则机制的详细说明
├─ visualize.py
│  └─ 对指定运行目录生成聚类分析图和人工解释报告
├─ test_fault_feature_rules.py
│  └─ 规则编译、确认门禁和表达式安全测试
└─ output/
   └─ 每次运行生成一个时间戳子目录
```

`__pycache__/` 是 Python 缓存，不属于业务配置或运行结果。

## 数据处理原则

`cluster_tracelog_samples.py` 会：

1. 读取样本索引 Excel，每行根据指定的 tracelog 路径列加载一个样本。
2. 从 `configs/preprocess_2D.yaml` 读取：
   - `constants.timestamp_window`
   - `channels.main_sensor_channels`
   - `data_processing` 中的窗口、时间偏置和缺失通道策略
   - `column_mapping_file`
3. 自动识别 tracelog 编码和分隔符。
4. 识别故障点相对时间轴并截取窗口。
5. 对缺失值进行插值或补零，并把样本重采样到统一长度。
6. 根据所选模式使用时序矩阵或故障特征矩阵进行聚类。

## 环境

以下命令均假设当前目录为工作区根目录：

```text
D:\Platform
```

使用项目虚拟环境：

```powershell
.\.venv\Scripts\python.exe --version
```

## 推荐流程：自然语言规则确认后聚类

### 1. 编辑故障特征描述文件

编辑：

```text
Platform\tracelog_clustering\fault_feature_descriptions.example.yaml
```

当前示例的数据来源是 `Platform/configs/14003.xlsx`，并记录了 Excel 中的：

- `标签` → `feature_id`
- `故障原因` → `fault_description`
- `判断方式` → `judgment_method`
- `序号` → `source_sequence`
- `所需标签点` → `source_required_tags`

主要配置字段：

- `version`：规则版本。
- `source_file`、`source_mapping`：规则来源和字段对应。
- `glossary`：中文术语到已有原子特征列的映射。
- `feature_id`：稳定标签，例如 A-H。
- `fault_description`：故障原因描述。
- `judgment_method`：原知识库中的完整判断方式。
- `natural_rule`：可编译的受控自然语言计算规则。
- `weight`：进入聚类前的权重，默认 `1.0`。
- `enabled`：是否启用该特征。

受控自然语言支持中文术语、数字、括号、加减乘除、比较、`且/或`，以及：

- `较大值(...)`
- `较小值(...)`
- `绝对值(...)`
- `clip(...)`

`fault_description` 和 `judgment_method` 可以写完整自然语言；`natural_rule` 必须使用 `glossary`
定义过的术语和受支持运算，以保证转换结果确定、可复现、可审计。

### 2. 生成故障特征确认表

```powershell
.\.venv\Scripts\python.exe Platform\tracelog_clustering\prepare_fault_feature_confirmation.py `
  --description-config Platform\tracelog_clustering\fault_feature_descriptions.example.yaml `
  --output Platform\tracelog_clustering\fault_feature_confirmation.xlsx
```

确认表支持 `.xlsx` 和 `.csv`，主要包含：

- 标签、故障描述和完整判断方式。
- 原始受控自然语言规则。
- 编译后的表达式和引用的原子特征。
- 编译状态、错误信息、规则版本和来源行号。
- 专家决定、专家修正公式和备注。

### 3. 人工确认规则

专家填写：

- `expert_decision=通过`：规则进入聚类。
- `expert_decision=拒绝`：规则不进入聚类。
- `expert_expression`：可选；用于填写人工修正后的最终表达式。
- `expert_comment`：确认依据或备注。

确认门禁规则：

- 任一启用规则仍为 `pending` 时拒绝运行。
- 任一启用规则编译失败时拒绝运行。
- 至少需要一个通过的故障特征。
- 专家公式使用 AST 白名单校验，禁止导入模块、属性访问、文件访问和系统命令。

### 4. 根据确认特征生成聚类结果

使用 `cluster_with_confirmed_features.py`，并设置 `--cluster-mode 2`：

```powershell
.\.venv\Scripts\python.exe Platform\tracelog_clustering\cluster_with_confirmed_features.py `
  --fault-feature-confirmation-path Platform\tracelog_clustering\fault_feature_confirmation.xlsx `
  --config configs/preprocess_2D.yaml `
  --excel-path "D:\data\14003_sc_rotorandgeneratorspeeddiffmax_1.xlsx" `
  --trace-path-column tr_path `
  --output-dir tracelog_clustering/output `
  --cluster-mode 2 `
  --n-clusters 6
```

若不传 `--n-clusters`，程序会在 `--cluster-range` 范围内按照 silhouette 自动选择簇数。

每次运行会生成独立目录，例如：

```text
Platform/tracelog_clustering/output/20260817_120000/
```

`run_metadata.json` 会额外记录确认表绝对路径、最终公式、输入原子特征和权重，便于追溯。

## 基础聚类入口

如果仍需使用内置 A-H 故障特征，可以直接运行：

```powershell
.\.venv\Scripts\python.exe Platform\tracelog_clustering\cluster_tracelog_samples.py `
  --config configs/preprocess_2D.yaml `
  --excel-path "D:\data\14003_sc_rotorandgeneratorspeeddiffmax_1.xlsx" `
  --trace-path-column tr_path `
  --output-dir tracelog_clustering/output `
  --cluster-mode 1 `
  --n-clusters 6
```

### 聚类模式

- `--cluster-mode 1` 或 `sequence_explain`：使用原始时序聚类，以故障树特征解释每个簇。
- `--cluster-mode 2` 或 `fault_tree`：使用核心故障树特征直接聚类。

对于经过确认的新规则，推荐使用兼容入口和模式 2。

### 常用参数

- `--n-clusters 4`：直接指定簇数。
- `--cluster-range 2-10`：自动选簇时的搜索范围。
- `--target-length 2000`：重采样后的统一序列长度。
- `--sample-limit 50`：调试时只处理前 50 条。
- `--cluster-method kmeans`：KMeans 聚类，默认值。
- `--cluster-method agglomerative`：层次聚类。
- `--random-state 42`：随机种子。
- `--representative-top-k 5`：每个簇或规则子组导出的代表样本数。
- `--batch-label-threshold 0.6`：可建议批量确认的簇内规则占比阈值。
- `--expert-confirmation-path <csv>`：应用聚类后的专家确认表。

## 聚类输出

### 样本和聚类结果

- `cluster_assignments.csv`：每个样本的簇标签、中心距离和规则候选。
- `cluster_summary.csv`：每个簇的样本数和占比。
- `cluster_metrics.csv`：自动选簇时的 silhouette 和 inertia；固定簇数时可能不存在。
- `clustering_features.npy`：本次实际进入聚类的特征矩阵。
- `preprocessed_sequences.npz`：预处理后的统一长度时序数据和通道名。

### 故障特征和规则解释

- `fault_tree_features.csv`：每个样本的故障特征、规则分数和规则候选标签。
- `cluster_fault_tree_feature_summary.csv`：各簇故障特征中位数。
- `cluster_rule_candidate_crosstab.csv`：簇与规则候选标签的交叉占比。
- `cluster_rule_group_summary.csv`：`cluster_label + rule_candidate_label` 子组汇总。

使用确认规则时，特征列名为 `configured_feature_<feature_id>`。

### 代表样本和专家复核

- `cluster_representative_samples.csv`：每个簇距离中心最近的代表样本。
- `cluster_rule_group_representative_samples.csv`：每个规则子组的代表样本。
- `cluster_expert_review_plan.csv`：可批量确认或需要进一步复核的建议。
- `cluster_expert_confirmation_template.csv`：聚类后的专家确认模板。

### 异常和审计

- `failed_samples.csv`：加载或预处理失败的样本。
- `missing_tracelog_rows.xlsx`：源 Excel 中 tracelog 缺失或失败的对应行；仅在存在失败时生成。
- `run_metadata.json`：本次运行参数、通道、簇数、特征和规则审计信息。

## 聚类后的专家批量标注

故障特征确认表与聚类专家确认表是两个不同阶段：

1. `fault_feature_confirmation.xlsx`：聚类前确认“描述如何转换为特征”。
2. `cluster_expert_confirmation_template.csv`：聚类后确认“某个样本子组如何赋标签”。

聚类后的确认表填写规则：

- `expert_decision=batch_accept`：批量接受该 `rule_group_key`，`expert_label` 必填。
- `expert_decision=review_more`：不批量赋标签，样本进入复核池。
- `expert_decision=reject`：拒绝批量赋标签，样本进入复核池。

再次运行基础入口或兼容入口，并传入已填写的确认表：

```powershell
.\.venv\Scripts\python.exe Platform\tracelog_clustering\cluster_tracelog_samples.py `
  --config configs/preprocess_2D.yaml `
  --excel-path "D:\data\14003_sc_rotorandgeneratorspeeddiffmax_1.xlsx" `
  --trace-path-column tr_path `
  --output-dir tracelog_clustering/output `
  --cluster-mode 1 `
  --expert-confirmation-path "D:\Platform\Platform\tracelog_clustering\output\某次运行目录\cluster_expert_confirmation_template.csv"
```

应用后会增加：

- `expert_labeled_samples.csv`：已应用专家批量标签的样本结果。
- `expert_review_samples.csv`：仍需逐条复核的样本。
- `fault_tree_features_with_expert_labels.csv`：附带专家标签的故障特征表。
- `expert_label_application_summary.csv`：标签应用数量汇总。

## 可视化

对某次完整运行目录执行：

```powershell
.\.venv\Scripts\python.exe Platform\tracelog_clustering\visualize.py `
  --run-dir Platform\tracelog_clustering\output\20260817_120000
```

运行目录必须至少包含 `preprocessed_sequences.npz`。如果存在 `clustering_features.npy`，轮廓系数
会按照本次真实聚类特征空间计算。脚本会在同一目录生成聚类分析图、特征热力图、重点簇诊断和人工解释文件。

## 测试

```powershell
.\.venv\Scripts\python.exe -m pytest `
  Platform\tracelog_clustering\test_fault_feature_rules.py -q
```

测试覆盖：

- YAML 中受控自然语言规则的编译。
- Excel/CSV 确认表读取。
- 未确认规则的运行门禁。
- 专家公式的安全语法校验。
- 已确认规则在样本特征表上的数值计算。

## 使用边界

- `glossary` 只能引用当前特征提取流程已经生成的原子特征。
- 新故障树如果只是组合、阈值或权重变化，只需更新 YAML 并重新确认。
- 新故障树如果引入全新通道或新的时序行为，需要先在基础特征提取中增加对应原子特征。
- 不要直接使用旧确认表匹配新 YAML；规则变化后应重新生成并确认，确保版本和公式可追溯。
