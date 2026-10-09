# 自然语言故障特征确认与聚类

该流程把故障树规则从聚类代码中拆出，分成四步：

1. 在 YAML 中填写故障描述、受控自然语言计算规则和中文术语表。
2. 编译为 Excel/CSV 确认表，展示“故障描述 → 生成公式 → 输入原子特征”的对应关系。
3. 专家填写 `expert_decision`；只有所有启用规则均完成确认后才能继续。
4. 用确认通过的特征生成每条样本的特征值，再执行标准化和聚类。

## 1. 编辑描述参数

复制 `fault_feature_descriptions.example.yaml`。主要字段如下：

- `version`：故障树版本。
- `glossary`：业务中文术语到程序已有原子特征列的映射。
- `feature_id`：稳定的故障特征/故障类别编号。
- `fault_description`：给专家阅读的故障含义。
- `natural_rule`：用于编译的受控自然语言规则。
- `weight`：进入聚类前的业务权重，默认 1。
- `enabled`：是否启用。

受控自然语言支持中文术语、数字、括号、`+ - * /`（也支持“加上、减去、乘以、除以”）、
比较、`且/或`，以及 `较大值(...)`、`较小值(...)`、`绝对值(...)`、`clip(...)`。

示例：

```yaml
glossary:
  发电机下降分数: generator_drop_score
  叶轮上升分数: rotor_rise_score
  传感器稳定分数: sensor_stable_score

features:
  - feature_id: C
    fault_description: 联轴器打滑，表现为发电机转速下降且叶轮转速上升
    natural_rule: 发电机下降分数 * 叶轮上升分数 * 传感器稳定分数
```

这里采用确定性编译，不调用大模型猜测公式。任意自由文本无法可靠、可复现地直接变成数值计算；
故障含义写在 `fault_description`，可执行部分写成接近自然语言的 `natural_rule`，专家可逐行核对。

## 2. 生成确认表

在项目根目录执行：

```powershell
python tracelog_clustering/prepare_fault_feature_confirmation.py `
  --description-config tracelog_clustering/fault_feature_descriptions.example.yaml `
  --output  Platform\tracelog_clustering\fault_feature_confirmation.xlsx
```

确认表包含原始描述、自然语言规则、生成公式、引用的原子特征、编译状态、规则版本和摘要。

专家填写：

- `expert_decision=通过`：该规则可用于聚类。
- `expert_decision=拒绝`：该规则不进入聚类。
- `expert_expression`：可选；若需修正编译公式，在此填写最终公式。
- `expert_comment`：确认备注。

只要存在 `pending` 或编译失败的启用规则，聚类入口就会拒绝运行。确认表中的公式使用 AST 白名单校验，
不能执行导入、属性访问、文件或系统命令。

## 3. 根据确认特征生成聚类

使用兼容入口，并设置 `--cluster-mode 2`：

```powershell
python tracelog_clustering/cluster_with_confirmed_features.py `
  --fault-feature-confirmation-path tracelog_clustering\fault_feature_confirmation.xlsx `
  --config configs/preprocess_2D.yaml `
  --excel-path "D:\data\14003_sc_rotorandgeneratorspeeddiffmax_1.xlsx" `
  --trace-path-column tr_path `
  --output-dir tracelog_clustering\output `
  --cluster-mode 2 `
  --n-clusters 6
```

原聚类脚本保持不变。兼容入口复用原有数据预处理、KMeans/层次聚类、代表样本和专家复核输出，
只替换实际进入聚类的故障特征列。

新增审计信息：

- `fault_tree_features.csv`：包含 `configured_feature_<feature_id>` 和每个样本的候选规则标签。
- `cluster_fault_tree_feature_summary.csv`：各簇已确认特征的中位数。
- `run_metadata.json`：记录确认表绝对路径、最终公式、输入原子特征、权重和本次实际特征列。

## 边界

`glossary` 只能引用现有时序处理已经产生的原子特征。若新故障树引入全新的信号或新的时序行为
（例如“某通道在 3 秒内连续振荡 5 次”且当前没有对应原子指标），仍需先补充一次原子特征提取器；
补充后，同类故障树组合变化只改 YAML 和确认表，无需再改聚类算法。
