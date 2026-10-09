import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples
from scipy.stats import ConstantInputWarning, f_oneway
import argparse
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ============================================================
# 1. 数据加载
# ============================================================
script_dir = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description="分析并可视化 tracelog 聚类结果")
parser.add_argument(
    "--run-dir",
    type=Path,
    default=script_dir / "output",
    help="包含 preprocessed_sequences.npz 的聚类结果目录",
)
args = parser.parse_args()
report_dir = args.run_dir.resolve()
data_path = report_dir / "preprocessed_sequences.npz"
cluster_features_path = report_dir / "clustering_features.npy"
if not data_path.exists():
    raise FileNotFoundError(f"找不到聚类序列文件: {data_path}")

data = np.load(data_path, allow_pickle=True, mmap_mode="r")

X = data["X"]              # (样本数, 通道数, 序列长度)
labels = data["cluster_labels"]  # (样本数,)
channels = data["channels"]      # 通道名列表

n_samples, n_channels, seq_len = X.shape
n_clusters = len(np.unique(labels))

print(f"数据形状: {X.shape}")
print(f"聚类数: {n_clusters}")
print(f"通道数: {n_channels}, 序列长度: {seq_len}")
print(f"通道名: {channels}")

# ============================================================
# 2. 簇分布概览（就是你最开始给的那张表）
# ============================================================
cluster_ids, counts = np.unique(labels, return_counts=True)
df_dist = pd.DataFrame({
    "cluster_label": cluster_ids,
    "sample_count": counts,
    "sample_ratio": counts / n_samples
})
df_dist = df_dist.sort_values("sample_count", ascending=False).reset_index(drop=True)
print("\n===== 簇分布 =====")
print(df_dist.to_string())

# ============================================================
# 3. 特征工程：把 3D 矩阵展平成 2D 用于分析
#    每个样本变成 (通道数 * 序列长度) 的向量
#    但这样维度很高。更实用的做法是：
#    对每个样本的每个通道，提取统计特征（均值、标准差、最大最小值等）
#    => 最终每个样本的特征维度 = 通道数 * 5
# ============================================================
def extract_features_per_channel(X):
    """
    X: (n_samples, n_channels, seq_len)
    返回: (n_samples, n_channels * 5)  5个统计量: mean, std, min, max, range
    """
    feat_list = []
    for ch in range(n_channels):
        ch_data = X[:, ch, :]  # (n_samples, seq_len)
        ch_feat = np.column_stack([
            np.mean(ch_data, axis=1),
            np.std(ch_data, axis=1),
            np.min(ch_data, axis=1),
            np.max(ch_data, axis=1),
            np.ptp(ch_data, axis=1)  # peak to peak (max - min)
        ])
        feat_list.append(ch_feat)
    return np.hstack(feat_list)

X_feat = extract_features_per_channel(X)
# 生成列名，方便后续查看
stat_names = ["mean", "std", "min", "max", "range"]
feature_cols = [f"{ch}_{stat}" for ch in channels for stat in stat_names]

print(f"\n提取后的特征矩阵形状: {X_feat.shape}")

# ============================================================
# 4. 构造更接近人工诊断逻辑的时序特征
#    注意：这些特征不是原始聚类输入，只用于判断簇是否有诊断含义。
#    与 min/max/std 不同，这里更关注：
#    - 转子/发电机转速偏差是否大、持续多久、何时出现；
#    - 不同转速来源之间是否一致；
#    - breaker feedback 是否动作、动作次数、动作先后/一致性。
# ============================================================
def _channel_index(name):
    matches = np.where(channels == name)[0]
    return int(matches[0]) if len(matches) else None


def _safe_corr(a, b):
    a_centered = a - a.mean(axis=1, keepdims=True)
    b_centered = b - b.mean(axis=1, keepdims=True)
    denom = np.sqrt(np.sum(a_centered ** 2, axis=1) * np.sum(b_centered ** 2, axis=1))
    corr = np.divide(
        np.sum(a_centered * b_centered, axis=1),
        denom,
        out=np.zeros(a.shape[0], dtype=float),
        where=denom > 1e-12,
    )
    return corr


def _transition_count(x, threshold=0.0):
    state = x > threshold
    return np.sum(state[:, 1:] != state[:, :-1], axis=1)


def _first_cross_time(x, threshold=0.0):
    state = x > threshold
    any_cross = state.any(axis=1)
    first_idx = np.argmax(state, axis=1)
    result = first_idx / max(x.shape[1] - 1, 1)
    result[~any_cross] = np.nan
    return result


def _add_signal_features(rows, prefix, values):
    rows[f"{prefix}_start_level"] = values[:, : max(1, seq_len // 20)].mean(axis=1)
    rows[f"{prefix}_end_level"] = values[:, -max(1, seq_len // 20):].mean(axis=1)
    rows[f"{prefix}_trend_end_minus_start"] = rows[f"{prefix}_end_level"] - rows[f"{prefix}_start_level"]
    rows[f"{prefix}_peak_time_ratio"] = np.argmax(values, axis=1) / max(seq_len - 1, 1)


def extract_diagnostic_features(X):
    rows = {}

    rotor_idx = _channel_index("rotor_speed")
    generator_idx = _channel_index("generator_speed")
    relay1_idx = _channel_index("grRotorSpeedFormSpeedRelay1")
    relay2_idx = _channel_index("grRotorSpeedFormSpeedRelay2")
    counter_idx = _channel_index("grRotorSpeedFromCounterModule1")
    breaker1_idx = _channel_index("breaker_on_feedback1")
    breaker2_idx = _channel_index("breaker_on_feedback2")

    if rotor_idx is not None:
        rotor = X[:, rotor_idx, :]
        _add_signal_features(rows, "rotor_speed", rotor)

    if generator_idx is not None:
        generator = X[:, generator_idx, :]
        _add_signal_features(rows, "generator_speed", generator)

    if rotor_idx is not None and generator_idx is not None:
        rotor = X[:, rotor_idx, :]
        generator = X[:, generator_idx, :]
        diff = generator - rotor
        abs_diff = np.abs(diff)
        rows["gen_rotor_dev_mean_abs"] = abs_diff.mean(axis=1)
        rows["gen_rotor_dev_peak_abs"] = abs_diff.max(axis=1)
        rows["gen_rotor_dev_over_1_ratio"] = (abs_diff > 1.0).mean(axis=1)
        rows["gen_rotor_dev_peak_time_ratio"] = np.argmax(abs_diff, axis=1) / max(seq_len - 1, 1)
        rows["gen_rotor_corr"] = _safe_corr(generator, rotor)

    if relay2_idx is not None and counter_idx is not None:
        relay2 = X[:, relay2_idx, :]
        counter = X[:, counter_idx, :]
        diff = relay2 - counter
        abs_diff = np.abs(diff)
        rows["relay2_counter_dev_mean_abs"] = abs_diff.mean(axis=1)
        rows["relay2_counter_dev_peak_abs"] = abs_diff.max(axis=1)
        rows["relay2_counter_corr"] = _safe_corr(relay2, counter)

    if relay1_idx is not None and relay2_idx is not None:
        relay1 = X[:, relay1_idx, :]
        relay2 = X[:, relay2_idx, :]
        abs_diff = np.abs(relay1 - relay2)
        rows["relay1_relay2_dev_mean_abs"] = abs_diff.mean(axis=1)
        rows["relay1_relay2_dev_peak_abs"] = abs_diff.max(axis=1)
        rows["relay1_relay2_corr"] = _safe_corr(relay1, relay2)

    for idx, name in [(breaker1_idx, "breaker1"), (breaker2_idx, "breaker2")]:
        if idx is None:
            continue
        signal = X[:, idx, :]
        rows[f"{name}_active_ratio"] = (signal > 0).mean(axis=1)
        rows[f"{name}_transition_count"] = _transition_count(signal)
        rows[f"{name}_first_active_time_ratio"] = _first_cross_time(signal)

    if breaker1_idx is not None and breaker2_idx is not None:
        b1 = X[:, breaker1_idx, :]
        b2 = X[:, breaker2_idx, :]
        rows["breaker_pair_dev_mean_abs"] = np.abs(b1 - b2).mean(axis=1)
        rows["breaker_pair_corr"] = _safe_corr(b1, b2)

    return pd.DataFrame(rows)


df_diag = extract_diagnostic_features(X)
diag_cols = df_diag.columns.tolist()
df_diag["cluster"] = labels

print(f"\n人工诊断候选特征矩阵形状: {df_diag[diag_cols].shape}")

# ============================================================
# 5. 全局聚类质量评估
#    先用聚类输入空间评估，再用人工诊断候选特征空间评估。
# ============================================================
if cluster_features_path.exists():
    X_cluster_space = StandardScaler().fit_transform(np.load(cluster_features_path, mmap_mode="r"))
    cluster_space_label = "实际聚类特征空间"
else:
    X_cluster_space = StandardScaler().fit_transform(X_feat)
    cluster_space_label = "时序统计特征空间"
sil_scores = silhouette_samples(X_cluster_space, labels)
avg_sil = np.mean(sil_scores)
print(f"\n===== 全局平均轮廓系数 (基于{cluster_space_label}): {avg_sil:.4f} =====")

diag_matrix = df_diag[diag_cols].fillna(-1.0).to_numpy()
diag_sil_scores = silhouette_samples(StandardScaler().fit_transform(diag_matrix), labels)
diag_avg_sil = np.mean(diag_sil_scores)
print(f"===== 全局平均轮廓系数 (基于人工诊断候选特征): {diag_avg_sil:.4f} =====")

# 每个簇的平均轮廓系数
df_sil = pd.DataFrame({"cluster": labels, "sil_score": sil_scores})
sil_per_cluster = df_sil.groupby("cluster")["sil_score"].mean().sort_values()
print("\n各簇平均轮廓系数 (基于聚类输入时序空间，从低到高):")
print(sil_per_cluster)

df_diag_sil = pd.DataFrame({"cluster": labels, "sil_score": diag_sil_scores})
diag_sil_per_cluster = df_diag_sil.groupby("cluster")["sil_score"].mean().sort_values()
print("\n各簇平均轮廓系数 (基于人工诊断候选特征，从低到高):")
print(diag_sil_per_cluster)

# ============================================================
# 6. 逐簇区分度分析：用 ANOVA F 值衡量每个特征在不同簇间的差异
#    F 值越大，说明该特征在不同簇之间差异越显著
# ============================================================
df_feat = pd.DataFrame(X_feat, columns=feature_cols)
df_feat["cluster"] = labels

f_scores = {}
for col in feature_cols:
    groups = [df_feat[df_feat["cluster"] == c][col].values for c in cluster_ids]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        f_stat, p_val = f_oneway(*groups)
    f_scores[col] = f_stat

# 挑出最具区分力的 Top 15 特征
top_features = sorted(f_scores.items(), key=lambda x: x[1], reverse=True)[:15]
print(f"\n===== 统计解释特征 Top 15 (非人工诊断依据，仅供参考) =====")
for feat, f_val in top_features:
    print(f"  {feat}: F={f_val:.2f}")

diag_f_scores = {}
for col in diag_cols:
    groups = [df_diag[df_diag["cluster"] == c][col].dropna().values for c in cluster_ids]
    groups = [group for group in groups if len(group) > 0]
    if len(groups) >= 2:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConstantInputWarning)
            f_stat, p_val = f_oneway(*groups)
        diag_f_scores[col] = f_stat

top_diag_features = sorted(diag_f_scores.items(), key=lambda x: x[1], reverse=True)[:15]
top_diag_feature_names = [f for f, _ in top_diag_features]
print(f"\n===== 人工诊断候选特征 Top 15 (ANOVA F 值) =====")
for feat, f_val in top_diag_features:
    print(f"  {feat}: F={f_val:.2f}")

# ============================================================
# 7. 逐簇特征画像：每个簇在每个特征上的均值，与全局均值的对比
#    (Heatmap 数据)
# ============================================================
cluster_profiles = df_feat.groupby("cluster")[feature_cols].mean()
# 标准化每个特征（按列），看每个簇相对于全局是偏高还是偏低
scaler_profile = StandardScaler()
profile_scaled = pd.DataFrame(
    scaler_profile.fit_transform(cluster_profiles),
    index=cluster_profiles.index,
    columns=feature_cols
)

diag_cluster_profiles = df_diag.groupby("cluster")[diag_cols].mean()
diag_profile_scaled = pd.DataFrame(
    StandardScaler().fit_transform(diag_cluster_profiles.fillna(-1.0)),
    index=diag_cluster_profiles.index,
    columns=diag_cols
)

# ============================================================
# 8. 全部簇的人工特征解释
#    目标：不只看重点簇，而是对当前所有 13 类都输出可读解释。
#    解释依据：
#    - 每个簇在人工诊断候选特征上的均值/中位数；
#    - 每个簇相对所有簇均值画像的 Z-score 偏移；
#    - 原始时序空间、人工特征空间的簇内轮廓系数。
# ============================================================
FEATURE_DESCRIPTIONS = {
    "rotor_speed_start_level": ("转子转速起始水平", "起始转子转速更高", "起始转子转速更低"),
    "rotor_speed_end_level": ("转子转速结束水平", "结束转子转速更高", "结束转子转速更低"),
    "rotor_speed_trend_end_minus_start": ("转子转速前后变化", "转子转速后段相对前段上升", "转子转速后段相对前段下降"),
    "rotor_speed_peak_time_ratio": ("转子转速峰值时刻", "转子转速峰值出现更晚", "转子转速峰值出现更早"),
    "generator_speed_start_level": ("发电机转速起始水平", "起始发电机转速更高", "起始发电机转速更低"),
    "generator_speed_end_level": ("发电机转速结束水平", "结束发电机转速更高", "结束发电机转速更低"),
    "generator_speed_trend_end_minus_start": ("发电机转速前后变化", "发电机转速后段相对前段上升", "发电机转速后段相对前段下降"),
    "generator_speed_peak_time_ratio": ("发电机转速峰值时刻", "发电机转速峰值出现更晚", "发电机转速峰值出现更早"),
    "gen_rotor_dev_mean_abs": ("发电机-转子转速平均偏差", "发电机与转子转速平均偏差更大", "发电机与转子转速平均偏差更小"),
    "gen_rotor_dev_peak_abs": ("发电机-转子转速峰值偏差", "发电机与转子转速峰值偏差更大", "发电机与转子转速峰值偏差更小"),
    "gen_rotor_dev_over_1_ratio": ("发电机-转子偏差超阈占比", "转速偏差超 1 的持续占比更高", "转速偏差超 1 的持续占比更低"),
    "gen_rotor_dev_peak_time_ratio": ("发电机-转子最大偏差时刻", "最大转速偏差出现更晚", "最大转速偏差出现更早"),
    "gen_rotor_corr": ("发电机-转子转速相关性", "发电机与转子转速同步性更强", "发电机与转子转速同步性更弱"),
    "relay2_counter_dev_mean_abs": ("继电器2-计数模块平均偏差", "继电器2与计数模块平均偏差更大", "继电器2与计数模块平均偏差更小"),
    "relay2_counter_dev_peak_abs": ("继电器2-计数模块峰值偏差", "继电器2与计数模块峰值偏差更大", "继电器2与计数模块峰值偏差更小"),
    "relay2_counter_corr": ("继电器2-计数模块相关性", "继电器2与计数模块同步性更强", "继电器2与计数模块同步性更弱"),
    "relay1_relay2_dev_mean_abs": ("继电器1-继电器2平均偏差", "两个继电器转速平均偏差更大", "两个继电器转速平均偏差更小"),
    "relay1_relay2_dev_peak_abs": ("继电器1-继电器2峰值偏差", "两个继电器转速峰值偏差更大", "两个继电器转速峰值偏差更小"),
    "relay1_relay2_corr": ("继电器1-继电器2相关性", "两个继电器转速同步性更强", "两个继电器转速同步性更弱"),
    "breaker1_active_ratio": ("断路器1反馈有效占比", "断路器1反馈有效持续占比更高", "断路器1反馈有效持续占比更低"),
    "breaker1_transition_count": ("断路器1反馈跳变次数", "断路器1反馈跳变更多", "断路器1反馈跳变更少"),
    "breaker1_first_active_time_ratio": ("断路器1首次有效时刻", "断路器1首次有效更晚", "断路器1首次有效更早"),
    "breaker2_active_ratio": ("断路器2反馈有效占比", "断路器2反馈有效持续占比更高", "断路器2反馈有效持续占比更低"),
    "breaker2_transition_count": ("断路器2反馈跳变次数", "断路器2反馈跳变更多", "断路器2反馈跳变更少"),
    "breaker2_first_active_time_ratio": ("断路器2首次有效时刻", "断路器2首次有效更晚", "断路器2首次有效更早"),
    "breaker_pair_dev_mean_abs": ("断路器双反馈不一致度", "两个断路器反馈不一致度更高", "两个断路器反馈不一致度更低"),
    "breaker_pair_corr": ("断路器双反馈相关性", "两个断路器反馈同步性更强", "两个断路器反馈同步性更弱"),
}


def _feature_phrase(feature, z_value):
    desc = FEATURE_DESCRIPTIONS.get(feature)
    if desc is None:
        direction = "偏高" if z_value >= 0 else "偏低"
        return f"{feature}{direction}"
    _, high_phrase, low_phrase = desc
    return high_phrase if z_value >= 0 else low_phrase


def _feature_label(feature):
    desc = FEATURE_DESCRIPTIONS.get(feature)
    return desc[0] if desc is not None else feature


def _format_feature_value(value):
    if pd.isna(value):
        return "NA"
    return f"{value:.3g}"


def _explanation_piece(feature, z_value, median_value):
    if pd.isna(median_value) and feature.endswith("_first_active_time_ratio"):
        return f"{_feature_label(feature)}未检测到有效触发(中位数=NA, Z={z_value:.2f})"
    return f"{_feature_phrase(feature, z_value)}(中位数={_format_feature_value(median_value)}, Z={z_value:.2f})"


def build_cluster_manual_explanations(top_n=5, z_threshold=0.75):
    summary_rows = []
    explanation_rows = []
    cluster_order = sorted(cluster_ids.astype(int).tolist())

    global_stats = df_diag[diag_cols].agg(["mean", "median", "std", "min", "max"])

    for c in cluster_order:
        part = df_diag[df_diag["cluster"] == c]
        z_series = diag_profile_scaled.loc[c, diag_cols].replace([np.inf, -np.inf], np.nan).dropna()
        dominant = z_series.reindex(z_series.abs().sort_values(ascending=False).index)
        dominant = dominant[dominant.abs() >= z_threshold]
        if dominant.empty:
            dominant = z_series.reindex(z_series.abs().sort_values(ascending=False).index)

        selected_features = dominant.head(top_n).index.tolist()
        high_features = [f for f in selected_features if z_series.get(f, 0) > 0]
        low_features = [f for f in selected_features if z_series.get(f, 0) < 0]

        for feature in selected_features:
            summary_rows.append({
                "cluster": c,
                "sample_count": int(len(part)),
                "sample_ratio": len(part) / n_samples,
                "feature": feature,
                "feature_label": _feature_label(feature),
                "cluster_mean": part[feature].mean(),
                "cluster_median": part[feature].median(),
                "cluster_std": part[feature].std(),
                "cluster_min": part[feature].min(),
                "cluster_max": part[feature].max(),
                "global_mean": global_stats.loc["mean", feature],
                "global_median": global_stats.loc["median", feature],
                "global_std": global_stats.loc["std", feature],
                "z_score_vs_global_cluster_profile": z_series.get(feature, np.nan),
            })

        phrases = []
        for feature in selected_features[:3]:
            value = part[feature].median()
            z_value = z_series.get(feature, 0)
            phrases.append(_explanation_piece(feature, z_value, value))

        seq_sil = sil_per_cluster.get(c, np.nan)
        diag_sil = diag_sil_per_cluster.get(c, np.nan)
        if pd.isna(seq_sil) or pd.isna(diag_sil):
            consistency = "轮廓系数缺失，需结合样本波形复核"
        elif seq_sil >= avg_sil and diag_sil >= diag_avg_sil:
            consistency = "原始时序和人工特征空间均相对清晰"
        elif seq_sil >= avg_sil:
            consistency = "原始时序空间较清晰，但人工诊断特征解释力一般"
        elif diag_sil >= diag_avg_sil:
            consistency = "人工诊断特征空间相对清晰，但原始时序边界一般"
        else:
            consistency = "两类空间轮廓系数均偏低，可能是过渡型或需合并复核"

        explanation_rows.append({
            "cluster": c,
            "sample_count": int(len(part)),
            "sample_ratio": len(part) / n_samples,
            "sequence_silhouette_mean": seq_sil,
            "diagnostic_feature_silhouette_mean": diag_sil,
            "top_high_features": "; ".join([_feature_label(f) for f in high_features]),
            "top_low_features": "; ".join([_feature_label(f) for f in low_features]),
            "manual_feature_explanation": "；".join(phrases),
            "interpretation_note": consistency,
        })

    return pd.DataFrame(summary_rows), pd.DataFrame(explanation_rows)


df_all_cluster_feature_summary, df_all_cluster_explanations = build_cluster_manual_explanations()
all_summary_path = report_dir / "叶轮转速_cluster_manual_feature_summary.csv"
all_explanation_path = report_dir / "叶轮转速_manual_explanations.csv"
all_explanation_md_path = report_dir / "叶轮转速_manual_explanations.md"
df_all_cluster_feature_summary.to_csv(all_summary_path, index=False, encoding="utf-8-sig")
df_all_cluster_explanations.to_csv(all_explanation_path, index=False, encoding="utf-8-sig")

with open(all_explanation_md_path, "w", encoding="utf-8") as f:
    f.write("# 全部簇人工特征解释\n\n")
    f.write("说明：以下解释基于人工诊断候选特征的簇间相对偏移，不代表聚类训练时直接使用了这些人工规则。\n\n")
    for _, row in df_all_cluster_explanations.sort_values("cluster").iterrows():
        f.write(f"## Cluster {int(row['cluster'])}\n\n")
        f.write(f"- 样本数/占比：{int(row['sample_count'])} / {row['sample_ratio']:.2%}\n")
        f.write(f"- 轮廓系数：时序空间 {row['sequence_silhouette_mean']:.3f}，人工特征空间 {row['diagnostic_feature_silhouette_mean']:.3f}\n")
        f.write(f"- 主要解释：{row['manual_feature_explanation']}\n")
        f.write(f"- 判读备注：{row['interpretation_note']}\n\n")

print("\n===== 全部簇人工特征解释 =====")
print(df_all_cluster_explanations.to_string(index=False))
print("\n全部簇人工特征解释已保存:")
print(f"  {all_summary_path}")
print(f"  {all_explanation_path}")
print(f"  {all_explanation_md_path}")

all_heatmap_features = df_all_cluster_feature_summary["feature"].drop_duplicates().tolist()
fig_all_heat, ax_all_heat = plt.subplots(figsize=(18, 8))
sns.heatmap(
    diag_profile_scaled.loc[sorted(cluster_ids.astype(int).tolist()), all_heatmap_features],
    cmap="RdBu_r",
    center=0,
    annot=True,
    fmt=".1f",
    linewidths=.5,
    cbar_kws={'label': 'Z-score vs global cluster mean'},
    ax=ax_all_heat
)
ax_all_heat.set_title("All Clusters: Manual Diagnostic Feature Profile", fontsize=14)
ax_all_heat.set_xlabel("Manual diagnostic feature")
ax_all_heat.set_ylabel("Cluster")
plt.xticks(rotation=45, ha="right")
fig_all_heat.tight_layout()
all_heatmap_path = report_dir / "叶轮转速_manual_feature_heatmap.png"
fig_all_heat.savefig(all_heatmap_path, dpi=150, bbox_inches="tight")
print(f"  {all_heatmap_path}")

# ============================================================
# 8. 可视化
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(22, 14))

# --- 图1: 簇样本量分布 ---
ax1 = axes[0, 0]
colors = sns.color_palette("tab20", n_clusters)
bars = ax1.bar(df_dist["cluster_label"].astype(str), df_dist["sample_count"], color=colors)
ax1.set_title("Cluster Sample Count", fontsize=14)
ax1.set_xlabel("Cluster")
ax1.set_ylabel("Sample Count")
for bar, ratio in zip(bars, df_dist["sample_ratio"]):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 3,
             f'{ratio:.1%}', ha='center', fontsize=8)

# --- 图2: 各簇轮廓系数（箱线图） ---
ax2 = axes[0, 1]
order = sil_per_cluster.index.tolist()
df_sil["cluster"] = df_sil["cluster"].astype(int)
sns.boxplot(x="cluster", y="sil_score",hue="cluster", data=df_sil, order=order, palette="Set3",legend=False, ax=ax2)
ax2.axhline(y=avg_sil, color='red', linestyle='--', label=f'Avg: {avg_sil:.3f}')
ax2.set_title("Silhouette Scores by Cluster", fontsize=14)
ax2.legend()

# --- 图3: PCA 降维散点图（看簇的分布和重叠程度） ---
ax3 = axes[0, 2]
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_cluster_space)
df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
df_pca["cluster"] = labels.astype(int)
scatter = sns.scatterplot(x="PC1", y="PC2", hue="cluster", data=df_pca,
                          palette="tab20", alpha=0.6, ax=ax3, legend="brief")
ax3.set_title(f"PCA on Sequence Space. Expl.Var: {pca.explained_variance_ratio_.sum():.1%}", fontsize=14)
ax3.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title="Cluster")

# --- 图4: 人工诊断候选特征在簇间的分布差异（热力图） ---
ax4 = axes[1, 0]
top_feat_names = top_diag_feature_names
sns.heatmap(diag_profile_scaled[top_feat_names].T, cmap="RdBu_r", center=0,
            annot=True, fmt=".1f", linewidths=.5, ax=ax4, cbar_kws={'label': 'Z-score'})
ax4.set_title("Top Diagnostic Candidate Features (Cluster Z-score vs Global)", fontsize=14)
ax4.set_ylabel("Feature")

# --- 图5: 每个簇的人工诊断候选特征 Z-score 全局热力图 ---
ax5 = axes[1, 1]
sns.heatmap(diag_profile_scaled.T, cmap="RdBu_r", center=0,
            xticklabels=True, yticklabels=False, ax=ax5,
            cbar_kws={'label': 'Z-score'})
ax5.set_title("Diagnostic Candidate Profile (Clusters vs Global Mean)", fontsize=14)
ax5.set_xlabel("Cluster")

# --- 图6: 簇大小（饼图） ---
ax6 = axes[1, 2]
threshold = 0.03  # 小于3%的标签合并显示
small_mask = df_dist["sample_ratio"] < threshold
pie_data = df_dist.copy()
if small_mask.any():
    other_sum = pie_data.loc[small_mask, "sample_ratio"].sum()
    pie_data = pie_data[~small_mask]
    pie_data = pd.concat([pie_data, pd.DataFrame({"cluster_label": ["Other(<3% each)"],
                                                  "sample_ratio": [other_sum]})],
                         ignore_index=True)
ax6.pie(pie_data["sample_ratio"], labels=pie_data["cluster_label"], autopct='%1.1f%%',
        colors=sns.color_palette("tab20", len(pie_data)), startangle=90)
ax6.set_title("Cluster Proportion", fontsize=14)

plt.tight_layout()
plt.savefig(report_dir / "cluster_analysis_report.png", dpi=150, bbox_inches="tight")
plt.show()

# ============================================================
# 9. 重点问题簇诊断：查看 cluster 3/8/1/12 在人工诊断候选特征上的真实分布
#    目标：
#    1) 看这些簇是否在诊断相关特征上有稳定、可解释的偏高/偏低；
#    2) 看簇内分布是否很宽，是否与其他簇大量重叠；
#    3) 判断它们更像物理状态，还是算法硬切出来的边界。
# ============================================================
focus_clusters = [3, 8, 1, 12]
focus_clusters = [c for c in focus_clusters if c in set(labels.astype(int))]
top_focus_features = top_diag_feature_names[:12]

if focus_clusters:
    df_focus = df_diag[df_diag["cluster"].isin(focus_clusters)].copy()
    focus_order = [c for c in focus_clusters if c in df_focus["cluster"].unique()]

    # 9.1 人工诊断候选特征箱线图：看中位数、离散程度和异常值
    df_focus_long = df_focus.melt(
        id_vars="cluster",
        value_vars=top_focus_features,
        var_name="feature",
        value_name="value"
    )

    n_cols = 3
    n_rows = int(np.ceil(len(top_focus_features) / n_cols))
    fig_box, axes_box = plt.subplots(n_rows, n_cols, figsize=(20, 4.2 * n_rows))
    axes_box = np.asarray(axes_box).reshape(-1)

    for ax, feature in zip(axes_box, top_focus_features):
        sns.boxplot(
            x="cluster",
            y="value",
            hue="cluster",
            data=df_focus_long[df_focus_long["feature"] == feature],
            order=focus_order,
            palette="Set2",
            legend=False,
            ax=ax
        )
        sns.stripplot(
            x="cluster",
            y="value",
            data=df_focus_long[df_focus_long["feature"] == feature],
            order=focus_order,
            color="black",
            alpha=0.25,
            size=2,
            jitter=0.25,
            ax=ax
        )
        ax.set_title(feature, fontsize=11)
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Feature value")

    for ax in axes_box[len(top_focus_features):]:
        ax.axis("off")

    fig_box.suptitle("Focused Clusters: Diagnostic Candidate Feature Distributions", fontsize=16, y=1.01)
    fig_box.tight_layout()
    boxplot_path = report_dir / "focused_cluster_diagnostic_feature_boxplots.png"
    fig_box.savefig(boxplot_path, dpi=150, bbox_inches="tight")
    plt.show()

    # 9.2 Z-score 偏移热力图：看每个重点簇相对全局均值的诊断画像
    focus_profile_scaled = diag_profile_scaled.loc[focus_order, top_focus_features]
    fig_heat, ax_heat = plt.subplots(figsize=(16, 4.8))
    sns.heatmap(
        focus_profile_scaled,
        cmap="RdBu_r",
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=.5,
        cbar_kws={'label': 'Z-score vs global cluster mean'},
        ax=ax_heat
    )
    ax_heat.set_title("Focused Clusters: Diagnostic Candidate Z-score Profile", fontsize=14)
    ax_heat.set_xlabel("Feature")
    ax_heat.set_ylabel("Cluster")
    plt.xticks(rotation=45, ha="right")
    fig_heat.tight_layout()
    heatmap_path = report_dir / "focused_cluster_diagnostic_profile_heatmap.png"
    fig_heat.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.show()

    # 9.3 PCA 局部图：只高亮重点簇，看它们在聚类输入时序空间里是否混在一起
    df_pca_focus = df_pca.copy()
    df_pca_focus["group"] = np.where(
        df_pca_focus["cluster"].isin(focus_order),
        df_pca_focus["cluster"].astype(str),
        "Other"
    )
    hue_order = [str(c) for c in focus_order] + ["Other"]
    palette = {str(c): color for c, color in zip(focus_order, sns.color_palette("Set2", len(focus_order)))}
    palette["Other"] = "lightgray"

    fig_pca, ax_pca = plt.subplots(figsize=(10, 7))
    sns.scatterplot(
        x="PC1",
        y="PC2",
        hue="group",
        hue_order=hue_order,
        data=df_pca_focus,
        palette=palette,
        alpha=0.7,
        s=35,
        ax=ax_pca
    )
    ax_pca.set_title("PCA Highlight: Focused Clusters vs Others", fontsize=14)
    ax_pca.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig_pca.tight_layout()
    pca_path = report_dir / "focused_cluster_pca_highlight.png"
    fig_pca.savefig(pca_path, dpi=150, bbox_inches="tight")
    plt.show()

    # 9.4 输出表格：均值、中位数、标准差，便于人工判读和写诊断报告
    summary_rows = []
    for c in focus_order:
        part = df_diag[df_diag["cluster"] == c]
        for feature in top_focus_features:
            summary_rows.append({
                "cluster": c,
                "feature": feature,
                "mean": part[feature].mean(),
                "median": part[feature].median(),
                "std": part[feature].std(),
                "min": part[feature].min(),
                "max": part[feature].max(),
                "z_score_vs_global": diag_profile_scaled.loc[c, feature]
            })
    df_focus_summary = pd.DataFrame(summary_rows)
    summary_path = report_dir / "focused_cluster_diagnostic_feature_summary.csv"
    df_focus_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n===== 重点问题簇人工诊断候选特征画像 =====")
    print(df_focus_summary.sort_values(["cluster", "z_score_vs_global"], ascending=[True, False]).to_string(index=False))
    print("\n重点簇诊断图片已保存:")
    print(f"  {boxplot_path}")
    print(f"  {heatmap_path}")
    print(f"  {pca_path}")
    print(f"重点簇统计表已保存: {summary_path}")

# ============================================================
# 10. 自动诊断：识别需要重点关注的“问题簇”
# ============================================================
print("\n===== 自动诊断报告 =====")

# 规则1：样本占比 < 2% 且轮廓系数低于全局平均
for _, row in df_dist.iterrows():
    c = row["cluster_label"]
    ratio = row["sample_ratio"]
    sil = sil_per_cluster.get(c, 0)
    if ratio < 0.02 and sil < avg_sil:
        print(f"[警告] 簇 {c}: 样本占比仅 {ratio:.2%}, 且轮廓系数 {sil:.3f} < 全局均值 {avg_sil:.3f}。建议讨论是否合并。")

# 规则2：样本占比 > 15%，但轮廓系数 < 0.1
for _, row in df_dist.iterrows():
    c = row["cluster_label"]
    ratio = row["sample_ratio"]
    sil = sil_per_cluster.get(c, 0)
    if ratio > 0.15 and sil < 0.1:
        print(f"[注意] 簇 {c}: 作为大簇(占比{ratio:.1%}), 轮廓系数仅为 {sil:.3f}，内部可能非常松散，需检查特征一致性。")

print(f"分析完成。完整图片已保存为 {report_dir / 'cluster_analysis_report.png'}")
