import argparse
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


LOGGER = logging.getLogger("tracelog_clustering")

SLIP_RING_RELAY = "grRotorSpeedFormSpeedRelay1"
SLIP_RING_COUNTER = "grRotorSpeedFromCounterModule1"
PROXIMITY_RELAY = "grRotorSpeedFormSpeedRelay2"
PROXIMITY_COUNTER = "grRotorSpeedFromCounterModule2"
GENERATOR_SPEED = "generator_speed"
ROTOR_SPEED = "rotor_speed"
CONVERTER_GENERATOR_SPEED = "converter_generator_speed"
OPERATION_MODE = "operation_mode"

CORE_FAULT_TREE_FEATURES = [
    "slip_ring_jump_score",
    "A_slip_ring_jump_score",
    "proximity_jump_score",
    "B_proximity_jump_score",
    "generator_drop_score",
    "rotor_rise_score",
    "gen_drop_rotor_rise_overlap_ratio",
    "C_coupling_slip_score",
    "generator_jump_score",
    "D_generator_change_rate_score",
    "E_both_sensor_jump_score",
    "both_sensor_jump_overlap_ratio",
    "converter_zero_score",
    "converter_drop_score",
    "F_converter_fault_score",
    "converter_negative_ratio",
    "G_reverse_rotation_score",
    "slip_ring_zero_score",
    "proximity_zero_score",
    "sensor_zero_score",
    "rotor_drop_score",
    "H_sensor_zero_rotor_drop_score",
    "gen_rotor_corr",
    "gen_rotor_dev_peak_abs",
    "gen_converter_corr",
    "gen_converter_dev_peak_abs",
    "slip_ring_relay_counter_corr",
    "slip_ring_relay_counter_dev_peak_abs",
    "proximity_relay_counter_corr",
    "proximity_relay_counter_dev_peak_abs",
]


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def resolve_project_root(script_path: Path) -> Path:
    return script_path.resolve().parents[1]


def load_yaml_config(config_path: Path, project_root: Path) -> Dict:
    with config_path.open("r", encoding="utf-8") as file:
        raw_cfg = yaml.safe_load(file) or {}

    def _replace(value):
        if isinstance(value, str):
            return value.replace("__PROJECT_ROOT__", str(project_root))
        if isinstance(value, list):
            return [_replace(item) for item in value]
        if isinstance(value, dict):
            return {key: _replace(item) for key, item in value.items()}
        return value

    return _replace(raw_cfg)


def load_column_mapping(mapping_file: Path) -> Dict[str, List[str]]:
    df = pd.read_excel(mapping_file)
    mapping: Dict[str, List[str]] = {}

    for _, row in df.iterrows():
        std_name = str(row.iloc[0]).strip()
        if not std_name or std_name == "nan":
            continue

        candidates: List[str] = []
        for value in row.iloc[2:]:
            value_str = str(value).strip()
            if value_str and value_str not in {"1", "nan"}:
                candidates.append(value_str)

        if candidates:
            mapping[std_name] = candidates

    if not mapping:
        raise ValueError(f"列名映射为空: {mapping_file}")

    return mapping


@dataclass
class SequenceSample:
    sample_index: int
    tracelog_path: str
    row_count: int
    time_start: float
    time_end: float
    data: np.ndarray
    auxiliary_data: Optional[Dict[str, np.ndarray]] = None


class TracelogWindowPreprocessor:
    def __init__(
        self,
        column_mapping: Dict[str, List[str]],
        main_sensor_channels: Sequence[str],
        timestamp_window: Tuple[float, float],
        default_fault_time: float,
        min_rows_after_window: int,
        time_offset_threshold: float,
        missing_channel_strategy: str = "zero",
    ) -> None:
        self.column_mapping = column_mapping
        self.main_sensor_channels = list(main_sensor_channels)
        self.timestamp_window = timestamp_window
        self.default_fault_time = default_fault_time
        self.min_rows_after_window = min_rows_after_window
        self.time_offset_threshold = time_offset_threshold
        self.missing_channel_strategy = missing_channel_strategy

        all_candidates: List[str] = []
        for values in column_mapping.values():
            all_candidates.extend(values)
        self._all_candidates = sorted(set(all_candidates), key=len, reverse=True)

    def _get_error_index(self, data: pd.DataFrame) -> Tuple[Optional[int], Optional[str], List[int]]:
        mode_columns = [
            "iOperationMode",
            "giWindTurbineOperationMode",
            "ioperation_mode",
            "iTurbineOperationMode",
            "giTurbineOperationMode",
            "GVL_OP_gS_OP_giOperationMode",
            "rOperationMode",
            "grOperationMode",
            "giOperationMode",
        ]

        mode_column = next((column for column in mode_columns if column in data.columns), None)
        if mode_column is None:
            return None, None, []

        mode_series = pd.to_numeric(data[mode_column], errors="coerce")
        error_indexes: List[int] = []

        for i in range(1, len(mode_series)):
            prev_mode = mode_series.iloc[i - 1]
            curr_mode = mode_series.iloc[i]
            if pd.isna(prev_mode) or pd.isna(curr_mode):
                continue
            prev_mode = int(prev_mode)
            curr_mode = int(curr_mode)
            if curr_mode in range(1, 8) and prev_mode in range(8, 16):
                error_indexes.append(i - 1)

        if error_indexes:
            return error_indexes[0], mode_column, error_indexes
        return None, mode_column, error_indexes

    def _robust_read_file(self, file_path: Path) -> pd.DataFrame:
        df: Optional[pd.DataFrame] = None

        for encoding in ["gbk", "utf-8", "utf-8-sig", "latin-1"]:
            try:
                with file_path.open("rb") as file:
                    raw_head = file.read(4096).decode(encoding, errors="ignore")

                sep_counts = {
                    ";": raw_head.count(";"),
                    ",": raw_head.count(","),
                    "\t": raw_head.count("\t"),
                }
                separator = max(sep_counts, key=sep_counts.get)
                if sep_counts[separator] == 0:
                    separator = ","

                try:
                    df = pd.read_csv(file_path, sep=separator, index_col=False, encoding=encoding, engine="c")
                except Exception:
                    df = pd.read_csv(file_path, sep=separator, index_col=False, encoding=encoding, engine="python")

                if len(df.columns) <= 1:
                    for skip_rows in range(1, 6):
                        try:
                            df_retry = pd.read_csv(
                                file_path,
                                sep=separator,
                                index_col=False,
                                encoding=encoding,
                                engine="c",
                                skiprows=skip_rows,
                            )
                        except Exception:
                            try:
                                df_retry = pd.read_csv(
                                    file_path,
                                    sep=separator,
                                    index_col=False,
                                    encoding=encoding,
                                    engine="python",
                                    skiprows=skip_rows,
                                )
                            except Exception:
                                continue
                        if len(df_retry.columns) > 1:
                            df = df_retry
                            break

                if df is not None and len(df.columns) > 1:
                    break
            except Exception:
                continue

        if df is None or len(df.columns) <= 1:
            raise ValueError(f"无法解析 tracelog 文件: {file_path}")

        first_col = str(df.columns[0])
        for candidate in self._all_candidates:
            prefix_len = len(first_col) - len(candidate)
            if prefix_len >= 10 and first_col.endswith(candidate):
                df.rename(columns={first_col: candidate}, inplace=True)
                break

        return df

    def _find_best_time_column(self, df: pd.DataFrame) -> Optional[str]:
        time_key = "timestamp" if "timestamp" in self.column_mapping else "time_stamp"
        potential_cols = [col for col in self.column_mapping.get(time_key, []) if col in df.columns]

        best_time_col = None
        max_std = -1.0
        for col in potential_cols:
            series = pd.to_numeric(df[col], errors="coerce").fillna(0)
            std_value = float(series.std())
            if std_value > max_std:
                max_std = std_value
                best_time_col = col

        if best_time_col is not None and max_std > 0:
            return best_time_col

        for col in potential_cols:
            if col not in df.columns:
                continue
            try:
                sample_series = df[col].dropna()
                if sample_series.empty:
                    continue
                sample_value = str(sample_series.iloc[0])
                parsed = None
                if len(sample_value.split("-")) >= 6:
                    parsed = pd.to_datetime(df[col], format="%Y-%m-%d-%H-%M-%S-%f", errors="coerce")
                if parsed is None or parsed.notna().sum() < len(df) * 0.5:
                    parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().sum() > len(df) * 0.5:
                    ref_time = parsed.dropna().iloc[len(parsed.dropna()) // 2]
                    df[col] = (parsed - ref_time).dt.total_seconds()
                    return col
            except Exception:
                continue

        return None

    def _build_mapped_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        error_index, mode_column, _ = self._get_error_index(df)
        mapped_cols: Dict[str, str] = {}

        if error_index is not None:
            if error_index <= 0:
                raise ValueError(f"识别到故障索引 {error_index}，无法提供故障前窗口")
            df["fault_index_timestamp"] = np.arange(len(df), dtype=np.float64) - float(error_index)
            mapped_cols["fault_index_timestamp"] = "timestamp"
        else:
            if mode_column is not None:
                raise ValueError(f"运行模式列 '{mode_column}' 未识别到 8~15 -> 1~7 跳变")
            best_time_col = self._find_best_time_column(df)
            if best_time_col is None:
                row_count = len(df)
                df["forced_time"] = np.linspace(-row_count / 2, row_count / 2, row_count)
                mapped_cols["forced_time"] = "timestamp"
            else:
                mapped_cols[best_time_col] = "timestamp"

        for std_col in self.main_sensor_channels:
            candidates = self.column_mapping.get(std_col, [])
            matched = next((candidate for candidate in candidates if candidate in df.columns), None)
            if matched is not None:
                mapped_cols[matched] = std_col

        if mode_column is not None and mode_column not in mapped_cols:
            mapped_cols[mode_column] = OPERATION_MODE

        if "timestamp" not in mapped_cols.values():
            raise ValueError("无法构造 timestamp 列")

        processed = df[list(mapped_cols.keys())].rename(columns=mapped_cols).copy()
        for col in processed.columns:
            processed[col] = pd.to_numeric(processed[col], errors="coerce")

        raw_mean = float(processed["timestamp"].mean())
        if abs(raw_mean) > self.time_offset_threshold:
            processed["timestamp"] = processed["timestamp"] - self.default_fault_time

        start_time, end_time = self.timestamp_window
        processed = processed[
            (processed["timestamp"] >= start_time) & (processed["timestamp"] <= end_time)
        ].copy()

        if len(processed) < self.min_rows_after_window:
            raise ValueError(
                f"时间窗口截取后数据不足 {self.min_rows_after_window} 行，实际 {len(processed)} 行"
            )

        processed.sort_values("timestamp", inplace=True)
        processed.drop_duplicates(subset=["timestamp"], keep="first", inplace=True)

        for channel in self.main_sensor_channels:
            if channel not in processed.columns:
                if self.missing_channel_strategy == "skip":
                    raise ValueError(f"缺失通道: {channel}")
                processed[channel] = 0.0

        auxiliary_columns = [OPERATION_MODE] if OPERATION_MODE in processed.columns else []
        ordered_columns = ["timestamp", *self.main_sensor_channels, *auxiliary_columns]
        processed = processed.reindex(columns=ordered_columns)
        processed.replace([np.inf, -np.inf], np.nan, inplace=True)
        processed.interpolate(method="linear", limit_direction="both", inplace=True)
        processed.fillna(0.0, inplace=True)

        return processed

    def preprocess(self, tracelog_path: str, sample_index: int) -> SequenceSample:
        file_path = Path(tracelog_path)
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {tracelog_path}")

        df = self._robust_read_file(file_path)
        processed = self._build_mapped_dataframe(df)

        timestamps = processed["timestamp"].to_numpy(dtype=np.float64)
        values = processed[self.main_sensor_channels].to_numpy(dtype=np.float64).T
        auxiliary_data = {
            column: processed[column].to_numpy(dtype=np.float64)
            for column in processed.columns
            if column not in {"timestamp", *self.main_sensor_channels}
        }

        return SequenceSample(
            sample_index=sample_index,
            tracelog_path=str(file_path),
            row_count=processed.shape[0],
            time_start=float(timestamps[0]),
            time_end=float(timestamps[-1]),
            data=np.vstack([timestamps[np.newaxis, :], values]),
            auxiliary_data=auxiliary_data,
        )


def infer_target_length(samples: Sequence[SequenceSample]) -> int:
    lengths = [sample.data.shape[1] for sample in samples]
    return int(np.median(lengths))


def resample_sample(sample: SequenceSample, main_sensor_channels: Sequence[str], target_length: int) -> np.ndarray:
    timestamps = sample.data[0]
    channel_values = sample.data[1:]

    if target_length < 2:
        raise ValueError("target_length 必须至少为 2")

    if np.unique(timestamps).size < 2:
        grid = np.linspace(timestamps[0] - 1.0, timestamps[0] + 1.0, target_length)
    else:
        grid = np.linspace(timestamps.min(), timestamps.max(), target_length)

    resampled_channels: List[np.ndarray] = []
    for channel_idx, _ in enumerate(main_sensor_channels):
        values = channel_values[channel_idx]
        if np.all(np.isnan(values)):
            resampled = np.zeros(target_length, dtype=np.float64)
        else:
            clean_values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            resampled = np.interp(grid, timestamps, clean_values)
        resampled_channels.append(resampled)

    matrix = np.stack(resampled_channels, axis=0)
    per_channel_mean = matrix.mean(axis=1, keepdims=True)
    per_channel_std = matrix.std(axis=1, keepdims=True)
    per_channel_std[per_channel_std == 0] = 1.0
    matrix = (matrix - per_channel_mean) / per_channel_std
    return matrix


def build_feature_matrix(samples: Sequence[SequenceSample], main_sensor_channels: Sequence[str], target_length: int) -> np.ndarray:
    matrices = [resample_sample(sample, main_sensor_channels, target_length) for sample in samples]
    return np.stack([matrix.reshape(-1) for matrix in matrices], axis=0)


def _safe_channel(sample: SequenceSample, channel_map: Dict[str, int], channel: str) -> np.ndarray:
    if channel in channel_map:
        return np.nan_to_num(sample.data[channel_map[channel] + 1], nan=0.0, posinf=0.0, neginf=0.0)
    return np.zeros(sample.data.shape[1], dtype=np.float64)


def _robust_scale(values: np.ndarray) -> float:
    finite_values = np.asarray(values, dtype=np.float64)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return 1.0
    median_value = float(np.median(finite_values))
    mad_value = float(np.median(np.abs(finite_values - median_value)))
    p95_value = float(np.percentile(np.abs(finite_values), 95))
    scale = max(abs(median_value) + 6.0 * mad_value, p95_value, 1e-6)
    return scale


def _diff_threshold(values: np.ndarray) -> float:
    diff_abs = np.abs(np.diff(values))
    if diff_abs.size == 0:
        return 1.0
    median_value = float(np.median(diff_abs))
    mad_value = float(np.median(np.abs(diff_abs - median_value)))
    p90_value = float(np.percentile(diff_abs, 90))
    signal_floor = _robust_scale(values) * 0.02
    return max(median_value + 6.0 * mad_value, p90_value * 1.5, signal_floor, 1e-6)


def _max_contiguous_ratio(mask: np.ndarray) -> float:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return 0.0
    max_run = 0
    current_run = 0
    for item in mask:
        if item:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    return float(max_run / mask.size)


def _corr(values_a: np.ndarray, values_b: np.ndarray) -> float:
    if values_a.size < 2 or values_b.size < 2:
        return 0.0
    std_a = float(np.std(values_a))
    std_b = float(np.std(values_b))
    if std_a <= 1e-12 or std_b <= 1e-12:
        return 0.0
    corr_value = np.corrcoef(values_a, values_b)[0, 1]
    if not np.isfinite(corr_value):
        return 0.0
    return float(corr_value)


def _jump_features(values: np.ndarray, prefix: str, timestamps: np.ndarray) -> Dict[str, float]:
    diff_values = np.diff(values)
    diff_abs = np.abs(diff_values)
    threshold = _diff_threshold(values)
    jump_mask = diff_abs > threshold
    max_abs_diff = float(diff_abs.max()) if diff_abs.size else 0.0
    jump_count = int(jump_mask.sum())
    peak_idx = int(np.argmax(diff_abs)) if diff_abs.size else 0
    time_span = max(float(timestamps[-1] - timestamps[0]), 1e-6)
    peak_time_ratio = float((timestamps[min(peak_idx + 1, len(timestamps) - 1)] - timestamps[0]) / time_span)
    score = max_abs_diff / threshold if threshold > 0 else 0.0
    return {
        f"{prefix}_jump_score": float(score),
        f"{prefix}_diff_abs_max": max_abs_diff,
        f"{prefix}_diff_abs_p95": float(np.percentile(diff_abs, 95)) if diff_abs.size else 0.0,
        f"{prefix}_jump_count": jump_count,
        f"{prefix}_jump_ratio": float(jump_count / max(diff_abs.size, 1)),
        f"{prefix}_jump_peak_time_ratio": peak_time_ratio,
    }


def _drop_features(values: np.ndarray, prefix: str, timestamps: np.ndarray) -> Dict[str, float]:
    diff_values = np.diff(values)
    drop_values = np.maximum(-diff_values, 0.0)
    threshold = _diff_threshold(values)
    drop_mask = drop_values > threshold
    max_drop = float(drop_values.max()) if drop_values.size else 0.0
    drop_count = int(drop_mask.sum())
    peak_idx = int(np.argmax(drop_values)) if drop_values.size else 0
    time_span = max(float(timestamps[-1] - timestamps[0]), 1e-6)
    peak_time_ratio = float((timestamps[min(peak_idx + 1, len(timestamps) - 1)] - timestamps[0]) / time_span)
    return {
        f"{prefix}_drop_score": float(max_drop / threshold if threshold > 0 else 0.0),
        f"{prefix}_drop_abs_max": max_drop,
        f"{prefix}_drop_count": drop_count,
        f"{prefix}_drop_peak_time_ratio": peak_time_ratio,
        f"{prefix}_drop_flag": float(drop_count > 0),
    }


def _rise_features(values: np.ndarray, prefix: str) -> Dict[str, float]:
    diff_values = np.diff(values)
    rise_values = np.maximum(diff_values, 0.0)
    threshold = _diff_threshold(values)
    rise_mask = rise_values > threshold
    return {
        f"{prefix}_rise_score": float((rise_values.max() if rise_values.size else 0.0) / threshold if threshold > 0 else 0.0),
        f"{prefix}_end_minus_start": float(values[-1] - values[0]) if values.size else 0.0,
        f"{prefix}_rise_ratio": float(rise_mask.sum() / max(rise_mask.size, 1)),
        f"{prefix}_rise_flag": float(rise_mask.any() or (values.size > 1 and values[-1] > values[0])),
    }


def _zero_features(values: np.ndarray, prefix: str) -> Dict[str, float]:
    near_zero_threshold = max(_robust_scale(values) * 0.01, 1e-6)
    zero_mask = np.abs(values) <= near_zero_threshold
    zero_ratio = float(zero_mask.mean()) if zero_mask.size else 0.0
    return {
        f"{prefix}_zero_ratio": zero_ratio,
        f"{prefix}_zero_score": zero_ratio,
        f"{prefix}_zero_duration_max": _max_contiguous_ratio(zero_mask),
        f"{prefix}_zero_flag": float(zero_ratio > 0.05),
    }


def _pair_features(values_a: np.ndarray, values_b: np.ndarray, prefix: str) -> Dict[str, float]:
    dev_abs = np.abs(values_a - values_b)
    return {
        f"{prefix}_dev_peak_abs": float(dev_abs.max()) if dev_abs.size else 0.0,
        f"{prefix}_dev_mean_abs": float(dev_abs.mean()) if dev_abs.size else 0.0,
        f"{prefix}_corr": _corr(values_a, values_b),
    }


def _jump_mask(values: np.ndarray) -> np.ndarray:
    diff_abs = np.abs(np.diff(values))
    return diff_abs > _diff_threshold(values)


def build_fault_tree_feature_frame(
    samples: Sequence[SequenceSample],
    main_sensor_channels: Sequence[str],
) -> pd.DataFrame:
    channel_map = {channel: idx for idx, channel in enumerate(main_sensor_channels)}
    rows: List[Dict[str, object]] = []

    for sample in samples:
        timestamps = sample.data[0]
        slip_relay = _safe_channel(sample, channel_map, SLIP_RING_RELAY)
        slip_counter = _safe_channel(sample, channel_map, SLIP_RING_COUNTER)
        prox_relay = _safe_channel(sample, channel_map, PROXIMITY_RELAY)
        prox_counter = _safe_channel(sample, channel_map, PROXIMITY_COUNTER)
        generator = _safe_channel(sample, channel_map, GENERATOR_SPEED)
        rotor = _safe_channel(sample, channel_map, ROTOR_SPEED)
        converter = _safe_channel(sample, channel_map, CONVERTER_GENERATOR_SPEED)

        features: Dict[str, object] = {"sample_index": sample.sample_index}

        slip_relay_jump = _jump_features(slip_relay, "slip_ring_relay", timestamps)
        slip_counter_jump = _jump_features(slip_counter, "slip_ring_counter", timestamps)
        prox_relay_jump = _jump_features(prox_relay, "proximity_relay", timestamps)
        prox_counter_jump = _jump_features(prox_counter, "proximity_counter", timestamps)
        generator_jump = _jump_features(generator, "generator", timestamps)
        features.update(slip_relay_jump)
        features.update(slip_counter_jump)
        features.update(prox_relay_jump)
        features.update(prox_counter_jump)
        features.update(generator_jump)

        features.update(_pair_features(slip_relay, slip_counter, "slip_ring_relay_counter"))
        features.update(_pair_features(prox_relay, prox_counter, "proximity_relay_counter"))
        features.update(_pair_features(generator, rotor, "gen_rotor"))
        features.update(_pair_features(generator, converter, "gen_converter"))

        slip_ring_jump_score = max(
            float(slip_relay_jump["slip_ring_relay_jump_score"]),
            float(slip_counter_jump["slip_ring_counter_jump_score"]),
        )
        proximity_jump_score = max(
            float(prox_relay_jump["proximity_relay_jump_score"]),
            float(prox_counter_jump["proximity_counter_jump_score"]),
        )
        slip_ring_jump_count = int(slip_relay_jump["slip_ring_relay_jump_count"]) + int(
            slip_counter_jump["slip_ring_counter_jump_count"]
        )
        proximity_jump_count = int(prox_relay_jump["proximity_relay_jump_count"]) + int(
            prox_counter_jump["proximity_counter_jump_count"]
        )
        slip_ring_jump_ratio = max(
            float(slip_relay_jump["slip_ring_relay_jump_ratio"]),
            float(slip_counter_jump["slip_ring_counter_jump_ratio"]),
        )
        proximity_jump_ratio = max(
            float(prox_relay_jump["proximity_relay_jump_ratio"]),
            float(prox_counter_jump["proximity_counter_jump_ratio"]),
        )

        sensor_jump_max = max(slip_ring_jump_score, proximity_jump_score)
        sensor_stable_score = 1.0 / (1.0 + sensor_jump_max)
        features.update(
            {
                "slip_ring_jump_score": slip_ring_jump_score,
                "slip_ring_jump_count": slip_ring_jump_count,
                "slip_ring_jump_ratio": slip_ring_jump_ratio,
                "proximity_jump_score": proximity_jump_score,
                "proximity_jump_count": proximity_jump_count,
                "proximity_jump_ratio": proximity_jump_ratio,
                "slip_ring_only_jump_score": max(slip_ring_jump_score - proximity_jump_score, 0.0),
                "proximity_only_jump_score": max(proximity_jump_score - slip_ring_jump_score, 0.0),
                "A_slip_ring_jump_score": slip_ring_jump_score - proximity_jump_score,
                "B_proximity_jump_score": proximity_jump_score - slip_ring_jump_score,
                "sensor_stable_score": sensor_stable_score,
                "no_slip_ring_jump_flag": float(slip_ring_jump_score <= 1.0),
                "no_proximity_jump_flag": float(proximity_jump_score <= 1.0),
            }
        )

        generator_drop = _drop_features(generator, "generator", timestamps)
        rotor_rise = _rise_features(rotor, "rotor")
        features.update(generator_drop)
        features.update(rotor_rise)
        generator_drop_mask = np.maximum(-np.diff(generator), 0.0) > _diff_threshold(generator)
        rotor_rise_mask = np.maximum(np.diff(rotor), 0.0) > _diff_threshold(rotor)
        overlap = generator_drop_mask & rotor_rise_mask
        overlap_ratio = float(overlap.sum() / max(overlap.size, 1))
        gen_rotor_corr = float(features["gen_rotor_corr"])
        features.update(
            {
                "generator_drop_abs_max": generator_drop["generator_drop_abs_max"],
                "generator_drop_count": generator_drop["generator_drop_count"],
                "generator_drop_peak_time_ratio": generator_drop["generator_drop_peak_time_ratio"],
                "rotor_end_minus_start": rotor_rise["rotor_end_minus_start"],
                "rotor_rise_ratio": rotor_rise["rotor_rise_ratio"],
                "gen_drop_rotor_rise_overlap_ratio": overlap_ratio,
                "gen_rotor_trend_opposite_flag": float(
                    generator.size > 1 and rotor.size > 1 and (generator[-1] - generator[0]) * (rotor[-1] - rotor[0]) < 0
                ),
                "C_coupling_slip_score": float(generator_drop["generator_drop_score"])
                * float(rotor_rise["rotor_rise_score"])
                * sensor_stable_score,
            }
        )

        operation_mode = None
        if sample.auxiliary_data and OPERATION_MODE in sample.auxiliary_data:
            op_values = sample.auxiliary_data[OPERATION_MODE]
            finite_modes = op_values[np.isfinite(op_values)]
            if finite_modes.size:
                operation_mode = float(np.nanmedian(finite_modes))
        features["operation_mode_less_than_14_flag"] = float(operation_mode is not None and operation_mode < 14.0)
        features["D_generator_change_rate_score"] = float(generator_jump["generator_jump_score"]) * sensor_stable_score

        slip_jump_mask = _jump_mask(slip_relay) | _jump_mask(slip_counter)
        prox_jump_mask = _jump_mask(prox_relay) | _jump_mask(prox_counter)
        both_jump_overlap = slip_jump_mask & prox_jump_mask
        if slip_jump_mask.any() and prox_jump_mask.any():
            gap = abs(int(np.flatnonzero(slip_jump_mask)[0]) - int(np.flatnonzero(prox_jump_mask)[0]))
            gap_ratio = float(gap / max(slip_jump_mask.size, 1))
        else:
            gap_ratio = 1.0
        features.update(
            {
                "both_sensor_jump_score": min(slip_ring_jump_score, proximity_jump_score),
                "both_sensor_jump_overlap_ratio": float(both_jump_overlap.sum() / max(both_jump_overlap.size, 1)),
                "slip_ring_proximity_jump_time_gap": gap_ratio,
                "E_both_sensor_jump_score": min(slip_ring_jump_score, proximity_jump_score),
            }
        )

        converter_zero = _zero_features(converter, "converter")
        converter_drop = _drop_features(converter, "converter", timestamps)
        features.update(converter_zero)
        features.update(converter_drop)
        features["converter_zero_score"] = converter_zero["converter_zero_score"]
        features["F_converter_fault_score"] = max(
            float(features["converter_zero_score"]),
            float(converter_drop["converter_drop_score"]),
        ) * sensor_stable_score

        converter_negative_mask = converter < 0
        converter_negative_ratio = float(converter_negative_mask.mean()) if converter_negative_mask.size else 0.0
        gen_converter_corr = float(features["gen_converter_corr"])
        gen_converter_trend_same_flag = float(
            generator.size > 1 and converter.size > 1 and (generator[-1] - generator[0]) * (converter[-1] - converter[0]) >= 0
        )
        features.update(
            {
                "converter_negative_ratio": converter_negative_ratio,
                "converter_negative_duration_max": _max_contiguous_ratio(converter_negative_mask),
                "converter_negative_min": float(converter.min()) if converter.size else 0.0,
                "converter_negative_flag": float(converter_negative_ratio > 0.05),
                "gen_converter_trend_same_flag": gen_converter_trend_same_flag,
                "G_reverse_rotation_score": converter_negative_ratio * max(gen_converter_corr, 0.0),
            }
        )

        slip_relay_zero = _zero_features(slip_relay, "slip_ring_relay")
        slip_counter_zero = _zero_features(slip_counter, "slip_ring_counter")
        prox_relay_zero = _zero_features(prox_relay, "proximity_relay")
        prox_counter_zero = _zero_features(prox_counter, "proximity_counter")
        rotor_zero = _zero_features(rotor, "rotor")
        rotor_drop = _drop_features(rotor, "rotor", timestamps)
        features.update(slip_relay_zero)
        features.update(slip_counter_zero)
        features.update(prox_relay_zero)
        features.update(prox_counter_zero)
        features.update(rotor_zero)
        features.update(rotor_drop)
        slip_ring_zero_score = max(float(slip_relay_zero["slip_ring_relay_zero_ratio"]), float(slip_counter_zero["slip_ring_counter_zero_ratio"]))
        proximity_zero_score = max(float(prox_relay_zero["proximity_relay_zero_ratio"]), float(prox_counter_zero["proximity_counter_zero_ratio"]))
        sensor_zero_score = max(slip_ring_zero_score, proximity_zero_score)
        rotor_drop_to_zero_flag = float(float(rotor_zero["rotor_zero_ratio"]) > 0.05 and float(rotor_drop["rotor_drop_score"]) > 1.0)
        features.update(
            {
                "slip_ring_zero_score": slip_ring_zero_score,
                "slip_ring_zero_duration_max": max(
                    float(slip_relay_zero["slip_ring_relay_zero_duration_max"]),
                    float(slip_counter_zero["slip_ring_counter_zero_duration_max"]),
                ),
                "proximity_zero_score": proximity_zero_score,
                "proximity_zero_duration_max": max(
                    float(prox_relay_zero["proximity_relay_zero_duration_max"]),
                    float(prox_counter_zero["proximity_counter_zero_duration_max"]),
                ),
                "sensor_zero_score": sensor_zero_score,
                "rotor_drop_score": rotor_drop["rotor_drop_score"],
                "rotor_drop_abs_max": rotor_drop["rotor_drop_abs_max"],
                "rotor_drop_to_zero_flag": rotor_drop_to_zero_flag,
                "H_sensor_zero_rotor_drop_score": sensor_zero_score * float(rotor_drop["rotor_drop_score"]),
            }
        )

        features.update(
            {
                "A_flag": float(features["A_slip_ring_jump_score"] > 1.0),
                "B_flag": float(features["B_proximity_jump_score"] > 1.0),
                "C_flag": float(
                    float(generator_drop["generator_drop_flag"]) > 0
                    and float(rotor_rise["rotor_rise_flag"]) > 0
                    and features["no_slip_ring_jump_flag"]
                    and features["no_proximity_jump_flag"]
                ),
                "D_flag": float(
                    float(generator_jump["generator_jump_score"]) > 1.0
                    and features["no_slip_ring_jump_flag"]
                    and features["no_proximity_jump_flag"]
                    and (operation_mode is None or operation_mode < 14.0)
                ),
                "E_flag": float(slip_ring_jump_score > 1.0 and proximity_jump_score > 1.0),
                "F_flag": float(
                    features["no_slip_ring_jump_flag"]
                    and features["no_proximity_jump_flag"]
                    and (float(features["converter_zero_flag"]) > 0 or float(converter_drop["converter_drop_flag"]) > 0)
                ),
                "G_flag": float(float(features["converter_negative_flag"]) > 0 and gen_converter_corr > 0.5),
                "H_flag": float(sensor_zero_score > 0.05 and float(rotor_drop["rotor_drop_flag"]) > 0),
            }
        )

        rule_scores = {
            "A": float(features["A_slip_ring_jump_score"]),
            "B": float(features["B_proximity_jump_score"]),
            "C": float(features["C_coupling_slip_score"]),
            "D": float(features["D_generator_change_rate_score"]),
            "E": float(features["E_both_sensor_jump_score"]),
            "F": float(features["F_converter_fault_score"]),
            "G": float(features["G_reverse_rotation_score"]),
            "H": float(features["H_sensor_zero_rotor_drop_score"]),
        }
        candidate_label = max(rule_scores, key=rule_scores.get)
        features.update({f"rule_score_{key}": value for key, value in rule_scores.items()})
        features["rule_candidate_label"] = candidate_label
        features["rule_candidate_score"] = rule_scores[candidate_label]
        features["gen_rotor_corr"] = gen_rotor_corr
        rows.append(features)

    feature_df = pd.DataFrame(rows)
    for column in CORE_FAULT_TREE_FEATURES:
        if column not in feature_df.columns:
            feature_df[column] = 0.0
    return feature_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def build_rule_feature_matrix(rule_feature_df: pd.DataFrame) -> np.ndarray:
    return rule_feature_df[CORE_FAULT_TREE_FEATURES].to_numpy(dtype=np.float64)


def choose_cluster_count(
    feature_matrix: np.ndarray,
    cluster_range: Sequence[int],
    random_state: int,
) -> Tuple[int, pd.DataFrame]:
    rows: List[Dict[str, float]] = []

    for n_clusters in cluster_range:
        if n_clusters < 2 or n_clusters >= feature_matrix.shape[0]:
            continue

        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
        labels = model.fit_predict(feature_matrix)
        if len(np.unique(labels)) < 2:
            silhouette = np.nan
        else:
            silhouette = float(silhouette_score(feature_matrix, labels))
        rows.append(
            {
                "n_clusters": n_clusters,
                "silhouette_score": silhouette,
                "inertia": float(model.inertia_),
            }
        )

    if not rows:
        raise ValueError("无法从给定 cluster_range 中选择聚类数")

    metrics_df = pd.DataFrame(rows)
    valid_df = metrics_df.dropna(subset=["silhouette_score"])
    if valid_df.empty:
        best_n_clusters = int(metrics_df.sort_values("inertia").iloc[0]["n_clusters"])
    else:
        best_n_clusters = int(valid_df.sort_values("silhouette_score", ascending=False).iloc[0]["n_clusters"])

    return best_n_clusters, metrics_df


def cluster_samples(
    feature_matrix: np.ndarray,
    method: str,
    n_clusters: int,
    random_state: int,
) -> Tuple[np.ndarray, object]:
    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
        labels = model.fit_predict(feature_matrix)
        return labels, model

    if method == "agglomerative":
        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(feature_matrix)
        return labels, model

    raise ValueError(f"不支持的聚类方法: {method}")


def compute_cluster_center_distances(feature_matrix: np.ndarray, labels: np.ndarray) -> np.ndarray:
    distances = np.zeros(feature_matrix.shape[0], dtype=np.float64)
    for cluster_label in sorted(pd.unique(labels)):
        member_mask = labels == cluster_label
        member_matrix = feature_matrix[member_mask]
        if member_matrix.size == 0:
            continue
        center = member_matrix.mean(axis=0)
        distances[member_mask] = np.linalg.norm(member_matrix - center, axis=1)
    return distances


def save_outputs(
    output_dir: Path,
    index_df: pd.DataFrame,
    samples: Sequence[SequenceSample],
    sequence_feature_matrix: np.ndarray,
    clustering_feature_matrix: np.ndarray,
    labels: np.ndarray,
    cluster_distances: np.ndarray,
    main_sensor_channels: Sequence[str],
    target_length: int,
    selected_n_clusters: int,
    metrics_df: Optional[pd.DataFrame],
    cluster_mode: str,
    rule_feature_df: Optional[pd.DataFrame],
    representative_top_k: int,
    batch_label_threshold: float,
    expert_confirmation_path: Optional[Path],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    result_rows: List[Dict[str, object]] = []
    label_counts = pd.Series(labels).value_counts().sort_index().to_dict()

    rule_by_sample = None
    if rule_feature_df is not None:
        rule_by_sample = rule_feature_df.set_index("sample_index", drop=False)

    for row_position, (sample, cluster_label) in enumerate(zip(samples, labels)):
        source_row = index_df.iloc[sample.sample_index].to_dict()
        source_row.update(
            {
                "sample_index": int(sample.sample_index),
                "cluster_label": int(cluster_label),
                "cluster_center_distance": float(cluster_distances[row_position]),
                "window_row_count": int(sample.row_count),
                "window_time_start": float(sample.time_start),
                "window_time_end": float(sample.time_end),
                "resampled_length": int(target_length),
            }
        )
        if rule_by_sample is not None and sample.sample_index in rule_by_sample.index:
            rule_row = rule_by_sample.loc[sample.sample_index]
            source_row["rule_candidate_label"] = rule_row["rule_candidate_label"]
            source_row["rule_candidate_score"] = float(rule_row["rule_candidate_score"])
        result_rows.append(source_row)

    result_df = pd.DataFrame(result_rows)
    result_df.to_csv(output_dir / "cluster_assignments.csv", index=False, encoding="utf-8-sig")

    cluster_summary = (
        result_df.groupby("cluster_label")
        .agg(sample_count=("cluster_label", "size"))
        .reset_index()
        .sort_values("cluster_label")
    )
    cluster_summary["sample_ratio"] = cluster_summary["sample_count"] / cluster_summary["sample_count"].sum()
    cluster_summary.to_csv(output_dir / "cluster_summary.csv", index=False, encoding="utf-8-sig")

    if metrics_df is not None:
        metrics_df.to_csv(output_dir / "cluster_metrics.csv", index=False, encoding="utf-8-sig")

    representative_rows = []
    for cluster_label in sorted(result_df["cluster_label"].dropna().unique()):
        cluster_rows = result_df[result_df["cluster_label"] == cluster_label].sort_values("cluster_center_distance")
        for rank, (_, row) in enumerate(cluster_rows.head(representative_top_k).iterrows(), start=1):
            representative_row = row.to_dict()
            representative_row["representative_rank"] = rank
            representative_rows.append(representative_row)
    representative_df = pd.DataFrame(representative_rows)
    if not representative_df.empty:
        leading_columns = [
            "cluster_label",
            "representative_rank",
            "cluster_center_distance",
            "rule_candidate_label",
            "rule_candidate_score",
        ]
        ordered_columns = [column for column in leading_columns if column in representative_df.columns]
        ordered_columns.extend([column for column in representative_df.columns if column not in ordered_columns])
        representative_df = representative_df.reindex(columns=ordered_columns)
    representative_df.to_csv(output_dir / "cluster_representative_samples.csv", index=False, encoding="utf-8-sig")

    np.save(output_dir / "clustering_features.npy", clustering_feature_matrix)

    if rule_feature_df is not None:
        rule_output_df = rule_feature_df.copy()
        label_df = pd.DataFrame(
            {
                "sample_index": [sample.sample_index for sample in samples],
                "cluster_label": labels,
                "cluster_center_distance": cluster_distances,
            }
        )
        rule_output_df = rule_output_df.merge(label_df, on="sample_index", how="left")
        leading_columns = ["sample_index", "cluster_label", "cluster_center_distance", "rule_candidate_label", "rule_candidate_score"]
        ordered_columns = [column for column in leading_columns if column in rule_output_df.columns]
        ordered_columns.extend([column for column in rule_output_df.columns if column not in ordered_columns])
        rule_output_df = rule_output_df.reindex(columns=ordered_columns)
        rule_output_df.to_csv(output_dir / "fault_tree_features.csv", index=False, encoding="utf-8-sig")

        summary_columns = [
            column
            for column in CORE_FAULT_TREE_FEATURES
            if column in rule_output_df.columns and pd.api.types.is_numeric_dtype(rule_output_df[column])
        ]
        cluster_feature_summary = (
            rule_output_df.groupby("cluster_label")[summary_columns]
            .median()
            .reset_index()
            .sort_values("cluster_label")
        )
        cluster_feature_summary.to_csv(
            output_dir / "cluster_fault_tree_feature_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

        rule_crosstab = pd.crosstab(
            rule_output_df["cluster_label"],
            rule_output_df["rule_candidate_label"],
            normalize="index",
        )
        rule_crosstab.to_csv(output_dir / "cluster_rule_candidate_crosstab.csv", encoding="utf-8-sig")

        group_rows = []
        review_rows = []
        cluster_sizes = rule_output_df.groupby("cluster_label").size().to_dict()
        group_counts = (
            rule_output_df.groupby(["cluster_label", "rule_candidate_label"])
            .agg(
                sample_count=("sample_index", "size"),
                median_cluster_center_distance=("cluster_center_distance", "median"),
                min_cluster_center_distance=("cluster_center_distance", "min"),
                median_rule_candidate_score=("rule_candidate_score", "median"),
            )
            .reset_index()
        )
        nearest_by_group = (
            rule_output_df.sort_values("cluster_center_distance")
            .groupby(["cluster_label", "rule_candidate_label"], as_index=False)
            .first()[["cluster_label", "rule_candidate_label", "sample_index", "cluster_center_distance"]]
            .rename(
                columns={
                    "sample_index": "nearest_sample_index_in_group",
                    "cluster_center_distance": "nearest_distance_in_group",
                }
            )
        )
        group_counts = group_counts.merge(nearest_by_group, on=["cluster_label", "rule_candidate_label"], how="left")
        for _, row in group_counts.iterrows():
            cluster_label = row["cluster_label"]
            sample_count = int(row["sample_count"])
            cluster_size = int(cluster_sizes.get(cluster_label, sample_count))
            group_ratio = float(sample_count / max(cluster_size, 1))
            group_row = row.to_dict()
            group_row["cluster_size"] = cluster_size
            group_row["group_ratio_in_cluster"] = group_ratio
            group_row["batch_label_candidate_flag"] = float(group_ratio >= batch_label_threshold)
            group_rows.append(group_row)
        group_summary_df = pd.DataFrame(group_rows).sort_values(
            ["cluster_label", "sample_count"], ascending=[True, False]
        )
        group_summary_df.to_csv(output_dir / "cluster_rule_group_summary.csv", index=False, encoding="utf-8-sig")

        group_representative_rows = []
        group_summary_lookup = group_summary_df.set_index(["cluster_label", "rule_candidate_label"])
        labeled_result_df = result_df.dropna(subset=["rule_candidate_label"]).copy()
        for (cluster_label, rule_label), group_df in labeled_result_df.groupby(["cluster_label", "rule_candidate_label"]):
            sorted_group_df = group_df.sort_values("cluster_center_distance")
            group_key = (cluster_label, rule_label)
            group_sample_count = np.nan
            group_ratio = np.nan
            batch_candidate_flag = np.nan
            if group_key in group_summary_lookup.index:
                group_summary_row = group_summary_lookup.loc[group_key]
                group_sample_count = group_summary_row.get("sample_count", np.nan)
                group_ratio = group_summary_row.get("group_ratio_in_cluster", np.nan)
                batch_candidate_flag = group_summary_row.get("batch_label_candidate_flag", np.nan)
            for rank, (_, row) in enumerate(sorted_group_df.head(representative_top_k).iterrows(), start=1):
                representative_row = row.to_dict()
                representative_row["rule_group_key"] = f"{cluster_label}_{rule_label}"
                representative_row["representative_rank_in_rule_group"] = rank
                representative_row["rule_group_sample_count"] = group_sample_count
                representative_row["rule_group_ratio_in_cluster"] = group_ratio
                representative_row["batch_label_candidate_flag"] = batch_candidate_flag
                group_representative_rows.append(representative_row)
        group_representative_df = pd.DataFrame(group_representative_rows)
        if not group_representative_df.empty:
            leading_columns = [
                "rule_group_key",
                "cluster_label",
                "rule_candidate_label",
                "representative_rank_in_rule_group",
                "cluster_center_distance",
                "rule_candidate_score",
                "rule_group_sample_count",
                "rule_group_ratio_in_cluster",
                "batch_label_candidate_flag",
            ]
            ordered_columns = [column for column in leading_columns if column in group_representative_df.columns]
            ordered_columns.extend([column for column in group_representative_df.columns if column not in ordered_columns])
            group_representative_df = group_representative_df.reindex(columns=ordered_columns)
        group_representative_df.to_csv(
            output_dir / "cluster_rule_group_representative_samples.csv",
            index=False,
            encoding="utf-8-sig",
        )

        representative_first = representative_df.sort_values("representative_rank").groupby("cluster_label", as_index=False).first()
        dominant = group_counts.sort_values(["cluster_label", "sample_count"], ascending=[True, False]).groupby(
            "cluster_label", as_index=False
        ).first()
        for _, row in dominant.iterrows():
            cluster_label = row["cluster_label"]
            cluster_size = int(cluster_sizes.get(cluster_label, row["sample_count"]))
            dominant_label = row["rule_candidate_label"]
            dominant_ratio = float(row["sample_count"] / max(cluster_size, 1))
            rep_match = representative_first[representative_first["cluster_label"] == cluster_label]
            representative_label = None
            representative_sample_index = None
            representative_distance = np.nan
            if not rep_match.empty:
                rep_row = rep_match.iloc[0]
                representative_label = rep_row.get("rule_candidate_label")
                representative_sample_index = rep_row.get("sample_index")
                representative_distance = rep_row.get("cluster_center_distance")
            can_batch = dominant_ratio >= batch_label_threshold and representative_label == dominant_label
            review_rows.append(
                {
                    "cluster_label": cluster_label,
                    "cluster_size": cluster_size,
                    "representative_sample_index": representative_sample_index,
                    "representative_rule_candidate_label": representative_label,
                    "representative_distance": representative_distance,
                    "dominant_rule_candidate_label": dominant_label,
                    "dominant_rule_ratio": dominant_ratio,
                    "suggested_action": "expert_confirm_then_batch_label" if can_batch else "manual_review_or_split",
                    "batch_group_key": f"{cluster_label}_{dominant_label}" if can_batch else "",
                }
            )
        review_plan_df = pd.DataFrame(review_rows).sort_values("cluster_label")
        review_plan_df.to_csv(
            output_dir / "cluster_expert_review_plan.csv",
            index=False,
            encoding="utf-8-sig",
        )

        confirmation_template_df = group_summary_df.copy()
        confirmation_template_df["rule_group_key"] = (
            confirmation_template_df["cluster_label"].astype(str)
            + "_"
            + confirmation_template_df["rule_candidate_label"].astype(str)
        )

        path_candidate_columns = [
            column
            for column in group_representative_df.columns
            if "path" in str(column).lower() or "tracelog" in str(column).lower()
        ]
        preferred_path_columns = [
            "tr_path",
            "trace_path",
            "tracelog_path",
            "missing_file_path",
        ]
        representative_path_column = next(
            (column for column in preferred_path_columns if column in group_representative_df.columns),
            path_candidate_columns[0] if path_candidate_columns else None,
        )
        representative_lookup_rows = []
        if not group_representative_df.empty:
            sorted_group_representative_df = group_representative_df.sort_values(
                ["rule_group_key", "representative_rank_in_rule_group"]
            )
            for rule_group_key, group_df in sorted_group_representative_df.groupby("rule_group_key"):
                lookup_row = {"rule_group_key": rule_group_key}
                sample_indexes = []
                sample_paths = []
                for _, representative_row in group_df.head(representative_top_k).iterrows():
                    rank = int(representative_row["representative_rank_in_rule_group"])
                    sample_index = representative_row.get("sample_index", "")
                    sample_path = representative_row.get(representative_path_column, "") if representative_path_column else ""
                    lookup_row[f"representative_{rank}_sample_index"] = sample_index
                    lookup_row[f"representative_{rank}_path"] = sample_path
                    if pd.notna(sample_index):
                        sample_indexes.append(str(sample_index))
                    if pd.notna(sample_path) and str(sample_path):
                        sample_paths.append(str(sample_path))
                lookup_row["representative_sample_indexes"] = ";".join(sample_indexes)
                lookup_row["representative_paths"] = ";".join(sample_paths)
                representative_lookup_rows.append(lookup_row)
        if representative_lookup_rows:
            representative_lookup_df = pd.DataFrame(representative_lookup_rows)
            confirmation_template_df = confirmation_template_df.merge(
                representative_lookup_df,
                on="rule_group_key",
                how="left",
            )

        confirmation_template_df["expert_decision"] = ""
        confirmation_template_df["expert_label"] = ""
        confirmation_template_df["expert_comment"] = ""
        template_columns = [
            "rule_group_key",
            "cluster_label",
            "rule_candidate_label",
            "sample_count",
            "group_ratio_in_cluster",
            "representative_paths",
            "representative_sample_indexes",
        ]
        for rank in range(1, representative_top_k + 1):
            template_columns.extend(
                [
                    f"representative_{rank}_path",
                    f"representative_{rank}_sample_index",
                ]
            )
        template_columns.extend(
            [
                "expert_decision",
                "expert_label",
                "expert_comment",
            ]
        )
        template_columns = [column for column in template_columns if column in confirmation_template_df.columns]
        confirmation_template_df = confirmation_template_df.reindex(columns=template_columns)
        confirmation_template_df.to_csv(
            output_dir / "cluster_expert_confirmation_template.csv",
            index=False,
            encoding="utf-8-sig",
        )

        if expert_confirmation_path is not None:
            apply_expert_confirmations(
                output_dir=output_dir,
                result_df=result_df,
                rule_output_df=rule_output_df,
                confirmation_path=expert_confirmation_path,
            )

    np.savez_compressed(
        output_dir / "preprocessed_sequences.npz",
        X=sequence_feature_matrix.reshape(sequence_feature_matrix.shape[0], len(main_sensor_channels), target_length),
        cluster_labels=labels,
        channels=np.array(main_sensor_channels, dtype=object),
    )

    metadata = {
        "sample_count": len(samples),
        "run_timestamp": output_dir.name,
        "cluster_mode": cluster_mode,
        "channel_count": len(main_sensor_channels),
        "channels": list(main_sensor_channels),
        "target_length": target_length,
        "selected_n_clusters": selected_n_clusters,
        "representative_top_k": representative_top_k,
        "batch_label_threshold": batch_label_threshold,
        "expert_confirmation_path": str(expert_confirmation_path) if expert_confirmation_path is not None else None,
        "clustering_feature_dim": int(clustering_feature_matrix.shape[1]),
        "core_fault_tree_features": CORE_FAULT_TREE_FEATURES,
        "cluster_sizes": {str(key): int(value) for key, value in label_counts.items()},
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _read_confirmation_table(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def apply_expert_confirmations(
    output_dir: Path,
    result_df: pd.DataFrame,
    rule_output_df: pd.DataFrame,
    confirmation_path: Path,
) -> None:
    if not confirmation_path.exists():
        raise FileNotFoundError(f"专家确认表不存在: {confirmation_path}")

    confirmation_df = _read_confirmation_table(confirmation_path)
    required_columns = {"rule_group_key", "expert_decision", "expert_label"}
    missing_columns = required_columns - set(confirmation_df.columns)
    if missing_columns:
        raise ValueError(f"专家确认表缺少列: {sorted(missing_columns)}")

    confirmation_df = confirmation_df.copy()
    confirmation_df["rule_group_key"] = confirmation_df["rule_group_key"].astype(str).str.strip()
    confirmation_df["expert_decision"] = confirmation_df["expert_decision"].astype(str).str.strip()
    confirmation_df["expert_label"] = confirmation_df["expert_label"].astype(str).str.strip()
    confirmation_df = confirmation_df[confirmation_df["rule_group_key"] != ""]
    confirmation_df = confirmation_df.drop_duplicates(subset=["rule_group_key"], keep="last")

    labeled_df = result_df.copy()
    labeled_df["rule_group_key"] = labeled_df["cluster_label"].astype(str) + "_" + labeled_df["rule_candidate_label"].astype(str)
    keep_columns = ["rule_group_key", "expert_decision", "expert_label"]
    optional_columns = ["expert_comment"]
    keep_columns.extend([column for column in optional_columns if column in confirmation_df.columns])
    labeled_df = labeled_df.merge(confirmation_df[keep_columns], on="rule_group_key", how="left")

    batch_accept_mask = labeled_df["expert_decision"].eq("batch_accept") & labeled_df["expert_label"].notna() & labeled_df["expert_label"].ne("")
    labeled_df["final_label"] = ""
    labeled_df.loc[batch_accept_mask, "final_label"] = labeled_df.loc[batch_accept_mask, "expert_label"]
    labeled_df["label_source"] = np.where(batch_accept_mask, "expert_batch_rule_group", "needs_review")

    labeled_df.to_csv(output_dir / "expert_labeled_samples.csv", index=False, encoding="utf-8-sig")
    labeled_df[~batch_accept_mask].to_csv(output_dir / "expert_review_samples.csv", index=False, encoding="utf-8-sig")

    rule_labeled_df = rule_output_df.copy()
    rule_labeled_df["rule_group_key"] = rule_labeled_df["cluster_label"].astype(str) + "_" + rule_labeled_df["rule_candidate_label"].astype(str)
    rule_labeled_df = rule_labeled_df.merge(confirmation_df[keep_columns], on="rule_group_key", how="left")
    rule_batch_accept_mask = rule_labeled_df["expert_decision"].eq("batch_accept") & rule_labeled_df["expert_label"].notna() & rule_labeled_df["expert_label"].ne("")
    rule_labeled_df["final_label"] = ""
    rule_labeled_df.loc[rule_batch_accept_mask, "final_label"] = rule_labeled_df.loc[rule_batch_accept_mask, "expert_label"]
    rule_labeled_df["label_source"] = np.where(rule_batch_accept_mask, "expert_batch_rule_group", "needs_review")
    rule_labeled_df.to_csv(output_dir / "fault_tree_features_with_expert_labels.csv", index=False, encoding="utf-8-sig")

    summary_df = (
        labeled_df.groupby(["rule_group_key", "expert_decision", "expert_label", "label_source"], dropna=False)
        .agg(sample_count=("rule_group_key", "size"))
        .reset_index()
        .sort_values(["label_source", "sample_count"], ascending=[True, False])
    )
    summary_df.to_csv(output_dir / "expert_label_application_summary.csv", index=False, encoding="utf-8-sig")


def save_failed_outputs(
    output_dir: Path,
    index_df: pd.DataFrame,
    failed_rows: Sequence[Dict[str, object]],
    trace_path_column: str,
) -> None:
    if not failed_rows:
        return

    failed_df = pd.DataFrame(failed_rows)
    failed_df.to_csv(output_dir / "failed_samples.csv", index=False, encoding="utf-8-sig")

    missing_file_df = failed_df[failed_df["error_type"] == "file_not_found"].copy()
    if missing_file_df.empty:
        return

    source_rows = index_df.iloc[missing_file_df["sample_index"].astype(int)].copy().reset_index(drop=True)
    source_rows["missing_file_path"] = missing_file_df["tracelog_path"].astype(str).to_list()
    source_rows["missing_file_error"] = missing_file_df["error"].astype(str).to_list()

    ordered_columns = list(source_rows.columns)
    if trace_path_column in ordered_columns:
        ordered_columns.remove(trace_path_column)
        ordered_columns = [trace_path_column, *ordered_columns]
    source_rows = source_rows.reindex(columns=ordered_columns)
    source_rows.to_excel(output_dir / "missing_tracelog_rows.xlsx", index=False)


def parse_cluster_range(value: str) -> List[int]:
    value = value.strip()
    if "-" in value:
        start_str, end_str = value.split("-", 1)
        start = int(start_str)
        end = int(end_str)
        if start > end:
            raise ValueError("cluster_range 起始值不能大于结束值")
        return list(range(start, end + 1))
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def create_timestamped_output_dir(base_output_dir: Path) -> Tuple[Path, str]:
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_output_dir / run_timestamp
    suffix = 1
    while output_dir.exists():
        output_dir = base_output_dir / f"{run_timestamp}_{suffix:03d}"
        suffix += 1
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir, output_dir.name


def main() -> None:
    parser = argparse.ArgumentParser(description="基于 tracelog 预处理结果的直接聚类脚本")
    parser.add_argument(
        "--config",
        default="configs/preprocess_2D.yaml",
        help="预处理配置文件路径",
    )
    parser.add_argument(
        "--excel-path",
        required=True,
        help="样本索引 Excel 路径",
    )
    parser.add_argument(
        "--trace-path-column",
        default="tracelog_path",
        help="Excel 中 tracelog 路径列名",
    )
    parser.add_argument(
        "--output-dir",
        default="tracelog_clustering/output",
        help="输出基础目录；每次运行会自动创建 YYYYMMDD_HHMMSS 子目录",
    )
    parser.add_argument(
        "--cluster-method",
        choices=["kmeans", "agglomerative"],
        default="kmeans",
        help="聚类方法",
    )
    parser.add_argument(
        "--cluster-mode",
        choices=["1", "2", "sequence_explain", "fault_tree"],
        default="1",
        help="1/sequence_explain=原始时序聚类+故障树解释；2/fault_tree=核心故障树特征聚类",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=None,
        help="指定聚类簇数；不传时按 silhouette 自动选择",
    )
    parser.add_argument(
        "--cluster-range",
        default="2-8",
        help="自动选簇时搜索范围，如 2-8 或 2,3,4,5",
    )
    parser.add_argument(
        "--target-length",
        type=int,
        default=None,
        help="重采样后的统一序列长度；不传时取窗口后样本长度中位数",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="仅调试时限制样本数",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="随机种子",
    )
    parser.add_argument(
        "--representative-top-k",
        type=int,
        default=5,
        help="每个簇导出的最近中心典型样本数量",
    )
    parser.add_argument(
        "--batch-label-threshold",
        type=float,
        default=0.6,
        help="cluster+规则候选组合可批量确认的簇内占比阈值",
    )
    parser.add_argument(
        "--expert-confirmation-path",
        default=None,
        help="专家填写后的 cluster_expert_confirmation_template.csv；传入后批量生成最终标签和复核样本",
    )
    args = parser.parse_args()

    setup_logging()

    script_path = Path(__file__)
    project_root = resolve_project_root(script_path)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (project_root / config_path).resolve()

    cfg = load_yaml_config(config_path, project_root)

    mapping_file = Path(cfg["column_mapping_file"])
    if not mapping_file.is_absolute():
        mapping_file = (project_root / mapping_file).resolve()
    column_mapping = load_column_mapping(mapping_file)

    main_sensor_channels = cfg["channels"]["main_sensor_channels"]
    timestamp_window = tuple(cfg["constants"]["timestamp_window"])
    default_fault_time = cfg["constants"].get("default_fault_time", 0)
    data_processing_cfg = cfg.get("data_processing", {})

    preprocessor = TracelogWindowPreprocessor(
        column_mapping=column_mapping,
        main_sensor_channels=main_sensor_channels,
        timestamp_window=(float(timestamp_window[0]), float(timestamp_window[1])),
        default_fault_time=float(default_fault_time),
        min_rows_after_window=int(data_processing_cfg.get("min_rows_after_window", 5)),
        time_offset_threshold=float(data_processing_cfg.get("time_offset_threshold", 10000)),
        missing_channel_strategy=str(data_processing_cfg.get("missing_channel_strategy", "zero")),
    )

    excel_path = Path(args.excel_path)
    index_df = pd.read_excel(excel_path)
    if args.trace_path_column not in index_df.columns:
        raise KeyError(f"Excel 中不存在列: {args.trace_path_column}")

    index_df = index_df.dropna(subset=[args.trace_path_column]).reset_index(drop=True)
    if args.sample_limit is not None:
        index_df = index_df.iloc[: args.sample_limit].copy()

    valid_samples: List[SequenceSample] = []
    failed_rows: List[Dict[str, object]] = []

    for idx, tracelog_path in enumerate(index_df[args.trace_path_column].astype(str)):
        try:
            sample = preprocessor.preprocess(tracelog_path=tracelog_path, sample_index=idx)
            valid_samples.append(sample)
        except Exception as exc:
            error_type = "file_not_found" if isinstance(exc, FileNotFoundError) else "preprocess_error"
            failed_rows.append(
                {
                    "sample_index": idx,
                    "tracelog_path": tracelog_path,
                    "error_type": error_type,
                    "error": str(exc),
                }
            )
            LOGGER.warning("跳过样本 %s: %s", tracelog_path, exc)

    if not valid_samples:
        raise RuntimeError("没有可用于聚类的有效样本")

    target_length = args.target_length or infer_target_length(valid_samples)
    sequence_feature_matrix = build_feature_matrix(valid_samples, main_sensor_channels, target_length)
    rule_feature_df = build_fault_tree_feature_frame(valid_samples, main_sensor_channels)
    rule_feature_matrix = build_rule_feature_matrix(rule_feature_df)

    cluster_mode = "fault_tree" if args.cluster_mode in {"2", "fault_tree"} else "sequence_explain"
    raw_clustering_feature_matrix = rule_feature_matrix if cluster_mode == "fault_tree" else sequence_feature_matrix

    scaler = StandardScaler()
    feature_matrix = scaler.fit_transform(raw_clustering_feature_matrix)

    metrics_df: Optional[pd.DataFrame] = None
    selected_n_clusters = args.n_clusters
    if selected_n_clusters is None:
        selected_n_clusters, metrics_df = choose_cluster_count(
            feature_matrix=feature_matrix,
            cluster_range=parse_cluster_range(args.cluster_range),
            random_state=args.random_state,
        )

    labels, _ = cluster_samples(
        feature_matrix=feature_matrix,
        method=args.cluster_method,
        n_clusters=selected_n_clusters,
        random_state=args.random_state,
    )
    cluster_distances = compute_cluster_center_distances(feature_matrix, labels)

    base_output_dir = Path(args.output_dir)
    if not base_output_dir.is_absolute():
        base_output_dir = (project_root / base_output_dir).resolve()
    output_dir, run_timestamp = create_timestamped_output_dir(base_output_dir)
    expert_confirmation_path = Path(args.expert_confirmation_path).resolve() if args.expert_confirmation_path else None

    save_outputs(
        output_dir=output_dir,
        index_df=index_df,
        samples=valid_samples,
        sequence_feature_matrix=sequence_feature_matrix,
        clustering_feature_matrix=raw_clustering_feature_matrix,
        labels=labels,
        cluster_distances=cluster_distances,
        main_sensor_channels=main_sensor_channels,
        target_length=target_length,
        selected_n_clusters=selected_n_clusters,
        metrics_df=metrics_df,
        cluster_mode=cluster_mode,
        rule_feature_df=rule_feature_df,
        representative_top_k=max(int(args.representative_top_k), 1),
        batch_label_threshold=float(args.batch_label_threshold),
        expert_confirmation_path=expert_confirmation_path,
    )

    save_failed_outputs(
        output_dir=output_dir,
        index_df=index_df,
        failed_rows=failed_rows,
        trace_path_column=args.trace_path_column,
    )

    LOGGER.info("聚类完成")
    LOGGER.info("聚类模式: %s", cluster_mode)
    LOGGER.info("有效样本数: %s", len(valid_samples))
    LOGGER.info("跳过样本数: %s", len(failed_rows))
    LOGGER.info("聚类簇数: %s", selected_n_clusters)
    LOGGER.info("输出目录: %s", output_dir)


if __name__ == "__main__":
    main()
