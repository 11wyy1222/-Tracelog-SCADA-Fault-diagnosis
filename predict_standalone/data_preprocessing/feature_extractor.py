import numpy as np
from scipy import signal
from scipy.stats import kurtosis, skew
from scipy.signal import find_peaks, hilbert
import warnings
import logging
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=RuntimeWarning)
# === 1. 引入外部配置 (确保路径正确) ===
from configs.config_loader import (
    DEFAULT_FAULT_TIME,          # 故障时间戳
    MULTI_ANALYSIS_CHANNELS      # 需要多通道同步的通道
)
from error_codes import ErrorCode

# === 2. 默认特征配置字典 (集成外部常量) ===
DEFAULT_FEATURE_CONFIG = {
    'general': {
        'fault_timestamp': DEFAULT_FAULT_TIME,
        'multi_channels': MULTI_ANALYSIS_CHANNELS
    },
    'stat': {
        'epsilon': 1e-8,
        'stability_segments': 4,
        'mutation_std_ratio': 1.5,
        'min_len_mutation': 10
    },
    'dynamic': {
        'window_pre_fault': 2000,
        'window_transient': 500,
        'window_post_fault': 2000,
        'response_thresh_ratio': 0.5,
        'rise_rate_idx_min': 10
    },
    'freq': {
        'bands': [(0, 0.1), (0.1, 0.3), (0.3, 0.5)],
        'high_freq_filter': {'order': 3, 'cutoff': 0.1, 'type': 'high'}
    },
    'period': {
        'peak_height_std_ratio': 0.5,
        'business_dist': 10,
        'min_len': 50
    },
    'business': {
        'jump_window': 5,
        'jump_ratio': 2.0,
        'sync_tolerance': 10,
        'fluctuation_w_ratio': 0.1,
        'pattern_thresh_high': 0.7,
        'pattern_thresh_low': 0.3
    }
}


class AdvancedFeatureExtractor:
    """
    增强的特征提取器
    """

    def __init__(self, config=None):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.config = config or DEFAULT_FEATURE_CONFIG

        self.fault_time_target = self.config['general']['fault_timestamp']
        self.default_channels = self.config['general']['multi_channels']
        self.eps = float(self.config['stat']['epsilon'])
        self.logger = logger
        self.logger.info("[INIT] AdvancedFeatureExtractor 初始化完成")
        try:
            f_cfg = self.config['freq']['high_freq_filter']
            self.b_high, self.a_high = signal.butter(
                f_cfg['order'], f_cfg['cutoff'], f_cfg['type']
            )
        except Exception as e:
            logger.error(f"滤波器系数初始化失败: {e}")
            self.b_high, self.a_high = None, None

        self.logger.debug("特征提取器初始化完成")

    # --------------------------------------------------------------------
    # 通用安全统计计算
    # --------------------------------------------------------------------
    def safe_statistical_calculation(self, values, calculation_func, default=0.0):
        if len(values) == 0 or np.all(np.isnan(values)):
            return default
        try:
            valid_vals = values[~np.isnan(values)]
            if len(valid_vals) == 0:
                return default
            result = calculation_func(valid_vals)
            return result if np.isfinite(result) else default
        except Exception:
            self.logger.debug(f"统计计算异常 [{calculation_func.__name__}]")
            return default

    # --------------------------------------------------------------------
    # 1. 基础统计特征
    # --------------------------------------------------------------------
    def extract_basic_statistical_features(self, sensor_data, channel):
        if channel not in sensor_data.columns:
            self.logger.error(f"[统计特征] [{ErrorCode.CHANNEL_MISSING.name}({ErrorCode.CHANNEL_MISSING.value})] 通道缺失: {channel}")
            return np.zeros(20)

        vals = sensor_data[channel].to_numpy()
        valid_vals = vals[~np.isnan(vals)]

        if valid_vals.size == 0:
            self.logger.warning(f"[统计特征] 通道 {channel} 数据全为空，填充零向量")
            return np.zeros(20)

        mean_val = np.mean(valid_vals)
        std_val = np.std(valid_vals)
        max_val = np.max(valid_vals)
        min_val = np.min(valid_vals)
        median_val = np.median(valid_vals)
        rms_val = np.sqrt(np.mean(valid_vals ** 2))
        value_range = max_val - min_val
        variance_val = np.var(valid_vals)

        if std_val < 1e-9:
            kurt_val = 0.0
            skewness_val = 0.0
        else:
            try:
                k_val = kurtosis(valid_vals)
                s_val = skew(valid_vals)
                kurt_val = k_val if np.isfinite(k_val) else 0.0
                skewness_val = s_val if np.isfinite(s_val) else 0.0
            except:
                kurt_val, skewness_val = 0.0, 0.0

        diff_vals = np.diff(valid_vals)
        if diff_vals.size > 0:
            mean_change = np.mean(np.abs(diff_vals))
            max_change = np.max(np.abs(diff_vals))
            diff_std = np.std(diff_vals)
            change_rate = mean_change / (np.abs(mean_val) + self.eps)

            diff2_vals = np.diff(diff_vals)
            if diff2_vals.size > 0:
                diff2_std = np.std(diff2_vals)
                diff2_max = np.max(np.abs(diff2_vals))
            else:
                diff2_std, diff2_max = 0.0, 0.0
        else:
            mean_change = max_change = diff_std = change_rate = 0.0
            diff2_std = diff2_max = 0.0

        try:
            stability_metric = self._calculate_stability_metric(valid_vals)
        except:
            stability_metric = 0.0

        try:
            mutation_features = self._detect_mutation_points(valid_vals)
        except:
            mutation_features = {'mutation_count': 0, 'mutation_intensity': 0}

        zero_crossings = np.where(np.diff(np.signbit(valid_vals)))[0]
        zero_crossing_rate = len(zero_crossings) / len(valid_vals) if len(valid_vals) > 0 else 0.0

        return np.array([
            mean_val, std_val, max_val, min_val, median_val,
            kurt_val, skewness_val, mean_change, max_change,
            rms_val, value_range, variance_val,
            change_rate, diff_std, diff2_std, diff2_max,
            stability_metric, zero_crossing_rate,
            mutation_features.get('mutation_count', 0),
            mutation_features.get('mutation_intensity', 0)
        ])

    def _calculate_stability_metric(self, values):
        n_segs = self.config['stat']['stability_segments']
        if len(values) < n_segs:
            return 0.0
        seg_size = len(values) // n_segs
        seg_stds = []
        for i in range(n_segs):
            start = i * seg_size
            end = start + seg_size if i < (n_segs - 1) else len(values)
            seg = values[start:end]
            if seg.size:
                seg_stds.append(np.std(seg))
        return np.std(seg_stds) / (np.mean(seg_stds) + self.eps) if seg_stds else 0.0

    def _detect_mutation_points(self, values):
        min_len = self.config['stat']['min_len_mutation']
        ratio = self.config['stat']['mutation_std_ratio']
        if len(values) < min_len:
            return {'mutation_count': 0, 'mutation_intensity': 0}
        diff_vals = np.diff(values)
        threshold = np.std(values) * ratio
        mutation_mask = np.abs(diff_vals) > threshold
        mutation_count = np.sum(mutation_mask)
        mutation_intensity = np.mean(np.abs(diff_vals[mutation_mask])) if mutation_count > 0 else 0
        return {'mutation_count': mutation_count, 'mutation_intensity': mutation_intensity}

    def _calculate_zero_crossing_rate(self, values):
        return 0.0

    # --------------------------------------------------------------------
    # 2. 动态响应特征
    # --------------------------------------------------------------------
    def extract_dynamic_features(self, sensor_data, channel):
        if channel not in sensor_data.columns or 'timestamp' not in sensor_data.columns:
            self.logger.warning(f"[动态特征] 缺少通道 {channel} 或 timestamp 列")
            return np.zeros(25)

        vals = sensor_data[channel].to_numpy()
        timestamps = sensor_data['timestamp'].to_numpy()

        if vals.size < 10:
            self.logger.debug(f"[动态特征] 数据点严重不足 ({vals.size}): {channel}")
            return np.zeros(25)

        cfg = self.config['dynamic']
        total = vals.size

        time_diff = np.abs(timestamps - self.fault_time_target)
        fault_idx = int(np.argmin(time_diff))

        pre_start = max(0, fault_idx - cfg['window_pre_fault'])
        pre_end = fault_idx
        tran_start = fault_idx
        tran_end = min(total, fault_idx + cfg['window_transient'])
        post_start = tran_end
        post_end = min(total, fault_idx + cfg['window_post_fault'])

        baseline = vals[pre_start:pre_end]
        transition = vals[tran_start:tran_end]
        steady = vals[post_start:post_end]

        feat = []

        # ---------- 过渡区特征 ----------
        if transition.size > 0:
            peak = np.nanmax(transition)
            valley = np.nanmin(transition)
            peak_to_valley = peak - valley

            base_std = np.nanstd(baseline) if baseline.size > 0 else 1e-8
            base_mean = np.nanmean(baseline) if baseline.size > 0 else 0

            overshoot = (peak - base_mean) / (base_std + self.eps)

            thresh = base_mean + cfg['response_thresh_ratio'] * (peak - base_mean)
            above = transition > thresh
            resp_time = np.argmax(above) if np.any(above) else transition.size

            peak_idx = np.argmax(transition)
            if peak_idx > cfg['rise_rate_idx_min'] and peak_idx < transition.size:
                rise_rate = (transition[peak_idx] - transition[0]) / peak_idx
            else:
                rise_rate = 0

            if transition.size >= 2:
                grad = np.gradient(transition)
                grad_mean = np.nanmean(grad)
                grad_std = np.nanstd(grad)
            else:
                grad_mean = 0.0
                grad_std = 0.0

            try:
                if transition.size >= 10:
                    analytic = hilbert(transition)
                    envelope = np.abs(analytic)
                    env_mean, env_std = np.mean(envelope), np.std(envelope)
                else:
                    env_mean, env_std = 0, 0
            except Exception:
                env_mean, env_std = 0, 0

            feat.extend([
                peak_to_valley, overshoot, resp_time, rise_rate,
                grad_mean, grad_std, env_mean, env_std
            ])
        else:
            feat.extend([0] * 8)

        # ---------- 稳态区特征 ----------
        if steady.size > 0:
            steady_mean = np.nanmean(steady)
            steady_std = np.nanstd(steady)
            steady_cv = steady_std / (steady_mean + self.eps)

            base_mean = np.nanmean(baseline) if baseline.size > 0 else 0
            steady_err = steady_mean - base_mean

            steady_fluc = np.nanstd(np.diff(steady)) if steady.size > 1 else 0

            if steady.size > 10:
                try:
                    ac = np.correlate(steady, steady, mode='full')
                    ac = ac[ac.size // 2:]
                    ac_min = np.argmin(ac[: min(20, ac.size)]) if ac.size >= 20 else 0
                except:
                    ac_min = 0
            else:
                ac_min = 0

            feat.extend([
                steady_mean, steady_std, steady_cv, steady_err,
                steady_fluc, ac_min
            ])
        else:
            feat.extend([0] * 6)

        # ---------- 全局特征 ----------
        if vals.size > 10:
            diff = np.diff(vals)
            if diff.size > 0:
                oscill = np.sum(np.diff(np.sign(diff)) != 0) / 2
            else:
                oscill = 0

            smoothness = np.nanmean(np.abs(np.diff(vals, n=2))) if vals.size > 2 else 0

            mono_seg = 0
            if len(diff) > 4:
                for i in range(len(diff) - 4):
                    seg = diff[i:i + 5]
                    if np.all(seg >= 0) or np.all(seg <= 0):
                        mono_seg += 1
            mono_ratio = mono_seg / (len(diff) - 4) if len(diff) > 4 else 0

            try:
                std_val = np.std(vals)
                h = std_val * self.config['period']['peak_height_std_ratio'] if std_val > 1e-6 else None

                peaks, props = find_peaks(np.abs(vals), height=h)
                avg_ph = np.mean(props['peak_heights']) if peaks.size else 0
                peak_cnt = peaks.size
            except Exception:
                peak_cnt, avg_ph = 0, 0

            feat.extend([oscill, smoothness, mono_ratio, peak_cnt, avg_ph])
        else:
            feat.extend([0] * 5)

        # ---------- 能量特征 ----------
        if transition.size > 0:
            base_mean = np.nanmean(baseline) if baseline.size > 0 else 0
            dev = transition - base_mean
            energy = np.nansum(dev ** 2)
            iae = np.nansum(np.abs(dev))

            if len(dev) > 1:
                half = len(dev) // 2
                denom = np.nansum(dev[:half] ** 2)
                e_ratio = np.nansum(dev[half:] ** 2) / (denom + self.eps)
            else:
                e_ratio = 0

            feat.extend([energy, iae, e_ratio])
        else:
            feat.extend([0] * 3)

        feat = np.pad(feat, (0, max(0, 25 - len(feat))), 'constant')
        return np.array(feat[:25])

    # --------------------------------------------------------------------
    # 3. 频域特征
    # --------------------------------------------------------------------
    def extract_frequency_features(self, sensor_data, channel):
        if channel not in sensor_data.columns:
            return np.zeros(12)
        vals = sensor_data[channel].to_numpy()
        if vals.size < 10:
            return np.zeros(12)

        try:
            try:
                detrended = signal.detrend(vals)
            except Exception:
                detrended = vals

            n = detrended.size
            if n < 2:
                return np.zeros(12)

            fft_vals = np.fft.fft(detrended)
            fft_freq = np.fft.fftfreq(n, d=1)
            pos_mask = fft_freq > 0
            freqs = fft_freq[pos_mask]
            mags = np.abs(fft_vals[pos_mask])

            if mags.size == 0:
                return np.zeros(12)

            centroid = np.sum(freqs * mags) / np.sum(mags)
            bandwidth = np.sqrt(np.sum((freqs - centroid) ** 2 * mags) / np.sum(mags)) if centroid > 0 else 0

            dom_idx   = np.argmax(mags)
            dom_freq  = freqs[dom_idx]
            dom_mag   = mags[dom_idx]
            high_mask = freqs > centroid
            high_ratio = np.sum(mags[high_mask]) / np.sum(mags) if high_mask.any() else 0

            skew_val = np.nan if np.isnan(skew(mags)) else skew(mags)
            kurt_val = np.nan if np.isnan(kurtosis(mags)) else kurtosis(mags)
            pdf  = mags / np.sum(mags)
            entropy = -np.sum(pdf * np.log(pdf + 1e-10))

            features = np.array([
                centroid, bandwidth, dom_freq, dom_mag, high_ratio,
                skew_val, kurt_val, entropy
            ])

            for low, high in self.config['freq']['bands']:
                idx = (freqs >= low) & (freqs < high)
                ratio = np.sum(mags[idx]) / np.sum(mags) if idx.any() else 0
                features = np.append(features, ratio)

            return np.pad(features, (0, max(0, 12 - features.size)), 'constant')[:12]
        except Exception as e:
            self.logger.debug(f"[频域特征] 计算错误 ({channel}): {e}")
            return np.zeros(12)

    # --------------------------------------------------------------------
    # 4. 稳定性特征
    # --------------------------------------------------------------------
    def extract_stability_specific_features(self, sensor_data, channel):
        if channel not in sensor_data.columns:
            return np.zeros(15)
        vals = sensor_data[channel].to_numpy()
        if vals.size < 10:
            return np.zeros(15)

        feat = []

        try:
            diff = np.diff(vals)
            abs_diff = np.abs(diff)
            mean_val = np.nanmean(np.abs(vals))

            rel_change_mean = np.nanmean(abs_diff) / (mean_val + self.eps)
            rel_change_std  = np.nanstd(abs_diff) / (mean_val + self.eps)
            feat.extend([rel_change_mean, rel_change_std])

            mut = self._detect_mutation_points(vals)
            feat.extend([mut['mutation_count'], mut['mutation_intensity']])

            feat.extend(self._calculate_segmented_stability(vals))

            try:
                analytic = hilbert(vals)
                env = np.abs(analytic)
                env_rate = np.nanmean(np.abs(np.diff(env)))
                env_std  = np.nanstd(env)
                feat.extend([env_rate, env_std])

                phase = np.unwrap(np.angle(analytic))
                inst_freq = np.diff(phase) / (2.0 * np.pi)
                feat.append(np.nanstd(inst_freq))
            except Exception:
                feat.extend([0, 0, 0])

            if vals.size > 10:
                try:
                    slope = np.polyfit(np.arange(vals.size), vals, 1)[0]
                    feat.append(slope)
                except Exception:
                    feat.append(0)
            else:
                feat.append(0)

            noise = np.nanstd(signal.filtfilt(self.b_high, self.a_high, vals)) / (np.nanstd(vals) + self.eps)
            feat.append(noise)
        except Exception as e:
            self.logger.debug(f"[稳定性] 计算错误 ({channel}): {e}")
            return np.zeros(15)

        return np.pad(feat, (0, max(0, 15 - len(feat))), 'constant')[:15]

    def _calculate_segmented_stability(self, values):
        n_segs = self.config['stat']['stability_segments']
        if len(values) < n_segs:
            return [0.0] * 3

        seg_size = len(values) // n_segs
        means, stds = [], []
        for i in range(n_segs):
            seg = values[i * seg_size: (i + 1) * seg_size if i < n_segs - 1 else len(values)]
            if seg.size:
                means.append(np.nanmean(seg))
                stds.append(np.nanstd(seg))

        if len(means) > 1:
            mean_var  = np.nanstd(means) / (np.nanmean(means) + self.eps)
            std_var   = np.nanstd(stds) / (np.nanmean(stds) + self.eps)
            max_diff  = np.max(means) - np.min(means)
            return [mean_var, std_var, max_diff]
        return [0.0] * 3

    # --------------------------------------------------------------------
    # 5. 周期性特征
    # --------------------------------------------------------------------
    def extract_periodicity_features(self, sensor_data, channel):
        if channel not in sensor_data.columns:
            return np.zeros(8)
        vals = sensor_data[channel].to_numpy()

        cfg = self.config['period']
        if vals.size < cfg['min_len']:
            return np.zeros(8)

        feat = []
        try:
            ac = np.correlate(vals, vals, mode='full')
            ac = ac[ac.size // 2:]
            peaks, _ = find_peaks(ac[:100],
                                  height=np.std(ac) * cfg['peak_height_std_ratio'])
            if peaks.size > 1:
                intervals = np.diff(peaks)
                consistency = 1.0 / (np.std(intervals) + self.eps)
                period = np.mean(intervals)
            else:
                consistency, period = 0, 0
            feat.extend([consistency, period, peaks.size])

            p_height = np.mean(vals) + cfg['peak_height_std_ratio'] * np.std(vals)
            v_height = -(np.mean(vals) - cfg['peak_height_std_ratio'] * np.std(vals))
            dist = cfg['business_dist']

            peaks_biz, _ = find_peaks(vals, height=p_height, distance=dist)
            valleys_biz, _ = find_peaks(-vals, height=v_height, distance=dist)

            feat.extend([
                peaks_biz.size, valleys_biz.size,
                peaks_biz.size + valleys_biz.size,
                1 if peaks_biz.size >= 3 else 0,
                1 if peaks_biz.size <= 2 else 0
            ])
        except Exception as e:
            self.logger.debug(f"[周期性特征] 计算错误 ({channel}): {e}")
            feat.extend([0] * 8)

        return np.pad(feat, (0, max(0, 8 - len(feat))), 'constant')[:8]

    # --------------------------------------------------------------------
    # 6. 多通道与业务规则特征
    # --------------------------------------------------------------------
    def extract_multi_sensor_sync_features(self, sensor_data, channels):
        if len(channels) < 2:
            self.logger.debug(f"[多通道] 通道数不足 ({len(channels)} < 2)，跳过")
            return np.zeros(3)
        try:
            sync_feats = self._detect_sync_jumps(sensor_data, channels)
            return np.array(sync_feats[:3])
        except Exception as e:
            self.logger.debug(f"[多通道] 同步特征计算失败: {e}")
            return np.zeros(3)

    def _detect_sync_jumps(self, sensor_data, channels):
        main_channel = channels[0]
        if main_channel not in sensor_data.columns:
            return [0, 0, 0]

        main_jumps = self._detect_jump_points(sensor_data[main_channel].to_numpy())
        scores = []

        for ch in channels[1:]:
            if ch in sensor_data.columns:
                jumps = self._detect_jump_points(sensor_data[ch].to_numpy())
                scores.append(self._calculate_jump_sync(main_jumps, jumps))
            else:
                scores.append(0)

        avg_score = np.mean(scores) if scores else 0
        max_score = np.max(scores) if scores else 0
        cnt = np.sum(np.array(scores) > 0.5)
        return [avg_score, max_score, cnt]

    def _detect_jump_points(self, values):
        w = self.config['business']['jump_window']
        ratio = self.config['business']['jump_ratio']

        values = np.array(values)

        if values.size < w * 2:
            return []

        try:
            moving_avg = np.convolve(values, np.ones(w) / w, mode='valid')
            right = moving_avg[w:]
            left = moving_avg[:len(moving_avg) - w]
            diff = np.abs(right - left)

            std_val = np.std(values)
            threshold = std_val * ratio

            if std_val < 1e-6:
                return []

            indices = np.where(diff > threshold)[0]

            return (indices + w).tolist()
        except Exception:
            return []

    def _calculate_jump_sync(self, main_jumps, jumps):
        main_jumps = np.array(main_jumps) if isinstance(main_jumps, list) else main_jumps
        jumps = np.array(jumps) if isinstance(jumps, list) else jumps

        if main_jumps.size == 0 or jumps.size == 0:
            return 0.0

        tol = self.config['business']['sync_tolerance']

        min_diffs = []
        for mj in main_jumps:
            diffs = np.abs(jumps - mj)
            min_diffs.append(np.min(diffs))

        min_diffs = np.array(min_diffs)
        return np.mean(min_diffs <= tol)

    def extract_business_rule_features(self, sensor_data, channels):
        if not channels:
            self.logger.warning("[业务规则] 未指定通道")
            return np.zeros(6)

        main_channel = channels[0]
        features = []

        try:
            if main_channel in sensor_data.columns:
                vals = sensor_data[main_channel].to_numpy()
                avg = np.mean(vals)
                features.extend([avg, 1 if avg > 1.0 else 0])
            else:
                features.extend([0, 0])

            pat = self._classify_fluctuation_pattern(sensor_data, channels)
            features.extend(pat)

            if main_channel in sensor_data.columns and 'timestamp' in sensor_data.columns:
                ts = sensor_data['timestamp'].to_numpy()
                vals = sensor_data[main_channel].to_numpy()
                idx = np.argmin(np.abs(ts - self.fault_time_target))
                pre = vals[max(0, idx - 1000):idx]
                if pre.size > 10:
                    features.extend([np.std(pre), np.std(pre) / (np.mean(pre) + self.eps)])
                else:
                    features.extend([0, 0])
            else:
                features.extend([0, 0])
        except Exception as e:
            self.logger.debug(f"[业务规则] 特征提取失败: {e}")
            features = [0] * 6

        return np.array(features[:6])

    def _classify_fluctuation_pattern(self, sensor_data, channels):
        if not channels or channels[0] not in sensor_data.columns:
            return [0, 0, 0]
        vals = sensor_data[channels[0]].to_numpy()
        cfg = self.config['business']

        reg_score = self._calculate_regularity_score(vals)
        drop_score = self._calculate_drop_score(vals)

        if reg_score > cfg['pattern_thresh_high'] and drop_score < cfg['pattern_thresh_low']:
            pat = 1
        elif drop_score > cfg['pattern_thresh_high'] and reg_score < cfg['pattern_thresh_low']:
            pat = 2
        else:
            pat = 0
        return [reg_score, drop_score, pat]

    def _calculate_regularity_score(self, values):
        if len(values) < 30 or np.nanstd(values) < 1e-9:
            return 0.0
        try:
            norm_vals = values - np.nanmean(values)
            ac = np.correlate(values - np.mean(values),
                              values - np.mean(values), mode='full')[len(values) - 1:]
            peaks, _ = find_peaks(ac[:50], height=0.3 * np.max(ac))
            return min(len(peaks) * 0.2, 1.0)
        except Exception:
            return 0.0

    def _calculate_drop_score(self, values):
        if len(values) < 10:
            return 0.0
        try:
            ratio = self.config['business']['fluctuation_w_ratio']
            w = min(50, int(len(values) * ratio) if len(values) * ratio > 1 else 1)
            moving_avg = np.convolve(values, np.ones(w) / w, mode='valid')
            drop = (moving_avg.max() - moving_avg.min()) / (moving_avg.max() or 1e-8)
            return min(drop, 1.0)
        except Exception:
            return 0.0

    # --------------------------------------------------------------------
    # 7. 对外接口
    # --------------------------------------------------------------------
    def extract_single_channel_features(self, sensor_data, channel):
        """
        对外接口：提取单通道所有特征（支持常量信号状态保留）
        """
        if channel not in sensor_data.columns:
            self.logger.error(f"[STAGE:FEAT] [{ErrorCode.CHANNEL_MISSING.name}({ErrorCode.CHANNEL_MISSING.value})] 通道缺失: {channel}")
            return np.zeros(80)

        vals = sensor_data[channel].to_numpy()

        if vals is None or len(vals) == 0 or np.all(np.isnan(vals)):
            self.logger.warning(f"[STAGE:FEAT] [{ErrorCode.CHANNEL_ALL_NAN.name}({ErrorCode.CHANNEL_ALL_NAN.value})] 通道 {channel} 数据全空或全为 NaN")
            return np.zeros(80)

        std_val = np.nanstd(vals)
        is_constant = std_val < 1e-9

        try:
            f1 = self.extract_basic_statistical_features(sensor_data, channel)

            if is_constant:
                self.logger.info(f"[STAGE:FEAT] 通道 {channel} 为静态信号 (均值: {np.nanmean(vals):.1f})，跳过高级特征提取")
                f2 = np.zeros(25)
                f3 = np.zeros(12)
                f4 = np.zeros(15)
                f5 = np.zeros(8)
            else:
                f2 = self.extract_dynamic_features(sensor_data, channel)
                f3 = self.extract_frequency_features(sensor_data, channel)
                f4 = self.extract_stability_specific_features(sensor_data, channel)
                f5 = self.extract_periodicity_features(sensor_data, channel)

            combined = np.concatenate([f1, f2, f3, f4, f5])
            return combined

        except Exception as e:
            self.logger.error(f"[STAGE:FEAT] [{ErrorCode.CHANNEL_EXTRACT_CRASH.name}({ErrorCode.CHANNEL_EXTRACT_CRASH.value})] 通道 {channel} 特征提取崩溃: {str(e)}")
            return np.zeros(80)

    def extract_multi_channel_features(self, sensor_data, multi_channels=None):
        """
        对外接口：多通道关联特征
        """
        if multi_channels is None:
            multi_channels = self.default_channels

        try:
            valid_channels = [c for c in multi_channels if c in sensor_data.columns]
            if len(valid_channels) < 2:
                self.logger.warning(f"[STAGE:FEAT] [{ErrorCode.MULTI_CH_INSUFFICIENT.name}({ErrorCode.MULTI_CH_INSUFFICIENT.value})] 多通道特征因列不足跳过: {valid_channels}")
                return np.zeros(9)

            f1 = self.extract_multi_sensor_sync_features(sensor_data, valid_channels)
            f2 = self.extract_business_rule_features(sensor_data, valid_channels)

            combined = np.concatenate([f1, f2])
            return combined

        except Exception as e:
            self.logger.error(f"[STAGE:FEAT] [{ErrorCode.MULTI_CH_EXTRACT_FAIL.name}({ErrorCode.MULTI_CH_EXTRACT_FAIL.value})] 多通道特征提取失败: {str(e)}")
            return np.zeros(9)
