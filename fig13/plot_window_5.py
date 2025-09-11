#!/usr/bin/env python3
"""
plot_p1_average.py — 对 p1 的所有固定长度窗口计算并绘制：
    (a) 平均 EDACF
    (b) 平均 EDSPEC
    (c) 聚合所有事件时刻（y 坐标固定为 1）

本版本将处理“不同窗口 EDACF/EDSPEC 文件长度不一致”问题：
- 统一将所有 EDACF 数据截取到最小公共长度求平均
- 统一将所有 EDSPEC 数据截取到最小公共频率长度求平均
"""

import glob
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# ———— 配置区 ————
path_id = 1
ed_dir = "."

# ---------- 全局美化 ----------
plt.rcParams.update({
    "font.family": "Times New Roman",
    "axes.labelsize": 14,
    "axes.titlesize": 12,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.figsize": (6, 4),
    "grid.linestyle": "--",
})

# 前缀目录
prefix = ed_dir.rstrip("/") + "/" if ed_dir else ""

# ---------- 1) 收集所有 EDACF 文件的 lags 和 C 数组长度 ----------
edacf_pattern = f"{prefix}noise_a1.1_s0.1_p{path_id}_spikes_win*_EDACF.txt"
edacf_files = sorted(glob.glob(edacf_pattern))
if not edacf_files:
    print(f"Error: 未找到任何文件匹配: {edacf_pattern}", file=sys.stderr)
    sys.exit(1)

lags_list = []
C_list = []

for fn in edacf_files:
    try:
        data = np.loadtxt(fn, unpack=True)
    except Exception as e:
        print(f"Warning: 读取 EDACF 文件失败，跳过: {fn} ({e})", file=sys.stderr)
        continue
    if data.shape[0] < 2:
        print(f"Warning: EDACF 数据列数异常，跳过: {fn}", file=sys.stderr)
        continue
    lags_i = data[0]
    C_i = data[1]
    if lags_i.size != C_i.size:
        print(f"Warning: lags 与 C 长度不一致，跳过: {fn}", file=sys.stderr)
        continue
    lags_list.append(lags_i)
    C_list.append(C_i)

if not C_list:
    print("Error: 没有任何有效的 EDACF 文件可用于平均。", file=sys.stderr)
    sys.exit(1)

min_len_lags = min(arr.size for arr in lags_list)
lags_ref = lags_list[0][:min_len_lags]

C_trimmed = []
for idx, C_i in enumerate(C_list):
    if C_i.size < min_len_lags:
        print(f"Warning: 跳过第 {idx} 个 EDACF（长度 {C_i.size} < {min_len_lags}）", file=sys.stderr)
        continue
    C_norm = C_i / C_i[0]
    C_trimmed.append(C_norm[:min_len_lags])

C_trimmed = np.vstack(C_trimmed)
C_mean = C_trimmed.mean(axis=0)
C_sem = C_trimmed.std(axis=0, ddof=1) / np.sqrt(C_trimmed.shape[0])

# ---------- 2) 收集所有 EDSPEC 文件的 freqs 和 P 数组长度 ----------
edspec_pattern = f"{prefix}noise_a1.1_s0.1_p{path_id}_spikes_win*_EDSPEC.txt"
edspec_files = sorted(glob.glob(edspec_pattern))
if not edspec_files:
    print(f"Error: 未找到任何文件匹配: {edspec_pattern}", file=sys.stderr)
    sys.exit(1)

freqs_list = []
P_list = []

for fn in edspec_files:
    try:
        data = np.loadtxt(fn, unpack=True)
    except Exception as e:
        print(f"Warning: 读取 EDSPEC 文件失败，跳过: {fn} ({e})", file=sys.stderr)
        continue
    if data.shape[0] < 2:
        print(f"Warning: EDSPEC 数据列数异常，跳过: {fn}", file=sys.stderr)
        continue
    f_i, P_i = data[0], data[1]
    if f_i.size != P_i.size:
        print(f"Warning: freqs 与 P 长度不一致，跳过: {fn}", file=sys.stderr)
        continue
    freqs_list.append(f_i)
    P_list.append(P_i)

if not P_list:
    print("Error: 没有任何有效的 EDSPEC 文件可用于平均。", file=sys.stderr)
    sys.exit(1)

min_len_freqs = min(arr.size for arr in freqs_list)
freqs_ref = freqs_list[0][:min_len_freqs]

P_trimmed = []
for idx, P_i in enumerate(P_list):
    if P_i.size < min_len_freqs:
        print(f"Warning: 跳过第 {idx} 个 EDSPEC（长度 {P_i.size} < {min_len_freqs}）", file=sys.stderr)
        continue
    P_trimmed.append(P_i[:min_len_freqs])

P_trimmed = np.vstack(P_trimmed)
P_mean = P_trimmed.mean(axis=0)
P_sem = P_trimmed.std(axis=0, ddof=1) / np.sqrt(P_trimmed.shape[0])

# ---------- 3) 聚合所有窗口的事件时刻 ----------
spike_pattern = f"{prefix}noise_a1.1_s0.1_p{path_id}_spikes_win*.txt"
spike_files = sorted(glob.glob(spike_pattern))
if not spike_files:
    print(f"Error: 未找到任何文件匹配: {spike_pattern}", file=sys.stderr)
    sys.exit(1)

all_times = []
for fn in spike_files:
    try:
        data = np.loadtxt(fn, unpack=True, skiprows=1)
    except Exception as e:
        print(f"Warning: 读取事件时刻文件失败，跳过: {fn} ({e})", file=sys.stderr)
        continue
    t_i = data if data.ndim == 1 else data[0]
    all_times.append(t_i)

if all_times:
    all_times = np.concatenate(all_times)
    all_times.sort()
else:
    all_times = np.array([])

# ---------- 4) 分别绘制并保存三个子图 ----------

# (a) 平均 EDACF + SEM 带
plt.figure()
plt.plot(lags_ref, C_mean, 'b-', linewidth=2, label="Average EDACF")
plt.fill_between(lags_ref, C_mean - C_sem, C_mean + C_sem,
                 color='blue', alpha=0.3, label="SEM")
plt.xlabel("Lag(dimensionless)")
plt.ylabel("EDACF")
plt.grid(True)
plt.legend()
outfn_edacf = f"p{path_id}_windows_average_EDACF.pdf"
plt.tight_layout()
plt.savefig(outfn_edacf, dpi=300)
plt.close()

# (b) 平均 EDSPEC + SEM 带，分成三段在 log-log 空间拟合（幂律）
# (b) 平均 EDSPEC + SEM 带（保留红色原谱），并在 log-log 空间按固定频段拟合三段（并绘出拟合线）
# (b) 平均 EDSPEC + SEM 带（保留红色原谱），并在 log-log 空间按固定频段用 P_sem 加权拟合三段
# (b) 平均 EDSPEC + SEM 带（保留红谱），使用“连续的加权分段线性拟合”（3 段，强制连续）
# (b) 平均 EDSPEC + SEM 带（保留红色原谱），三段独立加权拟合（P_sem -> weight），并绘出 ±1σ 带
plt.figure()

# 只使用有效的正频率和正谱值
valid_mask = (freqs_ref > 0) & np.isfinite(freqs_ref) & np.isfinite(P_mean) & (P_mean > 0) & np.isfinite(P_sem)
freqs_plot = freqs_ref[valid_mask]
P_mean_plot = P_mean[valid_mask]
P_sem_plot = P_sem[valid_mask]

if freqs_plot.size == 0:
    raise RuntimeError("没有可用于绘图的正频率/谱值。")

# 画原始红色谱线和红色 SEM 带（保持你原来的配色）
plt.loglog(freqs_plot, P_mean_plot, 'r-', linewidth=2, label="Average EDSPEC")
lower = np.maximum(P_mean_plot - P_sem_plot, 1e-30)
upper = P_mean_plot + P_sem_plot
plt.fill_between(freqs_plot, lower, upper, where=(upper > 0), color='red', alpha=0.3, label="SEM")

# 固定断点
break1 = 1e-1
break2 = 10**0.2  # ≈1.58489

# 计算 log-space 值与 sigma_log10（带下限保护）
ln10 = np.log(10.0)
logf_all = np.log10(freqs_plot)
logP_all = np.log10(P_mean_plot)
sigma_log10 = P_sem_plot / (P_mean_plot * ln10)

# 保护性下限（避免 sigma=0 或 nan）
finite_sigma = sigma_log10[np.isfinite(sigma_log10) & (sigma_log10 > 0)]
sigma_floor = (np.percentile(finite_sigma, 1) * 1e-3) if finite_sigma.size > 0 else 1e-12
sigma_floor = max(sigma_floor, 1e-12)
sigma_log10 = np.where((~np.isfinite(sigma_log10)) | (sigma_log10 <= 0), sigma_floor, sigma_log10)

# 定义三段掩码（严格边界）
seg_masks = [
    (freqs_plot <= break1),
    ((freqs_plot > break1) & (freqs_plot <= break2)),
    (freqs_plot > break2)
]

# 拟合线颜色（非红）
fit_colors = ['#1f77b4', '#2ca02c', '#9467bd']

fit_info = []
f_min_all = freqs_plot.min()
f_max_all = freqs_plot.max()

for i, seg_mask in enumerate(seg_masks):
    seg_no = i + 1
    idx = np.where(seg_mask)[0]
    if idx.size < 2:
        print(f"Warning: segment {seg_no} 样本点不足（{idx.size}），跳过拟合并继续绘图。", file=sys.stderr)
        continue

    f_seg = freqs_plot[idx]
    P_seg = P_mean_plot[idx]
    sigma_seg = sigma_log10[idx]

    logf_seg = np.log10(f_seg)
    logP_seg = np.log10(P_seg)

    # 权重 w = 1/sigma (np.polyfit 使用 w 直接)
    w_seg = 1.0 / sigma_seg

    # 加权线性拟合（log-space）： logP = slope * logf + intercept
    try:
        (slope, intercept), cov = np.polyfit(logf_seg, logP_seg, 1, w=w_seg, cov=True)
        slope_err = np.sqrt(cov[0, 0])
        intercept_err = np.sqrt(cov[1, 1])
    except Exception:
        # 退化为无协方差版本
        p = np.polyfit(logf_seg, logP_seg, 1, w=w_seg)
        slope, intercept = p[0], p[1]
        slope_err = np.nan
        intercept_err = np.nan

    # 保存拟合信息
    n_points = idx.size
    # 近似 95% CI（若 slope_err 可得）：用 z≈1.96（当样本很小可考虑 t 分布）
    if not np.isnan(slope_err):
        z95 = 1.96
        slope_95_lo = slope - z95 * slope_err
        slope_95_hi = slope + z95 * slope_err
    else:
        slope_95_lo = slope_95_hi = np.nan

    fit_info.append({
        'seg': seg_no,
        'slope': slope,
        'slope_err': slope_err,
        'slope_95': (slope_95_lo, slope_95_hi),
        'intercept': intercept,
        'intercept_err': intercept_err,
        'n_points': n_points,
        'f_range_data': (f_seg.min(), f_seg.max()),
        'color': fit_colors[i]
    })

    # 为美观在绘图时略微扩展绘制区间（但拟合仅用段内数据）
    # 扩展系数（不超过整体 data 范围）
    expand = 1.2
    plot_low = max(f_min_all, f_seg.min() / expand)
    plot_high = min(f_max_all, f_seg.max() * expand)
    # 但也确保不会画出与邻段重叠太多（只是为了好看）
    freq_fit = np.logspace(np.log10(plot_low), np.log10(plot_high), 300)
    logf_fit = np.log10(freq_fit)
    logP_fit = slope * logf_fit + intercept
    P_fit = 10**logP_fit

    # 用协方差在 log-space 上算预测方差 var(y_pred) = [x,1] cov [x,1]^T
    if 'cov' in locals() and cov is not None:
        # cov is 2x2 from last polyfit; but ensure we're using the cov for current segment
        cov_seg = cov
    else:
        cov_seg = None

    # 计算 ±1σ 带（在 log-space，然后转换回线性）
    if cov_seg is not None:
        y_se = np.sqrt((logf_fit**2) * cov_seg[0, 0] + 2 * logf_fit * cov_seg[0, 1] + cov_seg[1, 1])
        P_lower = 10**(logP_fit - y_se)
        P_upper = 10**(logP_fit + y_se)
        plt.fill_between(freq_fit, P_lower, P_upper, color=fit_colors[i], alpha=0.15)

    # 绘制拟合线（虚线）
    plt.loglog(freq_fit, P_fit, linestyle='--', linewidth=2, color=fit_colors[i],
               label=f"Fit seg{seg_no}: slope={slope:.3f}" + (f"±{slope_err:.3f}" if not np.isnan(slope_err) else ""))

    # 在段中间位置标注 slope±1σ（便于快速查看）
    try:
        text_x = np.sqrt(f_seg.min() * f_seg.max())  # 几何平均
        text_y = 10**(slope * np.log10(text_x) + intercept)
        # ------------------ 替换这里的 plt.text(...) 代码块 ------------------
        # 更稳健地计算每段注释位置：按段内相对位置 alpha，和在对数空间的十进制偏移 offsets_decades
        # 三段的相对位置和竖直偏移（你可以微调这些数组）
        alphas = [0.35, 0.50, 0.65]  # 段内位置（从左到右的比例）
        offsets_decades = [0.18, 0.3, 0.4]  # 十进制偏移（正值上移，负值下移）
        ha_opts = ['center', 'center', 'center']
        va_opts = ['bottom', 'top', 'bottom']

        # i 是当前段的索引（0-based），slope/intercept/fit color 已在上文定义
        try:
            alpha = alphas[i]
            # 在段内按 alpha 取一个 x（几何方式）
            logx_min = np.log10(f_seg.min())
            logx_max = np.log10(f_seg.max())
            logx_text = (1 - alpha) * logx_min + alpha * logx_max
            text_x = 10 ** (logx_text)

            # 文本基准 y 值：谱线上该 x 的值
            text_y_base = 10 ** (slope * logx_text + intercept)
            # 在对数坐标上偏移 offsets_decades[i] 个 decade（即乘以 10**offset）
            text_y = text_y_base * (10 ** offsets_decades[i])

            # 裁剪到当前 y 轴显示范围，避免超出画布
            ax = plt.gca()
            ymin, ymax = ax.get_ylim()
            # 留一点边距
            text_y = np.clip(text_y, ymin * 1.02, ymax * 0.98)

            plt.text(text_x, text_y,
                     f"{slope:.3f}±{slope_err:.3f}" if not np.isnan(slope_err) else f"{slope:.3f}",
                     fontsize=10, ha=ha_opts[i], va=va_opts[i],
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=fit_colors[i], alpha=0.9))
        except Exception as e:
            # 若任何错误（例如段内点数异常），保守地跳过注释，避免中断主流程
            print(f"Warning: failed to place label for segment {seg_no}: {e}", file=sys.stderr)
        # --------------------------------------------------------------------

    except Exception:
        pass

# 最后美化并保存（保持输出文件名不变）
plt.xlabel("Frequency")
plt.ylabel("EDSPEC")
plt.grid(True, which="both")
plt.legend()
outfn_edspec = f"p{path_id}_windows_average_EDSPEC.pdf"
plt.tight_layout()
plt.savefig(outfn_edspec, dpi=300)
plt.close()

# 打印拟合结果（控制台输出详细信息）
print("EDSPEC 三段独立加权拟合结果（基于 P_sem 权重）：")
for info in fit_info:
    fmin, fmax = info['f_range_data']
    s = info['slope']
    se = info['slope_err']
    n = info['n_points']
    lo95, hi95 = info['slope_95']
    if np.isnan(se):
        print(f"  段 {info['seg']}: freq [{fmin:.3g}, {fmax:.3g}]  slope = {s:.6g} (n={n})")
    else:
        print(f"  段 {info['seg']}: freq [{fmin:.3g}, {fmax:.3g}]  slope = {s:.6g} ± {se:.6g} (1σ, n={n})")
        print(f"    approx 95% CI: [{lo95:.6g}, {hi95:.6g}]")


# (c) 聚合事件时刻，y=1 常数
plt.figure()
if all_times.size > 0:
    plt.scatter(all_times, np.ones_like(all_times),
                s=10, marker="|", color="C2", alpha=0.7,
                rasterized=True)   # <--- 关键修改：避免 PDF 文件过大
plt.xlabel("Time")
plt.yticks([])
plt.xlim(2000, 2050)                # <--- 固定横坐标范围
plt.grid(False)
outfn_events = f"p{path_id}_windows_average_event_times.pdf"
plt.tight_layout()
plt.savefig(outfn_events, dpi=300)
plt.close()


print(f"Saved EDACF figure to {outfn_edacf}")
print(f"Saved EDSPEC figure to {outfn_edspec}")
print(f"Saved Event Times figure to {outfn_events}")