#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compute_and_plot_levy_intervals.py (版本：对数柱状图 + 统一美化)

这个脚本会：
 1. 读取目录 'noise_spikes2/' 下的所有 Lévy spike 文件。
 2. 计算 inter-burst 和 intra-burst 间隔。
 3. 合并所有间隔数据。
 4. 在对数等距 bin 上做概率密度直方图。
 5. 在对数–对数坐标图上，用柱状图画出分布，并应用统一的美化风格保存为 PDF。
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import sys

# ———— 配置区 ————
# Lévy spike 原始文件所在目录
INPUT_DIR = "noise_spikes2"

# burst 划分阈值
BURST_SPLIT_THRESHOLD = 200.0  # 秒

# 对数分箱数量
NUM_LOG_BINS = 40

# 【修改点 1】添加全局美化设置，与 linear 版本脚本同步
plt.rcParams.update({
    "font.family": "Times New Roman",
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.figsize": (6, 4),
    "grid.linestyle": "--",
})

# 输出目录
OUT_DIR = "levy_interval_distributions_bargraph"
os.makedirs(OUT_DIR, exist_ok=True)

# 【修改点 2】修改输出文件名和格式为 PDF
FN_PLOT = os.path.join(OUT_DIR, "intervals_loglog_bars.pdf")

# ————————————————

# --- 数据读取和处理部分 (保持不变) ---
if not os.path.isdir(INPUT_DIR):
    print(f"Error: 输入目录 '{INPUT_DIR}' 不存在，请检查路径。", file=sys.stderr)
    sys.exit(1)
pattern = os.path.join(INPUT_DIR, "noise_a1.1_s0.1_p*.txt")
file_list = sorted(glob.glob(pattern))
if len(file_list) == 0:
    print(f"Error: 在 '{INPUT_DIR}' 下找不到任何 'noise_a1.1_s0.1_p*.txt' 文件。", file=sys.stderr)
    sys.exit(1)
print(f"共找到 {len(file_list)} 个 Lévy spike 文件，开始处理...\n")
inter_all_paths = []
intra_all_paths = []
for fn in file_list:
    base = os.path.basename(fn)
    try:
        data = np.loadtxt(fn, delimiter='\t', skiprows=1, dtype=float)
    except Exception as e:
        print(f"  Warning: 无法读取 '{base}'，跳过。原因：{e}", file=sys.stderr)
        continue
    if data.size == 0: continue
    if data.ndim == 1:
        times = np.array([data[0]], dtype=float)
    else:
        times = data[:, 0].astype(float)
    times.sort()
    diffs = np.diff(times)
    if diffs.size == 0: continue
    split_idxs = np.where(diffs > BURST_SPLIT_THRESHOLD)[0]
    burst_starts, burst_ends, start_idx = [], [], 0
    for idx_split in split_idxs:
        burst_starts.append(start_idx)
        burst_ends.append(idx_split)
        start_idx = idx_split + 1
    burst_starts.append(start_idx)
    burst_ends.append(len(times) - 1)
    intra_list, inter_list = [], []
    for (s, e) in zip(burst_starts, burst_ends):
        segment = times[s : e+1]
        if segment.size >= 2:
            intra_list.append(np.diff(segment))
    for i in range(len(burst_starts) - 1):
        end_time = times[burst_ends[i]]
        next_start = times[burst_starts[i+1]]
        inter_list.append(next_start - end_time)
    inter_arr = np.array(inter_list, dtype=float) if len(inter_list) > 0 else np.empty(0, dtype=float)
    intra_arr = np.concatenate(intra_list).astype(float) if len(intra_list) > 0 else np.empty(0, dtype=float)
    inter_all_paths.append(inter_arr)
    intra_all_paths.append(intra_arr)
print("所有文件已处理完毕。\n")

# --- 数据汇总和直方图计算部分 (保持不变) ---
inter_all = np.concatenate([arr for arr in inter_all_paths if arr.size > 0]) if any(arr.size > 0 for arr in inter_all_paths) else np.empty(0)
intra_all = np.concatenate([arr for arr in intra_all_paths if arr.size > 0]) if any(arr.size > 0 for arr in intra_all_paths) else np.empty(0)
combined_all = np.concatenate([inter_all, intra_all]) if (inter_all.size > 0 or intra_all.size > 0) else np.empty(0)
print(f"汇总后：inter_all={inter_all.size}, intra_all={intra_all.size}, combined_all={combined_all.size}\n")
if combined_all.size == 0:
    print("Error: 没有任何正的间隔值，无法做对数分箱。", file=sys.stderr)
    sys.exit(1)
positive = combined_all[combined_all > 0]
log_min, log_max = positive.min(), positive.max()
print(f"对数分箱范围: [{log_min:.6g}, {log_max:.6g}]\n")
log_edges = np.logspace(np.log10(log_min), np.log10(log_max), NUM_LOG_BINS + 1)
log_centers = np.sqrt(log_edges[:-1] * log_edges[1:])
log_widths = np.diff(log_edges)
hist_inter, _ = np.histogram(inter_all, bins=log_edges, density=True)
hist_intra, _ = np.histogram(intra_all, bins=log_edges, density=True)
hist_combined, _ = np.histogram(combined_all, bins=log_edges, density=True)

# --- 最终绘图部分 (应用新的美化标准) ---
plt.figure() # figsize 会由 rcParams 自动设置

# 仅在 combined_all 不为空时绘制 (inter 和 intra 只是为了颜色区分)
if combined_all.size > 0:
    # 只画 combined 的曲线，使其与 linear 版本的图保持一致（只画一条线）
    plt.bar(log_centers, hist_combined, width=log_widths, align='center',
            facecolor="C2", edgecolor="black", alpha=0.75,
            label="Log-Bin Histogram") # 即使有label, plt.legend()被注释了也不会显示

# 设置坐标轴为对数尺度
plt.xscale("log")
plt.yscale("log")

# 设置Y轴范围，避免因0值导致范围问题
if np.any(hist_combined > 0):
    min_y = np.min(hist_combined[hist_combined > 0])
    plt.ylim(bottom=min_y * 0.5)

# 【修改点 3】简化标签，注释掉标题和图例
plt.xlabel("Interval")
plt.ylabel("PDF")
# plt.title("Lévy Spike Interval Distributions (Log-Bin Histogram)")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()

# 【修改点 4】更新保存命令
plt.savefig(FN_PLOT, dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ 已生成并保存对数坐标柱状图（应用新美化）：{FN_PLOT}\n")