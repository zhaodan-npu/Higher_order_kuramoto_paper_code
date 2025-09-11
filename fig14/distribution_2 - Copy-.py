#!/usr/bin/env python3
"""
compute_and_plot_logbin_intervals.py

从 “results/” 目录下所有 spikes_a0_s0_pX.txt 文件开始，
执行以下步骤：
 1. 对每个路径文件：
    a. 读取第一列 spike 时间（假定文件首行是表头，需要 skiprows=1）。
    b. 按相邻时间差 > 200 s 将 spike 划分为若干 burst。
    c. 计算每个 burst 内部相邻 spike 的 ISI（intra-burst ISI）以及相邻 burst 之间的间隔（inter-burst interval）。
    d. 将上述两种间隔合并，得到该路径的 combined_intervals_i 数组。
 2. 将所有路径的 combined_intervals_i 拼接得到一个一维数组 combined_intervals。
 3. 使用 np.logspace 在对数尺度上生成对数等距的 bin_edges，从最小正间隔到最大间隔等分 NUM_LOG_BINS 段。
 4. 对 combined_intervals 计算对数分箱直方图（density=True），得到概率密度 hist_vals。
 5. 绘图并保存对数分箱直方图，横坐标为对数刻度，纵坐标为概率密度。

使用方法：
    python compute_and_plot_logbin_intervals.py

请确保已安装：numpy、matplotlib

脚本会生成：
  - “combined_intervals.npy”：合并后的所有间隔数组
  - “log_bin_centers.npy/.txt”：对数 bin 的中心点
  - “hist_vals_log.npy/.txt”：对应每个中心的密度值
  - “log_edges.txt”：对数分箱时的所有 bin 边界（新增）
  - “logbin_histogram.png”：对数分箱下的概率密度条形图
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import sys

# ———— 配置区 ————
# 1) 存放所有 spike 文件的目录
results_dir = "results"

# 2) 匹配模式：所有路径文件名形如 spikes_a0_s0_p1.txt
pattern = os.path.join(results_dir, "spikes_a0_s0_p*.txt")

# 3) burst 划分阈值 (如果相邻两个 spike 时间差 > BURST_SPLIT_THRESHOLD, 视作新 burst)
BURST_SPLIT_THRESHOLD = 200.0  # 单位：秒

# 4) 对数分箱参数：在 [log_min, log_max] 范围内生成 NUM_LOG_BINS 个对数等距区间
NUM_LOG_BINS = 100

# 5) 输出目录和文件名
out_dir            = "logbin_results_a0s0"
fn_combined_npy    = os.path.join(out_dir, "combined_intervals.npy")
fn_bin_centers_npy = os.path.join(out_dir, "log_bin_centers.npy")
fn_hist_vals_npy   = os.path.join(out_dir, "hist_vals_log.npy")
fn_bin_centers_txt = os.path.join(out_dir, "log_bin_centers.txt")
fn_hist_vals_txt   = os.path.join(out_dir, "hist_vals_log.txt")
fn_edges_txt       = os.path.join(out_dir, "log_edges.txt")       # ＜－－ 新增：保存 log_edges
fn_plot            = os.path.join(out_dir, "logbin_histogram.png")
# ——————————————

# 确保输出目录存在
os.makedirs(out_dir, exist_ok=True)

# 1) 找到所有路径文件
path_files = sorted(glob.glob(pattern))
if not path_files:
    print(f"Error: 在目录 '{results_dir}' 下未找到任何匹配 '{pattern}' 的文件。", file=sys.stderr)
    sys.exit(1)

all_combined_intervals = []  # 用来存储每条路径的 combined_intervals_i

print(f"找到 {len(path_files)} 个路径文件，开始逐条处理...")

for fn in path_files:
    # 1.a 读取 spike 时间（假定第一列是时间，跳过表头行）
    try:
        data = np.loadtxt(fn, unpack=True, skiprows=1)
    except Exception as e:
        print(f"Warning: 无法读取 '{fn}'，跳过。原因: {e}", file=sys.stderr)
        continue

    # data 可能是一维或多维，取第一列时间
    if data.ndim == 1:
        t = data.copy()
    else:
        t = data[0].copy()

    # 如果少于两个 spike，则无法计算任何间隔
    if t.size < 2:
        print(f"  [跳过] '{os.path.basename(fn)}' 只有 {t.size} 个 spike", file=sys.stderr)
        continue

    # 1.b 保证时间升序
    t.sort()

    # 1.b 按相邻差 > BURST_SPLIT_THRESHOLD 划分 burst
    diffs = np.diff(t)
    split_idxs = np.where(diffs > BURST_SPLIT_THRESHOLD)[0]

    burst_starts = []
    burst_ends   = []
    start_idx    = 0
    for split_idx in split_idxs:
        burst_starts.append(start_idx)
        burst_ends.append(split_idx)
        start_idx = split_idx + 1
    # 添加最后一个 burst
    burst_starts.append(start_idx)
    burst_ends.append(len(t) - 1)

    # 1.c 计算 intra-burst ISI 和 inter-burst intervals
    intra_list = []
    inter_list = []

    # 对每个 burst 内部计算 ISI
    for (s, e) in zip(burst_starts, burst_ends):
        burst_times = t[s : e+1]
        if burst_times.size >= 2:
            local_isi = np.diff(burst_times)
            intra_list.append(local_isi)

    # 计算相邻两个 burst 之间的间隔：下一个 burst 开始 - 本 burst 结束
    for i in range(len(burst_starts) - 1):
        end_time         = t[ burst_ends[i] ]
        next_start_time  = t[ burst_starts[i+1] ]
        inter_list.append(next_start_time - end_time)

    # 1.d 合并成 numpy 数组
    if inter_list:
        inter_arr = np.asarray(inter_list, dtype=float)
    else:
        inter_arr = np.array([], dtype=float)

    if intra_list:
        intra_arr = np.concatenate(intra_list).astype(float)
    else:
        intra_arr = np.array([], dtype=float)

    # 如果两者都存在，就合并；否则按有的那一部分
    if inter_arr.size > 0 and intra_arr.size > 0:
        combined_i = np.concatenate([inter_arr, intra_arr])
    elif inter_arr.size > 0:
        combined_i = inter_arr.copy()
    else:
        combined_i = intra_arr.copy()

    all_combined_intervals.append(combined_i)
    print(f"  处理 '{os.path.basename(fn)}' → burst: {len(burst_starts)}, "
          f"inter={inter_arr.size}, intra={intra_arr.size}, combined={combined_i.size}")

print("所有路径处理完毕。\n")

# 2) 拼接所有路径的 combined_intervals_i，得到总的 combined_intervals
if all_combined_intervals:
    combined_intervals = np.concatenate([arr for arr in all_combined_intervals if arr.size > 0])
else:
    combined_intervals = np.array([], dtype=float)

print(f"合并后的总间隔数: {combined_intervals.size}")
# 保存到 .npy，方便后续直接载入
np.save(fn_combined_npy, combined_intervals)
print(f"已保存所有路径合并后的间隔到：{fn_combined_npy}")

if combined_intervals.size == 0:
    print("No intervals available to histogram. 程序退出。", file=sys.stderr)
    sys.exit(1)

# 3) 定义对数分箱的 bin_edges
#    取最小正间隔为 log_min(不能取 0)，最大间隔为 log_max
positive_intervals = combined_intervals[combined_intervals > 0]
if positive_intervals.size == 0:
    print("Error: combined_intervals 中没有正值，无法做对数分箱。", file=sys.stderr)
    sys.exit(1)

log_min = positive_intervals.min()
log_max = positive_intervals.max()
print(f"对数分箱范围: [{log_min:.3g}, {log_max:.3g}]")

# 构造 logspace，从 log10(log_min) 到 log10(log_max)，共 NUM_LOG_BINS+1 个边界
log_edges = np.logspace(np.log10(log_min), np.log10(log_max), NUM_LOG_BINS + 1)

# ＜－－ 在此处新增：保存 log_edges 到 txt 文件，供后续绘图使用
np.savetxt(fn_edges_txt, log_edges, fmt="%.6e")
print(f"已保存对数分箱边界 log_edges 到：{fn_edges_txt}")

# 计算每个 bin 的中心点：几何平均
log_centers = np.sqrt(log_edges[:-1] * log_edges[1:])

# 4) 用 np.histogram 计算对数分箱下的概率密度
hist_vals, _ = np.histogram(
    combined_intervals,
    bins=log_edges,
    density=True
)

# 保存 bin 中心和对应密度
np.save(fn_bin_centers_npy, log_centers)
np.save(fn_hist_vals_npy,   hist_vals)
np.savetxt(fn_bin_centers_txt, log_centers, fmt="%.6e")
np.savetxt(fn_hist_vals_txt,   hist_vals,   fmt="%.6e")
print(f"已保存 log-bin 中心到：{fn_bin_centers_npy} / {fn_bin_centers_txt}")
print(f"已保存密度值到：{fn_hist_vals_npy} / {fn_hist_vals_txt}")

# 5) 绘图：对数分箱直方图
plt.figure(figsize=(7, 4))
plt.bar(
    log_centers,
    hist_vals,
    width=(log_edges[1:] - log_edges[:-1]),  # 使用真实宽度
    align='center',
    color="C2",
    alpha=0.7,
    edgecolor="none"
)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Interval (s) [log scale]")
plt.ylabel("Probability Density")
plt.title("Combined Intervals Distribution (Log-Bin Histogram)")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(fn_plot, dpi=300)
plt.close()
print(f"已生成对数分箱直方图，保存为：{fn_plot}")
