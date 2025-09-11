#!/usr/bin/env python3
"""
plot_histogram_and_tail_fit.py

本脚本从当前目录读取 compute_and_plot_logbin_intervals.py 保存的数据：
  - log_bin_centers.npy   （对数分箱的中心点）
  - hist_vals_log.npy     （对应每个 bin 中心的概率密度）
  - log_edges.txt         （对数分箱的边界，可选，但建议提前保存）

然后在一张图中同时绘制：
  1) 对数–对数坐标下的柱状图（使用每个 bin 的真实宽度）
  2) 对 {Δt ≥ TAIL_MIN} 的尾部数据做幂律拟合并用虚线画出拟合曲线

使用方法：
    python plot_histogram_and_tail_fit.py

依赖：numpy, matplotlib, scipy
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# ———— 配置区 ————
# 只拟合 Δt ≥ TAIL_MIN 的那部分
TAIL_MIN = 0.1  # 单位：秒，可按需要调整

# 输入文件名（假设都在当前目录）
FN_BIN_CENTERS = "log_bin_centers.npy"
FN_HIST_VALS   = "hist_vals_log.npy"
FN_LOG_EDGES   = "log_edges.txt"  # 如果不存在，脚本会自动退化
# ---------- 全局美化 ----------
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
# 输出目录及文件名
OUT_DIR     = "plot_with_fit(final)"
OUTPUT_PNG  = os.path.join(OUT_DIR, "hist_and_tail_fit.pdf")

os.makedirs(OUT_DIR, exist_ok=True)
# ——————————————

# 1) 检查并加载数据
if not (os.path.isfile(FN_BIN_CENTERS) and os.path.isfile(FN_HIST_VALS)):
    print(f"Error: 当前目录下需要存在 '{FN_BIN_CENTERS}' 和 '{FN_HIST_VALS}'。", file=sys.stderr)
    sys.exit(1)

log_centers = np.load(FN_BIN_CENTERS)  # 形如 (NUM_LOG_BINS,)
hist_vals   = np.load(FN_HIST_VALS)    # 形如 (NUM_LOG_BINS,)

# 尝试加载对数 bin 的边缘，如果没有就自己近似生成
if os.path.isfile(FN_LOG_EDGES):
    try:
        log_edges = np.loadtxt(FN_LOG_EDGES)  # 形如 (NUM_LOG_BINS+1,)
    except Exception:
        print(f"Warning: 无法正确读取 '{FN_LOG_EDGES}'，将改用近似宽度。", file=sys.stderr)
        log_edges = None
else:
    log_edges = None

# 如果 log_edges 缺失或长度不符合，则用 centers 近似计算宽度
if log_edges is None or log_edges.ndim != 1 or log_edges.shape[0] != (log_centers.shape[0] + 1):
    # 中心点间距近似当做宽度
    approx_width = log_centers[1] - log_centers[0]
    widths = np.full_like(log_centers, fill_value=approx_width)
    print("Note: 未找到或无法使用 'log_edges.txt'，柱子宽度将近似为常数。", file=sys.stderr)
else:
    # 正确地计算出每个 bin 的线性宽度
    widths = log_edges[1:] - log_edges[:-1]

# 2) 筛选尾部数据（Δt ≥ TAIL_MIN 且密度 > 0）
mask_tail = (log_centers >= TAIL_MIN) & (hist_vals > 0)
x_tail = log_centers[mask_tail]
y_tail = hist_vals[mask_tail]

if x_tail.size < 2:
    print(f"Error: 筛选后尾部数据点太少 (仅 {x_tail.size} 个)，无法拟合幂律。", file=sys.stderr)
    sys.exit(1)

# # 对数–对数回归
# log10_x = np.log10(x_tail)
# log10_y = np.log10(y_tail)
# slope, intercept, r_value, p_value, std_err = linregress(log10_x, log10_y)
# alpha = -slope
# R2 = r_value**2
#
# print("\n===== 幂律尾部拟合结果 =====")
# print(f"拟合区间: Δt ∈ [{TAIL_MIN:.3g}, {x_tail.max():.3g}] s")
# print(f"log10(P) = {slope:.4f} · log10(Δt) + {intercept:.4f}")
# print(f"幂律指数 α = {alpha:.4f}, 决定系数 R² = {R2:.4f}\n")

# 3) 绘图：对数–对数柱状图 + 拟合虚线
plt.figure(figsize=(6, 4))

# —— 3a. 画出所有 bins（浅灰底色 + 浅蓝填充） ——
plt.bar(
    log_centers,
    hist_vals,
    width=widths,
    align="center",
    facecolor="#AED6F1",     # 浅蓝色填充
    edgecolor="#7FB3D5",     # 较深蓝灰边框
    linewidth=0.5,
    alpha=0.8,
    label="Log‐Bin Histogram"
)

# —— 3b. 高亮 Δt ≥ TAIL_MIN 的那部分 bins（深橙色） ——
plt.bar(
    x_tail,
    y_tail,
    width=widths[mask_tail],
    align="center",
    facecolor="#AED6F1",     # 深橙色填充
    edgecolor="#7FB3D5",     # 橙红边框
    linewidth=0.5,
    alpha=0.9
)

# # —— 3c. 拟合曲线 ——
# #    拟合曲线只画在 [TAIL_MIN, max(x_tail)] 区间内
# FIT_X_MAX = x_tail.max()
# fit_xs = np.logspace(np.log10(TAIL_MIN), np.log10(FIT_X_MAX), 200)
# fit_ys = (10**intercept) * (fit_xs**slope)
#
# plt.loglog(
#     fit_xs,
#     fit_ys,
#     "k--",    # 黑色虚线
#     linewidth=2,
#     label=f"Power‐Law Fit: α = {alpha:.2f}, R² = {R2:.3f}"
# )
# # plt.rcParams.update({
# #     "font.family":"Times New Roman",
# #     "axes.labelsize":10,
# #     "xtick.labelsize":10,
# #     "ytick.labelsize":10,
# #     "axes.linewidth":1,
# #     "figure.dpi":300,
# # })
# —— 3d. 坐标与样式设置 ——
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Interval",fontname="Times New Roman",fontsize=14)
plt.ylabel("PDF",fontname="Times New Roman",fontsize=14)
# plt.title("Combined Intervals (Log-Bin Histogram) and Tail Power‐Law Fit")
plt.grid(True, which="both", linestyle="--", alpha=0.3)
plt.legend(loc="upper right", framealpha=0.9)
plt.tight_layout()

# —— 3e. 保存并退出 ——
plt.savefig(OUTPUT_PNG)
plt.close()
print(f"已保存直方图与尾部拟合图到：{OUTPUT_PNG}")
