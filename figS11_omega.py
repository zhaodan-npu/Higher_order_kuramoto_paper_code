#!/usr/bin/env python3
"""
统计多次实验的 Cauchy(0,1) 频率样本 {omega_i}，绘制
1) |omega_i| 的直方图
2) 每个 realization 的 max |omega_i| 直方图

新增：
- 分别按出版级标准保存两个子图的 PDF 和 PNG（单图里不再画 (a)/(b)）
- 再把两个 PNG 合成一张 1x2 的 PNG/PDF（在合成图上统一加 (a)/(b)）
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import MaxNLocator
import cmcrameri.cm as cm
import matplotlib.image as mpimg

# ====== 参数，可根据论文实际情况调整 ======
N = 100                 # 每次实验的振子数
num_realizations = 1000 # 抽多少组 {omega_i}
delta = 1.0             # Cauchy(0, Delta) 的 Delta

np.random.seed(0)       # 固定随机种子，保证可复现

# ====== 生成多组 {omega_i} ======
# 形状: (num_realizations, N)
omega = delta * np.random.standard_cauchy(size=(num_realizations, N))

# 所有样本摊平后的 |omega|
omega_abs = np.abs(omega).ravel()

# 每组实验中最大的 |omega_i|
max_abs_per_real = np.max(np.abs(omega), axis=1)

# ====== 分位数统计（方便写在回复信中） ======
quantiles = [0.5, 0.9, 0.99, 0.999]
q_vals = np.quantile(omega_abs, quantiles)

print(f"Quantiles of |omega| over all realizations (N = {N}, M = {num_realizations}):")
for q, v in zip(quantiles, q_vals):
    print("  {:4.1f}% quantile: |omega| = {:.3f}".format(q * 100, v))

print("\nMean and std of max |omega_i| in each realization:")
print("  mean max |omega_i| = {:.3f}".format(max_abs_per_real.mean()))
print("  std  max |omega_i| = {:.3f}".format(max_abs_per_real.std()))

# =======================================================================
#   第一部分：分别画两张出版级直方图，并各自保存 PDF + PNG
#   （注意：这里不再在图内写 (a)/(b)）
# =======================================================================

# ====== 统一画图风格（对标你原来的 heatmap 标准） ======
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'figure.dpi': 300,
    'axes.linewidth': 0.8,
})

# --- |omega_i| 直方图（log-x），单独一张图 ---
fig_a, ax_a = plt.subplots(figsize=(5.5, 4))

bins_abs = np.logspace(-2, 3, 80)  # 1e-2 ~ 1e3

ax_a.hist(
    omega_abs,
    bins=bins_abs,
    density=True,
    alpha=0.9,
    color=cm.batlow(0.6),
    edgecolor='none'
)
ax_a.set_xscale('log')
ax_a.set_xlabel(r'$|\omega_i|$')
ax_a.set_ylabel('PDF')
ax_a.xaxis.set_major_locator(MaxNLocator(5))
ax_a.yaxis.set_major_locator(MaxNLocator(5))

plt.tight_layout()

out_pdf_a = Path("omega_abs_hist.pdf")
out_png_a = Path("omega_abs_hist.png")
fig_a.savefig(out_pdf_a, format='pdf', bbox_inches='tight')
fig_a.savefig(out_png_a, dpi=600, bbox_inches='tight')
plt.close(fig_a)
print(f"Saved: {out_pdf_a} and {out_png_a}")

# --- 每个 realization 的 max |omega_i| 直方图，单独一张图 ---
fig_b, ax_b = plt.subplots(figsize=(5.5, 4))

bins_max = np.logspace(0, 4, 60)  # 1 ~ 1e4

ax_b.hist(
    max_abs_per_real,
    bins=bins_max,
    density=True,
    alpha=0.9,
    color=cm.batlow(0.6),
    edgecolor='none'
)
ax_b.set_xscale('log')
ax_b.set_xlabel(r'$\max_i |\omega_i|$')
ax_b.set_ylabel('PDF')
ax_b.xaxis.set_major_locator(MaxNLocator(5))
ax_b.yaxis.set_major_locator(MaxNLocator(5))

plt.tight_layout()

out_pdf_b = Path("omega_max_hist.pdf")
out_png_b = Path("omega_max_hist.png")
fig_b.savefig(out_pdf_b, format='pdf', bbox_inches='tight')
fig_b.savefig(out_png_b, dpi=600, bbox_inches='tight')
plt.close(fig_b)
print(f"Saved: {out_pdf_b} and {out_png_b}")

# =======================================================================
#   第二部分：把两个 PNG 合成一张 1x2 的大图（在这里统一加 (a)/(b)）
# =======================================================================

# 合成大图的风格：Times New Roman + 基础字号 12
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 12,
})

img_paths = [
    str(out_png_a),  # "omega_abs_hist.png"
    str(out_png_b),  # "omega_max_hist.png"
]
labels = ['(a)', '(b)']

# 1 行 2 列布局
fig, axes = plt.subplots(1, 2, figsize=(6, 3))

for ax, img_path, label in zip(axes.flat, img_paths, labels):
    img = mpimg.imread(img_path)
    ax.imshow(img)
    ax.axis('off')

    # 左上角标注 (a), (b)
    ax.text(
        -0.1, 0.98, label,
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=14,
        fontweight='normal',
    )

plt.tight_layout()

combined_pdf = Path("combined_omega_hist_1x2.pdf")
combined_png = Path("combined_omega_hist_1x2.png")
plt.savefig(combined_pdf, bbox_inches='tight')
plt.savefig(combined_png, dpi=600, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {combined_pdf} and {combined_png}")
