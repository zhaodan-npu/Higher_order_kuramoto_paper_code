#!/usr/bin/env python3
"""
Basin Stability 热力图 —— 顺序色图 + 出版级美化
（横轴：K2；纵轴：K1）
Author: <your name>
Requires: matplotlib, numpy, cmcrameri, scipy
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter
from scipy.ndimage import gaussian_filter
import cmcrameri.cm as cm

# ---------- 1. 读取数据 ----------
data_file  = "basin_stability_alpha1_6_sigma0_5_further.txt"
# 假定 shape = (len(K2_list), len(K1_list))
basin_mat  = np.loadtxt(data_file)

# 与生成数据时一致
K1_list    = np.linspace(-1, 6, 100)
K2_list    = np.linspace(0, 10, 100)

# ---------- 2. 平滑 & 插值 选项 ----------
USE_GAUSSIAN = True
SIGMA_SMOOTH = 0.6
if USE_GAUSSIAN:
    plot_mat = gaussian_filter(basin_mat, SIGMA_SMOOTH)
else:
    plot_mat = basin_mat

# ---------- 3. 全局排版 ----------
plt.rcParams.update({
    "font.family":   "Times New Roman",
    "axes.labelsize":16,
    "xtick.labelsize":14,
    "ytick.labelsize":14,
    "axes.linewidth": 0.8,
    "figure.dpi":    300
})

# ---------- 4. 绘制热力图 ----------
fig, ax = plt.subplots(figsize=(6.5, 5))

cmap   = cm.batlow
interp = 'bilinear'

# 转置矩阵，把原来的 [K2,K1] → [K1,K2]
im = ax.imshow(
    plot_mat.T,
    extent=[K2_list.min(), K2_list.max(),  # 横轴：K2
            K1_list.min(), K1_list.max()], # 纵轴：K1
    origin='lower',
    aspect='auto',
    cmap=cmap,
    vmin=basin_mat.min(),
    vmax=basin_mat.max(),
    interpolation=interp
)

# ---------- 5. （可选）白色等高线突出高稳区 ----------
# threshold = 0.8
# cs = ax.contour(
#     plot_mat.T,                       # 注意矩阵也要用转置
#     levels=[threshold],
#     extent=[K2_list.min(), K2_list.max(),
#             K1_list.min(), K1_list.max()],
#     colors='white',
#     linewidths=1.2
# )
# ax.clabel(cs, fmt="%.2f", fontsize=10, inline=True, inline_spacing=4)

# ---------- 6. 颜色条 ----------
cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.04)
cbar.set_label(r"Basin Stability ($\alpha=1.6,\ \sigma=0.5$)", fontsize=15)
cbar.ax.tick_params(length=3, width=0.6)

# ---------- 7. 坐标轴 ----------
ax.set_xlabel(r'$K_2$', labelpad=6)  # 横轴改为 K2
ax.set_ylabel(r'$K_1$', labelpad=6)  # 纵轴改为 K1

ax.xaxis.set_major_locator(MultipleLocator(2.0))
ax.yaxis.set_major_locator(MultipleLocator(1.0))
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}"))
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}"))

# ---------- 8. 保存 & 显示 ----------
plt.tight_layout()
plt.savefig("basin_stability_alpha2_sigma0_511.pdf", bbox_inches='tight')
plt.show()
