"""
Mean Global Order Parameter 热力图  |  顺序色图 + 可选平滑 + 出版级美化
Author: <your name>
Requires: matplotlib, numpy, cmcrameri, scipy
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter
from scipy.ndimage import gaussian_filter
import cmcrameri.cm as cm

# ---------- 1. 读取数据 ----------
data_file = "mean_of_order/order_param_1.1_2.0_0.1_0.6.txt"
order_mat = np.loadtxt(data_file)                   # shape: (Nα, Nσ)

# 与模拟脚本保持一致的坐标
alpha_vals = np.append(np.arange(1.1, 2.0, 0.1), 2.0)
sigma_vals = np.arange(0.1, 0.6, 0.05)

# ---------- 2. 平滑 & 插值 选项 ----------
USE_GAUSSIAN = False    # True=数据级高斯平滑；False=仅插值
SIGMA_SMOOTH = 0.6      # 平滑强度 (0.5–1.0)
plot_mat     = gaussian_filter(order_mat, SIGMA_SMOOTH) if USE_GAUSSIAN else order_mat

order_plot = gaussian_filter(order_mat, SIGMA_SMOOTH) if USE_GAUSSIAN else order_mat

# ---------- 3. 全局排版 ----------
plt.rcParams.update({
    "font.family": "Times New Roman",
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "axes.linewidth": 0.8,
    "figure.dpi": 300
})

# ---------- 4. 绘制热力图 ----------
fig, ax = plt.subplots(figsize=(6.5, 5))

cmap       = cm.batlow         # 同 Basin‑Stability
interp     = 'bilinear'        # 'nearest' | 'bilinear' | 'bicubic'

im = ax.imshow(
    order_plot,
    extent=[sigma_vals.min(), sigma_vals.max(),
            alpha_vals.min(), alpha_vals.max()],
    origin='lower', aspect='auto',
    cmap=cmap, vmin=order_mat.min(), vmax=order_mat.max(),
    interpolation=interp
)

# ---------- 5. 添加白色等高线 -----------
# 强调高序参数区域：取最大值80%作为阈值
# threshold = 0.8 * order_mat.max()
threshold = 0.8
cs = ax.contour(
    plot_mat,
    levels=[threshold],
    extent=[sigma_vals.min(), sigma_vals.max(),
            alpha_vals.min(), alpha_vals.max()],
    colors='white',
    linewidths=1.6
)
ax.clabel(cs, fmt="%.2f", fontsize=16, inline=True, inline_spacing=4)


# ---------- 5. 颜色条 ----------
cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.04)
cbar.set_label(r'$\left< r \right>$', fontsize=15)
cbar.ax.tick_params(length=3, width=0.6)

# ---------- 6. 坐标轴美化 ----------
ax.set_xlabel(r'$\sigma$', labelpad=6)
ax.set_ylabel(r'$\alpha$', labelpad=6)
ax.xaxis.set_major_locator(MultipleLocator(0.05))
ax.yaxis.set_major_locator(MultipleLocator(0.2))
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}"))

# # ---------- 7. (可选) 等高线 ----------
# levels = np.linspace(order_mat.min()+0.1, order_mat.max(), 5)
# cs = ax.contour(order_plot, levels=levels,
#                 extent=[sigma_vals.min(), sigma_vals.max(),
#                         alpha_vals.min(), alpha_vals.max()],
#                 colors='k', linewidths=0.6)
# ax.clabel(cs, fmt="%.2f", fontsize=10, inline=True)

# ---------- 8. 保存 & 显示 ----------
plt.tight_layout()
plt.savefig("mean_of_order/order_param_1.1_2.0_0.1_0.6_heatmap_batlow.pdf", bbox_inches='tight')
plt.show()
