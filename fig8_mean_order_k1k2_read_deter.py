#!/usr/bin/env python3
"""
读取 `order_param_results.txt` 格式为 50×50 的 Mean Global Order Parameter 矩阵，
进行高斯平滑并绘制出版级热力图，使用 cmcrameri 的 'oslo_r' 顺序色图。
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter
import cmcrameri.cm as cm
from matplotlib.ticker import MaxNLocator


K1_vals = np.linspace(-1, 6, 50)
K2_vals = np.linspace( 0,10, 50)


data_file = Path("mean_of_order/order_param_k1k2_det.txt")
if not data_file.exists():
    raise FileNotFoundError(f"未找到数据文件: {data_file}")
order_mat = np.loadtxt(data_file)



SMOOTH_SIGMA_K1 = 1.0
SMOOTH_SIGMA_K2 = 1.0
order_smooth = gaussian_filter(order_mat, sigma=(SMOOTH_SIGMA_K1, SMOOTH_SIGMA_K2))
print(f"Applied Gaussian filter with sigma=({SMOOTH_SIGMA_K1},{SMOOTH_SIGMA_K2})")


plt.rcParams.update({
    'font.family': 'Times New Roman',
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'figure.dpi': 300,
    'axes.linewidth': 0.8,
})


fig, ax = plt.subplots(figsize=(6.5, 6))

im = ax.imshow(
    order_smooth,
    origin='lower',
    aspect='auto',
    extent=[K2_vals.min(), K2_vals.max(), K1_vals.min(), K1_vals.max()],
    cmap=cm.batlow,
    vmin=np.nanmin(order_mat), vmax=np.nanmax(order_mat),
    interpolation='bicubic'
)




cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.05)
cbar.set_label(r'$\left< r \right>$', fontsize=16)
cbar.ax.tick_params(labelsize=14)

ax.set_xlabel(r'$K_2$', labelpad=6)
ax.set_ylabel(r'$K_1$', labelpad=6)
ax.xaxis.set_major_locator(MaxNLocator(6))
ax.yaxis.set_major_locator(MaxNLocator(6))

plt.tight_layout()
out_pdf = Path('order_param_heatmap_oslo_smoothed.pdf')
fig.savefig(out_pdf, format='pdf', bbox_inches='tight')
print(f"Smoothed heatmap saved to {out_pdf}")
