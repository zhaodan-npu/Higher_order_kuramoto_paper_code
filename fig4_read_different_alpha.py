"""
PDF 热力图 —— 顺序色图 + 可选平滑（r–α）
Author: <your name>
Requires: matplotlib, numpy, cmcrameri, scipy
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter
from scipy.ndimage import gaussian_filter          # ← 仅当启用数据级平滑
import cmcrameri.cm as cm                          # 科研配色（pip install cmcrameri）


data_file  = "PDF/pdf_matrix_sigma_0.5.txt"
pdf_matrix = np.loadtxt(data_file)


alpha_vals = np.append(np.arange(1.1, 2.0, 0.1), 2.0)
x_vals     = np.linspace(0, 1, 200)


USE_GAUSSIAN = False
SIGMA_SMOOTH = 0.7

if USE_GAUSSIAN:
    pdf_plot = gaussian_filter(pdf_matrix, sigma=SIGMA_SMOOTH)
else:
    pdf_plot = pdf_matrix


plt.rcParams.update({
    "font.family": "Times New Roman",
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "axes.linewidth": 0.8,
    "figure.dpi": 300
})


fig, ax = plt.subplots(figsize=(6, 5))

cmap           = cm.batlow
interp_method  = 'bilinear'

im = ax.imshow(pdf_plot,
               extent=[x_vals.min(), x_vals.max(),
                       alpha_vals.min(), alpha_vals.max()],
               origin='lower',
               aspect='auto',
               cmap=cmap,
               vmin=pdf_matrix.min(),
               vmax=pdf_matrix.max(),
               interpolation=interp_method)


cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.04)
cbar.set_label(r"PDF($\sigma=0.5$)", fontsize=14)
cbar.ax.tick_params(length=3, width=0.6)


ax.set_xlabel(r'$r$')
ax.set_ylabel(r'$\alpha$')
ax.xaxis.set_major_locator(MultipleLocator(0.2))
ax.yaxis.set_major_locator(MultipleLocator(0.4))
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _ : f"{x:.1f}"))


plt.tight_layout()
fname = "PDF/pdf_heatmap_r_alpha_batlow_sigma_0.5.pdf"
plt.savefig(fname, bbox_inches='tight')
plt.show()
print(f"Saved to {fname}")
