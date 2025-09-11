"""
PDF 热力图 —— 顺序色图 + 平滑过渡
Author: <your name>
Requires: matplotlib, numpy, cmcrameri, scipy
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
from scipy.ndimage import gaussian_filter             # ← 方案 B 用
import cmcrameri.cm as cm


pdf_matrix = np.loadtxt("PDF/pdf_matrix_alpha_1.2.txt")
sigma_vals = np.arange(0.1, 2.1, 0.1)
x_vals     = np.linspace(0, 1, 200)


USE_GAUSSIAN  = False
SIGMA_SMOOTH  = 0.7


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

cmap = cm.batlow
interp_method = 'bilinear'

im = ax.imshow(pdf_plot,
               extent=[x_vals.min(), x_vals.max(),
                       sigma_vals.min(), sigma_vals.max()],
               origin='lower',
               aspect='auto',
               cmap=cmap,
               vmin=pdf_matrix.min(),
               vmax=pdf_matrix.max(),
               interpolation=interp_method)


cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.04)
cbar.set_label(r'PDF ($\alpha=1.2$)', fontsize=14)
cbar.ax.tick_params(length=3, width=0.6)


ax.set_xlabel(r'$r$')
ax.set_ylabel(r'$\sigma$')
ax.xaxis.set_major_locator(MultipleLocator(0.2))
ax.yaxis.set_major_locator(MultipleLocator(0.4))
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _ : f"{x:.1f}"))


plt.tight_layout()
fname = "PDF/pdf_heatmap_sigma_publication_alpha_1.2_smooth.pdf"
plt.savefig(fname, bbox_inches='tight')
plt.show()
print(f"Saved to {fname}")
