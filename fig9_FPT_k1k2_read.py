#!/usr/bin/env python3
"""

"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pathlib import Path
from scipy.ndimage import gaussian_filter
import cmcrameri.cm as cm



K1_vals = np.linspace(-1, 6, 50)
K2_vals = np.linspace( 0,10, 50)

data_file = Path("first_passage/fpt_k1k2_2_0.5.txt")
if not data_file.exists():
    raise FileNotFoundError(f"无法找到文件: {data_file}")
fpt_mat = np.loadtxt(data_file)


SMOOTH_SIGMA_K1 = 1.0
SMOOTH_SIGMA_K2 = 1.5
fpt_smooth = gaussian_filter(fpt_mat, sigma=(SMOOTH_SIGMA_K1, SMOOTH_SIGMA_K2))
print(f"Applied Gaussian filter with sigma=({SMOOTH_SIGMA_K1},{SMOOTH_SIGMA_K2})")


plt.rcParams.update({
    'font.family': 'Times New Roman',
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'figure.dpi': 300,
    'axes.linewidth': 0.8,
})


fig, ax = plt.subplots(figsize=(6.5, 5))

im = ax.imshow(
    fpt_smooth,
    origin='lower',
    aspect='auto',
    extent=[K2_vals.min(), K2_vals.max(), K1_vals.min(), K1_vals.max()],
    cmap=cm.lapaz_r,
    vmin=np.nanmin(fpt_mat), vmax=np.nanmax(fpt_mat),
    interpolation='bicubic'
)




cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.05)
cbar.set_label(r'$\mathrm{T}_{\text{first}}$', fontsize=16)
cbar.ax.tick_params(labelsize=14)


ax.set_xlabel(r'$K_2$', labelpad=6)
ax.set_ylabel(r'$K_1$', labelpad=6)
ax.xaxis.set_major_locator(MaxNLocator(6))
ax.yaxis.set_major_locator(MaxNLocator(6))

plt.tight_layout()
out_pdf = Path('fpt_heatmap_2_0.5.pdf')
fig.savefig(out_pdf, format='pdf', bbox_inches='tight')
print(f"Smoothed heatmap saved to {out_pdf}")
