#!/usr/bin/env python3

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import cmcrameri.cm as cm


results_dir = Path("results")
out_pdf     = Path("order_param_heatmap_batlow_smoothed22.pdf")


files = sorted(results_dir.glob('order_k1_*.npz'))
if not files:
    raise FileNotFoundError(f"在 {results_dir} 目录中未找到 order_k1_*.npz 文件。")

first = np.load(files[0])
K2_vals = first['K2_vals']
nK2     = K2_vals.size
nK1     = len(files)

order_mat = np.zeros((nK1, nK2))
K1_vals   = np.zeros(nK1)
for idx, f in enumerate(files):
    data = np.load(f)

    if 'k1' in data:
        K1_vals[idx] = data['k1']
    elif 'K1' in data:
        K1_vals[idx] = data['K1']
    else:
        raise KeyError("未在 npz 文件中找到 'k1' 或 'K1' 键")
    order_mat[idx, :] = data['r_mean']


sigma_k1 = 1.0
sigma_k2 = 1.0
order_smooth = gaussian_filter(order_mat, sigma=(sigma_k1, sigma_k2))
print(f"Applied Gaussian filter with sigma=(K1:{sigma_k1},K2:{sigma_k2})")


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
    vmin=order_mat.min(), vmax=order_mat.max(),
    interpolation='bicubic'
)




cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.05)
cbar.set_label(r'$\left< r \right>$', fontsize=16)
cbar.ax.tick_params(labelsize=14)

ax.set_xlabel(r'$K_2$', labelpad=6)
ax.set_ylabel(r'$K_1$', labelpad=6)
ax.xaxis.set_major_locator(plt.MaxNLocator(6))
ax.yaxis.set_major_locator(plt.MaxNLocator(6))

plt.tight_layout()
fig.savefig(out_pdf, format='pdf', bbox_inches='tight')
print(f"Smoothed heatmap saved to {out_pdf}")
