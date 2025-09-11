"""
Basin‑Stability 热力图  |  顺序色图 + 可选平滑 + 自定义等高线
Author: <your name>
Requires: matplotlib, numpy, cmcrameri, scipy
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter
from scipy.ndimage import gaussian_filter
import cmcrameri.cm as cm
from matplotlib.colors import ListedColormap


data_file = "basin_stability/basin_stability_results_1.1_2.0_0.1_0.6.txt"
stab_mat  = np.loadtxt(data_file)


alpha_vals = np.append(np.arange(1.1, 2.0, 0.1), 2.0)
sigma_vals = np.arange(0.1, 0.6, 0.05)


USE_GAUSSIAN, SIGMA_SMOOTH = True, 0.5
stab_plot = gaussian_filter(stab_mat, SIGMA_SMOOTH) if USE_GAUSSIAN else stab_mat


CONTOUR_STYLE = "A"
THRESHOLD     = 0.8
LABEL_FMT     = "%.2f"
NUM_LEVELS    = 4


plt.rcParams.update({
    "font.family": "Times New Roman",
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "axes.linewidth": 0.8,
    "figure.dpi": 300
})


fig, ax = plt.subplots(figsize=(6.5, 5))

cmap  = cm.batlow
interp = 'bilinear'

im = ax.imshow(
        stab_plot,
        extent=[sigma_vals.min(), sigma_vals.max(),
                alpha_vals.min(), alpha_vals.max()],
        origin='lower', aspect='auto',
        cmap=cmap, interpolation=interp,
        vmin=stab_mat.min(), vmax=stab_mat.max())


if CONTOUR_STYLE.upper() == "A":

    cs = ax.contour(
            stab_plot,
            levels=[THRESHOLD],
            extent=[sigma_vals.min(), sigma_vals.max(),
                    alpha_vals.min(), alpha_vals.max()],
            colors='white', linewidths=1.2)
    ax.clabel(cs, fmt=LABEL_FMT, fontsize=12, inline=True,
              inline_spacing=4, use_clabeltext=True)

elif CONTOUR_STYLE.upper() == "B":


    base_colors = cmap(np.linspace(0.65, 1.0, 128))
    darker      = np.clip(base_colors * [0.85, 0.85, 0.85, 1], 0, 1)
    fill_cmap   = ListedColormap(darker)


    cf = ax.contourf(
            stab_plot,
            levels=np.linspace(THRESHOLD, stab_mat.max(), NUM_LEVELS),
            extent=[sigma_vals.min(), sigma_vals.max(),
                    alpha_vals.min(), alpha_vals.max()],
            cmap=fill_cmap, alpha=0.35, antialiased=True)

    cs = ax.contour(
            stab_plot,
            levels=[THRESHOLD],
            extent=[sigma_vals.min(), sigma_vals.max(),
                    alpha_vals.min(), alpha_vals.max()],
            colors='black', linewidths=0.8)
    ax.clabel(cs, fmt=LABEL_FMT, fontsize=12, inline=True)


cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.04)
cbar.set_label(r'$\mathrm{BS}$', fontsize=16)
cbar.ax.tick_params(length=3, width=0.6)

ax.set_xlabel(r'$\sigma$')
ax.set_ylabel(r'$\alpha$')
ax.xaxis.set_major_locator(MultipleLocator(0.1))
ax.yaxis.set_major_locator(MultipleLocator(0.2))
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _ : f"{x:.1f}"))


plt.tight_layout()
fname = "basin_stability/basin_stability_heatmap_batlow.pdf"
plt.savefig(fname, bbox_inches='tight')
plt.show()
print(f"Saved to {fname}")
