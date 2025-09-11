#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib as mpl


outdir = Path("results")
txtfile = outdir / "basin_data.txt"
RASTER_DPI = 600
max_points = None
FIGSIZE_SINGLE = (8, 6)
out_png_template = outdir / "basin_view{v}.png"
out_pdf_template = outdir / "basin_view{v}.pdf"
# ------------------------------------------------

plt.rcParams.update({
    "font.family": "Times New Roman",
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "figure.dpi": 100,
})


data = np.loadtxt(txtfile, skiprows=1)
K1_flat, K2_flat, R0_flat, state_flat = data.T
sync_rgb = (251 / 255, 192 / 255, 217 / 255)
incoh_rgb = (8 / 255, 39 / 255, 90 / 255)
colors = np.empty((state_flat.size, 3))
colors[state_flat == 1] = sync_rgb
colors[state_flat == 0] = incoh_rgb

if (max_points is not None) and (K1_flat.size > max_points):
    rng = np.random.default_rng(seed=42)
    sel = rng.choice(K1_flat.size, size=max_points, replace=False)
    K1_plot, K2_plot, R0_plot, colors_plot = K1_flat[sel], K2_flat[sel], R0_flat[sel], colors[sel]
    print(f"Downsampled {K1_flat.size} -> {max_points} points for plotting.")
else:
    K1_plot, K2_plot, R0_plot, colors_plot = K1_flat, K2_flat, R0_flat, colors

views = [(25, 45), (25, 135), (90, 0)]
outdir.mkdir(parents=True, exist_ok=True)


for v_idx, (elev, azim) in enumerate(views, start=1):

    fig, ax = plt.subplots(
        figsize=FIGSIZE_SINGLE,
        subplot_kw={'projection': '3d'},
        constrained_layout=True
    )

    ax.scatter(
        K1_plot, K2_plot, R0_plot,
        c=colors_plot, s=20, alpha=0.9,
        edgecolors='none', linewidths=0
    )

    ax.view_init(elev=elev, azim=azim)



    ax.set_xlabel("$K_1$", labelpad=10)
    ax.set_ylabel("$K_2$", labelpad=10)


    if v_idx == 3:
        ax.set_zlabel("")
        ax.set_zticks([])
    else:  # 这是3D透视图
        ax.set_zlabel("$r_0$", labelpad=20)

    ax.grid(False)


    for axis in ['x','y','z']:
        try:
            pane = getattr(ax, f"{axis}axis").pane; pane.fill = True; pane.set_edgecolor('black')
        except Exception: pass
    try:
        for spine in ax.spines.values():
            spine.set_visible(True); spine.set_linewidth(1.2)
    except Exception: pass



    out_png = out_png_template.with_name(out_png_template.name.format(v=v_idx))
    print(f"Saving PNG (view {v_idx}) -> {out_png}  (dpi={RASTER_DPI})")
    fig.savefig(out_png, format='png', dpi=RASTER_DPI) # 可以去掉 bbox_inches 和 pad_inches
    plt.close(fig)


    try:
        from PIL import Image

        im = Image.open(out_png).convert('RGB')
        out_pdf = out_pdf_template.with_name(out_pdf_template.name.format(v=v_idx))
        im.save(out_pdf, "PDF", resolution=RASTER_DPI)
        print(f"Saved PDF (from PNG, view {v_idx}) -> {out_pdf}")
    except Exception as e:
        print(f"ERROR converting PNG->PDF for view {v_idx} via Pillow:", e)
        print("PNG was saved at:", out_png)

print("Done. Three separate view files saved in:", outdir)