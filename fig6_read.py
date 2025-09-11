import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# colormap fallback
try:
    import cmcrameri.cm as cm
    def get_colors(n):
        return cm.batlow(np.linspace(0.2, 0.8, n))
except ImportError:
    def get_colors(n):
        return plt.cm.viridis(np.linspace(0.2, 0.8, n))


data_dir  = "bifu_3D/data_txt"
out_dir   = "bifu_3D"
fig_size  = (6, 5)     # 单图大小 (width, height) 英寸
xlabel    = r'$K_1$'
ylabel    = r'$r$'
K2_target = 8          # 只要 K2 = 8

plt.rcParams.update({
    "font.family":     "Times New Roman",
    "axes.labelsize":  18,
    "axes.titlesize":  22,     # 全局标题大小
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "figure.dpi":      300,
})


scenarios = ["Deterministic", "Gaussian", "Lévy"]
colors = get_colors(len(scenarios))

os.makedirs(out_dir, exist_ok=True)

for j_col, scen in enumerate(scenarios):
    clr = colors[j_col]
    K2 = K2_target
    fname = os.path.join(data_dir, f"{scen}_K2_{K2}.txt")
    if not os.path.isfile(fname):
        print(f"Warning: file not found, skipping: {fname}")
        continue

    data = np.loadtxt(fname, comments='#')
    if data.size == 0:
        print(f"Warning: empty file, skipping: {fname}")
        continue

    K1, r_an, r_fwd, r_bwd = data.T

    fig, ax = plt.subplots(figsize=fig_size)
    ax.plot(K1, r_an, color=clr, linestyle='-', linewidth=2, label='Analytical')
    ax.plot(K1, r_fwd, color=clr, linestyle='--', linewidth=1.5, label='Forward')
    ax.plot(K1, r_bwd, color=clr, linestyle=':', linewidth=1.5, label='Backward')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(MultipleLocator(2))  # 横坐标间距为 2
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.set_xlim(K1.min(), K1.max())
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle=':', alpha=0.5)

    # legend 方框
    leg = ax.legend(fontsize=9, loc='upper right', frameon=True, fancybox=False,
                    handlelength=1.2, handletextpad=0.4, labelspacing=0.3)
    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_facecolor('white')
    leg.get_frame().set_alpha(1.0)

    outpath = os.path.join(out_dir, f"bifu_K2_{K2}_{scen}.pdf")
    plt.savefig(outpath, bbox_inches='tight')
    plt.close(fig)
    print("Saved:", outpath)
