"""
First‑Passage‑Time (FPT) 热力图  |  顺序色图(反向) + 可选平滑 + 自定义等高线
Author: <your name>
Requires: matplotlib, numpy, cmcrameri, scipy
"""

# ---------- 0. 依赖 ----------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter
from scipy.ndimage import gaussian_filter
import cmcrameri.cm as cm
from matplotlib.colors import ListedColormap
from matplotlib.colors import ListedColormap

# ---------- 1. 读入数据 ----------
data_file = "first_passage/fpt_1.1_2.0_0.1_0.9.txt"          # ← 你的 FPT 输出文件
fpt_mat   = np.loadtxt(data_file)      # shape: (Nα , Nσ)

# === 坐标 ===
alpha_vals = np.append(np.arange(1.1, 2.0, 0.1), 2.0)  # y
sigma_vals = np.arange(0.1, 1.0, 0.1)                  # x

# ---------- 2. 平滑 ----------
USE_GAUSSIAN, SIGMA_SMOOTH = True, 0.5     # True/False 与 σ
fpt_plot = gaussian_filter(fpt_mat, SIGMA_SMOOTH) if USE_GAUSSIAN else fpt_mat

# ---------- 3. 等高线 ----------
CONTOUR_STYLE = "A"      # "A" = 白边线；"B" = 半透明填充
THRESHOLD     = 0.8 * fpt_mat.max()  # 高 FPT 区域阈值
LABEL_FMT     = "%.1f"
NUM_LEVELS    = 4        # 方案 B 分级

# ---------- 4. 排版 ----------
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
# cmap     = cm.batlow_r             # ← 反向顺序色图
cmap = cm.oslo_r          # ❶ 纯冷色系，感知均匀
# cmap = cm.lapaz_r
interp   = 'bilinear'

im = ax.imshow(
        fpt_plot,
        extent=[sigma_vals.min(), sigma_vals.max(),
                alpha_vals.min(), alpha_vals.max()],
        origin='lower', aspect='auto',
        cmap=cmap, interpolation=interp,
        vmin=fpt_mat.min(), vmax=fpt_mat.max())

# ---------- 5. 等高线美化 ----------
if CONTOUR_STYLE.upper() == "A":
    cs = ax.contour(
            fpt_plot, levels=[THRESHOLD],
            extent=[sigma_vals.min(), sigma_vals.max(),
                    alpha_vals.min(), alpha_vals.max()],
            colors='white', linewidths=1.2)
    ax.clabel(cs, fmt=LABEL_FMT, fontsize=12, inline=True,
              inline_spacing=4, use_clabeltext=True)

elif CONTOUR_STYLE.upper() == "B":
    base = cmap(np.linspace(0.65, 1.0, 128))
    darker = np.clip(base * [0.85, 0.85, 0.85, 1], 0, 1)
    fill_cmap = ListedColormap(darker)
    cf = ax.contourf(
            fpt_plot,
            levels=np.linspace(THRESHOLD, fpt_mat.max(), NUM_LEVELS),
            extent=[sigma_vals.min(), sigma_vals.max(),
                    alpha_vals.min(), alpha_vals.max()],
            cmap=fill_cmap, alpha=0.35, antialiased=True)
    cs = ax.contour(
            fpt_plot, levels=[THRESHOLD],
            extent=[sigma_vals.min(), sigma_vals.max(),
                    alpha_vals.min(), alpha_vals.max()],
            colors='black', linewidths=0.8)
    ax.clabel(cs, fmt=LABEL_FMT, fontsize=12, inline=True)

# ---------- 6. 颜色条 & 轴 ----------
cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.04)
cbar.set_label("First Passage Time", fontsize=16)
cbar.ax.tick_params(length=3, width=0.6)

ax.set_xlabel(r'$\sigma$')
ax.set_ylabel(r'$\alpha$')
ax.xaxis.set_major_locator(MultipleLocator(0.2))
ax.yaxis.set_major_locator(MultipleLocator(0.2))
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _ : f"{x:.1f}"))

# ---------- 7. 保存 ----------
plt.tight_layout()
fname = "first_passage/fpt_1.1_2.0_0.1_0.9.pdf"
plt.savefig(fname, bbox_inches='tight')
plt.show()
print(f"Saved to {fname}")
