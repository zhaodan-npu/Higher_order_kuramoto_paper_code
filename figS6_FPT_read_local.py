import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# ---------- 1. 读入数据 ----------
data_file = "first_passage/fpt_1.1_2.0_0.1_0.9.txt"
fpt_mat = np.loadtxt(data_file)

alpha_vals = np.append(np.arange(1.1, 2.0, 0.1), 2.0)
sigma_vals = np.arange(0.1, 1.0, 0.1)

# ---------- 2. 平滑数据 ----------
USE_GAUSSIAN, SIGMA_SMOOTH = True, 0.5
fpt_plot = gaussian_filter(fpt_mat, SIGMA_SMOOTH) if USE_GAUSSIAN else fpt_mat

# ---------- 3. 设定用于提取等高线的阈值 ----------
THRESHOLD = 0.5 * fpt_mat.max()

# 提取等高线
cs = plt.contour(
        fpt_plot,
        levels=[THRESHOLD],
        extent=[sigma_vals.min(), sigma_vals.max(),
                alpha_vals.min(), alpha_vals.max()]
)

# ---------- 4. 提取并保存等高线数据 ----------
with open('fpt_contour_data.txt', 'w') as f:
    for i, seg in enumerate(cs.allsegs[0]):
        f.write(f"# FPT Contour Line {i+1}\n")
        sigma_line, alpha_line = seg[:, 0], seg[:, 1]
        for sigma, alpha in zip(sigma_line, alpha_line):
            f.write(f"{sigma:.6f}\t{alpha:.6f}\n")

plt.close()

print("✅ FPT等高线数据已保存至 fpt_contour_data.txt")
