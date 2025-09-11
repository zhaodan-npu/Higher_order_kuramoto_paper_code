import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    "font.family": "Times New Roman",
    "axes.labelsize": 18,     # 横纵坐标
    "axes.titlesize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16
})

fig, (ax_t, ax_p) = plt.subplots(2, 1, figsize=(8, 6))



base = Path('1.4_0.4')
dt, start_idx, win_len, path_idx = 0.01, 200_000, 300_000, 1
time = (np.arange(start_idx, start_idx + win_len) * dt)


sigmaA, alphaA = 0.0, 1.4
ts_fileA = base / f"r_all_sigma{sigmaA:.1f}_alpha{alphaA:.1f}.txt"
x_fileA  = base / f"x_vals_sigma{sigmaA:.1f}_alpha{alphaA:.1f}.txt"
y_fileA  = base / f"y_vals_sigma{sigmaA:.1f}_alpha{alphaA:.1f}.txt"

r_seg_A  = np.loadtxt(ts_fileA)[path_idx, start_idx:start_idx+win_len]
x_A      = np.loadtxt(x_fileA)
y_A      = np.loadtxt(y_fileA)


sigmaB, alphaB = 0.4, 1.4
ts_fileB = base / f"r_all_sigma{sigmaB:.1f}_alpha{alphaB:.1f}.txt"
x_fileB  = base / f"x_vals_sigma{sigmaB:.1f}_alpha{alphaB:.1f}.txt"
y_fileB  = base / f"y_vals_sigma{sigmaB:.1f}_alpha{alphaB:.1f}.txt"

r_seg_B  = np.loadtxt(ts_fileB)[path_idx, start_idx:start_idx+win_len]
x_B      = np.loadtxt(x_fileB)
y_B      = np.loadtxt(y_fileB)


fig, (ax_t, ax_p) = plt.subplots(2, 1, figsize=(8, 6))



ax_t.plot(time, r_seg_A, lw=1.3, label=r'$\sigma=0,\ \alpha=1.4$')
ax_t.plot(time, r_seg_B, lw=1.3, label=r'$\sigma=0.4,\ \alpha=1.4$')
# ax_t.axhline(0.8, ls='--', color='gray', lw=0.8)
ax_t.set_ylabel(r'$r(t)$', fontsize=15)
ax_t.legend()


ax_p.plot(x_A, y_A, lw=1.8, label=r'$\sigma=0,\ \alpha=1.4$')
ax_p.plot(x_B, y_B, lw=1.8, label=r'$\sigma=0.4,\ \alpha=1.4$')
ax_p.set_xlabel(r'$r$', fontsize=15)
ax_p.set_ylabel(r'$\mathrm{PDF}$', fontsize=15)
ax_p.legend()

plt.tight_layout()
plt.savefig('read_time_series/1.4_0.4/time_pdf_beautified.pdf')
