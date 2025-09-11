#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.ndimage import gaussian_filter
import cmcrameri.cm as cm
from matplotlib import cm as mpl_cm
from matplotlib.colors import LinearSegmentedColormap


plt.rcParams.update({
    "font.family":     "Times New Roman",
    "axes.labelsize":  18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "axes.linewidth":  0.8,
    "figure.dpi":      300
})


mean_max_amp = np.load("mean_max_amplitude.npy")
mean_spike_count = np.load("mean_spike_count.npy")


alphas = np.linspace(1.1, 2.0, 10)
sigmas = np.linspace(0.1, 1.0, 100)
X, Y = np.meshgrid(sigmas, alphas)

def smooth_masked(arr, sigma=0.6, eps=1e-12):
    arr = np.asarray(arr, dtype=float)
    valid = np.isfinite(arr)
    arr0  = np.where(valid, arr, 0.0)
    num = gaussian_filter(arr0, sigma=sigma, mode='constant', cval=0.0)
    den = gaussian_filter(valid.astype(float), sigma=sigma, mode='constant', cval=0.0)
    out = num / np.maximum(den, eps)
    out[den < eps] = np.nan
    return np.ma.masked_invalid(out)


mean_max_amp_smooth    = smooth_masked(mean_max_amp)
mean_spike_count_smooth = smooth_masked(mean_spike_count)


cmap_amp = cm.batlow.copy()
cmap_amp.set_bad('#d0d0d0')
levels_amp = np.linspace(
    np.nanpercentile(mean_max_amp_smooth.compressed(), 2),
    np.nanpercentile(mean_max_amp_smooth.compressed(), 98),
    12
)

base = mpl_cm.get_cmap('viridis')
cmap_cnt = LinearSegmentedColormap.from_list('viridis_trunc', base(np.linspace(0.15, 0.85, 256)))
cmap_cnt.set_bad('#f2f2f2')
levels_cnt = np.linspace(
    np.nanmin(mean_spike_count_smooth),
    np.nanmax(mean_spike_count_smooth),
    20
)


fig, ax = plt.subplots(figsize=(7,5))
pc = ax.contourf(X, Y, mean_max_amp_smooth, levels=levels_amp, cmap=cmap_amp, extend="both")
cbar = fig.colorbar(pc, ax=ax, pad=0.02, fraction=0.05, format="%.3f", extend='both', extendrect=True)
cbar.set_label(r'$R_{\mathrm{max}}(\alpha,\sigma)$', size=16)
ax.contour(X, Y, mean_max_amp_smooth, levels=levels_amp[::2], colors='k', linewidths=0.1, alpha=0.6)
ax.set_xlabel(r'$\sigma$'); ax.set_ylabel(r'$\alpha$')
ax.xaxis.set_major_locator(MaxNLocator(4)); ax.yaxis.set_major_locator(MaxNLocator(4))
# ax.text(-0.15, 1.05, '(a)', transform=ax.transAxes, fontsize=20, fontweight='bold', va='top', ha='right')
plt.tight_layout()
plt.savefig("fig_a_amp_heatmap.pdf", bbox_inches="tight")
plt.close(fig)


fig, ax = plt.subplots(figsize=(7,5))
pc = ax.contourf(X, Y, mean_spike_count_smooth, levels=levels_cnt, cmap=cmap_cnt, extend="both")
cbar = fig.colorbar(pc, ax=ax, pad=0.02, fraction=0.05, format="%.3f", extend='both', extendrect=True)
cbar.set_label(r'$N_{\mathrm{spikes}}(\alpha,\sigma)$', size=16)
ax.contour(X, Y, mean_spike_count_smooth, levels=levels_cnt[::2], colors='k', linewidths=0.1, alpha=0.6)
ax.set_xlabel(r'$\sigma$'); ax.set_ylabel(r'$\alpha$')
ax.xaxis.set_major_locator(MaxNLocator(4)); ax.yaxis.set_major_locator(MaxNLocator(4))
# ax.text(-0.15, 1.05, '(b)', transform=ax.transAxes, fontsize=20, fontweight='bold', va='top', ha='right')
plt.tight_layout()
plt.savefig("fig_b_spike_heatmap.pdf", bbox_inches="tight")
plt.close(fig)


selected_sigmas = [0.10, 0.25, 0.50]
selected_alphas = [1.1, 1.5, 2.0]


fig, ax = plt.subplots(figsize=(7,5))
for alpha_value in selected_alphas:
    alpha_idx = np.argmin(np.abs(alphas - alpha_value))
    amp_slice = mean_max_amp_smooth[alpha_idx, :]
    ax.plot(sigmas, amp_slice, marker='o', label=fr'$\alpha = {alphas[alpha_idx]:.2f}$')
ax.set_xlabel(r'$\sigma$'); ax.set_ylabel(r'$R_{\mathrm{max}}(\alpha,\sigma)$')
ax.legend(fontsize=14); ax.grid(True, linestyle='--', alpha=0.6)
# ax.text(-0.15, 1.05, '(c)', transform=ax.transAxes, fontsize=20, fontweight='bold', va='top', ha='right')
plt.tight_layout()
plt.savefig("fig_c_amp_vs_sigma.pdf", bbox_inches="tight")
plt.close(fig)


fig, ax = plt.subplots(figsize=(7,5))
for sigma_value in selected_sigmas:
    sigma_idx = np.argmin(np.abs(sigmas - sigma_value))
    amp_slice = mean_max_amp_smooth[:, sigma_idx]
    ax.plot(alphas, amp_slice, marker='o', label=fr'$\sigma = {sigmas[sigma_idx]:.2f}$')
ax.set_xlabel(r'$\alpha$'); ax.set_ylabel(r'$R_{\mathrm{max}}(\alpha,\sigma)$')
ax.legend(fontsize=14); ax.grid(True, linestyle='--', alpha=0.6)
# ax.text(-0.15, 1.05, '(d)', transform=ax.transAxes, fontsize=20, fontweight='bold', va='top', ha='right')
plt.tight_layout()
plt.savefig("fig_d_amp_vs_alpha.pdf", bbox_inches="tight")
plt.close(fig)


fig, ax = plt.subplots(figsize=(7,5))
for alpha_value in selected_alphas:
    alpha_idx = np.argmin(np.abs(alphas - alpha_value))
    spike_slice = mean_spike_count_smooth[alpha_idx, :]
    ax.plot(sigmas, spike_slice, marker='o', label=fr'$\alpha = {alphas[alpha_idx]:.2f}$')
ax.set_xlabel(r'$\sigma$'); ax.set_ylabel(r'$N_{\mathrm{spikes}}(\alpha,\sigma)$')
ax.legend(fontsize=14); ax.grid(True, linestyle='--', alpha=0.6)
# ax.text(-0.15, 1.05, '(e)', transform=ax.transAxes, fontsize=20, fontweight='bold', va='top', ha='right')
plt.tight_layout()
plt.savefig("fig_e_spike_vs_sigma.pdf", bbox_inches="tight")
plt.close(fig)


fig, ax = plt.subplots(figsize=(7,5))
for sigma_value in selected_sigmas:
    sigma_idx = np.argmin(np.abs(sigmas - sigma_value))
    spike_slice = mean_spike_count_smooth[:, sigma_idx]
    ax.plot(alphas, spike_slice, marker='o', label=fr'$\sigma = {sigmas[sigma_idx]:.2f}$')
ax.set_xlabel(r'$\alpha$'); ax.set_ylabel(r'$N_{\mathrm{spikes}}(\alpha,\sigma)$')
ax.legend(fontsize=14); ax.grid(True, linestyle='--', alpha=0.6)
# ax.text(-0.15, 1.05, '(f)', transform=ax.transAxes, fontsize=20, fontweight='bold', va='top', ha='right')
plt.tight_layout()
plt.savefig("fig_f_spike_vs_alpha.pdf", bbox_inches="tight")
plt.close(fig)

print("✅ Saved separate PDFs: fig_a ... fig_f")
