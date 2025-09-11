#!/usr/bin/env python3
"""
plot_p1_average.py — Compute and plot for p1 fixed-length windows:
    (a) Average EDACF
    (b) Average EDSPEC with average confidence level
    (c) Aggregated event times (y fixed to 1)
"""

import glob
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# ------ Configuration ------
# Path identifier p (fixed to 1)
path_id = 1

# If EDACF/EDSPEC results are in a subdirectory, use:
ed_dir = "."
# If results are in the current directory, set ed_dir to empty string:
# ed_dir = ""

# ---------- Global Styling ----------
plt.rcParams.update({
    "font.family": "Times New Roman",
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.figsize": (6, 4),
    "grid.linestyle": "--",
})

prefix = ed_dir.rstrip("/") + "/" if ed_dir else ""

# ---------- 1) Collect EDACF files and data ----------
edacf_pattern = f"{prefix}spikes_a0_s0_p{path_id}_win*_EDACF.txt"
edacf_files = sorted(glob.glob(edacf_pattern))
if not edacf_files:
    print(f"Error: No files found matching: {edacf_pattern}", file=sys.stderr)
    sys.exit(1)

lags_list = []
C_list    = []

for fn in edacf_files:
    try:
        data = np.loadtxt(fn, unpack=True)
    except Exception as e:
        print(f"Warning: Failed reading EDACF file, skipping: {fn} ({e})", file=sys.stderr)
        continue
    lags_i, C_i = data[0], data[1]
    lags_list.append(lags_i)
    C_list.append(C_i)

if not C_list:
    print("Error: No valid EDACF files for averaging.", file=sys.stderr)
    sys.exit(1)

min_len_lags = min(arr.size for arr in lags_list)
lags_ref = lags_list[0][:min_len_lags]
C_trimmed = np.vstack([c[:min_len_lags] for c in C_list])
C_mean = C_trimmed.mean(axis=0)
C_sem = C_trimmed.std(axis=0, ddof=1) / np.sqrt(C_trimmed.shape[0])

# ---------- 2) Collect EDSPEC files and data ----------
edspec_pattern = f"{prefix}spikes_a0_s0_p{path_id}_win*_EDSPEC.txt"
edspec_files = sorted(glob.glob(edspec_pattern))
if not edspec_files:
    print(f"Error: No files found matching: {edspec_pattern}", file=sys.stderr)
    sys.exit(1)

freqs_list = []
P_list     = []

for fn in edspec_files:
    try:
        data = np.loadtxt(fn, unpack=True)
    except Exception as e:
        print(f"Warning: Failed reading EDSPEC file, skipping: {fn} ({e})", file=sys.stderr)
        continue
    f_i, P_i = data[0], data[1]
    freqs_list.append(f_i)
    P_list.append(P_i)

if not P_list:
    print("Error: No valid EDSPEC files for averaging.", file=sys.stderr)
    sys.exit(1)

min_len_freqs = min(arr.size for arr in freqs_list)
freqs_ref = freqs_list[0][:min_len_freqs]
P_trimmed = np.vstack([p[:min_len_freqs] for p in P_list])
P_mean = P_trimmed.mean(axis=0)
P_sem = P_trimmed.std(axis=0, ddof=1) / np.sqrt(P_trimmed.shape[0])


# ---------- 2.5) Collect CONF files and data ----------
conf_pattern = f"{prefix}spikes_a0_s0_p{path_id}_win*_CONF.txt"
conf_files = sorted(glob.glob(conf_pattern))
SI_mean = None # Default value
fconf_ref = None

if conf_files:
    fconf_list = []
    SI_list = []
    for fn in conf_files:
        try:
            data = np.loadtxt(fn, unpack=True)
        except Exception as e:
            print(f"Warning: Failed reading CONF file, skipping: {fn} ({e})", file=sys.stderr)
            continue
        fconf_i, SI_i = data[0], data[1]
        fconf_list.append(fconf_i)
        SI_list.append(SI_i)

    if SI_list:
        min_len_si = min(arr.size for arr in SI_list)
        fconf_ref = fconf_list[0][:min_len_si]
        SI_trimmed = np.vstack([si[:min_len_si] for si in SI_list])
        SI_mean = SI_trimmed.mean(axis=0)
else:
    print("Warning: No CONF files found. Skipping confidence level plot.")


# ---------- 3) Collect event time files ----------
spike_pattern = f"spikes_a0_s0_p{path_id}_win*.txt"
spike_files = sorted(glob.glob(spike_pattern))
if not spike_files:
    print(f"Error: No files found matching: {spike_pattern}", file=sys.stderr)
    sys.exit(1)

all_times = []
for fn in spike_files:
    try:
        data = np.loadtxt(fn, unpack=True, skiprows=1)
    except Exception as e:
        print(f"Warning: Failed reading event time file, skipping: {fn} ({e})", file=sys.stderr)
        continue
    all_times.append(data if data.ndim == 1 else data[0])

if all_times:
    all_times = np.concatenate(all_times)
else:
    all_times = np.array([])


# ---------- 4) Plot and save separate figures ----------

# (a) Plot average EDACF
plt.figure()
plt.plot(lags_ref, C_mean, 'b-', linewidth=2, label="Average EDACF")
plt.fill_between(lags_ref, C_mean - C_sem, C_mean + C_sem,
                 color='blue', alpha=0.3, label="SEM")
plt.xlabel("Lag(dimensionless)")
plt.ylabel("EDACF")
plt.grid(True)
plt.legend()
outfn_edacf = f"p{path_id}_windows_average_EDACF.pdf"
plt.tight_layout()
plt.savefig(outfn_edacf, dpi=300)
plt.close()

# (b) Plot average EDSPEC
plt.figure()
plt.loglog(freqs_ref, P_mean, 'r-', linewidth=2, label="Average EDSPEC")
lower = np.maximum(P_mean - P_sem, 1e-30) # Prevent negative values in log scale
upper = P_mean + P_sem
plt.fill_between(freqs_ref, lower, upper, color='red', alpha=0.3, label="SEM")

# --- 添加置信度曲线 ---
if SI_mean is not None and fconf_ref is not None:
    plt.plot(fconf_ref, SI_mean, 'k--', linewidth=1.5, label='Average 95% CONF')

plt.xlabel("Frequency")
plt.ylabel("Power")
plt.grid(True, which="both")
plt.legend()
outfn_edspec = f"p{path_id}_windows_average_EDSPEC.pdf"
plt.tight_layout()
plt.savefig(outfn_edspec, dpi=300)
plt.close()

# (c) Plot aggregated event times
plt.figure()
if all_times.size > 0:
    plt.scatter(all_times, np.ones_like(all_times),
                s=10, marker="|", color="C2", alpha=0.7,
                rasterized=True)   # 减小PDF体积
    plt.xlim(2000, 2050)           # 固定横坐标范围
plt.xlabel("Time")
plt.yticks([])
plt.grid(False)
outfn_events = f"p{path_id}_windows_average_event_times.pdf"
plt.tight_layout()
plt.savefig(outfn_events, dpi=300)
plt.close()

print(f"Saved EDACF figure to {outfn_edacf}")
print(f"Saved EDSPEC figure to {outfn_edspec}")
print(f"Saved Event Times figure to {outfn_events}")
