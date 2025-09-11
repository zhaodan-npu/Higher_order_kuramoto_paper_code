#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import paramiko
import numpy as np
import matplotlib.pyplot as plt
import io
import getpass

# 配置区
alphas = np.linspace(1.1, 2.0, 10)
sigmas = np.linspace(0.1, 1.0, 100)
n_alpha, n_sigma = alphas.size, sigmas.size

HOST = **
USERNAME = **
KEY_PATH = **
REMOTE_DIR = **



mean_max_amp = np.full((n_alpha, n_sigma), np.nan)
mean_spike_count = np.full((n_alpha, n_sigma), np.nan)

# 建立 SSH & SFTP 连接
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
passphrase = getpass.getpass('Enter passphrase for private key: ')
pkey = paramiko.RSAKey.from_private_key_file(KEY_PATH, password=passphrase)
ssh.connect(HOST, username=USERNAME, pkey=pkey)
sftp = ssh.open_sftp()


def read_stats_file(alpha_idx, sigma_idx):
    pattern = f"stats_a{alpha_idx}_s{sigma_idx}_"
    files = [f for f in sftp.listdir(REMOTE_DIR) if f.startswith(pattern) and f.endswith(".txt")]

    if not files:
        return None, None

    all_max_amp = []
    all_spike_counts = []

    for fname in files:
        remote_path = f"{REMOTE_DIR}/{fname}"
        with sftp.open(remote_path, 'r') as f:
            data = np.loadtxt(io.StringIO(f.read().decode('utf-8')), skiprows=1)
            if data.ndim == 1:
                data = data[np.newaxis, :]
            max_amp = data[:, 1]
            spike_counts = data[:, 2]
            all_max_amp.append(max_amp)
            all_spike_counts.append(spike_counts)

    if all_max_amp:
        return np.concatenate(all_max_amp), np.concatenate(all_spike_counts)
    return None, None


# 主计算循环
for ai in range(n_alpha):
    for si in range(n_sigma):
        max_amp, spike_counts = read_stats_file(ai, si)
        if max_amp is not None:
            mask = spike_counts > 0
            if np.any(mask):
                mean_max_amp[ai, si] = max_amp[mask].mean()
                mean_spike_count[ai, si] = spike_counts[mask].mean()

# 关闭连接
sftp.close()
ssh.close()

# 保存结果
np.save("mean_max_amplitude.npy", mean_max_amp)
np.save("mean_spike_count.npy", mean_spike_count)

# 绘制热力图
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

im0 = axes[0].imshow(mean_max_amp, origin="lower", aspect="auto",
                     extent=[sigmas.min(), sigmas.max(), alphas.min(), alphas.max()], cmap="viridis")
fig.colorbar(im0, ax=axes[0], label="Mean Max Amplitude")
axes[0].set_xlabel(r"$\sigma$")
axes[0].set_ylabel(r"$\alpha$")
axes[0].set_title("Conditional Mean Max Amplitude")

im1 = axes[1].imshow(mean_spike_count, origin="lower", aspect="auto",
                     extent=[sigmas.min(), sigmas.max(), alphas.min(), alphas.max()], cmap="plasma")
fig.colorbar(im1, ax=axes[1], label="Mean Spike Count")
axes[1].set_xlabel(r"$\sigma$")
axes[1].set_ylabel(r"$\alpha$")
axes[1].set_title("Conditional Mean Spike Count")

plt.tight_layout()
plt.savefig("conditional_mean_heatmaps.pdf", dpi=300)
plt.show()

print("✅ 热力图已保存至 conditional_mean_heatmaps.pdf")