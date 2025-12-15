#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np

# ==== 参数网格：必须和你模拟脚本中一致 ====
# 注意：这里我用的是你 fig_ab 里给的网格
alphas = np.linspace(1.1, 2.0, 10)
sigmas = np.linspace(0.1, 1.0, 100)   # 确认你的 D_idx / sigma_idx 真的是 0..99 这一种
n_alpha, n_sigma = alphas.size, sigmas.size

# ==== 存放 txt 的目录 ====
DATA_DIR = "./results_constantD"   # 如果你 txt 在别的目录，这里改一下

# 结果矩阵：按 (alpha_idx, sigma_idx) 存放“有 spike 条件下的平均”
mean_max_amp     = np.full((n_alpha, n_sigma), np.nan)
mean_spike_count = np.full((n_alpha, n_sigma), np.nan)


def read_stats_one_point(alpha_idx, sigma_idx):
    """
    读取某个 (alpha_idx, sigma_idx) 对应的所有 stats 文件，
    拼接所有路径的 max_amplitude 和 num_spikes。

    文件名假定为：
    stats_a{alpha_idx}_D{sigma_idx}_paths{start}_{end}.txt
    """
    pattern = f"stats_a{alpha_idx}_D{sigma_idx}_"
    files = [
        f for f in os.listdir(DATA_DIR)
        if f.startswith(pattern) and f.endswith(".txt")
    ]

    if not files:
        # 这个 (alpha, sigma) 根本没跑过
        return None, None

    all_max_amp = []
    all_spike_counts = []

    for fname in files:
        path = os.path.join(DATA_DIR, fname)
        try:
            # 你的 txt 前两行是表头
            data = np.loadtxt(path, skiprows=2)
        except Exception as e:
            print(f"[警告] 读取失败，跳过 {path}: {e}")
            continue

        # 空文件：只有表头没有任何路径数据
        if data.size == 0:
            print(f"[警告] {path} 是空文件（没有数据），已跳过")
            continue

        # 只有一行数据时 -> 变成二维
        if data.ndim == 1:
            data = data[np.newaxis, :]

        # 防御一下列数不足
        if data.shape[1] < 3:
            print(f"[警告] {path} 列数不足，shape={data.shape}，已跳过")
            continue

        # 第 2 列: max_amplitude, 第 3 列: num_spikes
        all_max_amp.append(data[:, 1])
        all_spike_counts.append(data[:, 2])

    if not all_max_amp:
        # 有文件但全是空的 / 坏的
        return None, None

    return np.concatenate(all_max_amp), np.concatenate(all_spike_counts)


# ==== 主循环：对整个 (alpha, sigma) 网格做统计 ====
for ai in range(n_alpha):
    for si in range(n_sigma):
        max_amp, spike_counts = read_stats_one_point(ai, si)

        # 没有任何数据：保持 NaN
        if max_amp is None:
            continue

        # 只对“真的有 spike 的路径”做条件平均
        mask = (spike_counts > 0)
        if np.any(mask):
            mean_max_amp[ai, si]     = max_amp[mask].mean()
            mean_spike_count[ai, si] = spike_counts[mask].mean()
        else:
            # 所有路径都没 spike：这个点保持 NaN
            pass

# ==== 保存结果 ====
np.save("mean_max_amplitude.npy", mean_max_amp)
np.save("mean_spike_count.npy",   mean_spike_count)

print("✅ 已保存 mean_max_amplitude.npy / mean_spike_count.npy")
print("   现在可以直接运行你给的 fig_ab 那个画图脚本，颜色 & 平滑 & 无 spike 空白都自动对齐。")
