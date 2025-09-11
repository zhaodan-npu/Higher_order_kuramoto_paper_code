#!/usr/bin/env python3
# extract_noise_spikes_percentile.py

import os
import glob
import numpy as np

# ———— 配置区 ————
# 噪声文件所在目录：每个文件为两列带表头 (time, noise)
NOISE_DIR = "."
# 输出“噪声峰”所在目录
OUTPUT_DIR = "noise_spikes2"
# 百分位数阈值（例如 80 就表示第 80 百分位数）
PERCENTILE = 80
# 输出目录不存在则创建
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ————————————

# 1. 找到所有噪声文件（.txt）
pattern = os.path.join(NOISE_DIR, "*.txt")
noise_files = sorted(glob.glob(pattern))
if len(noise_files) == 0:
    print(f"[Error] 在目录 '{NOISE_DIR}' 中未找到任何 .txt 文件，请检查路径。")
    exit(1)

print(f"共找到 {len(noise_files)} 个噪声文件，准备计算全局阈值（第 {PERCENTILE} 百分位数）…")

# 2. 第一遍遍历：收集所有 noise 列的绝对值，用于计算百分位数
all_abs_values = []

for fn in noise_files:
    base = os.path.basename(fn)
    try:
        data = np.loadtxt(fn, skiprows=1)
    except Exception as e:
        print(f"  [Warning] 无法读取 '{base}'：{e}，将跳过此文件。")
        continue

    if data.ndim < 2 or data.shape[1] < 2:
        print(f"  [Warning] '{base}' 数据格式不符合“两列”要求，已跳过。")
        continue

    noise_col = data[:, 1]
    all_abs_values.append(np.abs(noise_col))

if len(all_abs_values) == 0:
    print("Error: 未能找到任何有效的噪声列。程序退出。")
    exit(1)

# 将所有路径的绝对值拼成一个大数组
all_abs_values = np.concatenate(all_abs_values)
# 计算全局第 PERCENTILE 百分位阈值
threshold = np.percentile(all_abs_values, PERCENTILE)
print(f"计算得到全局第 {PERCENTILE} 百分位数阈值：{threshold:.6e}\n")

# 3. 第二遍遍历：对每个文件提取噪声峰
for fn in noise_files:
    base = os.path.basename(fn)
    name, _ = os.path.splitext(base)
    print(f"→ 处理文件：{base}")

    try:
        data = np.loadtxt(fn, skiprows=1)
    except Exception as e:
        print(f"  [Warning] 无法读取 '{base}'：{e}，跳过。")
        continue

    if data.ndim < 2 or data.shape[1] < 2:
        print(f"  [Warning] '{base}' 格式不符合“两列”要求，已跳过。")
        continue

    # 第一列视为 time，第二列视为 noise
    times = data[:, 0]
    noise = data[:, 1]

    # 找到所有 |noise| ≥ threshold 的索引
    idx_spikes = np.where(np.abs(noise) >= threshold)[0]

    if idx_spikes.size == 0:
        print(f"  [Info] '{base}' 中无任何点超过阈值 {threshold:.6e}，跳过。")
        continue

    # 写入输出文件：两列“time \t noise”
    out_fn = os.path.join(OUTPUT_DIR, f"{name}_spikes.txt")
    with open(out_fn, "w") as f:
        f.write("time\tnoise\n")
        for i in idx_spikes:
            f.write(f"{times[i]:.6f}\t{noise[i]:.6f}\n")

    print(f"  [Done] 共提取 {idx_spikes.size} 个噪声峰，已保存：{out_fn}")

print("\n===== 全部文件处理完毕 =====")
