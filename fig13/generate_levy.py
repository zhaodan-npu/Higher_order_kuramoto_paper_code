#!/usr/bin/env python3
# generate_noise_fixed.py

import numpy as np
import os
from scipy.stats import levy_stable

"""
generate_noise_fixed.py

为每条路径生成固定 α=1.1、σ=0.1 的 Lévy 噪声时间序列，时间尺度与 Kuramoto 模型一致。

输出到指定目录，每个文件包含两列：time    noise_value。

使用方法：
    python generate_noise_fixed.py --outdir <output_folder>
"""

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--outdir', type=str, default='noise_fixed',
        help='输出文件夹'
    )
    args = parser.parse_args()

    # 固定参数
    alpha = 1.1
    sigma = 0.1

    # 与原模拟一致的配置
    num_realizations = 25     # 路径数
    dt = 0.01
    T_total = int(1e6)        # 总步数
    T_trans = int(1e5)        # 转瞬期步数
    L = T_total - T_trans     # 采样步数

    # 时间向量（对应采样阶段），单位秒
    t_sampling = (np.arange(T_trans, T_total) * dt).astype(np.float64)

    # 创建输出目录
    os.makedirs(args.outdir, exist_ok=True)

    print(f"[Info] 生成 Lévy 噪声：α={alpha}, σ={sigma}")
    print(f"[Info] 共 {num_realizations} 条路径，每条路径生成 {L} 个采样点")

    # 生成 Lévy 噪声：shape = (num_realizations, L)
    scale = sigma * (dt ** (1.0 / alpha))
    noise_all = levy_stable.rvs(alpha, 0.0, size=(num_realizations, L)) * scale

    # 保存每条路径
    for path_idx in range(num_realizations):
        fname = f"noise_a1.1_s0.1_p{path_idx}.txt"
        outpath = os.path.join(args.outdir, fname)
        with open(outpath, 'w') as f:
            f.write("time\tnoise\n")
            data = np.vstack((t_sampling, noise_all[path_idx])).T
            np.savetxt(f, data, fmt=['%.6f', '%.6e'], delimiter="\t")
        print(f"[Saved] {outpath}")

    print(f"[Done] 共保存 {num_realizations} 个噪声文件 到 '{args.outdir}'")

if __name__ == "__main__":
    main()
