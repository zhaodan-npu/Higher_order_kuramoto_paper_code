#!/usr/bin/env python3
"""
compute_basin_task.py

对 SLURM_ARRAY_TASK_ID 指定的 (K1,K2) 网格点计算 basin stability，
将结果保存在 basin_results/bs_i_j.npz。
"""
import os
import argparse
import numpy as np
from scipy.stats import levy_stable
from joblib import Parallel, delayed

# ——— 全局参数 ———
N         = 100        # 振子数量
alpha     = 1.6        # Lévy 指数
sigma     = 0.5        # Lévy 噪声强度
threshold = 0.8        # 有序阈值
n_samples = 100        # 每点蒙特卡洛样本数
T_sample  = 10000.0    # 单条轨迹总时长
dt        = 0.01       # 时间步长
steps     = int(T_sample / dt)

# 参数网格
K1_list = np.linspace(-1,  6, 50)
K2_list = np.linspace( 0, 10, 50)

out_dir = "basin_results"
os.makedirs(out_dir, exist_ok=True)

def simulate_r_series(K1, K2, omega, theta0):
    """
    Euler–Maruyama 模拟一次 r(t) 序列。
    """
    theta = theta0.copy()
    r_series = np.empty(steps)
    for i in range(steps):
        S = np.sin(theta).sum()
        C = np.cos(theta).sum()
        A = np.sin(2*theta).sum()
        B = np.cos(2*theta).sum()
        pair = (C * np.cos(theta) - S * np.sin(theta)) * (K1 / N)
        triad = (np.cos(theta)*(C*A - S*B)
               - np.sin(theta)*(C*B + S*A)) * (K2 / N**2)
        drift = omega + pair + triad
        noise = levy_stable.rvs(alpha, 0, size=N) * sigma * (dt**(1/alpha))
        theta = (theta + drift * dt + noise) % (2*np.pi)
        r_series[i] = np.sqrt(S*S + C*C) / N
    return r_series

def basin_stability_at(K1, K2, rng, n_jobs):
    """
    对 (K1,K2) 进行 n_samples 次独立模拟，统计 r̄>threshold 的比例。
    """
    omega = rng.standard_cauchy(N)
    def one_trial(seed):
        sub_rng = np.random.RandomState(seed)
        theta0 = sub_rng.uniform(0, 2*np.pi, size=N)
        r = simulate_r_series(K1, K2, omega, theta0)
        return (r.mean() > threshold)
    seeds = rng.randint(0, 2**30, size=n_samples)
    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(one_trial)(s) for s in seeds
    )
    return np.mean(results)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task_id', type=int, required=True,
                   help="SLURM_ARRAY_TASK_ID，从 0 到 2499")
    args = p.parse_args()

    # 计算对应的网格索引 i,j
    total = len(K1_list) * len(K2_list)
    if args.task_id < 0 or args.task_id >= total:
        raise ValueError(f"task_id 必须在 [0, {total}) 之间")
    i = args.task_id % len(K1_list)
    j = args.task_id // len(K1_list)
    K1 = K1_list[i]
    K2 = K2_list[j]

    # 并行参数
    n_jobs = int(os.environ.get('SLURM_CPUS_PER_TASK', '1'))
    rng    = np.random.RandomState(10000 + args.task_id)

    bs = basin_stability_at(K1, K2, rng, n_jobs)

    # 保存结果
    outpath = os.path.join(out_dir, f"bs_{i}_{j}.npz")
    np.savez(outpath, K1=K1, K2=K2, basin_stability=bs, i=i, j=j)
    print(f"[Done] task_id={args.task_id} (i={i}, j={j}), "
          f"K1={K1:.3f}, K2={K2:.3f} → BS={bs:.4f}")

if __name__ == '__main__':
    main()
