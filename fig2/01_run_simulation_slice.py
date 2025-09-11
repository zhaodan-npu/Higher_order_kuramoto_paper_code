#!/usr/bin/env python3
import numpy as np
import warnings
import argparse
from scipy.integrate import solve_ivp

# 1. 忽略数值溢出警告
warnings.filterwarnings("ignore", category=RuntimeWarning)

# 2. 参数网格（要和主脚本保持一致）
k1s = np.linspace(-1, 6.0, 100)
k2s = np.linspace(0, 10.0, 100)
r0s = np.linspace(0.0, 1.0, 200)

# 3. 时间区间与阈值
t_span = (0, 2000)
r_threshold = 0.8

# 4. 方程右侧
def drdt(t, r, k1, k2):
    rv = np.clip(r[0], 0.0, 1.0)
    return [-rv + 0.5*k1*(rv - rv**3) + 0.5*k2*(rv**3 - rv**5)]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k1_index", type=int, required=True,
                   help="index into k1s (0–99)")
    p.add_argument("--outdir",     type=str, default="results",
                   help="输出目录")
    args = p.parse_args()

    i = args.k1_index
    k1 = k1s[i]
    basin_slice = np.zeros((len(k2s), len(r0s)), dtype=int)

    for j, k2 in enumerate(k2s):
        for k, r0 in enumerate(r0s):
            sol = solve_ivp(
                drdt, t_span, [r0],
                args=(k1, k2),
                atol=1e-6, rtol=1e-6, max_step=0.5
            )
            basin_slice[j, k] = int(sol.y[0, -1] > r_threshold)

    # 保存第 i 片结果
    np.savez(
        f"{args.outdir}/basin_k1_{i:03d}.npz",
        k1_index=i, k1=k1,
        k2s=k2s, r0s=r0s,
        basin_slice=basin_slice
    )
    print(f"[Done] k1_index={i}, k1={k1:.3f}")

if __name__ == "__main__":
    main()
