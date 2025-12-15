#!/usr/bin/env python3
"""
Doubling test based on fig3_generate_data.py (deterministic case, sigma = 0).

在原代码基础上做的修改：
- 保留原有的 kuramoto_derivatives / rk4_step / simulate 结构不变；
- main() 里跑两次：
    (T, T_trans)  = (1e6, 1e5)
    (T', T_trans') = (2e6, 2e5)
  只看 sigma = 0 的确定性情形；
- 比较原始 run 的 <r> 与翻倍 run 最后同样长度窗口的 <r>，打印差值。
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
from scipy.stats import gaussian_kde, levy_stable

# ---- 保留你原来的画图字体设置（虽然这里主要是数值测试） ----
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["legend.fontsize"] = 12


@njit(parallel=True)
def kuramoto_derivatives(theta, omega, K1, K2):
    R, N = theta.shape
    dtheta = np.empty_like(theta)
    for r in prange(R):
        C = 0.0
        S = 0.0
        for j in range(N):
            C += np.cos(theta[r, j])
            S += np.sin(theta[r, j])
        for i in range(N):
            s_pair = 0.0
            for j in range(N):
                s_pair += np.sin(theta[r, j] - theta[r, i])
            pairwise = (K1 / N) * s_pair

            A = 0.0
            B = 0.0
            for j in range(N):
                X = 2 * theta[r, j] - theta[r, i]
                A += np.sin(X)
                B += np.cos(X)
            high_order = (K2 / (N * N)) * (C * A - S * B)

            dtheta[r, i] = omega[r, i] + pairwise + high_order
    return dtheta


@njit
def mod_2pi(theta):
    R, N = theta.shape
    for r in range(R):
        for i in range(N):
            theta[r, i] = theta[r, i] % (2 * np.pi)


@njit
def rk4_step(theta, omega, K1, K2, dt):
    R, N = theta.shape
    k1 = kuramoto_derivatives(theta, omega, K1, K2)

    theta_temp = np.empty_like(theta)
    for r in range(R):
        for i in range(N):
            theta_temp[r, i] = theta[r, i] + 0.5 * dt * k1[r, i]
    k2 = kuramoto_derivatives(theta_temp, omega, K1, K2)

    for r in range(R):
        for i in range(N):
            theta_temp[r, i] = theta[r, i] + 0.5 * dt * k2[r, i]
    k3 = kuramoto_derivatives(theta_temp, omega, K1, K2)

    for r in range(R):
        for i in range(N):
            theta_temp[r, i] = theta[r, i] + dt * k3[r, i]
    k4 = kuramoto_derivatives(theta_temp, omega, K1, K2)

    for r in range(R):
        for i in range(N):
            theta[r, i] = theta[r, i] + (dt / 6.0) * (
                k1[r, i] + 2 * k2[r, i] + 2 * k3[r, i] + k4[r, i]
            )
    mod_2pi(theta)


@njit(parallel=True)
def order_parameter(theta):
    R, N = theta.shape
    r_vals = np.empty(R)
    for r in prange(R):
        re = 0.0
        im = 0.0
        for i in range(N):
            re += np.cos(theta[r, i])
            im += np.sin(theta[r, i])
        r_vals[r] = np.sqrt((re / N) ** 2 + (im / N) ** 2)
    return r_vals


def simulate(theta0, omega, K1, K2, dt, T, sigma, alpha, T_trans):
    num_realizations, N = theta0.shape
    r_all = np.empty((num_realizations, T - T_trans))
    theta = theta0.copy()

    # transient
    for _ in range(T_trans):
        rk4_step(theta, omega, K1, K2, dt)
        if sigma != 0.0:
            noise = sigma * levy_stable.rvs(alpha, 0, size=theta.shape) * (
                dt ** (1 / alpha)
            )
            theta += noise
        mod_2pi(theta)

    # sampling
    for t in range(T - T_trans):
        rk4_step(theta, omega, K1, K2, dt)
        if sigma != 0.0:
            noise = sigma * levy_stable.rvs(alpha, 0, size=theta.shape) * (
                dt ** (1 / alpha)
            )
            theta += noise
        mod_2pi(theta)
        r_all[:, t] = order_parameter(theta)

    return r_all


def main():
    # ===== 参数（可以按你论文中 fig3 的设定改） =====
    N = 100
    num_realizations = 50
    K1 = 0.8
    K2 = 8.0
    dt = 0.01

    T1 = int(1e6)
    T_trans1 = int(1e5)

    alpha = 2
    sigma_det = 0.0  # 确定性情形

    # 翻倍
    T2 = 2 * T1
    T_trans2 = 2 * T_trans1

    # 固定 seed，保证两次用的是同一组 theta0, omega
    np.random.seed(0)
    theta_initial = 2 * np.pi * np.random.rand(num_realizations, N)
    omega_values = np.random.standard_cauchy((num_realizations, N))

    # ========== Run 1: 原始 (T1, T_trans1) ==========
    print("Running deterministic case with (T, T_trans) = (1e6, 1e5)...")
    theta_det1 = theta_initial.copy()
    omega_batch1 = omega_values.copy()
    r_all_det1 = simulate(theta_det1, omega_batch1, K1, K2, dt,
                          T1, sigma_det, alpha, T_trans1)
    mean_r1 = np.mean(r_all_det1)
    std_r1 = np.std(r_all_det1)

    # ========== Run 2: 翻倍 (T2, T_trans2) ==========
    print("Running deterministic case with (T, T_trans) = (2e6, 2e5)...")
    theta_det2 = theta_initial.copy()
    omega_batch2 = omega_values.copy()
    r_all_det2 = simulate(theta_det2, omega_batch2, K1, K2, dt,
                          T2, sigma_det, alpha, T_trans2)
    mean_r2_full = np.mean(r_all_det2)
    std_r2_full = np.std(r_all_det2)

    # ----- 比较：取 run2 末尾与 run1 等长的窗口 -----
    L1 = T1 - T_trans1
    L2 = T2 - T_trans2
    if L2 < L1:
        raise ValueError("L2 < L1, 请检查 T, T_trans 设定。")
    r_all_det2_last = r_all_det2[:, L2 - L1:]

    mean_r2_last = np.mean(r_all_det2_last)
    std_r2_last = np.std(r_all_det2_last)

    print("\n===== Doubling test (deterministic, sigma = 0) =====")
    print(f"Run 1:  T = {T1},  T_trans = {T_trans1}")
    print(f"  <r>_1 (full sampling window) = {mean_r1:.6f} ± {std_r1:.6f}")
    print(f"Run 2:  T = {T2},  T_trans = {T_trans2}")
    print(f"  <r>_2_full (full sampling)   = {mean_r2_full:.6f} ± {std_r2_full:.6f}")
    print(
        f"  <r>_2_last (last window, length = T1 - T_trans1) "
        f"= {mean_r2_last:.6f} ± {std_r2_last:.6f}"
    )

    diff = abs(mean_r1 - mean_r2_last)
    rel_diff = diff / max(abs(mean_r1), 1e-12)
    print("\nAbsolute difference between <r>_1 and <r>_2_last:")
    print(f"  |<r>_1 - <r>_2_last| = {diff:.6e}")
    print("Relative difference:")
    print(
        "  |<r>_1 - <r>_2_last| / max(|<r>_1|, 1e-12) "
        f"= {rel_diff:.6e}"
    )

    # 可选：保存两次的 r_all 用于后处理
    np.savetxt("r_det_T1.txt", r_all_det1.reshape(num_realizations, -1))
    np.savetxt("r_det_T2_last_window.txt", r_all_det2_last.reshape(num_realizations, -1))


if __name__ == "__main__":
    main()
