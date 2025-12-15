import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from joblib import Parallel, delayed
import time


# -----------------------------------------------------------------------------
# 1. Numba 加速的 Levy 噪声生成器 (Chambers-Mallows-Stuck 算法)
# -----------------------------------------------------------------------------
# Scipy 的 levy_stable.rvs 非常慢，无法在百万步循环中使用。
# 这里手动实现标准算法，速度快 100 倍以上。
@njit(fastmath=True)
def generate_levy_noise(alpha, sigma, dt, shape):
    """
    生成对称 Levy 稳定分布噪声 (beta=0, mu=0)。
    F = (sin(alpha * V) / (cos(V))^(1/alpha)) * ((cos((1-alpha)*V) / W)^((1-alpha)/alpha))
    """
    rows, cols = shape
    noise = np.empty((rows, cols), dtype=np.float64)

    scale_factor = sigma * (dt ** (1.0 / alpha))

    for i in range(rows):
        for j in range(cols):
            # V ~ Uniform(-pi/2, pi/2)
            V = np.random.uniform(-np.pi / 2, np.pi / 2)
            # W ~ Exponential(1)
            W = np.random.exponential(1.0)

            # 避免数值不稳定
            if abs(alpha - 1.0) < 1e-5:
                # alpha -> 1 (Cauchy)
                X = np.tan(V)
            else:
                term1 = np.sin(alpha * V) / (np.cos(V) ** (1.0 / alpha))
                term2 = (np.cos((1.0 - alpha) * V) / W) ** ((1.0 - alpha) / alpha)
                X = term1 * term2

            noise[i, j] = X * scale_factor

    return noise


# -----------------------------------------------------------------------------
# 2. 核心动力学 (Kuramoto)
# -----------------------------------------------------------------------------

@njit(fastmath=True)
def compute_derivatives(theta, omega, K1, K2, N):
    sin_th = np.sin(theta)
    cos_th = np.cos(theta)
    sum_sin = np.sum(sin_th)
    sum_cos = np.sum(cos_th)

    pairwise_term = (K1 / N) * (sum_sin * cos_th - sum_cos * sin_th)

    sin_2th = np.sin(2 * theta)
    cos_2th = np.cos(2 * theta)
    sum_sin2 = np.sum(sin_2th)
    sum_cos2 = np.sum(cos_2th)

    triadic_term = (K2 / N ** 2) * (
            sum_sin2 * (cos_th * sum_cos - sin_th * sum_sin)
            - sum_cos2 * (sin_th * sum_cos + cos_th * sum_sin)
    )

    return omega + pairwise_term + triadic_term


@njit(fastmath=True)
def run_simulation_chunk(theta, omega, K1, K2, dt, steps, alpha, sigma):
    """
    运行一段模拟。为了节省内存，分块生成噪声并更新。
    """
    chunk_size = 1000  # 每次生成 1000 步的噪声，避免内存爆炸
    N = len(theta)

    remaining_steps = steps

    while remaining_steps > 0:
        current_chunk = min(chunk_size, remaining_steps)

        # 现场生成噪声 (Fast!)
        noise_block = generate_levy_noise(alpha, sigma, dt, (current_chunk, N))

        for i in range(current_chunk):
            # RK4 Step
            k1 = compute_derivatives(theta, omega, K1, K2, N)
            k2 = compute_derivatives(theta + 0.5 * dt * k1, omega, K1, K2, N)
            k3 = compute_derivatives(theta + 0.5 * dt * k2, omega, K1, K2, N)
            k4 = compute_derivatives(theta + dt * k3, omega, K1, K2, N)

            theta += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

            # Add Levy Noise
            theta += noise_block[i]

            theta %= 2 * np.pi

        remaining_steps -= current_chunk

    return theta


# -----------------------------------------------------------------------------
# 3. 任务包装器 (每个 Grid Point 计算 basin stability)
# -----------------------------------------------------------------------------

def solve_grid_point(idx, alpha, sigma, K1, K2, N, num_samples, steps, dt, threshold):
    """
    计算单个 (alpha, sigma) 点的 Basin Stability。
    关键点：对每个 sample 生成独立的 omega。
    """
    success_count = 0

    # 这里不需要并行，因为外层已经是并行了。
    # 串行跑 num_samples 个样本，每个样本都在 Numba 里极速运行。
    for s in range(num_samples):
        # 【核心修改】每个样本使用独立的随机种子，生成独立的 omega
        # seed 策略：基于 idx (网格点ID) 和 s (样本ID) 组合
        unique_seed = (idx * 10000) + s
        np.random.seed(unique_seed)

        # 1. 随机生成 Omega (Standard Cauchy) -> 这里的 Omega 是随机的！
        omega = np.random.standard_cauchy(N)

        # 2. 随机生成初始 Theta (Uniform)
        theta = np.random.uniform(0, 2 * np.pi, N)

        # 3. 运行模拟
        # Numba 函数会自动处理内部循环
        final_theta = run_simulation_chunk(theta, omega, K1, K2, dt, steps, alpha, sigma)

        # 4. 判定同步
        r_final = np.abs(np.sum(np.exp(1j * final_theta))) / N
        if r_final >= threshold:
            success_count += 1

    stability = success_count / num_samples
    return idx, stability


# -----------------------------------------------------------------------------
# 4. 主程序
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    t0 = time.time()

    # 参数设置
    N = 100
    K1 = 0.8
    K2 = 8.0
    threshold = 0.8
    num_samples = 50  # 每个点跑 50 个不同的 Omega 组合
    steps = 1000000  # 100万步
    dt = 0.01

    # 网格
    alpha_vals = np.arange(1.1, 2.05, 0.1)
    alpha_vals[-1] = 2.0  # 修正浮点数尾数
    sigma_vals = np.arange(0.1, 0.6, 0.05)

    # 构建任务
    tasks = []
    grid_shape = (len(alpha_vals), len(sigma_vals))

    for i, alpha in enumerate(alpha_vals):
        for j, sigma in enumerate(sigma_vals):
            flat_idx = i * grid_shape[1] + j
            tasks.append((flat_idx, alpha, sigma))

    print(f"Computing Basin Stability Map.")
    print(f"Grid: {grid_shape}, Total Tasks: {len(tasks)}")
    print(f"Per Task: {num_samples} samples (Random Omega), {steps} steps.")

    # 并行计算
    # n_jobs=-1 使用所有核心。
    # 因为我们用 Numba 手写了噪声生成，去掉了 Scipy 开销，这里 CPU 利用率会非常高。
    results = Parallel(n_jobs=-1, verbose=1)(
        delayed(solve_grid_point)(
            idx, alpha, sigma, K1, K2, N, num_samples, steps, dt, threshold
        )
        for idx, alpha, sigma in tasks
    )

    # 整理结果
    bs_flat = np.zeros(len(tasks))
    for idx, val in results:
        bs_flat[idx] = val

    bs_matrix = bs_flat.reshape(grid_shape)

    # 保存与绘图
    np.savetxt("basin_stability_results.txt", bs_matrix)

    plt.figure(figsize=(10, 8))
    plt.imshow(bs_matrix,
               extent=[sigma_vals.min(), sigma_vals.max(), alpha_vals.min(), alpha_vals.max()],
               aspect='auto', origin='lower', cmap='viridis', vmin=0, vmax=1)
    plt.colorbar(label='Basin Stability')
    plt.xlabel("Sigma (Noise Intensity)")
    plt.ylabel("Alpha (Levy Index)")
    plt.title(f"Basin Stability (Random Omega, N={N})")
    plt.savefig("basin_stability_heatmap.pdf")

    print(f"Done. Total time: {time.time() - t0:.2f} s")