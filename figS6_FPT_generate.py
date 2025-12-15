import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from joblib import Parallel, delayed
from scipy.stats import levy_stable

@njit(fastmath=True)
def kuramoto_derivatives(theta, omega, K1, K2):
    N = len(theta)
    sin_th = np.sin(theta)
    cos_th = np.cos(theta)
    sum_sin = np.sum(sin_th)
    sum_cos = np.sum(cos_th)

    pairwise_term = (K1 / N) * (sum_sin * cos_th - sum_cos * sin_th)

    sin_2th = np.sin(2 * theta)
    cos_2th = np.cos(2 * theta)
    sum_sin2 = np.sum(sin_2th)
    sum_cos2 = np.sum(cos_2th)
    triadic_term = (K2 / N**2) * (
        sum_sin2 * (cos_th * sum_cos - sin_th * sum_sin)
        - sum_cos2 * (sin_th * sum_cos + cos_th * sum_sin)
    )

    return omega + pairwise_term + triadic_term

@njit(fastmath=True)
def rk4_step(theta, omega, K1, K2, dt, noise):
    k1 = kuramoto_derivatives(theta, omega, K1, K2)
    k2 = kuramoto_derivatives(theta + 0.5 * dt * k1, omega, K1, K2)
    k3 = kuramoto_derivatives(theta + 0.5 * dt * k2, omega, K1, K2)
    k4 = kuramoto_derivatives(theta + dt * k3, omega, K1, K2)
    theta += dt * (k1 + 2*k2 + 2*k3 + k4) / 6
    theta += noise
    theta %= 2 * np.pi

@njit(fastmath=True)
def integrate_until_threshold(theta, omega, K1, K2, dt, noise_chunk, threshold):
    for i in range(noise_chunk.shape[0]):
        rk4_step(theta, omega, K1, K2, dt, noise_chunk[i])
        r = np.abs(np.mean(np.exp(1j * theta)))
        if r >= threshold:
            return i + 1  # Return the step at which the threshold is crossed
    return noise_chunk.shape[0]  # Return the chunk size if threshold is not crossed

def first_passage_time(K1, K2, N, max_steps, dt, threshold, omega, sigma, alpha, beta):
    if sigma <= 0:
        raise ValueError("The `sigma` parameter must be positive.")
    if not (0 < alpha <= 2):
        raise ValueError("The `alpha` parameter must be in the range (0, 2].")
    if not (-1 <= beta <= 1):
        raise ValueError("The `beta` parameter must be in the range [-1, 1].")

    theta = np.random.uniform(0, 2*np.pi, N)
    chunk_size = 10000
    n_chunks = max_steps // chunk_size
    remainder = max_steps % chunk_size

    total_steps = 0
    for _ in range(n_chunks):
        noise_chunk = sigma * levy_stable.rvs(alpha, beta, size=(chunk_size, N)) * (dt**(1/alpha))
        steps = integrate_until_threshold(theta, omega, K1, K2, dt, noise_chunk, threshold)
        total_steps += steps
        if steps < chunk_size:
            return total_steps * dt  # Return the time at which the threshold is crossed

    if remainder > 0:
        noise_chunk = sigma * levy_stable.rvs(alpha, beta, size=(remainder, N)) * (dt**(1/alpha))
        steps = integrate_until_threshold(theta, omega, K1, K2, dt, noise_chunk, threshold)
        total_steps += steps
        if steps < remainder:
            return total_steps * dt

    return total_steps * dt  # Return the maximum time if threshold is not crossed

def average_first_passage_time(K1, K2, N, num_samples, max_steps, dt, threshold, sigma, alpha, beta, seed):
    np.random.seed(seed)
    omega = np.random.standard_cauchy(N)

    fpt_results = Parallel(n_jobs=2)(
        delayed(first_passage_time)(
            K1, K2, N, max_steps, dt, threshold, omega, sigma, alpha, beta
        )
        for _ in range(num_samples)
    )
    return np.mean(fpt_results)

if __name__ == '__main__':
    N = 100
    K2 = 8.0
    K1 = 0.8
    beta = 0
    threshold = 0.8
    num_samples = 50
    max_steps = 100000
    dt = 0.01

    # 防止alpha数值超过2.0
    alpha_vals = np.arange(1.1, 2.05, 0.1)
    alpha_vals[-1] = 2.0  # 强制最后一个值精确等于2.0
    sigma_vals = np.arange(0.1, 0.6, 0.05)
    fpt_results = np.zeros((len(alpha_vals), len(sigma_vals)))

    for i, alpha in enumerate(alpha_vals):
        for j, sigma in enumerate(sigma_vals):
            # 为避免浮点误差，再次检查并截断alpha的值
            alpha = min(max(alpha, 0.01), 2.0)

            fpt = average_first_passage_time(K1, K2, N, num_samples, max_steps, dt, threshold,
                                             sigma, alpha, beta, seed=42)
            fpt_results[i, j] = fpt
            print(f"alpha={alpha:.2f}, sigma={sigma:.2f}, FPT={fpt:.3f}")

    np.savetxt("fpt_results.txt", fpt_results)

    plt.figure(figsize=(10, 8))
    plt.imshow(fpt_results,
               extent=[sigma_vals.min(), sigma_vals.max(), alpha_vals.min(), alpha_vals.max()],
               aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='First Passage Time')
    plt.xlabel("Sigma")
    plt.ylabel("Alpha")
    plt.title(f"Average First Passage Time (K1={K1}, K2={K2}, N={N})")
    plt.savefig("fpt_heatmap.pdf")
