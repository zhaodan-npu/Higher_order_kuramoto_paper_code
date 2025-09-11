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
    beta = 0
    threshold = 0.8
    num_samples = 100
    max_steps = 1000000
    dt = 0.01
    alpha = 1.6  # Fixed alpha
    sigma = 0.5  # Fixed sigma

    K1_vals = np.linspace(-1, 6, 50)
    K2_vals = np.linspace(0, 10, 50)
    fpt_results = np.zeros((len(K1_vals), len(K2_vals)))

    for i, K1 in enumerate(K1_vals):
        for j, K2 in enumerate(K2_vals):
            fpt = average_first_passage_time(K1, K2, N, num_samples, max_steps, dt, threshold,
                                             sigma, alpha, beta, seed=42)
            fpt_results[i, j] = fpt
            print(f"K1={K1:.2f}, K2={K2:.2f}, FPT={fpt:.3f}")

    np.savetxt("fpt_results.txt", fpt_results)

    plt.figure(figsize=(10, 8))
    plt.imshow(fpt_results,
               extent=[K2_vals.min(), K2_vals.max(), K1_vals.min(), K1_vals.max()],
               aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='First Passage Time')
    plt.xlabel("K2")
    plt.ylabel("K1")
    plt.title(f"Average First Passage Time (alpha={alpha}, sigma={sigma}, N={N})")
    plt.savefig("fpt_heatmap.pdf")
    # plt.show()