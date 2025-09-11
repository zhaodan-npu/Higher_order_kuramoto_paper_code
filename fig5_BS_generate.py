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
    theta += noise  # Add Lévy noise
    theta %= 2 * np.pi

@njit(fastmath=True)
def integrate_chunk(theta, omega, K1, K2, dt, noise_chunk):
    """

    noise_chunk.shape = (chunk_size, N)
    """
    for i in range(noise_chunk.shape[0]):
        rk4_step(theta, omega, K1, K2, dt, noise_chunk[i])
    return theta

def single_sample_run(K1, K2, N, steps, dt, threshold, omega, sigma, alpha, beta):

    theta = np.random.uniform(0, 2*np.pi, N)
    chunk_size = 10000
    n_chunks = steps // chunk_size
    remainder = steps % chunk_size

    for _ in range(n_chunks):

        noise_chunk = sigma * levy_stable.rvs(alpha, beta, size=(chunk_size, N)) * (dt**(1/alpha))

        theta = integrate_chunk(theta, omega, K1, K2, dt, noise_chunk)

    if remainder > 0:
        noise_chunk = sigma * levy_stable.rvs(alpha, beta, size=(remainder, N)) * (dt**(1/alpha))
        theta = integrate_chunk(theta, omega, K1, K2, dt, noise_chunk)

    r_final = np.abs(np.mean(np.exp(1j * theta)))
    return r_final >= threshold

def basin_stability(K1, K2, N, num_samples, steps, dt, threshold, sigma, alpha, beta, seed):
    np.random.seed(seed)

    omega = np.random.standard_cauchy(N)


    results = Parallel(n_jobs=2)(
        delayed(single_sample_run)(
            K1, K2, N, steps, dt, threshold, omega, sigma, alpha, beta
        )
        for _ in range(num_samples)
    )
    return np.mean(results)

if __name__ == '__main__':
    N = 100
    K2 = 8.0
    K1 = 0.8  # Fixed K1 value
    beta = 0  # Symmetric Lévy distribution
    threshold = 0.8
    num_samples = 50
    steps = 1000000
    dt = 0.01

    alpha_vals = np.arange(1.1, 2.05, 0.1)
    alpha_vals[-1] = 2.0
    sigma_vals = np.arange(0.1, 0.6, 0.05)
    stability_results = np.zeros((len(alpha_vals), len(sigma_vals)))

    for i, alpha in enumerate(alpha_vals):
        for j, sigma in enumerate(sigma_vals):
            stability = basin_stability(K1, K2, N, num_samples, steps, dt, threshold,
                                        sigma, alpha, beta, seed=42)
            stability_results[i, j] = stability
            print(f"alpha={alpha:.2f}, sigma={sigma:.2f}, Stability={stability:.3f}")

    np.savetxt("basin_stability_results.txt", stability_results)

    plt.figure(figsize=(10, 8))
    plt.imshow(stability_results,
               extent=[sigma_vals.min(), sigma_vals.max(), alpha_vals.min(), alpha_vals.max()],
               aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='Basin Stability')
    plt.xlabel("Sigma")
    plt.ylabel("Alpha")
    plt.title(f"Basin Stability (K1={K1}, K2={K2}, N={N})")
    plt.savefig("basin_stability_heatmap.png")
    # plt.show()
