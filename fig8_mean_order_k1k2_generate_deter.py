import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from joblib import Parallel, delayed

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
def rk4_step(theta, omega, K1, K2, dt):
    k1 = kuramoto_derivatives(theta, omega, K1, K2)
    k2 = kuramoto_derivatives(theta + 0.5 * dt * k1, omega, K1, K2)
    k3 = kuramoto_derivatives(theta + 0.5 * dt * k2, omega, K1, K2)
    k4 = kuramoto_derivatives(theta + dt * k3, omega, K1, K2)

    theta += dt * (k1 + 2*k2 + 2*k3 + k4) / 6
    theta %= 2 * np.pi

@njit(fastmath=True)
def integrate_chunk(theta, omega, K1, K2, dt, num_steps):
    for _ in range(num_steps):
        rk4_step(theta, omega, K1, K2, dt)
    return theta

def single_sample_run(K1, K2, N, transient_steps, main_steps, dt, omega):
    theta = np.random.uniform(0, 2*np.pi, N)

    theta = integrate_chunk(theta, omega, K1, K2, dt, transient_steps)
    theta = integrate_chunk(theta, omega, K1, K2, dt, main_steps)

    r_final = np.abs(np.mean(np.exp(1j * theta)))
    return r_final

def mean_global_order_parameter(K1, K2, N, num_samples, transient_steps, main_steps, dt, seed):
    np.random.seed(seed)
    omega = np.random.standard_cauchy(N)

    results = Parallel(n_jobs=2)(
        delayed(single_sample_run)(
            K1, K2, N, transient_steps, main_steps, dt, omega
        )
        for _ in range(num_samples)
    )
    return np.mean(results)

if __name__ == '__main__':
    N = 100
    alpha = 1.5  # Fixed alpha
    sigma = 0.3  # Fixed sigma
    beta = 0
    num_samples = 100
    dt = 0.01
    transient_steps = 100000
    main_steps = 900000

    K1_vals = np.linspace(-1, 6, 50)
    K2_vals = np.linspace(0, 10, 50)
    order_param_results = np.zeros((len(K1_vals), len(K2_vals)))

    for i, K1 in enumerate(K1_vals):
        for j, K2 in enumerate(K2_vals):
            mean_r = mean_global_order_parameter(
                K1, K2, N, num_samples, transient_steps, main_steps, dt, seed=42
            )
            order_param_results[i, j] = mean_r
            print(f"K1={K1:.2f}, K2={K2:.2f}, Mean Order Parameter={mean_r:.3f}")

    np.savetxt("order_param_results.txt", order_param_results)

    plt.figure(figsize=(10, 8))
    plt.imshow(order_param_results,
               extent=[K2_vals.min(), K2_vals.max(), K1_vals.min(), K1_vals.max()],
               aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='Mean Global Order Parameter')
    plt.xlabel("K2")
    plt.ylabel("K1")
    # plt.title(f"Mean Global Order Parameter (alpha={alpha}, sigma={sigma}, N={N})")
    plt.savefig("order_param_heatmap.pdf")
    # plt.show()