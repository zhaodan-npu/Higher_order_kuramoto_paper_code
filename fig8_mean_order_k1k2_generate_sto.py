#!/usr/bin/env python3
import os
import argparse
import numpy as np
from numba import njit, prange
from joblib import Parallel, delayed
from scipy.stats import levy_stable


@njit(fastmath=True)
def kuramoto_derivatives(theta, omega, K1, K2):
    N = len(theta)
    sin_th = np.sin(theta)
    cos_th = np.cos(theta)
    sum_sin = np.sum(sin_th)
    sum_cos = np.sum(cos_th)


    pairwise = (K1/N)*(sum_sin * cos_th - sum_cos * sin_th)


    sin2 = np.sin(2*theta)
    cos2 = np.cos(2*theta)
    sum_sin2 = np.sum(sin2)
    sum_cos2 = np.sum(cos2)
    triadic = (K2/(N**2)) * (
        sum_sin2 * (cos_th*sum_cos - sin_th*sum_sin)
        - sum_cos2 * (sin_th*sum_cos + cos_th*sum_sin)
    )

    return omega + pairwise + triadic

@njit(fastmath=True)
def rk4_step(theta, omega, K1, K2, dt, noise):
    k1 = kuramoto_derivatives(theta, omega, K1, K2)
    k2 = kuramoto_derivatives(theta + 0.5*dt*k1, omega, K1, K2)
    k3 = kuramoto_derivatives(theta + 0.5*dt*k2, omega, K1, K2)
    k4 = kuramoto_derivatives(theta + dt*k3,   omega, K1, K2)
    theta += dt*(k1 + 2*k2 + 2*k3 + k4)/6.0
    theta += noise
    theta %= 2*np.pi

@njit(fastmath=True)
def integrate_chunk(theta, omega, K1, K2, dt, noise_chunk):
    for n in range(noise_chunk.shape[0]):
        rk4_step(theta, omega, K1, K2, dt, noise_chunk[n])
    return theta

def single_sample_run(K1, K2, N, transient_steps, main_steps, dt, omega, sigma, alpha, beta):
    theta = np.random.uniform(0, 2*np.pi, N)

    def run_steps(num_steps):
        chunk = 10000
        n_chunks = num_steps // chunk
        rem      = num_steps % chunk
        for _ in range(n_chunks):
            noise = sigma * levy_stable.rvs(alpha, beta, size=(chunk, N)) * np.sqrt(dt)
            integrate_chunk(theta, omega, K1, K2, dt, noise)
        if rem:
            noise = sigma * levy_stable.rvs(alpha, beta, size=(rem, N)) * np.sqrt(dt)
            integrate_chunk(theta, omega, K1, K2, dt, noise)

    run_steps(transient_steps)
    run_steps(main_steps)

    return np.abs(np.mean(np.exp(1j * theta)))

def mean_global_order_parameter(K1, K2, N, num_samples,
                                 transient_steps, main_steps,
                                 dt, sigma, alpha, beta, seed, n_jobs):
    np.random.seed(seed)
    omega = np.random.standard_cauchy(N)
    results = Parallel(n_jobs=n_jobs)(
        delayed(single_sample_run)(
            K1, K2, N, transient_steps, main_steps, dt,
            omega, sigma, alpha, beta
        )
        for _ in range(num_samples)
    )
    return np.mean(results)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--k1_idx', type=int, required=True,
                   help='Index into K1 grid (0–99)')
    p.add_argument('--outdir', type=str, default='results',
                   help='Output directory')
    args = p.parse_args()


    K1_vals = np.linspace(-1, 6.0, 100)
    K2_vals = np.linspace( 0, 10.0, 100)


    N               = 100
    num_samples     = 25
    dt              = 0.01
    transient_steps = 100_000
    main_steps      = 900_000
    sigma, alpha, beta = 0.5, 2.0, 0.0

    # Slurm 环境下并行度
    n_jobs = int(os.environ.get('SLURM_CPUS_PER_TASK','1'))

    k1_idx = args.k1_idx
    K1 = K1_vals[k1_idx]

    # 为这个 K1 计算所有 K2
    results = np.zeros(len(K2_vals))
    for j, K2 in enumerate(K2_vals):
        seed = k1_idx * len(K2_vals) + j
        r_mean = mean_global_order_parameter(
            K1, K2, N, num_samples,
            transient_steps, main_steps,
            dt, sigma, alpha, beta,
            seed, n_jobs
        )
        results[j] = r_mean
        print(f"[K1 idx={k1_idx}] K2={K2:.2f} → r̄={r_mean:.4f}")


    os.makedirs(args.outdir, exist_ok=True)
    np.savez(
        f"{args.outdir}/order_k1_{k1_idx:03d}.npz",
        k1_idx=k1_idx,
        K1=K1,
        K2_vals=K2_vals,
        r_mean=results
    )
    print(f"[Done] k1_idx={k1_idx}, K1={K1:.3f}")
