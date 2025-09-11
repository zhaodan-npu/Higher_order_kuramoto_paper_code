#!/usr/bin/env python3
import numpy as np
import argparse
import os
from numba import njit, prange
from scipy.stats import levy_stable

@njit(parallel=True)
def kuramoto_derivatives(theta, omega, K1, K2):
    R, N = theta.shape
    dtheta = np.empty_like(theta)
    for r in prange(R):
        C, S = np.sum(np.cos(theta[r])), np.sum(np.sin(theta[r]))
        for i in range(N):
            pairwise = (K1 / N) * np.sum(np.sin(theta[r] - theta[r, i]))
            A = np.sum(np.sin(2*theta[r] - theta[r, i]))
            B = np.sum(np.cos(2*theta[r] - theta[r, i]))
            high_order = (K2 / (N * N)) * (C * A - S * B)
            dtheta[r, i] = omega[r, i] + pairwise + high_order
    return dtheta

@njit
def mod_2pi(theta):
    theta %= 2*np.pi

@njit
def rk4_step(theta, omega, K1, K2, dt):
    k1 = kuramoto_derivatives(theta, omega, K1, K2)
    k2 = kuramoto_derivatives(theta + 0.5*dt*k1, omega, K1, K2)
    k3 = kuramoto_derivatives(theta + 0.5*dt*k2, omega, K1, K2)
    k4 = kuramoto_derivatives(theta + dt*k3, omega, K1, K2)
    theta += (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    mod_2pi(theta)

@njit(parallel=True)
def order_parameter(theta):
    R, N = theta.shape
    r_vals = np.empty(R)
    for r in prange(R):
        re, im = np.sum(np.cos(theta[r])), np.sum(np.sin(theta[r]))
        r_vals[r] = np.hypot(re/N, im/N)
    return r_vals

def simulate(theta0, omega, K1, K2, dt, T, sigma, alpha, T_trans):
    num_realizations, N = theta0.shape
    r_all = np.empty((num_realizations, T - T_trans))
    theta = theta0.copy()

    for _ in range(T_trans):
        rk4_step(theta, omega, K1, K2, dt)
        theta += sigma * levy_stable.rvs(alpha, 0, size=theta.shape) * dt**(1/alpha)
        mod_2pi(theta)

    for t in range(T - T_trans):
        rk4_step(theta, omega, K1, K2, dt)
        theta += sigma * levy_stable.rvs(alpha, 0, size=theta.shape) * dt**(1/alpha)
        mod_2pi(theta)
        r_all[:, t] = order_parameter(theta)

    return r_all

def extract_spike_sequences(r_all, threshold):
    seq_info = {}
    n_paths, L = r_all.shape
    for pi in range(n_paths):
        above = r_all[pi] > threshold
        segments, i = [], 0
        while i < L:
            if above[i]:
                start = i
                while i < L and above[i]:
                    i += 1
                segments.append((start, i))
            else:
                i += 1
        if segments:
            seq_info[pi] = segments
    return seq_info

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha_idx', type=int, required=True)
    parser.add_argument('--sigma_idx', type=int, required=True)
    parser.add_argument('--start_path', type=int, required=True)
    parser.add_argument('--end_path', type=int, required=True)
    parser.add_argument('--outdir', type=str, default='results_split')
    args = parser.parse_args()

    alphas = np.linspace(1.1, 2.0, 10)
    sigmas = np.linspace(0.1, 1.0, 100)
    alpha = alphas[args.alpha_idx]
    sigma = sigmas[args.sigma_idx]

    N = 100
    K1, K2, dt = 0.8, 8.0, 0.01
    T, T_trans, threshold = int(1e6), int(1e5), 0.8

    os.makedirs(args.outdir, exist_ok=True)
    np.random.seed(args.start_path)
    num_paths = args.end_path - args.start_path
    theta0 = 2*np.pi*np.random.rand(num_paths, N)
    omega = np.random.standard_cauchy((num_paths, N))

    r_all = simulate(theta0, omega, K1, K2, dt, T, sigma, alpha, T_trans)
    t_sampling = np.arange(T_trans, T) * dt
    seqs = extract_spike_sequences(r_all, threshold)

    stats_file = f"{args.outdir}/stats_a{args.alpha_idx}_s{args.sigma_idx}_paths{args.start_path}_{args.end_path}.txt"
    with open(stats_file, 'w') as f_stats:
        f_stats.write("path_idx\tmax_amplitude\tnum_spikes\n")
        for idx, path_idx in enumerate(range(args.start_path, args.end_path)):
            segments = seqs.get(idx, [])
            max_amp = max([r_all[idx, s:e].max() for s, e in segments], default=0.0)
            num_spikes = len(segments)
            f_stats.write(f"{path_idx}\t{max_amp:.6f}\t{num_spikes}\n")

            if segments:
                spike_file = f"{args.outdir}/spikes_a{args.alpha_idx}_s{args.sigma_idx}_p{path_idx}.txt"
                with open(spike_file, 'w') as f_spike:
                    f_spike.write("time\torder_parameter\n")
                    for s, e in segments:
                        data = np.column_stack((t_sampling[s:e], r_all[idx, s:e]))
                        np.savetxt(f_spike, data, fmt="%.6f", delimiter="\t")

    print(f"[Done] alpha={alpha:.3f}, sigma={sigma:.3f}, paths={args.start_path}-{args.end_path}")
