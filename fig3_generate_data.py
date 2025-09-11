import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
from scipy.stats import gaussian_kde, levy_stable

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
            theta[r, i] = theta[r, i] + (dt / 6.0) * (k1[r, i] + 2 * k2[r, i] + 2 * k3[r, i] + k4[r, i])
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

    # Run the transient period
    for _ in range(T_trans):
        rk4_step(theta, omega, K1, K2, dt)
        noise = sigma * levy_stable.rvs(alpha, 0, size=theta.shape) * (dt**(1/alpha))
        theta += noise
        mod_2pi(theta)

    # Run the sampling period
    for t in range(T - T_trans):
        rk4_step(theta, omega, K1, K2, dt)
        noise = sigma * levy_stable.rvs(alpha, 0, size=theta.shape) *  (dt**(1/alpha))
        mod_2pi(theta)
        r_all[:, t] = order_parameter(theta)

    return r_all

def main():
    N = 100
    num_realizations = 10
    K1 = 0.8
    K2 = 8.0
    dt = 0.01
    T = int(1e6)
    T_trans = int(1e5)  # Transient period
    sigma_det = 0.0
    sigma_rand = 0.5
    alpha = 1.2  # Lévy noise parameter

    theta_initial = 2 * np.pi * np.random.rand(num_realizations, N)
    omega_values = np.random.standard_cauchy((num_realizations, N))

    theta_det = theta_initial.copy()
    theta_rand = theta_initial.copy()
    omega_batch = omega_values.copy()

    print("Simulating deterministic case (sigma=0)...")
    r_all_det = simulate(theta_det, omega_batch, K1, K2, dt, T, sigma_det, alpha, T_trans)

    print("Simulating random case (sigma=0.1)...")
    r_all_rand = simulate(theta_rand, omega_batch, K1, K2, dt, T, sigma_rand, alpha, T_trans)

    time = np.arange(T - T_trans) * dt
    avg_r_det = np.mean(r_all_det, axis=0)
    avg_r_rand = np.mean(r_all_rand, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(time, r_all_det[1], label='Deterministic (sigma=0)', lw=2)
    plt.plot(time, r_all_rand[1], label='Random (sigma=0.1)', lw=2)
    plt.xlabel("Time")
    plt.ylabel("Order Parameter r")
    plt.legend(loc='best')
    plt.tight_layout()
    np.savetxt(f'r_all_det.txt', r_all_det, fmt='%.6f')
    np.savetxt(f'r_all_rand.txt', r_all_rand, fmt='%.6f')
    plt.savefig(f'plot0.pdf')

    # plt.figure(figsize=(10, 6))
    # plt.plot(time, avg_r_det, label='Deterministic (sigma=0)', lw=2)
    # plt.plot(time, avg_r_rand, label='Random (sigma=0.1)', lw=2)
    # plt.xlabel("Time")
    # plt.ylabel("Average Order Parameter r")
    # plt.legend(loc='best')
    # plt.tight_layout()
    # np.savetxt(f'avg_r_det.txt', avg_r_det, fmt='%.6f')
    # np.savetxt(f'avg_r_rand.txt', avg_r_rand, fmt='%.6f')
    # plt.savefig(f'plot1.pdf')

    all_r_det = r_all_det.flatten()
    all_r_rand = r_all_rand.flatten()
    density_det = gaussian_kde(all_r_det)
    density_rand = gaussian_kde(all_r_rand)
    x_vals = np.linspace(min(all_r_det.min(), all_r_rand.min()),
                         max(all_r_det.max(), all_r_rand.max()), 200)
    y_det = density_det(x_vals)
    y_rand = density_rand(x_vals)

    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, y_det, label='Deterministic (sigma=0)', lw=2)
    plt.plot(x_vals, y_rand, label='Random (sigma=0.1)', lw=2)
    plt.xlabel("r")
    plt.ylabel("PDF")
    plt.legend(loc='best')
    plt.tight_layout()
    np.savetxt(f'x_vals.txt', x_vals)
    np.savetxt(f'y_det.txt', y_det)
    np.savetxt(f'y_rand.txt', y_rand)
    plt.savefig(f'plot2.pdf')

if __name__ == '__main__':
    main()