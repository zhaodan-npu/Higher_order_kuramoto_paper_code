import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
from scipy.stats import gaussian_kde
from scipy.stats import levy_stable

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

def simulate(theta0, omega, K1, K2, dt, T, sigma, alpha, beta, transient_steps):
    num_realizations, N = theta0.shape
    r_all = np.empty((num_realizations, T - transient_steps))
    theta = theta0.copy()

    for t in range(T):
        rk4_step(theta, omega, K1, K2, dt)
        noise = sigma * levy_stable.rvs(alpha, beta, size=(num_realizations, N)) * np.sqrt(dt)
        theta += noise
        mod_2pi(theta)
        if t >= transient_steps:
            r_all[:, t - transient_steps] = order_parameter(theta)

    return r_all

def main():
    N = 100
    num_realizations = 10
    K1 = 0.8
    K2 = 8.0
    dt = 0.01
    T = int(1e6)
    transient_steps = int(1e5)  # Number of transient steps to discard
    beta = 0
    alpha = 2  # Fixed alpha value

    theta_initial = 2 * np.pi * np.random.rand(num_realizations, N)
    omega_values = np.random.standard_cauchy((num_realizations, N))

    sigma_vals = np.arange(0.1, 2.1, 0.5)
    x_vals = np.linspace(0, 1, 200)
    pdf_matrix = np.zeros((len(sigma_vals), len(x_vals)))

    for i, sigma in enumerate(sigma_vals):
        print(f"Simulating for sigma={sigma}...")
        r_all = simulate(theta_initial, omega_values, K1, K2, dt, T, sigma, alpha, beta, transient_steps)
        all_r = r_all.flatten()
        density = gaussian_kde(all_r)
        pdf_matrix[i, :] = density(x_vals)

    # Save the pdf_matrix to a text file
    np.savetxt("pdf_matrix_sigma.txt", pdf_matrix)

    plt.figure(figsize=(10, 8))
    plt.imshow(pdf_matrix, extent=[x_vals.min(), x_vals.max(), sigma_vals.min(), sigma_vals.max()],
               aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='PDF')
    plt.xlabel("Order Parameter r")
    plt.ylabel("Sigma")
    plt.title("PDF of Order Parameter r for Different Sigma Values (alpha=2)")
    plt.savefig("pdf_heatmap_sigma.png")
    plt.show()

if __name__ == '__main__':
    main()