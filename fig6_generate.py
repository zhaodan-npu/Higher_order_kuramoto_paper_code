import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import levy_stable
from mpl_toolkits.mplot3d import Axes3D


def f_r(r, K1, K2, Delta=1.0):
    return -Delta * r + 0.5 * K1 * (r - r**3) + 0.5 * K2 * (r**3 - r**5)

def df_dr(r, K1, K2, Delta=1.0):
    return -Delta + 0.5 * K1 * (1 - 3*r**2) + 0.5 * K2 * (3*r**2 - 5*r**4)

def find_roots_in_0_1(K1, K2, steps=300):
    r_vals = np.linspace(0, 1, steps)
    f_old = f_r(r_vals[0], K1, K2)
    roots = []
    for i in range(1, steps):
        f_new = f_r(r_vals[i], K1, K2)
        if abs(f_new) < 1e-14 or f_new * f_old < 0:
            a, b = r_vals[i-1], r_vals[i]
            fa = f_old
            for _ in range(20):
                m = 0.5*(a+b)
                fm = f_r(m, K1, K2)
                if fa*fm <= 0:
                    b, fb = m, fm
                else:
                    a, fa = m, fm
            roots.append(0.5*(a+b))
        f_old = f_new
    return sorted(set(np.round(roots,5)))


def kuramoto_derivatives(theta, omega, K1, K2):
    R, N = theta.shape
    dtheta = np.zeros_like(theta)
    for r in range(R):
        C = np.cos(theta[r]).sum()
        S = np.sin(theta[r]).sum()
        for i in range(N):
            p = (K1/N) * np.sin(theta[r] - theta[r,i]).sum()
            A = np.sin(2*theta[r] - theta[r,i]).sum()
            B = np.cos(2*theta[r] - theta[r,i]).sum()
            dtheta[r,i] = omega[r,i] + p + (K2/(N*N))*(C*A - S*B)
    return dtheta


def simulate_one_K(N, K1, K2, alpha_noise, sigma_noise,
                   dt, trans_steps, sample_steps,
                   omega, theta):
    # 过渡
    for _ in range(trans_steps):
        dθ = kuramoto_derivatives(theta, omega, K1, K2)
        noise = (levy_stable.rvs(alpha_noise, 0, size=(1,N))
                 * (sigma_noise*(dt**(1/alpha_noise)))) if sigma_noise>0 else 0
        theta = (theta + dt*dθ + noise) % (2*np.pi)
    # 采样
    rs = []
    for _ in range(sample_steps):
        dθ = kuramoto_derivatives(theta, omega, K1, K2)
        noise = (levy_stable.rvs(alpha_noise, 0, size=(1,N))
                 * (sigma_noise*(dt**(1/alpha_noise)))) if sigma_noise>0 else 0
        theta = (theta + dt*dθ + noise) % (2*np.pi)
        C, S = np.cos(theta).sum(), np.sin(theta).sum()
        rs.append(np.sqrt(C*C + S*S)/N)
    return theta, np.mean(rs)

def forward_backward_scan(N, K1_arr, K2,
                          alpha_noise, sigma_noise,
                          dt, trans_steps, sample_steps):
    np.random.seed(42)
    omega = np.random.standard_cauchy(N).reshape(1,N)
    # forward
    theta = 2*np.pi*np.random.rand(1,N)
    r_fwd = []
    for K1 in K1_arr:
        theta, r = simulate_one_K(N, K1, K2,
                                  alpha_noise, sigma_noise,
                                  dt, trans_steps, sample_steps,
                                  omega, theta)
        r_fwd.append(r)
    # backward
    theta = np.zeros((1,N))
    r_bwd = []
    for K1 in reversed(K1_arr):
        theta, r = simulate_one_K(N, K1, K2,
                                  alpha_noise, sigma_noise,
                                  dt, trans_steps, sample_steps,
                                  omega, theta)
        r_bwd.append(r)
    r_bwd.reverse()
    return np.array(r_fwd), np.array(r_bwd)


if __name__ == "__main__":
    # --- 场景定义 ---
    scenarios = [
        {"name":"Deterministic", "alpha":2.0, "sigma":0.0, "linestyle":"-"},
        {"name":"Gaussian",     "alpha":2.0, "sigma":0.5, "linestyle":"--"},
        {"name":"Levy",         "alpha":1.6, "sigma":0.5, "linestyle":":"}
    ]
    K2_list  = [0,4,8,10]
    K1_vals  = np.linspace(-2,6,50)
    N         = 100
    dt        = 0.01
    trans     = int(100/dt)
    samp      = int(50/dt)

    os.makedirs("results/data_txt", exist_ok=True)
    fig = plt.figure(figsize=(18,6))

    for idx, sc in enumerate(scenarios, start=1):
        ax = fig.add_subplot(1,3,idx, projection='3d')
        for k2i, K2 in enumerate(K2_list):
            clr = plt.rcParams['axes.prop_cycle'].by_key()['color'][k2i]

            # 解析分支
            r_an = []
            for K1 in K1_vals:
                roots  = find_roots_in_0_1(K1, K2)
                stable = [r for r in roots if df_dr(r,K1,K2)<0]
                r_an.append(max(stable) if stable else 0.0)
            r_an = np.array(r_an)

            # 数值扫描
            r_fwd, r_bwd = forward_backward_scan(
                N, K1_vals, K2,
                sc["alpha"], sc["sigma"],
                dt, trans, samp
            )

            # 保存 TXT
            data = np.column_stack([K1_vals, r_an, r_fwd, r_bwd])
            fname = f"results/data_txt/{sc['name']}_K2_{K2}.txt"
            np.savetxt(fname, data,
                       header="K1    analytic    forward    backward",
                       fmt="%.6f")

            # 绘制三条折线
            ax.plot( K1_vals, [K2]*len(K1_vals), r_an,
                     color=clr, linestyle=sc["linestyle"], linewidth=2,
                     label=f"Analytic, $K_2$={K2}")
            ax.plot( K1_vals, [K2]*len(K1_vals), r_fwd,
                     color=clr, linestyle=sc["linestyle"], linewidth=1,
                     marker='o', markersize=3,
                     label=f"Fwd, $K_2$={K2}")
            ax.plot( K1_vals, [K2]*len(K1_vals), r_bwd,
                     color=clr, linestyle=sc["linestyle"], linewidth=1,
                     marker='^', markersize=3,
                     label=f"Bwd, $K_2$={K2}")

        ax.set_title(f"{sc['name']} (α={sc['alpha']}, σ={sc['sigma']})", fontsize=14)
        ax.set_xlabel('$K_1$', fontsize=12)
        ax.set_ylabel('$K_2$', fontsize=12)
        ax.set_zlabel('$r$', fontsize=12)
        ax.view_init(elev=25, azim=45)
        if idx==1:
            ax.legend(loc='upper left', fontsize=8, ncol=2)

    plt.tight_layout()
    plt.savefig("results/bifurcation_all_scenarios.pdf")
    plt.show()
