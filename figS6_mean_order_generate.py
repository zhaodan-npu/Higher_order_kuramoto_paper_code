import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from joblib import Parallel, delayed
from scipy.stats import levy_stable

@njit(fastmath=True)
def kuramoto_derivatives(theta, omega, K1, K2):
    """
    计算高阶(二次+三次耦合) Kuramoto 模型下, 每个振子的瞬时导数 (theta_dot).
    theta: shape=(N,)
    omega: shape=(N,)
    返回 dtheta: shape=(N,)
    """
    N = len(theta)
    sin_th = np.sin(theta)
    cos_th = np.cos(theta)
    sum_sin = np.sum(sin_th)
    sum_cos = np.sum(cos_th)

    # 二次(一阶)耦合项
    pairwise_term = (K1 / N) * (sum_sin * cos_th - sum_cos * sin_th)

    # 三次(二阶)耦合项
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
    """
    用四阶RK4对系统进行一步积分, 然后加噪声 noise(列维或高斯).
    theta: shape=(N,)
    """
    k1 = kuramoto_derivatives(theta, omega, K1, K2)
    k2 = kuramoto_derivatives(theta + 0.5 * dt * k1, omega, K1, K2)
    k3 = kuramoto_derivatives(theta + 0.5 * dt * k2, omega, K1, K2)
    k4 = kuramoto_derivatives(theta + dt * k3, omega, K1, K2)

    theta += dt * (k1 + 2*k2 + 2*k3 + k4) / 6
    theta += noise
    theta %= 2 * np.pi

@njit(fastmath=True)
def integrate_chunk(theta, omega, K1, K2, dt, noise_chunk):
    """
    连续地对一段噪声 noise_chunk (shape=(chunk_size,N)) 做更新
    并最终返回更新后的 theta.
    """
    for i in range(noise_chunk.shape[0]):
        rk4_step(theta, omega, K1, K2, dt, noise_chunk[i])
    return theta

def single_sample_run(K1, K2, N,
                      transient_steps, main_steps,
                      dt, omega, sigma, alpha, beta):
    """
    单次模拟:
      - 初始化相位
      - 先跑 transient_steps 过渡, 不计统计
      - 再跑 main_steps, 最终取最后时刻的全局序参数 r
    返回: float, 表示该条模拟最终(或稳态)的全局序参数 r
    """

    # 初相: shape=(N,)
    theta = np.random.uniform(0, 2*np.pi, N)

    # 定义一个通用的分块函数
    def run_for_steps(num_steps):
        # 分块以避免一次生成过大噪声
        chunk_size = 10000
        n_chunks = num_steps // chunk_size
        remainder = num_steps % chunk_size

        for _ in range(n_chunks):
            noise_chunk = sigma * levy_stable.rvs(alpha, beta, size=(chunk_size, N)) * (dt**(1/alpha))
            integrate_chunk(theta, omega, K1, K2, dt, noise_chunk)
        if remainder > 0:
            noise_chunk = sigma * levy_stable.rvs(alpha, beta, size=(remainder, N)) * (dt**(1/alpha))
            integrate_chunk(theta, omega, K1, K2, dt, noise_chunk)

    # 1) 先跑 transient_steps
    run_for_steps(transient_steps)

    # 2) 再跑 main_steps
    run_for_steps(main_steps)

    # 取最后时刻的相位, 计算序参数
    r_final = np.abs(np.mean(np.exp(1j * theta)))
    return r_final

def mean_global_order_parameter(K1, K2, N,
                                num_samples,
                                transient_steps, main_steps,
                                dt, sigma, alpha, beta,
                                seed):
    """
    并行执行多条 single_sample_run, 返回平均全局序参数
    (一般是 main_steps后时刻的 r).
    """
    np.random.seed(seed)
    # 每个振子的固有频率
    omega = np.random.standard_cauchy(N)

    results = Parallel(n_jobs=2)(
        delayed(single_sample_run)(
            K1, K2, N,
            transient_steps, main_steps,
            dt, omega, sigma, alpha, beta
        )
        for _ in range(num_samples)
    )
    return np.mean(results)


if __name__ == '__main__':
    # ---------- 模型参数 ----------
    N = 100
    K2 = 8.0
    K1 = 0.8
    beta = 0              # 对称 Lévy 分布
    num_samples = 50
    dt = 0.01
    transient_steps = 100000
    main_steps = 900000   # 或者 1e6 - 1e5

    # 你要扫 alpha, sigma
    alpha_vals = np.arange(1.1, 2.05, 0.1)
    alpha_vals[-1] = 2.0  # 强制最后一个值精确等于2.0
    sigma_vals = np.arange(0.1, 0.6, 0.05)
    order_param_results = np.zeros((len(alpha_vals), len(sigma_vals)))

    for i, alpha in enumerate(alpha_vals):
        for j, sigma in enumerate(sigma_vals):
            mean_r = mean_global_order_parameter(
                K1, K2, N,
                num_samples,
                transient_steps, main_steps,
                dt, sigma, alpha, beta,
                seed=42
            )
            order_param_results[i, j] = mean_r
            print(f"alpha={alpha:.2f}, sigma={sigma:.2f}, Mean Order Parameter={mean_r:.3f}")

    # 保存数据
    np.savetxt("order_param_results.txt", order_param_results)

    # 绘图
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 8))
    plt.imshow(order_param_results,
               extent=[sigma_vals.min(), sigma_vals.max(),
                       alpha_vals.min(), alpha_vals.max()],
               aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='Mean Global Order Parameter')
    plt.xlabel("Sigma")
    plt.ylabel("Alpha")
    plt.title(f"Mean Global Order Parameter (K1={K1}, K2={K2}, N={N})")
    plt.savefig("order_param_heatmap.pdf")
    # plt.show()
