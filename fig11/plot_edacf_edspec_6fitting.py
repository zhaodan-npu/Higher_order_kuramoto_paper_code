#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
average_and_half_fit.py —

  (a) 平均 EDACF 并绘图；
  (b) 平均 EDSPEC + CONF + SEM，并对频率 f >= fit_min_freq 的部分做一次
      加权 log–log 拟合，得到幂律指数 beta 及其 95% 置信区间、R^2，
      同时与指数模型做 AIC 对比；
  (c) 不做局部放大 inset；
  (d) 所有拟合统计量会同时打印到控制台，并追加写入 'fit_stats.txt' 文件。
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.interpolate import interp1d

# — 全局配置 —
num_points = 5000
plt.rcParams.update({
    "font.family":     "Times New Roman",
    "axes.labelsize":  18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "axes.linewidth":  0.8,
    "figure.dpi":      300,
})

def safe_interp(x, y, num_points, log_scale=False):
    """
    对 (x,y) 去重并插值到公共网格。
    若 log_scale=True，则在 log10(x) 上等距取点（适合频率谱），
    否则在线性 x 上等距取点（适合自相关）。
    """
    x = np.asarray(x); y = np.asarray(y)
    if log_scale:
        mask = (x>0)&(y>0)&np.isfinite(x)&np.isfinite(y)
    else:
        mask = np.isfinite(x)&np.isfinite(y)
    x, y = x[mask], y[mask]
    xu, idx = np.unique(x, return_index=True)
    yu = y[idx]
    if xu.size < 2:
        return None, None
    if log_scale:
        x_new = np.logspace(np.log10(xu.min()), np.log10(xu.max()), num_points)
    else:
        x_new = np.linspace(xu.min(), xu.max(), num_points)
    f = interp1d(xu, yu, bounds_error=False, fill_value='extrapolate')
    return x_new, f(x_new)

def average_edacf(files):
    """
    读入多条 EDACF 曲线，插值到公共 lag 网格，计算平均值和 SEM。
    返回: lags, mean_EDACF(lag), SEM(lag)
    """
    interpolated = []
    x_ref = None
    for fn in files:
        data = np.loadtxt(fn)
        lags, vals = data[:,0], data[:,1]
        x, y = safe_interp(lags, vals, num_points)
        if x is not None:
            interpolated.append(y)
            x_ref = x
        else:
            print(f"Skipped EDACF: {fn}")
    if not interpolated:
        raise ValueError("No valid EDACF files found!")
    arr = np.vstack(interpolated)
    avg = arr.mean(axis=0)
    sem = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
    return x_ref, avg, sem

def average_spectrum(files):
    """
    读入多条频谱（EDSPEC 或 CONF），在 log10(f) 空间插值到公共频率网格，
    计算平均值和 SEM。返回: freqs, mean_P(f), SEM(f)
    """
    interpolated = []
    x_ref = None
    for fn in files:
        data = np.loadtxt(fn)
        freq, power = data[:,0], data[:,1]
        x, y = safe_interp(freq, power, num_points, log_scale=True)
        if x is not None:
            interpolated.append(y)
            x_ref = x
        else:
            print(f"Skipped Spectrum: {fn}")
    if not interpolated:
        raise ValueError("No valid spectrum files found!")
    arr = np.vstack(interpolated)
    avg = arr.mean(axis=0)
    sem = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
    return x_ref, avg, sem

def plot_edacf(lags, mean_c, sem_c):
    """绘制平均 EDACF + SEM。"""
    plt.figure(figsize=(8,5))
    plt.plot(lags, mean_c, 'b-', lw=2, label='Average EDACF')
    plt.fill_between(lags, mean_c-sem_c, mean_c+sem_c,
                     color='blue', alpha=0.3, label='SEM')
    plt.xlabel('Lag (dimensionless)', fontsize=16)
    plt.ylabel('EDACF', fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('average_EDACF_with_SE.pdf', dpi=300)
    plt.savefig('average_EDACF_with_SE.png', dpi=300)
    plt.close()
    print('Saved: average_EDACF_with_SE.pdf')

def plot_edspec_with_conf(freqs, powers, stderr, conf_powers,
                          fit_min_freq=0.1,
                          stats_file='fit_stats.txt'):
    """
    绘制 EDSPEC + SEM + CONF，并对 f >= fit_min_freq 的部分
    做加权 log–log 拟合，给出:
      - 幂律指数 beta 及其 95% 置信区间，
      - R^2（在 log10 空间），
      - 幂律模型与指数模型的 AIC 和 ΔAIC。
    同时在图上显示 beta 和 R^2，并将所有统计量写入 stats_file。
    """
    mask = np.isfinite(freqs)&np.isfinite(powers)&np.isfinite(conf_powers) \
           & (freqs>0)&(powers>0)&(conf_powers>0)
    f = freqs[mask]; p = powers[mask]; se = stderr[mask]; c = conf_powers[mask]

    fig, ax = plt.subplots(figsize=(8,5), constrained_layout=True)

    # 1) 原谱、SEM、置信带
    ax.loglog(f, p,   'r-',  lw=2, label='Average EDSPEC')
    sem_lo = np.clip(p-se, 1e-300, None); sem_hi = p+se
    ax.fill_between(f, sem_lo, sem_hi, color='red', alpha=0.3, label='SEM')
    ax.loglog(f, c,   'k--', lw=2, label='Average 95% CONF')

    # 高亮超线区，不进 legend
    ex = p > c
    if np.any(ex):
        ax.fill_between(f, c, p, where=ex, color='red', alpha=0.15, label='_nolegend_')

    # 2) 选取频率范围做加权 log–log 拟合
    fit_mask = f >= fit_min_freq
    if np.count_nonzero(fit_mask) < 3:
        print(f"Not enough data points for power-law fit (f >= {fit_min_freq}).")
    else:
        f0, p0, se0 = f[fit_mask], p[fit_mask], se[fit_mask]
        logf = np.log10(f0)
        logp = np.log10(p0)

        # 将平均谱的 SEM 转换为 log10 P 的误差: sigma_i ≈ SEM_P / (P ln 10)
        ln10 = np.log(10.0)
        sigma = se0/(p0*ln10)
        vs = sigma[np.isfinite(sigma)&(sigma>0)]
        floor = (np.percentile(vs,1)*1e-3) if vs.size else 1e-12
        sigma = np.where((~np.isfinite(sigma))|(sigma<=0), floor, sigma)
        w = 1.0/sigma   # 加权最小二乘的权重

        have_cov = False
        try:
            (slope, intercept), cov = np.polyfit(logf, logp, 1, w=w, cov=True)
            slope_err = np.sqrt(cov[0,0])
            have_cov = True
        except Exception as e:
            print("polyfit with covariance failed, falling back without cov:", e)
            slope, intercept = np.polyfit(logf, logp, 1, w=w)
            slope_err = np.nan

        # 幂律：P ~ f^{-beta}
        beta = -slope
        beta_err_95 = 1.96*slope_err if np.isfinite(slope_err) else np.nan

        # R^2（在 log10 空间）
        y_fit = slope*logf + intercept
        resid = logp - y_fit
        ss_res = np.sum(resid**2)
        ss_tot = np.sum((logp - np.mean(logp))**2)
        R2 = 1.0 - ss_res/ss_tot if ss_tot>0 else np.nan

        # AIC（幂律模型）
        N = logf.size
        k = 2  # slope + intercept
        AIC_power = np.nan
        if N > k and ss_res>0:
            AIC_power = 2*k + N*np.log(ss_res/N)

        # 指数模型：log10 P = A + B f
        try:
            (s_exp, b_exp), cov_exp = np.polyfit(f0, logp, 1, w=w, cov=True)
            y_fit_exp = s_exp*f0 + b_exp
            resid_exp = logp - y_fit_exp
            ss_res_exp = np.sum(resid_exp**2)
            AIC_exp = 2*k + N*np.log(ss_res_exp/N) if (N>k and ss_res_exp>0) else np.nan
        except Exception as e:
            print("Exponential polyfit failed:", e)
            s_exp = b_exp = np.nan
            AIC_exp = np.nan

        delta_AIC = AIC_exp - AIC_power if (np.isfinite(AIC_exp) and np.isfinite(AIC_power)) else np.nan

        # —— 控制台输出，方便写进论文 ——
        print(f"Power-law fit for f >= {fit_min_freq}:")
        print(f"  beta = {beta:.3f} ± {beta_err_95:.3f} (95% CI)")
        print(f"  R^2  = {R2:.3f}")
        if np.isfinite(AIC_power):
            print(f"  AIC_power = {AIC_power:.2f}")
        if np.isfinite(AIC_exp):
            print(f"  AIC_exp   = {AIC_exp:.2f}")
        if np.isfinite(delta_AIC):
            print(f"  ΔAIC (exp - power) = {delta_AIC:.2f}")

        # —— 结果写入文件，方便之后整理成表 —— 
        try:
            with open(stats_file, "a", encoding="utf-8") as fout:
                fout.write(
                    f"fit_min_freq={fit_min_freq}, "
                    f"beta={beta:.6f}, "
                    f"beta_95_CI_halfwidth={beta_err_95:.6f}, "
                    f"R2={R2:.6f}, "
                    f"AIC_power={AIC_power:.6f}, "
                    f"AIC_exp={AIC_exp:.6f}, "
                    f"delta_AIC={delta_AIC:.6f}\n"
                )
        except Exception as e:
            print(f"Warning: could not write stats to file '{stats_file}': {e}")

        # 拟合线
        freq_line = np.logspace(np.log10(f0.min()), np.log10(f0.max()), 400)
        logfl = np.log10(freq_line)
        logpl = slope*logfl + intercept
        P_fit = 10**logpl
        ax.loglog(freq_line, P_fit, '--', lw=2, color='#1f77b4',
                  label='Power-law fit')

        # 在图上标注 beta 和 R^2
        txt = (r"$\beta = {:.3f} \pm {:.3f}$ (95\% CI)".format(beta, beta_err_95)
               + "\n"
               + r"$R^2 = {:.3f}$".format(R2))
        ax.text(0.05, 0.95, txt,
                transform=ax.transAxes,
                va='top', ha='left',
                fontsize=14,
                bbox=dict(boxstyle='round', facecolor='white',
                          alpha=0.8, edgecolor='none'))

        # 拟合的不确定性带（1σ），不进 legend
        if have_cov:
            yse = np.sqrt(cov[0,0]*logfl**2 + 2*cov[0,1]*logfl + cov[1,1])
            P_lo = 10**(logpl-yse); P_hi = 10**(logpl+yse)
            ax.fill_between(freq_line,
                            np.clip(P_lo,1e-300,None),
                            P_hi,
                            color='#1f77b4', alpha=0.15, label='_nolegend_')

    # 3) 最终 legend
    ax.legend(fontsize=14)

    ax.set_xlabel('Frequency', fontsize=16)
    ax.set_ylabel('EDSPEC',   fontsize=16)
    ax.grid(True, which='both', ls='--')

    fig.savefig('average_EDSPEC_with_CONF_SE_and_fit.pdf', dpi=300, bbox_inches='tight')
    fig.savefig('average_EDSPEC_with_CONF_SE_and_fit.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('Saved: average_EDSPEC_with_CONF_SE_and_fit.pdf')

if __name__ == '__main__':
    edacf_files  = glob.glob('*_EDACF.txt')
    edspec_files = glob.glob('*_EDSPEC.txt')
    conf_files   = glob.glob('*_CONF.txt')

    print(f"EDACF files: {len(edacf_files)}, EDSPEC files: {len(edspec_files)}, CONF files: {len(conf_files)}")

    # (a) 平均 EDACF
    try:
        lags, avg_c, sem_c = average_edacf(edacf_files)
        plot_edacf(lags, avg_c, sem_c)
    except ValueError as e:
        print("EDACF error:", e)

    # (b) 平均 EDSPEC + 拟合
    try:
        freqs, avg_p, sem_p = average_spectrum(edspec_files)
        _,   avg_conf, _    = average_spectrum(conf_files)
        plot_edspec_with_conf(freqs, avg_p, sem_p, avg_conf)
    except ValueError as e:
        print("EDSPEC error:", e)
