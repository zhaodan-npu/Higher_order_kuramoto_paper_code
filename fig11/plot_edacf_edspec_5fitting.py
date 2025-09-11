#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.interpolate import interp1d
from matplotlib.lines import Line2D


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
    plt.close()
    print('Saved: average_EDACF_with_SE.pdf')

def plot_edspec_with_conf(freqs, powers, stderr, conf_powers):

    mask = np.isfinite(freqs)&np.isfinite(powers)&np.isfinite(conf_powers) \
           & (freqs>0)&(powers>0)&(conf_powers>0)
    f = freqs[mask]; p = powers[mask]; se = stderr[mask]; c = conf_powers[mask]

    fig, ax = plt.subplots(figsize=(8,5), constrained_layout=True)


    ax.loglog(f, p,   'r-',  lw=2, label='Average EDSPEC')
    sem_lo = np.clip(p-se, 1e-300, None); sem_hi = p+se
    ax.fill_between(f, sem_lo, sem_hi, color='red', alpha=0.3, label='SEM')
    ax.loglog(f, c,   'k--', lw=2, label='Average 95% CONF')


    ex = p > c
    if np.any(ex):
        ax.fill_between(f, c, p, where=ex, color='red', alpha=0.15, label='_nolegend_')


    half = f >= 6
    f0, p0, se0 = f[half], p[half], se[half]
    logf = np.log10(f0); logp = np.log10(p0)
    ln10 = np.log(10.0)
    sigma = se0/(p0*ln10)
    vs = sigma[np.isfinite(sigma)&(sigma>0)]
    floor = (np.percentile(vs,1)*1e-3) if vs.size else 1e-12
    sigma = np.where((~np.isfinite(sigma))|(sigma<=0), floor, sigma)
    w = 1.0/sigma

    try:
        (slope, intercept), cov = np.polyfit(logf, logp, 1, w=w, cov=True)
        slope_err = np.sqrt(cov[0,0])
    except:
        slope, intercept = np.polyfit(logf, logp, 1, w=w)
        slope_err = np.nan

    freq_line = np.logspace(np.log10(f0.min()), np.log10(f0.max()), 400)
    logfl = np.log10(freq_line)
    logpl = slope*logfl + intercept
    P_fit = 10**logpl


    ax.loglog(freq_line, P_fit, '--', lw=2, color='#1f77b4',
              label=f"Slope: {slope:.3f} ± {slope_err:.3f}")

    if 'cov' in locals():
        yse = np.sqrt(cov[0,0]*logfl**2 + 2*cov[0,1]*logfl + cov[1,1])
        P_lo = 10**(logpl-yse); P_hi = 10**(logpl+yse)
        ax.fill_between(freq_line,
                        np.clip(P_lo,1e-300,None),
                        P_hi,
                        color='#1f77b4', alpha=0.15, label='_nolegend_')


    ax.legend(fontsize=14)

    ax.set_xlabel('Frequency', fontsize=16)
    ax.set_ylabel('EDSPEC',   fontsize=16)
    ax.grid(True, which='both', ls='--')

    fig.savefig('average_EDSPEC_with_CONF_SE_and_fit.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('Saved: average_EDSPEC_with_CONF_SE_and_fit.pdf')

if __name__ == '__main__':
    edacf_files  = glob.glob('*_EDACF.txt')
    edspec_files = glob.glob('*_EDSPEC.txt')
    conf_files   = glob.glob('*_CONF.txt')

    print(f"EDACF files: {len(edacf_files)}, EDSPEC files: {len(edspec_files)}, CONF files: {len(conf_files)}")


    try:
        lags, avg_c, sem_c = average_edacf(edacf_files)
        plot_edacf(lags, avg_c, sem_c)
    except ValueError as e:
        print("EDACF error:", e)


    try:
        freqs, avg_p, sem_p = average_spectrum(edspec_files)
        _,   avg_conf, _    = average_spectrum(conf_files)
        plot_edspec_with_conf(freqs, avg_p, sem_p, avg_conf)
    except ValueError as e:
        print("EDSPEC error:", e)
