#!/usr/bin/env julia
#
# process_noise_windows.jl — 对单个 noise_pX_spikes_winY.txt 文件计算 EDACF 和 EDSPEC
#
# 用法：
#   julia process_noise_windows.jl <input_spike_window.txt> <output_dir>
#
# 输入文件：必须是两列文本，第一列是 spike 时间（载入后只使用时间列），
#          第二列是噪声值（此处不直接使用，仅保留格式对齐）。
#          文件第一行会被跳过（表头行）。
#
# 计算结果会输出到 <output_dir>，包括两个文件：
#   <basename>_EDACF.txt   （两列：lags C）
#   <basename>_EDSPEC.txt  （两列：freqs  pwr）
#
# 注意：脚本旁需要有同目录下的 editdistance.jl，以提供 EDACF/EDSPEC 函数。
#

using DelimitedFiles, FFTW, Statistics, Logging
import Base.Filesystem: basename, splitext, mkpath, joinpath

# —— 用户配置 —— 
const DELTA_T   = 0.01       # 自相关滞后步长 (s)
const MAXLAG    = 500.0      # 最大滞后 (s)
const P2        = 1.0        # 编辑距离“删除”成本

# 确保能找到 editdistance.jl（与本脚本同目录）
const SCRIPT_DIR = @__DIR__
include(joinpath(SCRIPT_DIR, "editdistance.jl"))  # 必须提供 EDACF, EDSPEC

function main()
    if length(ARGS) != 2
        println("Usage: julia process_noise_windows.jl <input_window.txt> <output_dir>")
        exit(1)
    end

    input_file, outdir = ARGS[1], ARGS[2]
    mkpath(outdir)

    base = splitext(basename(input_file))[1]
    @info "Processing $base → $outdir"

    # 1) 读入两列数据，跳过表头
    data = readdlm(input_file, '\t'; skipstart=1)
    # 只取第一列：事件时刻
    t = vec(data[:, 1])

    if length(t) < 3
        @warn "$base: too few events (< 3), skipping."
        return
    end

    # 2) 计算 EDACF
    lags = collect(0.0:DELTA_T:MAXLAG)
    C    = EDACF(t, lags, nothing, P2)

    # 3) 计算 EDSPEC
    freqs, pwr = EDSPEC(t, lags, C, P2)

    # 4) 写出结果
    writedlm(joinpath(outdir, base * "_EDACF.txt"),  hcat(lags, C),    '\t')
    writedlm(joinpath(outdir, base * "_EDSPEC.txt"), hcat(freqs, pwr), '\t')

    @info "Done $base"
end

main()
