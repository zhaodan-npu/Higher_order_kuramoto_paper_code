#!/usr/bin/env julia

using DelimitedFiles, FFTW, Statistics, Random, Logging
import Base.Filesystem: basename, splitext, mkpath, joinpath
using Base.Threads

# ==== 稳定精确计算参数 ====
const DELTA_T    = 0.01
const MAXLAG     = 500.0
const P2         = 1.0
const NSURR      = 10        # 建议提高NSURR以提升统计稳定性
const ALPHA      = 0.95
const EPS_STD    = 1e-8

const SCRIPT_DIR = @__DIR__
include(joinpath(SCRIPT_DIR, "editdistance.jl"))

# 稳定版CONF计算函数 (使用未压缩数据)
function EDSPEC_conf_stable(t_events, lags, nsurr, alph, p2)
    Random.seed!(12345)
    Nevents, tstart, tend, fixed_length = length(t_events), t_events[1], t_events[end], length(lags)
    pwr = zeros(Float64, nsurr, fixed_length)

    @threads for j in 1:nsurr
        tsurr = sort(rand(tstart:DELTA_T:tend, Nevents))

        C_surr = try
            tmp_C = EDACF(tsurr, lags, nothing, p2)
            length(tmp_C) == fixed_length ? tmp_C : fill(NaN, fixed_length)
        catch
            fill(NaN, fixed_length)
        end

        std_C = std(skipmissing(C_surr))
        if std_C < EPS_STD || isnan(std_C)
            normC = zeros(fixed_length)
        else
            normC = (C_surr .- median(skipmissing(C_surr))) ./ std_C
        end

        pwr[j, :] = abs.(fftshift(fft(normC)))
    end

    idx0 = Int64(ceil(fixed_length/2)+1)
    fs = 1.0 / (lags[2] - lags[1])
    freqs = fftshift(fftfreq(fixed_length, fs))[idx0:end]
    pwr_ = pwr[:, idx0:end]
    SI = vec(mapslices(x -> quantile(skipmissing(x), alph), pwr_, dims=1))

    return freqs, SI
end

# 主计算函数（所有数据使用原始数据）
function compute_final(input_file::String, outdir::String)
    mkpath(outdir)
    base = splitext(basename(input_file))[1]
    @info "[$(base)] Starting processing..."

    # 读取原始数据
    data = readdlm(input_file, '\t'; skipstart=1)
    t_ev = vec(data[:, 1])

    if length(t_ev) < 3
        @warn "[$(base)] Too few events (<3), skipping."
        return
    end

    # 计算EDACF和EDSPEC（使用原始数据）
    lags = collect(0.0:DELTA_T:MAXLAG)
    C = EDACF(t_ev, lags, nothing, P2)
    freqs, pwr = EDSPEC(t_ev, lags, C, P2)

    # 计算CONF（也使用完全相同的原始数据）
    fconf, SI = EDSPEC_conf_stable(t_ev, lags, NSURR, ALPHA, P2)

    # 单独保存数据，确保长度一致
    minlen_acf_spec = minimum([length(lags), length(C), length(freqs), length(pwr)])
    writedlm(joinpath(outdir, base*"_EDACF.txt"), hcat(lags[1:minlen_acf_spec], C[1:minlen_acf_spec]), '\t')
    writedlm(joinpath(outdir, base*"_EDSPEC.txt"), hcat(freqs[1:minlen_acf_spec], pwr[1:minlen_acf_spec]), '\t')

    # CONF单独保存
    minlen_conf = min(length(fconf), length(SI))
    if minlen_conf > 0
        writedlm(joinpath(outdir, base*"_CONF.txt"), hcat(fconf[1:minlen_conf], SI[1:minlen_conf]), '\t')
    end

    @info "[$(base)] Successfully done."
end

function main()
    if length(ARGS) != 2
        println("Usage: julia process_path_final.jl <input.txt> <output_dir>")
        exit(1)
    end
    input_file, outdir = ARGS[1], ARGS[2]
    compute_final(input_file, outdir)
end

main()
