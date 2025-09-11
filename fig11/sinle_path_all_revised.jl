#!/usr/bin/env julia

using DelimitedFiles, FFTW, Statistics, Random, Plots
import Base.Filesystem: basename, splitext, joinpath

const SCRIPT_DIR = @__DIR__
include(joinpath(SCRIPT_DIR, "editdistance.jl"))


const DELTA_T    = 0.01       
const MAXLAG     = 500.0     
const P2         = 1.0        
const NSURR      = 20         
const ALPHA      = 0.95       
const DEDUP_GAP  = 0.02      
const MAX_EVENTS = 20000      
const EPS_STD    = 1e-8     


function compress_events(t::Vector{Float64}; gap::Float64)
    isempty(t) && return t
    out = Float64[t[1]]
    for i in 2:length(t)
        if t[i] - t[i-1] > gap
            push!(out, t[i])
        end
    end
    return out
end

function EDSPEC_conf_safe(t_events, lags, nsurr, alph, p2)
    Nevents, tstart, tend, L = length(t_events), t_events[1], t_events[end], length(lags)
    pwr = Array{Float64}(undef, (nsurr, L))

    for j in 1:nsurr
   
        tsurr = sort(rand(tstart:DELTA_T:tend, Nevents))
        C_surr = EDACF(tsurr, lags, nothing, p2)
        std_C = std(C_surr)

        if std_C < EPS_STD
            normC = zeros(length(C_surr))
        else
            normC = (C_surr .- median(C_surr)) ./ std_C
        end

        pwr[j, :] = abs.(fftshift(fft(normC)))
    end

    idx0 = Int64(ceil(L/2)+1)
    fs = 1.0 / (lags[2] - lags[1])
    freqs = fftshift(fftfreq(L, fs))[idx0:end]
    pwr_ = pwr[:, idx0:end]
    SI = vec(mapslices(x -> quantile(x, alph), pwr_, dims=1))

    return freqs, SI
end


function compute_edacf_edspec(input_file::String, outdir::String)
    mkpath(outdir)

    data = readdlm(input_file, '\t'; skipstart=1)
    t_ev = vec(data[:, 1])

    t_ev = compress_events(t_ev; gap=DEDUP_GAP)

    if length(t_ev) > MAX_EVENTS
        shuffle!(t_ev)
        t_ev = sort!(t_ev[1:MAX_EVENTS])
    end

    if length(t_ev) < 3
        @warn "Too few events (<3), skipping."
        return
    end

    lags = collect(0.0:DELTA_T:MAXLAG)
    C = EDACF(t_ev, lags, nothing, P2)
    freqs, pwr = EDSPEC(t_ev, lags, C, P2)

    fconf, SI = NSURR > 0 ? EDSPEC_conf_safe(t_ev, lags, NSURR, ALPHA, P2) : (Float64[], Float64[])

    base = splitext(basename(input_file))[1]
    writedlm(joinpath(outdir, base*"_EDACF.txt"),  hcat(lags, C), '\t')
    writedlm(joinpath(outdir, base*"_EDSPEC.txt"), hcat(freqs, pwr), '\t')
    NSURR>0 && writedlm(joinpath(outdir, base*"_CONF.txt"), hcat(fconf, SI), '\t')

    p1 = plot(lags, C; ribbon=SI, xlabel="Lag", ylabel="EDACF", title=base*"_EDACF", legend=false, framestyle=:box)
    p2 = plot(freqs, pwr; ribbon=SI, xaxis=:log, yaxis=:log, xlabel="Frequency", ylabel="Power", title=base*"_EDSPEC", legend=false, framestyle=:box)

    savefig(p1, joinpath(outdir, base*"_EDACF.pdf"))
    savefig(p2, joinpath(outdir, base*"_EDSPEC.pdf"))

    @info "Done $input_file"
end


function main()
    if length(ARGS) != 2
        println("Usage: julia sinle_path_all_revised.jl <input.txt> <output_dir>")
        exit(1)
    end
    input_file, outdir = ARGS[1], ARGS[2]
    compute_edacf_edspec(input_file, outdir)
end

main()
