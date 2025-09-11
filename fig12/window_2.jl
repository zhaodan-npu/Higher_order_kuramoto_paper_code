#!/usr/bin/env julia
using DelimitedFiles
import Base.Filesystem: mkpath, basename, splitext, joinpath


const INPUT_DIR   = "results"                 
const OUTPUT_DIR  = "result_window_21"        
const WINDOW_LEN  = 200.0                      
const WINDOW_STEP = 200.0                      
const DELIM       = '\t'                       
const FILE_NAME   = "spikes_a0_s0_p21.txt"   


mkpath(OUTPUT_DIR)

"""
segment_fixed(fn, outdir; win_len, win_step)

将文件 fn 按固定窗口长度 win_len 和步长 win_step 分段。
写入 outdir，文件命名为 <basename>_win<i>.txt。
"""
function segment_fixed(fn::String, outdir::String; win_len::Float64, win_step::Float64)
    data = readdlm(fn, DELIM; skipstart=1)
    times = vec(data[:,1])
    t_min, t_max = minimum(times), maximum(times)
    starts = collect(t_min:win_step:t_max)
    base = splitext(basename(fn))[1]
    idx = 1
    for t0 in starts
        window = data[(times .>= t0) .& (times .< t0+win_len), :]
        if !isempty(window)
            outfn = joinpath(outdir, string(base, "_win", idx, ".txt"))
            writedlm(outfn, window, DELIM)
            @info "Wrote window $idx: $outfn (events=$(size(window,1)))"
            idx += 1
        end
    end
end

# 主程序：仅对指定文件进行固定窗口分段
fn = joinpath(INPUT_DIR, FILE_NAME)
if !isfile(fn)
    @error "Input file not found: $fn"
    exit(1)
end
@info "Segmenting $FILE_NAME into fixed windows of length $WINDOW_LEN"
segment_fixed(fn, OUTPUT_DIR; win_len=WINDOW_LEN, win_step=WINDOW_STEP)
