#!/usr/bin/env julia
# segment_noise_windows.jl — 将列维噪声“spike”按固定窗口分割

using DelimitedFiles
import Base.Filesystem: mkpath, basename, splitext, joinpath

# —— 用户配置 —— 
const INPUT_DIR   = "noise_spikes2"      # 存放所有噪声 spike txt 的目录
const OUTPUT_DIR  = "noise_windows"      # 分段后 txt 存放目录
const WINDOW_LEN  = 200.0                # 窗口长度（与时间单位保持一致）
const WINDOW_STEP = 200.0                # 窗口步长，与窗口长度相同表示不重叠
const DELIM       = '\t'                 # 文本列之间的分隔符

# 创建输出目录
mkpath(OUTPUT_DIR)

"""
segment_fixed(fn, outdir; win_len, win_step)

将文件 fn 的“spike”按固定窗口长度 win_len 和步长 win_step 分段：
  - fn 必须是两列文件，第一列为时间，第二列为噪声值，且有表头。
  - 对每个 t0 从 t_min 到 t_max 以步长 win_step 构造窗口 [t0, t0+win_len),
    将所有 times ∈ [t0, t0+win_len) 的行写入单独文件。
  - 输出文件命名为 <basename>_win<i>.txt，放在 outdir 中。
"""
function segment_fixed(fn::String, outdir::String; win_len::Float64, win_step::Float64)
    # 1) 读取带表头的两列数据
    data = readdlm(fn, DELIM; skipstart=1)
    # 第一列是时间
    times = vec(data[:, 1])
    if isempty(times)
        @warn "文件中无数据，跳过：$fn"
        return
    end

    # 2) 计算该文件时间范围
    t_min, t_max = minimum(times), maximum(times)
    # 3) 生成所有窗口的起点
    starts = collect(t_min:win_step:t_max)

    base = splitext(basename(fn))[1]
    idx = 1

    for t0 in starts
        # 取出属于 [t0, t0+win_len) 的行
        mask = (times .>= t0) .& (times .< t0 + win_len)
        if any(mask)
            window = data[mask, :]
            outfn = joinpath(outdir, "$(base)_win$(idx).txt")
            writedlm(outfn, window, DELIM)
            @info "Wrote window $idx: $outfn  (events=$(size(window, 1)))"
            idx += 1
        end
    end
end

# ------------------- 主程序 -------------------

# 1) 列出 INPUT_DIR 下所有 .txt 文件
files = filter(fn -> endswith(fn, ".txt"), readdir(INPUT_DIR; join=true))
if isempty(files)
    @error "在目录 '$INPUT_DIR' 中未找到任何 .txt 文件"
    exit(1)
end

# 2) 对每个文件调用 segment_fixed
for fn in sort(files)
    @info "Segmenting $(basename(fn)) into fixed windows of length $WINDOW_LEN"
    segment_fixed(fn, OUTPUT_DIR; win_len=WINDOW_LEN, win_step=WINDOW_STEP)
end

@info "All files segmented. Output directory: $OUTPUT_DIR"
