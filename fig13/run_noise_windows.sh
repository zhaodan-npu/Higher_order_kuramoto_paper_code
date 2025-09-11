#!/usr/bin/env bash
#
# run_noise_windows.sh — 处理单个 noise_p*_spikes_win*.txt 窗口文件，调用 process_noise_windows.jl
#

#SBATCH --job-name=noise_edacf
#SBATCH --account=ff4
#SBATCH --output=logs/noise_edacf_%A_%a.out
#SBATCH --error=logs/noise_edacf_%A_%a.err
#SBATCH --qos=medium

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8    # Julia 线程数
#SBATCH --mem=8G

JULIA_EXE="/p/system/packages/tools/julia/1.11.0/bin/julia"

export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

# 进入提交目录（一般 run_noise_windows.sh 与 noise_windows/ 同级）
cd "$SLURM_SUBMIT_DIR"

echo "===== DEBUG: pwd = $(pwd)"
echo "===== DEBUG: ls -l ./noise_fixed/noise_windows/ ====="
ls -l ./noise_windows/


mkdir -p logs1 noise_results1

# 列出当前目录下所有 noise_p*_spikes_win*.txt
mapfile -t FILES < <(ls noise_windows/noise_a1.1_s0.1_p*_spikes_win*.txt 2>/dev/null)
N=${#FILES[@]}

: "${BATCH_START:=0}"
TASK_ID=$(( SLURM_ARRAY_TASK_ID + BATCH_START ))

if (( TASK_ID >= N )); then
  echo "[$SLURM_ARRAY_TASK_ID] No file to process (only $N files)."
  exit 0
fi

INPUT_FILE="${FILES[$TASK_ID]}"
OUTDIR="noise_results"

echo "[$SLURM_ARRAY_TASK_ID] Processing $INPUT_FILE → $OUTDIR"

# 调用 Julia 脚本。若使用绝对路径 Julia，请改成 "$JULIA_EXE"


srun "$JULIA_EXE" --threads $JULIA_NUM_THREADS process_noise_windows.jl \
    "$INPUT_FILE" "$OUTDIR"

echo "[$SLURM_ARRAY_TASK_ID] Done."
