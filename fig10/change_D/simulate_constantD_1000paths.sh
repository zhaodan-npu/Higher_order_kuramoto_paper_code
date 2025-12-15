#!/bin/bash
#SBATCH --job-name=spike_constD_1000       # 作业名称
#SBATCH --account=ff4                      # 换成你的账户（如需要）
#SBATCH --output=spike_constD_1000-%A_%a.out  # 标准输出
#SBATCH --error=spike_constD_1000-%A_%a.err   # 标准错误
#SBATCH --qos=medium                       # 队列: short, medium, long, priority

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --array=0-999

## #SBATCH --mem=64G                       # 如需显式内存限制可打开

# ==================== 参数维度设置 ====================
# 必须和 spike_task_split.py 里的 alphas 和 D_vals 个数一致
num_alpha=10       # len(alphas)
num_D=10           # len(D_vals)
num_chunks=10     # 每组参数把 1000 条路径拆成 10 段
chunk_size=100    # 每段 100 条路径 -> 10 * 100 = 1000 条

# 总 job 数 = 5 * 5 * 10 = 250
# 请确保上面 SBATCH --array=0-249
global_id=${SLURM_ARRAY_TASK_ID}

# -------------- 将 global_id 映射到 (alpha_idx, D_idx, chunk_idx) --------------
# 思路：alpha 是最外层，D 次之，最后是 path-chunk
params_per_alpha=$(( num_D * num_chunks ))

alpha_idx=$(( global_id / params_per_alpha ))
rem=$(( global_id % params_per_alpha ))

D_idx=$(( rem / num_chunks ))
chunk_idx=$(( rem % num_chunks ))

start_path=$(( chunk_idx * chunk_size ))
end_path=$(( start_path + chunk_size ))

cd "$SLURM_SUBMIT_DIR"

export MPLBACKEND=Agg
module load python

echo "=============================================="
echo " SLURM_ARRAY_TASK_ID = ${global_id}"
echo " alpha_idx = ${alpha_idx} (0..$((num_alpha-1)))"
echo " D_idx     = ${D_idx}     (0..$((num_D-1)))"
echo " chunk_idx = ${chunk_idx} (0..$((num_chunks-1)))"
echo " paths     = ${start_path}..${end_path} (共 ${chunk_size} 条)"
echo "=============================================="

python spike_task_split.py \
    --alpha_idx "${alpha_idx}" \
    --D_idx "${D_idx}" \
    --start_path "${start_path}" \
    --end_path "${end_path}" \
    --outdir "results_constantD"

echo "Done: alpha_idx=${alpha_idx}, D_idx=${D_idx}, paths=${start_path}-${end_path}"
