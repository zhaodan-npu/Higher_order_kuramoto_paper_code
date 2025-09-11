#!/bin/bash
#SBATCH --job-name=spike_array_1000paths
#SBATCH --account=**
#SBATCH --output=logs/spike_array_%A_%a.out
#SBATCH --error=logs/spike_array_%A_%a.err
#SBATCH --qos=medium
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-999  # ←这里修改为小于集群限制的数量

module load python
cd "$SLURM_SUBMIT_DIR"

mkdir -p logs
mkdir -p ***

alphas=(0 1 2 3 4 5 6 7 8 9)
sigmas=({0..99})
path_starts=(0 100 200 300 400 500 600 700 800 900)
path_ends=(100 200 300 400 500 600 700 800 900 1000)

global_id=$SLURM_ARRAY_TASK_ID

alpha_idx=${alphas[$(( global_id / (100*10) ))]}
sigma_idx=${sigmas[$(( (global_id / 10) % 100 ))]}
path_idx=$(( global_id % 10 ))
start_path=${path_starts[$path_idx]}
end_path=${path_ends[$path_idx]}

echo "[Task $global_id] alpha_idx=$alpha_idx sigma_idx=$sigma_idx paths=$start_path-$end_path"

python spike_task_split.py \
    --alpha_idx $alpha_idx \
    --sigma_idx $sigma_idx \
    --start_path $start_path \
    --end_path $end_path \
    --outdir "/p/tmp/danzhao/results_1000paths"
