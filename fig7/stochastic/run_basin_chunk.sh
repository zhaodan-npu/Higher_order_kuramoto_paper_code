#!/usr/bin/env bash
#SBATCH --job-name=basin_chunk
#SBATCH --account=**
#SBATCH --output=logs/BS_%A_%a.out
#SBATCH --error=logs/BS_%A_%a.err
#SBATCH --qos=medium

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G


# limit each chunk to array indices 0–999
#SBATCH --array=0-999%100

module load python

cd "$SLURM_SUBMIT_DIR"
# 激活你的虚拟环境
source /home/danzhao/myenv/bin/activate
mkdir -p basin_results logs

# OFFSET comes from master_submit.sh’s --export
TASK_ID=$(( SLURM_ARRAY_TASK_ID + OFFSET ))

echo "[$SLURM_ARRAY_TASK_ID] real task_id=$TASK_ID"
python compute_basin_task.py --task_id "$TASK_ID"
