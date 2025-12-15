#!/bin/bash
#SBATCH --job-name=fig3_double_det       # 作业名称
#SBATCH --account=ff4                    # 账户（按你们集群要求改）
#SBATCH --output=fig3_double_det-%j.out  # 标准输出
#SBATCH --error=fig3_double_det-%j.err   # 标准错误
#SBATCH --qos=medium                     # 队列: short, medium, long, priority

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

## #SBATCH --mem=64G                     # 如需显式内存可打开

cd "$SLURM_SUBMIT_DIR"

export MPLBACKEND=Agg

module load python

echo "Starting deterministic doubling test (fig3_doubling_det.py)..."
python fig3_doubling_det.py
echo "Done."
