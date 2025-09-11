#!/bin/bash
#SBATCH --job-name=my_python_job
#SBATCH --account=**

#SBATCH --output=python_job-%j.out
#SBATCH --error=python_job-%j.err
#SBATCH --qos=medium  # short, medium, long, priority


#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --array=0-99            # 100 个任务，对应 k1s 的每个 index

module load python          # 根据实际环境改
# source ~/venv/bin/activate    # 如有虚拟环境

mkdir -p results logs

srun python 01_run_simulation_slice.py \
     --k1_index $SLURM_ARRAY_TASK_ID \
     --outdir results


