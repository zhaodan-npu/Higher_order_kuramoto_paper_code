#!/bin/bash
#SBATCH --job-name=edacf_conf_final
#SBATCH --account=ff4
#SBATCH --output=logs/edacf_conf_%A_%a.out
#SBATCH --error=logs/edacf_conf_%A_%a.err
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --array=0-44

module purge
module load julia/1.10.0

JULIA_EXE="/p/system/packages/tools/julia/1.11.0/bin/julia"
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results

FILES=(spikes_a0_s0_p1_win*.txt)

if (( SLURM_ARRAY_TASK_ID >= ${#FILES[@]} )); then
    echo "Task ID out of range."
    exit 1
fi

INPUT_FILE="${FILES[$SLURM_ARRAY_TASK_ID]}"
OUTDIR="results"

echo "[Task $SLURM_ARRAY_TASK_ID] Processing $INPUT_FILE → $OUTDIR"
srun "$JULIA_EXE" --threads $JULIA_NUM_THREADS process_path_1.jl "$INPUT_FILE" "$OUTDIR"
echo "[Task $SLURM_ARRAY_TASK_ID] Done"
