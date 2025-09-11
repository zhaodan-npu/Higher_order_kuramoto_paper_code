#!/usr/bin/env bash
#SBATCH --job-name=edacf_edspec
#SBATCH --account=**
#SBATCH --output=edacf_edspec-%A_%a.out
#SBATCH --error=edacf_edspec-%A_%a.err
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

# 根据实际文件数量调整 0-<N-1>
#SBATCH --array=0-99

module load julia
export JULIA_DEPOT_PATH="$HOME/.julia"

cd "$SLURM_SUBMIT_DIR"

RESULTS_DIR="/p"
OUTDIR="/p/tmp"
mkdir -p "$OUTDIR"

alpha_idx=3  # alpha=1.4
sigma_idx=33 # sigma=0.4
FILES=("$RESULTS_DIR"/spikes_a"${alpha_idx}"_s"${sigma_idx}"_p*.txt)
INPUT=${FILES[$SLURM_ARRAY_TASK_ID]}

echo "[$SLURM_ARRAY_TASK_ID] Processing $INPUT"

julia sinle_path_all_revised.jl "$INPUT" "$OUTDIR"
echo "[$SLURM_ARRAY_TASK_ID] Done."
