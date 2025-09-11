#!/bin/bash
# run_noise_fixed.sh — SLURM batch script for fixed α=1.1, σ=0.1 Lévy noise

#SBATCH --job-name=levy_noise_fixed
#SBATCH --account=ff4
#SBATCH --output=logs/noise_fixed_%j.out
#SBATCH --error=logs/noise_fixed_%j.err
#SBATCH --qos=short

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

# 载入一个可以调用 python3 的模块（请根据集群实际情况改成正确的版本）
module load python

# 确保输出目录和日志目录存在
mkdir -p noise_fixed logs

echo "[Info] Generating Lévy noise (α=1.1, σ=0.1) with python3…"

# 用 python3 来运行生成噪声的脚本
srun python generate_levy.py --outdir noise_fixed

echo "[Done] Noise files saved in 'noise_fixed/'"
