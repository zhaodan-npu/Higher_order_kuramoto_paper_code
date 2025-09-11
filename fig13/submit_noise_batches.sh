#!/usr/bin/env bash
#
# submit_noise_batches.sh — 按 MaxArraySize=1001 分批提交 run_noise_windows.sh
#

# -------- 配置区 --------
# 存放 run_noise_windows.sh 的目录（脚本与 noise_windows/ 同级）
BASE_DIR="$(dirname "$(readlink -f "$0")")"

# 每批最多可以提交的数组任务数量（0…1000 共 1001 个任务）
BATCH_SIZE=1001

# 同一批内部最大并发任务数
CONCURRENCY=200
# ------------------------

cd "$BASE_DIR"

# 统计 noise_windows/ 下所有 window 文件数量
N=$(ls noise_windows/noise_a1.1_s0.1_p*_spikes_win*.txt 2>/dev/null | wc -l | xargs)
if (( N == 0 )); then
  echo "[Error] 在目录 'noise_windows/' 中未找到任何 noise_a1.1_s0.1_p*_spikes_win*.txt 文件。"
  exit 1
fi

echo "Total window files: $N. Submitting in batches of $BATCH_SIZE …"

# 分批提交：全局索引从 0 到 N-1
for (( start=0; start<N; start += BATCH_SIZE )); do
  end=$(( start + BATCH_SIZE - 1 ))
  (( end >= N )) && end=$(( N - 1 ))
  max_idx=$(( end - start ))
  echo "⇒ Submitting batch: global indices $start–$end  as array 0–$max_idx"
  sbatch \
    --array=0-${max_idx}%${CONCURRENCY} \
    --export=BATCH_START=${start} \
    "${BASE_DIR}/run_noise_windows.sh"
done
