#!/usr/bin/env bash
for OFFSET in 0 1000 2000; do
  echo "Submitting chunk with OFFSET=$OFFSET"
  sbatch --export=OFFSET=$OFFSET run_basin_chunk.sh
done
