#!/bin/bash

set -euo pipefail

source ../VAE/.venv/bin/activate

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="run_history/train_$TIMESTAMP"
mkdir -p "$LOG_DIR"

TRAIN_LOG="$LOG_DIR/train.log"
GPU_LOG="$LOG_DIR/gpu_mem.log"
RAM_LOG="$LOG_DIR/ram_usage.log"
PID_FILE="$LOG_DIR/pid.txt"
CONFIG_PATH="configs/vae.yaml"
DATASETS=(data1 data2 data3 data4 data5 data6 data7)

echo "Starting experiment at: $TIMESTAMP"
echo "Logs will be saved in $LOG_DIR"

export COLUMNS=150
 nohup python3 main.py --model vae --config "$CONFIG_PATH" --run train > "$TRAIN_LOG" 2>&1 < /dev/null &

 PID=$!
 echo "$PID" > "$PID_FILE"
 echo "Training started with PID $PID"

 (
   while kill -0 "$PID" 2>/dev/null; do
     echo "$(date +%F' '%T) - RAM: $(free -m | awk '/^Mem:/ {print $3 "MB / " $2 "MB"}')" >> "$RAM_LOG"
     if command -v nvidia-smi >/dev/null 2>&1; then
       nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total \
         --format=csv,noheader,nounits >> "$GPU_LOG"
     fi
     sleep 30
   done
   echo "Monitoring stopped for PID $PID." >> "$RAM_LOG"
   if command -v nvidia-smi >/dev/null 2>&1; then
     echo "Monitoring stopped for PID $PID." >> "$GPU_LOG"
   fi
 ) &
 MONITOR_PID=$!

 echo "PID: $PID"
 echo "Tail logs with: tail -f $TRAIN_LOG"

 wait "$PID"
 wait "$MONITOR_PID" || true
 echo "Training completed successfully."

for dataset in "${DATASETS[@]}"; do
  INFER_LOG="$LOG_DIR/inference_${dataset}.log"
  INFER_PID_FILE="$LOG_DIR/inference_${dataset}.pid"
  echo "Running inference for $dataset"
  nohup python3 main.py --model vae --config "$CONFIG_PATH" --dataset "$dataset" --run inference > "$INFER_LOG" 2>&1 < /dev/null &
  INFER_PID=$!
  echo "$INFER_PID" > "$INFER_PID_FILE"
  wait "$INFER_PID"

  latent_src="artifacts/latent_space"
  latent_dst="artifacts/latent_space_${dataset}"
  viz_src="artifacts/visualizations"
  viz_dst="artifacts/visualizations_${dataset}"

  [[ -d "$latent_src" ]] || { echo "Missing latent output for $dataset: $latent_src"; exit 1; }
  [[ -d "$viz_src" ]] || { echo "Missing visualization output for $dataset: $viz_src"; exit 1; }

  rm -rf "$latent_dst" "$viz_dst"
  mv "$latent_src" "$latent_dst"
  mv "$viz_src" "$viz_dst"
done

python3 plot_all_latent_spaces.py > "$LOG_DIR/plot_all_latent_spaces.log" 2>&1
echo "Pipeline completed successfully." > END
