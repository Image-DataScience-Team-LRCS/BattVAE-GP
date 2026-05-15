#!/bin/bash

set -euo pipefail

if [[ "${PIPELINE_DETACHED:-0}" != "1" ]]; then
  TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
  LAUNCH_LOG_DIR="run_history/launcher_$TIMESTAMP"
  mkdir -p "$LAUNCH_LOG_DIR"
  nohup env PIPELINE_DETACHED=1 bash "$0" "$@" > "$LAUNCH_LOG_DIR/run.log" 2>&1 < /dev/null &
  PIPELINE_PID=$!
  echo "$PIPELINE_PID" > "$LAUNCH_LOG_DIR/pipeline.pid"
  echo "Pipeline detached with PID $PIPELINE_PID"
  echo "Launcher log: $LAUNCH_LOG_DIR/run.log"
  echo "Follow it with: tail -f $LAUNCH_LOG_DIR/run.log"
  exit 0
fi

source ../VAE/.venv/bin/activate

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="run_history/train_$TIMESTAMP"
mkdir -p "$LOG_DIR"

TRAIN_LOG="$LOG_DIR/train.log"
CONFIG_PATH="configs/vae.yaml"
DATASETS=(data1 data2 data3 data4 data5 data6 data7)

echo "Starting experiment at: $TIMESTAMP"
echo "Logs will be saved in $LOG_DIR"

export COLUMNS=150
echo "Training VAE"
echo "Tail logs with: tail -f $TRAIN_LOG"
python3 main.py --model vae --config "$CONFIG_PATH" --run train > "$TRAIN_LOG" 2>&1 < /dev/null
echo "Training completed successfully."

for dataset in "${DATASETS[@]}"; do
  INFER_LOG="$LOG_DIR/inference_${dataset}.log"
  echo "Running inference for $dataset"
  python3 main.py --model vae --config "$CONFIG_PATH" --dataset "$dataset" --run inference > "$INFER_LOG" 2>&1 < /dev/null

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

mv artifacts/latent_spaces_overlay_with_gp_interpolation.png artifacts/latent_spaces_overlay.png


echo "Training GP"
python3 main.py --model gp --run train > "$LOG_DIR/gp_train.log" 2>&1 < /dev/null

echo "Running GP inference"
python3 main.py --model gp --run interpolation > "$LOG_DIR/gp_interpolation.log" 2>&1 < /dev/null

python3 plot_all_latent_spaces.py --gp-data all > "$LOG_DIR/plot_gp_results.log" 2>&1

echo "Running interpolation"
python3 main.py --model vae --run interpolation > "$LOG_DIR/interpolation.log" 2>&1 < /dev/null

echo "Pipeline completed successfully." > "$LOG_DIR/completion.log"
