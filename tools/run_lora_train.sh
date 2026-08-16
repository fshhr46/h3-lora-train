#!/bin/bash
# SDXL LoRA training wrapper for furnance.
#
# Both 4090s normally belong to llama-brain.service (the dsh "furnance" model).
# This script stops the brain, trains on GPU 0, and ALWAYS restores the brain
# on exit (success, failure, or `systemctl --user stop lora-train`).
#
# Run it as a transient user unit so it survives SSH drops:
#   systemd-run --user --unit=lora-train --collect tools/run_lora_train.sh
#
# Override hyper-parameters via env vars, e.g. EPOCHS=20 DATA_DIR=data/filtered.
set -u
cd "$(dirname "$0")/.."

DATA_DIR="${DATA_DIR:-data/cropped}"
OUT="${OUT:-output/models/lora}"
EPOCHS="${EPOCHS:-15}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LR="${LR:-1e-4}"
RES="${RES:-512,512}"
GRAD_ACC="${GRAD_ACC:-4}"
SAVE_EVERY="${SAVE_EVERY:-2}"
SEED="${SEED:-42}"
GPU="${GPU:-0}"

mkdir -p "$OUT"
LOG="$OUT/training_log.txt"

restore() { echo "[wrapper] $(date +%T) restoring llama-brain.service"; systemctl --user start llama-brain.service; }
trap restore EXIT

echo "[wrapper] $(date +%T) stopping llama-brain.service to free the GPUs"
systemctl --user stop llama-brain.service
sleep 5
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

source .venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1

echo "[wrapper] $(date +%T) starting train_lora.py (data=$DATA_DIR out=$OUT epochs=$EPOCHS)"
python train_lora.py \
  --data_dir "$DATA_DIR" --output_dir "$OUT" \
  --lora_rank "$LORA_RANK" --lora_alpha "$LORA_ALPHA" --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" \
  --learning_rate "$LR" --resolution "$RES" --gradient_accumulation "$GRAD_ACC" \
  --save_every "$SAVE_EVERY" --seed "$SEED" \
  2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
echo "[wrapper] $(date +%T) train_lora.py exit code $rc"
exit "$rc"
