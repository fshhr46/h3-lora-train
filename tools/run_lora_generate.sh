#!/bin/bash
# SDXL LoRA image generation wrapper for furnance.
#
# Stops llama-brain, generates images with trained LoRA on GPU 0,
# and ALWAYS restores the brain on exit (success, failure, or ctrl-c).
#
# Usage:
#   tools/run_lora_generate.sh
#     --lora_path output/models/lora/best_lora
#     --prompts prompts/positive_prompts.txt
#     [--output output/videos]
#     [--lora_weight 0.7]
#     [--num_steps 30]
#     [--seed 42]
#     [--gpu 0]
#
# One-liner SSH example:
#   ssh furnance 'cd ~/workspace/test-deepseek-harness && \
#     systemctl --user stop llama-brain && \
#     source .venv/bin/activate && \
#     HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 python generate.py \
#       --lora_path output/models/lora/best_lora \
#       --prompts prompts/positive_prompts.txt \
#       --output output/videos --lora_weight 0.7 --num_steps 30 && \
#     systemctl --user start llama-brain'

set -u
cd "$(dirname "$0")/.."

LORA_PATH="${LORA_PATH:-output/models/lora/best_lora}"
PROMPTS="${PROMPTS:-prompts/positive_prompts.txt}"
OUTPUT="${OUTPUT:-output/videos}"
LORA_WEIGHT="${LORA_WEIGHT:-0.7}"
NUM_STEPS="${NUM_STEPS:-30}"
SEED="${SEED:-}"
GPU="${GPU:-0}"

restore() { echo "[wrapper] $(date +%T) restoring llama-brain.service"; systemctl --user start llama-brain.service; }
trap restore EXIT

echo "[wrapper] $(date +%T) stopping llama-brain.service"
systemctl --user stop llama-brain.service
sleep 5

source .venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1

echo "[wrapper] $(date +%T) generate.py lora=$LORA_PATH prompts=$PROMPTS weight=$LORA_WEIGHT steps=$NUM_STEPS seed=$SEED"
python generate.py \
  --lora_path "$LORA_PATH" \
  --prompts "$PROMPTS" \
  --output "$OUTPUT" \
  --lora_weight "$LORA_WEIGHT" \
  --num_steps "$NUM_STEPS" \
  --seed "$SEED" \
  2>&1
rc=${PIPESTATUS[0]}
echo "[wrapper] $(date +%T) generate.py exit code $rc"
exit "$rc"
