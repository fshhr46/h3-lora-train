#!/bin/bash
# 读取 ~/shared/prompt.txt（SMB 可编辑），用 SDXL(+可选 LoRA) 出图到 ~/shared/lora-images/<ts>-<name>/
# 用法: tools/run_gen.sh [批次名]      环境变量: BASE_MODEL(sdxl|realvis|juggernaut|repo id, 默认 realvis) N(每条prompt出几张,默认1) LORA_WEIGHT(默认0.0) STEPS(30) SEED(42) W(1024) H(1024) GPU(1 = gpu2 图片卡) PROMPTS(~/shared/prompt.txt)
# 注意: 需要 GPU 空闲；若 llama-brain 在跑，先 systemctl --user stop llama-brain.service，跑完 start。
set -u
cd "$(dirname "$0")/.."
NAME="${1:-prompt}"; TS=$(date +%Y%m%d-%H%M)
PROMPTS="${PROMPTS:-$HOME/shared/prompt.txt}"
OUT="$HOME/shared/lora-images/${TS}-${NAME}"; mkdir -p "$OUT"
n=$(grep -vE '^\s*#|^\s*$' "$PROMPTS" | wc -l); [ "$n" -gt 0 ] || { echo "prompt 文件里没有有效行: $PROMPTS"; exit 1; }
cp "$PROMPTS" "$OUT/prompts.txt"
N="${N:-1}"
if [ "$N" -gt 1 ]; then  # 每条 prompt 重复 N 行 → 每条出 N 张（seed 递增）
  EXP="$OUT/prompts_expanded.txt"; grep -vE '^\s*#|^\s*$' "$PROMPTS" | awk -v n="$N" '{for(i=0;i<n;i++) print}' > "$EXP"; PROMPTS="$EXP"
fi
source .venv/bin/activate
export HF_HOME=/mnt/elements/hf-cache HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES="${GPU:-1}" PYTHONUNBUFFERED=1
python generate.py --base_model "${BASE_MODEL:-realvis}" --lora_path "${LORA_PATH:-none}" --prompts "$PROMPTS" --output "$OUT" \
  --lora_weight "${LORA_WEIGHT:-0.0}" --num_steps "${STEPS:-30}" --seed "${SEED:-42}" --width "${W:-1024}" --height "${H:-1024}" \
  2>&1 | tee "$OUT/generation_log.txt" | grep -E "^\[|💾|🎉|Traceback|Error"
echo "OUT=$OUT"
