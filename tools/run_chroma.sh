#!/bin/bash
# Chroma1-HD 出图：读 ~/shared/prompt.txt（SMB 可编辑）→ ~/shared/chroma-images/<时间>-<批次名>/
# 用法: tools/run_chroma.sh [批次名]
# 环境变量: N(每条 prompt 出几张,默认1) STEPS(26) GUIDANCE(3.5) SEED(42) W(1024) H(1024) GPU(1 = gpu2 图片卡)
#           PROMPTS(~/shared/prompt.txt) NEG(负面词文件或文字,默认空) FP8=1(transformer fp8 存储,更省显存)
# gen-mode 下 gpu2(索引1) 专用于图片；brain mode 下不能跑（brain 占满两卡）。
set -u
cd "$(dirname "$0")/.."
NAME="${1:-prompt}"; TS=$(date +%Y%m%d-%H%M)
PROMPTS="${PROMPTS:-$HOME/shared/prompt.txt}"
OUT="$HOME/shared/chroma-images/${TS}-${NAME}"; mkdir -p "$OUT"
n=$(grep -vE '^\s*#|^\s*$' "$PROMPTS" | wc -l); [ "$n" -gt 0 ] || { echo "prompt 文件里没有有效行: $PROMPTS"; exit 1; }
cp "$PROMPTS" "$OUT/prompts.txt"
N="${N:-1}"
if [ "$N" -gt 1 ]; then
  EXP="$OUT/prompts_expanded.txt"; grep -vE '^\s*#|^\s*$' "$PROMPTS" | awk -v n="$N" '{for(i=0;i<n;i++) print}' > "$EXP"; PROMPTS="$EXP"
fi
if [ "$(systemctl --user is-active llama-brain.service)" = active ]; then echo "⚠️  brain mode 中（llama-brain 在跑），先切到 gen-mode: systemctl --user stop llama-brain.service"; exit 1; fi
source .venv/bin/activate
export HF_HOME=/mnt/elements/hf-cache HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES="${GPU:-1}" PYTHONUNBUFFERED=1
extra=(); [ -n "${NEG:-}" ] && extra+=(--negative_prompts "$NEG"); [ "${FP8:-0}" = 1 ] && extra+=(--fp8)
python tools/generate_chroma.py --prompts "$PROMPTS" --output "$OUT" --num_steps "${STEPS:-26}" --guidance "${GUIDANCE:-3.5}" \
  --seed "${SEED:-42}" --width "${W:-1024}" --height "${H:-1024}" "${extra[@]}" \
  2>&1 | tee "$OUT/generation_log.txt" | grep -E "^\[|💾|🎉|✅|Traceback|Error|❌"
echo "OUT=$OUT   (smb://guest@100.89.241.44/shared/chroma-images/${TS}-${NAME}/)"
