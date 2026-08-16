#!/bin/bash
# MiniMax H3 视频生成（经 vLLM-Omni 服务，见 tools/h3_server.sh）
#
# 用法:
#   tools/run_h3.sh --first <图片> [--last <图片>] --prompt "文字"        # 首帧(+尾帧) → 视频 (fl2va)
#   tools/run_h3.sh --prompt "文字" [--aspect 16:9]                          # 纯文生视频 (t2va)
#   tools/run_h3.sh --first a.png --last b.png --prompt-file ~/shared/h3prompt.txt
#
# 图片路径可以是 furnance 本地路径，也可以直接给 SMB 地址:
#   smb://100.89.241.44/shared/lora-images/xxx/lora_test_0002.png  → /home/fshhr46/shared/lora-images/xxx/lora_test_0002.png
#
# 可选: --duration 5 (秒, 4–15)  --steps 50  --seed 42  --name 批次名  --width W --height H
#       --flow-shift 12  --audio-flow-shift 3.0  --dry-run (只打印 curl 不执行)
# 环境变量: H3_URL (默认 http://127.0.0.1:8091)
# 输出: ~/shared/h3-videos/<时间>-<批次名>/video.mp4 + request.json + 首/尾帧副本
#       即 smb://guest@100.89.241.44/shared/h3-videos/...
set -u

H3_URL="${H3_URL:-http://127.0.0.1:8091}"
FIRST=""; LAST=""; PROMPT=""; PROMPT_FILE=""; DURATION="5"; STEPS="50"; SEED="42"; NAME=""
WIDTH=""; HEIGHT=""; ASPECT="16:9"; FLOW_SHIFT="12"; AUDIO_FLOW_SHIFT="3.0"; DRY=0

while [ $# -gt 0 ]; do
  case "$1" in
    --first) FIRST="$2"; shift 2;;
    --last) LAST="$2"; shift 2;;
    --prompt) PROMPT="$2"; shift 2;;
    --prompt-file) PROMPT_FILE="$2"; shift 2;;
    --duration) DURATION="$2"; shift 2;;
    --steps) STEPS="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;
    --name) NAME="$2"; shift 2;;
    --width) WIDTH="$2"; shift 2;;
    --height) HEIGHT="$2"; shift 2;;
    --aspect) ASPECT="$2"; shift 2;;
    --flow-shift) FLOW_SHIFT="$2"; shift 2;;
    --audio-flow-shift) AUDIO_FLOW_SHIFT="$2"; shift 2;;
    --dry-run) DRY=1; shift;;
    -h|--help) sed -n '2,20p' "$0"; exit 0;;
    *) echo "未知参数: $1"; exit 2;;
  esac
done

# smb:// 或 /Volumes/shared 路径 → furnance 本地路径
to_local() {
  local p="$1"
  p="${p#smb://guest@}"; p="${p#smb://}"
  case "$p" in
    100.89.241.44/shared/*) p="$HOME/shared/${p#100.89.241.44/shared/}";;
    furnance/shared/*)      p="$HOME/shared/${p#furnance/shared/}";;
    /Volumes/shared/*)      p="$HOME/shared/${p#/Volumes/shared/}";;
  esac
  printf '%s' "$p"
}

if [ -n "$PROMPT_FILE" ]; then
  PROMPT="$(grep -vE '^\s*#|^\s*$' "$PROMPT_FILE" | tr '\n' ' ' | sed 's/ *$//')"
fi
[ -n "$PROMPT" ] || { echo "❌ 需要 --prompt 或 --prompt-file"; exit 2; }

TASK="t2va"; FRAME_IDX=""
if [ -n "$FIRST" ]; then
  FIRST="$(to_local "$FIRST")"; [ -f "$FIRST" ] || { echo "❌ 首帧不存在: $FIRST"; exit 2; }
  TASK="fl2va"; FRAME_IDX='[0]'
  if [ -n "$LAST" ]; then
    LAST="$(to_local "$LAST")"; [ -f "$LAST" ] || { echo "❌ 尾帧不存在: $LAST"; exit 2; }
    FRAME_IDX='[0,-1]'
  fi
elif [ -n "$LAST" ]; then
  echo "❌ 只给尾帧不支持，请同时给 --first"; exit 2
fi

# 服务健康检查（--dry-run 跳过）
if [ "$DRY" != 1 ] && ! curl -s -m 5 -o /dev/null "$H3_URL/health"; then
  echo "❌ H3 服务不可达: $H3_URL   （启动: tools/h3_server.sh start；查看: tools/h3_server.sh status）"; exit 1
fi

TS=$(date +%Y%m%d-%H%M%S); NAME="${NAME:-$TASK}"
OUT="$HOME/shared/h3-videos/${TS}-${NAME}"; mkdir -p "$OUT"
EXTRA="{\"task\":\"$TASK\",\"duration\":$DURATION,\"audio_flow_shift\":$AUDIO_FLOW_SHIFT,\"flow_shift\":$FLOW_SHIFT"
[ -n "$FRAME_IDX" ] && EXTRA="$EXTRA,\"frame_indices\":$FRAME_IDX"
EXTRA="$EXTRA}"

args=(-sS -X POST "$H3_URL/v1/videos/sync" --max-time 3600
      -F "prompt=$PROMPT" -F "fps=24" -F "num_inference_steps=$STEPS" -F "flow_shift=$FLOW_SHIFT" -F "seed=$SEED"
      -F "extra_params=$EXTRA" -o "$OUT/video.mp4")
[ "$TASK" = "t2va" ] && args+=(-F "aspect_ratio=$ASPECT")
[ -n "$WIDTH" ]  && args+=(-F "width=$WIDTH")
[ -n "$HEIGHT" ] && args+=(-F "height=$HEIGHT")
if [ -n "$FIRST" ] && [ -n "$LAST" ]; then
  args+=(-F "input_references=@$FIRST" -F "input_references=@$LAST")
elif [ -n "$FIRST" ]; then
  args+=(-F "input_reference=@$FIRST")
fi

# 记录请求
python3 - "$OUT/request.json" "$TASK" "$PROMPT" "$FIRST" "$LAST" "$DURATION" "$STEPS" "$SEED" "$WIDTH" "$HEIGHT" "$ASPECT" "$EXTRA" <<'PY'
import json,sys
k=["task","prompt","first","last","duration","steps","seed","width","height","aspect","extra_params"]
json.dump(dict(zip(k,sys.argv[2:])),open(sys.argv[1],"w"),ensure_ascii=False,indent=2)
PY
[ -n "$FIRST" ] && cp "$FIRST" "$OUT/first.${FIRST##*.}"
[ -n "$LAST" ]  && cp "$LAST"  "$OUT/last.${LAST##*.}"

echo "🎬 task=$TASK duration=${DURATION}s steps=$STEPS seed=$SEED"
echo "   prompt: ${PROMPT:0:120}"
[ -n "$FIRST" ] && echo "   first:  $FIRST"
[ -n "$LAST" ]  && echo "   last:   $LAST"
echo "   out:    $OUT/video.mp4"
if [ "$DRY" = 1 ]; then printf 'curl'; printf ' %q' "${args[@]}"; echo; exit 0; fi

t0=$(date +%s)
if curl "${args[@]}"; then
  if head -c 4 "$OUT/video.mp4" | grep -q "ftyp" || [ "$(stat -c %s "$OUT/video.mp4")" -gt 100000 ]; then
    echo "✅ 完成 $(( $(date +%s)-t0 ))s → $OUT/video.mp4  ($(du -h "$OUT/video.mp4" | cut -f1))"
    echo "   SMB: smb://guest@100.89.241.44/shared/h3-videos/${TS}-${NAME}/video.mp4"
  else
    echo "❌ 服务返回的不是视频（可能是错误 JSON）:"; head -c 800 "$OUT/video.mp4"; echo; mv "$OUT/video.mp4" "$OUT/error.txt"; exit 1
  fi
else
  echo "❌ 请求失败（$(( $(date +%s)-t0 ))s）"; exit 1
fi
