#!/bin/bash
# MiniMax H3 后端（方案 B）：无界面 ComfyUI 服务管理 —— start | stop | status | logs
#
# - ComfyUI ~/ComfyUI（0.30.0，内置 H3 节点），conda 环境 comfy
# - 权重：/mnt/elements/models/minimax-h3/（Comfy-Org 官方 int8 DiT + nvfp4 编码器 + 2 个 VAE，已软链进 ~/ComfyUI/models）
# - 单卡：默认 GPU 索引 0（gen-mode: gpu1=视频, gpu2(索引1)=图片）；跑 H3 时 llama-brain 必须停（它占满两张卡）
# - 只监听 127.0.0.1:8188，不开界面；由 tools/h3_comfy.py / tools/run_h3.sh 通过 HTTP API 驱动
# 环境变量: H3_GPU(默认0 = gpu1 视频卡) COMFY_PORT(8188)
set -u
UNIT=h3-comfy
H3_GPU="${H3_GPU:-0}"; PORT="${COMFY_PORT:-8188}"
PY="$HOME/miniforge3/envs/comfy/bin/python"
LOG=/mnt/elements/logs/h3-comfy.log

case "${1:-status}" in
  start)
    if [ "$(systemctl --user is-active llama-brain.service)" = active ]; then
      echo "⚠️  llama-brain 正在运行并占用两张卡。先: systemctl --user stop llama-brain.service"; exit 1
    fi
    [ -x "$PY" ] || { echo "❌ 没找到 $PY"; exit 1; }
    systemctl --user reset-failed $UNIT.service 2>/dev/null
    mkdir -p "$(dirname "$LOG")"
    systemd-run --user --unit=$UNIT --collect --description="ComfyUI headless for MiniMax H3 (GPU $H3_GPU :$PORT)" \
      -E CUDA_VISIBLE_DEVICES="$H3_GPU" -E PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      -p WorkingDirectory="$HOME/ComfyUI" \
      bash -c "exec $PY main.py --listen 127.0.0.1 --port $PORT --disable-auto-launch --dont-print-server >> $LOG 2>&1"
    echo "🚀 已启动 $UNIT（GPU $H3_GPU，端口 $PORT）。首个请求会加载权重（~40 GB，从外接盘读约 3–5 分钟）。"
    ;;
  stop)   systemctl --user stop $UNIT.service; echo "⏹  已停止 $UNIT";;
  status)
    echo "unit:   $(systemctl --user is-active $UNIT.service 2>/dev/null)"
    echo "api:    $(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/system_stats 2>/dev/null || echo down)"
    curl -s -m 3 http://127.0.0.1:$PORT/queue 2>/dev/null | python3 -c "import sys,json; q=json.load(sys.stdin); print(f'queue:  running={len(q.get(\"queue_running\",[]))} pending={len(q.get(\"queue_pending\",[]))}')" 2>/dev/null
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
    free -g | awk 'NR==2{print "RAM used/total: "$3"/"$2" GB"}'
    ;;
  logs)   tail -n "${2:-40}" -f "$LOG";;
  *) echo "用法: $0 {start|stop|status|logs [N]}"; exit 2;;
esac
