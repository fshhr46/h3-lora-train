#!/bin/bash
# MiniMax H3 服务（vLLM-Omni）管理：start | stop | status | logs
#
# - 单卡：默认 GPU 1（GPU 0 留给 SDXL 出图）；跑 H3 时 llama-brain 必须是停止状态（brain 占满两张卡）
# - 权重：官方 bf16 MiniMaxAI/MiniMax-H3 的 FL2VA 分区，缓存在 HF_HOME=/mnt/elements/hf-cache（外接盘）
# - 加载时 FP8 量化（DiT + 文本编码器）+ 模型级 CPU offload；内存常驻约 70 GB
# - 首次启动要把 ~130 GB 权重读入并量化，可能 10–20 分钟；之后每条请求几分钟
# 环境变量: H3_GPU(默认1) H3_PORT(8091) H3_TASK(fl2va)
set -u
UNIT=h3-server
VENV="$HOME/vllm-omni/.venv"
H3_GPU="${H3_GPU:-1}"; H3_PORT="${H3_PORT:-8091}"; H3_TASK="${H3_TASK:-fl2va}"
LOG=/mnt/elements/logs/h3-server.log

case "${1:-status}" in
  start)
    if [ "$(systemctl --user is-active llama-brain.service)" = active ]; then
      echo "⚠️  llama-brain 正在运行并占用两张卡。先: systemctl --user stop llama-brain.service"; exit 1
    fi
    [ -x "$VENV/bin/vllm" ] || { echo "❌ 没找到 $VENV/bin/vllm（vLLM-Omni 未装好）"; exit 1; }
    systemctl --user reset-failed $UNIT.service 2>/dev/null
    mkdir -p "$(dirname "$LOG")"
    systemd-run --user --unit=$UNIT --collect --description="MiniMax H3 (vLLM-Omni) on GPU $H3_GPU :$H3_PORT" \
      -E HF_HOME=/mnt/elements/hf-cache -E HF_HUB_OFFLINE=1 \
      -E CUDA_VISIBLE_DEVICES="$H3_GPU" -E VLLM_OMNI_VIDEO_SYNC_TIMEOUT=3600 -E VLLM_WORKER_MULTIPROC_METHOD=spawn \
      -p WorkingDirectory="$HOME/vllm-omni" \
      bash -c "source $VENV/bin/activate && exec vllm serve MiniMaxAI/MiniMax-H3 --omni --trust-remote-code \
        --host 0.0.0.0 --port $H3_PORT --num-gpus 1 --enable-cpu-offload --quantization fp8 \
        --task-type $H3_TASK --enforce-eager --diffusion-attention-backend FLASH_ATTN --vae-use-tiling \
        >> $LOG 2>&1"
    echo "🚀 已启动 $UNIT（GPU $H3_GPU，端口 $H3_PORT）。首次加载需 10–20 分钟。"
    echo "   进度: tools/h3_server.sh logs    就绪判断: curl -s http://127.0.0.1:$H3_PORT/health"
    ;;
  stop)   systemctl --user stop $UNIT.service; echo "⏹  已停止 $UNIT";;
  status)
    echo "unit:   $(systemctl --user is-active $UNIT.service 2>/dev/null)"
    echo "health: $(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:$H3_PORT/health 2>/dev/null || echo down)"
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
    free -g | awk 'NR==2{print "RAM used/total: "$3"/"$2" GB"}'
    ;;
  logs)   tail -n "${2:-40}" -f "$LOG";;
  *) echo "用法: $0 {start|stop|status|logs [N]}"; exit 2;;
esac
