#!/bin/bash
# gen-queue（FastAPI FIFO 队列服务）管理：start | stop | restart | status | logs
# 服务：tools/gen_server.py，systemd 用户单元 gen-api，监听 0.0.0.0:8090（tailnet 内 http://100.89.241.44:8090）
# 队列：image → gpu2(索引1) SDXL/Chroma；video → gpu1(索引0) H3(ComfyUI 后端，按需自动拉起)
set -u
UNIT=gen-api
cd "$(dirname "$0")/.."
PY="$PWD/.venv/bin/python"
LOG=/mnt/elements/logs/gen-api.log
case "${1:-status}" in
  start)
    [ -x "$PY" ] || { echo "❌ 没有 .venv"; exit 1; }
    "$PY" -c "import fastapi, uvicorn" 2>/dev/null || { echo "安装 fastapi/uvicorn..."; source .venv/bin/activate; (uv pip install -q fastapi uvicorn 2>/dev/null || pip install -q fastapi uvicorn); }
    systemctl --user reset-failed $UNIT.service 2>/dev/null
    mkdir -p "$(dirname "$LOG")" /mnt/elements/gen-queue
    systemd-run --user --unit=$UNIT --collect --description="gen-queue FastAPI (image gpu2 / video gpu1) :8090" \
      -p WorkingDirectory="$PWD" -p Restart=on-failure -p RestartSec=5 \
      -E GENQ_PORT=8090 -E GENQ_DIR=/mnt/elements/gen-queue -E COMFY_URL=http://127.0.0.1:8188 -E GPU_IMAGE=1 -E GPU_VIDEO=0 \
      bash -c "exec $PY tools/gen_server.py >> $LOG 2>&1"
    sleep 3; echo "🚀 $UNIT: $(systemctl --user is-active $UNIT.service)  → http://100.89.241.44:8090/docs";;
  stop)    systemctl --user stop $UNIT.service; echo "⏹  已停止 $UNIT";;
  restart) "$0" stop; sleep 1; "$0" start;;
  status)
    echo "unit: $(systemctl --user is-active $UNIT.service 2>/dev/null)"
    curl -s -m 5 http://127.0.0.1:8090/queues 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "api down";;
  logs)    tail -n "${2:-40}" -f "$LOG";;
  *) echo "用法: $0 {start|stop|restart|status|logs [N]}"; exit 2;;
esac
