#!/bin/bash
# MiniMax H3 视频生成（方案 B：ComfyUI 无界面后端）。参数见 tools/h3_comfy.py --help
# 例: tools/run_h3.sh --first smb://100.89.241.44/shared/lora-images/X/a.png --last .../b.png --prompt "..." --duration 5 --name test1
exec "$HOME/miniforge3/envs/comfy/bin/python" "$(dirname "$0")/h3_comfy.py" "$@"
