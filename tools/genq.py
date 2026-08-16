#!/usr/bin/env python3
"""
genq.py — gen-queue 的命令行客户端（只依赖标准库，任何有 python3 的机器都能用）。

  genq.py image  "prompt 文字" [-m sdxl|chroma] [--base realvis|juggernaut|sdxl|noobai] [--lora 文件名 --lora_weight 0.8] [-n 4] [--steps 36] [--guidance 4.5] [--seed 7] [--w 832 --h 1216] [--neg "..."] [--name batch] [--wait]
  genq.py image  -f ~/shared/prompt.txt  ...            # 从文件读多条 prompt（# 和空行忽略）
  genq.py video  "prompt 文字" [--first 图片] [--last 图片] [--duration 5] [--steps 20] [--seed 42] [--name batch] [--wait]
  genq.py status                                       # 两个队列状态
  genq.py list   [-q image|video] [-s pending|running|done|failed] [-l 20]
  genq.py get    <job_id>          genq.py cancel <job_id>          genq.py wait <job_id>
环境变量 GENQ_URL（默认 http://127.0.0.1:8090；从 bastion/runtime 用 http://100.89.241.44:8090）
"""
import argparse, json, os, sys, time, urllib.request, urllib.error
from pathlib import Path

URL = os.environ.get("GENQ_URL", "http://127.0.0.1:8090").rstrip("/")


def call(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(URL + path, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r: return json.loads(r.read())
    except urllib.error.HTTPError as e:
        sys.exit(f"❌ {e.code}: {e.read().decode()[:500]}")
    except Exception as e:
        sys.exit(f"❌ 连不上 {URL}: {e}  （furnance: tools/gen_server.sh start）")


def show(j):
    print(f"[{j['id']}] {j['queue']:5s} {j['status']:9s}" + (f" pos={j['position']}" if j.get('position') is not None else "") +
          (f"  → {j.get('smb') or j.get('output_dir')}" if j.get('output_dir') else "") + (f"  ⚠ {j['error'][:120]}" if j.get('error') else ""))


def wait(jid):
    last = None
    while True:
        j = call("GET", f"/jobs/{jid}")
        if j["status"] != last: show(j); last = j["status"]
        if j["status"] in ("done", "failed", "cancelled", "interrupted"): return j
        time.sleep(5)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("image"); a.add_argument("prompt", nargs="?"); a.add_argument("-f", "--file")
    a.add_argument("-m", "--model", default="sdxl", choices=["chroma", "sdxl"]); a.add_argument("--base", help="sdxl 底模: sdxl|realvis|juggernaut|<repo id>（默认 realvis）"); a.add_argument("-n", type=int, default=1)
    a.add_argument("--steps", type=int); a.add_argument("--guidance", type=float); a.add_argument("--seed", type=int)
    a.add_argument("--w", type=int); a.add_argument("--h", type=int); a.add_argument("--neg"); a.add_argument("--lora", help="sdxl 单文件 LoRA 名字/路径（在 /mnt/elements/models/loras/sdxl/ 下，多个逗号分隔）或 PEFT 目录"); a.add_argument("--lora_weight", type=float)
    a.add_argument("--name"); a.add_argument("--wait", action="store_true")
    v = sub.add_parser("video"); v.add_argument("prompt"); v.add_argument("--first"); v.add_argument("--last")
    v.add_argument("--duration", type=float, default=5); v.add_argument("--steps", type=int, default=20); v.add_argument("--seed", type=int, default=42)
    v.add_argument("--w", type=int); v.add_argument("--h", type=int); v.add_argument("--aspect", default="16:9"); v.add_argument("--name"); v.add_argument("--wait", action="store_true")
    sub.add_parser("status")
    l = sub.add_parser("list"); l.add_argument("-q"); l.add_argument("-s"); l.add_argument("-l", type=int, default=20)
    g = sub.add_parser("get"); g.add_argument("id")
    c = sub.add_parser("cancel"); c.add_argument("id")
    w = sub.add_parser("wait"); w.add_argument("id")
    args = ap.parse_args()

    if args.cmd == "image":
        prompts = None
        if args.file:
            prompts = [x.strip() for x in Path(os.path.expanduser(args.file)).read_text().splitlines() if x.strip() and not x.lstrip().startswith("#")]
        elif not args.prompt: sys.exit("需要 prompt 或 -f 文件")
        body = {k: v for k, v in dict(model=args.model, base_model=args.base, prompt=None if prompts else args.prompt, prompts=prompts, n=args.n, steps=args.steps,
                guidance=args.guidance, seed=args.seed, width=args.w, height=args.h, negative=args.neg, lora_path=args.lora, lora_weight=args.lora_weight, name=args.name).items() if v is not None}
        j = call("POST", "/jobs/image", body); show(j)
        if args.wait: wait(j["id"])
    elif args.cmd == "video":
        body = {k: v for k, v in dict(prompt=args.prompt, first=args.first, last=args.last, duration=args.duration, steps=args.steps, seed=args.seed,
                width=args.w, height=args.h, aspect=args.aspect, name=args.name).items() if v is not None}
        j = call("POST", "/jobs/video", body); show(j)
        if args.wait: wait(j["id"])
    elif args.cmd == "status":
        print(json.dumps(call("GET", "/queues"), ensure_ascii=False, indent=2))
    elif args.cmd == "list":
        q = "&".join(f"{k}={v}" for k, v in dict(queue=args.q, status=args.s, limit=args.l).items() if v is not None)
        for j in call("GET", "/jobs?" + q): show(j)
    elif args.cmd == "get":
        print(json.dumps(call("GET", f"/jobs/{args.id}"), ensure_ascii=False, indent=2))
    elif args.cmd == "cancel":
        print(call("DELETE", f"/jobs/{args.id}"))
    elif args.cmd == "wait":
        wait(args.id)


if __name__ == "__main__":
    main()
