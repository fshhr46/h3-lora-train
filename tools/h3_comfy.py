#!/usr/bin/env python3
"""
MiniMax H3 视频生成 —— 通过无界面 ComfyUI 后端（tools/h3_server.sh 管理，默认 http://127.0.0.1:8188）。

用法（与 tools/run_h3.sh 相同的参数）:
  h3_comfy.py --first a.png [--last b.png] --prompt "..."     # 首帧(+尾帧) → 视频
  h3_comfy.py --prompt "..." [--aspect 16:9]                 # 纯文生视频
可选: --duration 5  --steps 20  --seed 42  --name 批次名  --width W --height H  --prompt-file F  --dry-run
图片路径可给 smb://100.89.241.44/shared/... 或 /Volumes/shared/...，会换算成 ~/shared/...
输出: ~/shared/h3-videos/<时间>-<批次名>/video.mp4 + request.json + 首/尾帧副本

图结构 = 官方模板 video_minimax_h3_{t2v,i2v}.json 的子图手工展开:
  UNETLoader / CLIPLoader(type=minimax) / VAELoader×2 → MiniMaxH3ImageToVideo → BasicGuider
  → SamplerCustomAdvanced(res_multistep, simple, 20 steps) → VAEDecode + VAEDecodeAudio → CreateVideo(24fps) → SaveVideo
"""
import argparse, json, mimetypes, os, sys, time, uuid, shutil, urllib.request, urllib.parse, urllib.error
from pathlib import Path

COMFY = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")
HOME = Path.home()
UNET = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
CLIP = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VAE_V = "minimax_h3_video_vae_fp16.safetensors"
VAE_A = "minimax_h3_audio_vae_fp32.safetensors"


def to_local(p: str) -> str:
    for pre in ("smb://guest@", "smb://"):
        if p.startswith(pre): p = p[len(pre):]
    for pre in ("100.89.241.44/shared/", "furnance/shared/", "/Volumes/shared/"):
        if p.startswith(pre): return str(HOME / "shared" / p[len(pre):])
    return p


def frames_for(duration: float) -> int:
    n = max(5, round(duration * 24))
    return n + (5 - (n % 17)) % 17


def canvas(aspect: str = None, w: int = None, h: int = None, img_wh=None):
    """H3 原生画布：768 短边，上限 768x1344，32 的倍数。"""
    if w and h:
        return (w // 32) * 32, (h // 32) * 32
    if img_wh:
        iw, ih = img_wh; r = iw / ih
    else:
        a, b = (aspect or "16:9").split(":"); r = int(a) / int(b)
    if r >= 1:
        h_ = 768; w_ = min(1344, round(768 * r))
    else:
        w_ = 768; h_ = min(1344, round(768 / r))
    return (w_ // 32) * 32, (h_ // 32) * 32


def http(method, path, data=None, headers=None, timeout=60):
    req = urllib.request.Request(COMFY + path, data=data, method=method, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def upload_image(path: str) -> str:
    boundary = uuid.uuid4().hex
    name = f"h3_{int(time.time())}_{Path(path).name}"
    ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\n"
            f"Content-Type: {ctype}\r\n\r\n").encode() + Path(path).read_bytes() + \
           f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n--{boundary}--\r\n".encode()
    r = json.loads(http("POST", "/upload/image", body, {"Content-Type": f"multipart/form-data; boundary={boundary}"}))
    return r["name"]


def image_size(path):
    try:
        from PIL import Image
        with Image.open(path) as im: return im.size
    except Exception:
        return None


def build_graph(prompt, width, height, length, seed, steps, first=None, last=None, prefix="video/h3"):
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": CLIP, "type": "minimax", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_V}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_A}},
        "10": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {
            "clip": ["2", 0], "vae": ["3", 0], "prompt": prompt, "width": width, "height": height, "length": length}},
        "11": {"class_type": "BasicGuider", "inputs": {"model": ["1", 0], "conditioning": ["10", 0]}},
        "12": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "13": {"class_type": "BasicScheduler", "inputs": {"model": ["1", 0], "scheduler": "simple", "steps": steps, "denoise": 1.0}},
        "14": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "15": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["14", 0], "guider": ["11", 0], "sampler": ["12", 0], "sigmas": ["13", 0], "latent_image": ["10", 1]}},
        "16": {"class_type": "VAEDecode", "inputs": {"samples": ["15", 0], "vae": ["3", 0]}},
        "17": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["15", 0], "vae": ["4", 0]}},
        "18": {"class_type": "CreateVideo", "inputs": {"images": ["16", 0], "fps": 24, "audio": ["17", 0]}},
        "19": {"class_type": "SaveVideo", "inputs": {"video": ["18", 0], "filename_prefix": prefix, "format": "auto", "codec": "auto"}},
    }
    if first:
        g["20"] = {"class_type": "LoadImage", "inputs": {"image": first}}
        g["21"] = {"class_type": "ImageScale", "inputs": {"image": ["20", 0], "upscale_method": "lanczos", "width": width, "height": height, "crop": "center"}}
        g["10"]["inputs"]["first_frame"] = ["21", 0]
    if last:
        g["22"] = {"class_type": "LoadImage", "inputs": {"image": last}}
        g["23"] = {"class_type": "ImageScale", "inputs": {"image": ["22", 0], "upscale_method": "lanczos", "width": width, "height": height, "crop": "center"}}
        g["10"]["inputs"]["last_frame"] = ["23", 0]
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first"); ap.add_argument("--last")
    ap.add_argument("--prompt"); ap.add_argument("--prompt-file")
    ap.add_argument("--duration", type=float, default=5); ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--name")
    ap.add_argument("--width", type=int); ap.add_argument("--height", type=int); ap.add_argument("--aspect", default="16:9")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    prompt = a.prompt
    if a.prompt_file:
        prompt = " ".join(l.strip() for l in Path(a.prompt_file).read_text().splitlines() if l.strip() and not l.lstrip().startswith("#"))
    if not prompt: sys.exit("❌ 需要 --prompt 或 --prompt-file")
    first = to_local(a.first) if a.first else None
    last = to_local(a.last) if a.last else None
    if last and not first: sys.exit("❌ 只给尾帧不支持，请同时给 --first")
    for p in (first, last):
        if p and not Path(p).is_file(): sys.exit(f"❌ 图片不存在: {p}")

    task = "fl2va" if first else "t2va"
    width, height = canvas(a.aspect, a.width, a.height, image_size(first) if first else None)
    length = frames_for(a.duration)
    ts = time.strftime("%Y%m%d-%H%M%S"); name = a.name or task
    out = HOME / "shared" / "h3-videos" / f"{ts}-{name}"

    print(f"🎬 task={task} {width}x{height} length={length} frames (~{length/24:.1f}s) steps={a.steps} seed={a.seed}")
    print(f"   prompt: {prompt[:120]}")
    if first: print(f"   first:  {first}")
    if last:  print(f"   last:   {last}")
    print(f"   out:    {out}/video.mp4")

    if not a.dry_run:
        try: http("GET", "/system_stats", timeout=5)
        except Exception as e: sys.exit(f"❌ ComfyUI 后端不可达 {COMFY}（启动: tools/h3_server.sh start）: {e}")

    up_first = upload_image(first) if (first and not a.dry_run) else (Path(first).name if first else None)
    up_last = upload_image(last) if (last and not a.dry_run) else (Path(last).name if last else None)
    graph = build_graph(prompt, width, height, length, a.seed, a.steps, up_first, up_last, prefix=f"video/h3/{ts}-{name}")
    if a.dry_run:
        print(json.dumps(graph, ensure_ascii=False, indent=1)[:1500]); return

    out.mkdir(parents=True, exist_ok=True)
    (out / "request.json").write_text(json.dumps({"task": task, "prompt": prompt, "first": first, "last": last, "width": width, "height": height,
        "length": length, "duration": a.duration, "steps": a.steps, "seed": a.seed, "backend": "comfyui", "graph": graph}, ensure_ascii=False, indent=2))
    if first: shutil.copy(first, out / f"first{Path(first).suffix}")
    if last:  shutil.copy(last, out / f"last{Path(last).suffix}")

    t0 = time.time()
    r = json.loads(http("POST", "/prompt", json.dumps({"prompt": graph, "client_id": uuid.uuid4().hex}).encode(), {"Content-Type": "application/json"}))
    if "prompt_id" not in r: sys.exit(f"❌ 提交失败: {json.dumps(r, ensure_ascii=False)[:800]}")
    pid = r["prompt_id"]; print(f"   submitted prompt_id={pid}")
    last_msg = ""
    while True:
        time.sleep(10)
        h = json.loads(http("GET", f"/history/{pid}"))
        if pid in h:
            st = h[pid].get("status", {})
            if st.get("status_str") == "error" or st.get("completed") is False and st.get("status_str"):
                msgs = [m for m in st.get("messages", []) if m[0] == "execution_error"]
                sys.exit("❌ 执行出错: " + json.dumps(msgs, ensure_ascii=False)[:1200])
            outputs = h[pid].get("outputs", {})
            files = [f for o in outputs.values() for k in ("images", "gifs", "video", "videos") for f in o.get(k, []) if isinstance(f, dict) and "filename" in f]
            if files:
                f = files[-1]
                q = urllib.parse.urlencode({"filename": f["filename"], "subfolder": f.get("subfolder", ""), "type": f.get("type", "output")})
                data = http("GET", f"/view?{q}", timeout=600)
                (out / "video.mp4").write_bytes(data)
                print(f"✅ 完成 {time.time()-t0:.0f}s → {out}/video.mp4 ({len(data)/1e6:.1f} MB)")
                print(f"   SMB: smb://guest@100.89.241.44/shared/h3-videos/{ts}-{name}/video.mp4"); return
        try:
            q = json.loads(http("GET", "/queue"))
            running = len(q.get("queue_running", [])); pending = len(q.get("queue_pending", []))
            msg = f"   … {time.time()-t0:.0f}s (running={running} pending={pending})"
            if msg != last_msg and int(time.time()-t0) % 60 < 10: print(msg); last_msg = msg
        except Exception: pass


if __name__ == "__main__":
    main()
