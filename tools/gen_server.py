#!/usr/bin/env python3
"""
gen_server.py — 图片 / 视频生成的 FIFO 队列服务（FastAPI）。

两个独立队列、各一个 worker 线程、严格先进先出、一次一个任务：
  image  → gpu2（CUDA 索引 1）：SDXL (generate.py) 或 Chroma (generate_chroma.py)
  video  → gpu1（CUDA 索引 0）：MiniMax H3（h3_comfy.py → 无界面 ComfyUI :8188）
任务持久化在 SQLite（/mnt/elements/gen-queue/jobs.db），服务重启后 pending 任务继续排队、running 任务标为 interrupted。
输出目录沿用现有约定：~/shared/{chroma-images,lora-images,h3-videos}/<时间>-<name>/，SMB 可见。

启动：tools/gen_server.sh start   （systemd 用户单元 gen-api，监听 0.0.0.0:8090，tailnet 内可访问）
API：
  POST /jobs/image   {"model":"chroma"|"sdxl", "prompts":[...] 或 "prompt":"...", "n":1, "steps":..,
                      "guidance":.., "seed":42, "width":1024, "height":1024, "negative":"...", "lora_weight":0.0, "name":"batch"}
  POST /jobs/video   {"prompt":"...", "first":"<路径或smb地址>", "last":"...", "duration":5, "steps":20,
                      "seed":42, "width":null, "height":null, "aspect":"16:9", "name":"batch"}
  GET  /jobs/{id}    GET /jobs?queue=image|video&status=pending|running|done|failed&limit=50
  DELETE /jobs/{id}  取消 pending 任务（running 的会被 kill）
  GET  /queues       两个队列的长度、当前任务、GPU/模式状态
  GET  /health
客户端：tools/genq.py（见 --help）。
"""
import json, os, signal, sqlite3, subprocess, sys, threading, time, uuid
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
import uvicorn

ROOT = Path(__file__).resolve().parent.parent          # 项目根 (test-deepseek-harness)
TOOLS = ROOT / "tools"
HOME = Path.home()
DB_DIR = Path(os.environ.get("GENQ_DIR", "/mnt/elements/gen-queue")); DB_DIR.mkdir(parents=True, exist_ok=True)
DB = DB_DIR / "jobs.db"
LOGDIR = DB_DIR / "logs"; LOGDIR.mkdir(exist_ok=True)
VENV_PY = ROOT / ".venv" / "bin" / "python"
COMFY_PY = HOME / "miniforge3" / "envs" / "comfy" / "bin" / "python"
COMFY_URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")
GPU_IMAGE = os.environ.get("GPU_IMAGE", "1")   # gpu2
GPU_VIDEO = os.environ.get("GPU_VIDEO", "0")   # gpu1
SMB = "smb://guest@100.89.241.44/shared"

app = FastAPI(title="gen-queue", version="1.0")
_lock = threading.Lock()
_running_proc = {"image": None, "video": None}
_current = {"image": None, "video": None}
_wake = {"image": threading.Event(), "video": threading.Event()}


# ---------------- storage ----------------
def db():
    c = sqlite3.connect(DB, timeout=30, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c

def init_db():
    with db() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS jobs(
            id TEXT PRIMARY KEY, queue TEXT, status TEXT, created REAL, started REAL, finished REAL,
            request TEXT, output_dir TEXT, log TEXT, error TEXT, seq INTEGER)""")
        c.execute("CREATE INDEX IF NOT EXISTS ix_q ON jobs(queue,status,seq)")
        # 服务重启：正在跑的标 interrupted
        c.execute("UPDATE jobs SET status='interrupted', finished=? WHERE status='running'", (time.time(),))

def insert(queue, req: dict) -> dict:
    jid = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    with _lock, db() as c:
        seq = (c.execute("SELECT COALESCE(MAX(seq),0)+1 FROM jobs").fetchone()[0])
        c.execute("INSERT INTO jobs(id,queue,status,created,request,seq) VALUES(?,?,?,?,?,?)",
                  (jid, queue, "pending", time.time(), json.dumps(req, ensure_ascii=False), seq))
    _wake[queue].set()
    return get(jid)

def get(jid) -> Optional[dict]:
    with db() as c:
        r = c.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone()
    return row2dict(r) if r else None

def row2dict(r):
    d = dict(r); d["request"] = json.loads(d["request"]) if d.get("request") else {}
    if d.get("output_dir"):
        od = d["output_dir"]; d["smb"] = od.replace(str(HOME / "shared"), SMB) if od.startswith(str(HOME / "shared")) else None
    pos = None
    if d["status"] == "pending":
        with db() as c:
            pos = c.execute("SELECT COUNT(*) FROM jobs WHERE queue=? AND status='pending' AND seq<?", (d["queue"], d["seq"])).fetchone()[0]
    d["position"] = pos
    return d

def update(jid, **kw):
    with _lock, db() as c:
        c.execute("UPDATE jobs SET " + ",".join(f"{k}=?" for k in kw) + " WHERE id=?", (*kw.values(), jid))

def next_pending(queue):
    with db() as c:
        r = c.execute("SELECT * FROM jobs WHERE queue=? AND status='pending' ORDER BY seq LIMIT 1", (queue,)).fetchone()
    return row2dict(r) if r else None


# ---------------- helpers ----------------
def brain_active() -> bool:
    return subprocess.run(["systemctl", "--user", "is-active", "llama-brain.service"], capture_output=True, text=True).stdout.strip() == "active"

def to_local(p: str) -> str:
    for pre in ("smb://guest@", "smb://"):
        if p.startswith(pre): p = p[len(pre):]
    for pre in ("100.89.241.44/shared/", "furnance/shared/", "/Volumes/shared/"):
        if p.startswith(pre): return str(HOME / "shared" / p[len(pre):])
    return os.path.expanduser(p)

def ensure_comfy():
    import urllib.request
    try:
        urllib.request.urlopen(COMFY_URL + "/system_stats", timeout=3); return
    except Exception:
        pass
    subprocess.run([str(TOOLS / "h3_server.sh"), "start"], capture_output=True, text=True)
    for _ in range(60):
        time.sleep(2)
        try:
            urllib.request.urlopen(COMFY_URL + "/system_stats", timeout=3); return
        except Exception:
            continue
    raise RuntimeError("ComfyUI 后端 (H3) 起不来，见 tools/h3_server.sh logs")

def run_cmd(queue, jid, cmd, env, logfile):
    with open(logfile, "ab") as lf:
        lf.write((f"$ {' '.join(map(str, cmd))}\n").encode())
        p = subprocess.Popen([str(c) for c in cmd], stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=str(ROOT))
        _running_proc[queue] = p
        rc = p.wait()
        _running_proc[queue] = None
    return rc


# ---------------- executors ----------------
def exec_image(job):
    r = job["request"]; model = r.get("model", "chroma")
    ts = time.strftime("%Y%m%d-%H%M"); name = r.get("name") or job["id"]
    sub = "chroma-images" if model == "chroma" else "lora-images"
    out = HOME / "shared" / sub / f"{ts}-{name}"; out.mkdir(parents=True, exist_ok=True)
    prompts = r.get("prompts") or [r.get("prompt")]
    n = int(r.get("n") or 1)
    pf = out / "prompts.txt"; pf.write_text("\n".join(p for p in prompts for _ in range(n)) + "\n")
    update(job["id"], output_dir=str(out))
    env = dict(os.environ, HF_HOME="/mnt/elements/hf-cache", HF_HUB_OFFLINE="1", CUDA_VISIBLE_DEVICES=GPU_IMAGE, PYTHONUNBUFFERED="1")
    log = LOGDIR / f"{job['id']}.log"
    if model == "chroma":
        cmd = [VENV_PY, TOOLS / "generate_chroma.py", "--prompts", pf, "--output", out, "--num_steps", r.get("steps") or 26,
               "--guidance", r.get("guidance") or 3.5, "--width", r.get("width") or 1024, "--height", r.get("height") or 1024]
        if r.get("seed") is not None: cmd += ["--seed", r["seed"]]
        if r.get("negative"): cmd += ["--negative_prompts", r["negative"]]
        if r.get("fp8"): cmd += ["--fp8"]
    else:  # sdxl
        cmd = [VENV_PY, ROOT / "generate.py", "--lora_path", r.get("lora_path") or "output/models/lora/best_lora",
               "--prompts", pf, "--output", out, "--lora_weight", r.get("lora_weight") if r.get("lora_weight") is not None else 0.0,
               "--num_steps", r.get("steps") or 30, "--guidance", r.get("guidance") or 7.0,
               "--width", r.get("width") or 1024, "--height", r.get("height") or 1024, "--seed", r.get("seed") if r.get("seed") is not None else 42]
        cmd += ["--negative_prompts", r["negative"] if r.get("negative") is not None else "prompts/negative_prompts.txt"]
    rc = run_cmd("image", job["id"], cmd, env, log)
    if rc != 0: raise RuntimeError(f"exit code {rc}，见 {log}")
    return str(out)

def exec_video(job):
    r = job["request"]
    ensure_comfy()
    cmd = [COMFY_PY, TOOLS / "h3_comfy.py", "--prompt", r["prompt"], "--duration", r.get("duration") or 5,
           "--steps", r.get("steps") or 20, "--seed", r.get("seed") if r.get("seed") is not None else 42, "--name", r.get("name") or job["id"]]
    if r.get("first"): cmd += ["--first", to_local(r["first"])]
    if r.get("last"): cmd += ["--last", to_local(r["last"])]
    if r.get("width") and r.get("height"): cmd += ["--width", r["width"], "--height", r["height"]]
    if r.get("aspect"): cmd += ["--aspect", r["aspect"]]
    env = dict(os.environ, COMFY_URL=COMFY_URL, PYTHONUNBUFFERED="1")
    log = LOGDIR / f"{job['id']}.log"
    rc = run_cmd("video", job["id"], cmd, env, log)
    if rc != 0: raise RuntimeError(f"exit code {rc}，见 {log}")
    # h3_comfy.py 把输出目录写在日志 "out:    <dir>/video.mp4"
    out = None
    for line in log.read_text(errors="ignore").splitlines():
        if line.strip().startswith("out:"): out = line.split("out:", 1)[1].strip().rsplit("/", 1)[0]
    return out


def worker(queue, fn):
    while True:
        job = next_pending(queue)
        if not job:
            _wake[queue].wait(timeout=5); _wake[queue].clear(); continue
        if brain_active():
            update(job["id"], status="failed", finished=time.time(), error="brain mode 中（llama-brain 在跑）。先 systemctl --user stop llama-brain.service 切到 gen-mode")
            continue
        _current[queue] = job["id"]
        update(job["id"], status="running", started=time.time(), log=str(LOGDIR / f"{job['id']}.log"))
        try:
            out = fn(job)
            update(job["id"], status="done", finished=time.time(), output_dir=out)
        except Exception as e:
            st = "cancelled" if (get(job["id"]) or {}).get("status") == "cancelling" else "failed"
            update(job["id"], status=st, finished=time.time(), error=str(e)[:2000])
        finally:
            _current[queue] = None


# ---------------- API ----------------
class ImageJob(BaseModel):
    model: str = Field("chroma", pattern="^(chroma|sdxl)$")
    prompt: Optional[str] = None
    prompts: Optional[List[str]] = None
    n: int = 1
    steps: Optional[int] = None
    guidance: Optional[float] = None
    seed: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    negative: Optional[str] = None
    lora_weight: Optional[float] = None
    lora_path: Optional[str] = None
    fp8: bool = False
    name: Optional[str] = None

class VideoJob(BaseModel):
    prompt: str
    first: Optional[str] = None
    last: Optional[str] = None
    duration: float = 5
    steps: int = 20
    seed: Optional[int] = 42
    width: Optional[int] = None
    height: Optional[int] = None
    aspect: Optional[str] = "16:9"
    name: Optional[str] = None

@app.get("/health")
def health(): return {"ok": True, "brain_mode": brain_active(), "gen_mode": not brain_active()}

@app.get("/queues")
def queues():
    with db() as c:
        cnt = {q: c.execute("SELECT COUNT(*) FROM jobs WHERE queue=? AND status='pending'", (q,)).fetchone()[0] for q in ("image", "video")}
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip().splitlines()
    return {"mode": "brain" if brain_active() else "gen", "gpu": gpu,
            "image": {"gpu": f"cuda:{GPU_IMAGE} (gpu2)", "pending": cnt["image"], "running": _current["image"]},
            "video": {"gpu": f"cuda:{GPU_VIDEO} (gpu1)", "pending": cnt["video"], "running": _current["video"]}}

@app.post("/jobs/image")
def submit_image(j: ImageJob):
    if not (j.prompt or j.prompts): raise HTTPException(400, "需要 prompt 或 prompts")
    return insert("image", j.model_dump(exclude_none=True))

@app.post("/jobs/video")
def submit_video(j: VideoJob):
    if j.last and not j.first: raise HTTPException(400, "只给 last 不支持，请同时给 first")
    for p in (j.first, j.last):
        if p and not Path(to_local(p)).is_file(): raise HTTPException(400, f"图片不存在: {p}")
    return insert("video", j.model_dump(exclude_none=True))

@app.get("/jobs/{jid}")
def job(jid: str):
    d = get(jid)
    if not d: raise HTTPException(404, "no such job")
    return d

@app.get("/jobs")
def jobs(queue: Optional[str] = None, status: Optional[str] = None, limit: int = Query(50, le=500)):
    q = "SELECT * FROM jobs"; cond = []; args = []
    if queue: cond.append("queue=?"); args.append(queue)
    if status: cond.append("status=?"); args.append(status)
    if cond: q += " WHERE " + " AND ".join(cond)
    q += " ORDER BY seq DESC LIMIT ?"; args.append(limit)
    with db() as c:
        return [row2dict(r) for r in c.execute(q, args).fetchall()]

@app.delete("/jobs/{jid}")
def cancel(jid: str):
    d = get(jid)
    if not d: raise HTTPException(404, "no such job")
    if d["status"] == "pending":
        update(jid, status="cancelled", finished=time.time()); return {"id": jid, "status": "cancelled"}
    if d["status"] == "running":
        update(jid, status="cancelling")
        p = _running_proc.get(d["queue"])
        if p: p.send_signal(signal.SIGTERM)
        return {"id": jid, "status": "cancelling"}
    return {"id": jid, "status": d["status"], "note": "不是 pending/running，无需取消"}


if __name__ == "__main__":
    init_db()
    threading.Thread(target=worker, args=("image", exec_image), daemon=True, name="image-worker").start()
    threading.Thread(target=worker, args=("video", exec_video), daemon=True, name="video-worker").start()
    uvicorn.run(app, host=os.environ.get("GENQ_HOST", "0.0.0.0"), port=int(os.environ.get("GENQ_PORT", "8090")), log_level="info")
