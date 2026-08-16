#!/usr/bin/env python3
"""
Chroma1-HD 文生图（diffusers ChromaPipeline，FLUX 架构、Apache 2.0、未过滤的社区模型）。

与 generate.py（SDXL）同一套 CLI/输出约定：
    python generate_chroma.py --prompts prompts.txt --output ~/shared/chroma-images/xxx --num_steps 26 --guidance 3.5 --seed 42
权重：lodestones/Chroma1-HD（HF 缓存，HF_HOME=/mnt/elements/hf-cache，离线加载）
显存：transformer bf16 17.8 GB + T5-XXL 9.5 GB > 24 GB，因此默认 enable_model_cpu_offload（分阶段上卡，峰值 ~18 GB）。
GPU：由 CUDA_VISIBLE_DEVICES 决定（tools/run_chroma.sh 默认 gpu2 = 索引 1）。
"""
import argparse, json, os, sys, time
from pathlib import Path

try:
    import torch
    from diffusers import ChromaPipeline
except ImportError as e:
    print(f"❌ 缺少依赖: {e}  (需要 diffusers>=0.34, torch)"); sys.exit(1)

MODEL_ID = os.environ.get("CHROMA_MODEL", "lodestones/Chroma1-HD")


def resolve_model(model_id: str) -> str:
    """离线：优先用 HF 缓存里的 snapshot 目录（diffusers 离线按 repo id 加载会去联网查元数据）。"""
    if os.path.isdir(model_id):
        return model_id
    import glob
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    snaps = sorted(glob.glob(os.path.join(hf_home, "hub", "models--" + model_id.replace("/", "--"), "snapshots", "*")))
    return snaps[-1] if snaps else model_id


def load_prompt_file(path, joiner=None):
    p = Path(path)
    if not p.is_file():
        return None
    lines = [l.strip() for l in p.read_text().splitlines() if l.strip() and not l.lstrip().startswith("#")]
    return joiner.join(lines) if joiner is not None else lines


def main():
    ap = argparse.ArgumentParser(description="Chroma1-HD 文生图")
    ap.add_argument("--prompts", type=str, required=True, help="提示词文件（每行一条，# 忽略）或一段内联文字")
    ap.add_argument("--negative_prompts", type=str, default="", help="负面提示词文件（各行逗号拼接）或内联文字；默认不用")
    ap.add_argument("--output", type=str, default="output/chroma")
    ap.add_argument("--num_steps", type=int, default=26)
    ap.add_argument("--guidance", type=float, default=3.5, help="Chroma 是真 CFG（非蒸馏），推荐 3–4.5")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=None, help="基础种子；第 i 张用 seed+i")
    ap.add_argument("--no_offload", action="store_true", help="不做 CPU offload（需要 >30 GB 显存）")
    ap.add_argument("--fp8", action="store_true", help="transformer 用 fp8 存储（layerwise casting），更省显存")
    args = ap.parse_args()

    prompts = load_prompt_file(args.prompts)
    if prompts is None:
        prompts = [args.prompts]; print("📝 使用内联提示词")
    else:
        print(f"📝 从 {args.prompts} 加载了 {len(prompts)} 条提示词")
    negative = ""
    if args.negative_prompts:
        negative = load_prompt_file(args.negative_prompts, joiner=", ")
        if negative is None: negative = args.negative_prompts
        print(f"🚫 负面提示词: {negative[:100]}")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  设备: {dev} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','all')}) | 📦 {MODEL_ID}")
    t0 = time.time()
    MODEL_PATH = resolve_model(MODEL_ID); print(f"   路径: {MODEL_PATH}")
    pipe = ChromaPipeline.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16)
    if args.fp8:
        pipe.transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
    if args.no_offload:
        pipe.to(dev)
    else:
        pipe.enable_model_cpu_offload()
    print(f"✅ 加载完成 {time.time()-t0:.0f}s")

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    existing = len(list(out.glob("chroma_*.png")))
    manifest = []
    for i, prompt in enumerate(prompts):
        s = None if args.seed is None else args.seed + i
        g = torch.Generator("cuda").manual_seed(s) if s is not None else None
        print(f"\n[{i+1}/{len(prompts)}] {prompt[:100]}  (steps={args.num_steps} cfg={args.guidance} seed={s})")
        t1 = time.time()
        img = pipe(prompt=prompt, negative_prompt=negative or None, num_inference_steps=args.num_steps,
                   guidance_scale=args.guidance, width=args.width, height=args.height, generator=g).images[0]
        fn = out / f"chroma_{existing+i+1:04d}.png"; img.save(str(fn))
        manifest.append({"file": fn.name, "prompt": prompt, "negative_prompt": negative, "steps": args.num_steps,
                         "guidance": args.guidance, "seed": s, "width": args.width, "height": args.height, "model": MODEL_ID})
        print(f"   💾 {fn}  ({time.time()-t1:.0f}s)")
    (out / "manifest.jsonl").open("a").write("".join(json.dumps(m, ensure_ascii=False) + "\n" for m in manifest))
    print(f"\n🎉 生成完成: {len(prompts)} 张 → {out}")


if __name__ == "__main__":
    main()
