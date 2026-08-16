#!/usr/bin/env python3
"""
使用训练好的 SDXL LoRA（PEFT 适配器）生成测试图片。

与 train_lora.py 配套：
  - 基础模型从项目 .models/ 里的 HF 缓存离线加载（StableDiffusionXLPipeline）
  - LoRA 是 train_lora.py 用 unet.save_pretrained() 存的 PEFT 目录
    （adapter_config.json + adapter_model.safetensors），用 PeftModel 注入回 UNet
  - --lora_weight 通过 cross_attention_kwargs={"scale": w} 控制强度

使用方法（在 furnance 上，需先停 llama-brain 腾出显卡，见 operation.md §3）:
    python generate.py --lora_path output/models/lora/best_lora \
        --prompts prompts/positive_prompts.txt --output output/videos \
        --lora_weight 0.7 --num_steps 30 --seed 42
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

try:
    import torch
    from diffusers import StableDiffusionXLPipeline
    from peft import PeftModel
except ImportError as e:
    print(f"❌ 缺少依赖: {e}")
    print("   pip install torch diffusers peft transformers")
    sys.exit(1)


def resolve_base_model(base_model: str) -> str:
    """把 HF repo id 解析成项目 .models/ 缓存里的 snapshot 目录（离线）。已是本地目录则原样返回。"""
    if os.path.isdir(base_model) and os.path.exists(os.path.join(base_model, "model_index.json")):
        return base_model
    here = Path(__file__).resolve().parent
    repo_dir = "models--" + base_model.replace("/", "--")
    for cache in (here / ".models", Path.home() / ".cache" / "huggingface" / "hub"):
        snaps = sorted(glob.glob(str(cache / repo_dir / "snapshots" / "*")))
        if snaps:
            return snaps[-1]
    print(f"❌ 在 .models/ 或 ~/.cache/huggingface/hub 里找不到 {base_model} 的缓存")
    sys.exit(1)


def load_prompt_file(path, joiner=None):
    p = Path(path)
    if not p.is_file():
        return None
    lines = [l.strip() for l in p.read_text().splitlines() if l.strip() and not l.lstrip().startswith("#")]
    if joiner is not None:
        return joiner.join(lines)
    return lines


class LoRAImageGenerator:
    def __init__(self, base_model, lora_path, device="cuda"):
        self.device = torch.device(device)
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        snapshot = resolve_base_model(base_model)
        print(f"🖥️  设备: {self.device}")
        print(f"📦 基础模型: {snapshot}")

        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            snapshot, torch_dtype=dtype, use_safetensors=True, add_watermarker=False,
        )

        lora_path = Path(lora_path)
        print(f"🎯 加载 LoRA: {lora_path}")
        if not (lora_path / "adapter_config.json").exists():
            print("❌ 不是 PEFT 适配器目录（缺 adapter_config.json）")
            sys.exit(1)
        cfg = json.load(open(lora_path / "adapter_config.json"))
        print(f"   r={cfg.get('r')} alpha={cfg.get('lora_alpha')} targets={cfg.get('target_modules')}")
        side_cfg = lora_path.parent / f"{lora_path.name}_config.json"
        if side_cfg.exists():
            print(f"   训练配置: {side_cfg.read_text().strip()}")

        # 把 LoRA 层注入 UNet，然后取回带 LoRA 层的原始 UNet 对象交给管线
        peft_unet = PeftModel.from_pretrained(self.pipe.unet, str(lora_path))
        self.pipe.unet = peft_unet.base_model.model
        self.pipe.unet.to(dtype)
        self.pipe.to(self.device)
        print("✅ 加载完成")

    @torch.inference_mode()
    def generate(self, prompt, negative_prompt="", lora_weight=0.7, width=1024, height=1024,
                 num_steps=30, guidance=7.0, seed=None):
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            num_inference_steps=num_steps,
            guidance_scale=guidance,
            width=width, height=height,
            generator=generator,
            cross_attention_kwargs={"scale": lora_weight},
        ).images[0]
        return image

    def generate_batch(self, prompts, output_dir, negative_prompt="", lora_weight=0.7, seed=None, **kw):
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        existing = len(list(out.glob("lora_test_*.png")))
        manifest = []
        for i, prompt in enumerate(prompts):
            s = None if seed is None else seed + i
            print(f"\n[{i+1}/{len(prompts)}] {prompt[:100]}  (lora_weight={lora_weight}, seed={s})")
            img = self.generate(prompt, negative_prompt=negative_prompt, lora_weight=lora_weight, seed=s, **kw)
            fn = out / f"lora_test_{existing + i + 1:04d}.png"
            img.save(str(fn))
            manifest.append({"file": fn.name, "prompt": prompt, "negative_prompt": negative_prompt,
                             "lora_weight": lora_weight, "seed": s, **kw})
            print(f"   💾 {fn}")
        (out / "manifest.jsonl").open("a").write("".join(json.dumps(m, ensure_ascii=False) + "\n" for m in manifest))
        print(f"\n🎉 生成完成: {len(prompts)} 张 → {out}")


def main():
    ap = argparse.ArgumentParser(description="SDXL LoRA 生图")
    ap.add_argument("--lora_path", type=str, required=True, help="PEFT 适配器目录，如 output/models/lora/best_lora")
    ap.add_argument("--base_model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    ap.add_argument("--prompts", type=str, default="prompts/positive_prompts.txt",
                    help="提示词文件（每行一条，# 开头忽略）；也可以直接给一段文字")
    ap.add_argument("--negative_prompts", type=str, default="prompts/negative_prompts.txt",
                    help="负面提示词文件（所有行用逗号拼成一条）；传 '' 表示不用")
    ap.add_argument("--output", type=str, default="output/videos")
    ap.add_argument("--lora_weight", type=float, default=0.7)
    ap.add_argument("--num_steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.0)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=None, help="基础种子；第 i 张用 seed+i")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    prompts = load_prompt_file(args.prompts)
    if prompts is None:
        prompts = [args.prompts]  # 当作一条内联提示词
        print("📝 使用内联提示词")
    else:
        print(f"📝 从 {args.prompts} 加载了 {len(prompts)} 条提示词")
    negative = load_prompt_file(args.negative_prompts, joiner=", ") if args.negative_prompts else ""
    if negative:
        print(f"🚫 负面提示词: {negative[:120]}...")

    gen = LoRAImageGenerator(args.base_model, args.lora_path, args.device)
    gen.generate_batch(prompts, args.output, negative_prompt=negative or "", lora_weight=args.lora_weight,
                       seed=args.seed, width=args.width, height=args.height,
                       num_steps=args.num_steps, guidance=args.guidance)


if __name__ == "__main__":
    main()
