#!/usr/bin/env python3
"""
SDXL LoRA 训练脚本（v2，2026-08-16）

与 diffusers 官方 SDXL LoRA 配方对齐，修掉 v1 里导致“出图全是噪点”的几处训练/推理不一致：
  1. 文本条件：两个 text encoder 都取 hidden_states[-2]（v1 用的是 last_hidden_state），pooled 用 TE2 的 text_embeds
  2. SDXL 微条件 time_ids = [orig_h, orig_w, crop_top, crop_left, target_h, target_w]（v1 是 0/0/h/2/w/2/h/w）
  3. 精度：UNet 用 bf16 权重 + autocast，LoRA 参数 fp32，loss 用 fp32（v1 是 fp16 权重且无 loss scaling）
  4. LoRA 只挂在注意力投影 to_q/to_k/to_v/to_out.0（v1 还挂了全部 conv 层，扰动过大）
  5. 每帧可用 captions/<stem>.txt；没有则统一用 --caption（默认带触发词 "hhcs style"），推理时把触发词写进 prompt
  6. VAE fp32 编码、latents 预计算（含水平翻转版本做增强）；梯度裁剪 1.0；cosine 学习率 + warmup

CLI 与 v1 兼容（train.sh / tools/run_lora_train.sh 不用改），新增 --caption --max_steps --lora_dropout --target_modules。
产出仍是 PEFT 适配器目录（adapter_config.json + adapter_model.safetensors），generate.py 直接可用。
"""

import argparse
import glob
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from peft import LoraConfig, get_peft_model

IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def resolve_snapshot(base_model: str) -> str:
    """HF repo id → 项目 .models/ 里的 snapshot 目录（离线）。本地目录则原样返回。"""
    if os.path.isdir(base_model) and os.path.exists(os.path.join(base_model, "model_index.json")):
        return base_model
    here = Path(__file__).resolve().parent
    repo_dir = "models--" + base_model.replace("/", "--")
    for cache in (here / ".models", Path.home() / ".cache" / "huggingface" / "hub"):
        snaps = sorted(glob.glob(str(cache / repo_dir / "snapshots" / "*")))
        if snaps:
            return snaps[-1]
    raise FileNotFoundError(f"找不到 {base_model} 的本地缓存（.models/ 或 ~/.cache/huggingface/hub）")


class FrameDataset(Dataset):
    """图片 + caption。resize 短边到 res 后中心裁剪，同时记录 SDXL 需要的 original_size / crop 坐标。"""

    def __init__(self, data_dir, resolution, default_caption):
        self.res_w, self.res_h = resolution
        self.files = sorted(p for p in Path(data_dir).rglob("*") if p.suffix.lower() in IMG_EXTS)
        if not self.files:
            raise RuntimeError(f"{data_dir} 里没有图片")
        cap_dir = Path(data_dir).parent / "captions"
        self.captions = []
        n_cap = 0
        for f in self.files:
            c = cap_dir / f"{f.stem}.txt"
            if c.exists():
                self.captions.append(c.read_text().strip()); n_cap += 1
            else:
                self.captions.append(default_caption)
        print(f"✅ 加载了 {len(self.files)} 张图；带专属 caption 的 {n_cap} 张，其余用默认 caption: “{default_caption}”")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(self.files[i]).convert("RGB")
        ow, oh = img.size
        scale = max(self.res_w / ow, self.res_h / oh)
        nw, nh = max(self.res_w, round(ow * scale)), max(self.res_h, round(oh * scale))
        img = img.resize((nw, nh), Image.BICUBIC)
        left, top = (nw - self.res_w) // 2, (nh - self.res_h) // 2
        img = img.crop((left, top, left + self.res_w, top + self.res_h))
        x = torch.from_numpy(__import__("numpy").asarray(img)).permute(2, 0, 1).float() / 127.5 - 1.0
        return {"pixel_values": x, "orig": (oh, ow), "crop": (top, left), "caption": self.captions[i], "idx": i}


def encode_prompt(tokenizers, text_encoders, caption, device):
    """SDXL 文本条件：concat(TE1.hidden[-2], TE2.hidden[-2]) 和 TE2 的 pooled text_embeds。"""
    hs = []
    pooled = None
    for tok, te in zip(tokenizers, text_encoders):
        ids = tok(caption, padding="max_length", max_length=tok.model_max_length, truncation=True,
                  return_tensors="pt").input_ids.to(device)
        out = te(ids, output_hidden_states=True)
        pooled = out[0]  # 对 TE2 是 text_embeds（投影后 pooled）；最终取 TE2 的
        hs.append(out.hidden_states[-2])
    return torch.cat(hs, dim=-1), pooled


def parse_args():
    p = argparse.ArgumentParser(description="SDXL LoRA 训练 (v2)")
    p.add_argument("--base_model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument("--data_dir", type=str, default="data/filtered")
    p.add_argument("--output_dir", type=str, default="output/models")
    p.add_argument("--caption", type=str, default="hhcs style, photo of a woman",
                   help="没有 captions/<stem>.txt 时使用的统一 caption（推理时把触发词写进 prompt）")
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--target_modules", type=str, default="to_q,to_k,to_v,to_out.0")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--max_steps", type=int, default=0, help=">0 时提前结束（冒烟测试用）")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--resolution", type=str, default="512,512", help="W,H")
    p.add_argument("--gradient_accumulation", type=int, default=1)
    p.add_argument("--save_every", type=int, default=2, help="每 N 个 epoch 保存一次检查点")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_flip", action="store_true", help="关闭水平翻转增强")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed); random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("❌ 需要 CUDA GPU（bf16 训练）")
    res_w, res_h = (int(v) for v in args.resolution.split(","))
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    snapshot = resolve_snapshot(args.base_model)
    print(f"🖥️  设备: {device} | 📦 基础模型: {snapshot}")

    # ---------- 冻结组件 ----------
    tok1 = CLIPTokenizer.from_pretrained(snapshot, subfolder="tokenizer")
    tok2 = CLIPTokenizer.from_pretrained(snapshot, subfolder="tokenizer_2")
    te1 = CLIPTextModel.from_pretrained(snapshot, subfolder="text_encoder", torch_dtype=torch.float32).to(device).eval()
    te2 = CLIPTextModelWithProjection.from_pretrained(snapshot, subfolder="text_encoder_2", torch_dtype=torch.float32).to(device).eval()
    vae = AutoencoderKL.from_pretrained(snapshot, subfolder="vae", torch_dtype=torch.float32).to(device).eval()
    noise_scheduler = DDPMScheduler.from_pretrained(snapshot, subfolder="scheduler")
    for m in (te1, te2, vae):
        m.requires_grad_(False)

    # ---------- 数据 → 预计算 latents / 文本嵌入 ----------
    ds = FrameDataset(args.data_dir, (res_w, res_h), args.caption)
    print("📦 预计算 VAE latents（含水平翻转）...")
    latents, latents_flip, time_ids = [], [], []
    with torch.no_grad():
        for i in range(len(ds)):
            item = ds[i]
            x = item["pixel_values"].unsqueeze(0).to(device)
            latents.append(vae.encode(x).latent_dist.sample().squeeze(0) * vae.config.scaling_factor)
            if not args.no_flip:
                latents_flip.append(vae.encode(torch.flip(x, dims=[3])).latent_dist.sample().squeeze(0) * vae.config.scaling_factor)
            oh, ow = item["orig"]; top, left = item["crop"]
            time_ids.append(torch.tensor([oh, ow, top, left, res_h, res_w], dtype=torch.float32))
    latents = torch.stack(latents)                       # [N,4,h/8,w/8] fp32 on GPU
    latents_flip = torch.stack(latents_flip) if latents_flip else None
    time_ids = torch.stack(time_ids).to(device)
    print(f"   ✅ latents: {tuple(latents.shape)}")

    print("📦 预计算文本嵌入...")
    uniq = sorted(set(ds.captions))
    emb_cache = {}
    with torch.no_grad():
        for c in uniq:
            emb_cache[c] = encode_prompt((tok1, tok2), (te1, te2), c, device)
    cap_idx = torch.tensor([uniq.index(c) for c in ds.captions], device=device)
    hidden_all = torch.stack([emb_cache[c][0].squeeze(0) for c in uniq])   # [U,77,2048]
    pooled_all = torch.stack([emb_cache[c][1].squeeze(0) for c in uniq])   # [U,1280]
    print(f"   ✅ {len(uniq)} 条不同 caption")
    del te1, te2, vae, emb_cache
    torch.cuda.empty_cache()

    # ---------- UNet + LoRA ----------
    unet = UNet2DConditionModel.from_pretrained(snapshot, subfolder="unet", torch_dtype=torch.bfloat16).to(device)
    unet.requires_grad_(False)
    unet.enable_gradient_checkpointing()
    lora_cfg = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                          init_lora_weights="gaussian", target_modules=args.target_modules.split(","))
    unet = get_peft_model(unet, lora_cfg)   # 适配器权重会被 PEFT 自动转成 fp32
    unet.print_trainable_parameters()
    params = [p for p in unet.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()

    # ---------- 优化器 / 调度 ----------
    N = len(ds)
    steps_per_epoch = math.ceil(N / args.batch_size)
    total_updates = math.ceil(steps_per_epoch * args.epochs / args.gradient_accumulation)
    if args.max_steps > 0:
        total_updates = min(total_updates, args.max_steps)
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-8)
    lr_sched = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=max(1, int(0.05 * total_updates)),
                             num_training_steps=total_updates)

    def save(name, extra=None):
        path = out_dir / name
        unet.save_pretrained(str(path))
        meta = {"base_model": args.base_model, "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha,
                "target_modules": args.target_modules, "resolution": args.resolution, "epochs": args.epochs,
                "batch_size": args.batch_size, "learning_rate": args.learning_rate, "seed": args.seed,
                "caption": args.caption, "trainer": "train_lora.py v2 (bf16, hidden[-2], time_ids fixed)"}
        if extra: meta.update(extra)
        (out_dir / f"{name}_config.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        print(f"💾 已保存: {path}")

    print("\n🚀 开始训练...")
    print(f"   样本 {N} | 每 epoch {steps_per_epoch} 步 | 累积 {args.gradient_accumulation} | 优化器更新总数 {total_updates}")
    print("=" * 60)
    unet.train()
    global_update, best_loss, t0 = 0, float("inf"), time.time()
    done = False
    for epoch in range(args.epochs):
        perm = torch.randperm(N, device=device)
        epoch_loss, epoch_n = 0.0, 0
        for step in range(steps_per_epoch):
            idx = perm[step * args.batch_size:(step + 1) * args.batch_size]
            bs = idx.numel()
            if latents_flip is not None:
                flip = torch.rand(bs, device=device) < 0.5
                lat = torch.where(flip[:, None, None, None], latents_flip[idx], latents[idx])
            else:
                lat = latents[idx]
            noise = torch.randn_like(lat)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=device).long()
            noisy = noise_scheduler.add_noise(lat, noise, timesteps)
            ehs = hidden_all[cap_idx[idx]]
            pooled = pooled_all[cap_idx[idx]]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = unet(noisy.to(torch.bfloat16), timesteps, encoder_hidden_states=ehs.to(torch.bfloat16),
                            added_cond_kwargs={"text_embeds": pooled.to(torch.bfloat16),
                                               "time_ids": time_ids[idx].to(torch.bfloat16)}).sample
            if noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(lat, noise, timesteps)
            else:
                target = noise
            loss = F.mse_loss(pred.float(), target.float()) / args.gradient_accumulation
            loss.backward()
            epoch_loss += loss.item() * args.gradient_accumulation; epoch_n += 1
            if (step + 1) % args.gradient_accumulation == 0 or step + 1 == steps_per_epoch:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step(); lr_sched.step(); optimizer.zero_grad(set_to_none=True)
                global_update += 1
                if global_update % 25 == 0:
                    print(f"  Epoch {epoch+1}/{args.epochs} | update {global_update}/{total_updates} | "
                          f"loss {epoch_loss/epoch_n:.4f} | lr {lr_sched.get_last_lr()[0]:.2e} | {time.time()-t0:.0f}s")
                if args.max_steps > 0 and global_update >= args.max_steps:
                    done = True; break
        avg = epoch_loss / max(1, epoch_n)
        print(f"✅ Epoch {epoch+1}/{args.epochs} | Average Loss: {avg:.4f}")
        if avg < best_loss:
            best_loss = avg; save("best_lora", {"epoch": epoch + 1, "avg_loss": avg})
        if (epoch + 1) % args.save_every == 0:
            save(f"lora_epoch_{epoch+1}", {"epoch": epoch + 1, "avg_loss": avg})
        print("-" * 60)
        if done:
            break
    save("final_lora", {"epoch": epoch + 1, "avg_loss": avg})
    print(f"\n🎉 训练完成！模型已保存到: {out_dir}  (用时 {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
