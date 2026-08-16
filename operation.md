# operation.md — 给 Claude 的操作手册（test-deepseek-harness：LoRA 训练 / SDXL 生图 / MiniMax H3 生视频）

> 读者是 Claude（或其它 agent）。目标：不需要人解释背景，就能安全地在 furnance 上跑一轮
> SDXL LoRA 训练并把环境恢复原状。所有命令默认从 **bastion**（Mac mini）或 **runtime** 上通过
> `ssh furnance` 执行；runtime 已配置免密直连 furnance。
> 最后更新：2026-08-16（训练 §1 与生图 §3 均已实跑验证）。
> 用户说“训练”“跑 LoRA”→ §1；用户发 `[gen-img sdxl|chroma] …` → §3（图片，默认 chroma）；用户发 `[gen-v] …` → §5（视频）；“gen-mode / brain mode”→ §0.5；提交任务优先用 §6 的队列；都不需要再向用户要说明。

---

## 0. 机器拓扑（先看这个）

| 主机 | 角色 | 关键信息 |
|---|---|---|
| **furnance** (`100.89.241.44`, Ubuntu, 2×RTX 4090 24 GB) | 训练机 + 主脑 | `llama-brain.service`（用户级 systemd）跑 Qwen3.6-35B-A3B，**默认占满两张卡**（各 ~22.5 GB）；13 TB 外接盘 `/mnt/elements`；SMB 共享 `smb://guest@100.89.241.44/shared` = `~/shared`（生图/视频放这里）；H3 权重与日志在 `/mnt/elements/{hf-cache,logs}`；vLLM-Omni 在 `~/vllm-omni/.venv`；本项目路径 `~/workspace/test-deepseek-harness` |
| **runtime** (`100.120.147.125`, Ubuntu, RTX 2070 8 GB) | dsh 宿主 + 备胎模型 | `dsh.service` → https://runtime.tail97d99a.ts.net/ ；`runtime-brain.service`（Qwen3.5-9B, Vulkan, `127.0.0.1:8001`）；本项目有一份同步副本 |
| **bastion** (`100.110.50.87`, Mac mini M2 8 GB) | 跳板/客户端 | 磁盘 256 GB 且很紧，**不要在上面下载模型、跑训练或起 dsh** |

模型 API 都是 llama-server 的 OpenAI 兼容接口，端口 8001。dsh 里两个 provider：`furnance`（主脑）和 `runtime`（备胎），手动切换。

**核心约束**：furnance 的 GPU 被 brain 占着，任何离线 GPU 任务（LoRA 训练、生图、bake 视频）都必须
**先停 brain、做完再启 brain**。停 brain 期间 dsh 的 `furnance` provider 不可用，用 `runtime`。

---

## 0.5 两种运行模式（用户定义的术语）

| 模式 | brain（llama-brain.service） | gpu1 = nvidia-smi 索引 0 | gpu2 = nvidia-smi 索引 1 | 什么时候用 |
|---|---|---|---|---|
| **brain mode** | 运行，占满两张卡（各 ~22.5 GB） | brain | brain | dsh 用 furnance 主脑对话时 |
| **gen-mode** | 停止 | **视频**：MiniMax H3（ComfyUI 后端 :8188） | **图片**：SDXL / Chroma / LoRA 训练 | 出图、出视频、训练时 |

- 切换：`systemctl --user stop llama-brain.service`（进 gen-mode）/ `systemctl --user start llama-brain.service`（回 brain mode，加载 ~7 分钟）。gen-mode 期间 dsh 用 `runtime` provider。用户已明确“停 brain 不是问题”。
- 脚本默认已按 gen-mode 绑定：`tools/h3_server.sh` → `H3_GPU=0`；`tools/run_gen.sh` / `tools/run_chroma.sh` / `tools/run_lora_train.sh` → `GPU=1`。图片和视频可以**同时跑**（各占一张卡），只共享内存。
- 用户说“gpu1/gpu2”指的是这个 1 起数的编号；命令行里的 `CUDA_VISIBLE_DEVICES` 是 0 起数，别弄混。

---

## 1. 训练一次 LoRA（标准流程）

### 1.1 前置检查（30 秒）

```bash
ssh furnance bash -s <<'EOF'
cd ~/workspace/test-deepseek-harness
echo "brain: $(systemctl --user is-active llama-brain.service)"
echo "lora-train running? $(systemctl --user is-active lora-train.service 2>/dev/null)"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
echo "data/cropped: $(find data/cropped -type f | wc -l) files"
ls .models/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/*/unet/diffusion_pytorch_model* >/dev/null && echo "SDXL cache ok"
.venv/bin/python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
EOF
```

期望：brain `active`；`lora-train` 不是 active（否则已有训练在跑，别重复起）；`data/cropped` 有几百个文件；SDXL cache ok；cuda True。

数据目录说明：`data/raw` 原始视频 → `preprocess_enhanced.py --mode images` 抽帧到 `data/cropped`（570 帧）；`data/filtered` 是人工筛选目录，**目前为空**（所以不要用 `train.sh` 的默认值）。

### 1.2 启动（用项目里的包装脚本，跑成临时 systemd 单元）

```bash
ssh furnance 'cd ~/workspace/test-deepseek-harness && systemctl --user reset-failed lora-train.service 2>/dev/null; systemd-run --user --unit=lora-train --collect --description="SDXL LoRA training" tools/run_lora_train.sh'
```

`tools/run_lora_train.sh` 做的事：`systemctl --user stop llama-brain` → 在 GPU 0 上跑 `train_lora.py` → **trap EXIT 无条件 `start llama-brain`**。
默认超参：`data/cropped`、rank 32 / alpha 16、15 epochs、batch 4、grad-acc 4、lr 1e-4、512×512、每 2 epoch 存检查点、seed 42。
改超参用环境变量：`systemd-run --user --unit=lora-train --collect -E EPOCHS=20 -E DATA_DIR=data/filtered tools/run_lora_train.sh`。

为什么不用 `train.sh`：它传的 `--gradient_accumulation_steps/--mixed_precision/--max_steps` 与 `train_lora.py` 的 argparse 不匹配（会直接报错），而且有交互式 `read -p`。

### 1.3 监控

```bash
# 实时日志
ssh furnance 'journalctl --user -u lora-train -f'
# 只看 epoch 汇总
ssh furnance 'tr "\r" "\n" < ~/workspace/test-deepseek-harness/output/models/lora/training_log.txt | grep -E "Average Loss|已保存|Traceback|训练完成"'
# GPU
ssh furnance 'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader'
```

参考基线（570 帧、上述默认超参、单张 4090）：加载 + latents 预计算 ~1 分钟；~0.8 step/s；**全程约 11 分钟**；
loss 0.43 → 0.20 → 0.18 → … 第 6 轮后在 0.145–0.155 之间收敛。显存 ~18 GB。

### 1.4 完成后验证（必须做）

```bash
ssh furnance bash -s <<'EOF'
systemctl --user show lora-train.service -p Result -p ExecMainStatus 2>/dev/null
ls ~/workspace/test-deepseek-harness/output/models/lora/
echo "brain: $(systemctl --user is-active llama-brain.service)"
curl -s -m 3 http://127.0.0.1:8001/health || echo "(brain still loading, ~7 min after start)"
EOF
```

期望：`Result=success`；输出目录里有 `best_lora/`（loss 最低）、`final_lora/`、`lora_epoch_N/`，每个是 PEFT 适配器（`adapter_model.safetensors` ≈ 230 MB + `adapter_config.json`），旁边有 `*_config.json` 记录超参；brain `active`，加载完成后 `/health` 返回 `{"status":"ok"}`。

**如果 brain 不是 active，立刻 `ssh furnance 'systemctl --user start llama-brain.service'`。** 这是最不能漏的一步。

### 1.5 中止

```bash
ssh furnance 'systemctl --user stop lora-train.service'
```

trap 仍会恢复 brain。之后同样做 1.4 的 brain 检查。

---

## 2. 已知坑 / 历史

- `train_lora.py` 曾缺 `--save_every` 参数（代码引用了 `args.save_every`），第一次运行在 epoch 1 结束时 `AttributeError` 崩溃。已于 2026-08-16 补上（默认 2）。如果看到类似 `'Namespace' object has no attribute 'xxx'`，就是同类问题：对照 `grep -oE "args\.[a-z_]+" train_lora.py | sort -u` 和 `parse_args()` 里定义的参数。
- 直接在 shell 里 `python train_lora.py` 而不停 brain → CUDA OOM 或 5 秒后 failed（`train_checkpoint.json` 里那条 22:03 的记录就是）。
- `train_loop.py` 也有“停 brain”逻辑，但它是 `kill llama-server` 且 `BRAIN_CMD=None`——**训练完不会恢复 brain**。不要用它，用 `tools/run_lora_train.sh`。
- 停 brain 是 `systemctl --user stop`，不是 kill；恢复后要 ~7 分钟加载才 `/health` ok。
- 项目在三台机器上都有副本，**furnance 上的是训练用的权威副本**；runtime 上的是 dsh 工作区（agent 会改代码），bastion 上的不要再用。同步方向：改了代码后 `rsync -a --exclude .venv --exclude venv --exclude .models --exclude data --exclude output` 从改动的那台推到另一台。
- 一次 15 epoch 训练产出 ~2 GB（9 个检查点）。多跑几轮后把旧的搬到外接盘：`rsync -a output/models/lora/ /mnt/elements/lora-runs/<日期>/ && rm -rf output/models/lora/lora_epoch_*`。
- 大文件（模型、数据集、产出）放 furnance 的 `/mnt/elements`，不要放 bastion。

---

## 3. 用训练好的 LoRA 生图（已验证流程，2026-08-16）

### 3.1 约定

- 用户会以 **`[gen-img]`** 开头发提示词（旧写法 `[prompt]` 同义）。**模式后缀**：`[gen-img sdxl] …` 走 SDXL（标签式关键词、77 token 上限、LoRA 生态），`[gen-img chroma] …` 走 Chroma1-HD（自然语言长句、真 CFG，默认 36 步 / CFG 4.5）；**不写模式默认 chroma**（2026-08-16 用户定）。两者都通过队列提交：`tools/genq.py image "..." -m sdxl|chroma`（可能是完整句子，也可能是按类别列的词条）。收到后**不用再问**，直接执行；一行一条 prompt。词条式的输入自己组合成若干条完整 prompt（每条以 `beautiful woman, ...` 开头、以 `photorealistic` 结尾即可）。
- 默认参数：`--lora_path output/models/lora/best_lora --lora_weight 0.7 --num_steps 30 --seed 42`，1024×1024，guidance 7.0，负面词默认取 `prompts/negative_prompts.txt` 全部拼接。用户在同一条消息里说了强度/步数/尺寸/张数就按用户的。
- **输出必须放到 SMB 共享**下：`~/shared/lora-images/<YYYYMMDD-HHMM>-<批次名>/`（furnance 的 `/home/fshhr46/shared` 通过 Samba `[shared]` 公开、guest 可读写）。用户在任意 tailnet 设备上用 `smb://guest@100.89.241.44/shared` 看图。把本次 prompt 文件也复制一份到该目录（`prompts.txt`），脚本会自动写 `manifest.jsonl`（每张图的 prompt / seed / 参数）和 `generation_log.txt`。
- 显卡：gen-mode 下图片固定 **gpu2（索引 1）**，`run_gen.sh` / `run_chroma.sh` 默认就是；SDXL 1024²、30 步约 5–8 秒/张。brain mode 下不能出图，先切 gen-mode。视频在 gpu1 上跑时图片照常可跑。
- `generate.py` 已在 2026-08-16 重写为 SDXL + PEFT 版本（原文件在 `generate.py.orig`）：从 `.models/` 离线加载 `StableDiffusionXLPipeline`，用 `PeftModel.from_pretrained` 把 `best_lora/`（`adapter_config.json`+`adapter_model.safetensors`）注入 UNet，`--lora_weight` 走 `cross_attention_kwargs`。参数：`--lora_path --prompts --negative_prompts --output --lora_weight --num_steps --guidance --width --height --seed --device`。`--prompts` 既可以是文件也可以直接是一段文字。
- **Chroma1-HD**（FLUX 架构、Apache 2.0、未过滤社区模型，2026-08-16 加入）：`tools/run_chroma.sh <批次名>` 读同一个 `~/shared/prompt.txt`，出到 `~/shared/chroma-images/<时间>-<批次>/`；变量 `N/STEPS(26)/GUIDANCE(3.5)/SEED/W/H/GPU(1)/NEG/FP8`。底层 `tools/generate_chroma.py`（diffusers `ChromaPipeline`，权重 `lodestones/Chroma1-HD` 在 `HF_HOME=/mnt/elements/hf-cache`，24 GB 卡靠 `enable_model_cpu_offload`，每张 1024² 约 30–60 秒）。用户发 `[gen-img]` 未指定模型时默认仍是 SDXL；说“用 chroma”就走它。
- 用户自助出图：编辑 SMB 根目录的 `~/shared/prompt.txt`（一行一条），然后 `tools/run_gen.sh <批次名>`；`N=4` 每条出 4 张、`LORA_WEIGHT=0.3`、`STEPS/SEED/W/H/GPU/PROMPTS` 均可用环境变量覆盖。用户说“跑 prompt.txt”就是这个。
- LoRA 现状（2026-08-16）：v1 适配器一套上就出噪点（训练侧不一致导致），`train_lora.py` 已重写为 v2 但**用户决定暂不重训**；因此默认 `LORA_WEIGHT=0.0`（= 纯 SDXL 出图）。
- runtime 上 `prompts/generation_instructions.txt` 是 dsh agent 写的早期版本，里面 `~/.models` 路径写错了（实际是项目下 `.models/`），以本节为准。

### 3.2 执行模板（brain 正在跑的情况）

```bash
ssh furnance bash -s <<'EOF'
cd ~/workspace/test-deepseek-harness
TS=$(date +%Y%m%d-%H%M); NAME=<批次名>; OUT=~/shared/lora-images/${TS}-${NAME}; mkdir -p "$OUT"
cat > prompts/user_${TS}_${NAME}.txt <<'P'
<第 1 条 prompt>
<第 2 条 prompt>
P
cp prompts/user_${TS}_${NAME}.txt "$OUT/prompts.txt"
systemctl --user reset-failed lora-gen.service 2>/dev/null
systemd-run --user --unit=lora-gen --collect --description="SDXL LoRA gen ${NAME}" bash -c "cd ~/workspace/test-deepseek-harness; trap 'systemctl --user start llama-brain.service' EXIT; systemctl --user stop llama-brain.service; sleep 5; source .venv/bin/activate; HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python generate.py --lora_path output/models/lora/best_lora --prompts prompts/user_${TS}_${NAME}.txt --output $OUT --lora_weight 0.7 --num_steps 30 --seed 42 2>&1 | tee $OUT/generation_log.txt"
echo "OUT=$OUT"
EOF
```

brain 已经停着时（例如训练在跑）：去掉 `trap …` 和 `systemctl --user stop …`，`CUDA_VISIBLE_DEVICES=1`，单元名换成 `lora-gen-user` 之类避免冲突。

### 3.3 监控 / 完成后

```bash
ssh furnance 'systemctl --user is-active lora-gen.service; journalctl --user -u lora-gen -n 5'
ssh furnance 'ls ~/shared/lora-images/<目录>/ ; systemctl --user is-active llama-brain.service'
```

完成后向用户报告：输出目录路径（furnance 路径 + SMB 路径）、张数、用了哪些参数；并确认 brain `active`（停过 brain 的情况下）。若 brain 不是 active 且没有别的单元在用 GPU：`systemctl --user start llama-brain.service`。

### 3.4 内容边界（Claude 自己遵守）

内衣/泳装/姿态/表情类词条可以直接执行；明确的裸露与性行为姿势类词条（如 nipples showing / naked / see-through / spread legs / on all fours 等）Claude 不代为执行——告诉用户可以自己把词条写进文件后用 3.2 的命令跑，工具是现成的。这一条是 Claude 的行为约束，不是项目限制。

---

## 5. MiniMax H3 生视频（方案 B：无界面 ComfyUI 后端，gpu1）

### 5.1 约定
- 用户以 **`[gen-v]`** 开头请求视频。参数：`first=<图片路径或smb地址> last=<图片路径或smb地址> prompt=…`；只给 first = 单图生视频，不给图 = 纯文生视频。可附 `duration=5 steps=20 seed=7 name=批次名`。
- 用户通常**自己跑**；Claude 的职责是让后端可用、脚本正确、文档最新。若用户让 Claude 跑，直接执行。
- 图片路径接受 SMB 地址（`smb://100.89.241.44/shared/...`、`/Volumes/shared/...`），脚本自动换算到 `~/shared/...`。
- 输出固定：`~/shared/h3-videos/<时间>-<批次名>/video.mp4`（+ `request.json`、首/尾帧副本），即 `smb://guest@100.89.241.44/shared/h3-videos/`。
- 内容边界同 §3.4。

### 5.2 架构与决策记录
- **方案 A（vLLM-Omni）已放弃**（2026-08-16，六次尝试）：单卡放不下 32B 文本编码器；在线 fp8 量化要整模先上 GPU（24 GB 必 OOM）；bf16 + 双卡 DLO 需要 ≥200 GB 内存（furnance 125 GB，被 OOM-killer 杀）。代码保留在 `~/vllm-omni`（.venv 已装好 vLLM 0.27.1），脚本改名 `tools/run_h3_vllm.sh` / `tools/h3_server_vllm.sh`，日志 `/mnt/elements/logs/h3-server.log.try1..6`。**加内存到 256 GB 后**可复用双卡 DLO 配方。furnance 上多了 `/swapfile.h3`（64 GB，无害，留着）。
- **方案 B（现役）**：ComfyUI 0.30.0（`~/ComfyUI`，conda 环境 `comfy`，内置 H3 节点）当**无界面后端**，只监听 `127.0.0.1:8188`；权重用 Comfy-Org 官方量化包（DiT int8 剪枝 21 GB + Qwen3-VL 编码器 nvfp4 15.7 GB + 视频/音频 VAE），在 `/mnt/elements/models/minimax-h3/`，已软链进 `~/ComfyUI/models/{diffusion_models,text_encoders,vae}`。用户不想“用 ComfyUI”指的是界面；这里只是后台进程，用户操作仍是脚本。
- GPU：**gpu1（索引 0）**；显存峰值约 17–20 GB；内存 ~25 GB；nvfp4 在 Ada 上是软件模拟（编码慢一点，无碍）。
- 生成图 = 官方模板 `video_minimax_h3_{t2v,i2v}.json` 的子图手工展开（`tools/h3_comfy.py` 里 `build_graph`）：UNETLoader / CLIPLoader(type=minimax) / VAELoader×2 → MiniMaxH3ImageToVideo(clip, vae, prompt, width, height, length, first_frame?, last_frame?) → BasicGuider → SamplerCustomAdvanced(res_multistep, simple, 20 步) → VAEDecode + VAEDecodeAudio → CreateVideo(24fps) → SaveVideo。首/尾帧先 ImageScale 到画布尺寸。画布：768 短边、上限 768×1344、32 倍数，按首帧比例；帧数 = max(5, round(秒×24)) 向上贴到 17k+5。模板 JSON 存在 `tools/h3-workflows/`。

### 5.3 服务：`tools/h3_server.sh {start|stop|status|logs}`
```bash
ssh furnance 'systemctl --user is-active llama-brain.service'   # 必须 inactive（gen-mode）
ssh furnance '~/workspace/test-deepseek-harness/tools/h3_server.sh start'   # 秒起；首个请求加载权重 ~40 GB（外接盘，3–5 分钟）
ssh furnance '~/workspace/test-deepseek-harness/tools/h3_server.sh status'  # api=200 即可用；显示队列与显存
```
日志 `/mnt/elements/logs/h3-comfy.log`。后端常驻不用停；模型加载一次后留在显存/内存里。

### 5.4 生成：`tools/run_h3.sh`（= `tools/h3_comfy.py`，用 comfy 环境的 python）
```bash
# 首帧 + 尾帧 + prompt
ssh furnance '~/workspace/test-deepseek-harness/tools/run_h3.sh --first smb://100.89.241.44/shared/lora-images/X/a.png --last smb://100.89.241.44/shared/lora-images/X/b.png --prompt "The subject moves naturally from the first image to the last, gentle camera push-in" --duration 5 --name test1'
# 单图 / 纯文生视频
ssh furnance '~/workspace/test-deepseek-harness/tools/run_h3.sh --first ~/shared/lora-images/X/a.png --prompt "..." --duration 5'
ssh furnance '~/workspace/test-deepseek-harness/tools/run_h3.sh --prompt "..." --aspect 16:9 --duration 5'
```
可选：`--steps 20 --seed 42 --width W --height H --prompt-file 文件 --dry-run（打印 API 图不提交）`。脚本会上传图片到 ComfyUI、提交 `/prompt`、轮询 `/history`、把 mp4 从 `/view` 取回到输出目录。

### 5.5 H3 的 prompt 写法
分镜叙述式：主体+场景 → 有序动作（the shot begins… then… the video ends）→ 镜头运动 → 光线质感 → `Audio: …` → 结尾。示例 `prompts/h3_example_prompts.txt`。用图做首帧时写 “Preserve the subject from the reference image exactly”。

### 5.6 实测记录
- 2026-08-16 首条 5 秒 I2V（768×768，20 步）：见下方追加的耗时记录（跑通后由 Claude 写入）。

---

## 6. 队列服务 gen-queue（FastAPI，FIFO；`[gen-img]` / `[gen-v]` 的推荐入口）

- 服务：`tools/gen_server.py`，systemd 用户单元 `gen-api`，`tools/gen_server.sh {start|stop|restart|status|logs}`；监听 `0.0.0.0:8090`（tailnet：`http://100.89.241.44:8090`，Swagger 在 `/docs`）。日志 `/mnt/elements/logs/gen-api.log`，任务库 `/mnt/elements/gen-queue/jobs.db`，每个任务的执行日志 `/mnt/elements/gen-queue/logs/<id>.log`。
- 两个独立 FIFO 队列，各一个 worker、一次一个任务、互不阻塞：
  - `image` → gpu2（`CUDA_VISIBLE_DEVICES=1`）：`model=chroma`（`tools/generate_chroma.py`）或 `model=sdxl`（`generate.py`）；输出 `~/shared/{chroma-images|lora-images}/<时间>-<name>/`
  - `video` → gpu1（索引 0）：H3，经 `tools/h3_comfy.py` → 无界面 ComfyUI :8188（后端没起会自动 `h3_server.sh start`）；输出 `~/shared/h3-videos/<时间>-<name>/`
- brain mode 下提交的任务会直接 `failed`（错误里写明先切 gen-mode）。服务重启：pending 继续排队，running 标 `interrupted`。
- 客户端 `tools/genq.py`（纯标准库，bastion/runtime 上设 `GENQ_URL=http://100.89.241.44:8090` 即可用）：
```bash
tools/genq.py image "a photorealistic ..." -m chroma -n 4 --steps 36 --guidance 4.5 --seed 7 --name batch1 [--wait]
tools/genq.py image -f ~/shared/prompt.txt -m sdxl -n 2 --name batch2         # 从文件读多条
tools/genq.py video "The shot begins ..." --first smb://100.89.241.44/shared/chroma-images/X/a.png --last .../b.png --duration 5 --name v1 [--wait]
tools/genq.py status | list [-q image|video] [-s pending|running|done|failed] | get <id> | wait <id> | cancel <id>
```
- 直接调 HTTP：`POST /jobs/image` / `POST /jobs/video`（JSON 字段见 `gen_server.py` 顶部注释）、`GET /jobs/{id}`、`GET /jobs?queue=&status=`、`DELETE /jobs/{id}`、`GET /queues`。返回里 `position` 是排队位置，`smb` 是输出目录的 SMB 地址。
- Claude 处理 `[gen-img]` / `[gen-v]` 时优先走 `genq.py`（排队、不抢卡、有记录）；`run_gen.sh / run_chroma.sh / run_h3.sh` 仍可直接用（会和队列里的任务抢同一张卡，自己注意）。

---

## 4. 相关服务速查

| 服务 | 主机 | 命令 |
|---|---|---|
| 主脑 | furnance | `systemctl --user {status,start,stop,restart} llama-brain.service`；`journalctl --user -u llama-brain -n 50` |
| 训练 | furnance | `systemctl --user status lora-train.service`；`journalctl --user -u lora-train -f` |
| 生图 (SDXL) | furnance | `tools/run_gen.sh <批次名>`（gpu2=索引1，读 `~/shared/prompt.txt`）；产出 `~/shared/lora-images/<批次>/`；SMB `smb://guest@100.89.241.44/shared` |
| 队列服务 gen-queue | furnance | `tools/gen_server.sh {start,stop,status,logs}`（:8090，`/docs`）；`tools/genq.py image|video|status|list|get|wait|cancel` |
| 生视频 (H3, ComfyUI 后端) | furnance | `tools/h3_server.sh {start,stop,status,logs}`（gpu1=索引0，:8188）；`tools/run_h3.sh --first … --last … --prompt …`；产出 `~/shared/h3-videos/<批次>/`；日志 `/mnt/elements/logs/h3-comfy.log` |
| 生图 (Chroma) | furnance | `tools/run_chroma.sh <批次名>`（gpu2=索引1，读 `~/shared/prompt.txt`）；产出 `~/shared/chroma-images/<批次>/` |
| dsh | runtime | `systemctl --user restart dsh.service`；unit 在 `~/.config/systemd/user/dsh.service`（含 `DSH_LOCAL_KEY` 与 node PATH） |
| 备胎模型 | runtime | `systemctl --user status runtime-brain.service`（`127.0.0.1:8001`，alias `runtime-brain`） |
| dsh 配置 | runtime | `~/.dsh/cordis.patch.yml`（provider 列表）、`~/.dsh/sessions/`（对话，多帧 zstd，第一帧只放头行） |
