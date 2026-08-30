# STARTUP.md — 系统总览与启动步骤

面向：重启/断电/唤醒之后，快速把整套系统拉到可用状态。详细操作见 `operation.md`。
最后更新：2026-08-22。

## 机器拓扑

| 机器 | 角色 | 地址 | 硬件 |
|---|---|---|---|
| **furnance** | 模型 + 生成主力 | LAN `10.0.0.182` / tailnet `100.89.241.44` | 2×RTX 4090、125 GB RAM、13 TB 外接盘 `/mnt/elements` |
| **runtime** | dsh 宿主 + 备胎模型 | LAN `10.0.0.171` / tailnet `100.120.147.125` | RTX 2070、16 GB RAM |
| **bastion** | 客户端 / 跳板（Mac mini） | LAN `10.0.0.168` | M2、8 GB |

三台同一局域网 `10.0.0.0/24`，SSH 别名已配好（`ssh furnance` / `ssh runtime`），runtime 也能免密 `ssh furnance`。

## 服务清单与自启情况（两台 Linux 都开了 linger，用户级服务无需登录即自启）

| 服务 | 机器 | 作用 | 重启后 |
|---|---|---|---|
| `llama-brain.service` | furnance | 35B 主脑（dsh 的 furnance provider，占**两张**卡 ~45 GB） | **自启** ✅ |
| `dsh.service` | runtime | DeepSeek harness Web（https://runtime.tail97d99a.ts.net/） | **自启** ✅ |
| `runtime-brain.service` | runtime | 9B 备胎模型（dsh 的 runtime provider） | **自启** ✅（2026-08-22 起 enable） |
| `gen-api`（gen-queue） | furnance | 图片/视频生成队列（:8090） | ❌ 手动（与主脑抢 GPU） |
| `h3-comfy`（H3 后端） | furnance | MiniMax H3 视频（ComfyUI :8188） | ❌ 手动（gen-api 提交视频时自动拉起） |

## 场景一：只用 dsh 对话（默认，开箱即用）

重启后 `llama-brain`、`dsh`、`runtime-brain` 都会自启，**通常什么都不用做**。验证：

```bash
ssh furnance 'curl -s -m3 http://127.0.0.1:8001/health'      # {"status":"ok"} = 35B 主脑就绪（加载需 ~7 分钟）
ssh runtime  'curl -s -m3 http://127.0.0.1:8001/health'      # 9B 备胎
curl -s -o /dev/null -w "%{http_code}\n" https://runtime.tail97d99a.ts.net/   # 200 = dsh 在线
```

打开 **https://runtime.tail97d99a.ts.net/**，Models 页面切 `furnance`(35B) 或 `runtime`(9B)。

若某个模型没起：
```bash
ssh furnance 'systemctl --user start llama-brain.service'    # 主脑
ssh runtime  'systemctl --user start runtime-brain.service'  # 备胎
ssh runtime  'systemctl --user restart dsh.service'          # dsh 本身
```

## 场景二：出图 / 出视频（gen-mode）

furnance 的生成栈和 35B 主脑**抢同一批 GPU**，必须先停主脑（gen-mode），dsh 期间改用 runtime 备胎。

```bash
# 1) 切 gen-mode：停主脑
ssh furnance 'systemctl --user stop llama-brain.service'
# 2) 起队列服务（视频后端会按需自动拉起）
ssh furnance '~/workspace/test-deepseek-harness/tools/gen_server.sh start'
ssh furnance '~/workspace/test-deepseek-harness/tools/genq.py status'   # 确认 image/video 队列就绪
```

提交任务（gpu 索引 0=视频/gpu1、索引 1=图片/gpu2，可同时跑）：
```bash
# 图片：SDXL 系（默认 realvis）/ chroma / 动漫 noobai+LoRA
ssh furnance '~/workspace/test-deepseek-harness/tools/genq.py image "..." --base realvis -n 4 --name b1'
ssh furnance '~/workspace/test-deepseek-harness/tools/genq.py image "..." -m chroma --steps 36 --guidance 4.5 --name b2'
ssh furnance '~/workspace/test-deepseek-harness/tools/genq.py image "..." --base noobai --lora reparkz_cknb_v01-000025 --lora_weight 0.8 --name b3'
# 视频：MiniMax H3（首帧/尾帧 + prompt）
ssh furnance '~/workspace/test-deepseek-harness/tools/genq.py video "..." --first smb://100.89.241.44/shared/xxx/a.png --duration 5 --name v1'
ssh furnance '~/workspace/test-deepseek-harness/tools/genq.py status | list | get <id> | wait <id> | cancel <id>'
```
产出：`smb://guest@100.89.241.44/shared/{lora-images,chroma-images,h3-videos}/<批次>/`（Finder 连 `smb://100.89.241.44/shared` 访客登录）。

切回 dsh 对话（brain mode）：
```bash
ssh furnance '~/workspace/test-deepseek-harness/tools/gen_server.sh stop; ~/workspace/test-deepseek-harness/tools/h3_server.sh stop'
ssh furnance 'systemctl --user start llama-brain.service'
```

## 场景三：省电 / 唤醒（Wake-on-LAN，在 bastion 上）

```bash
zzz furnance      # ssh 让它睡（睡前最好先停占显存的服务）
wake furnance     # WoL 唤醒，~5–10 秒可 ssh；服务/显存自动恢复
zzz runtime  /  wake runtime
```
前提：机器一直插网线且通电（poweroff 无法 WoL，只有 suspend 可以）。细节见 `operation.md §7`。

## 训练 LoRA
见 `operation.md §1`：`ssh furnance '... systemd-run --user --unit=lora-train ... tools/run_lora_train.sh'`（自动停/启主脑）。

## 关键路径
- 项目（本仓库）：furnance & runtime 的 `~/workspace/test-deepseek-harness`（GitHub `fshhr46/h3-lora-train` 分支 `clean`）
- 大文件 / 权重 / 队列库：furnance `/mnt/elements`（外接盘）
- SMB 共享看图看视频：`smb://guest@100.89.241.44/shared`
- dsh 配置：runtime `~/.dsh/`
