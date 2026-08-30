# SHUTDOWN.md — 关闭 / 停机方式

对应 `STARTUP.md`。按“想省多少电 / 要不要还能远程唤醒”选一种。命令都在 **bastion** 上执行（或直接 ssh 到对应机器）。
最后更新：2026-08-22。

---

## 方式对照

| 方式 | 省电 | 还能远程恢复？ | 恢复速度 | 何时用 |
|---|---|---|---|---|
| A. 只停服务（不关机） | 少（GPU 空载 ~60 W，整机 ~150 W） | ✅ ssh 一直在 | 秒级起服务 | 只想腾出 GPU / 临时停 |
| B. Suspend（睡眠） | 多（整机降到待机几十 W） | ✅ Wake-on-LAN | ~5–10 秒 | 白天不用、晚上还想随时叫醒 |
| C. Poweroff（关机） | 最多（~0 W） | ❌ 只能到现场按电源键 | 开机 + 加载 ~几分钟 | 长期不用 / 要断电搬动 |

> ⚠️ **poweroff 之后无法 Wake-on-LAN**（网卡断电），只有 suspend 能远程唤醒。要能远程叫醒就用 B。

---

## A. 只停服务（保持开机，最常用）

**furnance** —— 停生成栈 + 主脑：
```bash
ssh furnance '~/workspace/test-deepseek-harness/tools/gen_server.sh stop 2>/dev/null; \
              ~/workspace/test-deepseek-harness/tools/h3_server.sh stop 2>/dev/null; \
              systemctl --user stop llama-brain.service'
ssh furnance 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader'   # 应为 ~1 MiB
```
**runtime** —— 停备胎模型（dsh 想留着就别停 dsh.service）：
```bash
ssh runtime 'systemctl --user stop runtime-brain.service'
# 连 dsh 也停：ssh runtime 'systemctl --user stop dsh.service'
```

恢复：见 `STARTUP.md`（或直接 `systemctl --user start <服务>`）。

---

## B. Suspend（睡眠，可 WoL 唤醒）

**睡之前务必先停占显存的服务**（否则挂起要把几十 GB 显存写盘，很慢甚至失败）：
```bash
# furnance：先按 A 停掉 llama-brain / 生成栈，再睡
ssh furnance '~/workspace/test-deepseek-harness/tools/gen_server.sh stop 2>/dev/null; \
              ~/workspace/test-deepseek-harness/tools/h3_server.sh stop 2>/dev/null; \
              systemctl --user stop llama-brain.service'
zzz furnance          # bastion 上：ssh 让它睡 + 确认已下线

# runtime：备胎模型 5.6 GB，可停可不停；dsh 睡后会断线
ssh runtime 'systemctl --user stop runtime-brain.service'
zzz runtime
```
唤醒（bastion 上）：
```bash
wake furnance    # WoL，~5–10 秒可 ssh；autostart 服务自动恢复
wake runtime     # 唤醒后 dsh + runtime-brain 自动回来
```
细节 / 原理见 `operation.md §7`。`zzz`/`wake` 脚本在 bastion `~/bin/`。

---

## C. Poweroff（彻底关机，需 sudo，到现场才能开）

```bash
ssh furnance 'sudo systemctl poweroff'     # 会提示输密码；ssh 随即断开
ssh runtime  'sudo systemctl poweroff'
```
重启（reboot，不是关机，会自己回来）：
```bash
ssh furnance 'sudo systemctl reboot'
ssh runtime  'sudo systemctl reboot'
```
关机后再开机只能**现场按电源键**（无 WoL）。开机后 autostart 服务（furnance `llama-brain`、runtime `dsh`+`runtime-brain`）会自动起来，主脑加载 ~7 分钟；生成队列仍需手动（见 STARTUP.md 场景二）。

---

## 快速决策
- 马上还要用 / 只是换 gen-mode → **A**
- 今天不用了、想省电又要能远程叫醒 → **B（suspend + WoL）**
- 出差、搬机器、长期闲置 → **C（poweroff）**，回来现场开机
