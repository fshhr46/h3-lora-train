# dsh (DeepSeek harness) 服务化建议

> 写于 2026-08-15，起因：今晚重启 `https://furnance.tail97d99a.ts.net/` 时踩了两个坑，
> 说明现在的 nohup 裸跑方式不够稳。本文记录现状、问题和建议的 systemd 方案。

## 现状（截至 2026-08-15 22:40）

| 项目 | 值 |
|---|---|
| 运行位置 | **furnance 本机**（不是 bastion） |
| 程序 | `@deepseek-ai/dsh` 0.1.0-rc.6，`dsh web` 子命令 |
| 启动命令 | `npx @deepseek-ai/dsh web --host 127.0.0.1 --port 3080 --trusted-host furnance.tail97d99a.ts.net` |
| node 来源 | conda 环境 `~/miniforge3/envs/dsh`（node v26.6.0）——**不在**非交互 shell 的 PATH 里 |
| 必需环境变量 | `DSH_LOCAL_KEY`（任意非空占位值） |
| 对外入口 | `tailscale serve`：`https://furnance.tail97d99a.ts.net/` → `http://127.0.0.1:3080` |
| 后端模型 | `llama-brain.service`（llama-server，`127.0.0.1:8001`，alias `suno-brain`） |
| dsh 配置 | `~/.dsh/settings.yaml`、`~/.dsh/cordis.patch.yml`、`~/.dsh/storages/` |
| 工作区 | `/home/fshhr46/workspace/test-deepseek-harness`（本目录） |
| 日志 | `/tmp/dsh.log`（nohup 重定向） |
| 进程管理 | **无**——nohup 后台进程，机器重启不会拉起，崩了不会重启 |

## 今晚踩的坑

1. **`npx: No such file or directory`**
   非交互 ssh（甚至 `bash -lc`）的 PATH 里没有 node。当初启动时是在激活了 conda `dsh` 环境的交互 shell 里跑的，
   这个前提没有记录在任何地方。
2. **`no credential for provider route "furnance-local"; its profile resolves DSH_LOCAL_KEY, which is not set`**
   `~/.dsh/cordis.patch.yml` 里 `furnance-local` 声明了 `apiKeyEnv: DSH_LOCAL_KEY`。llama-server 本身不要 key，
   但 pi-ai 的 OpenAI 兼容适配器强制要一个非空凭证，所以启动 dsh 时必须 `export DSH_LOCAL_KEY=<任意值>`。
   patch 文件的注释里写了这一点，但启动命令本身没有固化它。
3. `pkill -f "dsh web"` 会把自己所在的 shell 一起杀掉（命令行里也含这个字符串）。用 `pgrep -f "bin/dsh web"` 更安全。

结论：两条隐式前提（PATH、DSH_LOCAL_KEY）都应该写进一个 unit 文件里，和 `llama-brain.service` 一样受 systemd 管理。

## 建议：`~/.config/systemd/user/dsh.service`

```ini
[Unit]
Description=dsh — DeepSeek harness web UI (127.0.0.1:3080, behind tailscale serve)
# 后端是 llama-brain；dsh 起来后再拉起也没关系，但顺序上放它后面更合理。
After=network.target llama-brain.service
Wants=llama-brain.service

[Service]
WorkingDirectory=%h
# node 来自 conda 的 dsh 环境；非交互 shell 的 PATH 里没有它，所以在这里显式给出。
Environment=PATH=%h/miniforge3/envs/dsh/bin:/usr/local/bin:/usr/bin:/bin
# 占位凭证。~/.dsh/cordis.patch.yml 里 furnance-local 的 apiKeyEnv 指向它；
# llama-server 不校验，但 pi-ai 的 openai-completions 适配器要求非空，否则每轮对话都报
# "no credential for provider route furnance-local"。
Environment=DSH_LOCAL_KEY=local-placeholder
# 用 npx 是为了和当前行为一致（复用 ~/.npm/_npx 里已装好的 0.1.0-rc.6）。
# 若以后想锁版本，可改成 `npm i -g @deepseek-ai/dsh@<ver>` 后直接 ExecStart=%h/miniforge3/envs/dsh/bin/dsh web ...
ExecStart=%h/miniforge3/envs/dsh/bin/npx @deepseek-ai/dsh web \
  --host 127.0.0.1 --port 3080 \
  --trusted-host furnance.tail97d99a.ts.net
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

## 安装 / 切换步骤

```bash
# 1. 停掉当前 nohup 进程
kill $(pgrep -f "bin/dsh web")

# 2. 写入 unit（把上面的内容保存为下面这个文件）
$EDITOR ~/.config/systemd/user/dsh.service

# 3. 加载并启用（开机自启，与 llama-brain 相同）
systemctl --user daemon-reload
systemctl --user enable --now dsh.service

# 4. 验证
systemctl --user status dsh.service --no-pager
ss -ltnp | grep 3080
curl -s -o /dev/null -w "%{http_code}\n" https://furnance.tail97d99a.ts.net/
```

> 如果 `loginctl show-user fshhr46 | grep Linger` 不是 `Linger=yes`，用户级 unit 在没人登录时不会启动；
> `llama-brain.service` 已经在用同样的机制并且能开机自启，所以应该已经开了 linger，可以顺手确认一下。

## 日常操作（切换后）

```bash
systemctl --user restart dsh.service      # 重启
systemctl --user stop dsh.service         # 停止
journalctl --user -u dsh.service -f       # 看日志（替代 /tmp/dsh.log）
```

## 之后升级 dsh 版本

`npx @deepseek-ai/dsh` 会用 `~/.npm/_npx` 里的缓存；要升级就 `npx @deepseek-ai/dsh@latest --version` 拉一次新包，
然后 `systemctl --user restart dsh.service`。升级前先看 `~/.dsh/cordis.patch.yml` 里 `llm-pi-ai` 的 provider
配置格式有没有变（尤其是 `apiKeyEnv` 那条规则）。
