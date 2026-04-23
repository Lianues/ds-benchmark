# DeepSeek API 流式性能基准测试工具

一个用于测量 DeepSeek API 流式输出性能指标的命令行工具。

## 测量指标

| 指标 | 说明 |
|------|------|
| **TTFT (首字时间)** | 从请求发出到收到第一个有内容的 chunk 的耗时 |
| **实测输出速度** | 基于流式 chunk 到达时间实测的 tokens/s |
| **TPS (流式时间)** | `completion_tokens / 流式持续时间` — 基于 API 返回的 usage 信息 |
| **TPS (总耗时)** | `completion_tokens / 总耗时` — 包含首字等待时间的整体吞吐量 |

## 安装

```bash
cd ds-benchmark
pip install -r requirements.txt
```

## 使用

### 基本用法

```bash
# 方式 1: 通过命令行参数传入 API Key
python ds_benchmark.py -k sk-xxxx "你好，请介绍一下你自己"

# 方式 2: 通过环境变量
export DS_API_KEY=sk-xxxx
python ds_benchmark.py "你好，请介绍一下你自己"

# 方式 3: 通过 .env 文件
cp .env.example .env
# 编辑 .env 填入你的 API Key
python ds_benchmark.py "你好"
```

### 多轮测试

```bash
# 执行 5 轮测试，最后输出汇总统计
python ds_benchmark.py -k sk-xxxx -n 5 "请写一首关于春天的诗"
```

### 高级选项

```bash
python ds_benchmark.py -k sk-xxxx \
  -m deepseek-chat \          # 指定模型
  --max-tokens 2048 \         # 最大生成 token 数
  -t 0.7 \                    # 温度参数
  -s "你是一个诗人" \           # 系统 prompt
  --timeout 60 \              # 超时秒数
  -q \                        # 静默模式 (不实时打印)
  "写一首诗"
```

### 参数说明

```
positional arguments:
  prompt                用户 prompt (默认: 介绍四大发明)

options:
  -k, --api-key         DeepSeek API Key
  -m, --model           模型名称 (默认: deepseek-chat)
  -n, --rounds          测试轮数 (默认: 1)
  -s, --system          系统 prompt (默认: You are a helpful assistant)
  --max-tokens          最大生成 token 数 (默认: 4096)
  -t, --temperature     温度参数 (默认: 1.0)
  --timeout             请求超时秒数 (默认: 120)
  -q, --quiet           静默模式
```

## 输出示例

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  🔄 第 1/1 轮测试
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
────────────────────────────────────────────────────────────
📡 Model: deepseek-chat
📝 Prompt: 你好
────────────────────────────────────────────────────────────

你好！有什么我可以帮助你的吗？😊

┌──────────────── ⏱  时间指标 ────────────────┐
│ 指标                      │   数值           │
├───────────────────────────┼──────────────────┤
│ 🚀 TTFT (首字时间)        │     532.3 ms     │
│ 📦 首 Chunk 时间          │     530.1 ms     │
│ 📨 流式输出持续时间       │     287.5 ms     │
│ ⏳ 总耗时                 │     823.6 ms     │
└───────────────────────────┴──────────────────┘

┌──────────────── ⚡ 速度指标 ────────────────┐
│ 指标                      │   数值           │
├───────────────────────────┼──────────────────┤
│ 📊 实测输出速度 (chunk)   │   38.26 tokens/s │
│ 📊 TPS (usage/流式时间)   │   34.78 tokens/s │
│ 📊 TPS (usage/总耗时)     │   13.35 tokens/s │
└───────────────────────────┴──────────────────┘
```

## 指标计算说明

### TTFT (Time To First Token) — 首字时间
```
TTFT = 第一个有 delta.content 的 chunk 到达时间 - 请求发出时间
```
反映了 API 的响应延迟，包含网络传输 + 服务端排队 + 预填充(prefill)时间。

### 实测输出速度 (Output Speed)
```
Output Speed = (有内容的 chunk 数 - 1) / (最后一个 chunk 时间 - 第一个内容 chunk 时间)
```
基于流式 chunk 的实际到达时间计算。由于每个 chunk 通常对应 1 个 token，可以近似看作实际的 token 输出速率。

### TPS — 基于 usage 的 Tokens Per Second
```
TPS (流式) = (completion_tokens - 1) / 流式持续时间
TPS (总体) = completion_tokens / 总耗时
```
使用 API 返回的 `usage.completion_tokens` 精确数值计算。流式 TPS 更能反映解码(decode)阶段的实际速度；总体 TPS 包含了首字等待时间。

> 注: 流式 TPS 计算中 `-1` 是因为第一个 token 标志着输出开始，不应计入生成速率的分子中。
