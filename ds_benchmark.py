#!/usr/bin/env python3
"""
DeepSeek API 流式性能基准测试工具
===================================
支持模型：
  - deepseek-chat      (V3)  : 普通对话模型，max_tokens=8192
  - deepseek-reasoner  (R1)  : 推理模型，max_tokens=65536

测量指标：
  - TTFT (Time To First Token)  : 首字时间 — 从请求发出到收到第一个内容 chunk 的耗时
  - 思考时间                    : 推理模型的思维链持续时间
  - 输出速度 (Output Speed)     : 基于流式 chunk 实测的 tokens/s
  - TPS (Tokens Per Second)     : 基于响应 usage 字段 + 总耗时计算的吞吐量
"""

import json
import os
import sys
import time
import argparse
from dataclasses import dataclass, field
from typing import Optional

import httpx

# ─── 配置 ───────────────────────────────────────────────────────────────────────

API_URL = "https://api.deepseek.com/chat/completions"

# 各模型默认 max_tokens
MODEL_MAX_TOKENS = {
    "deepseek-chat":     8192,    # V3: 8K
    "deepseek-reasoner": 65536,   # R1: 64K
}

ALL_MODELS = list(MODEL_MAX_TOKENS.keys())


def get_default_max_tokens(model: str) -> int:
    """根据模型名获取默认 max_tokens"""
    for key, val in MODEL_MAX_TOKENS.items():
        if key in model:
            return val
    return 8192


# ─── 数据类 ─────────────────────────────────────────────────────────────────────

@dataclass
class StreamMetrics:
    """流式性能测量数据"""
    model: str = ""

    # 时间戳
    request_start: float = 0.0
    first_chunk_time: float = 0.0
    first_reasoning_time: float = 0.0
    last_reasoning_time: float = 0.0
    first_content_time: float = 0.0
    last_chunk_time: float = 0.0
    request_end: float = 0.0

    # 计数
    chunk_count: int = 0
    reasoning_chunk_count: int = 0
    content_chunk_count: int = 0
    reasoning_char_count: int = 0
    content_char_count: int = 0

    # chunk 时间记录
    reasoning_timestamps: list = field(default_factory=list)
    content_timestamps: list = field(default_factory=list)

    # API 返回的 usage 信息
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    prompt_cache_hit_tokens: int = 0
    prompt_cache_miss_tokens: int = 0

    # 最终内容
    full_reasoning: str = ""
    full_content: str = ""

    # ── 是否有推理 ──

    @property
    def has_reasoning(self) -> bool:
        return self.reasoning_chunk_count > 0

    # ── 输出 token 数 (去掉 reasoning) ──

    @property
    def output_tokens(self) -> int:
        return self.completion_tokens - self.reasoning_tokens

    # ── 首字时间 ──

    @property
    def ttft(self) -> float:
        """首字时间 (秒) — 第一个有实际输出的 chunk (思维链或内容)"""
        first = 0.0
        if self.first_reasoning_time > 0:
            first = self.first_reasoning_time
        elif self.first_content_time > 0:
            first = self.first_content_time
        return (first - self.request_start) if first > 0 else 0.0

    @property
    def ttft_content(self) -> float:
        """正式内容首字时间 (秒) — 从请求发出到第一个 content chunk"""
        if self.first_content_time > 0:
            return self.first_content_time - self.request_start
        return 0.0

    # ── 思考时间 ──

    @property
    def thinking_duration(self) -> float:
        if self.first_reasoning_time > 0 and self.last_reasoning_time > 0:
            return self.last_reasoning_time - self.first_reasoning_time
        return 0.0

    @property
    def thinking_total(self) -> float:
        if self.last_reasoning_time > 0:
            return self.last_reasoning_time - self.request_start
        return 0.0

    # ── 总耗时 ──

    @property
    def total_time(self) -> float:
        return self.request_end - self.request_start

    @property
    def content_streaming_duration(self) -> float:
        if self.first_content_time > 0 and self.last_chunk_time > 0:
            return self.last_chunk_time - self.first_content_time
        return 0.0

    @property
    def reasoning_streaming_duration(self) -> float:
        if self.first_reasoning_time > 0 and self.last_reasoning_time > 0:
            return self.last_reasoning_time - self.first_reasoning_time
        return 0.0

    # ── 速度指标 ──

    @property
    def reasoning_speed_by_chunks(self) -> float:
        d = self.reasoning_streaming_duration
        if d > 0 and self.reasoning_chunk_count > 1:
            return (self.reasoning_chunk_count - 1) / d
        return 0.0

    @property
    def content_speed_by_chunks(self) -> float:
        d = self.content_streaming_duration
        if d > 0 and self.content_chunk_count > 1:
            return (self.content_chunk_count - 1) / d
        return 0.0

    @property
    def reasoning_tps(self) -> float:
        d = self.reasoning_streaming_duration
        if d > 0 and self.reasoning_tokens > 1:
            return (self.reasoning_tokens - 1) / d
        return 0.0

    @property
    def content_tps(self) -> float:
        d = self.content_streaming_duration
        ot = self.output_tokens
        if d > 0 and ot > 1:
            return (ot - 1) / d
        return 0.0

    @property
    def tps_overall(self) -> float:
        if self.total_time > 0 and self.completion_tokens > 0:
            return self.completion_tokens / self.total_time
        return 0.0


# ─── 核心逻辑 ────────────────────────────────────────────────────────────────────

def build_request_body(
    prompt: str,
    model: str = "deepseek-chat",
    max_tokens: int = 8192,
) -> dict:
    """构建请求体，根据模型自动调整参数"""
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "model": model,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    # deepseek-chat (V3) 特有参数
    if "reasoner" not in model:
        body.update({
            "thinking": {"type": "disabled"},
            "response_format": {"type": "text"},
        })

    return body


def parse_sse_line(line: str) -> Optional[dict]:
    line = line.strip()
    if not line or not line.startswith("data:"):
        return None
    data_str = line[len("data:"):].strip()
    if data_str == "[DONE]":
        return None
    try:
        return json.loads(data_str)
    except json.JSONDecodeError:
        return None


def stream_chat(
    api_key: str,
    prompt: str,
    model: str = "deepseek-chat",
    max_tokens: int = None,
    timeout: float = 600.0,
    verbose: bool = True,
) -> StreamMetrics:
    """发起流式请求并收集性能指标"""

    # 自动设置 max_tokens
    if max_tokens is None:
        max_tokens = get_default_max_tokens(model)

    metrics = StreamMetrics(model=model)
    body = build_request_body(prompt, model, max_tokens)
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "Authorization": f"Bearer {api_key}",
    }

    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    is_reasoning_phase = True

    if verbose:
        print("─" * 60)
        print(f"📡 Model     : {model}")
        print(f"🔧 Max Tokens: {max_tokens:,}")
        print(f"📝 Prompt    : {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
        print("─" * 60)

    metrics.request_start = time.perf_counter()

    with httpx.Client(timeout=httpx.Timeout(timeout)) as client:
        with client.stream("POST", API_URL, headers=headers, json=body) as response:
            if response.status_code != 200:
                body_text = response.read().decode()
                print(f"\n❌ 请求失败 HTTP {response.status_code}: {body_text}")
                metrics.request_end = time.perf_counter()
                return metrics

            if verbose:
                print()

            for raw_line in response.iter_lines():
                now = time.perf_counter()

                chunk = parse_sse_line(raw_line)
                if chunk is None:
                    continue

                metrics.chunk_count += 1
                if metrics.first_chunk_time == 0:
                    metrics.first_chunk_time = now

                # ── 提取 usage ──
                usage = chunk.get("usage")
                if usage:
                    metrics.prompt_tokens = usage.get("prompt_tokens", 0)
                    metrics.completion_tokens = usage.get("completion_tokens", 0)
                    metrics.total_tokens = usage.get("total_tokens", 0)
                    # cache 信息
                    metrics.prompt_cache_hit_tokens = usage.get("prompt_cache_hit_tokens", 0)
                    metrics.prompt_cache_miss_tokens = usage.get("prompt_cache_miss_tokens", 0)
                    # deepseek-reasoner: completion_tokens_details
                    details = usage.get("completion_tokens_details", {})
                    if details:
                        metrics.reasoning_tokens = details.get("reasoning_tokens", 0)

                # ── 提取 delta ──
                choices = chunk.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})

                # 思维链 (reasoning_content)
                reasoning = delta.get("reasoning_content", "")
                if reasoning:
                    metrics.reasoning_chunk_count += 1
                    metrics.reasoning_char_count += len(reasoning)
                    metrics.reasoning_timestamps.append(now)
                    reasoning_parts.append(reasoning)

                    if metrics.first_reasoning_time == 0:
                        metrics.first_reasoning_time = now
                        if verbose:
                            print("💭 [思维链开始]")

                    metrics.last_reasoning_time = now

                    if verbose:
                        print(f"\033[2m{reasoning}\033[0m", end="", flush=True)

                # 正式内容 (content)
                content = delta.get("content", "")
                if content:
                    if is_reasoning_phase and metrics.has_reasoning and verbose:
                        is_reasoning_phase = False
                        print(f"\n\n✍️  [正式输出开始]")

                    metrics.content_chunk_count += 1
                    metrics.content_char_count += len(content)
                    metrics.content_timestamps.append(now)
                    content_parts.append(content)

                    if metrics.first_content_time == 0:
                        metrics.first_content_time = now

                    metrics.last_chunk_time = now

                    if verbose:
                        print(content, end="", flush=True)

    metrics.request_end = time.perf_counter()
    metrics.full_reasoning = "".join(reasoning_parts)
    metrics.full_content = "".join(content_parts)

    if verbose:
        print("\n")

    return metrics


# ─── 报告输出 ────────────────────────────────────────────────────────────────────

def fmt_ms(seconds: float) -> str:
    """格式化时间：<1s 显示 ms，>=1s 显示 s"""
    if seconds < 1.0:
        return f"{seconds * 1000:.1f} ms"
    elif seconds < 60.0:
        return f"{seconds:.2f} s"
    else:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m {s:.1f}s"


def print_report(metrics: StreamMetrics):
    """打印格式化的性能报告"""
    has_reasoning = metrics.has_reasoning
    model_tag = "R1 推理" if has_reasoning else "V3 对话"

    W = 60  # 表宽

    print()
    print("╔" + "═" * W + "╗")
    title = f"DeepSeek {model_tag} 模型 · 流式性能报告"
    print("║" + title.center(W - 8) + "        ║")
    print("║" + f"Model: {metrics.model}".center(W - 8) + "        ║")
    print("╚" + "═" * W + "╝")

    # ── 时间指标 ──
    print()
    print("  ⏱  时间指标")
    print("  " + "─" * (W - 2))
    print(f"  │ {'🚀 TTFT (首字时间)':<30} │ {fmt_ms(metrics.ttft):>14}  │")

    if has_reasoning:
        print(f"  │ {'💭 思维链持续时间':<30} │ {fmt_ms(metrics.thinking_duration):>14}  │")
        print(f"  │ {'💭 请求到思考结束':<30} │ {fmt_ms(metrics.thinking_total):>14}  │")
        print(f"  │ {'✍️  正式内容首字时间':<29} │ {fmt_ms(metrics.ttft_content):>14}  │")

    print(f"  │ {'📨 内容流式持续时间':<30} │ {fmt_ms(metrics.content_streaming_duration):>14}  │")
    print(f"  │ {'⏳ 总耗时':<30} │ {fmt_ms(metrics.total_time):>14}  │")
    print("  " + "─" * (W - 2))

    # ── Token 信息 ──
    print()
    print("  🔢 Token 信息 (API usage)")
    print("  " + "─" * (W - 2))
    print(f"  │ {'Prompt Tokens':<34} │ {metrics.prompt_tokens:>10}      │")

    if metrics.prompt_cache_hit_tokens > 0:
        print(f"  │ {'  ├ Cache Hit':<34} │ {metrics.prompt_cache_hit_tokens:>10}      │")
        print(f"  │ {'  └ Cache Miss':<34} │ {metrics.prompt_cache_miss_tokens:>10}      │")

    if has_reasoning:
        print(f"  │ {'Reasoning Tokens (思维链)':<30} │ {metrics.reasoning_tokens:>10}      │")
        print(f"  │ {'Output Tokens (正式内容)':<31} │ {metrics.output_tokens:>10}      │")

    print(f"  │ {'Completion Tokens (总生成)':<30} │ {metrics.completion_tokens:>10}      │")
    print(f"  │ {'Total Tokens':<34} │ {metrics.total_tokens:>10}      │")
    print("  " + "─" * (W - 2))

    # ── Chunk 统计 ──
    print()
    print("  📦 流式 Chunk 统计")
    print("  " + "─" * (W - 2))

    if has_reasoning:
        print(f"  │ {'思维链 Chunk / 字符数':<30} │ {metrics.reasoning_chunk_count:>6} / {metrics.reasoning_char_count:<6}   │")

    print(f"  │ {'内容 Chunk / 字符数':<31} │ {metrics.content_chunk_count:>6} / {metrics.content_char_count:<6}   │")
    print(f"  │ {'总 Chunk 数':<34} │ {metrics.chunk_count:>10}      │")
    print("  " + "─" * (W - 2))

    # ── 速度指标 ──
    print()
    print("  ⚡ 速度指标")
    print("  " + "═" * (W - 2))

    if has_reasoning:
        print(f"  │ {'💭 思维链实测速度 (chunk)':<30} │ \033[33m{metrics.reasoning_speed_by_chunks:>10.2f} t/s\033[0m  │")
        print(f"  │ {'💭 思维链 TPS (usage)':<34} │ \033[33m{metrics.reasoning_tps:>10.2f} t/s\033[0m  │")
        print(f"  │ {'✍️  内容实测速度 (chunk)':<29} │ \033[32m{metrics.content_speed_by_chunks:>10.2f} t/s\033[0m  │")
        print(f"  │ {'✍️  内容 TPS (usage)':<33} │ \033[32m{metrics.content_tps:>10.2f} t/s\033[0m  │")
    else:
        print(f"  │ {'📊 实测输出速度 (chunk)':<30} │ \033[32m{metrics.content_speed_by_chunks:>10.2f} t/s\033[0m  │")
        print(f"  │ {'📊 内容 TPS (usage)':<34} │ \033[32m{metrics.content_tps:>10.2f} t/s\033[0m  │")

    print(f"  │ {'📊 整体 TPS (总token/总耗时)':<30} │ \033[36m{metrics.tps_overall:>10.2f} t/s\033[0m  │")
    print("  " + "═" * (W - 2))
    print()


def print_comparison(results: dict[str, StreamMetrics]):
    """打印多模型对比表"""
    print()
    print("╔" + "═" * 72 + "╗")
    print("║" + "📊 多模型性能对比".center(64) + "        ║")
    print("╚" + "═" * 72 + "╝")
    print()

    # 表头
    models = list(results.keys())
    header = f"  {'指标':<28}"
    for m in models:
        short = "V3 Chat" if "chat" in m else "R1 Reasoner"
        header += f" │ {short:>16}"
    print(header)
    print("  " + "─" * (28 + len(models) * 19))

    def row(label: str, values: list[str]):
        line = f"  {label:<28}"
        for v in values:
            line += f" │ {v:>16}"
        print(line)

    # TTFT
    row("🚀 TTFT (首字时间)",
        [fmt_ms(results[m].ttft) for m in models])

    # 内容首字时间 (reasoner 额外显示)
    row("✍️  内容首字时间",
        [fmt_ms(results[m].ttft_content) for m in models])

    # 思考时间
    if any(results[m].has_reasoning for m in models):
        row("💭 思维链时间",
            [fmt_ms(results[m].thinking_duration) if results[m].has_reasoning else "—" for m in models])

    # 内容流式时间
    row("📨 内容流式时间",
        [fmt_ms(results[m].content_streaming_duration) for m in models])

    # 总耗时
    row("⏳ 总耗时",
        [fmt_ms(results[m].total_time) for m in models])

    print("  " + "─" * (28 + len(models) * 19))

    # Tokens
    row("Prompt Tokens",
        [str(results[m].prompt_tokens) for m in models])

    if any(results[m].has_reasoning for m in models):
        row("Reasoning Tokens",
            [str(results[m].reasoning_tokens) if results[m].has_reasoning else "—" for m in models])

    row("Completion Tokens",
        [str(results[m].completion_tokens) for m in models])

    row("Output Tokens (内容)",
        [str(results[m].output_tokens) for m in models])

    print("  " + "─" * (28 + len(models) * 19))

    # 速度
    if any(results[m].has_reasoning for m in models):
        row("💭 思维链 TPS",
            [f"{results[m].reasoning_tps:.2f} t/s" if results[m].has_reasoning else "—" for m in models])

    row("✍️  内容 TPS (usage)",
        [f"{results[m].content_tps:.2f} t/s" for m in models])

    row("📊 实测速度 (chunk)",
        [f"{results[m].content_speed_by_chunks:.2f} t/s" for m in models])

    row("📊 整体 TPS",
        [f"{results[m].tps_overall:.2f} t/s" for m in models])

    print("  " + "═" * (28 + len(models) * 19))
    print()


# ─── 多轮测试 ────────────────────────────────────────────────────────────────────

def run_multi_round(
    api_key: str,
    prompt: str,
    rounds: int = 3,
    **kwargs,
) -> list[StreamMetrics]:
    """执行多轮测试并打印汇总"""
    all_metrics: list[StreamMetrics] = []

    for i in range(rounds):
        print(f"\n{'━' * 60}")
        print(f"  🔄 第 {i + 1}/{rounds} 轮测试")
        print(f"{'━' * 60}")
        m = stream_chat(api_key, prompt, **kwargs)
        print_report(m)
        all_metrics.append(m)

    if rounds > 1:
        has_reasoning = any(m.has_reasoning for m in all_metrics)

        print(f"\n{'━' * 60}")
        print(f"  📋 {rounds} 轮测试汇总 ({all_metrics[0].model})")
        print(f"{'━' * 60}")

        ttfts = [m.ttft * 1000 for m in all_metrics if m.ttft > 0]
        if ttfts:
            print(f"  TTFT         — 平均: {sum(ttfts)/len(ttfts):>8.1f} ms │ "
                  f"最小: {min(ttfts):>8.1f} ms │ 最大: {max(ttfts):>8.1f} ms")

        if has_reasoning:
            r_tps = [m.reasoning_tps for m in all_metrics if m.reasoning_tps > 0]
            c_tps = [m.content_tps for m in all_metrics if m.content_tps > 0]
            think_t = [m.thinking_duration * 1000 for m in all_metrics if m.thinking_duration > 0]

            if think_t:
                print(f"  思考时间     — 平均: {sum(think_t)/len(think_t):>8.1f} ms │ "
                      f"最小: {min(think_t):>8.1f} ms │ 最大: {max(think_t):>8.1f} ms")
            if r_tps:
                print(f"  思维链 TPS   — 平均: {sum(r_tps)/len(r_tps):>8.2f} t/s │ "
                      f"最小: {min(r_tps):>8.2f} t/s │ 最大: {max(r_tps):>8.2f} t/s")
            if c_tps:
                print(f"  内容 TPS     — 平均: {sum(c_tps)/len(c_tps):>8.2f} t/s │ "
                      f"最小: {min(c_tps):>8.2f} t/s │ 最大: {max(c_tps):>8.2f} t/s")
        else:
            tps_list = [m.content_tps for m in all_metrics if m.content_tps > 0]
            if tps_list:
                print(f"  内容 TPS     — 平均: {sum(tps_list)/len(tps_list):>8.2f} t/s │ "
                      f"最小: {min(tps_list):>8.2f} t/s │ 最大: {max(tps_list):>8.2f} t/s")

        overall = [m.tps_overall for m in all_metrics if m.tps_overall > 0]
        if overall:
            print(f"  整体 TPS     — 平均: {sum(overall)/len(overall):>8.2f} t/s │ "
                  f"最小: {min(overall):>8.2f} t/s │ 最大: {max(overall):>8.2f} t/s")
        print()

    return all_metrics


# ─── CLI ─────────────────────────────────────────────────────────────────────────

DEFAULT_PROMPT = """请以"时光的河流"为题，写一篇不少于3000字的散文。

要求：
1. 文章需要有深度的思考和细腻的情感表达
2. 结构完整，分为多个章节，每个章节有小标题
3. 融入对历史、人生、自然的思考
4. 运用丰富的修辞手法：比喻、拟人、排比、对比等
5. 结尾要有升华，给人以启迪
6. 请用优美的中文写作"""


def main():
    parser = argparse.ArgumentParser(
        description="DeepSeek API 流式性能基准测试工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单模型测试
  python ds_benchmark.py -k sk-xxx -m deepseek-chat "写一篇散文"
  python ds_benchmark.py -k sk-xxx -m deepseek-reasoner "写一篇散文"

  # 对比测试 (同时测 V3 + R1)
  python ds_benchmark.py -k sk-xxx -c "写一篇散文"

  # 多轮测试
  python ds_benchmark.py -k sk-xxx -m deepseek-chat -n 3 "Hello"
        """,
    )
    parser.add_argument("prompt", nargs="?", default=DEFAULT_PROMPT,
                        help="用户 prompt (默认: 写一篇3000字散文)")
    parser.add_argument("-k", "--api-key", default=None,
                        help="DeepSeek API Key")
    parser.add_argument("-m", "--model", default=None,
                        help="模型名称: deepseek-chat (V3) 或 deepseek-reasoner (R1)")
    parser.add_argument("-c", "--compare", action="store_true",
                        help="对比模式：依次测试 deepseek-chat 和 deepseek-reasoner")
    parser.add_argument("-n", "--rounds", type=int, default=1,
                        help="每个模型的测试轮数 (默认: 1)")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="最大生成 token 数 (默认: chat=8192, reasoner=65536)")
    parser.add_argument("--timeout", type=float, default=600.0,
                        help="请求超时秒数 (默认: 600)")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="静默模式，不实时打印生成内容")

    args = parser.parse_args()

    # API Key
    api_key = args.api_key or os.environ.get("DS_API_KEY", "")
    if not api_key:
        env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        if os.path.exists(env_file):
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("DS_API_KEY="):
                        api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break

    if not api_key:
        print("❌ 请提供 API Key:")
        print("   方式 1: python ds_benchmark.py -k sk-xxx 'your prompt'")
        print("   方式 2: 设置环境变量 DS_API_KEY=sk-xxx")
        print("   方式 3: 创建 .env 文件并写入 DS_API_KEY=sk-xxx")
        sys.exit(1)

    verbose = not args.quiet

    # ── 对比模式 ──
    if args.compare:
        comparison_results: dict[str, StreamMetrics] = {}

        for model in ALL_MODELS:
            max_tok = args.max_tokens if args.max_tokens else get_default_max_tokens(model)

            print(f"\n{'━' * 60}")
            print(f"  🏁 测试模型: {model} (max_tokens={max_tok:,})")
            print(f"{'━' * 60}")

            if args.rounds > 1:
                results = run_multi_round(
                    api_key=api_key, prompt=args.prompt, rounds=args.rounds,
                    model=model, max_tokens=max_tok,
                    timeout=args.timeout, verbose=verbose,
                )
                # 取最后一轮作为对比数据
                comparison_results[model] = results[-1]
            else:
                m = stream_chat(
                    api_key=api_key, prompt=args.prompt,
                    model=model, max_tokens=max_tok,
                    timeout=args.timeout, verbose=verbose,
                )
                print_report(m)
                comparison_results[model] = m

        # 打印对比表
        print_comparison(comparison_results)
        return

    # ── 单模型模式 ──
    model = args.model or "deepseek-chat"
    max_tok = args.max_tokens if args.max_tokens else get_default_max_tokens(model)

    kwargs = dict(
        model=model,
        max_tokens=max_tok,
        timeout=args.timeout,
        verbose=verbose,
    )

    if args.rounds > 1:
        run_multi_round(api_key=api_key, prompt=args.prompt, rounds=args.rounds, **kwargs)
    else:
        metrics = stream_chat(api_key=api_key, prompt=args.prompt, **kwargs)
        print_report(metrics)


if __name__ == "__main__":
    main()
