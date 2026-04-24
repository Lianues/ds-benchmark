"""
定时调度模块 —— 每隔 N 分钟并行调用 API 进行性能测试。

每个周期对所有已配置 API Key 的模型分别并行发起多次请求，
计算平均值后存入数据库，供 Web 前端查询。
"""

import threading
import time
import traceback
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

from ds_benchmark import (
    stream_chat,
    StreamMetrics,
    get_default_max_tokens,
    get_api_key,
    get_parallel,
    ALL_MODELS,
    DEFAULT_PROMPT,
)
from database import init_db, save_test_run, save_batch_result

# ---------------------------------------------------------------------------
# 全局状态 —— 供 web server 读取
# ---------------------------------------------------------------------------
scheduler_status: dict = {
    "is_running": False,
    "last_run_time": None,
    "next_run_time": None,
    "total_cycles": 0,
    "current_model": None,
}


# ---------------------------------------------------------------------------
# 1. run_single_test
# ---------------------------------------------------------------------------
def run_single_test(
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
) -> dict:
    """执行一次 stream_chat 并将 StreamMetrics 转为 dict（时间单位 ms）。

    失败时返回 success=False + error_msg，不抛异常。
    """
    try:
        metrics: StreamMetrics = stream_chat(
            api_key,
            prompt,
            model=model,
            max_tokens=max_tokens,
            verbose=False,
        )
        return {
            "success": True,
            "error_msg": "",
            # 时间指标 (ms)
            "ttft_ms": round(metrics.ttft * 1000, 2),
            "content_ttft_ms": round(metrics.ttft_content * 1000, 2),
            "thinking_duration_ms": round(metrics.thinking_duration * 1000, 2),
            "content_streaming_ms": round(metrics.content_streaming_duration * 1000, 2),
            "total_time_ms": round(metrics.total_time * 1000, 2),
            # token 统计
            "prompt_tokens": metrics.prompt_tokens,
            "completion_tokens": metrics.completion_tokens,
            "reasoning_tokens": metrics.reasoning_tokens,
            "output_tokens": metrics.output_tokens,
            "total_tokens": metrics.total_tokens,
            # 速度
            "content_tps": round(metrics.content_tps, 2),
            "reasoning_tps": round(metrics.reasoning_tps, 2),
            "overall_tps": round(metrics.tps_overall, 2),
            "chunk_speed": round(metrics.content_speed_by_chunks, 2),
        }
    except Exception as exc:
        return {
            "success": False,
            "error_msg": f"{type(exc).__name__}: {exc}",
            "ttft_ms": 0,
            "content_ttft_ms": 0,
            "thinking_duration_ms": 0,
            "content_streaming_ms": 0,
            "total_time_ms": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "content_tps": 0,
            "reasoning_tps": 0,
            "overall_tps": 0,
            "chunk_speed": 0,
        }


# ---------------------------------------------------------------------------
# 2. run_batch
# ---------------------------------------------------------------------------
def run_batch(
    api_key: str,
    model: str,
    prompt: str,
    parallel: int = 3,
) -> tuple:
    """对指定模型并行执行 *parallel* 次测试，保存结果到数据库。

    Returns:
        (batch_id, runs_list)
    """
    now = datetime.now()
    batch_id = f"{model}_{now.strftime('%Y%m%d_%H%M%S')}"
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    max_tokens = get_default_max_tokens(model)

    print(f"  ▶ [{model}] 并行 {parallel} 次测试 (max_tokens={max_tokens}) ...")

    # --- 并行执行 ---
    runs: list[dict] = [None] * parallel
    with ThreadPoolExecutor(max_workers=parallel) as executor:
        future_to_idx = {
            executor.submit(run_single_test, api_key, model, prompt, max_tokens): i
            for i in range(parallel)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            runs[idx] = future.result()

    # --- 保存每条记录 ---
    for idx, run in enumerate(runs):
        save_test_run(batch_id, model, idx, timestamp, run)

    # --- 保存批次聚合 ---
    save_batch_result(batch_id, model, timestamp, runs)

    # --- 控制台摘要 ---
    ok_runs = [r for r in runs if r["success"]]
    fail_count = parallel - len(ok_runs)

    if ok_runs:
        avg_ttft = sum(r["ttft_ms"] for r in ok_runs) / len(ok_runs)
        avg_total = sum(r["total_time_ms"] for r in ok_runs) / len(ok_runs)
        avg_tps = sum(r["overall_tps"] for r in ok_runs) / len(ok_runs)
        print(
            f"    ✔ {model}  成功 {len(ok_runs)}/{parallel}  "
            f"TTFT={avg_ttft:.0f}ms  总耗时={avg_total:.0f}ms  "
            f"TPS={avg_tps:.1f} t/s"
            + (f"  (失败 {fail_count})" if fail_count else "")
        )
    else:
        print(f"    ✘ {model}  全部失败 ({parallel}/{parallel})")

    return batch_id, runs


# ---------------------------------------------------------------------------
# 3. run_cycle
# ---------------------------------------------------------------------------
def run_cycle(prompt: str, parallel: int = 3) -> list:
    """对所有已配置 API Key 的模型并行执行一轮批次测试。

    每个模型使用各自的 API Key 和并发数，内部再并行 N 次。

    Returns:
        [(batch_id, runs_list), ...]
    """
    global scheduler_status

    # 筛选出有 API Key 的模型
    active_models = []
    for model in ALL_MODELS:
        key = get_api_key(model)
        if key:
            active_models.append((model, key))
        else:
            print(f"  ⚠ {model} 未配置 API Key，跳过")

    if not active_models:
        print("  ❌ 没有可用的模型（未配置任何 API Key）")
        return []

    cycle_start = datetime.now()
    model_names = [m for m, _ in active_models]
    total_concurrent = sum(get_parallel(m, parallel) for m, _ in active_models)
    print(f"\n{'='*60}")
    print(f"  🚀 新一轮测试  {cycle_start.strftime('%Y-%m-%d %H:%M:%S')}"
          f"  (共 {total_concurrent} 并发)")
    print(f"{'='*60}")

    scheduler_status["current_model"] = " + ".join(model_names)

    # 所有模型并行跑
    with ThreadPoolExecutor(max_workers=len(active_models)) as executor:
        futures = {
            executor.submit(run_batch, api_key, model, prompt,
                            get_parallel(model, parallel)): model
            for model, api_key in active_models
        }
        results = []
        for future in as_completed(futures):
            results.append(future.result())

    scheduler_status["current_model"] = None
    elapsed = (datetime.now() - cycle_start).total_seconds()
    print(f"{'─'*60}")
    print(f"  ✅ 本轮完成  耗时 {elapsed:.1f}s"
          f"  模型数 {len(active_models)}  并发 {total_concurrent}")
    print(f"{'='*60}\n")

    return results


# ---------------------------------------------------------------------------
# 4. start_scheduler
# ---------------------------------------------------------------------------
def start_scheduler(
    api_key: str = None,
    prompt: str = None,
    interval_minutes: int = 10,
    run_immediately: bool = True,
    parallel: int = 3,
) -> threading.Event:
    """启动后台定时调度器。

    Args:
        api_key: DeepSeek API Key（可选，向后兼容；优先使用 .env 中的配置）
        prompt: 测试 prompt（默认 DEFAULT_PROMPT）
        interval_minutes: 调度间隔（分钟）
        run_immediately: 是否立即执行第一轮
        parallel: 每模型并行调用数

    Returns:
        stop_event: 调用 stop_event.set() 可优雅停止调度器
    """
    if prompt is None:
        prompt = DEFAULT_PROMPT

    # 如果传入了 api_key，作为 DS_API_KEY 的备选
    if api_key:
        from ds_benchmark import set_api_key, _api_keys
        if "DS_API_KEY" not in _api_keys:
            set_api_key("DS_API_KEY", api_key)

    global scheduler_status
    stop_event = threading.Event()

    # 初始化数据库
    init_db()

    def _loop():
        global scheduler_status
        scheduler_status["is_running"] = True

        try:
            # --- 立即执行 ---
            if run_immediately:
                scheduler_status["last_run_time"] = datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                scheduler_status["total_cycles"] += 1
                run_cycle(prompt, parallel=parallel)

            # --- 定时循环 ---
            while not stop_event.is_set():
                next_time = datetime.now() + timedelta(minutes=interval_minutes)
                scheduler_status["next_run_time"] = next_time.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                print(
                    f"  ⏳ 下次执行: {scheduler_status['next_run_time']}  "
                    f"(间隔 {interval_minutes} 分钟)"
                )

                # 可中断地等待
                if stop_event.wait(timeout=interval_minutes * 60):
                    break  # 收到停止信号

                scheduler_status["last_run_time"] = datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                scheduler_status["next_run_time"] = None
                scheduler_status["total_cycles"] += 1
                run_cycle(prompt, parallel=parallel)

        except Exception:
            traceback.print_exc()
        finally:
            scheduler_status["is_running"] = False
            scheduler_status["next_run_time"] = None
            scheduler_status["current_model"] = None
            print("  🛑 调度器已停止")

    thread = threading.Thread(target=_loop, daemon=True, name="benchmark-scheduler")
    thread.start()

    return stop_event
