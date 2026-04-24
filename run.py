#!/usr/bin/env python3
"""
DeepSeek / GLM API 性能监控服务
================================
启动后会:
  1. 在后台定时(每10分钟)对所有已配置 Key 的模型各并行调用 3 次
  2. 将测试结果(含平均值)存入 SQLite 数据库
  3. 启动 Web 服务器，提供可视化仪表盘

用法:
  python run.py -k sk-xxxx
  python run.py -k sk-xxxx --port 8080 --interval 5
"""

import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(
        description="DeepSeek / GLM API 性能监控服务",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run.py -k sk-xxxx                          # 默认配置启动
  python run.py -k sk-xxxx --port 8080               # 自定义端口
  python run.py -k sk-xxxx --interval 5              # 每5分钟测试
  python run.py -k sk-xxxx --no-immediate            # 启动时不立即测试
        """,
    )
    parser.add_argument("-k", "--api-key", default=None, help="DeepSeek API Key")
    parser.add_argument("-p", "--port", type=int, default=8080, help="Web 服务器端口 (默认: 8080)")
    parser.add_argument("-i", "--interval", type=int, default=None, help="测试间隔(分钟) (默认: 读取 .env 的 INTERVAL_MINUTES，否则 10)")
    parser.add_argument("--no-immediate", action="store_true", help="启动时不立即执行测试")
    parser.add_argument("--parallel", type=int, default=3, help="每模型并行调用数 (默认: 3)")
    parser.add_argument("--prompt", default=None, help="自定义测试 prompt")

    args = parser.parse_args()

    # ── API Keys ──
    from ds_benchmark import load_api_keys, set_api_key, get_api_key, ALL_MODELS, _load_env_file

    # 先从 .env 和环境变量加载所有 Key
    load_api_keys()

    # CLI -k 参数覆盖 DS_API_KEY
    if args.api_key:
        set_api_key("DS_API_KEY", args.api_key)

    # ── 调度间隔（分钟）──
    # 优先级：CLI -i > 环境变量 INTERVAL_MINUTES > .env 文件 > 默认 10
    if args.interval is not None:
        interval = args.interval
    else:
        env_val = os.environ.get("INTERVAL_MINUTES") or _load_env_file().get("INTERVAL_MINUTES")
        try:
            interval = int(env_val) if env_val else 10
            if interval < 1:
                interval = 10
        except ValueError:
            print(f"⚠ INTERVAL_MINUTES 无效值 ({env_val!r})，使用默认 10 分钟")
            interval = 10

    # 检查哪些模型有可用的 Key
    available_models = [m for m in ALL_MODELS if get_api_key(m)]
    if not available_models:
        print("❌ 请提供至少一个 API Key:")
        print("   python run.py -k sk-xxxx                  # DeepSeek API Key")
        print("   或在 .env 文件中设置 DS_API_KEY / GLM_API_KEY")
        sys.exit(1)

    # ── Import ──
    from database import init_db, get_db_path
    from scheduler import start_scheduler
    from server import start_server

    # ── Init DB ──
    init_db()

    # ── 启动信息 ──
    prompt = args.prompt or None  # None 时 scheduler 会用 DEFAULT_PROMPT
    run_immediately = not args.no_immediate

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║         🚀 API 性能监控服务                             ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  📊 仪表盘   : http://localhost:{args.port:<24}   ║")
    print(f"║  ⏱  测试间隔 : 每 {interval} 分钟{' ' * (28 - len(str(interval)))}   ║")
    print(f"║  🔄 并行调用 : 每模型 {args.parallel} 次{' ' * (24 - len(str(args.parallel)))}   ║")
    print(f"║  💾 数据库   : {os.path.basename(get_db_path()):<30} ║")
    print(f"║  🏁 立即测试 : {'是' if run_immediately else '否':<30} ║")
    print("╠══════════════════════════════════════════════════════════╣")
    for m in ALL_MODELS:
        key = get_api_key(m)
        status = "✓ 已配置" if key else "✗ 未配置 (跳过)"
        label = f"  {m}: {status}"
        print(f"║  {label:<55}║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()
    print("  按 Ctrl+C 停止服务")
    print()

    # ── 启动调度器(后台线程) ──
    stop_event = start_scheduler(
        prompt=prompt,
        interval_minutes=interval,
        run_immediately=run_immediately,
        parallel=args.parallel,
    )

    # ── 启动 Web 服务器(阻塞主线程) ──
    # Ctrl+C 由 server.serve_forever() 内部的 try/except 处理，
    # 不在这里注册 SIGINT —— 否则强制 sys.exit() 会导致 daemon 线程
    # 持有 stdout 锁时崩溃 (_enter_buffered_busy)
    try:
        start_server(port=args.port)
    except KeyboardInterrupt:
        pass
    finally:
        print("\n⏹  正在停止服务...")
        stop_event.set()
        # 给后台线程极短时间打印尾声
        import time
        time.sleep(0.2)
        print("✅ 服务已停止")
        # 强制退出，绕过 ThreadPoolExecutor 的 atexit join
        # （否则正在跑的 HTTP 请求会让进程卡到请求结束）
        os._exit(0)



if __name__ == "__main__":
    main()
