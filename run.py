#!/usr/bin/env python3
"""
DeepSeek API 性能监控服务
========================
启动后会:
  1. 在后台定时(每10分钟)对 deepseek-chat 和 deepseek-reasoner 各并行调用 3 次
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
        description="DeepSeek API 性能监控服务",
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
    parser.add_argument("-i", "--interval", type=int, default=10, help="测试间隔(分钟) (默认: 10)")
    parser.add_argument("--no-immediate", action="store_true", help="启动时不立即执行测试")
    parser.add_argument("--parallel", type=int, default=3, help="每模型并行调用数 (默认: 3)")
    parser.add_argument("--prompt", default=None, help="自定义测试 prompt")

    args = parser.parse_args()

    # ── API Key ──
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
        print("   python run.py -k sk-xxxx")
        print("   或设置环境变量 DS_API_KEY=sk-xxxx")
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
    print("║         🚀 DeepSeek API 性能监控服务                   ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  📊 仪表盘   : http://localhost:{args.port:<24}   ║")
    print(f"║  ⏱  测试间隔 : 每 {args.interval} 分钟{' ' * (28 - len(str(args.interval)))}   ║")
    print(f"║  🔄 并行调用 : 每模型 {args.parallel} 次{' ' * (24 - len(str(args.parallel)))}   ║")
    print(f"║  💾 数据库   : {os.path.basename(get_db_path()):<30} ║")
    print(f"║  🏁 立即测试 : {'是' if run_immediately else '否':<30} ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()
    print("  按 Ctrl+C 停止服务")
    print()

    # ── 启动调度器(后台线程) ──
    stop_event = start_scheduler(
        api_key=api_key,
        prompt=prompt,
        interval_minutes=args.interval,
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
