#!/usr/bin/env bash
# ═══════════════════════════════════════════
#  DeepSeek API 性能监控 — tmux 管理脚本
#  用法: ./run.sh [start|stop|restart|log|status]
# ═══════════════════════════════════════════

SESSION="ds-benchmark"
DIR="$(cd "$(dirname "$0")" && pwd)"
PORT=8080

start() {
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "⚠️  已在运行中（session: $SESSION）"
    echo "   查看日志: $0 log"
    echo "   重启服务: $0 restart"
    return 1
  fi

  # 检查 .env 是否存在
  if [ ! -f "$DIR/.env" ]; then
    echo "❌ 未找到 .env 文件"
    echo "   请创建 $DIR/.env 并写入: DS_API_KEY=sk-xxxx"
    echo "   可参考 .env.example"
    return 1
  fi

  # 检查 python3 是否可用
  if ! command -v python3 >/dev/null 2>&1; then
    echo "❌ 未找到 python3 命令"
    return 1
  fi

  tmux new-session -d -s "$SESSION" -c "$DIR" "python3 run.py"
  sleep 1

  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "✅ 已启动（session: $SESSION）"
    echo "   📊 仪表盘: http://localhost:$PORT"
    echo "   📋 查看日志: $0 log"
    echo "   ⛔ 停止服务: $0 stop"
  else
    echo "❌ 启动失败，请检查依赖是否安装："
    echo "   pip3 install -r requirements.txt"
  fi
}

stop() {
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    # 发送 Ctrl+C 让 Python 优雅退出
    tmux send-keys -t "$SESSION" C-c
    sleep 1
    tmux kill-session -t "$SESSION" 2>/dev/null
    echo "⛔ 已停止"
  else
    echo "⚠️  未在运行"
  fi
}

status() {
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "✅ 运行中（session: $SESSION）"
    echo "   📊 仪表盘: http://localhost:$PORT"
  else
    echo "⏹  未运行"
  fi
}

case "${1:-start}" in
  start)   start ;;
  stop)    stop ;;
  restart) stop; sleep 1; start ;;
  status)  status ;;
  log)
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      tmux attach -t "$SESSION"
    else
      echo "⚠️  未在运行，先执行: $0 start"
    fi
    ;;
  *)
    echo "用法: $0 {start|stop|restart|status|log}"
    echo ""
    echo "  start    — 启动服务"
    echo "  stop     — 停止服务"
    echo "  restart  — 重启服务"
    echo "  status   — 查看状态"
    echo "  log      — 查看实时日志（Ctrl+B 然后 D 退出）"
    exit 1
    ;;
esac
