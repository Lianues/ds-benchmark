"""轻量 Web 服务器 - 基于 Python 标准库 http.server，零额外依赖"""

import json
import os
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse, parse_qs

from database import (
    get_batch_results, get_latest_batch, get_test_runs,
    get_all_latest, get_stats, count_batch_results,
)
from ds_benchmark import ALL_MODELS
from scheduler import scheduler_status

# server.py 所在目录，用于定位 static 文件
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class APIHandler(BaseHTTPRequestHandler):
    """处理所有 HTTP 请求的核心 Handler"""

    # ── 响应工具方法 ──────────────────────────────────────────────

    def _send_json(self, data, status=200):
        """发送 JSON 响应"""
        body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, content):
        """发送 HTML 响应"""
        body = content.encode("utf-8") if isinstance(content, str) else content
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_404(self, message="Not Found"):
        """发送 404 JSON 错误响应"""
        self._send_json({"success": False, "error": message}, status=404)

    def _ok(self, data):
        """发送统一成功响应"""
        self._send_json({"success": True, "data": data})

    # ── 请求分发 ──────────────────────────────────────────────────

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        params = parse_qs(parsed.query)

        routes = {
            "/":                self._handle_index,
            "/api/status":      self._handle_status,
            "/api/latest":      self._handle_latest,
            "/api/batches":     self._handle_batches,
            "/api/runs":        self._handle_runs,
            "/api/stats":       self._handle_stats,
            "/api/chart-data":  self._handle_chart_data,
        }

        handler = routes.get(path)
        if handler:
            try:
                handler(params)
            except Exception as e:
                print(f"[server] ERROR {path}: {e}")
                self._send_json({"success": False, "error": str(e)}, status=500)
        else:
            self._send_404(f"No route: {path}")

    # ── 路由实现 ──────────────────────────────────────────────────

    def _handle_index(self, params):
        """GET / → 返回 static/index.html"""
        filepath = os.path.join(BASE_DIR, "static", "index.html")
        if not os.path.isfile(filepath):
            self._send_404("static/index.html not found")
            return
        with open(filepath, "r", encoding="utf-8") as f:
            self._send_html(f.read())

    def _handle_status(self, params):
        """GET /api/status → 调度器状态"""
        self._ok(scheduler_status)

    def _handle_latest(self, params):
        """GET /api/latest → 所有模型最新批次结果"""
        self._ok(get_all_latest())

    def _handle_batches(self, params):
        """GET /api/batches?model=xxx&limit=50&offset=0 → 批次历史 (含总数)"""
        model = params.get("model", [None])[0]
        limit = int(params.get("limit", [50])[0])
        offset = max(0, int(params.get("offset", [0])[0]))
        items = get_batch_results(model=model, limit=limit, offset=offset)
        total = count_batch_results(model=model)
        self._ok({"items": items, "total": total, "limit": limit, "offset": offset})

    def _handle_runs(self, params):
        """GET /api/runs?batch_id=xxx → 某批次测试详情"""
        batch_id = params.get("batch_id", [None])[0]
        if not batch_id:
            self._send_json({"success": False, "error": "batch_id is required"}, status=400)
            return
        self._ok(get_test_runs(batch_id))

    def _handle_stats(self, params):
        """GET /api/stats?model=xxx&hours=24 → 统计摘要"""
        model = params.get("model", [None])[0]
        hours = int(params.get("hours", [24])[0])
        self._ok(get_stats(model=model, hours=hours))

    def _handle_chart_data(self, params):
        """GET /api/chart-data?hours=24 → 图表时序数据"""
        hours = int(params.get("hours", [24])[0])

        models = ALL_MODELS
        chart = {}

        for model in models:
            batches = get_batch_results(model=model, limit=9999)
            # 按时间正序（数据库通常返回倒序，翻转一下）
            if batches and isinstance(batches, list):
                batches = list(reversed(batches))

            # 过滤时间范围：用 hours 参数
            from datetime import datetime, timedelta
            # hours=0 表示不过滤，返回全部
            cutoff = datetime.now() - timedelta(hours=hours) if hours > 0 else datetime.min

            series = {
                "timestamps": [],
                "ttft_ms": [],
                "content_tps": [],
                "overall_tps": [],
                "total_time_ms": [],
                "content_streaming_ms": [],
                "completion_tokens": [],
                "output_tokens": [],
                "success_rate": [],
                "success_count": [],
                "total_count": [],
            }
            # 所有模型都启用思考模式，统一包含思维链字段
            series["thinking_duration_ms"] = []
            series["reasoning_tps"] = []
            series["reasoning_tokens"] = []

            for b in (batches or []):
                # 解析时间戳
                ts_raw = b.get("timestamp") or b.get("created_at") or ""
                try:
                    ts = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00").replace("+00:00", ""))
                except Exception:
                    try:
                        ts = datetime.strptime(str(ts_raw)[:19], "%Y-%m-%d %H:%M:%S")
                    except Exception:
                        continue

                if ts < cutoff:
                    continue

                # 输出 UTC ISO（含秒、带 Z 后缀）。
                # 与卡片端时间戳的语义保持一致（数据库里存的是服务器 UTC 时间），
                # 前端 new Date(s) 会按 UTC 解析后自动转成浏览器本地时区显示。
                ts_label = ts.strftime("%Y-%m-%dT%H:%M:%SZ")
                series["timestamps"].append(ts_label)
                series["ttft_ms"].append(b.get("ttft_ms") or b.get("avg_ttft_ms"))
                series["content_tps"].append(b.get("content_tps") or b.get("avg_content_tps"))
                series["overall_tps"].append(b.get("overall_tps") or b.get("avg_overall_tps"))
                series["total_time_ms"].append(b.get("total_time_ms") or b.get("avg_total_time_ms"))
                series["content_streaming_ms"].append(b.get("content_streaming_ms") or b.get("avg_content_streaming_ms"))
                series["completion_tokens"].append(b.get("completion_tokens") or b.get("avg_completion_tokens"))
                series["output_tokens"].append(b.get("output_tokens") or b.get("avg_output_tokens"))

                # 成功率（百分比）
                sc = b.get("success_count")
                tc = b.get("total_count")
                if sc is not None and tc:
                    rate = round(sc / tc * 100, 2)
                else:
                    rate = None
                series["success_rate"].append(rate)
                series["success_count"].append(sc)
                series["total_count"].append(tc)

                series["thinking_duration_ms"].append(b.get("thinking_duration_ms") or b.get("avg_thinking_duration_ms"))
                series["reasoning_tps"].append(b.get("reasoning_tps") or b.get("avg_reasoning_tps"))
                series["reasoning_tokens"].append(b.get("reasoning_tokens") or b.get("avg_reasoning_tokens"))

            chart[model] = series

        self._ok(chart)

    # ── 日志格式简化 ──────────────────────────────────────────────

    def log_message(self, format, *args):
        # 只打印错误响应（4xx/5xx），2xx 静默
        # args 一般是 (request_line, status_code, size)
        try:
            status = int(args[1])
            if status < 400:
                return
        except (IndexError, ValueError):
            pass
        print(f"[server] {self.address_string()} {format % args}")


# ── 启动函数 ──────────────────────────────────────────────────────

class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """多线程 HTTP 服务器，防止单个请求阻塞"""
    daemon_threads = True


def start_server(port=8080):
    """启动 HTTP 服务器（阻塞调用）"""
    server = ThreadingHTTPServer(("0.0.0.0", port), APIHandler)
    print(f"[server] Listening on http://0.0.0.0:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()


def start_server_thread(port=8080):
    """在后台守护线程启动服务器"""
    t = threading.Thread(target=start_server, args=(port,), daemon=True)
    t.start()
    print(f"[server] Background thread started on port {port}")
    return t


if __name__ == "__main__":
    start_server()
