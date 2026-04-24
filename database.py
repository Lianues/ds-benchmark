"""
SQLite 数据库操作模块 - 存储基准测试结果
"""

import os
import sqlite3
import threading
from pathlib import Path
from statistics import mean

# 数据库路径
_DB_DIR = Path(__file__).parent / "data"
_DB_PATH = _DB_DIR / "benchmark.db"

# 线程锁
_lock = threading.Lock()


def get_db_path() -> str:
    """返回数据库文件路径"""
    return str(_DB_PATH)


def _get_conn() -> sqlite3.Connection:
    """获取数据库连接（row_factory 设为 sqlite3.Row 以支持 dict 访问）"""
    conn = sqlite3.connect(str(_DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _row_to_dict(row: sqlite3.Row) -> dict:
    """将 sqlite3.Row 转换为 dict"""
    return dict(row)


def _rows_to_dicts(rows: list) -> list[dict]:
    """将 sqlite3.Row 列表转换为 dict 列表"""
    return [_row_to_dict(r) for r in rows]


def init_db():
    """初始化数据库：创建目录和表"""
    _DB_DIR.mkdir(parents=True, exist_ok=True)

    with _lock:
        conn = _get_conn()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS test_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    batch_id TEXT NOT NULL,
                    model TEXT NOT NULL,
                    run_index INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    ttft_ms REAL,
                    content_ttft_ms REAL,
                    thinking_duration_ms REAL,
                    content_streaming_ms REAL,
                    total_time_ms REAL,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    reasoning_tokens INTEGER,
                    output_tokens INTEGER,
                    total_tokens INTEGER,
                    content_tps REAL,
                    reasoning_tps REAL,
                    overall_tps REAL,
                    chunk_speed REAL,
                    success INTEGER DEFAULT 1,
                    error_msg TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_test_runs_batch_id
                    ON test_runs(batch_id);
                CREATE INDEX IF NOT EXISTS idx_test_runs_model
                    ON test_runs(model);

                CREATE TABLE IF NOT EXISTS batch_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    batch_id TEXT UNIQUE NOT NULL,
                    model TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    avg_ttft_ms REAL,
                    min_ttft_ms REAL,
                    max_ttft_ms REAL,
                    avg_content_ttft_ms REAL,
                    avg_thinking_duration_ms REAL,
                    avg_content_streaming_ms REAL,
                    avg_total_time_ms REAL,
                    avg_prompt_tokens REAL,
                    avg_completion_tokens REAL,
                    avg_reasoning_tokens REAL,
                    avg_output_tokens REAL,
                    avg_content_tps REAL,
                    min_content_tps REAL,
                    max_content_tps REAL,
                    avg_reasoning_tps REAL,
                    avg_overall_tps REAL,
                    avg_chunk_speed REAL,
                    success_count INTEGER DEFAULT 0,
                    total_count INTEGER DEFAULT 3
                );

                CREATE INDEX IF NOT EXISTS idx_batch_results_model
                    ON batch_results(model);
                CREATE INDEX IF NOT EXISTS idx_batch_results_timestamp
                    ON batch_results(timestamp);
            """)
            conn.commit()
        finally:
            conn.close()


def save_test_run(batch_id: str, model: str, run_index: int,
                  timestamp: str, metrics: dict):
    """
    保存单次测试结果到 test_runs 表。

    Args:
        batch_id: 批次 ID
        model: 模型名称
        run_index: 本批次中的运行索引 (0, 1, 2)
        timestamp: ISO 8601 时间戳
        metrics: 指标字典，包含 ttft_ms, content_tps 等字段
    """
    with _lock:
        conn = _get_conn()
        try:
            conn.execute("""
                INSERT INTO test_runs (
                    batch_id, model, run_index, timestamp,
                    ttft_ms, content_ttft_ms, thinking_duration_ms,
                    content_streaming_ms, total_time_ms,
                    prompt_tokens, completion_tokens, reasoning_tokens,
                    output_tokens, total_tokens,
                    content_tps, reasoning_tps, overall_tps, chunk_speed,
                    success, error_msg
                ) VALUES (
                    ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?,
                    ?, ?, ?,
                    ?, ?,
                    ?, ?, ?, ?,
                    ?, ?
                )
            """, (
                batch_id, model, run_index, timestamp,
                metrics.get("ttft_ms"),
                metrics.get("content_ttft_ms"),
                metrics.get("thinking_duration_ms"),
                metrics.get("content_streaming_ms"),
                metrics.get("total_time_ms"),
                metrics.get("prompt_tokens"),
                metrics.get("completion_tokens"),
                metrics.get("reasoning_tokens"),
                metrics.get("output_tokens"),
                metrics.get("total_tokens"),
                metrics.get("content_tps"),
                metrics.get("reasoning_tps"),
                metrics.get("overall_tps"),
                metrics.get("chunk_speed"),
                1 if metrics.get("success", True) else 0,
                metrics.get("error_msg"),
            ))
            conn.commit()
        finally:
            conn.close()


def _safe_values(runs: list[dict], key: str) -> list[float]:
    """从 runs 中提取指定 key 的非 None 数值列表"""
    return [r[key] for r in runs if r.get(key) is not None]


def _safe_avg(values: list[float]) -> float | None:
    return mean(values) if values else None


def _safe_min(values: list[float]) -> float | None:
    return min(values) if values else None


def _safe_max(values: list[float]) -> float | None:
    return max(values) if values else None


def save_batch_result(batch_id: str, model: str, timestamp: str,
                      runs: list[dict]):
    """
    从多次测试结果计算平均值并保存到 batch_results 表。

    Args:
        batch_id: 批次 ID
        model: 模型名称
        timestamp: ISO 8601 时间戳
        runs: 单次测试结果的 dict 列表
    """
    if not runs:
        # 空列表：记录一条全空的批次结果
        with _lock:
            conn = _get_conn()
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO batch_results (
                        batch_id, model, timestamp,
                        avg_ttft_ms, min_ttft_ms, max_ttft_ms,
                        avg_content_ttft_ms, avg_thinking_duration_ms,
                        avg_content_streaming_ms, avg_total_time_ms,
                        avg_prompt_tokens, avg_completion_tokens,
                        avg_reasoning_tokens, avg_output_tokens,
                        avg_content_tps, min_content_tps, max_content_tps,
                        avg_reasoning_tps, avg_overall_tps, avg_chunk_speed,
                        success_count, total_count
                    ) VALUES (
                        ?, ?, ?,
                        NULL, NULL, NULL,
                        NULL, NULL,
                        NULL, NULL,
                        NULL, NULL,
                        NULL, NULL,
                        NULL, NULL, NULL,
                        NULL, NULL, NULL,
                        0, 0
                    )
                """, (batch_id, model, timestamp))
                conn.commit()
            finally:
                conn.close()
        return

    # 只用成功的 run 来计算统计值
    successful = [r for r in runs if r.get("success", True)]

    ttft_vals = _safe_values(successful, "ttft_ms")
    content_tps_vals = _safe_values(successful, "content_tps")

    with _lock:
        conn = _get_conn()
        try:
            conn.execute("""
                INSERT OR REPLACE INTO batch_results (
                    batch_id, model, timestamp,
                    avg_ttft_ms, min_ttft_ms, max_ttft_ms,
                    avg_content_ttft_ms, avg_thinking_duration_ms,
                    avg_content_streaming_ms, avg_total_time_ms,
                    avg_prompt_tokens, avg_completion_tokens,
                    avg_reasoning_tokens, avg_output_tokens,
                    avg_content_tps, min_content_tps, max_content_tps,
                    avg_reasoning_tps, avg_overall_tps, avg_chunk_speed,
                    success_count, total_count
                ) VALUES (
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?,
                    ?, ?,
                    ?, ?,
                    ?, ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?
                )
            """, (
                batch_id, model, timestamp,
                _safe_avg(ttft_vals),
                _safe_min(ttft_vals),
                _safe_max(ttft_vals),
                _safe_avg(_safe_values(successful, "content_ttft_ms")),
                _safe_avg(_safe_values(successful, "thinking_duration_ms")),
                _safe_avg(_safe_values(successful, "content_streaming_ms")),
                _safe_avg(_safe_values(successful, "total_time_ms")),
                _safe_avg(_safe_values(successful, "prompt_tokens")),
                _safe_avg(_safe_values(successful, "completion_tokens")),
                _safe_avg(_safe_values(successful, "reasoning_tokens")),
                _safe_avg(_safe_values(successful, "output_tokens")),
                _safe_avg(content_tps_vals),
                _safe_min(content_tps_vals),
                _safe_max(content_tps_vals),
                _safe_avg(_safe_values(successful, "reasoning_tps")),
                _safe_avg(_safe_values(successful, "overall_tps")),
                _safe_avg(_safe_values(successful, "chunk_speed")),
                len(successful),
                len(runs),
            ))
            conn.commit()
        finally:
            conn.close()


def get_batch_results(model: str | None = None, limit: int = 200,
                      offset: int = 0) -> list[dict]:
    """
    查询批次结果，支持按模型筛选。

    Args:
        model: 模型名称（None 则查询全部）
        limit: 返回数量上限
        offset: 偏移量

    Returns:
        dict 列表，按时间倒序
    """
    with _lock:
        conn = _get_conn()
        try:
            if model:
                rows = conn.execute(
                    "SELECT * FROM batch_results WHERE model = ? "
                    "ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                    (model, limit, offset)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM batch_results "
                    "ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                    (limit, offset)
                ).fetchall()
            return _rows_to_dicts(rows)
        finally:
            conn.close()


def get_latest_batch(model: str) -> dict | None:
    """
    获取指定模型最新的一批结果。

    Args:
        model: 模型名称

    Returns:
        最新批次结果 dict，无结果返回 None
    """
    with _lock:
        conn = _get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM batch_results WHERE model = ? "
                "ORDER BY timestamp DESC LIMIT 1",
                (model,)
            ).fetchone()
            return _row_to_dict(row) if row else None
        finally:
            conn.close()


def count_batch_results(model: str | None = None) -> int:
    """
    统计 batch_results 表的总条数，支持按模型筛选。

    Args:
        model: 模型名称（None 则统计全部）

    Returns:
        记录总数
    """
    with _lock:
        conn = _get_conn()
        try:
            if model:
                row = conn.execute(
                    "SELECT COUNT(*) FROM batch_results WHERE model = ?",
                    (model,)
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(*) FROM batch_results"
                ).fetchone()
            return int(row[0]) if row else 0
        finally:
            conn.close()


def get_test_runs(batch_id: str) -> list[dict]:
    """
    获取批次下的所有单次测试结果。

    Args:
        batch_id: 批次 ID

    Returns:
        dict 列表，按 run_index 排序
    """
    with _lock:
        conn = _get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM test_runs WHERE batch_id = ? "
                "ORDER BY run_index",
                (batch_id,)
            ).fetchall()
            return _rows_to_dicts(rows)
        finally:
            conn.close()


def get_all_latest() -> dict[str, dict]:
    """
    获取所有模型的最新批次结果。

    Returns:
        dict，key 为模型名称，value 为该模型最新的 batch_result dict
    """
    with _lock:
        conn = _get_conn()
        try:
            # 使用子查询取每个模型 timestamp 最大的记录
            rows = conn.execute("""
                SELECT b.* FROM batch_results b
                INNER JOIN (
                    SELECT model, MAX(timestamp) AS max_ts
                    FROM batch_results
                    GROUP BY model
                ) latest ON b.model = latest.model
                        AND b.timestamp = latest.max_ts
            """).fetchall()
            return {row["model"]: _row_to_dict(row) for row in rows}
        finally:
            conn.close()


def get_stats(model: str, hours: int = 24) -> dict | None:
    """
    获取过去 N 小时的统计摘要。

    Args:
        model: 模型名称
        hours: 回溯小时数（默认 24）

    Returns:
        统计摘要 dict，包含 avg/min/max TPS 和 TTFT；无数据返回 None
    """
    with _lock:
        conn = _get_conn()
        try:
            row = conn.execute("""
                SELECT
                    COUNT(*) AS batch_count,
                    COUNT(*) AS count,
                    AVG(avg_content_tps) AS avg_content_tps,
                    MIN(min_content_tps) AS min_content_tps,
                    MAX(max_content_tps) AS max_content_tps,
                    AVG(avg_reasoning_tps) AS avg_reasoning_tps,
                    MIN(avg_reasoning_tps) AS min_reasoning_tps,
                    MAX(avg_reasoning_tps) AS max_reasoning_tps,
                    AVG(avg_overall_tps) AS avg_overall_tps,
                    MIN(avg_overall_tps) AS min_overall_tps,
                    MAX(avg_overall_tps) AS max_overall_tps,
                    AVG(avg_ttft_ms) AS avg_ttft_ms,
                    MIN(min_ttft_ms) AS min_ttft_ms,
                    MAX(max_ttft_ms) AS max_ttft_ms,
                    AVG(avg_content_ttft_ms) AS avg_content_ttft_ms,
                    AVG(avg_thinking_duration_ms) AS avg_thinking_duration_ms,
                    MIN(avg_thinking_duration_ms) AS min_thinking_duration_ms,
                    MAX(avg_thinking_duration_ms) AS max_thinking_duration_ms,
                    AVG(avg_total_time_ms) AS avg_total_time_ms,
                    AVG(avg_chunk_speed) AS avg_chunk_speed,
                    AVG(avg_completion_tokens) AS avg_completion_tokens,
                    AVG(avg_output_tokens) AS avg_output_tokens,
                    AVG(avg_reasoning_tokens) AS avg_reasoning_tokens,
                    SUM(success_count) AS total_success,
                    SUM(total_count) AS total_runs
                FROM batch_results
                WHERE model = ?
                  AND timestamp >= datetime('now', ? || ' hours')
            """, (model, -abs(hours))).fetchone()

            if not row or row["batch_count"] == 0:
                return None

            return _row_to_dict(row)
        finally:
            conn.close()
