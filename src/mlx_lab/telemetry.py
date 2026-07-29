from __future__ import annotations

import json
import math
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

from .user_store import user_paths


DEFAULT_MAX_HISTORY = 5_000
DEFAULT_RETENTION_DAYS = 30
SECONDS_PER_DAY = 86_400
AGENT_LOOP_MIN_TOOL_STEPS = 8
AGENT_LOOP_HIGH_TOOL_STEPS = 30
AGENT_LOOP_SEGMENT_GAP_SECONDS = 1_800
AGENT_LOOP_CACHE_WINDOW = 4
AGENT_LOOP_LARGE_CONTEXT_TOKENS = 32_768


def configured_telemetry_path(logs_dir: Path | None = None) -> Path:
    configured = os.environ.get("MLX_SERVER_TELEMETRY_PATH", "")
    if configured:
        return Path(configured).expanduser()
    return (logs_dir or user_paths().logs) / "mlx-server-telemetry.jsonl"


def configured_telemetry_retention_days() -> int:
    try:
        return max(
            1,
            min(
                int(
                    os.environ.get(
                        "MLX_SERVER_TELEMETRY_RETENTION_DAYS",
                        str(DEFAULT_RETENTION_DAYS),
                    )
                ),
                365,
            ),
        )
    except ValueError:
        return DEFAULT_RETENTION_DAYS


def telemetry_archive_directory(path: Path) -> Path:
    return path.parent / f"{path.stem}-archive"


def _telemetry_paths(path: Path) -> list[Path]:
    archive_dir = telemetry_archive_directory(path)
    archives = sorted(archive_dir.glob("*.jsonl")) if archive_dir.is_dir() else []
    return [*archives, path]


def load_telemetry_history(
    path: Path,
    *,
    max_history: int = DEFAULT_MAX_HISTORY,
    retention_days: int | None = None,
    now: float | None = None,
) -> list[dict[str, Any]]:
    retention = max(
        1,
        int(retention_days or configured_telemetry_retention_days()),
    )
    cutoff = float(now if now is not None else time.time()) - (
        retention * SECONDS_PER_DAY
    )
    records: dict[str, dict[str, Any]] = {}
    unkeyed_index = 0
    for telemetry_path in _telemetry_paths(path):
        try:
            with telemetry_path.open("r", encoding="utf-8") as file:
                for line in file:
                    try:
                        item = json.loads(line)
                    except (json.JSONDecodeError, TypeError, ValueError):
                        continue
                    if not isinstance(item, dict):
                        continue
                    timestamp = _record_timestamp(item)
                    if timestamp < cutoff:
                        continue
                    record_id = str(item.get("id") or "").strip()
                    if record_id:
                        key = f"id:{record_id}"
                    else:
                        key = f"unkeyed:{unkeyed_index}"
                        unkeyed_index += 1
                    records[key] = item
        except OSError:
            continue
    history = sorted(records.values(), key=_record_timestamp)
    return history[-max(1, int(max_history)) :]


def mcp_server_for_tool(name: Any) -> str:
    tool_name = str(name or "").strip()
    lower_name = tool_name.casefold()
    if lower_name.startswith("mcp__"):
        parts = tool_name.split("__")
        return parts[1] if len(parts) > 2 and parts[1] else "mcp"
    if lower_name.startswith("mcp_") or lower_name.startswith("mcp-"):
        return "mcp"
    return ""


def _number(record: dict[str, Any], name: str) -> int:
    try:
        return max(0, int(record.get(name) or 0))
    except (TypeError, ValueError):
        return 0


def _duration(record: dict[str, Any], now: float) -> float:
    completed_at = float(record.get("completed_at") or now)
    started_at = float(record.get("started_at") or completed_at)
    return max(0.0, completed_at - started_at)


def _record_timestamp(record: dict[str, Any]) -> float:
    try:
        return float(record.get("completed_at") or record.get("started_at") or 0)
    except (TypeError, ValueError):
        return 0.0


def _context_tokens(record: dict[str, Any]) -> int:
    return _number(record, "prompt_tokens") + _number(
        record,
        "cached_prompt_tokens",
    )


def _cache_hit_percent(record: dict[str, Any]) -> float:
    context_tokens = _context_tokens(record)
    if context_tokens <= 0:
        return 0.0
    return round(
        (_number(record, "cached_prompt_tokens") / context_tokens) * 100,
        1,
    )


def _telemetry_failure_kind(record: dict[str, Any]) -> tuple[str, str]:
    status = str(record.get("status") or "").strip().casefold()
    if status == "cancelled":
        return ("cancelled", "Cancelled")

    error = str(record.get("error") or "").strip().casefold()
    failure_patterns = (
        (
            "client_disconnected",
            "Client disconnected",
            (
                "broken pipe",
                "client disconnected",
                "connection aborted",
                "connection reset",
            ),
        ),
        (
            "safety_admission",
            "Safety admission",
            (
                "safety admission rejected",
                "effective context cap",
                "exceeds the server cap",
                "available after the",
            ),
        ),
        (
            "memory",
            "Memory",
            (
                "out of memory",
                "memory pressure",
                "memory guard",
                "metal memory",
            ),
        ),
        (
            "timeout",
            "Timeout",
            ("timed out", "timeout"),
        ),
        (
            "authentication",
            "Authentication",
            (
                "api key",
                "authentication",
                "unauthorized",
                "forbidden",
            ),
        ),
        (
            "model",
            "Model",
            (
                "failed to load model",
                "model not found",
                "no model loaded",
                "requested model",
            ),
        ),
    )
    for code, label, patterns in failure_patterns:
        if any(pattern in error for pattern in patterns):
            return (code, label)
    return ("request_error", "Request error")


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2


def _agent_group_key(record: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(record.get("client") or "Unknown client").strip(),
        str(record.get("client_address") or "").strip(),
        str(record.get("model") or "unknown").strip(),
        str(record.get("query_preview") or "").strip(),
    )


def _is_agent_request(record: dict[str, Any]) -> bool:
    return bool(
        record.get("offered_tools")
        or _number(record, "offered_tool_count")
        or _number(record, "tool_call_count")
    )


def _starts_new_agent_segment(
    previous: dict[str, Any],
    current: dict[str, Any],
) -> bool:
    gap = _record_timestamp(current) - _record_timestamp(previous)
    if gap > AGENT_LOOP_SEGMENT_GAP_SECONDS:
        return True

    previous_messages = _number(previous, "message_count")
    current_messages = _number(current, "message_count")
    if (
        previous_messages
        and current_messages
        and current_messages <= previous_messages
    ):
        return True

    previous_context = _context_tokens(previous)
    current_context = _context_tokens(current)
    return (
        previous_context >= 4_096
        and current_context
        and current_context < previous_context * 0.6
    )


def detect_agent_loop_risks(
    records: list[dict[str, Any]],
    *,
    active: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return advisory agent-loop risks derived from privacy-bounded telemetry."""
    segments: dict[
        tuple[str, str, str, str],
        list[list[dict[str, Any]]],
    ] = {}
    for record in sorted(records, key=_record_timestamp):
        if not _is_agent_request(record):
            continue
        key = _agent_group_key(record)
        grouped = segments.setdefault(key, [])
        if not grouped or _starts_new_agent_segment(grouped[-1][-1], record):
            grouped.append([])
        grouped[-1].append(record)

    active_keys = {
        _agent_group_key(record)
        for record in (active or [])
        if _is_agent_request(record)
    }
    risks: list[dict[str, Any]] = []
    for key, grouped in segments.items():
        segment = grouped[-1]
        trailing_tool_steps: list[dict[str, Any]] = []
        for record in reversed(segment):
            if (
                record.get("status") != "completed"
                or _number(record, "tool_call_count") <= 0
            ):
                break
            trailing_tool_steps.append(record)
        trailing_tool_steps.reverse()
        step_count = len(trailing_tool_steps)
        if step_count < AGENT_LOOP_MIN_TOOL_STEPS:
            continue

        if step_count >= AGENT_LOOP_HIGH_TOOL_STEPS:
            score = 70
        elif step_count >= 20:
            score = 55
        elif step_count >= 12:
            score = 40
        else:
            score = 25

        latest = trailing_tool_steps[-1]
        latest_context = _context_tokens(latest)
        tool_names = [
            str(call.get("name") or "unknown")
            for record in trailing_tool_steps
            for call in record.get("tool_calls", [])
            if isinstance(call, dict)
        ]
        tool_counts = {
            name: tool_names.count(name)
            for name in set(tool_names)
        }
        dominant_tool = ""
        dominant_tool_count = 0
        if tool_counts:
            dominant_tool, dominant_tool_count = max(
                tool_counts.items(),
                key=lambda item: (item[1], item[0]),
            )
        dominant_share = (
            dominant_tool_count / max(1, len(tool_names))
        )

        signals = [
            {
                "code": "sustained_tool_chain",
                "message": (
                    f"{step_count} consecutive model responses ended with "
                    "structured tool calls."
                ),
            }
        ]
        if dominant_tool_count >= 6 and dominant_share >= 0.75:
            score += 15
            signals.append(
                {
                    "code": "repeated_tool_pattern",
                    "message": (
                        f"{dominant_tool} accounts for "
                        f"{dominant_share * 100:.0f}% of recent tool calls."
                    ),
                }
            )

        recent = trailing_tool_steps[-AGENT_LOOP_CACHE_WINDOW:]
        prior = trailing_tool_steps[:-AGENT_LOOP_CACHE_WINDOW]
        recent_cache_hits = [_cache_hit_percent(record) for record in recent]
        prior_cache_hits = [_cache_hit_percent(record) for record in prior]
        cache_thrashing = (
            len(recent) == AGENT_LOOP_CACHE_WINDOW
            and latest_context >= AGENT_LOOP_LARGE_CONTEXT_TOKENS
            and max(recent_cache_hits, default=0.0) <= 5.0
            and max(prior_cache_hits, default=0.0) >= 25.0
        )
        if cache_thrashing:
            score += 20
            signals.append(
                {
                    "code": "prompt_cache_thrashing",
                    "message": (
                        f"Prompt-cache reuse fell to "
                        f"{sum(recent_cache_hits) / len(recent_cache_hits):.1f}% "
                        f"across the last {len(recent)} steps at "
                        f"{latest_context:,} context tokens."
                    ),
                }
            )

        recent_latency = (
            sum(_duration(record, _record_timestamp(record)) for record in recent)
            / len(recent)
        )
        baseline_latency = _median(
            [
                _duration(record, _record_timestamp(record))
                for record in prior
                if _duration(record, _record_timestamp(record)) > 0
            ]
        )
        latency_degraded = (
            len(recent) == AGENT_LOOP_CACHE_WINDOW
            and recent_latency >= 60
            and baseline_latency > 0
            and recent_latency >= baseline_latency * 3
        )
        if latency_degraded:
            score += 10
            signals.append(
                {
                    "code": "latency_degradation",
                    "message": (
                        f"Recent steps average {recent_latency:.0f}s versus "
                        f"a {baseline_latency:.0f}s earlier median."
                    ),
                }
            )

        if latest_context >= AGENT_LOOP_LARGE_CONTEXT_TOKENS:
            signals.append(
                {
                    "code": "large_agent_context",
                    "message": (
                        f"The latest agent context is approximately "
                        f"{latest_context:,} tokens."
                    ),
                }
            )

        score = min(100, score)
        level = "high" if score >= 70 else "elevated" if score >= 40 else "watch"
        recommendation = (
            "Review or stop the client, then compact or start a fresh agent "
            "session; the current context is no longer benefiting from prompt "
            "caching."
            if cache_thrashing
            else
            "Review the client before allowing more tool actions. Consider a "
            "step budget or ask the agent to summarize and finish."
        )
        risks.append(
            {
                "level": level,
                "score": score,
                "active": key in active_keys,
                "client": key[0],
                "client_address": key[1],
                "model": key[2],
                "query_preview": key[3],
                "first_seen_at": _record_timestamp(trailing_tool_steps[0]),
                "last_seen_at": _record_timestamp(latest),
                "consecutive_tool_steps": step_count,
                "message_count": _number(latest, "message_count"),
                "context_tokens": latest_context,
                "cache_hit_percent": round(
                    sum(recent_cache_hits) / len(recent_cache_hits),
                    1,
                ),
                "average_recent_latency_seconds": round(recent_latency, 3),
                "dominant_tool": dominant_tool,
                "dominant_tool_percent": round(dominant_share * 100, 1),
                "signals": signals,
                "recommendation": recommendation,
            }
        )

    level_order = {"high": 3, "elevated": 2, "watch": 1}
    risks.sort(
        key=lambda item: (
            level_order.get(str(item["level"]), 0),
            int(item["score"]),
            float(item["last_seen_at"]),
        ),
        reverse=True,
    )
    return risks


def aggregate_telemetry(
    history: list[dict[str, Any]],
    *,
    active: list[dict[str, Any]] | None = None,
    window_seconds: int = 86_400,
    limit: int = 100,
    request_page: int = 1,
    request_page_size: int | None = None,
    retention_days: int | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    generated_at = float(now or time.time())
    retained_days = max(
        1,
        int(retention_days or configured_telemetry_retention_days()),
    )
    since = generated_at - window_seconds if window_seconds > 0 else 0
    records = [
        dict(record)
        for record in history
        if float(record.get("completed_at") or record.get("started_at") or 0) >= since
    ]
    records.sort(
        key=lambda record: float(
            record.get("completed_at") or record.get("started_at") or 0
        ),
        reverse=True,
    )
    active_records = [dict(record) for record in (active or [])]
    for record in active_records:
        record["duration_seconds"] = round(_duration(record, generated_at), 3)

    prompt_tokens = sum(_number(record, "prompt_tokens") for record in records)
    output_tokens = sum(_number(record, "output_tokens") for record in records)
    cached_tokens = sum(_number(record, "cached_prompt_tokens") for record in records)
    tool_calls = sum(_number(record, "tool_call_count") for record in records)
    mcp_tool_calls = sum(_number(record, "mcp_tool_call_count") for record in records)
    durations = [_duration(record, generated_at) for record in records]
    completed = sum(record.get("status") == "completed" for record in records)
    error_records: list[dict[str, Any]] = []
    failure_types: dict[str, dict[str, Any]] = {}
    for record in records:
        if record.get("status") == "completed":
            continue
        failure_code, failure_type = _telemetry_failure_kind(record)
        error_record = dict(record)
        error_record["failure_code"] = failure_code
        error_record["failure_type"] = failure_type
        error_records.append(error_record)
        failure = failure_types.setdefault(
            failure_code,
            {
                "code": failure_code,
                "label": failure_type,
                "count": 0,
            },
        )
        failure["count"] += 1
    errors = len(error_records)
    token_total = prompt_tokens + output_tokens
    agent_loop_risks = detect_agent_loop_risks(
        records,
        active=active_records,
    )

    models: dict[str, dict[str, Any]] = {}
    tools: dict[str, dict[str, Any]] = {}
    sources: dict[str, dict[str, Any]] = {}

    for record in records:
        model_name = str(record.get("model") or "unknown")
        model = models.setdefault(
            model_name,
            {
                "model": model_name,
                "requests": 0,
                "prompt_tokens": 0,
                "output_tokens": 0,
                "cached_prompt_tokens": 0,
                "tool_calls": 0,
                "errors": 0,
                "duration_seconds": 0.0,
                "last_active_at": 0.0,
            },
        )
        model["requests"] += 1
        model["prompt_tokens"] += _number(record, "prompt_tokens")
        model["output_tokens"] += _number(record, "output_tokens")
        model["cached_prompt_tokens"] += _number(record, "cached_prompt_tokens")
        model["tool_calls"] += _number(record, "tool_call_count")
        model["errors"] += record.get("status") != "completed"
        model["duration_seconds"] += _duration(record, generated_at)
        model["last_active_at"] = max(
            model["last_active_at"],
            float(record.get("completed_at") or record.get("started_at") or 0),
        )

        source_name = str(record.get("client") or "Unknown client")
        source = sources.setdefault(
            source_name,
            {
                "client": source_name,
                "requests": 0,
                "prompt_tokens": 0,
                "output_tokens": 0,
                "tool_calls": 0,
                "errors": 0,
            },
        )
        source["requests"] += 1
        source["prompt_tokens"] += _number(record, "prompt_tokens")
        source["output_tokens"] += _number(record, "output_tokens")
        source["tool_calls"] += _number(record, "tool_call_count")
        source["errors"] += record.get("status") != "completed"

        offered_names = {
            str(name)
            for name in record.get("offered_tools", [])
            if str(name or "").strip()
        }
        called_names = [
            str(call.get("name") or "unknown")
            for call in record.get("tool_calls", [])
            if isinstance(call, dict)
        ]
        for name in offered_names | set(called_names):
            server = mcp_server_for_tool(name)
            tool = tools.setdefault(
                name,
                {
                    "name": name,
                    "offered_in_requests": 0,
                    "invocations": 0,
                    "mcp_server": server,
                },
            )
            tool["offered_in_requests"] += name in offered_names
            tool["invocations"] += called_names.count(name)

    model_rows = []
    for model in models.values():
        request_count = max(1, int(model["requests"]))
        model["average_latency_seconds"] = round(
            model.pop("duration_seconds") / request_count,
            3,
        )
        model_rows.append(model)
    model_rows.sort(
        key=lambda item: (item["last_active_at"], item["requests"]),
        reverse=True,
    )
    tool_rows = sorted(
        tools.values(),
        key=lambda item: (item["invocations"], item["offered_in_requests"], item["name"]),
        reverse=True,
    )
    source_rows = sorted(
        sources.values(),
        key=lambda item: (item["requests"], item["client"]),
        reverse=True,
    )
    failure_type_rows = sorted(
        failure_types.values(),
        key=lambda item: (item["count"], item["label"]),
        reverse=True,
    )
    page_size = max(1, min(int(request_page_size or limit), 100))
    total_request_pages = max(1, math.ceil(len(records) / page_size))
    current_request_page = max(1, min(int(request_page), total_request_pages))
    request_offset = (current_request_page - 1) * page_size
    request_page_records = records[request_offset : request_offset + page_size]
    first_request_number = request_offset + 1 if records else 0
    last_request_number = request_offset + len(request_page_records)

    if records:
        earliest = min(
            float(record.get("completed_at") or record.get("started_at") or generated_at)
            for record in records
        )
    else:
        earliest = since or generated_at
    span = max(60.0, generated_at - earliest)
    bucket_seconds = max(60, math.ceil(span / 24))
    bucket_start = math.floor(earliest / bucket_seconds) * bucket_seconds
    bucket_count = max(1, min(24, math.ceil((generated_at - bucket_start) / bucket_seconds)))
    bucket_start = generated_at - (bucket_count * bucket_seconds)
    buckets = [
        {
            "started_at": bucket_start + (index * bucket_seconds),
            "requests": 0,
            "prompt_tokens": 0,
            "output_tokens": 0,
            "tool_calls": 0,
            "errors": 0,
        }
        for index in range(bucket_count)
    ]
    for record in records:
        timestamp = float(
            record.get("completed_at") or record.get("started_at") or generated_at
        )
        index = int((timestamp - bucket_start) // bucket_seconds)
        index = max(0, min(bucket_count - 1, index))
        bucket = buckets[index]
        bucket["requests"] += 1
        bucket["prompt_tokens"] += _number(record, "prompt_tokens")
        bucket["output_tokens"] += _number(record, "output_tokens")
        bucket["tool_calls"] += _number(record, "tool_call_count")
        bucket["errors"] += record.get("status") != "completed"

    return {
        "ok": True,
        "generated_at": generated_at,
        "window_seconds": window_seconds,
        "retention": {
            "days": retained_days,
            "archive_interval": "daily",
            "automatic_deletion": True,
        },
        "summary": {
            "requests": len(records),
            "active_requests": len(active_records),
            "completed_requests": completed,
            "errors": errors,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "total_tokens": token_total,
            "cached_prompt_tokens": cached_tokens,
            "cache_hit_percent": (
                round(
                    (cached_tokens / (prompt_tokens + cached_tokens)) * 100,
                    1,
                )
                if prompt_tokens or cached_tokens
                else 0.0
            ),
            "tool_calls": tool_calls,
            "mcp_tool_calls": mcp_tool_calls,
            "agent_loop_risks": len(agent_loop_risks),
            "high_agent_loop_risks": sum(
                risk["level"] == "high" for risk in agent_loop_risks
            ),
            "average_latency_seconds": (
                round(sum(durations) / len(durations), 3) if durations else 0.0
            ),
            "average_tokens_per_second": (
                round(token_total / sum(durations), 2)
                if sum(durations) > 0
                else 0.0
            ),
        },
        "active": active_records,
        "timeline": {
            "bucket_seconds": bucket_seconds,
            "buckets": buckets,
        },
        "models": model_rows,
        "tools": tool_rows,
        "sources": source_rows,
        "failure_types": failure_type_rows,
        "error_events": error_records[: max(1, min(limit, 500))],
        "agent_loop_risks": agent_loop_risks,
        "request_pagination": {
            "page": current_request_page,
            "page_size": page_size,
            "total_items": len(records),
            "total_pages": total_request_pages,
            "has_previous": current_request_page > 1,
            "has_next": current_request_page < total_request_pages,
            "first_item": first_request_number,
            "last_item": last_request_number,
        },
        "requests": request_page_records,
    }


class TelemetryStore:
    def __init__(
        self,
        path: Path | None = None,
        *,
        max_history: int = DEFAULT_MAX_HISTORY,
        retention_days: int | None = None,
    ) -> None:
        self.path = path or configured_telemetry_path()
        self.max_history = max(100, int(max_history))
        self.retention_days = max(
            1,
            int(retention_days or configured_telemetry_retention_days()),
        )
        self.lock = threading.RLock()
        self.history: deque[dict[str, Any]] = deque(
            load_telemetry_history(
                self.path,
                max_history=self.max_history,
                retention_days=self.retention_days,
            ),
            maxlen=self.max_history,
        )
        self.active: dict[str, dict[str, Any]] = {}
        self.records_since_compaction = 0
        self.last_housekeeping_day = ""
        self._compact()

    def start(self, record: dict[str, Any]) -> None:
        request_id = str(record["id"])
        with self.lock:
            self.active[request_id] = dict(record)

    def update(self, request_id: str, **updates: Any) -> None:
        with self.lock:
            record = self.active.get(request_id)
            if record is not None:
                record.update(updates)

    def record_response(self, request_id: str, response: dict[str, Any]) -> None:
        with self.lock:
            record = self.active.get(request_id)
            if record is None:
                return
            usage = response.get("usage")
            if isinstance(usage, dict):
                record["prompt_tokens"] = _number(usage, "prompt_tokens")
                record["output_tokens"] = _number(usage, "completion_tokens")
                prompt_details = usage.get("prompt_tokens_details")
                if isinstance(prompt_details, dict):
                    record["cached_prompt_tokens"] = _number(
                        prompt_details,
                        "cached_tokens",
                    )

            choices = response.get("choices")
            if not isinstance(choices, list):
                return
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                message = choice.get("message") or choice.get("delta") or {}
                if not isinstance(message, dict):
                    continue
                calls = message.get("tool_calls")
                if not isinstance(calls, list):
                    continue
                for call in calls:
                    if not isinstance(call, dict):
                        continue
                    function = call.get("function")
                    if not isinstance(function, dict):
                        continue
                    name = str(function.get("name") or "unknown")
                    normalized = {
                        "name": name,
                        "arguments_preview": "",
                        "mcp_server": mcp_server_for_tool(name),
                    }
                    existing = record.setdefault("tool_calls", [])
                    if normalized not in existing:
                        existing.append(normalized)
                        record["tool_call_count"] = len(existing)
                        record["mcp_tool_call_count"] = sum(
                            bool(item.get("mcp_server")) for item in existing
                        )

    def finish(
        self,
        request_id: str,
        *,
        request_state: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        now = time.time()
        with self.lock:
            record = self.active.pop(request_id, None)
            if record is None:
                return None
            state = request_state or {}
            for target, source in (
                ("prompt_tokens", "prompt_tokens"),
                ("output_tokens", "output_tokens"),
                ("cached_prompt_tokens", "cached_prompt_tokens"),
            ):
                if source in state:
                    record[target] = _number(state, source)

            if state.get("cancel_requested"):
                record["status"] = "cancelled"
                record["error"] = str(state.get("cancel_reason") or "")
            elif state.get("status") == "rejected":
                record["status"] = "error"
                record["error"] = str(state.get("error") or "")
            else:
                record["status"] = "completed"

            record["completed_at"] = now
            record["duration_seconds"] = round(
                max(0.0, now - float(record.get("started_at") or now)),
                3,
            )
            record["total_tokens"] = (
                _number(record, "prompt_tokens") + _number(record, "output_tokens")
            )
            self.history.append(record)
            self._append(record)
            return dict(record)

    def active_snapshot(self) -> list[dict[str, Any]]:
        now = time.time()
        with self.lock:
            records = [dict(record) for record in self.active.values()]
        for record in records:
            record["duration_seconds"] = round(_duration(record, now), 3)
        return records

    def snapshot(
        self,
        *,
        window_seconds: int = 86_400,
        limit: int = 100,
        request_page: int = 1,
        request_page_size: int | None = None,
    ) -> dict[str, Any]:
        with self.lock:
            current_day = time.strftime("%Y-%m-%d", time.gmtime())
            if current_day != self.last_housekeeping_day:
                self._compact()
            history = [dict(record) for record in self.history]
            active = [dict(record) for record in self.active.values()]
        return aggregate_telemetry(
            history,
            active=active,
            window_seconds=window_seconds,
            limit=limit,
            request_page=request_page,
            request_page_size=request_page_size,
            retention_days=self.retention_days,
        )

    def _append(self, record: dict[str, Any]) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, separators=(",", ":")) + "\n")
            self.records_since_compaction += 1
            current_day = time.strftime("%Y-%m-%d", time.gmtime())
            if (
                self.records_since_compaction >= 500
                or current_day != self.last_housekeeping_day
            ):
                self._compact()
        except (OSError, TypeError, ValueError):
            return

    def _compact(self) -> None:
        try:
            now = time.time()
            cutoff = now - (self.retention_days * SECONDS_PER_DAY)
            retained = [
                dict(record)
                for record in self.history
                if _record_timestamp(record) >= cutoff
            ]
            self.history = deque(retained, maxlen=self.max_history)
            current_day = time.strftime("%Y-%m-%d", time.gmtime(now))
            active_records: list[dict[str, Any]] = []
            archived_records: dict[str, list[dict[str, Any]]] = {}
            for record in retained:
                record_day = time.strftime(
                    "%Y-%m-%d",
                    time.gmtime(_record_timestamp(record)),
                )
                if record_day == current_day:
                    active_records.append(record)
                else:
                    archived_records.setdefault(record_day, []).append(record)

            archive_dir = telemetry_archive_directory(self.path)
            existing_archives = (
                set(archive_dir.glob("*.jsonl")) if archive_dir.is_dir() else set()
            )
            expected_archives: set[Path] = set()
            for record_day, records in archived_records.items():
                archive_path = archive_dir / f"{record_day}.jsonl"
                self._write_records(archive_path, records)
                expected_archives.add(archive_path)
            self._write_records(self.path, active_records)
            for stale_archive in existing_archives - expected_archives:
                stale_archive.unlink(missing_ok=True)
            if archive_dir.is_dir() and not any(archive_dir.iterdir()):
                archive_dir.rmdir()

            self.records_since_compaction = 0
            self.last_housekeeping_day = current_day
        except (OSError, TypeError, ValueError):
            return

    @staticmethod
    def _write_records(path: Path, records: list[dict[str, Any]]) -> None:
        if not records:
            path.unlink(missing_ok=True)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, separators=(",", ":")) + "\n")
        temporary.replace(path)
