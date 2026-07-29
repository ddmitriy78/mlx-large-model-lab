from __future__ import annotations

import hmac
import json
import mimetypes
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import psutil

from .cli import (
    LOGS_DIR,
    PROMPTS_DIR,
    ROOT,
    RUNS_DIR,
    configured_models,
    model_is_downloaded,
    model_label,
    model_path,
    read_benchmark_records,
    summarize_benchmark_records,
)
from .telemetry import (
    aggregate_telemetry,
    configured_telemetry_path,
    load_telemetry_history,
)
from .user_store import control_center_settings, update_control_center_settings

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
PROMPT_PROGRESS_RE = re.compile(
    r"Prompt processing progress:\s+(?P<current>\d+)/(?P<total>\d+)"
)
PROMPT_CACHE_RE = re.compile(
    r"Prompt Cache:\s+(?P<sequences>\d+) sequences,\s+(?P<gigabytes>[\d.]+) GB"
)
BENCHMARK_MODELS_RE = re.compile(r"^Benchmark models:\s+(?P<total>\d+)$")
BENCHMARK_PROMPTS_RE = re.compile(r"^Prompts:\s+(?P<total>\d+)$")
BENCHMARK_MODEL_RE = re.compile(r"^Benchmark model:\s+(?P<name>.+)$")
BENCHMARK_PROMPT_RE = re.compile(
    r"^Prompt:\s+(?P<name>.+?)(?:\s+\((?P<category>[^()]*)\))?$"
)


def benchmark_progress(
    lines: list[str],
    *,
    running: bool,
    started_at: float | None,
    finished_at: float | None,
    returncode: int | None,
) -> dict[str, Any]:
    model_total = 0
    prompt_total = 0
    model_current = 0
    prompt_current = 0
    current_model = ""
    current_prompt = ""
    current_category = ""
    stage = "Preparing benchmark"

    for raw_line in lines:
        line = raw_line.strip()
        if match := BENCHMARK_MODELS_RE.match(line):
            model_total = int(match.group("total"))
        elif match := BENCHMARK_PROMPTS_RE.match(line):
            prompt_total = int(match.group("total"))
        elif match := BENCHMARK_MODEL_RE.match(line):
            model_current += 1
            prompt_current = 0
            current_model = match.group("name")
            current_prompt = ""
            current_category = ""
            stage = "Loading model"
        elif match := BENCHMARK_PROMPT_RE.match(line):
            prompt_current += 1
            current_prompt = match.group("name")
            current_category = match.group("category") or ""
            stage = "Running prompt"
        elif line.startswith("Load time:") and model_current:
            stage = "Starting prompt set"
        elif line.startswith("Benchmark failed:"):
            stage = "Continuing after model failure"
        elif line.startswith("Model Comparison:") or line.startswith(
            "HTML report written:"
        ):
            stage = "Finalizing report"

    total_steps = model_total * prompt_total
    completed_steps = 0
    if total_steps and model_current:
        completed_steps = (
            (model_current - 1) * prompt_total + max(0, prompt_current - 1)
        )
    if not running and returncode == 0:
        completed_steps = total_steps
        stage = "Benchmark complete"
    elif not running and returncode is not None:
        stage = "Benchmark failed"
    completed_steps = min(total_steps, completed_steps)
    percent = (
        round((completed_steps / total_steps) * 100, 1) if total_steps else 0.0
    )

    if started_at is None:
        state = "idle"
        stage = "Waiting to start"
        elapsed_seconds = 0.0
    else:
        elapsed_until = time.time() if running else (finished_at or time.time())
        elapsed_seconds = max(0.0, elapsed_until - started_at)
        if running:
            state = "running"
        elif returncode == 0:
            state = "complete"
            percent = 100.0 if total_steps else 0.0
        elif returncode is not None:
            state = "failed"
        else:
            state = "starting"

    return {
        "state": state,
        "stage": stage,
        "model_current": model_current,
        "model_total": model_total,
        "prompt_current": prompt_current,
        "prompt_total": prompt_total,
        "current_model": current_model,
        "current_prompt": current_prompt,
        "current_category": current_category,
        "completed_steps": completed_steps,
        "total_steps": total_steps,
        "percent": percent,
        "elapsed_seconds": round(elapsed_seconds, 1),
    }


class JobState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.process: subprocess.Popen[str] | None = None
        self.command: list[str] = []
        self.lines: list[str] = []
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.returncode: int | None = None

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            running = (
                self.started_at is not None
                and self.finished_at is None
                and self.returncode is None
            )
            lines = self.lines[-500:]
            return {
                "running": running,
                "command": self.command,
                "lines": lines,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "returncode": self.returncode,
                "progress": benchmark_progress(
                    lines,
                    running=running,
                    started_at=self.started_at,
                    finished_at=self.finished_at,
                    returncode=self.returncode,
                ),
            }

    def start(self, command: list[str]) -> None:
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                raise RuntimeError("A benchmark is already running.")
            self.command = command
            self.lines = []
            self.started_at = time.time()
            self.finished_at = None
            self.returncode = None

        thread = threading.Thread(target=self._run, args=(command,), daemon=True)
        thread.start()

    def _run(self, command: list[str]) -> None:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        with self.lock:
            self.process = process

        assert process.stdout is not None
        for line in process.stdout:
            clean_line = ANSI_RE.sub("", line.rstrip())
            with self.lock:
                self.lines.append(clean_line)
                self.lines = self.lines[-500:]

        returncode = process.wait()
        with self.lock:
            self.returncode = returncode
            self.finished_at = time.time()


JOB = JobState()
MLX_SERVER_SCRIPT = ROOT / "scripts" / "mlx-server"
MLX_SERVER_LOG = LOGS_DIR / "mlx-server.log"
MLX_SERVER_RUNTIME_STATE = LOGS_DIR / "mlx-server-runtime.json"
MLX_SERVER_TELEMETRY = configured_telemetry_path(LOGS_DIR)
MLX_SERVER_MUTATING_COMMANDS = {"load", "switch", "unload", "restart"}
MLX_SERVER_COMMANDS = {"current", "list", *MLX_SERVER_MUTATING_COMMANDS}
MLX_SERVER_LOCK = threading.Lock()
MLX_INTERNAL_PATHS = {
    "status": "/mlx-lab/status",
    "cancel": "/mlx-lab/requests/cancel",
    "safety": "/mlx-lab/safety",
    "cache_clear": "/mlx-lab/cache/clear",
    "telemetry": "/mlx-lab/telemetry",
}


def control_center_access_payload(
    *,
    active_host: str | None = None,
    active_port: int | None = None,
    active_authentication: str | None = None,
    active_api_key: str | None = None,
    active_model_api_authentication: str | None = None,
) -> dict[str, Any]:
    settings = control_center_settings()
    configured_model_api_authentication = settings.get(
        "model_api_authentication",
        "none",
    )
    running_authentication = active_authentication or settings["authentication"]
    authentication_changed = (
        active_authentication is not None
        and settings["authentication"] != active_authentication
    )
    api_key_changed = False
    if (
        settings["authentication"] == "api_key"
        and running_authentication == "api_key"
        and active_api_key is not None
    ):
        configured_api_key = control_center_settings(include_secret=True)["api_key"]
        api_key_changed = not hmac.compare_digest(
            configured_api_key,
            active_api_key,
        )
    network_changed = (
        active_host is not None
        and active_port is not None
        and (
            settings["bind_host"] != active_host
            or settings["port"] != active_port
        )
    )
    control_center_restart_required = (
        network_changed
        or authentication_changed
        or api_key_changed
    )
    model_api_restart_required = (
        active_model_api_authentication is not None
        and configured_model_api_authentication != active_model_api_authentication
    )
    restart_required = (
        control_center_restart_required or model_api_restart_required
    )
    return {
        "ok": True,
        **settings,
        "model_api_authentication": configured_model_api_authentication,
        "authentication_required": running_authentication == "api_key",
        "configured_authentication_required": (
            settings["authentication"] == "api_key"
        ),
        "active_host": active_host,
        "active_port": active_port,
        "active_authentication": running_authentication,
        "active_model_api_authentication": active_model_api_authentication,
        "control_center_restart_required": control_center_restart_required,
        "model_api_restart_required": model_api_restart_required,
        "restart_required": restart_required,
    }


def valid_control_center_api_key(
    candidate: str,
    active_settings: dict[str, Any] | None = None,
) -> bool:
    settings = active_settings or control_center_settings()
    if settings["authentication"] != "api_key":
        return True
    secret = settings.get("api_key")
    if not secret:
        secret = control_center_settings(include_secret=True)["api_key"]
    return bool(candidate) and hmac.compare_digest(candidate, secret)


def control_center_restart_command() -> list[str]:
    return [
        sys.executable,
        "-m",
        "mlx_lab.cli",
        "ui",
    ]


def schedule_control_center_restart(*, delay_seconds: float = 0.35) -> None:
    command = control_center_restart_command()

    def restart() -> None:
        time.sleep(delay_seconds)
        os.execv(command[0], command)

    threading.Thread(
        target=restart,
        daemon=True,
        name="mlx-control-center-restart",
    ).start()


def rel_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def artifact_href(path: Path) -> str:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(RUNS_DIR.resolve())
    except ValueError:
        relative = resolved.relative_to(ROOT.resolve())
        namespace = "workspace"
    else:
        namespace = "runs"
    return "/artifact/" + urllib.parse.quote(f"{namespace}/{relative.as_posix()}")


def prompt_files() -> list[dict[str, Any]]:
    files = []
    for path in sorted(PROMPTS_DIR.glob("*.jsonl")):
        count = 0
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    count += 1
        files.append({"path": rel_path(path), "name": path.stem, "count": count})
    return files


def canonical_model_name(model: dict[str, Any]) -> str:
    repo_name = (
        str(model.get("repo_id") or "").strip().rstrip("/").rsplit("/", 1)[-1]
    )
    if repo_name:
        return repo_name
    return Path(str(model.get("local_dir") or "")).name


def model_options() -> list[dict[str, Any]]:
    models = []
    for key, model in configured_models():
        downloaded = model_is_downloaded(model)
        models.append(
            {
                "key": key,
                "label": model_label(key, model),
                "full_name": canonical_model_name(model),
                "repo_id": model["repo_id"],
                "local_dir": model["local_dir"],
                "downloaded": downloaded,
            }
        )
    return models


def recent_runs() -> list[dict[str, Any]]:
    RUNS_DIR.mkdir(exist_ok=True)
    benchmark_names: dict[str, str] = {}
    for key, model in configured_models():
        full_name = canonical_model_name(model)
        for identity in (
            key,
            model_label(key, model),
            model["repo_id"],
            model["local_dir"],
            full_name,
            *model.get("aliases", []),
        ):
            benchmark_names[normalized_model_identity(identity)] = full_name

    runs = []
    for jsonl_path in sorted(RUNS_DIR.glob("*.jsonl"), reverse=True)[:20]:
        html_path = jsonl_path.with_suffix(".html")
        try:
            records = read_benchmark_records(jsonl_path)
            summaries = summarize_benchmark_records(records) if records else []
            for summary in summaries:
                summary["full_name"] = benchmark_names.get(
                    normalized_model_identity(summary.get("label")),
                    summary.get("label") or "Unknown model",
                )
        except Exception as exc:
            records = []
            summaries = []
            error = str(exc)
        else:
            error = ""
        runs.append(
            {
                "name": jsonl_path.name,
                "jsonl_href": artifact_href(jsonl_path),
                "html_href": artifact_href(html_path) if html_path.exists() else "",
                "record_count": len(records),
                "summaries": summaries,
                "error": error,
                "mtime": jsonl_path.stat().st_mtime,
            }
        )
    return runs


def normalized_model_identity(value: Any) -> str:
    return str(value or "").strip().rstrip("/").casefold()


def model_statistics_payload() -> dict[str, Any]:
    RUNS_DIR.mkdir(exist_ok=True)
    configured = list(configured_models())
    identities: dict[str, str] = {}
    accumulators: dict[str, dict[str, Any]] = {}
    models: list[dict[str, Any]] = []

    for key, model in configured:
        local_path = model_path(model)
        aliases = list(model.get("aliases", []))
        identity_values = {
            key,
            model["repo_id"],
            model["local_dir"],
            str(local_path),
            local_path.name,
            model_label(key, model),
            *aliases,
        }
        for value in identity_values:
            normalized = normalized_model_identity(value)
            if normalized:
                identities.setdefault(normalized, key)
            if "/" in str(value) and not str(value).startswith(("http://", "https://")):
                identities.setdefault(normalized_model_identity(Path(str(value)).name), key)

        models.append(
            {
                "key": key,
                "label": model_label(key, model),
                "full_name": canonical_model_name(model),
                "repo_id": model["repo_id"],
                "aliases": aliases,
                "local_dir": model["local_dir"],
                "downloaded": model_is_downloaded(model),
                "estimated_size_gb": model.get("estimated_size_gb"),
                "role": model.get("role", ""),
                "notes": model.get("notes", ""),
            }
        )
        accumulators[key] = {
            "records": 0,
            "runs": set(),
            "prompts": set(),
            "categories": set(),
            "prompt_tokens": 0,
            "generation_tokens": 0,
            "prompt_tps": [],
            "generation_tps": [],
            "wall_seconds": [],
            "peak_memory_gb": 0.0,
            "last_benchmarked_at": None,
        }

    unmatched_records = 0
    source_errors: list[dict[str, str]] = []
    for run_path in sorted(RUNS_DIR.glob("*.jsonl")):
        try:
            records = read_benchmark_records(run_path)
        except Exception as exc:
            source_errors.append({"run": run_path.name, "error": str(exc)})
            continue

        run_mtime = run_path.stat().st_mtime
        for record in records:
            model_key = ""
            for field in ("model_key", "model_ref", "model_label", "model"):
                candidate = normalized_model_identity(record.get(field))
                if candidate in identities:
                    model_key = identities[candidate]
                    break
                if (
                    candidate
                    and "/" in str(record.get(field) or "")
                    and not str(record.get(field) or "").startswith(("http://", "https://"))
                ):
                    basename = normalized_model_identity(
                        Path(str(record.get(field))).name
                    )
                    if basename in identities:
                        model_key = identities[basename]
                        break
            if not model_key:
                unmatched_records += 1
                continue

            accumulator = accumulators[model_key]
            stats = record.get("stats") or {}

            def metric(name: str) -> float:
                try:
                    return float(stats.get(name) or 0)
                except (TypeError, ValueError):
                    return 0.0

            accumulator["records"] += 1
            accumulator["runs"].add(run_path.name)
            if record.get("prompt_name"):
                accumulator["prompts"].add(str(record["prompt_name"]))
            if record.get("category"):
                accumulator["categories"].add(str(record["category"]))
            accumulator["prompt_tokens"] += int(metric("prompt_tokens"))
            accumulator["generation_tokens"] += int(metric("generation_tokens"))
            if metric("prompt_tps") > 0:
                accumulator["prompt_tps"].append(metric("prompt_tps"))
            if metric("generation_tps") > 0:
                accumulator["generation_tps"].append(metric("generation_tps"))
            if metric("wall_seconds") > 0:
                accumulator["wall_seconds"].append(metric("wall_seconds"))
            accumulator["peak_memory_gb"] = max(
                accumulator["peak_memory_gb"],
                metric("peak_memory_gb"),
            )
            accumulator["last_benchmarked_at"] = max(
                accumulator["last_benchmarked_at"] or 0,
                run_mtime,
            )

    def average(values: list[float]) -> float | None:
        return round(sum(values) / len(values), 2) if values else None

    total_records = 0
    latest_benchmark_at: float | None = None
    for model in models:
        accumulator = accumulators[model["key"]]
        generation_rates = accumulator["generation_tps"]
        benchmark = {
            "record_count": accumulator["records"],
            "run_count": len(accumulator["runs"]),
            "unique_prompts": len(accumulator["prompts"]),
            "category_count": len(accumulator["categories"]),
            "prompt_tokens": accumulator["prompt_tokens"],
            "generation_tokens": accumulator["generation_tokens"],
            "average_prompt_tps": average(accumulator["prompt_tps"]),
            "average_generation_tps": average(generation_rates),
            "best_generation_tps": (
                round(max(generation_rates), 2) if generation_rates else None
            ),
            "average_wall_seconds": average(accumulator["wall_seconds"]),
            "peak_memory_gb": (
                round(accumulator["peak_memory_gb"], 2)
                if accumulator["peak_memory_gb"] > 0
                else None
            ),
            "last_benchmarked_at": accumulator["last_benchmarked_at"],
        }
        model["benchmark"] = benchmark
        total_records += benchmark["record_count"]
        if benchmark["last_benchmarked_at"]:
            latest_benchmark_at = max(
                latest_benchmark_at or 0,
                benchmark["last_benchmarked_at"],
            )

    benchmarked_models = [
        model for model in models if model["benchmark"]["record_count"] > 0
    ]
    speed_models = [
        model
        for model in benchmarked_models
        if model["benchmark"]["average_generation_tps"] is not None
    ]
    fastest_model = (
        max(
            speed_models,
            key=lambda item: item["benchmark"]["average_generation_tps"],
        )
        if speed_models
        else None
    )

    return {
        "generated_at": time.time(),
        "summary": {
            "total_models": len(models),
            "downloaded_models": sum(1 for model in models if model["downloaded"]),
            "benchmarked_models": len(benchmarked_models),
            "total_records": total_records,
            "unmatched_records": unmatched_records,
            "latest_benchmark_at": latest_benchmark_at,
            "fastest_model": (
                {
                    "key": fastest_model["key"],
                    "label": fastest_model["label"],
                    "full_name": fastest_model["full_name"],
                    "average_generation_tps": fastest_model["benchmark"][
                        "average_generation_tps"
                    ],
                }
                if fastest_model
                else None
            ),
        },
        "models": models,
        "source_errors": source_errors,
    }


def allowed_prompt_paths() -> set[str]:
    return {item["path"] for item in prompt_files()}


def validate_output_dir(value: str) -> str:
    if not value:
        return ""
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Output directory must be relative to the user run store.")
    return path.as_posix()


def build_benchmark_command(payload: dict[str, Any]) -> list[str]:
    prompts = str(payload.get("prompts") or "prompts/first_model.jsonl")
    if prompts not in allowed_prompt_paths():
        raise ValueError("Unknown prompt file.")

    max_tokens = int(payload.get("max_tokens") or 512)
    if max_tokens < 1 or max_tokens > 20000:
        raise ValueError("max_tokens must be between 1 and 20000.")

    command = [
        sys.executable,
        "-m",
        "mlx_lab.cli",
        "benchmark",
        "--prompts",
        prompts,
        "--max-tokens",
        str(max_tokens),
    ]

    limit = payload.get("limit")
    if limit not in (None, ""):
        limit_value = int(limit)
        if limit_value < 1:
            raise ValueError("limit must be positive.")
        command.extend(["--limit", str(limit_value)])

    output_dir = validate_output_dir(str(payload.get("output_dir") or ""))
    if output_dir:
        command.extend(["--output-dir", output_dir])

    if payload.get("all_models"):
        command.append("--all-models")
    else:
        model_key = str(payload.get("model_key") or "")
        allowed_models = {item["key"] for item in model_options() if item["downloaded"]}
        if model_key not in allowed_models:
            raise ValueError("Choose a downloaded model.")
        command.extend(["--model", model_key])

    return command


def clean_process_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return ANSI_RE.sub("", value).strip()


def mlx_server_control_timeout() -> int:
    try:
        return int(os.environ.get("MLX_SERVER_CONTROL_TIMEOUT", "360"))
    except ValueError:
        return 360


def mlx_server_port() -> int:
    try:
        return int(os.environ.get("MLX_SERVER_PORT", "48080"))
    except ValueError as exc:
        raise ValueError("MLX_SERVER_PORT must be an integer.") from exc


def model_api_authorization_headers() -> dict[str, str]:
    settings = control_center_settings()
    if not settings.get("api_key_configured"):
        return {}
    api_key = control_center_settings(include_secret=True)["api_key"]
    return {"authorization": f"Bearer {api_key}"}


def mlx_internal_request(
    action: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    timeout: float = 3.0,
) -> dict[str, Any]:
    if action not in MLX_INTERNAL_PATHS:
        raise ValueError("Unsupported MLX runtime action.")
    port = mlx_server_port()

    body = None
    headers = model_api_authorization_headers()
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["content-type"] = "application/json"
    query = urllib.parse.urlencode(params or {})
    url = f"http://127.0.0.1:{port}{MLX_INTERNAL_PATHS[action]}"
    if query:
        url += f"?{query}"
    request = urllib.request.Request(
        url,
        data=body,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            error_payload = json.loads(exc.read().decode("utf-8"))
        except (ValueError, TypeError):
            error_payload = {}
        raise RuntimeError(
            str(error_payload.get("error") or f"MLX runtime returned HTTP {exc.code}.")
        ) from exc
    except (OSError, urllib.error.URLError) as exc:
        raise RuntimeError(f"MLX runtime control is unavailable: {exc}") from exc

    if not isinstance(result, dict):
        raise RuntimeError("MLX runtime returned an invalid control response.")
    return result


def telemetry_payload(
    *,
    window_seconds: int = 86_400,
    limit: int = 100,
    request_page: int = 1,
    request_page_size: int | None = None,
) -> dict[str, Any]:
    if not 0 <= window_seconds <= 31_536_000:
        raise ValueError("window_seconds must be between 0 and 31536000.")
    if not 1 <= limit <= 500:
        raise ValueError("limit must be between 1 and 500.")
    if not 1 <= request_page <= 1_000_000:
        raise ValueError("request_page must be between 1 and 1000000.")
    page_size = min(limit, 100) if request_page_size is None else request_page_size
    if not 1 <= page_size <= 100:
        raise ValueError("request_page_size must be between 1 and 100.")

    try:
        result = mlx_internal_request(
            "telemetry",
            params={
                "window_seconds": window_seconds,
                "limit": limit,
                "request_page": request_page,
                "request_page_size": page_size,
            },
        )
        if (
            not isinstance(result.get("failure_types"), list)
            or not isinstance(result.get("error_events"), list)
            or any(
                "errors" not in source
                for source in result.get("sources", [])
                if isinstance(source, dict)
            )
            or not isinstance(result.get("request_pagination"), dict)
        ):
            retained = aggregate_telemetry(
                load_telemetry_history(MLX_SERVER_TELEMETRY),
                active=result.get("active"),
                window_seconds=window_seconds,
                limit=limit,
                request_page=request_page,
                request_page_size=page_size,
            )
            result["failure_types"] = retained["failure_types"]
            result["error_events"] = retained["error_events"]
            source_errors = {
                source["client"]: source["errors"]
                for source in retained["sources"]
            }
            for source in result.get("sources", []):
                if isinstance(source, dict):
                    source["errors"] = source_errors.get(source.get("client"), 0)
            if not isinstance(result.get("request_pagination"), dict):
                result["request_pagination"] = retained["request_pagination"]
                result["requests"] = retained["requests"]
                result["retention"] = retained["retention"]
        result["runtime_available"] = True
        return result
    except RuntimeError as exc:
        history = load_telemetry_history(MLX_SERVER_TELEMETRY)
        result = aggregate_telemetry(
            history,
            window_seconds=window_seconds,
            limit=limit,
            request_page=request_page,
            request_page_size=page_size,
        )
        result["runtime_available"] = False
        result["runtime_error"] = str(exc)
        return result


def validate_harness_chat_payload(
    payload: dict[str, Any],
    *,
    active_model: str,
    max_output_tokens: int = 32_000,
) -> dict[str, Any]:
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("Add at least one message.")
    if len(messages) > 128:
        raise ValueError("Conversation history is limited to 128 messages.")

    normalized_messages: list[dict[str, str]] = []
    total_characters = 0
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"Message {index + 1} must be an object.")
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"Message {index + 1} has an unsupported role.")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"Message {index + 1} must include text content.")
        total_characters += len(content)
        normalized_messages.append({"role": role, "content": content})

    if total_characters > 2_000_000:
        raise ValueError("Conversation text is limited to 2,000,000 characters.")

    max_tokens = payload.get("max_tokens", 1_200)
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
        raise ValueError("max_tokens must be an integer.")
    if not 1 <= max_tokens <= max_output_tokens:
        raise ValueError(
            f"max_tokens must be between 1 and {max_output_tokens} for the active server."
        )

    temperature = payload.get("temperature", 0.2)
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValueError("temperature must be a number.")
    temperature = float(temperature)
    if not 0 <= temperature <= 2:
        raise ValueError("temperature must be between 0 and 2.")

    return {
        "model": active_model,
        "messages": normalized_messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def build_harness_chat_request(payload: dict[str, Any]) -> urllib.request.Request:
    server = mlx_server_status_payload()
    status = server.get("status") or {}
    if not status.get("http_ready"):
        raise RuntimeError("Load an MLX model before sending a message.")

    active_model = str(status.get("loaded_model") or "")
    if active_model in {"", "none", "unknown"}:
        raise RuntimeError("The active MLX model could not be determined.")

    control = server.get("control") or {}
    safety = control.get("safety") or {}
    policy = safety.get("policy") or {}
    max_output_tokens = int(policy.get("max_output_tokens") or 32_000)
    upstream_payload = validate_harness_chat_payload(
        payload,
        active_model=active_model,
        max_output_tokens=max_output_tokens,
    )
    return urllib.request.Request(
        f"http://127.0.0.1:{mlx_server_port()}/v1/chat/completions",
        data=json.dumps(upstream_payload).encode("utf-8"),
        headers={
            "accept": "text/event-stream",
            "content-type": "application/json",
            "x-mlx-client": "Control Center Harness",
            **model_api_authorization_headers(),
        },
        method="POST",
    )


def mlx_harness_timeout() -> int:
    try:
        return max(1, int(os.environ.get("MLX_HARNESS_TIMEOUT", "900")))
    except ValueError:
        return 900


def run_mlx_server_command(command: str, model_key: str | None = None) -> dict[str, Any]:
    if command not in MLX_SERVER_COMMANDS:
        raise ValueError("Unsupported MLX server command.")

    args = [str(MLX_SERVER_SCRIPT), command]
    if model_key:
        args.append(model_key)

    def run() -> dict[str, Any]:
        try:
            completed = subprocess.run(
                args,
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=mlx_server_control_timeout(),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "command": args,
                "returncode": None,
                "timed_out": True,
                "stdout": clean_process_text(exc.stdout),
                "stderr": clean_process_text(exc.stderr),
            }

        return {
            "command": args,
            "returncode": completed.returncode,
            "timed_out": False,
            "stdout": clean_process_text(completed.stdout),
            "stderr": clean_process_text(completed.stderr),
        }

    if command in MLX_SERVER_MUTATING_COMMANDS:
        with MLX_SERVER_LOCK:
            return run()
    return run()


def configured_server_models() -> list[dict[str, Any]]:
    models = []
    for key, model in configured_models():
        local = model_path(model)
        aliases = list(model.get("aliases", []))
        accepted_inputs = [key, *aliases, local.name, model["repo_id"]]
        models.append(
            {
                "key": key,
                "label": model_label(key, model),
                "repo_id": model["repo_id"],
                "aliases": aliases,
                "local_dir": model["local_dir"],
                "local_path": str(local),
                "downloaded": model_is_downloaded(model),
                "estimated_size_gb": model.get("estimated_size_gb"),
                "role": model.get("role", ""),
                "accepted_inputs": accepted_inputs,
            }
        )
    return models


def resolve_server_model_input(model_input: str) -> tuple[str, dict[str, Any]]:
    normalized = model_input.strip()
    if not normalized:
        raise ValueError("Missing model.")

    input_path: Path | None = None
    try:
        path = Path(normalized).expanduser()
        input_path = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    except Exception:
        input_path = None

    for key, model in configured_models():
        local = model_path(model)
        candidates = {
            key,
            model["repo_id"],
            model["local_dir"],
            local.name,
            *model.get("aliases", []),
        }
        path_matches = input_path is not None and input_path == local.resolve()
        if normalized in candidates or path_matches:
            if not model_is_downloaded(model):
                raise ValueError("Choose a downloaded configured model.")
            return key, model

    raise ValueError("Unknown configured model. Use /api/mlx-server/models first.")


def line_value(lines: list[str], prefix: str) -> str:
    for line in lines:
        if line.startswith(prefix):
            return line.removeprefix(prefix).strip()
    return ""


def parse_mlx_server_current(output: str) -> dict[str, Any]:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    ready_endpoint = line_value(lines, "HTTP ready: ")
    not_ready_endpoint = line_value(lines, "HTTP not ready: ")
    listener_value = line_value(lines, "Listener PID(s) on port 48080: ")
    if not listener_value:
        listener_match = next(
            (line for line in lines if line.startswith("Listener PID(s) on port ")),
            "",
        )
        listener_value = listener_match.split(": ", 1)[1] if ": " in listener_match else ""

    listener_pids = [] if listener_value in ("", "none") else listener_value.split()
    return {
        "http_ready": bool(ready_endpoint),
        "endpoint": ready_endpoint or not_ready_endpoint,
        "managed_pid": line_value(lines, "Managed PID: ") or "none",
        "listener_pids": listener_pids,
        "loaded_model": line_value(lines, "Loaded model: ") or "unknown",
        "startup_model": line_value(lines, "Startup model: "),
        "default_model": line_value(lines, "Default model: "),
        "prompt_cache": line_value(lines, "Prompt cache: "),
        "prompt_cache_limit": line_value(lines, "Prompt cache limit: "),
        "command_line": line_value(lines, "Command: "),
    }


def canonical_model_ref(value: str) -> str:
    if not value:
        return ""
    path = Path(value).expanduser()
    if path.exists():
        return str(path.resolve())
    return value


def configured_model_for_ref(value: str) -> dict[str, Any] | None:
    canonical = canonical_model_ref(value)
    for model in configured_server_models():
        candidates = {
            canonical_model_ref(model["local_path"]),
            model["repo_id"],
            model["key"],
            *model["aliases"],
        }
        if canonical in candidates or value in candidates:
            return model
    return None


def read_runtime_state(pid: int | None = None) -> dict[str, Any]:
    try:
        state = json.loads(MLX_SERVER_RUNTIME_STATE.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    if not isinstance(state, dict):
        return {}
    if pid is not None and state.get("pid") != pid:
        return {}
    return state


def read_log_activity() -> dict[str, Any]:
    try:
        with MLX_SERVER_LOG.open("rb") as file:
            file.seek(0, os.SEEK_END)
            size = file.tell()
            file.seek(max(0, size - 512 * 1024))
            text = file.read().decode("utf-8", errors="replace")
    except OSError:
        return {
            "active": False,
            "prompt_current": 0,
            "prompt_total": 0,
            "prompt_percent": 0.0,
            "cache_sequences": 0,
            "cache_gb": 0.0,
            "last_error": "",
            "recent_lines": [],
        }

    lines = [line for line in text.splitlines() if line.strip()]
    start_index = max(
        (index for index, line in enumerate(lines) if "Starting httpd at " in line),
        default=0,
    )
    session_lines = lines[start_index:]
    progress_index = -1
    prompt_current = 0
    prompt_total = 0
    cache_sequences = 0
    cache_gb = 0.0
    completion_index = -1
    last_error = ""

    for index, line in enumerate(session_lines):
        if match := PROMPT_PROGRESS_RE.search(line):
            progress_index = index
            prompt_current = int(match.group("current"))
            prompt_total = int(match.group("total"))
        if match := PROMPT_CACHE_RE.search(line):
            cache_sequences = int(match.group("sequences"))
            cache_gb = float(match.group("gigabytes"))
        if '"POST /v1/' in line or "BrokenPipeError" in line:
            completion_index = index
        if "OutOfMemory" in line or "Insufficient Memory" in line:
            last_error = line.strip()

    active = progress_index > completion_index and prompt_current < prompt_total
    percent = (
        round((prompt_current / prompt_total) * 100, 1) if prompt_total else 0.0
    )
    return {
        "active": active,
        "prompt_current": prompt_current,
        "prompt_total": prompt_total,
        "prompt_percent": percent,
        "cache_sequences": cache_sequences,
        "cache_gb": cache_gb,
        "last_error": last_error,
        "recent_lines": session_lines[-35:],
    }


def system_memory_payload(pid: int | None) -> dict[str, Any]:
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    total = int(memory.total)
    available = int(memory.available)
    used = max(0, total - available)
    used_percent = round((used / total) * 100, 1) if total else 0.0
    if used_percent >= 90:
        pressure = "critical"
    elif used_percent >= 75:
        pressure = "elevated"
    else:
        pressure = "healthy"

    process: dict[str, Any] = {}
    if pid is not None:
        try:
            server_process = psutil.Process(pid)
            info = server_process.memory_info()
            process = {
                "pid": pid,
                "rss": int(info.rss),
                "vms": int(info.vms),
                "status": server_process.status(),
                "started_at": server_process.create_time(),
                "cpu_seconds": round(sum(server_process.cpu_times()[:2]), 2),
            }
        except (psutil.Error, OSError):
            process = {}

    return {
        "pressure": pressure,
        "total": total,
        "available": available,
        "used": used,
        "used_percent": used_percent,
        "free": int(memory.free),
        "active": int(getattr(memory, "active", 0)),
        "inactive": int(getattr(memory, "inactive", 0)),
        "wired": int(getattr(memory, "wired", 0)),
        "cached": int(getattr(memory, "cached", 0)),
        "swap_used": int(swap.used),
        "swap_percent": float(swap.percent),
        "process": process,
        "load_average": [round(value, 2) for value in os.getloadavg()],
    }


def mlx_server_status_payload() -> dict[str, Any]:
    result = run_mlx_server_command("current")
    status = parse_mlx_server_current(result["stdout"])
    try:
        pid = int(status["managed_pid"])
    except (TypeError, ValueError):
        pid = None
    runtime = read_runtime_state(pid)
    runtime_model = str(runtime.get("runtime_model") or "")
    if runtime_model:
        status["loaded_model"] = runtime_model
    model = configured_model_for_ref(status["loaded_model"])
    control: dict[str, Any] = {}
    if status["http_ready"]:
        try:
            control = mlx_internal_request("status")
        except RuntimeError as exc:
            control = {"ok": False, "error": str(exc)}
    return {
        "ok": result["returncode"] == 0 and not result["timed_out"],
        "collected_at": time.time(),
        "status": status,
        "runtime": runtime,
        "control": control,
        "model": model,
        "memory": system_memory_payload(pid),
        "activity": read_log_activity(),
        "result": result,
    }


def mlx_server_models_payload() -> dict[str, Any]:
    result = run_mlx_server_command("list")
    return {
        "ok": result["returncode"] == 0 and not result["timed_out"],
        "models": configured_server_models(),
        "result": result,
    }


def mlx_server_help_payload() -> dict[str, Any]:
    return {
        "ok": True,
        "endpoints": [
            {
                "method": "GET",
                "path": "/api/auth/status",
                "description": "Show whether Control Center API key authentication is required and satisfied.",
            },
            {
                "method": "GET",
                "path": "/api/settings/access",
                "description": "Show the configured Control Center bind scope, port, and authentication mode.",
            },
            {
                "method": "POST",
                "path": "/api/settings/access",
                "body": {
                    "access_mode": "network",
                    "port": 8765,
                    "authentication": "api_key",
                    "model_api_authentication": "api_key",
                    "api_key": "optional-replacement-with-24-or-more-characters",
                },
                "description": "Save next-start Control Center and Model API access settings and optionally replace their shared API key.",
            },
            {
                "method": "POST",
                "path": "/api/control-center/restart",
                "description": "Restart the Control Center process and apply its saved access settings.",
            },
            {
                "method": "GET",
                "path": "/api/mlx-server/status",
                "description": "Show HTTP readiness, PID, listener, and loaded model.",
            },
            {
                "method": "GET",
                "path": "/api/mlx-server/current",
                "description": "Alias for /api/mlx-server/status.",
            },
            {
                "method": "GET",
                "path": "/api/mlx-server/models",
                "description": "List configured model aliases and download status.",
            },
            {
                "method": "GET",
                "path": "/api/model-statistics",
                "description": "Aggregate registry and benchmark statistics for every configured model.",
            },
            {
                "method": "GET",
                "path": "/api/telemetry?window_seconds=86400&limit=100&request_page=1&request_page_size=25",
                "description": "Summarize MLX activity and return one page of retained recent queries.",
            },
            {
                "method": "GET",
                "path": "/api/mlx-server/list",
                "description": "Alias for /api/mlx-server/models.",
            },
            {
                "method": "POST",
                "path": "/api/mlx-server/load",
                "body": {"model": "qwen3-coder-next-8bit"},
                "description": "Restart the MLX server with one downloaded model loaded.",
            },
            {
                "method": "POST",
                "path": "/api/mlx-server/switch",
                "body": {"model": "qwen3-coder-next"},
                "description": "Alias for load.",
            },
            {
                "method": "POST",
                "path": "/api/mlx-server/restart",
                "body": {"model": "qwen3-coder-next"},
                "description": "Restart the MLX server with the selected downloaded model.",
            },
            {
                "method": "POST",
                "path": "/api/mlx-server/unload",
                "description": "Stop the MLX server and release model memory.",
            },
            {
                "method": "POST",
                "path": "/api/mlx-server/cancel",
                "body": {"request_id": "chatcmpl-optional"},
                "description": "Cancel one request, or every active request when omitted.",
            },
            {
                "method": "POST",
                "path": "/api/mlx-server/safety",
                "body": {
                    "enabled": True,
                    "max_context_tokens": 128000,
                    "max_output_tokens": 32000,
                    "memory_reserve_bytes": 17179869184,
                    "memory_abort_bytes": 8589934592,
                },
                "description": "Update in-memory admission and memory-guard limits.",
            },
            {
                "method": "POST",
                "path": "/api/mlx-server/cache/clear",
                "body": {"entry_id": "cache-optional"},
                "description": "Clear one retained prompt cache, or all when omitted.",
            },
            {
                "method": "POST",
                "path": "/api/harness/chat",
                "body": {
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 1200,
                    "temperature": 0.2,
                },
                "description": "Stream a chat completion from the active MLX model.",
            },
        ],
    }


CONTROL_PLANE_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="icon" href="data:,">
  <title>MLX Control Center</title>
  <style>
    :root {
      --bg: #f5f7fa;
      --panel: #ffffff;
      --text: #1f2933;
      --muted: #687385;
      --line: #d9dee7;
      --accent: #0f766e;
      --accent-dark: #115e59;
      --danger: #b42318;
      --warning: #b54708;
      --healthy: #16803c;
      --code: #101827;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
        "Segoe UI", sans-serif;
      letter-spacing: 0;
    }
    main {
      width: min(1280px, calc(100vw - 32px));
      margin: 24px auto 40px;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 18px;
    }
    .eyebrow {
      margin: 0 0 4px;
      color: var(--accent);
      font-size: 11px;
      font-weight: 800;
      letter-spacing: .12em;
      text-transform: uppercase;
    }
    h1 {
      margin: 0;
      font-size: 26px;
      letter-spacing: 0;
    }
    h2 {
      margin: 0 0 12px;
      font-size: 16px;
      letter-spacing: 0;
    }
    .status {
      min-width: 170px;
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      text-align: right;
      font-size: 13px;
      color: var(--muted);
    }
    .status strong {
      display: block;
      color: var(--text);
      font-size: 15px;
    }
    .status[data-state="healthy"] strong { color: var(--healthy); }
    .status[data-state="elevated"] strong { color: var(--warning); }
    .status[data-state="critical"] strong { color: var(--danger); }
    .status small {
      display: block;
      margin-top: 2px;
      font-variant-numeric: tabular-nums;
    }
    .overview {
      margin-bottom: 16px;
      padding: 18px;
    }
    .overview-heading {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 14px;
    }
    .overview-heading h2 { margin-bottom: 3px; font-size: 18px; }
    .muted { margin: 0; color: var(--muted); font-size: 13px; }
    .metric-grid {
      display: grid;
      grid-template-columns: 2fr repeat(4, minmax(120px, 1fr));
      gap: 10px;
    }
    .metric {
      min-width: 0;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #f8fafc;
    }
    .metric span {
      display: block;
      margin-bottom: 5px;
      color: var(--muted);
      font-size: 10px;
      font-weight: 750;
      letter-spacing: .06em;
      text-transform: uppercase;
    }
    .metric strong {
      display: block;
      overflow: hidden;
      font-size: 16px;
      font-variant-numeric: tabular-nums;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .metric small {
      display: block;
      overflow: hidden;
      margin-top: 3px;
      color: var(--muted);
      font-size: 11px;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .memory-panel {
      margin-top: 12px;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 7px;
    }
    .memory-labels,
    .request-labels {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 7px;
      font-size: 12px;
    }
    .memory-track,
    .request-track {
      height: 12px;
      overflow: hidden;
      border-radius: 999px;
      background: #e5e9f0;
    }
    .memory-fill,
    .request-fill {
      width: 0;
      height: 100%;
      border-radius: inherit;
      background: var(--healthy);
      transition: width .25s ease, background .25s ease;
    }
    .memory-fill.elevated { background: var(--warning); }
    .memory-fill.critical { background: var(--danger); }
    .request-fill { background: var(--accent); }
    .request-meta {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 8px;
    }
    .request-meta small {
      display: block;
      margin-top: 2px;
      color: var(--muted);
      font-size: 11px;
      overflow-wrap: anywhere;
    }
    .request-meta button {
      flex: 0 0 auto;
      width: auto;
      height: 32px;
      margin: 0;
      padding: 0 12px;
    }
    .control-grid {
      display: grid;
      grid-template-columns: minmax(300px, .8fr) minmax(360px, 1.2fr);
      gap: 12px;
      margin-top: 12px;
    }
    .control-panel {
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 7px;
    }
    .control-panel label:first-child { margin-top: 0; }
    .control-actions {
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 8px;
    }
    .control-actions button { margin-top: 10px; }
    .safety-grid {
      display: grid;
      grid-template-columns: minmax(360px, 1fr) minmax(360px, 1fr);
      gap: 12px;
      margin-top: 12px;
    }
    .panel-heading {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 8px;
    }
    .panel-heading h3 {
      margin: 0 0 3px;
      font-size: 14px;
    }
    .field-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0 10px;
    }
    .safety-summary {
      min-height: 36px;
      margin: 10px 0 0;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
    }
    .cache-panel .table-wrap { max-height: 230px; overflow: auto; }
    .cache-panel td:last-child { text-align: right; }
    button.inline-button {
      width: auto;
      height: 28px;
      margin: 0;
      padding: 0 10px;
      font-size: 11px;
    }
    .empty-row {
      color: var(--muted);
      text-align: center;
    }
    .activity-panel {
      margin-top: 12px;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #fbfcfd;
    }
    .activity-panel[hidden] { display: none; }
    button.compact {
      width: auto;
      height: 34px;
      margin: 0;
      padding: 0 14px;
    }
    .tab-list {
      display: flex;
      gap: 4px;
      margin-bottom: 16px;
      padding: 4px;
      border: 1px solid var(--line);
      border-radius: 9px;
      background: #e9edf3;
    }
    .tab-button {
      width: auto;
      min-width: 150px;
      height: 38px;
      margin: 0;
      padding: 0 18px;
      border: 0;
      background: transparent;
      color: var(--muted);
      box-shadow: none;
    }
    .tab-button:hover {
      background: rgb(255 255 255 / 65%);
      color: var(--text);
    }
    .tab-button.active {
      background: var(--panel);
      color: var(--accent-dark);
      box-shadow: 0 1px 3px rgb(15 23 42 / 10%);
    }
    .tab-panel[hidden] { display: none; }
    .benchmark-section {
      min-height: 620px;
    }
    .benchmark-heading {
      margin-bottom: 14px;
    }
    .benchmark-workspace {
      display: grid;
      grid-template-columns: minmax(300px, 360px) minmax(0, 1fr);
      gap: 16px;
      align-items: start;
    }
    .benchmark-live-column {
      min-width: 0;
    }
    .latest-comparison-bars:empty::before {
      display: block;
      color: var(--muted);
      content: "No completed benchmark to compare yet.";
      font-size: 12px;
      line-height: 1.45;
    }
    .latest-comparison-bars .bar {
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 6px 12px;
      align-items: start;
      margin: 0;
      padding: 10px 0;
      border-bottom: 1px solid var(--line);
    }
    .latest-comparison-bars .bar:first-child {
      padding-top: 2px;
    }
    .latest-comparison-bars .bar:last-child {
      padding-bottom: 2px;
      border-bottom: 0;
    }
    .latest-comparison-model {
      display: -webkit-box;
      min-width: 0;
      overflow: hidden;
      font-weight: 650;
      line-height: 1.35;
      overflow-wrap: anywhere;
      -webkit-box-orient: vertical;
      -webkit-line-clamp: 2;
    }
    .latest-comparison-rate {
      color: var(--text);
      font-variant-numeric: tabular-nums;
      font-weight: 750;
      white-space: nowrap;
    }
    .latest-comparison-rate small {
      color: var(--muted);
      font-size: 10px;
      font-weight: 650;
    }
    .latest-comparison-bars .bar-track {
      grid-column: 1 / -1;
      width: 100%;
      height: 8px;
    }
    .benchmark-progress-panel {
      overflow: hidden;
    }
    .benchmark-panel-heading,
    .console-heading {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 12px;
    }
    .benchmark-panel-heading h2,
    .console-heading h2 {
      margin-bottom: 3px;
    }
    .benchmark-status-badge {
      flex: 0 0 auto;
      padding: 4px 9px;
      border: 1px solid #cfd8e5;
      border-radius: 999px;
      background: #f3f6fa;
      color: #526174;
      font-size: 10px;
      font-weight: 800;
      letter-spacing: .06em;
      text-transform: uppercase;
    }
    .benchmark-status-badge[data-state="running"],
    .benchmark-status-badge[data-state="starting"] {
      border-color: #b8ded9;
      background: #ecfdf9;
      color: var(--accent-dark);
    }
    .benchmark-status-badge[data-state="complete"] {
      border-color: #b9dfc5;
      background: #effbf3;
      color: var(--healthy);
    }
    .benchmark-status-badge[data-state="failed"] {
      border-color: #e5c3be;
      background: #fff4f2;
      color: var(--danger);
    }
    .benchmark-progress-summary {
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 12px;
      margin-top: 16px;
    }
    .benchmark-progress-summary strong {
      font-size: 26px;
      font-variant-numeric: tabular-nums;
      letter-spacing: -.02em;
    }
    .benchmark-progress-summary span {
      color: var(--muted);
      font-size: 12px;
      font-variant-numeric: tabular-nums;
    }
    .benchmark-progress-track {
      height: 14px;
      margin-top: 8px;
      overflow: hidden;
      border-radius: 999px;
      background: #e5e9f0;
    }
    .benchmark-progress-fill {
      width: 0;
      height: 100%;
      border-radius: inherit;
      background: var(--accent);
      transition: width .35s ease;
    }
    .benchmark-progress-fill.running {
      background-image: linear-gradient(
        110deg,
        var(--accent) 0 42%,
        #14968a 42% 58%,
        var(--accent) 58% 100%
      );
      background-size: 44px 100%;
      animation: benchmark-progress-stripes 1.1s linear infinite;
    }
    @keyframes benchmark-progress-stripes {
      to { background-position: 44px 0; }
    }
    .benchmark-progress-metrics {
      display: grid;
      grid-template-columns: 1fr 1fr .7fr;
      gap: 8px;
      margin-top: 12px;
    }
    .benchmark-progress-metric {
      min-width: 0;
      padding: 10px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #f8fafc;
    }
    .benchmark-progress-metric span,
    .benchmark-progress-metric strong,
    .benchmark-progress-metric small {
      display: block;
    }
    .benchmark-progress-metric span {
      color: var(--muted);
      font-size: 10px;
      font-weight: 750;
      letter-spacing: .06em;
      text-transform: uppercase;
    }
    .benchmark-progress-metric strong {
      margin-top: 4px;
      font-size: 15px;
      font-variant-numeric: tabular-nums;
    }
    .benchmark-progress-metric small {
      overflow: hidden;
      margin-top: 3px;
      color: var(--muted);
      font-size: 11px;
      line-height: 1.35;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .benchmark-progress-activity {
      min-height: 18px;
      margin: 11px 0 0;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
    }
    .benchmark-progress-activity.running::before {
      display: inline-block;
      width: 7px;
      height: 7px;
      margin: 0 7px 1px 0;
      border-radius: 999px;
      background: var(--accent);
      content: "";
      animation: benchmark-pulse 1.3s ease-in-out infinite;
    }
    @keyframes benchmark-pulse {
      50% { opacity: .3; transform: scale(.75); }
    }
    .benchmark-console-panel {
      min-width: 0;
    }
    .console-heading {
      align-items: center;
      margin-bottom: 10px;
    }
    .console-heading h2 {
      margin: 0;
    }
    .console-tools {
      display: flex;
      align-items: center;
      gap: 12px;
      color: var(--muted);
      font-size: 11px;
    }
    .console-follow {
      display: flex;
      align-items: center;
      gap: 6px;
      margin: 0;
      color: var(--muted);
      font-size: 11px;
      font-weight: 650;
      text-transform: none;
      white-space: nowrap;
    }
    .console-follow input {
      width: 14px;
      height: 14px;
      margin: 0;
    }
    #logOutput {
      height: clamp(380px, 48vh, 620px);
      min-height: 380px;
      max-height: none;
      font-size: 13px;
      line-height: 1.55;
      white-space: pre-wrap;
    }
    .statistics-section {
      margin-top: 16px;
    }
    .statistics-summary {
      margin: 16px 0;
    }
    .statistics-grid {
      display: grid;
      grid-template-columns: minmax(0, 1.4fr) minmax(280px, .6fr);
      gap: 12px;
      margin-bottom: 16px;
    }
    .statistics-panel {
      min-width: 0;
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfd;
    }
    .statistics-panel h3 {
      margin: 0 0 4px;
      font-size: 14px;
    }
    .statistics-speed-bars {
      margin-top: 12px;
    }
    .statistics-speed-bars .bar {
      grid-template-columns: minmax(150px, 230px) 1fr 82px;
    }
    .statistics-model {
      min-width: 230px;
      white-space: normal;
    }
    .statistics-model strong,
    .statistics-model small {
      display: block;
    }
    .statistics-model small {
      max-width: 360px;
      margin-top: 3px;
      color: var(--muted);
      line-height: 1.35;
    }
    .statistics-table table {
      min-width: 1160px;
    }
    .availability-badge {
      display: inline-block;
      padding: 3px 7px;
      border: 1px solid #b8ded9;
      border-radius: 999px;
      background: #ecfdf9;
      color: var(--accent-dark);
      font-size: 10px;
      font-weight: 750;
      text-transform: uppercase;
    }
    .availability-badge.missing {
      border-color: #e5c3be;
      background: #fff4f2;
      color: var(--danger);
    }
    .statistics-note {
      margin: 10px 0 0;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.5;
    }
    .telemetry-section {
      min-height: 650px;
    }
    .telemetry-toolbar {
      display: flex;
      align-items: end;
      gap: 8px;
    }
    .telemetry-toolbar label {
      margin: 0;
    }
    .telemetry-toolbar select {
      min-width: 128px;
    }
    .telemetry-summary {
      grid-template-columns: repeat(7, minmax(0, 1fr));
      margin: 16px 0;
    }
    .telemetry-summary .metric:first-child {
      grid-column: auto;
    }
    .telemetry-grid {
      display: grid;
      grid-template-columns: minmax(0, 1.25fr) minmax(320px, .75fr);
      gap: 12px;
      margin-bottom: 12px;
    }
    .telemetry-panel {
      min-width: 0;
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfd;
    }
    .telemetry-panel h3 {
      margin: 0 0 4px;
      font-size: 14px;
    }
    .telemetry-chart {
      display: flex;
      height: 150px;
      align-items: end;
      gap: 4px;
      margin-top: 14px;
      padding-top: 8px;
      border-bottom: 1px solid var(--line);
    }
    .telemetry-column {
      position: relative;
      display: flex;
      flex: 1;
      min-width: 3px;
      height: 100%;
      align-items: end;
      border-radius: 3px 3px 0 0;
      background: #e5e9f0;
      overflow: hidden;
    }
    .telemetry-column span {
      display: block;
      width: 100%;
      min-height: 2px;
      background: var(--accent);
    }
    .telemetry-column.has-error span {
      background: var(--warning);
    }
    button.telemetry-column {
      width: auto;
      margin: 0;
      padding: 0;
      border: 0;
      border-radius: 3px 3px 0 0;
      background: #e5e9f0;
      box-shadow: none;
      color: var(--warning);
    }
    button.telemetry-column:hover,
    button.telemetry-column:focus-visible {
      background: #f4dcc2;
      outline: 2px solid #e39b53;
      outline-offset: 1px;
    }
    .telemetry-column-error-count {
      position: absolute;
      z-index: 1;
      top: 3px;
      left: 50%;
      min-width: 16px;
      padding: 1px 4px;
      border-radius: 999px;
      background: #fff7ed;
      box-shadow: 0 0 0 1px #efc18f;
      font-size: 9px;
      font-style: normal;
      line-height: 14px;
      text-align: center;
      transform: translateX(-50%);
    }
    .telemetry-legend {
      display: flex;
      flex-wrap: wrap;
      gap: 6px 14px;
      margin-top: 9px;
      color: var(--muted);
      font-size: 11px;
    }
    .telemetry-legend strong {
      color: var(--text);
    }
    .telemetry-active {
      margin-bottom: 12px;
      padding: 11px 12px;
      border: 1px solid #b8ded9;
      border-radius: 7px;
      background: #ecfdf9;
      color: var(--accent-dark);
      font-size: 12px;
    }
    .telemetry-active[hidden] {
      display: none;
    }
    .telemetry-active strong {
      display: block;
      margin-bottom: 3px;
    }
    .agent-loop-alerts {
      display: grid;
      grid-template-columns: minmax(0, 1fr);
      gap: 10px;
      margin-bottom: 12px;
    }
    .agent-loop-alerts[hidden] {
      display: none;
    }
    .agent-loop-alert {
      min-width: 0;
      padding: 13px 14px;
      border: 1px solid #f0c48a;
      border-left: 4px solid var(--warning);
      border-radius: 7px;
      background: #fff9ef;
    }
    .agent-loop-alert.high {
      border-color: #efb1a9;
      border-left-color: var(--danger);
      background: #fff4f2;
    }
    .agent-loop-alert.watch {
      border-color: #cfd8e5;
      border-left-color: #6b7c93;
      background: #f8fafc;
    }
    .agent-loop-alert-heading {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 6px;
    }
    .agent-loop-alert-heading strong {
      min-width: 0;
      overflow: hidden;
      font-size: 13px;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .agent-loop-alert p {
      margin: 4px 0 0;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
    }
    .agent-loop-alert .recommendation {
      color: var(--text);
      font-weight: 650;
    }
    .agent-loop-metrics {
      display: flex;
      flex-wrap: wrap;
      gap: 5px 12px;
      margin-top: 8px;
      color: var(--text);
      font-size: 11px;
      font-variant-numeric: tabular-nums;
    }
    .agent-loop-signals {
      margin: 8px 0 0;
      padding-left: 18px;
      color: var(--muted);
      font-size: 11px;
      line-height: 1.5;
    }
    .telemetry-table table {
      table-layout: fixed;
    }
    .telemetry-table th:first-child,
    .telemetry-table td:first-child {
      width: 36%;
    }
    .telemetry-table .telemetry-model {
      min-width: 0;
    }
    .telemetry-collapsible:not(.expanded) tr:nth-child(n + 6) {
      display: none;
    }
    .telemetry-toggle-row {
      display: flex;
      justify-content: center;
      margin-top: 8px;
    }
    .telemetry-query-table table {
      min-width: 1180px;
    }
    .telemetry-query-table {
      max-height: 680px;
      overflow: auto;
    }
    .telemetry-query-heading-meta {
      display: grid;
      justify-items: end;
      gap: 3px;
      text-align: right;
    }
    .telemetry-retention-policy {
      color: var(--healthy);
      font-size: 11px;
      font-weight: 700;
    }
    .telemetry-pagination {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      min-height: 34px;
      margin-top: 10px;
    }
    .telemetry-pagination-controls {
      display: flex;
      align-items: center;
      justify-content: flex-end;
      gap: 8px;
    }
    .telemetry-pagination label {
      margin: 0;
      font-size: 10px;
      white-space: nowrap;
    }
    .telemetry-pagination select {
      width: auto;
      min-width: 68px;
      height: 28px;
      padding: 3px 24px 3px 7px;
      font-size: 11px;
    }
    .telemetry-page-status {
      min-width: 76px;
      text-align: center;
      white-space: nowrap;
    }
    .telemetry-query {
      min-width: 300px;
      max-width: 520px;
      line-height: 1.4;
      white-space: normal;
    }
    .telemetry-model {
      min-width: 220px;
      max-width: 340px;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .telemetry-badge {
      display: inline-block;
      padding: 2px 6px;
      border-radius: 999px;
      background: #e8edf4;
      color: #46556a;
      font-size: 10px;
      font-weight: 750;
      text-transform: uppercase;
    }
    .telemetry-badge.completed {
      background: #e9f8ef;
      color: var(--healthy);
    }
    .telemetry-badge.error,
    .telemetry-badge.cancelled {
      background: #fff0ed;
      color: var(--danger);
    }
    .telemetry-badge.mcp {
      background: #eeeafd;
      color: #5b3db1;
    }
    .telemetry-note {
      min-height: 18px;
      margin: 8px 0 0;
      color: var(--muted);
      font-size: 11px;
      line-height: 1.45;
    }
    .telemetry-failure-panel {
      margin-bottom: 12px;
      border-left: 4px solid var(--warning);
    }
    .telemetry-failure-actions {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .telemetry-failure-count {
      padding: 3px 8px;
      border-radius: 999px;
      background: #fff0ed;
      color: var(--danger);
      font-size: 10px;
      font-weight: 750;
      white-space: nowrap;
    }
    .telemetry-failure-count.clear {
      background: #e9f8ef;
      color: var(--healthy);
    }
    .telemetry-failure-summary {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin: 10px 0;
    }
    .telemetry-failure-summary-group {
      min-width: 0;
      padding: 9px 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
    }
    .telemetry-failure-summary-group > strong {
      display: block;
      margin-bottom: 6px;
      color: var(--muted);
      font-size: 10px;
      letter-spacing: .06em;
      text-transform: uppercase;
    }
    .telemetry-failure-chips {
      display: flex;
      flex-wrap: wrap;
      gap: 5px;
    }
    .telemetry-failure-chip {
      max-width: 100%;
      padding: 3px 7px;
      overflow: hidden;
      border-radius: 999px;
      background: #fff7ed;
      color: #8a3d08;
      font-size: 10px;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .telemetry-failure-chip strong {
      margin-right: 3px;
      color: var(--danger);
    }
    .telemetry-failure-filter {
      margin: 0 0 8px;
      color: var(--warning);
      font-size: 11px;
      font-weight: 650;
    }
    .telemetry-failure-table {
      max-height: 440px;
      overflow: auto;
    }
    .telemetry-failure-table table {
      min-width: 1050px;
    }
    .telemetry-failure-row td {
      background: #fffaf8;
    }
    .telemetry-failure-source strong,
    .telemetry-failure-details strong {
      display: block;
      font-size: 11px;
      line-height: 1.4;
    }
    .telemetry-failure-source small,
    .telemetry-failure-details small,
    .telemetry-failure-endpoint small {
      display: block;
      margin-top: 2px;
      color: var(--muted);
      font-size: 10px;
      line-height: 1.4;
    }
    .telemetry-failure-endpoint code {
      font-size: 10px;
      white-space: nowrap;
    }
    .telemetry-source-count small {
      display: block;
      margin-top: 2px;
      color: var(--muted);
      font-size: 9px;
      white-space: nowrap;
    }
    .telemetry-source-count small.has-error {
      color: var(--danger);
      font-weight: 700;
    }
    .harness-section {
      min-height: 650px;
    }
    .harness-heading {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 14px;
    }
    .harness-heading h2 { margin-bottom: 3px; }
    .model-pill {
      max-width: 380px;
      padding: 6px 10px;
      overflow: hidden;
      border: 1px solid #b8ded9;
      border-radius: 999px;
      background: #ecfdf9;
      color: var(--accent-dark);
      font-size: 12px;
      font-weight: 750;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .harness-layout {
      display: grid;
      grid-template-columns: minmax(260px, 330px) minmax(0, 1fr);
      gap: 14px;
      align-items: start;
    }
    .harness-settings {
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #fbfcfd;
    }
    .harness-settings label:first-child { margin-top: 0; }
    .harness-settings textarea { min-height: 142px; }
    .harness-workspace {
      min-width: 0;
    }
    .harness-transcript {
      min-height: 340px;
      max-height: 560px;
      overflow: auto;
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #f8fafc;
      scroll-behavior: smooth;
    }
    .harness-empty {
      display: grid;
      min-height: 310px;
      place-items: center;
      color: var(--muted);
      font-size: 13px;
      text-align: center;
    }
    .harness-message {
      width: min(88%, 760px);
      margin: 0 0 12px;
      padding: 11px 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      box-shadow: 0 1px 1px rgb(15 23 42 / 4%);
    }
    .harness-message.user {
      margin-left: auto;
      border-color: #b8ded9;
      background: #ecfdf9;
    }
    .harness-message.pending {
      border-style: dashed;
    }
    .harness-message-header {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 6px;
      color: var(--muted);
      font-size: 10px;
      font-weight: 750;
      letter-spacing: .06em;
      text-transform: uppercase;
    }
    .harness-message-body {
      font-size: 14px;
      line-height: 1.55;
      overflow-wrap: anywhere;
      white-space: pre-wrap;
    }
    .harness-message-metrics {
      margin-top: 9px;
      padding-top: 7px;
      border-top: 1px solid var(--line);
      color: var(--muted);
      font-size: 11px;
      font-variant-numeric: tabular-nums;
    }
    .harness-message-notice {
      margin-top: 9px;
      padding: 7px 9px;
      border-left: 3px solid var(--warning);
      background: #fff7ed;
      color: #9a3412;
      font-size: 12px;
      line-height: 1.45;
    }
    .harness-reasoning {
      margin: 0 0 9px;
      padding: 8px 9px;
      border-left: 3px solid #cbd5e1;
      background: #f8fafc;
      color: #526174;
      font-size: 12px;
    }
    .harness-reasoning summary {
      cursor: pointer;
      font-weight: 700;
    }
    .harness-reasoning div {
      margin-top: 7px;
      line-height: 1.5;
      white-space: pre-wrap;
    }
    .harness-composer {
      margin-top: 10px;
    }
    .harness-composer label {
      margin-top: 0;
    }
    .harness-actions {
      display: grid;
      grid-template-columns: 1fr auto auto;
      gap: 8px;
      align-items: end;
    }
    .harness-actions button {
      width: auto;
      min-width: 112px;
      margin-top: 10px;
      padding: 0 16px;
    }
    .harness-status-row {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      min-height: 20px;
      margin-top: 8px;
      color: var(--muted);
      font-size: 12px;
    }
    .harness-status-row .error {
      min-height: 0;
      margin: 0;
      text-align: right;
    }
    .settings-section {
      min-height: 560px;
    }
    .settings-layout {
      display: grid;
      grid-template-columns: minmax(0, 1.05fr) minmax(320px, .95fr);
      gap: 14px;
    }
    .settings-panel {
      min-width: 0;
      padding: 16px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfd;
    }
    .settings-panel h3 {
      margin: 0 0 4px;
      font-size: 15px;
    }
    .scope-options {
      display: grid;
      gap: 8px;
      margin-top: 14px;
    }
    .scope-option {
      display: grid;
      grid-template-columns: auto 1fr;
      gap: 10px;
      align-items: start;
      margin: 0;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #fff;
      color: var(--text);
      font-size: 13px;
      text-transform: none;
      cursor: pointer;
    }
    .scope-option:has(input:checked) {
      border-color: #8bc8c1;
      background: #f0fdfa;
    }
    .scope-option input {
      width: 16px;
      height: 16px;
      margin-top: 2px;
    }
    .scope-option strong,
    .scope-option small {
      display: block;
    }
    .scope-option small {
      margin-top: 3px;
      color: var(--muted);
      line-height: 1.45;
    }
    .settings-status {
      margin: 14px 0 0;
      padding: 11px 12px;
      border: 1px solid #b8ded9;
      border-radius: 7px;
      background: #ecfdf9;
      color: var(--accent-dark);
      font-size: 12px;
      line-height: 1.5;
    }
    .settings-warning {
      border-color: #efc48f;
      background: #fff8eb;
      color: #854d0e;
    }
    .key-row {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
    }
    .key-row button {
      width: auto;
      min-width: 112px;
      height: 38px;
      margin: 0;
      padding: 0 12px;
    }
    .settings-actions {
      display: flex;
      justify-content: flex-end;
      gap: 8px;
      margin-top: 16px;
    }
    .settings-actions button {
      width: auto;
      min-width: 150px;
      margin: 0;
      padding: 0 16px;
    }
    .service-control-panel {
      margin-top: 14px;
    }
    .service-control-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-top: 12px;
    }
    .service-control {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #fff;
    }
    .service-control strong {
      display: block;
      font-size: 13px;
    }
    .service-control small {
      display: block;
      margin-top: 3px;
      color: var(--muted);
      font-size: 11px;
      line-height: 1.4;
    }
    .service-control button {
      flex: 0 0 auto;
      width: auto;
      min-width: 152px;
      margin: 0;
      padding: 0 14px;
    }
    .auth-gate {
      position: fixed;
      z-index: 100;
      inset: 0;
      display: grid;
      place-items: center;
      padding: 18px;
      background: rgb(15 23 42 / 56%);
      backdrop-filter: blur(4px);
    }
    .auth-gate[hidden] {
      display: none;
    }
    .auth-card {
      width: min(440px, 100%);
      padding: 22px;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: var(--panel);
      box-shadow: 0 18px 50px rgb(15 23 42 / 24%);
    }
    .auth-card h2 {
      margin-bottom: 5px;
      font-size: 20px;
    }
    .auth-card button {
      margin-top: 12px;
    }
    .grid {
      display: grid;
      grid-template-columns: minmax(320px, 420px) 1fr;
      gap: 16px;
      align-items: start;
    }
    section {
      padding: 16px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
    }
    label {
      display: block;
      margin: 12px 0 6px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
      text-transform: uppercase;
    }
    select,
    input,
    textarea {
      width: 100%;
      padding: 7px 9px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--text);
      font: inherit;
    }
    select,
    input {
      height: 38px;
    }
    textarea {
      min-height: 92px;
      line-height: 1.45;
      resize: vertical;
    }
    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    .check {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-top: 12px;
      color: var(--text);
      font-size: 14px;
    }
    .check input {
      width: 16px;
      height: 16px;
    }
    button {
      width: 100%;
      height: 40px;
      margin-top: 16px;
      border: 0;
      border-radius: 6px;
      background: var(--accent);
      color: #fff;
      font-weight: 700;
      cursor: pointer;
    }
    button:hover { background: var(--accent-dark); }
    button:disabled {
      cursor: not-allowed;
      opacity: .6;
    }
    button.secondary {
      background: #526174;
    }
    button.secondary:hover {
      background: #3d4b5c;
    }
    button.danger {
      background: var(--danger);
    }
    button.danger:hover {
      background: #8f1f17;
    }
    .button-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    .table-wrap {
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    th,
    td {
      padding: 9px 10px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      white-space: nowrap;
      vertical-align: top;
    }
    th {
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      background: #f8fafc;
    }
    tr:last-child td { border-bottom: 0; }
    .number {
      text-align: right;
      font-variant-numeric: tabular-nums;
    }
    .bar {
      display: grid;
      grid-template-columns: 130px 1fr 64px;
      gap: 10px;
      align-items: center;
      margin: 8px 0;
      font-size: 13px;
    }
    .bar-track {
      height: 10px;
      border-radius: 999px;
      background: #e5e9f0;
      overflow: hidden;
    }
    .bar-fill {
      height: 100%;
      background: var(--accent);
    }
    pre {
      min-height: 230px;
      max-height: 420px;
      overflow: auto;
      margin: 0;
      padding: 12px;
      border-radius: 8px;
      background: var(--code);
      color: #f8fafc;
      font-size: 12px;
      line-height: 1.45;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
    .mini-pre {
      min-height: 104px;
      max-height: 190px;
      margin-top: 12px;
    }
    a {
      color: var(--accent);
      text-decoration: none;
      font-weight: 650;
    }
    a:hover { text-decoration: underline; }
    .stack {
      display: grid;
      gap: 16px;
    }
    .error {
      color: var(--danger);
      min-height: 18px;
      margin-top: 10px;
      font-size: 13px;
    }
    @media (max-width: 860px) {
      main {
        width: calc(100% - 20px);
        max-width: 760px;
        margin: 14px auto 40px;
      }
      header { align-items: stretch; flex-direction: column; }
      .status { text-align: left; }
      .grid { grid-template-columns: 1fr; }
      .benchmark-workspace { grid-template-columns: 1fr; }
      .metric-grid { grid-template-columns: 1fr 1fr; }
      .metric:first-child { grid-column: 1 / -1; }
      .control-grid { grid-template-columns: 1fr; }
      .safety-grid { grid-template-columns: 1fr; }
      .statistics-grid { grid-template-columns: 1fr; }
      .statistics-speed-bars .bar { grid-template-columns: 1fr; gap: 4px; }
      .telemetry-summary { grid-template-columns: repeat(2, 1fr); }
      .telemetry-grid { grid-template-columns: 1fr; }
      .telemetry-table table { min-width: 720px; table-layout: auto; }
      .telemetry-table .telemetry-model { min-width: 220px; }
      .telemetry-failure-summary { grid-template-columns: 1fr; }
      .harness-layout { grid-template-columns: 1fr; }
      .settings-layout { grid-template-columns: 1fr; }
      .service-control-grid { grid-template-columns: 1fr; }
      .harness-settings textarea { min-height: 100px; }
      .row { grid-template-columns: 1fr; }
      .bar { grid-template-columns: 1fr; gap: 4px; }
      th, td { white-space: normal; }
    }
    @media (max-width: 520px) {
      .metric-grid { grid-template-columns: 1fr; }
      .benchmark-progress-metrics { grid-template-columns: 1fr; }
      #logOutput {
        height: 340px;
        min-height: 340px;
      }
      .metric:first-child { grid-column: auto; }
      .telemetry-summary { grid-template-columns: 1fr; }
      .telemetry-toolbar { align-items: stretch; flex-direction: column; }
      .telemetry-toolbar select { min-width: 0; }
      .agent-loop-alert-heading {
        align-items: flex-start;
        flex-wrap: wrap;
      }
      .agent-loop-alert-heading strong {
        width: 100%;
        white-space: normal;
      }
      .telemetry-failure-actions {
        align-items: flex-end;
        flex-direction: column;
      }
      .telemetry-query-heading-meta {
        justify-items: start;
        text-align: left;
      }
      .telemetry-query-heading {
        align-items: stretch;
        flex-direction: column;
      }
      .telemetry-pagination {
        align-items: stretch;
        flex-direction: column;
      }
      .telemetry-pagination-controls {
        display: grid;
        grid-template-columns: 1fr auto auto;
        justify-content: initial;
      }
      .telemetry-pagination-controls label {
        grid-column: 1;
      }
      .telemetry-pagination-controls select {
        grid-column: 2 / 4;
        justify-self: end;
      }
      #telemetryQueryPrevious {
        grid-column: 1;
      }
      #telemetryQueryPageStatus {
        grid-column: 2;
      }
      #telemetryQueryNext {
        grid-column: 3;
      }
      .control-actions { grid-template-columns: 1fr; }
      .field-grid { grid-template-columns: 1fr; }
      .overview-heading { align-items: stretch; flex-direction: column; }
      .harness-heading { align-items: stretch; flex-direction: column; }
      .model-pill { max-width: none; }
      .harness-transcript { min-height: 280px; padding: 10px; }
      .harness-empty { min-height: 250px; }
      .harness-message { width: 96%; }
      .harness-actions { grid-template-columns: 1fr 1fr; }
      .harness-actions button:first-child { grid-column: 1 / -1; width: 100%; }
      .harness-actions button { width: 100%; min-width: 0; }
      .harness-status-row { align-items: flex-start; flex-direction: column; }
      .harness-status-row .error { text-align: left; }
      .request-meta { align-items: stretch; flex-direction: column; }
      .request-meta button { width: 100%; }
      .key-row { grid-template-columns: 1fr; }
      .key-row button { width: 100%; }
      .settings-actions { align-items: stretch; flex-direction: column; }
      .settings-actions button { width: 100%; }
      .service-control {
        align-items: stretch;
        flex-direction: column;
      }
      .service-control button { width: 100%; }
      button.compact { width: 100%; }
      .tab-button {
        flex: 1;
        min-width: 0;
        height: 44px;
        padding: 0 6px;
        font-size: 11px;
        line-height: 1.15;
      }
    }
  </style>
</head>
<body>
  <div id="authGate" class="auth-gate" role="dialog" aria-modal="true" hidden
    aria-labelledby="authGateTitle">
    <form id="authForm" class="auth-card">
      <p class="eyebrow">Protected control plane</p>
      <h2 id="authGateTitle">API key required</h2>
      <p class="muted">Enter the key shown in the terminal that started MLX Control Center.</p>
      <label for="authApiKey">API key</label>
      <input id="authApiKey" type="password" autocomplete="current-password"
        placeholder="Paste your Control Center API key">
      <button type="submit">Unlock Control Center</button>
      <div id="authErrorText" class="error" aria-live="polite"></div>
    </form>
  </div>
  <main inert>
    <header>
      <div>
        <p class="eyebrow">Local Apple Silicon operations</p>
        <h1>MLX Control Center</h1>
      </div>
      <div class="status" id="healthCard" data-state="healthy">
        <span>Server</span>
        <strong id="serverHealth">Checking…</strong>
        <small id="lastUpdated">Waiting for telemetry</small>
      </div>
    </header>

    <nav class="tab-list" role="tablist" aria-label="Control Center views">
      <button id="operationsTabButton" class="tab-button active" type="button"
        role="tab" aria-selected="true" aria-controls="operationsTab">Operations</button>
      <button id="telemetryTabButton" class="tab-button" type="button"
        role="tab" aria-selected="false" aria-controls="telemetryTab">Telemetry</button>
      <button id="benchmarkTabButton" class="tab-button" type="button"
        role="tab" aria-selected="false" aria-controls="benchmarkTab">Benchmarks</button>
      <button id="harnessTabButton" class="tab-button" type="button"
        role="tab" aria-selected="false" aria-controls="harnessTab">Model Harness</button>
      <button id="settingsTabButton" class="tab-button" type="button"
        role="tab" aria-selected="false" aria-controls="settingsTab">Settings</button>
    </nav>

    <div id="operationsTab" class="tab-panel" role="tabpanel"
      aria-labelledby="operationsTabButton">
    <section class="overview">
      <div class="overview-heading">
        <div>
          <h2>Runtime overview</h2>
          <p class="muted">Live model, unified memory, KV cache, and request activity.</p>
        </div>
        <button id="refreshServerButton" class="secondary compact" type="button">Refresh</button>
      </div>

      <div class="metric-grid">
        <div class="metric">
          <span>Runtime model</span>
          <strong id="loadedModelMetric">—</strong>
          <small id="modelRoleMetric">No model loaded</small>
        </div>
        <div class="metric">
          <span>System memory</span>
          <strong id="memoryMetric">—</strong>
          <small id="availableMetric">— available</small>
        </div>
        <div class="metric">
          <span>Prompt cache</span>
          <strong id="cacheMetric">—</strong>
          <small id="cacheDetailMetric">—</small>
        </div>
        <div class="metric">
          <span>Swap used</span>
          <strong id="swapMetric">—</strong>
          <small>macOS unified memory</small>
        </div>
        <div class="metric">
          <span>Server process</span>
          <strong id="serverPidMetric">—</strong>
          <small id="processMetric">—</small>
        </div>
      </div>

      <div class="memory-panel">
        <div class="memory-labels">
          <strong id="memoryLabel">System memory</strong>
          <span id="memoryPercent">—</span>
        </div>
        <div class="memory-track"><div id="memoryFill" class="memory-fill"></div></div>
      </div>

      <div id="requestPanel" class="activity-panel" hidden>
        <div class="request-meta">
          <div>
            <strong id="requestTitle">Active request</strong>
            <small id="requestId">Waiting for request details</small>
          </div>
          <button id="cancelRequestButton" class="danger" type="button">Cancel request</button>
        </div>
        <div class="request-labels">
          <span>Progress</span>
          <span id="requestText">—</span>
        </div>
        <div class="request-track"><div id="requestFill" class="request-fill"></div></div>
      </div>

      <div class="control-grid">
        <div class="control-panel">
          <label for="serverModelSelect">Managed model</label>
          <select id="serverModelSelect"></select>
          <div class="control-actions">
            <button id="loadServerButton" type="button">Load / Switch</button>
            <button id="restartServerButton" class="secondary" type="button">Restart</button>
            <button id="unloadServerButton" class="danger" type="button">Unload</button>
          </div>
          <div class="error" id="serverErrorText"></div>
        </div>
        <div class="control-panel">
          <label>Runtime details</label>
          <pre id="serverStatusOutput" class="mini-pre"></pre>
        </div>
      </div>

      <div class="safety-grid">
        <div class="control-panel">
          <div class="panel-heading">
            <div>
              <h3>Safety guard</h3>
              <p class="muted">Admission limits and automatic low-memory cancellation.</p>
            </div>
          </div>
          <label class="check">
            <input id="guardEnabled" type="checkbox" checked>
            Memory guard enabled
          </label>
          <div class="field-grid">
            <div>
              <label for="contextCap">Context cap</label>
              <input id="contextCap" type="number" min="1024" max="1048576" step="1024">
            </div>
            <div>
              <label for="outputCap">Output cap</label>
              <input id="outputCap" type="number" min="1" max="131072" step="256">
            </div>
            <div>
              <label for="reserveGib">Admission reserve · GiB</label>
              <input id="reserveGib" type="number" min="4" max="128" step="1">
            </div>
            <div>
              <label for="abortGib">Emergency cancel · GiB</label>
              <input id="abortGib" type="number" min="2" max="127" step="1">
            </div>
          </div>
          <button id="saveSafetyButton" type="button">Save safety limits</button>
          <p id="safetySummary" class="safety-summary">Safety telemetry will appear after the server is ready.</p>
        </div>

        <div class="control-panel cache-panel">
          <div class="panel-heading">
            <div>
              <h3>Retained prompt cache</h3>
              <p class="muted">Inspect or release reusable KV-cache entries.</p>
            </div>
            <button id="clearCacheButton" class="danger inline-button" type="button">Clear all</button>
          </div>
          <div class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Type</th>
                  <th class="number">Tokens</th>
                  <th class="number">Memory</th>
                  <th></th>
                </tr>
              </thead>
              <tbody id="cacheRows">
                <tr><td class="empty-row" colspan="4">No retained cache entries</td></tr>
              </tbody>
            </table>
          </div>
          <div class="error" id="safetyErrorText"></div>
        </div>
      </div>
    </section>
    </div>

    <section id="telemetryTab" class="telemetry-section tab-panel" role="tabpanel"
      aria-labelledby="telemetryTabButton" hidden>
      <div class="overview-heading">
        <div>
          <h2>Server telemetry</h2>
          <p class="muted">Request volume, token usage, model activity, agent-loop risk, clients, tools, and MCP-attributed calls.</p>
        </div>
        <div class="telemetry-toolbar">
          <label for="telemetryWindow">Time range</label>
          <select id="telemetryWindow">
            <option value="3600">Last hour</option>
            <option value="86400" selected>Last 24 hours</option>
            <option value="604800">Last 7 days</option>
            <option value="2592000">Last 30 days</option>
            <option value="0">All retained</option>
          </select>
          <button id="refreshTelemetryButton" class="secondary compact" type="button">Refresh</button>
        </div>
      </div>

      <div id="telemetryActive" class="telemetry-active" hidden></div>
      <div id="telemetryLoopAlerts" class="agent-loop-alerts" aria-live="polite" hidden></div>

      <div class="metric-grid telemetry-summary">
        <div class="metric">
          <span>Requests</span>
          <strong id="telemetryRequests">—</strong>
          <small id="telemetryRequestDetail">— active</small>
        </div>
        <div class="metric">
          <span>Input tokens</span>
          <strong id="telemetryInputTokens">—</strong>
          <small id="telemetryCacheRate">— cached</small>
        </div>
        <div class="metric">
          <span>Output tokens</span>
          <strong id="telemetryOutputTokens">—</strong>
          <small>Generated tokens</small>
        </div>
        <div class="metric">
          <span>Total tokens</span>
          <strong id="telemetryTotalTokens">—</strong>
          <small id="telemetryThroughput">— tok/s</small>
        </div>
        <div class="metric">
          <span>Tool calls</span>
          <strong id="telemetryToolCalls">—</strong>
          <small>Model invocations</small>
        </div>
        <div class="metric">
          <span>MCP calls</span>
          <strong id="telemetryMcpCalls">—</strong>
          <small>Namespaced tools</small>
        </div>
        <div class="metric">
          <span>Average latency</span>
          <strong id="telemetryLatency">—</strong>
          <small id="telemetryErrorRate">— errors</small>
        </div>
      </div>

      <div class="telemetry-grid">
        <div class="telemetry-panel">
          <h3>Token activity</h3>
          <p class="muted">Input plus output tokens per time bucket. Select an amber bucket to inspect its errors and cancellations.</p>
          <div id="telemetryChart" class="telemetry-chart" aria-label="Token activity chart"></div>
          <div id="telemetryLegend" class="telemetry-legend"></div>
        </div>
        <div class="telemetry-panel">
          <h3>Client activity</h3>
          <p class="muted">Which local clients are using the MLX endpoint.</p>
          <div id="telemetrySources"></div>
          <p id="telemetryRuntimeNote" class="telemetry-note">Waiting for telemetry.</p>
        </div>
      </div>

      <div id="telemetryFailurePanel" class="telemetry-panel telemetry-failure-panel">
        <div class="panel-heading">
          <div>
            <h3>Errors &amp; cancellations</h3>
            <p class="muted">Failure type, originating client, API endpoint, model, and retained error message.</p>
          </div>
          <div class="telemetry-failure-actions">
            <span id="telemetryFailureCount" class="telemetry-failure-count">Waiting for telemetry</span>
            <button id="telemetryFailureClearFilter" class="secondary inline-button" type="button" hidden>Show all failures</button>
          </div>
        </div>
        <div id="telemetryFailureSummary" class="telemetry-failure-summary"></div>
        <p id="telemetryFailureFilter" class="telemetry-failure-filter" hidden></p>
        <div class="table-wrap telemetry-failure-table">
          <table>
            <thead>
              <tr>
                <th>Time</th>
                <th>Type</th>
                <th>Source</th>
                <th>Endpoint</th>
                <th>Model</th>
                <th>Details</th>
              </tr>
            </thead>
            <tbody id="telemetryFailureRows">
              <tr><td class="empty-row" colspan="6">No errors or cancellations in this time range.</td></tr>
            </tbody>
          </table>
        </div>
      </div>

      <div class="telemetry-grid">
        <div class="telemetry-panel">
          <h3>Model activity</h3>
          <p class="muted">Usage grouped by the requested resident model.</p>
          <div class="table-wrap telemetry-table">
            <table>
              <thead>
                <tr>
                  <th>Model</th>
                  <th class="number">Requests</th>
                  <th class="number">Input</th>
                  <th class="number">Output</th>
                  <th class="number">Tools</th>
                  <th class="number">Failures</th>
                  <th class="number">Avg latency</th>
                </tr>
              </thead>
              <tbody id="telemetryModelRows" class="telemetry-collapsible">
                <tr><td class="empty-row" colspan="7">No model activity yet.</td></tr>
              </tbody>
            </table>
          </div>
          <div class="telemetry-toggle-row">
            <button id="telemetryModelToggle" class="secondary inline-button" type="button"
              aria-controls="telemetryModelRows" aria-expanded="false" hidden>Show all models</button>
          </div>
        </div>
        <div class="telemetry-panel">
          <h3>Tool and MCP activity</h3>
          <p class="muted">Tools offered to models and structured calls they emitted.</p>
          <div class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Tool</th>
                  <th class="number">Offered</th>
                  <th class="number">Calls</th>
                </tr>
              </thead>
              <tbody id="telemetryToolRows" class="telemetry-collapsible">
                <tr><td class="empty-row" colspan="3">No tool activity yet.</td></tr>
              </tbody>
            </table>
          </div>
          <div class="telemetry-toggle-row">
            <button id="telemetryToolToggle" class="secondary inline-button" type="button"
              aria-controls="telemetryToolRows" aria-expanded="false" hidden>Show all tools</button>
          </div>
          <p class="telemetry-note">MCP is attributed when the client preserves an <code>mcp__server__tool</code> or <code>mcp_*</code> name. Flattened tool names remain visible but cannot be identified as MCP reliably.</p>
        </div>
      </div>

      <div class="telemetry-panel">
        <div class="panel-heading telemetry-query-heading">
          <div>
            <h3>Recent queries</h3>
            <p class="muted">Short previews only; full conversations are not written to telemetry.</p>
          </div>
          <div class="telemetry-query-heading-meta">
            <span id="telemetryRetentionPolicy" class="telemetry-retention-policy">30-day retention · daily archive · automatic deletion</span>
            <span id="telemetryUpdatedAt" class="muted">Waiting for telemetry</span>
          </div>
        </div>
        <div class="table-wrap telemetry-query-table">
          <table>
            <thead>
              <tr>
                <th>Time</th>
                <th>Client</th>
                <th>Model</th>
                <th>Query preview</th>
                <th class="number">Input</th>
                <th class="number">Cached</th>
                <th class="number">Output</th>
                <th>Tools called</th>
                <th class="number">Latency</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody id="telemetryRequestRows">
              <tr><td class="empty-row" colspan="10">No requests in this time range.</td></tr>
            </tbody>
          </table>
        </div>
        <div class="telemetry-pagination" aria-label="Recent query pages">
          <span id="telemetryQueryCount" class="muted">0 queries</span>
          <div class="telemetry-pagination-controls">
            <label for="telemetryQueryPageSize">Rows per page</label>
            <select id="telemetryQueryPageSize">
              <option value="10">10</option>
              <option value="25" selected>25</option>
              <option value="50">50</option>
              <option value="100">100</option>
            </select>
            <button id="telemetryQueryPrevious" class="secondary inline-button" type="button" disabled>Previous</button>
            <span id="telemetryQueryPageStatus" class="muted telemetry-page-status" aria-live="polite">Page 1 of 1</span>
            <button id="telemetryQueryNext" class="secondary inline-button" type="button" disabled>Next</button>
          </div>
        </div>
        <div id="telemetryErrorText" class="error"></div>
      </div>
    </section>

    <section id="benchmarkTab" class="benchmark-section tab-panel" role="tabpanel"
      aria-labelledby="benchmarkTabButton" hidden>
    <div class="overview-heading benchmark-heading">
      <div>
        <h2>Model benchmarks</h2>
        <p class="muted">Run repeatable prompt sets, inspect results, and compare every configured model.</p>
      </div>
    </div>
    <div class="benchmark-workspace">
      <div class="stack">
        <section>
          <h2>Run benchmark · <span id="jobState">Idle</span></h2>
          <form id="benchmarkForm">
            <label for="promptSelect">Prompt Set</label>
            <select id="promptSelect"></select>

            <label class="check">
              <input id="allModels" type="checkbox" checked>
              All downloaded models
            </label>

            <label for="modelSelect">Benchmark target</label>
            <select id="modelSelect"></select>

            <div class="row">
              <div>
                <label for="maxTokens">Max Tokens</label>
                <input id="maxTokens" type="number" min="1" max="20000" value="1200">
              </div>
              <div>
                <label for="limit">Limit</label>
                <input id="limit" type="number" min="1" placeholder="all">
              </div>
            </div>

            <label for="outputDir">Output Directory</label>
            <input id="outputDir" value="control-plane-outputs">

            <button id="startButton" type="submit">Start Benchmark</button>
            <div class="error" id="errorText"></div>
          </form>
        </section>

        <section class="latest-comparison-panel">
          <h2>Latest Comparison</h2>
          <div id="bars" class="latest-comparison-bars"></div>
        </section>
      </div>

      <div class="stack benchmark-live-column">
        <section class="benchmark-progress-panel">
          <div class="benchmark-panel-heading">
            <div>
              <h2>Benchmark progress</h2>
              <p id="benchmarkProgressStage" class="muted">Waiting to start</p>
            </div>
            <span id="benchmarkProgressStatus" class="benchmark-status-badge"
              data-state="idle">Idle</span>
          </div>
          <div class="benchmark-progress-summary">
            <strong id="benchmarkProgressPercent">0%</strong>
            <span id="benchmarkProgressSteps">0 / 0 prompts complete</span>
          </div>
          <div id="benchmarkProgressTrack" class="benchmark-progress-track"
            role="progressbar" aria-label="Benchmark completion"
            aria-valuemin="0" aria-valuemax="100" aria-valuenow="0">
            <div id="benchmarkProgressFill" class="benchmark-progress-fill"></div>
          </div>
          <div class="benchmark-progress-metrics">
            <div class="benchmark-progress-metric">
              <span>Model</span>
              <strong id="benchmarkProgressModelCount">— / —</strong>
              <small id="benchmarkProgressModel">Waiting for benchmark</small>
            </div>
            <div class="benchmark-progress-metric">
              <span>Prompt</span>
              <strong id="benchmarkProgressPromptCount">— / —</strong>
              <small id="benchmarkProgressPrompt">Waiting for prompt</small>
            </div>
            <div class="benchmark-progress-metric">
              <span>Elapsed</span>
              <strong id="benchmarkProgressElapsed">0s</strong>
              <small>Live runtime</small>
            </div>
          </div>
          <p id="benchmarkProgressActivity" class="benchmark-progress-activity">
            Start a benchmark to see live progress.
          </p>
        </section>

        <section class="benchmark-console-panel">
          <div class="console-heading">
            <h2>Live console</h2>
            <div class="console-tools">
              <span id="consoleLineCount">0 lines</span>
              <label class="console-follow">
                <input id="followConsoleOutput" type="checkbox" checked>
                Follow output
              </label>
            </div>
          </div>
          <pre id="logOutput" aria-label="Benchmark console output"></pre>
        </section>

        <section>
          <h2>Recent runs</h2>
          <div class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Run</th>
                  <th class="number">Records</th>
                  <th>Report</th>
                  <th>JSONL</th>
                </tr>
              </thead>
              <tbody id="runsBody"></tbody>
            </table>
          </div>
        </section>
      </div>
    </div>
    <section class="statistics-section">
      <div class="overview-heading">
        <div>
          <h2>Benchmark statistics</h2>
          <p class="muted">Registry coverage and performance aggregated from every benchmark JSONL run.</p>
        </div>
        <button id="refreshStatisticsButton" class="secondary compact" type="button">Refresh</button>
      </div>

      <div class="metric-grid statistics-summary">
        <div class="metric">
          <span>Configured models</span>
          <strong id="statisticsTotalModels">—</strong>
          <small id="statisticsDownloadedDetail">— downloaded</small>
        </div>
        <div class="metric">
          <span>Benchmarked</span>
          <strong id="statisticsBenchmarkedModels">—</strong>
          <small id="statisticsCoveragePercent">— coverage</small>
        </div>
        <div class="metric">
          <span>Prompt samples</span>
          <strong id="statisticsSamples">—</strong>
          <small>Across all saved runs</small>
        </div>
        <div class="metric">
          <span>Fastest average</span>
          <strong id="statisticsFastestRate">—</strong>
          <small id="statisticsFastestModel">No benchmark data</small>
        </div>
        <div class="metric">
          <span>Latest benchmark</span>
          <strong id="statisticsLatestDate">—</strong>
          <small id="statisticsUpdatedAt">Waiting for statistics</small>
        </div>
      </div>

      <div class="statistics-grid">
        <div class="statistics-panel">
          <h3>Average generation speed</h3>
          <p class="muted">Tokens per second across all recorded prompts for each model.</p>
          <div id="statisticsSpeedBars" class="statistics-speed-bars"></div>
        </div>
        <div class="statistics-panel">
          <h3>Coverage</h3>
          <p id="statisticsCoverageText" class="statistics-note">
            Statistics are collected from saved benchmark runs.
          </p>
          <p class="statistics-note">
            A dash means that model has no recorded value yet. Run it from the
            benchmark controls above to add samples.
          </p>
          <div id="statisticsErrorText" class="error"></div>
        </div>
      </div>

      <div class="table-wrap statistics-table">
        <table>
          <thead>
            <tr>
              <th>Model</th>
              <th>Status</th>
              <th class="number">Size</th>
              <th class="number">Samples</th>
              <th class="number">Runs</th>
              <th class="number">Avg prompt tok/s</th>
              <th class="number">Avg gen tok/s</th>
              <th class="number">Best gen tok/s</th>
              <th class="number">Peak memory</th>
              <th>Last benchmark</th>
            </tr>
          </thead>
          <tbody id="statisticsRows">
            <tr><td class="empty-row" colspan="10">Open this tab to load model statistics.</td></tr>
          </tbody>
        </table>
      </div>
    </section>
    </section>

    <section id="harnessTab" class="harness-section tab-panel" role="tabpanel"
      aria-labelledby="harnessTabButton" hidden>
      <div class="harness-heading">
        <div>
          <h2>Model harness</h2>
          <p class="muted">Chat with the active model using multi-turn history and streamed output.</p>
        </div>
        <div id="harnessModelName" class="model-pill">No model loaded</div>
      </div>

      <div class="harness-layout">
        <aside class="harness-settings">
          <label for="harnessSystemPrompt">System prompt</label>
          <textarea id="harnessSystemPrompt">You are a helpful local assistant. Answer directly and clearly.</textarea>

          <div class="row">
            <div>
              <label for="harnessMaxTokens">Max tokens</label>
              <input id="harnessMaxTokens" type="number" min="1" max="32000" value="1200">
            </div>
            <div>
              <label for="harnessTemperature">Temperature</label>
              <input id="harnessTemperature" type="number" min="0" max="2" step="0.1" value="0.2">
            </div>
          </div>
          <p class="safety-summary">Conversation history stays in this browser tab. Generation uses the active server safety limits.</p>
        </aside>

        <div class="harness-workspace">
          <div id="harnessTranscript" class="harness-transcript" aria-live="polite">
            <div class="harness-empty">Send a message to begin a local conversation.</div>
          </div>

          <form id="harnessForm" class="harness-composer">
            <label for="harnessInput">Message</label>
            <textarea id="harnessInput" aria-describedby="harnessInputHelp"
              placeholder="Ask the active model anything…"></textarea>
            <p id="harnessInputHelp" class="safety-summary">Enter to send · Shift+Enter for a new line</p>
            <div class="harness-actions">
              <button id="harnessSendButton" type="submit">Send message</button>
              <button id="harnessCancelButton" class="danger" type="button" disabled>Cancel</button>
              <button id="harnessClearButton" class="secondary" type="button">Clear chat</button>
            </div>
          </form>
          <div class="harness-status-row">
            <span id="harnessStatus">Waiting for an active model</span>
            <span id="harnessErrorText" class="error"></span>
          </div>
        </div>
      </div>
    </section>

    <section id="settingsTab" class="settings-section tab-panel" role="tabpanel"
      aria-labelledby="settingsTabButton" hidden>
      <div class="overview-heading">
        <div>
          <h2>Control Center settings</h2>
          <p class="muted">Control access to this interface and the MLX model API with one shared key.</p>
        </div>
        <button id="lockControlCenterButton" class="secondary compact" type="button">
          Lock this tab
        </button>
      </div>

      <form id="accessSettingsForm">
        <div class="settings-layout">
          <div class="settings-panel">
            <h3>Access scope</h3>
            <p class="muted">Saved access changes take effect the next time Control Center starts.</p>
            <div class="scope-options">
              <label class="scope-option">
                <input id="localAccessMode" type="radio" name="accessMode" value="local">
                <span>
                  <strong>Local only</strong>
                  <small>Bind to 127.0.0.1. Only applications on this Mac can connect.</small>
                </span>
              </label>
              <label class="scope-option">
                <input id="networkAccessMode" type="radio" name="accessMode" value="network">
                <span>
                  <strong>Local network</strong>
                  <small>Bind to all interfaces. Other devices on trusted networks can connect.</small>
                </span>
              </label>
            </div>

            <label for="controlCenterPort">Port</label>
            <input id="controlCenterPort" type="number" min="1" max="65535" step="1">
            <p id="accessEndpointSummary" class="settings-status">Loading access settings…</p>
          </div>

          <div class="settings-panel">
            <h3>Authentication</h3>
            <p class="muted">Optional for local access and mandatory for network access.</p>
            <label class="check">
              <input id="apiKeyAuthentication" type="checkbox">
              Require an API key for every Control Center request
            </label>
            <label class="check">
              <input id="modelApiKeyAuthentication" type="checkbox">
              Require the same API key for every Model API request on port 48080
            </label>

            <label for="replacementApiKey">Replacement API key</label>
            <div class="key-row">
              <input id="replacementApiKey" type="password" minlength="24"
                autocomplete="new-password" placeholder="Leave blank to keep the current key">
              <button id="generateApiKeyButton" class="secondary" type="button">Generate</button>
            </div>
            <p class="safety-summary">
              The same key protects both services when their checkboxes are enabled. A
              replacement must be at least 24 characters. Copy generated keys before saving;
              saved keys are never returned by the API.
            </p>
            <div id="settingsErrorText" class="error" aria-live="polite"></div>
          </div>
        </div>
        <p id="settingsRestartNotice" class="settings-status settings-warning" hidden>
          Restart the affected service to apply saved access settings.
        </p>
        <div class="settings-actions">
          <button id="saveAccessSettingsButton" type="submit">Save access settings</button>
        </div>
      </form>

      <div class="settings-panel service-control-panel">
        <h3>Service controls</h3>
        <p class="muted">Restart either service after configuration changes or to recover a stale process.</p>
        <div class="service-control-grid">
          <div class="service-control">
            <div>
              <strong>Control Center</strong>
              <small>Briefly reconnects this page and applies saved bind, port, and authentication settings.</small>
            </div>
            <button id="restartControlCenterButton" class="secondary" type="button">Restart Control Center</button>
          </div>
          <div class="service-control">
            <div>
              <strong>MLX model server</strong>
              <small>Reloads the selected model and clears active requests and retained prompt cache.</small>
            </div>
            <button id="restartMlxServerButton" class="secondary" type="button">Restart MLX Server</button>
          </div>
        </div>
        <p id="serviceRestartStatus" class="settings-status" aria-live="polite" hidden></p>
      </div>
    </section>
  </main>

  <script>
    const form = document.querySelector("#benchmarkForm");
    const promptSelect = document.querySelector("#promptSelect");
    const modelSelect = document.querySelector("#modelSelect");
    const allModels = document.querySelector("#allModels");
    const maxTokens = document.querySelector("#maxTokens");
    const limit = document.querySelector("#limit");
    const outputDir = document.querySelector("#outputDir");
    const errorText = document.querySelector("#errorText");
    const serverErrorText = document.querySelector("#serverErrorText");
    const jobState = document.querySelector("#jobState");
    const startButton = document.querySelector("#startButton");
    const benchmarkProgressStage = document.querySelector("#benchmarkProgressStage");
    const benchmarkProgressStatus = document.querySelector("#benchmarkProgressStatus");
    const benchmarkProgressPercent = document.querySelector("#benchmarkProgressPercent");
    const benchmarkProgressSteps = document.querySelector("#benchmarkProgressSteps");
    const benchmarkProgressTrack = document.querySelector("#benchmarkProgressTrack");
    const benchmarkProgressFill = document.querySelector("#benchmarkProgressFill");
    const benchmarkProgressModelCount = document.querySelector("#benchmarkProgressModelCount");
    const benchmarkProgressModel = document.querySelector("#benchmarkProgressModel");
    const benchmarkProgressPromptCount = document.querySelector("#benchmarkProgressPromptCount");
    const benchmarkProgressPrompt = document.querySelector("#benchmarkProgressPrompt");
    const benchmarkProgressElapsed = document.querySelector("#benchmarkProgressElapsed");
    const benchmarkProgressActivity = document.querySelector("#benchmarkProgressActivity");
    const consoleLineCount = document.querySelector("#consoleLineCount");
    const followConsoleOutput = document.querySelector("#followConsoleOutput");
    const serverModelSelect = document.querySelector("#serverModelSelect");
    const loadServerButton = document.querySelector("#loadServerButton");
    const restartServerButton = document.querySelector("#restartServerButton");
    const unloadServerButton = document.querySelector("#unloadServerButton");
    const refreshServerButton = document.querySelector("#refreshServerButton");
    const serverStatusOutput = document.querySelector("#serverStatusOutput");
    const healthCard = document.querySelector("#healthCard");
    const serverHealth = document.querySelector("#serverHealth");
    const lastUpdated = document.querySelector("#lastUpdated");
    const loadedModelMetric = document.querySelector("#loadedModelMetric");
    const modelRoleMetric = document.querySelector("#modelRoleMetric");
    const memoryMetric = document.querySelector("#memoryMetric");
    const availableMetric = document.querySelector("#availableMetric");
    const cacheMetric = document.querySelector("#cacheMetric");
    const cacheDetailMetric = document.querySelector("#cacheDetailMetric");
    const swapMetric = document.querySelector("#swapMetric");
    const serverPidMetric = document.querySelector("#serverPidMetric");
    const processMetric = document.querySelector("#processMetric");
    const memoryLabel = document.querySelector("#memoryLabel");
    const memoryPercent = document.querySelector("#memoryPercent");
    const memoryFill = document.querySelector("#memoryFill");
    const requestPanel = document.querySelector("#requestPanel");
    const requestTitle = document.querySelector("#requestTitle");
    const requestId = document.querySelector("#requestId");
    const requestText = document.querySelector("#requestText");
    const requestFill = document.querySelector("#requestFill");
    const cancelRequestButton = document.querySelector("#cancelRequestButton");
    const guardEnabled = document.querySelector("#guardEnabled");
    const contextCap = document.querySelector("#contextCap");
    const outputCap = document.querySelector("#outputCap");
    const reserveGib = document.querySelector("#reserveGib");
    const abortGib = document.querySelector("#abortGib");
    const saveSafetyButton = document.querySelector("#saveSafetyButton");
    const safetySummary = document.querySelector("#safetySummary");
    const safetyErrorText = document.querySelector("#safetyErrorText");
    const clearCacheButton = document.querySelector("#clearCacheButton");
    const cacheRows = document.querySelector("#cacheRows");
    const operationsTabButton = document.querySelector("#operationsTabButton");
    const telemetryTabButton = document.querySelector("#telemetryTabButton");
    const benchmarkTabButton = document.querySelector("#benchmarkTabButton");
    const harnessTabButton = document.querySelector("#harnessTabButton");
    const settingsTabButton = document.querySelector("#settingsTabButton");
    const operationsTab = document.querySelector("#operationsTab");
    const telemetryTab = document.querySelector("#telemetryTab");
    const benchmarkTab = document.querySelector("#benchmarkTab");
    const harnessTab = document.querySelector("#harnessTab");
    const settingsTab = document.querySelector("#settingsTab");
    const authGate = document.querySelector("#authGate");
    const mainContent = document.querySelector("main");
    const authForm = document.querySelector("#authForm");
    const authApiKey = document.querySelector("#authApiKey");
    const authErrorText = document.querySelector("#authErrorText");
    const lockControlCenterButton = document.querySelector("#lockControlCenterButton");
    const accessSettingsForm = document.querySelector("#accessSettingsForm");
    const localAccessMode = document.querySelector("#localAccessMode");
    const networkAccessMode = document.querySelector("#networkAccessMode");
    const controlCenterPort = document.querySelector("#controlCenterPort");
    const apiKeyAuthentication = document.querySelector("#apiKeyAuthentication");
    const modelApiKeyAuthentication = document.querySelector("#modelApiKeyAuthentication");
    const replacementApiKey = document.querySelector("#replacementApiKey");
    const generateApiKeyButton = document.querySelector("#generateApiKeyButton");
    const accessEndpointSummary = document.querySelector("#accessEndpointSummary");
    const settingsRestartNotice = document.querySelector("#settingsRestartNotice");
    const settingsErrorText = document.querySelector("#settingsErrorText");
    const saveAccessSettingsButton = document.querySelector("#saveAccessSettingsButton");
    const restartControlCenterButton = document.querySelector("#restartControlCenterButton");
    const restartMlxServerButton = document.querySelector("#restartMlxServerButton");
    const serviceRestartStatus = document.querySelector("#serviceRestartStatus");
    const telemetryWindow = document.querySelector("#telemetryWindow");
    const refreshTelemetryButton = document.querySelector("#refreshTelemetryButton");
    const telemetryActive = document.querySelector("#telemetryActive");
    const telemetryLoopAlerts = document.querySelector("#telemetryLoopAlerts");
    const telemetryRequests = document.querySelector("#telemetryRequests");
    const telemetryRequestDetail = document.querySelector("#telemetryRequestDetail");
    const telemetryInputTokens = document.querySelector("#telemetryInputTokens");
    const telemetryCacheRate = document.querySelector("#telemetryCacheRate");
    const telemetryOutputTokens = document.querySelector("#telemetryOutputTokens");
    const telemetryTotalTokens = document.querySelector("#telemetryTotalTokens");
    const telemetryThroughput = document.querySelector("#telemetryThroughput");
    const telemetryToolCalls = document.querySelector("#telemetryToolCalls");
    const telemetryMcpCalls = document.querySelector("#telemetryMcpCalls");
    const telemetryLatency = document.querySelector("#telemetryLatency");
    const telemetryErrorRate = document.querySelector("#telemetryErrorRate");
    const telemetryChart = document.querySelector("#telemetryChart");
    const telemetryLegend = document.querySelector("#telemetryLegend");
    const telemetryFailurePanel = document.querySelector("#telemetryFailurePanel");
    const telemetryFailureCount = document.querySelector("#telemetryFailureCount");
    const telemetryFailureClearFilter = document.querySelector("#telemetryFailureClearFilter");
    const telemetryFailureSummary = document.querySelector("#telemetryFailureSummary");
    const telemetryFailureFilter = document.querySelector("#telemetryFailureFilter");
    const telemetryFailureRows = document.querySelector("#telemetryFailureRows");
    const telemetrySources = document.querySelector("#telemetrySources");
    const telemetryRuntimeNote = document.querySelector("#telemetryRuntimeNote");
    const telemetryModelRows = document.querySelector("#telemetryModelRows");
    const telemetryModelToggle = document.querySelector("#telemetryModelToggle");
    const telemetryToolRows = document.querySelector("#telemetryToolRows");
    const telemetryToolToggle = document.querySelector("#telemetryToolToggle");
    const telemetryRequestRows = document.querySelector("#telemetryRequestRows");
    const telemetryRetentionPolicy = document.querySelector("#telemetryRetentionPolicy");
    const telemetryQueryCount = document.querySelector("#telemetryQueryCount");
    const telemetryQueryPageSize = document.querySelector("#telemetryQueryPageSize");
    const telemetryQueryPrevious = document.querySelector("#telemetryQueryPrevious");
    const telemetryQueryPageStatus = document.querySelector("#telemetryQueryPageStatus");
    const telemetryQueryNext = document.querySelector("#telemetryQueryNext");
    const telemetryUpdatedAt = document.querySelector("#telemetryUpdatedAt");
    const telemetryErrorText = document.querySelector("#telemetryErrorText");
    const refreshStatisticsButton = document.querySelector("#refreshStatisticsButton");
    const statisticsTotalModels = document.querySelector("#statisticsTotalModels");
    const statisticsDownloadedDetail = document.querySelector("#statisticsDownloadedDetail");
    const statisticsBenchmarkedModels = document.querySelector("#statisticsBenchmarkedModels");
    const statisticsCoveragePercent = document.querySelector("#statisticsCoveragePercent");
    const statisticsSamples = document.querySelector("#statisticsSamples");
    const statisticsFastestRate = document.querySelector("#statisticsFastestRate");
    const statisticsFastestModel = document.querySelector("#statisticsFastestModel");
    const statisticsLatestDate = document.querySelector("#statisticsLatestDate");
    const statisticsUpdatedAt = document.querySelector("#statisticsUpdatedAt");
    const statisticsSpeedBars = document.querySelector("#statisticsSpeedBars");
    const statisticsCoverageText = document.querySelector("#statisticsCoverageText");
    const statisticsErrorText = document.querySelector("#statisticsErrorText");
    const statisticsRows = document.querySelector("#statisticsRows");
    const harnessForm = document.querySelector("#harnessForm");
    const harnessSystemPrompt = document.querySelector("#harnessSystemPrompt");
    const harnessMaxTokens = document.querySelector("#harnessMaxTokens");
    const harnessTemperature = document.querySelector("#harnessTemperature");
    const harnessTranscript = document.querySelector("#harnessTranscript");
    const harnessInput = document.querySelector("#harnessInput");
    const harnessSendButton = document.querySelector("#harnessSendButton");
    const harnessCancelButton = document.querySelector("#harnessCancelButton");
    const harnessClearButton = document.querySelector("#harnessClearButton");
    const harnessStatus = document.querySelector("#harnessStatus");
    const harnessErrorText = document.querySelector("#harnessErrorText");
    const harnessModelName = document.querySelector("#harnessModelName");
    const logOutput = document.querySelector("#logOutput");
    const runsBody = document.querySelector("#runsBody");
    const bars = document.querySelector("#bars");
    let currentServerModelKey = "";
    let activeRequestId = "";
    let safetyFormModel = "";
    let harnessMessages = [];
    let harnessGenerating = false;
    let harnessServerReady = false;
    let harnessActiveModel = "No model loaded";
    let harnessAbortController = null;
    let statisticsLoaded = false;
    let telemetryLoaded = false;
    let telemetryErrorEvents = [];
    let telemetryFailureTypes = [];
    let telemetrySourceData = [];
    let telemetryFailureTotal = 0;
    let telemetryFailureBucket = null;
    let telemetryQueryPage = 1;
    let settingsLoaded = false;
    let activeTabName = "operations";
    let benchmarkJobRunning = false;
    let localAuthenticationPreference = false;
    let apiKeyConfigured = false;
    let initialized = false;
    let apiKey = "";
    try {
      apiKey = sessionStorage.getItem("mlx-control-center-api-key") || "";
    } catch (_) {}

    function setActiveTab(tabName) {
      const tabs = [
        {name: "operations", button: operationsTabButton, panel: operationsTab},
        {name: "telemetry", button: telemetryTabButton, panel: telemetryTab},
        {name: "benchmark", button: benchmarkTabButton, panel: benchmarkTab},
        {name: "harness", button: harnessTabButton, panel: harnessTab},
        {name: "settings", button: settingsTabButton, panel: settingsTab}
      ];
      activeTabName = tabName;
      for (const tab of tabs) {
        const active = tab.name === tabName;
        tab.panel.hidden = !active;
        tab.button.classList.toggle("active", active);
        tab.button.setAttribute("aria-selected", String(active));
        tab.button.tabIndex = active ? 0 : -1;
      }
      if (tabName === "benchmark" && !statisticsLoaded) {
        refreshModelStatistics().catch(error => {
          statisticsErrorText.textContent = error.message;
        });
      }
      if (tabName === "telemetry" && !telemetryLoaded) {
        refreshTelemetry().catch(error => {
          telemetryErrorText.textContent = error.message;
        });
      }
      if (tabName === "settings" && !settingsLoaded) {
        refreshAccessSettings().catch(error => {
          settingsErrorText.textContent = error.message;
        });
      }
    }

    function authorizedOptions(options = {}) {
      const headers = new Headers(options.headers || {});
      if (apiKey) headers.set("authorization", `Bearer ${apiKey}`);
      return {...options, headers};
    }

    async function api(path, options = {}) {
      const response = await fetch(path, authorizedOptions(options));
      const data = await response.json().catch(() => ({}));
      if (response.status === 401) {
        showAuthenticationGate("Enter the API key to continue.");
      }
      if (!response.ok) throw new Error(data.error || "Request failed");
      return data;
    }

    async function authenticationStatus(candidate = apiKey) {
      const headers = new Headers();
      if (candidate) headers.set("authorization", `Bearer ${candidate}`);
      const response = await fetch("/api/auth/status", {headers});
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Authentication check failed");
      return data;
    }

    function rememberApiKey(value) {
      apiKey = value;
      try {
        if (value) {
          sessionStorage.setItem("mlx-control-center-api-key", value);
        } else {
          sessionStorage.removeItem("mlx-control-center-api-key");
        }
      } catch (_) {}
    }

    function showAuthenticationGate(message = "") {
      authErrorText.textContent = message;
      authApiKey.value = "";
      mainContent.inert = true;
      authGate.hidden = false;
      setTimeout(() => authApiKey.focus(), 0);
    }

    async function unlockControlCenter(candidate) {
      const status = await authenticationStatus(candidate);
      if (status.authentication_required && !status.authenticated) {
        throw new Error("That API key was not accepted.");
      }
      rememberApiKey(status.authentication_required ? candidate : "");
      mainContent.inert = false;
      authGate.hidden = true;
      if (!initialized) {
        await init();
      }
    }

    function escapeHtml(value) {
      return String(value ?? "").replace(/[&<>"']/g, character => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;"
      })[character]);
    }

    function setHarnessGenerating(generating) {
      harnessGenerating = generating;
      harnessSendButton.disabled = generating || !harnessServerReady;
      harnessCancelButton.disabled = !generating;
      harnessClearButton.disabled = generating;
      harnessInput.disabled = generating;
      harnessSystemPrompt.disabled = generating;
      harnessMaxTokens.disabled = generating;
      harnessTemperature.disabled = generating;
    }

    function harnessUsageSummary(message) {
      if (!message.usage) return "";
      const inputTokens = Number(message.usage.prompt_tokens || 0);
      const outputTokens = Number(message.usage.completion_tokens || 0);
      const tokensPerSecond = Number(message.tokensPerSecond || 0);
      return [
        `${inputTokens.toLocaleString()} input tokens`,
        `${outputTokens.toLocaleString()} output tokens`,
        `${tokensPerSecond.toLocaleString(undefined, {
          minimumFractionDigits: 1,
          maximumFractionDigits: 1
        })} output tok/s`
      ].join(" · ");
    }

    function renderHarnessConversation() {
      harnessTranscript.replaceChildren();
      if (!harnessMessages.length) {
        const empty = document.createElement("div");
        empty.className = "harness-empty";
        empty.textContent = "Send a message to begin a local conversation.";
        harnessTranscript.append(empty);
        return;
      }

      for (const message of harnessMessages) {
        const item = document.createElement("article");
        item.className = `harness-message ${message.role}${message.pending ? " pending" : ""}`;

        const header = document.createElement("div");
        header.className = "harness-message-header";
        const role = document.createElement("span");
        role.textContent = message.role === "user" ? "You" : "Assistant";
        const model = document.createElement("span");
        model.textContent = message.role === "assistant"
          ? (message.model || harnessActiveModel)
          : "";
        header.append(role, model);
        item.append(header);

        if (message.reasoning) {
          const reasoning = document.createElement("details");
          reasoning.className = "harness-reasoning";
          reasoning.open = Boolean(message.pending);
          const summary = document.createElement("summary");
          summary.textContent = "Model reasoning";
          const content = document.createElement("div");
          content.textContent = message.reasoning;
          reasoning.append(summary, content);
          item.append(reasoning);
        }

        const body = document.createElement("div");
        body.className = "harness-message-body";
        body.textContent = message.content || (
          message.pending
            ? "Generating…"
            : (message.finishReason === "length"
              ? "No final response before the max-token limit."
              : "No response content.")
        );
        item.append(body);
        if (message.stopReason === "repetition") {
          const notice = document.createElement("div");
          notice.className = "harness-message-notice";
          notice.textContent = message.stopMessage
            || "Generation stopped automatically because repeated output was detected.";
          item.append(notice);
        }
        const usageSummary = harnessUsageSummary(message);
        if (usageSummary) {
          const metrics = document.createElement("div");
          metrics.className = "harness-message-metrics";
          metrics.textContent = usageSummary;
          item.append(metrics);
        }
        harnessTranscript.append(item);
      }
      harnessTranscript.scrollTop = harnessTranscript.scrollHeight;
    }

    function harnessRequestMessages() {
      const messages = [];
      const systemPrompt = harnessSystemPrompt.value.trim();
      if (systemPrompt) {
        messages.push({role: "system", content: systemPrompt});
      }
      for (const message of harnessMessages) {
        if (!message.pending && message.content) {
          messages.push({role: message.role, content: message.content});
        }
      }
      return messages;
    }

    function applyHarnessStreamEvent(payload, assistant) {
      const choice = Array.isArray(payload.choices) ? payload.choices[0] : null;
      const delta = choice?.delta || {};
      const content = typeof delta.content === "string"
        ? delta.content
        : (typeof choice?.text === "string" ? choice.text : "");
      const reasoning = typeof delta.reasoning_content === "string"
        ? delta.reasoning_content
        : (typeof delta.reasoning === "string" ? delta.reasoning : "");
      assistant.content += content;
      assistant.reasoning += reasoning;
      if (choice?.finish_reason) {
        assistant.finishReason = choice.finish_reason;
      }
      if (payload.usage) {
        assistant.usage = payload.usage;
      }
      if (payload.mlx_stop_reason === "repetition") {
        assistant.stopReason = "repetition";
        assistant.stopMessage = typeof payload.mlx_stop_message === "string"
          ? payload.mlx_stop_message
          : "";
      }
      renderHarnessConversation();
    }

    async function streamHarnessCompletion(messages, assistant) {
      harnessAbortController = new AbortController();
      const response = await fetch("/api/harness/chat", {
        method: "POST",
        headers: authorizedOptions({
          headers: {"content-type": "application/json"}
        }).headers,
        body: JSON.stringify({
          messages,
          max_tokens: Number(harnessMaxTokens.value),
          temperature: Number(harnessTemperature.value)
        }),
        signal: harnessAbortController.signal
      });
      if (!response.ok) {
        const responseText = await response.text();
        let error = responseText || "Generation failed.";
        try {
          const payload = JSON.parse(responseText);
          error = payload.error || error;
        } catch (_) {}
        throw new Error(error);
      }
      if (!response.body) throw new Error("Streaming is unavailable in this browser.");

      assistant.generationStartedAt = performance.now();
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const {done, value} = await reader.read();
        buffer += decoder.decode(value || new Uint8Array(), {stream: !done});
        let newline = buffer.indexOf("\n");
        while (newline >= 0) {
          const line = buffer.slice(0, newline).trim();
          buffer = buffer.slice(newline + 1);
          if (line.startsWith("data:")) {
            const data = line.slice(5).trim();
            if (data && data !== "[DONE]") {
              applyHarnessStreamEvent(JSON.parse(data), assistant);
            }
          }
          newline = buffer.indexOf("\n");
        }
        if (done) break;
      }
      const completionTokens = Number(assistant.usage?.completion_tokens || 0);
      const generationSeconds = Math.max(
        (performance.now() - assistant.generationStartedAt) / 1000,
        0.001
      );
      assistant.tokensPerSecond = completionTokens / generationSeconds;
    }

    function renderState(data) {
      promptSelect.innerHTML = data.prompts.map(item =>
        `<option value="${escapeHtml(item.path)}">${escapeHtml(item.name)} (${item.count})</option>`
      ).join("");
      modelSelect.innerHTML = data.models.filter(item => item.downloaded).map(item =>
        `<option value="${escapeHtml(item.key)}">${escapeHtml(item.full_name || item.label)}</option>`
      ).join("");
      serverModelSelect.innerHTML = data.models.filter(item => item.downloaded).map(item =>
        `<option value="${escapeHtml(item.key)}">${escapeHtml(item.label)}</option>`
      ).join("");
      modelSelect.disabled = allModels.checked;
      renderRuns(data.runs);
    }

    function renderAccessSettings(data) {
      const network = data.access_mode === "network";
      localAccessMode.checked = !network;
      networkAccessMode.checked = network;
      controlCenterPort.value = String(data.port);
      apiKeyConfigured = Boolean(data.api_key_configured);
      localAuthenticationPreference = network
        ? false
        : data.authentication === "api_key";
      apiKeyAuthentication.checked = network || localAuthenticationPreference;
      modelApiKeyAuthentication.checked =
        data.model_api_authentication === "api_key";
      lockControlCenterButton.hidden = data.active_authentication !== "api_key";
      apiKeyAuthentication.disabled = network;
      const endpointHost = network ? "<this-mac>" : "127.0.0.1";
      const protection = data.authentication === "api_key"
        ? "API key required"
        : "Authentication disabled";
      const modelProtection = data.model_api_authentication === "api_key"
        ? "Model API key required"
        : "Model API open locally";
      accessEndpointSummary.textContent =
        `Next start: http://${endpointHost}:${data.port} · ${protection} · ${modelProtection}`;
      accessEndpointSummary.classList.toggle(
        "settings-warning",
        network && data.authentication !== "api_key"
      );
      const pendingRestarts = [];
      if (data.control_center_restart_required) pendingRestarts.push("Control Center");
      if (data.model_api_restart_required) pendingRestarts.push("Model API");
      settingsRestartNotice.textContent = pendingRestarts.length
        ? `Restart ${pendingRestarts.join(" and ")} to apply saved access settings.`
        : "";
      settingsRestartNotice.hidden = pendingRestarts.length === 0;
      settingsLoaded = true;
    }

    async function refreshAccessSettings() {
      const data = await api("/api/settings/access");
      renderAccessSettings(data);
      return data;
    }

    function renderRuns(runs) {
      runsBody.innerHTML = runs.map(run => `
        <tr>
          <td>${escapeHtml(run.name)}</td>
          <td class="number">${run.record_count}</td>
          <td>${run.html_href ? `<a href="#" data-artifact="${escapeHtml(run.html_href)}">open</a>` : ""}</td>
          <td><a href="#" data-artifact="${escapeHtml(run.jsonl_href)}">jsonl</a></td>
        </tr>
      `).join("");
      const latest = runs.find(run => run.summaries && run.summaries.length);
      renderBars(latest ? latest.summaries : []);
    }

    async function openArtifact(path) {
      const artifactWindow = window.open("", "_blank");
      if (artifactWindow) artifactWindow.opener = null;
      try {
        const response = await fetch(path, authorizedOptions());
        if (response.status === 401) {
          artifactWindow?.close();
          showAuthenticationGate("Enter the API key again to open artifacts.");
          return;
        }
        if (!response.ok) throw new Error("Artifact download failed.");
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        if (artifactWindow) {
          artifactWindow.location.href = url;
        } else {
          window.open(url, "_blank", "noopener");
        }
        setTimeout(() => URL.revokeObjectURL(url), 60_000);
      } catch (error) {
        artifactWindow?.close();
        throw error;
      }
    }

    function renderBars(summaries) {
      if (!summaries.length) {
        bars.innerHTML = "";
        return;
      }
      const maxRate = Math.max(...summaries.map(item => item.generation_tps || 0), 1);
      bars.innerHTML = summaries.map(item => {
        const width = Math.max(3, Math.round(((item.generation_tps || 0) / maxRate) * 100));
        const name = item.full_name || item.label;
        return `
          <div class="bar">
            <div class="latest-comparison-model" title="${escapeHtml(name)}">
              ${escapeHtml(name)}
            </div>
            <div class="latest-comparison-rate">
              ${(item.generation_tps || 0).toFixed(2)} <small>tok/s</small>
            </div>
            <div class="bar-track" aria-hidden="true">
              <div class="bar-fill" style="width: ${width}%"></div>
            </div>
          </div>
        `;
      }).join("");
    }

    function formatStatistic(value, digits = 2) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) {
        return "—";
      }
      return Number(value).toLocaleString(undefined, {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits
      });
    }

    function formatBenchmarkDate(value, includeTime = true) {
      if (!value) return "—";
      const date = new Date(Number(value) * 1000);
      return includeTime ? date.toLocaleString() : date.toLocaleDateString();
    }

    function renderModelStatistics(data) {
      const summary = data.summary || {};
      const models = Array.isArray(data.models) ? data.models : [];
      const totalModels = Number(summary.total_models || 0);
      const benchmarkedModels = Number(summary.benchmarked_models || 0);
      const fastest = summary.fastest_model || null;

      statisticsTotalModels.textContent = totalModels.toLocaleString();
      statisticsDownloadedDetail.textContent =
        `${Number(summary.downloaded_models || 0).toLocaleString()} downloaded`;
      statisticsBenchmarkedModels.textContent = benchmarkedModels.toLocaleString();
      statisticsCoveragePercent.textContent = totalModels
        ? `${Math.round((benchmarkedModels / totalModels) * 100)}% coverage`
        : "No configured models";
      statisticsSamples.textContent =
        Number(summary.total_records || 0).toLocaleString();
      statisticsFastestRate.textContent = fastest
        ? `${formatStatistic(fastest.average_generation_tps)} tok/s`
        : "—";
      statisticsFastestModel.textContent =
        fastest?.full_name || fastest?.label || "No benchmark data";
      statisticsLatestDate.textContent = formatBenchmarkDate(
        summary.latest_benchmark_at,
        false
      );
      statisticsUpdatedAt.textContent =
        `Aggregated ${formatBenchmarkDate(data.generated_at)}`;

      const unmatched = Number(summary.unmatched_records || 0);
      const sourceErrors = Array.isArray(data.source_errors) ? data.source_errors.length : 0;
      statisticsCoverageText.textContent = [
        `${benchmarkedModels} of ${totalModels} configured models have saved benchmark samples.`,
        unmatched
          ? `${unmatched} historical record${unmatched === 1 ? "" : "s"} could not be matched to the current registry.`
          : "All readable records matched the current model registry.",
        sourceErrors
          ? `${sourceErrors} run file${sourceErrors === 1 ? "" : "s"} could not be read.`
          : ""
      ].filter(Boolean).join(" ");

      const speedModels = models
        .filter(model =>
          Number.isFinite(Number(model.benchmark?.average_generation_tps))
          && model.benchmark?.average_generation_tps !== null
        )
        .sort((left, right) =>
          right.benchmark.average_generation_tps - left.benchmark.average_generation_tps
        );
      if (speedModels.length) {
        const maxRate = Math.max(
          ...speedModels.map(model => model.benchmark.average_generation_tps),
          1
        );
        statisticsSpeedBars.innerHTML = speedModels.map(model => {
          const rate = model.benchmark.average_generation_tps;
          const width = Math.max(3, Math.round((rate / maxRate) * 100));
          return `
            <div class="bar">
              <div>${escapeHtml(model.full_name || model.label)}</div>
              <div class="bar-track"><div class="bar-fill" style="width: ${width}%"></div></div>
              <div class="number">${formatStatistic(rate)}</div>
            </div>
          `;
        }).join("");
      } else {
        statisticsSpeedBars.innerHTML =
          '<div class="empty-row">No generation-speed samples yet.</div>';
      }

      statisticsRows.innerHTML = models.map(model => {
        const benchmark = model.benchmark || {};
        return `
          <tr>
            <td class="statistics-model">
              <strong>${escapeHtml(model.full_name || model.label)}</strong>
              <small>${escapeHtml(model.role || model.repo_id)}</small>
            </td>
            <td>
              <span class="availability-badge${model.downloaded ? "" : " missing"}">
                ${model.downloaded ? "Available" : "Not downloaded"}
              </span>
            </td>
            <td class="number">${model.estimated_size_gb === null || model.estimated_size_gb === undefined
              ? "—"
              : `${formatStatistic(model.estimated_size_gb, 1)} GB`}</td>
            <td class="number">${Number(benchmark.record_count || 0).toLocaleString()}</td>
            <td class="number">${Number(benchmark.run_count || 0).toLocaleString()}</td>
            <td class="number">${formatStatistic(benchmark.average_prompt_tps)}</td>
            <td class="number">${formatStatistic(benchmark.average_generation_tps)}</td>
            <td class="number">${formatStatistic(benchmark.best_generation_tps)}</td>
            <td class="number">${benchmark.peak_memory_gb === null || benchmark.peak_memory_gb === undefined
              ? "—"
              : `${formatStatistic(benchmark.peak_memory_gb, 1)} GB`}</td>
            <td>${formatBenchmarkDate(benchmark.last_benchmarked_at, false)}</td>
          </tr>
        `;
      }).join("") || (
        '<tr><td class="empty-row" colspan="10">No configured models found.</td></tr>'
      );
    }

    function formatTelemetryNumber(value) {
      const number = Number(value || 0);
      if (Math.abs(number) < 10_000) return number.toLocaleString();
      return new Intl.NumberFormat(undefined, {
        notation: "compact",
        maximumFractionDigits: 1
      }).format(number);
    }

    function formatTelemetryDuration(value) {
      const seconds = Number(value || 0);
      if (seconds < 1) return `${Math.round(seconds * 1000)} ms`;
      if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 1 : 0)} s`;
      return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
    }

    function formatTelemetryTime(value) {
      if (!value) return "—";
      return new Date(Number(value) * 1000).toLocaleString([], {
        month: "short",
        day: "numeric",
        hour: "numeric",
        minute: "2-digit",
        second: "2-digit"
      });
    }

    function updateTelemetryActivityToggle(button, rows, itemCount, itemLabel) {
      const hasAdditionalItems = itemCount > 5;
      button.hidden = !hasAdditionalItems;
      if (!hasAdditionalItems) {
        rows.classList.remove("expanded");
      }
      const expanded = hasAdditionalItems && rows.classList.contains("expanded");
      button.setAttribute("aria-expanded", String(expanded));
      button.textContent = expanded
        ? "Show top 5"
        : `Show all ${itemCount.toLocaleString()} ${itemLabel}`;
    }

    function renderTelemetryFailures() {
      const loadedFailures = telemetryErrorEvents.length;
      const totalFailures = telemetryFailureTotal;
      const displayedFailures = telemetryFailureBucket
        ? telemetryErrorEvents.filter(record => {
            const timestamp = Number(record.completed_at || record.started_at || 0);
            return timestamp >= telemetryFailureBucket.start
              && timestamp < telemetryFailureBucket.end;
          })
        : telemetryErrorEvents;

      telemetryFailureCount.textContent = telemetryFailureBucket
        ? `${displayedFailures.length.toLocaleString()} in selected bucket · ${totalFailures.toLocaleString()} total`
        : (
            totalFailures === loadedFailures
              ? `${totalFailures.toLocaleString()} failure${totalFailures === 1 ? "" : "s"}`
              : `${totalFailures.toLocaleString()} failures · latest ${loadedFailures.toLocaleString()} shown`
          );
      telemetryFailureCount.classList.toggle("clear", totalFailures === 0);
      telemetryFailureClearFilter.hidden = !telemetryFailureBucket;
      telemetryFailureFilter.hidden = !telemetryFailureBucket;
      telemetryFailureFilter.textContent = telemetryFailureBucket
        ? `Showing failures from ${formatTelemetryTime(telemetryFailureBucket.start)} to ${formatTelemetryTime(telemetryFailureBucket.end)}.`
        : "";

      let typeRows = telemetryFailureTypes;
      let sourceRows = telemetrySourceData.filter(source => Number(source.errors || 0) > 0);
      if (telemetryFailureBucket) {
        const typeCounts = new Map();
        const sourceCounts = new Map();
        displayedFailures.forEach(record => {
          const typeKey = String(record.failure_code || "request_error");
          const type = typeCounts.get(typeKey) || {
            code: typeKey,
            label: record.failure_type || "Request error",
            count: 0
          };
          type.count += 1;
          typeCounts.set(typeKey, type);

          const sourceKey = `${record.client || "Unknown client"}\u0000${record.endpoint || "Unknown endpoint"}`;
          const source = sourceCounts.get(sourceKey) || {
            client: record.client || "Unknown client",
            endpoint: record.endpoint || "Unknown endpoint",
            errors: 0
          };
          source.errors += 1;
          sourceCounts.set(sourceKey, source);
        });
        typeRows = [...typeCounts.values()].sort((a, b) => b.count - a.count);
        sourceRows = [...sourceCounts.values()].sort((a, b) => b.errors - a.errors);
      }

      if (totalFailures) {
        const typeMarkup = typeRows.map(item => `
          <span class="telemetry-failure-chip">
            <strong>${Number(item.count || 0).toLocaleString()}</strong>${escapeHtml(item.label || "Request error")}
          </span>
        `).join("");
        const sourceMarkup = sourceRows.map(source => `
          <span class="telemetry-failure-chip" title="${escapeHtml(source.client || "Unknown client")} · ${escapeHtml(source.endpoint || "/v1/chat/completions")}">
            <strong>${Number(source.errors || 0).toLocaleString()}</strong>${escapeHtml(source.client || "Unknown client")} · ${escapeHtml(source.endpoint || "/v1/chat/completions")}
          </span>
        `).join("");
        telemetryFailureSummary.innerHTML = `
          <div class="telemetry-failure-summary-group">
            <strong>Types</strong>
            <div class="telemetry-failure-chips">${typeMarkup || '<span class="muted">No failures in the selected bucket.</span>'}</div>
          </div>
          <div class="telemetry-failure-summary-group">
            <strong>Origins</strong>
            <div class="telemetry-failure-chips">${sourceMarkup || '<span class="muted">No failures in the selected bucket.</span>'}</div>
          </div>
        `;
      } else {
        telemetryFailureSummary.innerHTML = `
          <div class="telemetry-failure-summary-group">
            <strong>Status</strong>
            <span class="muted">No errors or cancellations were recorded in this time range.</span>
          </div>
        `;
      }

      telemetryFailureRows.innerHTML = displayedFailures.map(record => {
        const status = String(record.status || "error");
        const error = record.error || (
          status === "cancelled" ? "Request was cancelled without a reason." : "No error message was retained."
        );
        const query = record.query_preview || "Query preview disabled or unavailable";
        return `
          <tr class="telemetry-failure-row">
            <td>${formatTelemetryTime(record.completed_at || record.started_at)}</td>
            <td>
              <span class="telemetry-badge ${escapeHtml(status)}">${escapeHtml(record.failure_type || "Request error")}</span>
            </td>
            <td class="telemetry-failure-source">
              <strong>${escapeHtml(record.client || "Unknown client")}</strong>
              <small>${escapeHtml(record.client_address || "Address unavailable")}</small>
            </td>
            <td class="telemetry-failure-endpoint">
              <code>${escapeHtml(record.endpoint || "Unknown endpoint")}</code>
              <small>${escapeHtml(record.request_type || "unknown")} request</small>
            </td>
            <td class="telemetry-model" title="${escapeHtml(record.model || "")}">${escapeHtml(shortModelName(record.model))}</td>
            <td class="telemetry-failure-details">
              <strong>${escapeHtml(error)}</strong>
              <small title="${escapeHtml(query)}">Query: ${escapeHtml(query)}</small>
              <small>Request ID: ${escapeHtml(record.id || "unavailable")}</small>
            </td>
          </tr>
        `;
      }).join("") || (
        '<tr><td class="empty-row" colspan="6">No errors or cancellations in this selection.</td></tr>'
      );
    }

    function renderTelemetry(data) {
      const summary = data.summary || {};
      const active = Array.isArray(data.active) ? data.active : [];
      const models = Array.isArray(data.models) ? data.models : [];
      const tools = Array.isArray(data.tools) ? data.tools : [];
      const sources = Array.isArray(data.sources) ? data.sources : [];
      const failureTypes = Array.isArray(data.failure_types) ? data.failure_types : [];
      const errorEvents = Array.isArray(data.error_events) ? data.error_events : [];
      const requests = Array.isArray(data.requests) ? data.requests : [];
      const requestPagination = data.request_pagination || {};
      const retention = data.retention || {};
      const loopRisks = Array.isArray(data.agent_loop_risks)
        ? data.agent_loop_risks
        : [];
      const buckets = Array.isArray(data.timeline?.buckets)
        ? data.timeline.buckets
        : [];
      telemetryErrorEvents = errorEvents;
      telemetryFailureTypes = failureTypes;
      telemetrySourceData = sources;
      telemetryFailureTotal = Number(summary.errors || errorEvents.length);

      telemetryRequests.textContent = formatTelemetryNumber(summary.requests);
      telemetryRequestDetail.textContent =
        `${Number(summary.active_requests || 0).toLocaleString()} active · ${Number(summary.errors || 0).toLocaleString()} failures`;
      telemetryInputTokens.textContent = formatTelemetryNumber(summary.prompt_tokens);
      telemetryCacheRate.textContent =
        `${Number(summary.cache_hit_percent || 0).toFixed(1)}% cache hit`;
      telemetryOutputTokens.textContent = formatTelemetryNumber(summary.output_tokens);
      telemetryTotalTokens.textContent = formatTelemetryNumber(summary.total_tokens);
      telemetryThroughput.textContent =
        `${Number(summary.average_tokens_per_second || 0).toLocaleString(undefined, {maximumFractionDigits: 1})} avg tok/s`;
      telemetryToolCalls.textContent = formatTelemetryNumber(summary.tool_calls);
      telemetryMcpCalls.textContent = formatTelemetryNumber(summary.mcp_tool_calls);
      telemetryLatency.textContent = formatTelemetryDuration(summary.average_latency_seconds);
      const requestCount = Number(summary.requests || 0);
      const errorPercent = requestCount
        ? (Number(summary.errors || 0) / requestCount) * 100
        : 0;
      telemetryErrorRate.textContent = `${errorPercent.toFixed(1)}% failure rate`;

      if (active.length) {
        telemetryActive.hidden = false;
        telemetryActive.innerHTML = active.map(record => `
          <strong>Live · ${escapeHtml(record.client || "Unknown client")} · ${escapeHtml(shortModelName(record.model))}</strong>
          <span>${escapeHtml(record.query_preview || "Request in progress")} ·
            ${Number(record.prompt_tokens || 0).toLocaleString()} input ·
            ${Number(record.output_tokens || 0).toLocaleString()} output ·
            ${formatTelemetryDuration(record.duration_seconds)}</span>
        `).join("<br>");
      } else {
        telemetryActive.hidden = true;
        telemetryActive.textContent = "";
      }

      if (loopRisks.length) {
        telemetryLoopAlerts.hidden = false;
        telemetryLoopAlerts.innerHTML = loopRisks.map(risk => {
          const level = String(risk.level || "watch");
          const signals = Array.isArray(risk.signals) ? risk.signals : [];
          const query = risk.query_preview || "Query preview disabled";
          return `
            <article class="agent-loop-alert ${escapeHtml(level)}">
              <div class="agent-loop-alert-heading">
                <strong title="${escapeHtml(query)}">${escapeHtml(risk.client || "Unknown client")} · ${escapeHtml(shortModelName(risk.model))}</strong>
                <span class="telemetry-badge ${level === "high" ? "error" : ""}">
                  ${risk.active ? "Active · " : ""}${escapeHtml(level)} loop risk · ${Number(risk.score || 0)}
                </span>
              </div>
              <p>${escapeHtml(query)}</p>
              <div class="agent-loop-metrics">
                <span><strong>${Number(risk.consecutive_tool_steps || 0).toLocaleString()}</strong> consecutive tool steps</span>
                <span><strong>${formatTelemetryNumber(risk.context_tokens)}</strong> context tokens</span>
                <span><strong>${Number(risk.cache_hit_percent || 0).toFixed(1)}%</strong> recent cache hit</span>
                <span><strong>${formatTelemetryDuration(risk.average_recent_latency_seconds)}</strong> recent latency</span>
              </div>
              <ul class="agent-loop-signals">
                ${signals.map(signal => `<li>${escapeHtml(signal.message || "")}</li>`).join("")}
              </ul>
              <p class="recommendation">${escapeHtml(risk.recommendation || "")}</p>
            </article>
          `;
        }).join("");
      } else {
        telemetryLoopAlerts.hidden = true;
        telemetryLoopAlerts.textContent = "";
      }

      const maxBucketTokens = Math.max(
        1,
        ...buckets.map(bucket =>
          Number(bucket.prompt_tokens || 0) + Number(bucket.output_tokens || 0)
        )
      );
      telemetryChart.innerHTML = buckets.map(bucket => {
        const tokens =
          Number(bucket.prompt_tokens || 0) + Number(bucket.output_tokens || 0);
        const height = tokens ? Math.max(3, (tokens / maxBucketTokens) * 100) : 0;
        const title = [
          formatTelemetryTime(bucket.started_at),
          `${Number(bucket.requests || 0).toLocaleString()} requests`,
          `${Number(bucket.prompt_tokens || 0).toLocaleString()} input tokens`,
          `${Number(bucket.output_tokens || 0).toLocaleString()} output tokens`,
          `${Number(bucket.tool_calls || 0).toLocaleString()} tool calls`,
          `${Number(bucket.errors || 0).toLocaleString()} errors or cancellations`
        ].join(" · ");
        if (Number(bucket.errors || 0) > 0) {
          const bucketStart = Number(bucket.started_at || 0);
          const bucketEnd = bucketStart + Number(data.timeline?.bucket_seconds || 0);
          return `
            <button class="telemetry-column has-error" type="button"
              data-failure-bucket-start="${bucketStart}" data-failure-bucket-end="${bucketEnd}"
              aria-label="${escapeHtml(title)}. Select to inspect failures."
              title="${escapeHtml(title)} · Select to inspect failures">
              <em class="telemetry-column-error-count">${Number(bucket.errors).toLocaleString()}</em>
              <span style="height: ${height}%"></span>
            </button>
          `;
        }
        return `
          <div class="telemetry-column" title="${escapeHtml(title)}">
            <span style="height: ${height}%"></span>
          </div>
        `;
      }).join("");
      telemetryLegend.innerHTML = buckets.length
        ? `
          <span><strong>${formatTelemetryTime(buckets[0].started_at)}</strong> start</span>
          <span><strong>${formatTelemetryTime(buckets[buckets.length - 1].started_at)}</strong> latest bucket</span>
          <span><strong>${Math.round(Number(data.timeline?.bucket_seconds || 0) / 60).toLocaleString()} min</strong> per bucket</span>
        `
        : "<span>No completed requests in this time range.</span>";

      if (sources.length) {
        const maxRequests = Math.max(...sources.map(source => Number(source.requests || 0)), 1);
        telemetrySources.innerHTML = sources.map(source => `
          <div class="bar">
            <div>${escapeHtml(source.client)}</div>
            <div class="bar-track"><div class="bar-fill" style="width: ${Math.max(3, (Number(source.requests || 0) / maxRequests) * 100)}%"></div></div>
            <div class="number telemetry-source-count">
              ${Number(source.requests || 0).toLocaleString()}
              <small class="${Number(source.errors || 0) ? "has-error" : ""}">${Number(source.errors || 0).toLocaleString()} failures</small>
            </div>
          </div>
        `).join("");
      } else {
        telemetrySources.innerHTML = '<div class="empty-row">No client activity yet.</div>';
      }
      telemetryRuntimeNote.textContent = data.runtime_available
        ? "Live telemetry is connected to the running MLX server."
        : "The MLX server is offline; showing retained history from disk.";

      telemetryModelRows.innerHTML = models.map(model => `
        <tr>
          <td class="telemetry-model" title="${escapeHtml(model.model)}">${escapeHtml(shortModelName(model.model))}</td>
          <td class="number">${Number(model.requests || 0).toLocaleString()}</td>
          <td class="number">${Number(model.prompt_tokens || 0).toLocaleString()}</td>
          <td class="number">${Number(model.output_tokens || 0).toLocaleString()}</td>
          <td class="number">${Number(model.tool_calls || 0).toLocaleString()}</td>
          <td class="number">${Number(model.errors || 0).toLocaleString()}</td>
          <td class="number">${formatTelemetryDuration(model.average_latency_seconds)}</td>
        </tr>
      `).join("") || (
        '<tr><td class="empty-row" colspan="7">No model activity in this time range.</td></tr>'
      );
      updateTelemetryActivityToggle(
        telemetryModelToggle,
        telemetryModelRows,
        models.length,
        "models"
      );

      telemetryToolRows.innerHTML = tools.map(tool => `
        <tr>
          <td>
            ${escapeHtml(tool.name)}
            ${tool.mcp_server
              ? `<span class="telemetry-badge mcp" title="MCP server: ${escapeHtml(tool.mcp_server)}">MCP</span>`
              : ""}
          </td>
          <td class="number">${Number(tool.offered_in_requests || 0).toLocaleString()}</td>
          <td class="number">${Number(tool.invocations || 0).toLocaleString()}</td>
        </tr>
      `).join("") || (
        '<tr><td class="empty-row" colspan="3">No tool activity in this time range.</td></tr>'
      );
      updateTelemetryActivityToggle(
        telemetryToolToggle,
        telemetryToolRows,
        tools.length,
        "tools"
      );

      telemetryRequestRows.innerHTML = requests.map(record => {
        const calls = Array.isArray(record.tool_calls) ? record.tool_calls : [];
        const callMarkup = calls.length
          ? calls.map(call => `
              <span title="${escapeHtml(call.arguments_preview || "")}">
                ${escapeHtml(call.name || "unknown")}${call.mcp_server ? " · MCP" : ""}
              </span>
            `).join("<br>")
          : "—";
        const status = String(record.status || "unknown");
        return `
          <tr class="${status === "completed" ? "" : "telemetry-failure-row"}">
            <td>${formatTelemetryTime(record.completed_at || record.started_at)}</td>
            <td>${escapeHtml(record.client || "Unknown client")}</td>
            <td class="telemetry-model" title="${escapeHtml(record.model || "")}">${escapeHtml(shortModelName(record.model))}</td>
            <td class="telemetry-query">${escapeHtml(record.query_preview || "Preview disabled or unavailable")}</td>
            <td class="number">${Number(record.prompt_tokens || 0).toLocaleString()}</td>
            <td class="number">${Number(record.cached_prompt_tokens || 0).toLocaleString()}</td>
            <td class="number">${Number(record.output_tokens || 0).toLocaleString()}</td>
            <td>${callMarkup}</td>
            <td class="number">${formatTelemetryDuration(record.duration_seconds)}</td>
            <td><span class="telemetry-badge ${escapeHtml(status)}" title="${escapeHtml(record.error || "")}">${escapeHtml(status)}</span></td>
          </tr>
        `;
      }).join("") || (
        '<tr><td class="empty-row" colspan="10">No requests in this time range.</td></tr>'
      );

      telemetryQueryPage = Number(requestPagination.page || 1);
      const totalQueryPages = Number(requestPagination.total_pages || 1);
      const totalQueries = Number(requestPagination.total_items || 0);
      const firstQuery = Number(requestPagination.first_item || 0);
      const lastQuery = Number(requestPagination.last_item || 0);
      telemetryQueryCount.textContent = totalQueries
        ? `${firstQuery.toLocaleString()}–${lastQuery.toLocaleString()} of ${totalQueries.toLocaleString()} queries`
        : "0 queries";
      telemetryQueryPageStatus.textContent =
        `Page ${telemetryQueryPage.toLocaleString()} of ${totalQueryPages.toLocaleString()}`;
      telemetryQueryPrevious.disabled = !requestPagination.has_previous;
      telemetryQueryNext.disabled = !requestPagination.has_next;
      const renderedPageSize = String(requestPagination.page_size || 25);
      if ([...telemetryQueryPageSize.options].some(option => option.value === renderedPageSize)) {
        telemetryQueryPageSize.value = renderedPageSize;
      }
      telemetryRetentionPolicy.textContent =
        `${Number(retention.days || 30).toLocaleString()}-day retention · daily archive · automatic deletion`;
      telemetryUpdatedAt.textContent =
        `Updated ${formatTelemetryTime(data.generated_at)}`;
      renderTelemetryFailures();
      telemetryErrorText.textContent = "";
    }

    function formatBytes(value) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
      const bytes = Number(value);
      const units = ["B", "KiB", "MiB", "GiB", "TiB"];
      let size = bytes;
      let unit = 0;
      while (size >= 1024 && unit < units.length - 1) {
        size /= 1024;
        unit += 1;
      }
      const precision = unit >= 3 ? 1 : (unit === 0 ? 0 : 1);
      return `${size.toFixed(precision)} ${units[unit]}`;
    }

    function shortModelName(value) {
      if (!value || value === "unknown" || value === "none") return "No model loaded";
      const parts = value.split("/");
      return parts[parts.length - 1] || value;
    }

    function renderSafety(safety) {
      const policy = safety.policy || {};
      const profile = safety.profile || {};
      const event = safety.last_event || {};
      const profileModel = profile.model || "";
      if (policy.max_context_tokens && safetyFormModel !== profileModel) {
        guardEnabled.checked = policy.enabled !== false;
        contextCap.value = policy.max_context_tokens;
        outputCap.value = policy.max_output_tokens;
        reserveGib.value = (Number(policy.memory_reserve_bytes || 0) / (1024 ** 3)).toFixed(0);
        abortGib.value = (Number(policy.memory_abort_bytes || 0) / (1024 ** 3)).toFixed(0);
        safetyFormModel = profileModel;
      }

      if (!policy.max_context_tokens) {
        safetySummary.textContent = "Safety controls are unavailable until the managed MLX server is restarted.";
        saveSafetyButton.disabled = true;
        return;
      }

      saveSafetyButton.disabled = false;
      const effective = Number(profile.effective_context_tokens || policy.max_context_tokens);
      const dynamic = Math.min(effective, Number(safety.dynamic_token_budget || effective));
      const eventText = event.message ? ` Last action: ${event.message}` : "";
      safetySummary.textContent = [
        policy.enabled === false ? "Guard off." : "Guard on.",
        `Effective context ${effective.toLocaleString()} tokens.`,
        `Current memory budget allows about ${dynamic.toLocaleString()} total active tokens.`,
        `KV estimate ${formatBytes(profile.kv_bytes_per_token)} per token.`,
        eventText
      ].join(" ");
    }

    function renderCache(cache) {
      const entries = Array.isArray(cache.entries) ? cache.entries : [];
      clearCacheButton.disabled = entries.length === 0;
      if (!entries.length) {
        cacheRows.innerHTML = '<tr><td class="empty-row" colspan="4">No retained cache entries</td></tr>';
        return;
      }
      cacheRows.innerHTML = entries.map(entry => `
        <tr title="${escapeHtml(shortModelName(entry.model))}">
          <td>${escapeHtml(entry.cache_type)}</td>
          <td class="number">${Number(entry.tokens || 0).toLocaleString()}</td>
          <td class="number">${formatBytes(entry.bytes)}</td>
          <td><button class="danger inline-button cache-remove" type="button" data-entry-id="${escapeHtml(entry.id)}">Clear</button></td>
        </tr>
      `).join("");
    }

    function renderServerStatus(data) {
      const status = data.status || {};
      const runtime = data.runtime || {};
      const control = data.control || {};
      const memory = data.memory || {};
      const process = memory.process || {};
      const activity = data.activity || {};
      const model = data.model || {};
      const safety = control.safety || runtime.safety || {};
      const cache = control.prompt_cache || runtime.prompt_cache || {};
      const activeRequests = Array.isArray(safety.active_requests)
        ? safety.active_requests
        : [];
      const activeRequest = activeRequests[0] || null;
      const listeners = status.listener_pids && status.listener_pids.length
        ? status.listener_pids.join(", ")
        : "none";
      const ready = Boolean(status.http_ready);
      const pressure = memory.pressure || (ready ? "healthy" : "critical");
      const loadedName = model.label || shortModelName(status.loaded_model);
      const cacheBytes = cache.bytes ?? Math.round((activity.cache_gb || 0) * 1e9);
      const cacheSequences = cache.sequences ?? activity.cache_sequences ?? 0;
      const cacheLimit = cache.max_bytes;
      const serverOutputCap = Number(safety.policy?.max_output_tokens || 32000);

      healthCard.dataset.state = ready ? pressure : "critical";
      serverHealth.textContent = ready
        ? (pressure === "healthy" ? "Online · healthy" : `Online · ${pressure}`)
        : "Offline";
      lastUpdated.textContent = `Updated ${new Date((data.collected_at || Date.now() / 1000) * 1000).toLocaleTimeString()}`;

      loadedModelMetric.textContent = loadedName;
      loadedModelMetric.title = status.loaded_model || "";
      modelRoleMetric.textContent = model.role
        ? `${model.role}${model.estimated_size_gb ? ` · ~${model.estimated_size_gb} GB` : ""}`
        : (ready ? "Runtime model reported by MLX" : "No model loaded");
      harnessServerReady = ready;
      harnessActiveModel = loadedName;
      harnessModelName.textContent = loadedName;
      harnessModelName.title = status.loaded_model || "";
      harnessMaxTokens.max = String(serverOutputCap);
      if (!harnessGenerating && Number(harnessMaxTokens.value) > serverOutputCap) {
        harnessMaxTokens.value = String(serverOutputCap);
      }
      if (!harnessGenerating && (!ready || !harnessMessages.length)) {
        harnessStatus.textContent = ready
          ? "Ready · press Enter to send"
          : "Waiting for an active model";
      }
      setHarnessGenerating(harnessGenerating);

      memoryMetric.textContent = memory.total
        ? `${formatBytes(memory.used)} / ${formatBytes(memory.total)}`
        : "—";
      availableMetric.textContent = memory.available
        ? `${formatBytes(memory.available)} available`
        : "— available";
      cacheMetric.textContent = `${formatBytes(cacheBytes)} · ${cacheSequences}`;
      cacheDetailMetric.textContent = cacheLimit
        ? `${formatBytes(cacheLimit)} retained limit`
        : `${cacheSequences === 1 ? "sequence" : "sequences"} retained`;
      swapMetric.textContent = formatBytes(memory.swap_used);
      serverPidMetric.textContent = status.managed_pid && status.managed_pid !== "none"
        ? `PID ${status.managed_pid}`
        : "Stopped";
      processMetric.textContent = process.rss
        ? `${formatBytes(process.rss)} RSS · ${process.status || "running"}`
        : "No managed process";

      const usedPercent = Number(memory.used_percent || 0);
      memoryLabel.textContent = `${pressure[0]?.toUpperCase() || ""}${pressure.slice(1)} · ${formatBytes(memory.wired)} wired`;
      memoryPercent.textContent = `${usedPercent.toFixed(1)}% used`;
      memoryFill.style.width = `${Math.min(100, usedPercent)}%`;
      memoryFill.className = `memory-fill ${pressure === "healthy" ? "" : pressure}`.trim();

      requestPanel.hidden = !activeRequest && !activity.active;
      if (activeRequest) {
        activeRequestId = activeRequest.id || "";
        const promptTotal = Number(activeRequest.prompt_tokens || 0);
        const promptProcessed = Number(activeRequest.prompt_processed || 0);
        const outputTokens = Number(activeRequest.output_tokens || 0);
        const statusName = String(activeRequest.status || "active");
        const kvBytesPerToken = Number(safety.profile?.kv_bytes_per_token || 0);
        const progress = promptTotal
          ? Math.min(100, (promptProcessed / promptTotal) * 100)
          : 0;
        const promptKvText = kvBytesPerToken
          ? ` · ~${formatBytes(promptProcessed * kvBytesPerToken)} / ${formatBytes(promptTotal * kvBytesPerToken)} KV memory`
          : "";
        requestTitle.textContent = `Active request · ${statusName}`;
        requestId.textContent = `${activeRequest.id || "unknown"} · output ${outputTokens.toLocaleString()} / ${Number(activeRequest.max_output_tokens || 0).toLocaleString()}`;
        requestText.textContent = statusName === "generating"
          ? `${outputTokens.toLocaleString()} output tokens`
          : `${promptProcessed.toLocaleString()} / ${promptTotal.toLocaleString()} prompt tokens${promptKvText} · ${progress.toFixed(1)}%`;
        requestFill.style.width = `${statusName === "generating" ? 100 : progress}%`;
        cancelRequestButton.disabled = statusName === "cancelling";
      } else if (activity.active) {
        activeRequestId = "";
        requestTitle.textContent = "Active prompt prefill";
        requestId.textContent = "Legacy log telemetry; request ID unavailable";
        requestText.textContent = `${Number(activity.prompt_current || 0).toLocaleString()} / ${Number(activity.prompt_total || 0).toLocaleString()} tokens · ${activity.prompt_percent || 0}%`;
        requestFill.style.width = `${Math.min(100, Number(activity.prompt_percent || 0))}%`;
        cancelRequestButton.disabled = true;
      } else {
        activeRequestId = "";
      }

      renderSafety(safety);
      renderCache(cache);

      if (model.key) {
        currentServerModelKey = model.key;
        if ([...serverModelSelect.options].some(option => option.value === model.key)) {
          serverModelSelect.value = model.key;
        }
      }

      serverStatusOutput.textContent = [
        `HTTP: ${status.http_ready ? "ready" : "not ready"} ${status.endpoint || ""}`.trim(),
        `Managed PID: ${status.managed_pid || "none"}`,
        `Listener PID(s): ${listeners}`,
        `Runtime model: ${status.loaded_model || "unknown"}`,
        status.startup_model ? `Startup model: ${status.startup_model}` : "",
        `Pinned model: ${runtime.pinned_model === false ? "no" : "yes"}`,
        `Prompt cache: ${cacheSequences} sequence(s), ${formatBytes(cacheBytes)}`,
        cacheLimit ? `Cache limit: ${formatBytes(cacheLimit)}` : "",
        safety.policy
          ? `Safety: ${safety.policy.enabled === false ? "off" : "on"} · context ${Number(safety.profile?.effective_context_tokens || safety.policy.max_context_tokens).toLocaleString()} · output ${Number(safety.policy.max_output_tokens).toLocaleString()}`
          : "Safety: unavailable until restart",
        `Active requests: ${activeRequests.length}`,
        activeRequest
          ? `Request: ${activeRequest.id} · ${activeRequest.status} · prompt ${activeRequest.prompt_processed}/${activeRequest.prompt_tokens}`
          : (activity.active
            ? `Prefill: ${activity.prompt_current}/${activity.prompt_total} (${activity.prompt_percent}%)`
            : "Prefill: idle"),
        safety.last_event?.message ? `Last safety action: ${safety.last_event.message}` : "",
        activity.last_error ? `Last memory error: ${activity.last_error}` : "",
        status.default_model ? `Default: ${status.default_model}` : "",
      ].filter(Boolean).join("\n");
    }

    function setServerButtonsDisabled(disabled) {
      loadServerButton.disabled = disabled;
      restartServerButton.disabled = disabled;
      restartMlxServerButton.disabled = disabled;
      unloadServerButton.disabled = disabled;
      refreshServerButton.disabled = disabled;
    }

    async function refreshServerStatus() {
      const data = await api("/api/mlx-server/status");
      renderServerStatus(data);
      return data;
    }

    async function restartManagedMlxServer() {
      const data = await api("/api/mlx-server/restart", {
        method: "POST",
        headers: {"content-type": "application/json"},
        body: JSON.stringify({model: currentServerModelKey || serverModelSelect.value})
      });
      renderServerStatus(data.status_payload || data);
      return data;
    }

    function formatBenchmarkDuration(value) {
      const totalSeconds = Math.max(0, Math.floor(Number(value || 0)));
      const hours = Math.floor(totalSeconds / 3600);
      const minutes = Math.floor((totalSeconds % 3600) / 60);
      const seconds = totalSeconds % 60;
      if (hours) return `${hours}h ${minutes}m ${seconds}s`;
      if (minutes) return `${minutes}m ${seconds}s`;
      return `${seconds}s`;
    }

    function renderBenchmarkProgress(job) {
      const progress = job.progress || {};
      const state = progress.state || (job.running ? "running" : "idle");
      const statusLabels = {
        idle: "Idle",
        starting: "Starting",
        running: "Running",
        complete: "Complete",
        failed: "Failed"
      };
      const modelCurrent = Number(progress.model_current || 0);
      const modelTotal = Number(progress.model_total || 0);
      const promptCurrent = Number(progress.prompt_current || 0);
      const promptTotal = Number(progress.prompt_total || 0);
      const completedSteps = Number(progress.completed_steps || 0);
      const totalSteps = Number(progress.total_steps || 0);
      const percent = Math.min(100, Math.max(0, Number(progress.percent || 0)));

      benchmarkProgressStatus.dataset.state = state;
      benchmarkProgressStatus.textContent = statusLabels[state] || state;
      benchmarkProgressStage.textContent = progress.stage || "Waiting to start";
      benchmarkProgressPercent.textContent = `${percent.toFixed(percent % 1 ? 1 : 0)}%`;
      benchmarkProgressSteps.textContent = totalSteps
        ? `${completedSteps.toLocaleString()} / ${totalSteps.toLocaleString()} prompts complete`
        : "Waiting for benchmark plan";
      benchmarkProgressTrack.setAttribute("aria-valuenow", String(percent));
      benchmarkProgressTrack.setAttribute(
        "aria-valuetext",
        totalSteps
          ? `${completedSteps} of ${totalSteps} prompts complete`
          : statusLabels[state] || state
      );
      benchmarkProgressFill.style.width = `${percent}%`;
      benchmarkProgressFill.classList.toggle(
        "running",
        state === "running" || state === "starting"
      );

      benchmarkProgressModelCount.textContent = modelTotal
        ? `${modelCurrent.toLocaleString()} / ${modelTotal.toLocaleString()}`
        : "— / —";
      benchmarkProgressModel.textContent =
        progress.current_model || "Waiting for benchmark";
      benchmarkProgressPromptCount.textContent = promptTotal
        ? `${promptCurrent.toLocaleString()} / ${promptTotal.toLocaleString()}`
        : "— / —";
      benchmarkProgressPrompt.textContent = progress.current_prompt
        ? `${progress.current_prompt}${progress.current_category
          ? ` · ${progress.current_category}`
          : ""}`
        : "Waiting for prompt";
      benchmarkProgressElapsed.textContent =
        formatBenchmarkDuration(progress.elapsed_seconds);

      benchmarkProgressActivity.classList.toggle(
        "running",
        state === "running" || state === "starting"
      );
      if (state === "running" && progress.current_prompt) {
        benchmarkProgressActivity.textContent =
          `Running “${progress.current_prompt}” on ${progress.current_model}. `
          + "Progress advances as prompts finish.";
      } else if (state === "running" && progress.current_model) {
        benchmarkProgressActivity.textContent =
          `${progress.stage || "Preparing"} · ${progress.current_model}`;
      } else if (state === "complete") {
        benchmarkProgressActivity.textContent =
          `Finished ${totalSteps.toLocaleString()} prompt run${totalSteps === 1 ? "" : "s"} in `
          + `${formatBenchmarkDuration(progress.elapsed_seconds)}.`;
      } else if (state === "failed") {
        benchmarkProgressActivity.textContent =
          "The benchmark stopped with an error. Review the latest console lines.";
      } else if (state === "starting") {
        benchmarkProgressActivity.textContent =
          "Starting the benchmark process and reading its execution plan…";
      } else {
        benchmarkProgressActivity.textContent =
          "Start a benchmark to see live progress.";
      }
    }

    async function refreshJob() {
      const job = await api("/api/job");
      const benchmarkCompleted = benchmarkJobRunning && !job.running;
      benchmarkJobRunning = Boolean(job.running);
      const state = job.progress?.state || (job.running ? "running" : "idle");
      jobState.textContent = {
        idle: "Idle",
        starting: "Starting",
        running: "Running",
        complete: "Complete",
        failed: "Failed"
      }[state] || "Idle";
      startButton.disabled = job.running;
      renderBenchmarkProgress(job);
      const lines = Array.isArray(job.lines) ? job.lines : [];
      const output = lines.join("\n");
      if (logOutput.textContent !== output) {
        const previousScrollTop = logOutput.scrollTop;
        logOutput.textContent = output;
        logOutput.scrollTop = followConsoleOutput.checked
          ? logOutput.scrollHeight
          : previousScrollTop;
      }
      consoleLineCount.textContent =
        `${lines.length.toLocaleString()} line${lines.length === 1 ? "" : "s"}`;
      if (benchmarkCompleted && job.returncode === 0) {
        refreshRuns().catch(error => {
          errorText.textContent = error.message;
        });
        refreshModelStatistics().catch(error => {
          statisticsErrorText.textContent = error.message;
        });
      }
      return job;
    }

    async function refreshRuns() {
      const data = await api("/api/runs");
      renderRuns(data.runs);
    }

    async function refreshModelStatistics() {
      statisticsErrorText.textContent = "";
      refreshStatisticsButton.disabled = true;
      try {
        const data = await api("/api/model-statistics");
        renderModelStatistics(data);
        statisticsLoaded = true;
        return data;
      } finally {
        refreshStatisticsButton.disabled = false;
      }
    }

    async function refreshTelemetry(page = telemetryQueryPage) {
      telemetryErrorText.textContent = "";
      refreshTelemetryButton.disabled = true;
      try {
        const windowSeconds = Number(telemetryWindow.value || 86400);
        const pageSize = Number(telemetryQueryPageSize.value || 25);
        const data = await api(
          `/api/telemetry?window_seconds=${encodeURIComponent(windowSeconds)}&limit=150`
          + `&request_page=${encodeURIComponent(page)}`
          + `&request_page_size=${encodeURIComponent(pageSize)}`
        );
        renderTelemetry(data);
        telemetryLoaded = true;
        return data;
      } finally {
        refreshTelemetryButton.disabled = false;
      }
    }

    harnessForm.addEventListener("submit", async event => {
      event.preventDefault();
      const prompt = harnessInput.value.trim();
      harnessErrorText.textContent = "";
      if (!prompt || harnessGenerating) return;
      if (!harnessServerReady) {
        harnessErrorText.textContent = "Load an MLX model before sending a message.";
        return;
      }

      harnessMessages.push({role: "user", content: prompt});
      harnessInput.value = "";
      const requestMessages = harnessRequestMessages();
      const assistant = {
        role: "assistant",
        content: "",
        reasoning: "",
        model: harnessActiveModel,
        pending: true,
        usage: null
      };
      harnessMessages.push(assistant);
      renderHarnessConversation();
      setHarnessGenerating(true);
      harnessStatus.textContent = "Generating streamed response…";

      try {
        await streamHarnessCompletion(requestMessages, assistant);
        assistant.pending = false;
        const usageSummary = harnessUsageSummary(assistant);
        if (assistant.stopReason === "repetition") {
          harnessStatus.textContent = usageSummary
            ? `Stopped: repeated output detected · ${usageSummary}`
            : "Stopped: repeated output detected";
        } else if (assistant.finishReason === "length") {
          harnessStatus.textContent = usageSummary
            ? `Stopped at max-token limit · ${usageSummary}`
            : "Stopped at max-token limit";
        } else {
          harnessStatus.textContent = usageSummary
            ? `Complete · ${usageSummary}`
            : "Complete";
        }
      } catch (error) {
        assistant.pending = false;
        const cancelled = error.name === "AbortError";
        if (!assistant.content && !assistant.reasoning) {
          harnessMessages = harnessMessages.filter(message => message !== assistant);
        }
        harnessStatus.textContent = cancelled ? "Generation cancelled" : "Generation stopped";
        if (!cancelled) {
          harnessErrorText.textContent = error.message;
        }
      } finally {
        harnessAbortController = null;
        setHarnessGenerating(false);
        renderHarnessConversation();
        harnessInput.focus();
      }
    });

    harnessCancelButton.addEventListener("click", async () => {
      if (!harnessGenerating) return;
      harnessStatus.textContent = "Cancelling generation…";
      harnessAbortController?.abort();
      try {
        await api("/api/mlx-server/cancel", {
          method: "POST",
          headers: {"content-type": "application/json"},
          body: JSON.stringify(activeRequestId ? {request_id: activeRequestId} : {})
        });
      } catch (error) {
        harnessErrorText.textContent = error.message;
      }
    });

    harnessClearButton.addEventListener("click", () => {
      harnessMessages = [];
      harnessErrorText.textContent = "";
      harnessStatus.textContent = harnessServerReady
        ? "Ready · press Enter to send"
        : "Waiting for an active model";
      renderHarnessConversation();
      harnessInput.focus();
    });

    harnessInput.addEventListener("keydown", event => {
      if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
        event.preventDefault();
        if (!harnessSendButton.disabled) {
          harnessForm.requestSubmit();
        }
      }
    });

    operationsTabButton.addEventListener("click", () => setActiveTab("operations"));
    telemetryTabButton.addEventListener("click", () => setActiveTab("telemetry"));
    benchmarkTabButton.addEventListener("click", () => setActiveTab("benchmark"));
    harnessTabButton.addEventListener("click", () => setActiveTab("harness"));
    settingsTabButton.addEventListener("click", () => setActiveTab("settings"));
    authForm.addEventListener("submit", async event => {
      event.preventDefault();
      authErrorText.textContent = "";
      const candidate = authApiKey.value.trim();
      if (!candidate) {
        authErrorText.textContent = "Enter the Control Center API key.";
        return;
      }
      try {
        await unlockControlCenter(candidate);
      } catch (error) {
        authErrorText.textContent = error.message;
      }
    });
    lockControlCenterButton.addEventListener("click", () => {
      rememberApiKey("");
      showAuthenticationGate();
    });
    networkAccessMode.addEventListener("change", () => {
      if (!networkAccessMode.checked) return;
      localAuthenticationPreference = apiKeyAuthentication.checked;
      apiKeyAuthentication.checked = true;
      apiKeyAuthentication.disabled = true;
      if (!apiKeyConfigured && !replacementApiKey.value) {
        generateReplacementApiKey();
      }
    });
    localAccessMode.addEventListener("change", () => {
      if (!localAccessMode.checked) return;
      apiKeyAuthentication.disabled = false;
      apiKeyAuthentication.checked = localAuthenticationPreference;
    });
    apiKeyAuthentication.addEventListener("change", () => {
      if (localAccessMode.checked) {
        localAuthenticationPreference = apiKeyAuthentication.checked;
      }
    });
    function generateReplacementApiKey() {
      const bytes = new Uint8Array(32);
      crypto.getRandomValues(bytes);
      const generated = btoa(String.fromCharCode(...bytes))
        .replaceAll("+", "-")
        .replaceAll("/", "_")
        .replaceAll("=", "");
      replacementApiKey.type = "text";
      replacementApiKey.value = generated;
      replacementApiKey.focus();
      replacementApiKey.select();
    }
    generateApiKeyButton.addEventListener("click", generateReplacementApiKey);
    accessSettingsForm.addEventListener("submit", async event => {
      event.preventDefault();
      settingsErrorText.textContent = "";
      saveAccessSettingsButton.disabled = true;
      const replacement = replacementApiKey.value.trim();
      try {
        if (
          (apiKeyAuthentication.checked || modelApiKeyAuthentication.checked)
          && !apiKeyConfigured
          && !replacement
        ) {
          throw new Error("Enter or generate a replacement key before enabling authentication.");
        }
        const payload = {
          access_mode: networkAccessMode.checked ? "network" : "local",
          port: Number(controlCenterPort.value),
          authentication: apiKeyAuthentication.checked ? "api_key" : "none",
          model_api_authentication: modelApiKeyAuthentication.checked ? "api_key" : "none"
        };
        if (
          replacement
          && (apiKeyAuthentication.checked || modelApiKeyAuthentication.checked)
        ) {
          payload.api_key = replacement;
        }
        const data = await api("/api/settings/access", {
          method: "POST",
          headers: {"content-type": "application/json"},
          body: JSON.stringify(payload)
        });
        replacementApiKey.value = "";
        replacementApiKey.type = "password";
        renderAccessSettings(data);
      } catch (error) {
        settingsErrorText.textContent = error.message;
      } finally {
        saveAccessSettingsButton.disabled = false;
      }
    });
    restartMlxServerButton.addEventListener("click", async () => {
      if (!window.confirm(
        "Restart the MLX model server? Active requests will stop and retained prompt cache will be cleared."
      )) return;
      restartMlxServerButton.disabled = true;
      serviceRestartStatus.hidden = false;
      serviceRestartStatus.classList.remove("settings-warning");
      serviceRestartStatus.textContent = "Restarting the MLX model server…";
      try {
        await restartManagedMlxServer();
        serviceRestartStatus.textContent = "MLX model server restarted and is ready.";
      } catch (error) {
        serviceRestartStatus.classList.add("settings-warning");
        serviceRestartStatus.textContent = error.message;
      } finally {
        restartMlxServerButton.disabled = false;
      }
    });
    restartControlCenterButton.addEventListener("click", async () => {
      if (!window.confirm(
        "Restart the Control Center? This page will reconnect using the saved access settings."
      )) return;
      restartControlCenterButton.disabled = true;
      serviceRestartStatus.hidden = false;
      serviceRestartStatus.classList.remove("settings-warning");
      serviceRestartStatus.textContent = "Restarting the Control Center…";
      try {
        const data = await api("/api/control-center/restart", {method: "POST"});
        const nextUrl = new URL(window.location.href);
        if (data.access_mode === "local") nextUrl.hostname = "127.0.0.1";
        nextUrl.port = String(data.port || nextUrl.port);
        window.setTimeout(() => window.location.assign(nextUrl.toString()), 1400);
      } catch (error) {
        restartControlCenterButton.disabled = false;
        serviceRestartStatus.classList.add("settings-warning");
        serviceRestartStatus.textContent = error.message;
      }
    });
    runsBody.addEventListener("click", event => {
      const link = event.target.closest("[data-artifact]");
      if (!link) return;
      event.preventDefault();
      openArtifact(link.dataset.artifact).catch(error => {
        errorText.textContent = error.message;
      });
    });
    refreshTelemetryButton.addEventListener("click", () => {
      refreshTelemetry(1).catch(error => {
        telemetryErrorText.textContent = error.message;
      });
    });
    telemetryWindow.addEventListener("change", () => {
      telemetryFailureBucket = null;
      refreshTelemetry(1).catch(error => {
        telemetryErrorText.textContent = error.message;
      });
    });
    telemetryQueryPageSize.addEventListener("change", () => {
      refreshTelemetry(1).catch(error => {
        telemetryErrorText.textContent = error.message;
      });
    });
    telemetryQueryPrevious.addEventListener("click", () => {
      refreshTelemetry(Math.max(1, telemetryQueryPage - 1)).catch(error => {
        telemetryErrorText.textContent = error.message;
      });
    });
    telemetryQueryNext.addEventListener("click", () => {
      refreshTelemetry(telemetryQueryPage + 1).catch(error => {
        telemetryErrorText.textContent = error.message;
      });
    });
    telemetryChart.addEventListener("click", event => {
      const bucket = event.target.closest("[data-failure-bucket-start]");
      if (!bucket) return;
      telemetryFailureBucket = {
        start: Number(bucket.dataset.failureBucketStart || 0),
        end: Number(bucket.dataset.failureBucketEnd || 0)
      };
      renderTelemetryFailures();
      telemetryFailurePanel.scrollIntoView({behavior: "smooth", block: "start"});
    });
    telemetryFailureClearFilter.addEventListener("click", () => {
      telemetryFailureBucket = null;
      renderTelemetryFailures();
    });
    telemetryModelToggle.addEventListener("click", () => {
      telemetryModelRows.classList.toggle("expanded");
      updateTelemetryActivityToggle(
        telemetryModelToggle,
        telemetryModelRows,
        telemetryModelRows.children.length,
        "models"
      );
    });
    telemetryToolToggle.addEventListener("click", () => {
      telemetryToolRows.classList.toggle("expanded");
      updateTelemetryActivityToggle(
        telemetryToolToggle,
        telemetryToolRows,
        telemetryToolRows.children.length,
        "tools"
      );
    });
    refreshStatisticsButton.addEventListener("click", () => {
      refreshModelStatistics().catch(error => {
        statisticsErrorText.textContent = error.message;
      });
    });
    const tabOrder = [
      {name: "operations", button: operationsTabButton},
      {name: "telemetry", button: telemetryTabButton},
      {name: "benchmark", button: benchmarkTabButton},
      {name: "harness", button: harnessTabButton},
      {name: "settings", button: settingsTabButton}
    ];
    for (const [index, tab] of tabOrder.entries()) {
      tab.button.addEventListener("keydown", event => {
        if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
        event.preventDefault();
        const offset = event.key === "ArrowRight" ? 1 : -1;
        const next = tabOrder[(index + offset + tabOrder.length) % tabOrder.length];
        setActiveTab(next.name);
        next.button.focus();
      });
    }

    allModels.addEventListener("change", () => {
      modelSelect.disabled = allModels.checked;
    });

    form.addEventListener("submit", async event => {
      event.preventDefault();
      errorText.textContent = "";
      followConsoleOutput.checked = true;
      try {
        await api("/api/benchmark", {
          method: "POST",
          headers: {"content-type": "application/json"},
          body: JSON.stringify({
            prompts: promptSelect.value,
            all_models: allModels.checked,
            model_key: modelSelect.value,
            max_tokens: maxTokens.value,
            limit: limit.value,
            output_dir: outputDir.value
          })
        });
        await refreshJob();
      } catch (error) {
        errorText.textContent = error.message;
      }
    });

    cancelRequestButton.addEventListener("click", async () => {
      if (!activeRequestId) return;
      safetyErrorText.textContent = "";
      cancelRequestButton.disabled = true;
      try {
        const data = await api("/api/mlx-server/cancel", {
          method: "POST",
          headers: {"content-type": "application/json"},
          body: JSON.stringify({request_id: activeRequestId})
        });
        renderServerStatus(data.status_payload || data);
      } catch (error) {
        safetyErrorText.textContent = error.message;
      } finally {
        cancelRequestButton.disabled = false;
      }
    });

    saveSafetyButton.addEventListener("click", async () => {
      safetyErrorText.textContent = "";
      saveSafetyButton.disabled = true;
      try {
        const data = await api("/api/mlx-server/safety", {
          method: "POST",
          headers: {"content-type": "application/json"},
          body: JSON.stringify({
            enabled: guardEnabled.checked,
            max_context_tokens: Number(contextCap.value),
            max_output_tokens: Number(outputCap.value),
            memory_reserve_bytes: Math.round(Number(reserveGib.value) * (1024 ** 3)),
            memory_abort_bytes: Math.round(Number(abortGib.value) * (1024 ** 3))
          })
        });
        safetyFormModel = "";
        renderServerStatus(data.status_payload || data);
      } catch (error) {
        safetyErrorText.textContent = error.message;
      } finally {
        saveSafetyButton.disabled = false;
      }
    });

    clearCacheButton.addEventListener("click", async () => {
      safetyErrorText.textContent = "";
      clearCacheButton.disabled = true;
      try {
        const data = await api("/api/mlx-server/cache/clear", {
          method: "POST",
          headers: {"content-type": "application/json"},
          body: "{}"
        });
        renderServerStatus(data.status_payload || data);
      } catch (error) {
        safetyErrorText.textContent = error.message;
        clearCacheButton.disabled = false;
      }
    });

    cacheRows.addEventListener("click", async event => {
      const button = event.target.closest(".cache-remove");
      if (!button) return;
      safetyErrorText.textContent = "";
      button.disabled = true;
      try {
        const data = await api("/api/mlx-server/cache/clear", {
          method: "POST",
          headers: {"content-type": "application/json"},
          body: JSON.stringify({entry_id: button.dataset.entryId})
        });
        renderServerStatus(data.status_payload || data);
      } catch (error) {
        safetyErrorText.textContent = error.message;
        button.disabled = false;
      }
    });

    loadServerButton.addEventListener("click", async () => {
      serverErrorText.textContent = "";
      setServerButtonsDisabled(true);
      try {
        const data = await api("/api/mlx-server/switch", {
          method: "POST",
          headers: {"content-type": "application/json"},
          body: JSON.stringify({model: serverModelSelect.value})
        });
        renderServerStatus(data.status_payload || data);
      } catch (error) {
        serverErrorText.textContent = error.message;
      } finally {
        setServerButtonsDisabled(false);
      }
    });

    unloadServerButton.addEventListener("click", async () => {
      serverErrorText.textContent = "";
      setServerButtonsDisabled(true);
      try {
        const data = await api("/api/mlx-server/unload", {method: "POST"});
        renderServerStatus(data.status_payload || data);
      } catch (error) {
        serverErrorText.textContent = error.message;
      } finally {
        setServerButtonsDisabled(false);
      }
    });

    restartServerButton.addEventListener("click", async () => {
      serverErrorText.textContent = "";
      setServerButtonsDisabled(true);
      try {
        await restartManagedMlxServer();
      } catch (error) {
        serverErrorText.textContent = error.message;
      } finally {
        setServerButtonsDisabled(false);
      }
    });

    refreshServerButton.addEventListener("click", async () => {
      serverErrorText.textContent = "";
      setServerButtonsDisabled(true);
      try {
        await refreshServerStatus();
      } catch (error) {
        serverErrorText.textContent = error.message;
      } finally {
        setServerButtonsDisabled(false);
      }
    });

    async function init() {
      if (initialized) return;
      initialized = true;
      try {
        const state = await api("/api/state");
        renderState(state);
        await refreshServerStatus();
        await refreshJob();
        setInterval(refreshJob, 2000);
        setInterval(refreshRuns, 5000);
        setInterval(() => {
          refreshServerStatus().catch(error => {
            serverErrorText.textContent = error.message;
          });
        }, 2000);
        setInterval(() => {
          if (activeTabName !== "telemetry") return;
          refreshTelemetry().catch(error => {
            telemetryErrorText.textContent = error.message;
          });
        }, 5000);
      } catch (error) {
        initialized = false;
        throw error;
      }
    }

    async function bootstrap() {
      const status = await authenticationStatus();
      if (status.authentication_required && !status.authenticated) {
        showAuthenticationGate();
        return;
      }
      mainContent.inert = false;
      authGate.hidden = true;
      await init();
    }

    bootstrap().catch(error => {
      showAuthenticationGate(error.message);
    });
  </script>
</body>
</html>
"""


class ControlPlaneHandler(BaseHTTPRequestHandler):
    server_version = "MLXLabControlPlane/0.1"

    def log_message(self, format: str, *args: Any) -> None:
        return

    def send_json(self, data: dict[str, Any], status: int = 200) -> None:
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("cache-control", "no-store")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def send_html(self, html: str) -> None:
        payload = html.encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "text/html; charset=utf-8")
        self.send_header("cache-control", "no-store")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("content-length", "0"))
        payload = self.rfile.read(length)
        body = json.loads(payload or b"{}")
        if not isinstance(body, dict):
            raise ValueError("JSON body must be an object.")
        return body

    def supplied_api_key(self) -> str:
        authorization = str(self.headers.get("authorization") or "")
        scheme, _, value = authorization.partition(" ")
        if scheme.casefold() == "bearer":
            return value.strip()
        return str(self.headers.get("x-api-key") or "").strip()

    def is_authenticated(self) -> bool:
        active_settings = getattr(
            self.server,
            "control_center_access",
            None,
        )
        return valid_control_center_api_key(
            self.supplied_api_key(),
            active_settings,
        )

    def require_authentication(self) -> bool:
        if self.is_authenticated():
            return True
        self.send_response(401)
        payload = json.dumps(
            {"error": "A valid Control Center API key is required."}
        ).encode("utf-8")
        self.send_header("content-type", "application/json")
        self.send_header("cache-control", "no-store")
        self.send_header("www-authenticate", 'Bearer realm="MLX Control Center"')
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
        return False

    def access_payload(self) -> dict[str, Any]:
        active_host, active_port = self.server.server_address[:2]
        active_settings = getattr(self.server, "control_center_access", None)
        if active_settings is None:
            active_settings = control_center_settings()
        runtime = read_runtime_state()
        return control_center_access_payload(
            active_host=str(active_host),
            active_port=int(active_port),
            active_authentication=active_settings["authentication"],
            active_api_key=active_settings.get("api_key"),
            active_model_api_authentication=runtime.get(
                "model_api_authentication"
            ),
        )

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            self.send_html(CONTROL_PLANE_HTML)
            return
        if parsed.path == "/api/auth/status":
            payload = self.access_payload()
            payload["authenticated"] = self.is_authenticated()
            self.send_json(payload)
            return
        if not self.require_authentication():
            return
        if parsed.path == "/api/settings/access":
            self.send_json(self.access_payload())
            return
        if parsed.path == "/api/state":
            self.send_json(
                {
                    "models": model_options(),
                    "prompts": prompt_files(),
                    "runs": recent_runs(),
                }
            )
            return
        if parsed.path == "/api/job":
            self.send_json(JOB.snapshot())
            return
        if parsed.path == "/api/runs":
            self.send_json({"runs": recent_runs()})
            return
        if parsed.path == "/api/model-statistics":
            self.send_json(model_statistics_payload())
            return
        if parsed.path == "/api/telemetry":
            query = urllib.parse.parse_qs(parsed.query)
            try:
                window_seconds = int(query.get("window_seconds", ["86400"])[0])
                limit = int(query.get("limit", ["100"])[0])
                request_page = int(query.get("request_page", ["1"])[0])
                request_page_size = int(
                    query.get("request_page_size", [str(min(limit, 100))])[0]
                )
                payload = telemetry_payload(
                    window_seconds=window_seconds,
                    limit=limit,
                    request_page=request_page,
                    request_page_size=request_page_size,
                )
            except (TypeError, ValueError) as exc:
                self.send_json({"error": str(exc)}, status=400)
                return
            self.send_json(payload)
            return
        if parsed.path in ("/api/mlx-server/status", "/api/mlx-server/current"):
            payload = mlx_server_status_payload()
            self.send_json(payload)
            return
        if parsed.path in ("/api/mlx-server/models", "/api/mlx-server/list"):
            payload = mlx_server_models_payload()
            self.send_json(payload, status=200 if payload["ok"] else 500)
            return
        if parsed.path == "/api/mlx-server/help":
            self.send_json(mlx_server_help_payload())
            return
        if parsed.path.startswith("/artifact/"):
            self.send_artifact(parsed.path.removeprefix("/artifact/"))
            return
        self.send_json({"error": "Not found"}, status=404)

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if not self.require_authentication():
            return
        if parsed.path == "/api/settings/access":
            try:
                update_control_center_settings(self.read_json_body())
            except (ValueError, json.JSONDecodeError) as exc:
                self.send_json({"error": str(exc)}, status=400)
                return
            self.send_json(self.access_payload())
            return
        if parsed.path == "/api/control-center/restart":
            access = control_center_settings()
            self.send_json(
                {
                    "ok": True,
                    "access_mode": access["access_mode"],
                    "port": access["port"],
                    "message": "Control Center restart scheduled.",
                }
            )
            schedule_control_center_restart()
            return
        if parsed.path == "/api/benchmark":
            self.handle_benchmark_post()
            return
        if parsed.path == "/api/harness/chat":
            self.handle_harness_chat_post()
            return
        if parsed.path in (
            "/api/mlx-server/load",
            "/api/mlx-server/switch",
            "/api/mlx-server/unload",
            "/api/mlx-server/restart",
        ):
            self.handle_mlx_server_post(parsed.path.rsplit("/", 1)[-1])
            return
        runtime_actions = {
            "/api/mlx-server/cancel": "cancel",
            "/api/mlx-server/safety": "safety",
            "/api/mlx-server/cache/clear": "cache_clear",
        }
        if parsed.path in runtime_actions:
            self.handle_mlx_runtime_post(runtime_actions[parsed.path])
            return

        self.send_json({"error": "Not found"}, status=404)

    def handle_benchmark_post(self) -> None:
        try:
            body = self.read_json_body()
            command = build_benchmark_command(body)
            JOB.start(command)
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=400)
            return

        self.send_json({"ok": True, "command": command})

    def handle_harness_chat_post(self) -> None:
        try:
            request = build_harness_chat_request(self.read_json_body())
            response = urllib.request.urlopen(request, timeout=mlx_harness_timeout())
        except urllib.error.HTTPError as exc:
            try:
                payload = json.loads(exc.read().decode("utf-8"))
            except (UnicodeDecodeError, ValueError, TypeError):
                payload = {}
            error = payload.get("error") if isinstance(payload, dict) else None
            if isinstance(error, dict):
                error = error.get("message")
            self.send_json(
                {"error": str(error or f"MLX server returned HTTP {exc.code}.")},
                status=exc.code,
            )
            return
        except (
            RuntimeError,
            ValueError,
            json.JSONDecodeError,
            OSError,
            urllib.error.URLError,
        ) as exc:
            self.send_json({"error": str(exc)}, status=400)
            return

        self.send_response(200)
        self.send_header("content-type", "text/event-stream; charset=utf-8")
        self.send_header("cache-control", "no-cache")
        self.send_header("connection", "close")
        self.end_headers()
        self.close_connection = True
        try:
            with response:
                for line in response:
                    self.wfile.write(line)
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return

    def handle_mlx_server_post(self, action: str) -> None:
        try:
            model_key = None
            model: dict[str, Any] | None = None
            if action in ("load", "switch", "restart"):
                body = self.read_json_body()
                model_key, model = resolve_server_model_input(str(body.get("model") or ""))

            result = run_mlx_server_command(action, model_key)
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=400)
            return

        if result["returncode"] != 0 or result["timed_out"]:
            error = result["stderr"] or result["stdout"] or "MLX server command failed."
            self.send_json(
                {
                    "ok": False,
                    "error": error,
                    "action": action,
                    "result": result,
                },
                status=500,
            )
            return

        status_payload = mlx_server_status_payload()
        self.send_json(
            {
                "ok": True,
                "action": action,
                "model": (
                    {
                        "key": model_key,
                        "label": model_label(model_key, model),
                        "repo_id": model["repo_id"],
                        "local_dir": model["local_dir"],
                    }
                    if model_key and model
                    else None
                ),
                "result": result,
                "status_payload": status_payload,
            }
        )

    def handle_mlx_runtime_post(self, action: str) -> None:
        try:
            body = self.read_json_body()
            result = mlx_internal_request(
                action,
                method="POST",
                payload=body,
            )
        except (RuntimeError, ValueError, json.JSONDecodeError) as exc:
            self.send_json({"error": str(exc)}, status=400)
            return

        self.send_json(
            {
                "ok": True,
                "action": action,
                "result": result,
                "status_payload": mlx_server_status_payload(),
            }
        )

    def send_artifact(self, artifact_path: str) -> None:
        try:
            relative = Path(urllib.parse.unquote(artifact_path))
            namespace, *parts = relative.parts
            if namespace == "runs":
                base = RUNS_DIR.resolve()
            elif namespace == "workspace":
                base = ROOT.resolve()
            else:
                raise ValueError("Missing artifact namespace")
            target = base.joinpath(*parts).resolve()
            target.relative_to(base)
        except Exception:
            self.send_json({"error": "Invalid artifact path"}, status=400)
            return

        if not target.is_file():
            self.send_json({"error": "Not found"}, status=404)
            return

        mime_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        payload = target.read_bytes()
        self.send_response(200)
        self.send_header("content-type", mime_type)
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def serve_control_plane(host: str | None = None, port: int | None = None) -> None:
    access = control_center_settings()
    if access["authentication"] == "api_key":
        access = {
            **access,
            "api_key": control_center_settings(include_secret=True)["api_key"],
        }
    bind_host = host or access["bind_host"]
    bind_port = port or access["port"]
    server = ThreadingHTTPServer((bind_host, bind_port), ControlPlaneHandler)
    server.control_center_access = access
    display_host = "127.0.0.1" if bind_host == "0.0.0.0" else bind_host
    url = f"http://{display_host}:{bind_port}"
    print(f"MLX Lab control plane: {url}")
    if access["authentication"] == "api_key":
        print(f"Control Center API key: {access['api_key']}")
        print("Authentication: required (Authorization: Bearer <api-key>)")
    else:
        print("Authentication: disabled for local-only access")
    if bind_host == "0.0.0.0":
        print("Network access: listening on all interfaces")
    print("Press Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
