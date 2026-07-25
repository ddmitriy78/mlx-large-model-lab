from __future__ import annotations

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
    ROOT,
    RUNS_DIR,
    configured_models,
    model_is_downloaded,
    model_label,
    read_benchmark_records,
    summarize_benchmark_records,
)

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
PROMPT_PROGRESS_RE = re.compile(
    r"Prompt processing progress:\s+(?P<current>\d+)/(?P<total>\d+)"
)
PROMPT_CACHE_RE = re.compile(
    r"Prompt Cache:\s+(?P<sequences>\d+) sequences,\s+(?P<gigabytes>[\d.]+) GB"
)


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
            running = self.process is not None and self.process.poll() is None
            return {
                "running": running,
                "command": self.command,
                "lines": self.lines[-500:],
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "returncode": self.returncode,
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
MLX_SERVER_LOG = ROOT / "logs" / "mlx-server.log"
MLX_SERVER_RUNTIME_STATE = ROOT / "logs" / "mlx-server-runtime.json"
MLX_SERVER_MUTATING_COMMANDS = {"load", "switch", "unload"}
MLX_SERVER_COMMANDS = {"current", "list", *MLX_SERVER_MUTATING_COMMANDS}
MLX_SERVER_LOCK = threading.Lock()
MLX_INTERNAL_PATHS = {
    "status": "/mlx-lab/status",
    "cancel": "/mlx-lab/requests/cancel",
    "safety": "/mlx-lab/safety",
    "cache_clear": "/mlx-lab/cache/clear",
}


def rel_path(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def artifact_href(path: Path) -> str:
    return "/artifact/" + urllib.parse.quote(rel_path(path))


def prompt_files() -> list[dict[str, Any]]:
    prompts_dir = ROOT / "prompts"
    files = []
    for path in sorted(prompts_dir.glob("*.jsonl")):
        count = 0
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    count += 1
        files.append({"path": rel_path(path), "name": path.stem, "count": count})
    return files


def model_options() -> list[dict[str, Any]]:
    models = []
    for key, model in configured_models():
        downloaded = model_is_downloaded(model)
        models.append(
            {
                "key": key,
                "label": model_label(key, model),
                "repo_id": model["repo_id"],
                "local_dir": model["local_dir"],
                "downloaded": downloaded,
            }
        )
    return models


def recent_runs() -> list[dict[str, Any]]:
    RUNS_DIR.mkdir(exist_ok=True)
    runs = []
    for jsonl_path in sorted(RUNS_DIR.glob("*.jsonl"), reverse=True)[:20]:
        html_path = jsonl_path.with_suffix(".html")
        try:
            records = read_benchmark_records(jsonl_path)
            summaries = summarize_benchmark_records(records) if records else []
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


def allowed_prompt_paths() -> set[str]:
    return {item["path"] for item in prompt_files()}


def validate_output_dir(value: str) -> str:
    if not value:
        return ""
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Output directory must be a relative path inside the workspace.")
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


def mlx_internal_request(
    action: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: float = 3.0,
) -> dict[str, Any]:
    if action not in MLX_INTERNAL_PATHS:
        raise ValueError("Unsupported MLX runtime action.")
    try:
        port = int(os.environ.get("MLX_SERVER_PORT", "48080"))
    except ValueError as exc:
        raise ValueError("MLX_SERVER_PORT must be an integer.") from exc

    body = None
    headers: dict[str, str] = {}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["content-type"] = "application/json"
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{MLX_INTERNAL_PATHS[action]}",
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
        local = ROOT / model["local_dir"]
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
        local = ROOT / model["local_dir"]
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
    input {
      width: 100%;
      height: 38px;
      padding: 7px 9px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--text);
      font: inherit;
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
      main { width: min(100vw - 20px, 760px); margin-top: 14px; }
      header { align-items: stretch; flex-direction: column; }
      .status { text-align: left; }
      .grid { grid-template-columns: 1fr; }
      .metric-grid { grid-template-columns: 1fr 1fr; }
      .metric:first-child { grid-column: 1 / -1; }
      .control-grid { grid-template-columns: 1fr; }
      .safety-grid { grid-template-columns: 1fr; }
      .row { grid-template-columns: 1fr; }
      .bar { grid-template-columns: 1fr; gap: 4px; }
      th, td { white-space: normal; }
    }
    @media (max-width: 520px) {
      .metric-grid { grid-template-columns: 1fr; }
      .metric:first-child { grid-column: auto; }
      .control-actions { grid-template-columns: 1fr; }
      .field-grid { grid-template-columns: 1fr; }
      .overview-heading { align-items: stretch; flex-direction: column; }
      .request-meta { align-items: stretch; flex-direction: column; }
      .request-meta button { width: 100%; }
      button.compact { width: 100%; }
    }
  </style>
</head>
<body>
  <main>
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

    <div class="grid">
      <div class="stack">
        <section>
          <h2>Benchmark · <span id="jobState">Idle</span></h2>
          <form id="benchmarkForm">
            <label for="promptSelect">Prompt Set</label>
            <select id="promptSelect"></select>

            <label class="check">
              <input id="allModels" type="checkbox" checked>
              All downloaded models
            </label>

            <label for="modelSelect">Model</label>
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
            <input id="outputDir" value="runs/control-plane-outputs">

            <button id="startButton" type="submit">Start Benchmark</button>
            <div class="error" id="errorText"></div>
          </form>
        </section>

      </div>

      <div class="stack">
        <section>
          <h2>Latest Comparison</h2>
          <div id="bars"></div>
        </section>

        <section>
          <h2>Runs</h2>
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

        <section>
          <h2>Console</h2>
          <pre id="logOutput"></pre>
        </section>
      </div>
    </div>
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
    const logOutput = document.querySelector("#logOutput");
    const runsBody = document.querySelector("#runsBody");
    const bars = document.querySelector("#bars");
    let currentServerModelKey = "";
    let activeRequestId = "";
    let safetyFormModel = "";

    async function api(path, options) {
      const response = await fetch(path, options);
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Request failed");
      return data;
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

    function renderState(data) {
      promptSelect.innerHTML = data.prompts.map(item =>
        `<option value="${item.path}">${item.name} (${item.count})</option>`
      ).join("");
      modelSelect.innerHTML = data.models.filter(item => item.downloaded).map(item =>
        `<option value="${item.key}">${item.label}</option>`
      ).join("");
      serverModelSelect.innerHTML = data.models.filter(item => item.downloaded).map(item =>
        `<option value="${item.key}">${item.label}</option>`
      ).join("");
      modelSelect.disabled = allModels.checked;
      renderRuns(data.runs);
    }

    function renderRuns(runs) {
      runsBody.innerHTML = runs.map(run => `
        <tr>
          <td>${run.name}</td>
          <td class="number">${run.record_count}</td>
          <td>${run.html_href ? `<a href="${run.html_href}" target="_blank">open</a>` : ""}</td>
          <td><a href="${run.jsonl_href}" target="_blank">jsonl</a></td>
        </tr>
      `).join("");
      const latest = runs.find(run => run.summaries && run.summaries.length);
      renderBars(latest ? latest.summaries : []);
    }

    function renderBars(summaries) {
      if (!summaries.length) {
        bars.innerHTML = "";
        return;
      }
      const maxRate = Math.max(...summaries.map(item => item.generation_tps || 0), 1);
      bars.innerHTML = summaries.map(item => {
        const width = Math.max(3, Math.round(((item.generation_tps || 0) / maxRate) * 100));
        return `
          <div class="bar">
            <div>${item.label}</div>
            <div class="bar-track"><div class="bar-fill" style="width: ${width}%"></div></div>
            <div class="number">${(item.generation_tps || 0).toFixed(2)}</div>
          </div>
        `;
      }).join("");
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
        const progress = promptTotal
          ? Math.min(100, (promptProcessed / promptTotal) * 100)
          : 0;
        requestTitle.textContent = `Active request · ${statusName}`;
        requestId.textContent = `${activeRequest.id || "unknown"} · output ${outputTokens.toLocaleString()} / ${Number(activeRequest.max_output_tokens || 0).toLocaleString()}`;
        requestText.textContent = statusName === "generating"
          ? `${outputTokens.toLocaleString()} output tokens`
          : `${promptProcessed.toLocaleString()} / ${promptTotal.toLocaleString()} prompt tokens · ${progress.toFixed(1)}%`;
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
      unloadServerButton.disabled = disabled;
      refreshServerButton.disabled = disabled;
    }

    async function refreshServerStatus() {
      const data = await api("/api/mlx-server/status");
      renderServerStatus(data);
      return data;
    }

    async function refreshJob() {
      const job = await api("/api/job");
      jobState.textContent = job.running ? "Running" : (job.returncode === 0 ? "Complete" : "Idle");
      startButton.disabled = job.running;
      logOutput.textContent = job.lines.join("\n");
      logOutput.scrollTop = logOutput.scrollHeight;
      return job;
    }

    async function refreshRuns() {
      const data = await api("/api/runs");
      renderRuns(data.runs);
    }

    allModels.addEventListener("change", () => {
      modelSelect.disabled = allModels.checked;
    });

    form.addEventListener("submit", async event => {
      event.preventDefault();
      errorText.textContent = "";
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
        const data = await api("/api/mlx-server/switch", {
          method: "POST",
          headers: {"content-type": "application/json"},
          body: JSON.stringify({model: currentServerModelKey || serverModelSelect.value})
        });
        renderServerStatus(data.status_payload || data);
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
    }

    init().catch(error => {
      errorText.textContent = error.message;
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
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def send_html(self, html: str) -> None:
        payload = html.encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "text/html; charset=utf-8")
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

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            self.send_html(CONTROL_PLANE_HTML)
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
        if parsed.path in ("/api/mlx-server/status", "/api/mlx-server/current"):
            payload = mlx_server_status_payload()
            self.send_json(payload, status=200 if payload["ok"] else 500)
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
        if parsed.path == "/api/benchmark":
            self.handle_benchmark_post()
            return
        if parsed.path in (
            "/api/mlx-server/load",
            "/api/mlx-server/switch",
            "/api/mlx-server/unload",
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

    def handle_mlx_server_post(self, action: str) -> None:
        try:
            model_key = None
            model: dict[str, Any] | None = None
            if action in ("load", "switch"):
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
            target = (ROOT / relative).resolve()
            target.relative_to(ROOT.resolve())
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


def serve_control_plane(host: str, port: int) -> None:
    server = ThreadingHTTPServer((host, port), ControlPlaneHandler)
    url = f"http://{host}:{port}"
    print(f"MLX Lab control plane: {url}")
    print("Press Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
