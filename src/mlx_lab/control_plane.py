from __future__ import annotations

import json
import mimetypes
import os
import re
import subprocess
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

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
MLX_SERVER_MUTATING_COMMANDS = {"load", "switch", "unload"}
MLX_SERVER_COMMANDS = {"current", "list", *MLX_SERVER_MUTATING_COMMANDS}
MLX_SERVER_LOCK = threading.Lock()


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
        "default_model": line_value(lines, "Default model: "),
        "command_line": line_value(lines, "Command: "),
    }


def mlx_server_status_payload() -> dict[str, Any]:
    result = run_mlx_server_command("current")
    return {
        "ok": result["returncode"] == 0 and not result["timed_out"],
        "status": parse_mlx_server_current(result["stdout"]),
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
        ],
    }


CONTROL_PLANE_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MLX Lab Control Plane</title>
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
      .row { grid-template-columns: 1fr; }
      .bar { grid-template-columns: 1fr; gap: 4px; }
      th, td { white-space: normal; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>MLX Lab Control Plane</h1>
      <div class="status"><span>Job</span><strong id="jobState">Idle</strong></div>
    </header>

    <div class="grid">
      <div class="stack">
        <section>
          <h2>Benchmark</h2>
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

        <section>
          <h2>MLX Server</h2>
          <label for="serverModelSelect">Model</label>
          <select id="serverModelSelect"></select>
          <div class="button-row">
            <button id="loadServerButton" type="button">Load / Switch</button>
            <button id="unloadServerButton" class="danger" type="button">Unload</button>
          </div>
          <button id="refreshServerButton" class="secondary" type="button">Refresh Status</button>
          <pre id="serverStatusOutput" class="mini-pre"></pre>
          <div class="error" id="serverErrorText"></div>
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
    const unloadServerButton = document.querySelector("#unloadServerButton");
    const refreshServerButton = document.querySelector("#refreshServerButton");
    const serverStatusOutput = document.querySelector("#serverStatusOutput");
    const logOutput = document.querySelector("#logOutput");
    const runsBody = document.querySelector("#runsBody");
    const bars = document.querySelector("#bars");

    async function api(path, options) {
      const response = await fetch(path, options);
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Request failed");
      return data;
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

    function renderServerStatus(data) {
      const status = data.status || {};
      const listeners = status.listener_pids && status.listener_pids.length
        ? status.listener_pids.join(", ")
        : "none";
      serverStatusOutput.textContent = [
        `HTTP: ${status.http_ready ? "ready" : "not ready"} ${status.endpoint || ""}`.trim(),
        `Managed PID: ${status.managed_pid || "none"}`,
        `Listener PID(s): ${listeners}`,
        `Loaded: ${status.loaded_model || "unknown"}`,
        status.default_model ? `Default: ${status.default_model}` : "",
      ].filter(Boolean).join("\n");
    }

    function setServerButtonsDisabled(disabled) {
      loadServerButton.disabled = disabled;
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
