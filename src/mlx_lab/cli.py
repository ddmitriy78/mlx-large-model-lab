from __future__ import annotations

import argparse
import gc
import html
import json
import os
import platform
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil
from huggingface_hub import hf_hub_download, snapshot_download
from rich import box
from rich.console import Console
from rich.table import Table

from .model_sizing import (
    GB,
    GIB,
    build_memory_plan,
    model_memory_profile,
    recommended_abort_threshold_bytes,
    recommended_memory_reserve_bytes,
)
from .user_store import (
    capabilities_path,
    control_center_settings,
    merged_config,
    model_is_present,
    project_root,
    register_model_path,
    sync_user_store,
    user_paths,
)

ROOT = project_root()
CONFIG_PATH = capabilities_path()
SOURCE_PROMPTS_DIR = ROOT / "prompts"
PROMPTS_DIR = (
    SOURCE_PROMPTS_DIR
    if SOURCE_PROMPTS_DIR.is_dir()
    else Path(__file__).resolve().parent / "data" / "prompts"
)
PROMPT_PATH = PROMPTS_DIR / "first_model.jsonl"
USER_PATHS = user_paths()
RUNS_DIR = USER_PATHS.runs
LOGS_DIR = USER_PATHS.logs
CACHE_DIR = USER_PATHS.cache

SYSTEM_PROMPT = (
    "You are a local coding and defensive cybersecurity assistant. "
    "Prioritize correctness, practical tradeoffs, and safe defensive guidance."
)

console = Console()


def load_config() -> dict[str, Any]:
    return merged_config()


def first_model_config() -> dict[str, Any]:
    config = load_config()
    return config["models"][config["first_model"]]


def configured_models() -> list[tuple[str, dict[str, Any]]]:
    config = load_config()
    return list(config["models"].items())


def configured_model_map() -> dict[str, dict[str, Any]]:
    return dict(configured_models())


def configured_benchmark_sets() -> dict[str, list[str]]:
    config = load_config()
    return config.get("benchmark_sets", {})


def model_path(model: dict[str, Any]) -> Path:
    path = Path(model["local_dir"]).expanduser()
    return path if path.is_absolute() else ROOT / path


def model_is_downloaded(model: dict[str, Any]) -> bool:
    local = model_path(model)
    return (local / "config.json").exists() and any(local.glob("*.safetensors"))


def resolved_model_ref(model: dict[str, Any]) -> str:
    local = model_path(model)
    if model_is_downloaded(model):
        return str(local)
    raise SystemExit(
        "Primary model is not downloaded yet. Run "
        "`uv run mlx-lab download --dry-run` and then "
        "`uv run mlx-lab download`, or pass --model explicitly."
    )


def model_label(model_key: str, model: dict[str, Any]) -> str:
    aliases = model.get("aliases", [])
    if aliases:
        return aliases[0]
    return model_key


def find_configured_model(model_input: str) -> tuple[str, dict[str, Any]] | None:
    for key, model in configured_models():
        local = model_path(model)
        candidates = {
            key,
            model["repo_id"],
            model["local_dir"],
            str(local),
            local.name,
            *model.get("aliases", []),
        }
        if model_input in candidates:
            return key, model

        input_path = Path(model_input).expanduser()
        if input_path.exists() and input_path.resolve() == local.resolve():
            return key, model
    return None


def benchmark_target(key: str, model: dict[str, Any]) -> dict[str, Any]:
    if model_is_downloaded(model):
        model_ref = str(model_path(model))
    else:
        model_ref = model["repo_id"]
    return {
        "key": key,
        "label": model_label(key, model),
        "repo_id": model["repo_id"],
        "model_ref": model_ref,
    }


def downloaded_benchmark_target(key: str, model: dict[str, Any]) -> dict[str, Any] | None:
    if not model_is_downloaded(model):
        console.print(
            "[yellow]Skipping not downloaded:[/yellow] "
            f"{model_label(key, model)} ({model['local_dir']})"
        )
        return None
    return benchmark_target(key, model)


def benchmark_targets(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.all_models:
        targets = []
        for key, model in configured_models():
            target = downloaded_benchmark_target(key, model)
            if target:
                targets.append(target)
        if not targets:
            raise SystemExit("No downloaded configured models were found.")
        return targets

    config = load_config()
    if args.model_set:
        model_map = config["models"]
        benchmark_sets = config.get("benchmark_sets", {})
        model_keys = benchmark_sets.get(args.model_set)
        if not model_keys:
            raise SystemExit(f"Unknown benchmark model set: {args.model_set}")

        targets = []
        for key in model_keys:
            if key not in model_map:
                raise SystemExit(
                    f"Benchmark set {args.model_set} references unknown model: {key}"
                )
            target = downloaded_benchmark_target(key, model_map[key])
            if target:
                targets.append(target)
        if not targets:
            raise SystemExit(
                f"No downloaded configured models were found in set: {args.model_set}"
            )
        return targets

    if args.model:
        matched = find_configured_model(args.model)
        if matched:
            key, model = matched
            return [benchmark_target(key, model)]
        return [
            {
                "key": args.model,
                "label": Path(args.model).name,
                "repo_id": args.model,
                "model_ref": args.model,
            }
        ]

    key = config["first_model"]
    model = config["models"][key]
    return [
        {
            "key": key,
            "label": model_label(key, model),
            "repo_id": model["repo_id"],
            "model_ref": resolved_model_ref(model),
        }
    ]


def release_model_memory() -> None:
    gc.collect()
    try:
        import mlx.core as mx

        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        else:
            mx.metal.clear_cache()
    except Exception:
        pass


def format_seconds(value: float) -> str:
    return f"{value:.2f}s"


def format_rate(value: float) -> str:
    return f"{value:.2f}"


def html_text(value: Any) -> str:
    return html.escape(str(value), quote=True)


def table_cell(value: Any, *, number: bool = False) -> str:
    class_attr = ' class="number"' if number else ""
    return f"<td{class_attr}>{html_text(value)}</td>"


def workspace_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return ROOT / path


def run_store_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    parts = path.parts[1:] if path.parts[:1] == ("runs",) else path.parts
    user_path = RUNS_DIR.joinpath(*parts)
    legacy_path = ROOT / path
    if legacy_path.exists() and not user_path.exists():
        return legacy_path
    return user_path


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._").lower()
    return slug or "item"


def html_file_link(report_path: Path, file_path: str | None) -> str:
    if not file_path:
        return ""
    path = Path(file_path).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    try:
        href = os.path.relpath(path, report_path.parent)
    except ValueError:
        href = path.as_uri()
    return f'<a href="{html_text(href)}">{html_text(Path(file_path).name)}</a>'


def write_output_file(
    *,
    output_dir: Path,
    prompt_index: int,
    target: dict[str, Any],
    item: dict[str, str],
    prompt_path: Path,
    stats: dict[str, Any],
    text: str,
) -> Path:
    model_dir = output_dir / safe_slug(target["label"])
    model_dir.mkdir(parents=True, exist_ok=True)
    output_path = model_dir / f"{prompt_index:02d}-{safe_slug(item['name'])}.md"
    metadata = {
        "model": target["repo_id"],
        "model_key": target["key"],
        "model_label": target["label"],
        "model_ref": target["model_ref"],
        "prompt_name": item["name"],
        "category": item["category"],
        "prompt_file": str(prompt_path),
        "stats": stats,
    }
    output_path.write_text(
        "# MLX Benchmark Output\n\n"
        "## Metadata\n\n"
        "```json\n"
        f"{json.dumps(metadata, indent=2)}\n"
        "```\n\n"
        "## Prompt\n\n"
        f"{item['prompt']}\n\n"
        "## Output\n\n"
        f"{text}\n",
        encoding="utf-8",
    )
    return output_path


def write_benchmark_html_report(
    *,
    report_path: Path,
    run_path: Path,
    summaries: list[dict[str, Any]],
    records: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prompt_file = getattr(args, "prompts", "")
    output_dir = getattr(args, "output_dir", "")
    summary_rows = []
    for summary in summaries:
        summary_rows.append(
            "<tr>"
            + table_cell(summary["label"])
            + table_cell(summary["status"])
            + table_cell(format_seconds(summary["load_seconds"]), number=True)
            + table_cell(format_seconds(summary["total_seconds"]), number=True)
            + table_cell(summary["prompt_count"], number=True)
            + table_cell(summary["prompt_tokens"], number=True)
            + table_cell(format_rate(summary["prompt_tps"]), number=True)
            + table_cell(summary["generation_tokens"], number=True)
            + table_cell(format_rate(summary["generation_tps"]), number=True)
            + table_cell(f"{summary['peak_memory_gb']:.2f}", number=True)
            + "</tr>"
        )

    prompt_rows = []
    for record in records:
        stats = record["stats"]
        output_link = html_file_link(report_path, record.get("output_path"))
        prompt_rows.append(
            "<tr>"
            + table_cell(record["model_label"])
            + table_cell(record["prompt_name"])
            + table_cell(record["category"])
            + table_cell(format_seconds(stats["wall_seconds"]), number=True)
            + table_cell(stats["prompt_tokens"], number=True)
            + table_cell(format_rate(stats["prompt_tps"]), number=True)
            + table_cell(stats["generation_tokens"], number=True)
            + table_cell(format_rate(stats["generation_tps"]), number=True)
            + table_cell(f"{stats['peak_memory_gb']:.2f}", number=True)
            + table_cell(stats["finish_reason"])
            + f"<td>{output_link}</td>"
            + "</tr>"
        )

    output_blocks = []
    for record in records:
        stats = record["stats"]
        summary = (
            f"{record['model_label']} / {record['prompt_name']} "
            f"({stats['generation_tokens']} gen tokens, "
            f"{format_rate(stats['generation_tps'])} tok/s)"
        )
        output_blocks.append(
            "<details>"
            f"<summary>{html_text(summary)}</summary>"
            f"<p class=\"output-link\">{html_file_link(report_path, record.get('output_path'))}</p>"
            f"<pre>{html_text(record['output'])}</pre>"
            "</details>"
        )

    html_document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MLX Benchmark Report</title>
  <style>
    :root {{
      color-scheme: light dark;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #1f2933;
      --muted: #687385;
      --border: #d9dee7;
      --accent: #0f766e;
      --code-bg: #101827;
    }}
    @media (prefers-color-scheme: dark) {{
      :root {{
        --bg: #111418;
        --panel: #181d24;
        --text: #edf2f7;
        --muted: #a6b0bf;
        --border: #303845;
        --accent: #2dd4bf;
        --code-bg: #0b1018;
      }}
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
        "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    main {{
      width: min(1180px, calc(100vw - 32px));
      margin: 32px auto 48px;
    }}
    header {{
      margin-bottom: 24px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 28px 0 12px;
      font-size: 18px;
      letter-spacing: 0;
    }}
    .meta {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 10px;
      margin-top: 16px;
    }}
    .metric {{
      padding: 12px 14px;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
    }}
    .metric span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
    }}
    .metric strong {{
      display: block;
      margin-top: 4px;
      font-size: 15px;
      word-break: break-word;
    }}
    .table-wrap {{
      overflow-x: auto;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th,
    td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--border);
      text-align: left;
      vertical-align: top;
      white-space: nowrap;
    }}
    th {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      background: color-mix(in srgb, var(--panel) 88%, var(--accent));
    }}
    tr:last-child td {{
      border-bottom: 0;
    }}
    .number {{
      text-align: right;
      font-variant-numeric: tabular-nums;
    }}
    details {{
      margin-bottom: 10px;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
    }}
    summary {{
      cursor: pointer;
      padding: 12px 14px;
      color: var(--accent);
      font-weight: 650;
    }}
    pre {{
      margin: 0;
      padding: 16px;
      overflow-x: auto;
      border-top: 1px solid var(--border);
      background: var(--code-bg);
      color: #f8fafc;
      font-size: 13px;
      line-height: 1.5;
      white-space: pre-wrap;
    }}
    .note {{
      color: var(--muted);
      font-size: 13px;
    }}
    .output-link {{
      margin: 0;
      padding: 0 14px 12px;
      color: var(--muted);
      font-size: 13px;
    }}
    a {{
      color: var(--accent);
      text-decoration: none;
    }}
    a:hover {{
      text-decoration: underline;
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>MLX Benchmark Report</h1>
      <p class="note">Generated {html_text(generated_at)}. Raw JSONL: {html_text(run_path.name)}</p>
      <div class="meta">
        <div class="metric"><span>Models</span><strong>{len(summaries)}</strong></div>
        <div class="metric"><span>Prompt runs</span><strong>{len(records)}</strong></div>
        <div class="metric"><span>Max tokens</span><strong>{html_text(args.max_tokens)}</strong></div>
        <div class="metric"><span>Temperature</span><strong>{html_text(args.temp)}</strong></div>
        <div class="metric"><span>Prompt file</span><strong>{html_text(prompt_file)}</strong></div>
        <div class="metric"><span>Output dir</span><strong>{html_text(output_dir or "embedded only")}</strong></div>
      </div>
    </header>

    <section>
      <h2>Model Summary</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Model</th>
              <th>Status</th>
              <th class="number">Load</th>
              <th class="number">Total</th>
              <th class="number">Prompts</th>
              <th class="number">Prompt Tok</th>
              <th class="number">Prompt Tok/s</th>
              <th class="number">Gen Tok</th>
              <th class="number">Gen Tok/s</th>
              <th class="number">Peak GB</th>
            </tr>
          </thead>
          <tbody>
            {''.join(summary_rows)}
          </tbody>
        </table>
      </div>
    </section>

    <section>
      <h2>Prompt Runs</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Model</th>
              <th>Prompt</th>
              <th>Category</th>
              <th class="number">Wall</th>
              <th class="number">Prompt Tok</th>
              <th class="number">Prompt Tok/s</th>
              <th class="number">Gen Tok</th>
              <th class="number">Gen Tok/s</th>
              <th class="number">Peak GB</th>
              <th>Finish</th>
              <th>Output File</th>
            </tr>
          </thead>
          <tbody>
            {''.join(prompt_rows)}
          </tbody>
        </table>
      </div>
    </section>

    <section>
      <h2>Outputs</h2>
      {''.join(output_blocks)}
    </section>
  </main>
</body>
</html>
"""
    report_path.write_text(html_document, encoding="utf-8")


def normalize_benchmark_record(record: dict[str, Any]) -> dict[str, Any]:
    stats = record.setdefault("stats", {})
    stats.setdefault("wall_seconds", 0.0)
    stats.setdefault("prompt_tokens", 0)
    stats.setdefault("prompt_tps", 0.0)
    stats.setdefault("generation_tokens", 0)
    stats.setdefault("generation_tps", 0.0)
    stats.setdefault("peak_memory_gb", 0.0)
    stats.setdefault("finish_reason", "")

    model_ref = record.get("model_ref") or record.get("model") or "model"
    record.setdefault("model_label", Path(str(model_ref)).name)
    record.setdefault("prompt_name", "")
    record.setdefault("category", "")
    record.setdefault("output", "")
    record.setdefault("output_path", "")
    record.setdefault("output_dir", "")
    record.setdefault("prompt_file", "")
    record.setdefault("load_seconds", 0.0)
    record.setdefault("max_tokens", "")
    record.setdefault("temperature", "")
    return record


def read_benchmark_records(run_path: Path) -> list[dict[str, Any]]:
    records = []
    with run_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(normalize_benchmark_record(json.loads(line)))
    return records


def prompt_file_id(path_value: str | Path) -> str:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def normalized_benchmark_settings(
    *,
    max_tokens: Any,
    temp: Any,
    top_p: Any,
    top_k: Any,
    max_kv_size: Any,
) -> tuple[int, float, float, int, str]:
    kv_size = "" if max_kv_size in (None, "") else str(int(max_kv_size))
    return (int(max_tokens), float(temp), float(top_p), int(top_k), kv_size)


def benchmark_completion_key(
    *,
    model_key: str,
    prompt_file: str | Path,
    prompt_name: str,
    max_tokens: Any,
    temp: Any,
    top_p: Any,
    top_k: Any,
    max_kv_size: Any,
) -> tuple[Any, ...]:
    return (
        model_key,
        prompt_file_id(prompt_file),
        prompt_name,
        *normalized_benchmark_settings(
            max_tokens=max_tokens,
            temp=temp,
            top_p=top_p,
            top_k=top_k,
            max_kv_size=max_kv_size,
        ),
    )


def completed_benchmark_keys() -> set[tuple[Any, ...]]:
    completed: set[tuple[Any, ...]] = set()
    if not RUNS_DIR.exists():
        return completed

    for run_path in RUNS_DIR.glob("*.jsonl"):
        with run_path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    prompt_file = record.get("prompt_file")
                    model_key = record.get("model_key")
                    prompt_name = record.get("prompt_name")
                    if not (prompt_file and model_key and prompt_name):
                        continue
                    completed.add(
                        benchmark_completion_key(
                            model_key=model_key,
                            prompt_file=prompt_file,
                            prompt_name=prompt_name,
                            max_tokens=record.get("max_tokens", 512),
                            temp=record.get("temperature", 0.0),
                            top_p=record.get("top_p", 1.0),
                            top_k=record.get("top_k", 0),
                            max_kv_size=record.get("max_kv_size"),
                        )
                    )
                except Exception:
                    continue
    return completed


def summarize_benchmark_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(record["model_label"], []).append(record)

    summaries = []
    for label, group in groups.items():
        prompt_count = len(group)
        prompt_tokens = sum(item["stats"]["prompt_tokens"] for item in group)
        generation_tokens = sum(item["stats"]["generation_tokens"] for item in group)
        prompt_tps = (
            sum(item["stats"]["prompt_tps"] for item in group) / prompt_count
            if prompt_count
            else 0.0
        )
        generation_tps = (
            sum(item["stats"]["generation_tps"] for item in group) / prompt_count
            if prompt_count
            else 0.0
        )
        peak_memory_gb = max(item["stats"]["peak_memory_gb"] for item in group)
        load_seconds = max(item.get("load_seconds", 0.0) for item in group)
        total_seconds = load_seconds + sum(
            item["stats"].get("wall_seconds", 0.0) for item in group
        )
        summaries.append(
            {
                "label": label,
                "load_seconds": load_seconds,
                "total_seconds": total_seconds,
                "prompt_count": prompt_count,
                "prompt_tokens": prompt_tokens,
                "generation_tokens": generation_tokens,
                "prompt_tps": prompt_tps,
                "generation_tps": generation_tps,
                "peak_memory_gb": peak_memory_gb,
                "status": "ok",
            }
        )
    return summaries


def read_prompts(prompt_path: Path, limit: int | None = None) -> list[dict[str, str]]:
    prompts: list[dict[str, str]] = []
    with prompt_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                prompts.append(json.loads(line))
            if limit is not None and len(prompts) >= limit:
                break
    return prompts


def apply_chat_template(tokenizer: Any, prompt: str, system_prompt: str) -> str | list[int]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    return f"{system_prompt}\n\nUser:\n{prompt}\n\nAssistant:\n"


def make_sampler(temp: float, top_p: float, top_k: int):
    from mlx_lm.sample_utils import make_sampler as mlx_make_sampler

    return mlx_make_sampler(temp=temp, top_p=top_p, top_k=top_k)


def load_model(model_ref: str) -> tuple[Any, Any]:
    from mlx_lm import load

    console.print(f"[bold]Loading model:[/bold] {model_ref}")
    return load(model_ref)


def generate_with_loaded_model(
    *,
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_tokens: int,
    temp: float,
    top_p: float,
    top_k: int,
    max_kv_size: int | None,
    show_output: bool,
    system_prompt: str,
) -> tuple[str, dict[str, Any]]:
    from mlx_lm import stream_generate

    formatted_prompt = apply_chat_template(tokenizer, prompt, system_prompt)
    sampler = make_sampler(temp=temp, top_p=top_p, top_k=top_k)

    text_parts: list[str] = []
    final_response = None
    for response in stream_generate(
        model,
        tokenizer,
        formatted_prompt,
        max_tokens=max_tokens,
        sampler=sampler,
        max_kv_size=max_kv_size,
    ):
        text_parts.append(response.text)
        final_response = response
        if show_output:
            print(response.text, end="", flush=True)

    if show_output:
        print()

    text = "".join(text_parts)
    stats = {
        "prompt_tokens": getattr(final_response, "prompt_tokens", 0),
        "prompt_tps": getattr(final_response, "prompt_tps", 0.0),
        "generation_tokens": getattr(final_response, "generation_tokens", 0),
        "generation_tps": getattr(final_response, "generation_tps", 0.0),
        "peak_memory_gb": getattr(final_response, "peak_memory", 0.0),
        "finish_reason": getattr(final_response, "finish_reason", None),
    }
    return text, stats


def command_info(_: argparse.Namespace) -> None:
    model = first_model_config()
    local = model_path(model)
    disk = shutil.disk_usage(USER_PATHS.root)
    memory_bytes = psutil.virtual_memory().total

    table = Table(title="MLX Large Model Lab")
    table.add_column("Item")
    table.add_column("Value")
    table.add_row("Machine", platform.platform())
    table.add_row("Python", sys.version.split()[0])
    table.add_row(
        "Memory",
        f"{memory_bytes / (1024 ** 3):.1f} GiB ({memory_bytes / 1e9:.1f} GB)",
    )
    table.add_row("Disk free", f"{disk.free / 1e9:.1f} GB")
    table.add_row("Primary model", model["repo_id"])
    table.add_row("Expected model size", f"{model['estimated_size_gb']} GB")
    table.add_row("Local model path", str(local))
    table.add_row("Downloaded", "yes" if model_is_downloaded(model) else "no")
    table.add_row("Capabilities", str(CONFIG_PATH))
    table.add_row("User configuration", str(USER_PATHS.config))
    table.add_row("User store", str(USER_PATHS.root))
    console.print(table)


def selected_configured_model(
    model_input: str | None,
) -> tuple[str, dict[str, Any]]:
    config = load_config()
    if not model_input:
        key = config["first_model"]
        return key, config["models"][key]
    matched = find_configured_model(model_input)
    if matched:
        return matched
    raise SystemExit(
        f"Unknown model capability: {model_input}. "
        "Run `uv run mlx-lab models` to list supported models."
    )


def repository_download_sizes(
    repo_id: str,
) -> tuple[int, int, int]:
    files = snapshot_download(
        repo_id=repo_id,
        cache_dir=CACHE_DIR / "huggingface",
        dry_run=True,
    )
    total_bytes = sum(int(item.file_size or 0) for item in files)
    weight_bytes = sum(
        int(item.file_size or 0)
        for item in files
        if str(getattr(item, "filename", "")).endswith(".safetensors")
    )
    return len(files), total_bytes, weight_bytes or total_bytes


def format_gib(value: int) -> str:
    return f"{value / GIB:.1f} GiB"


def format_gb(value: int) -> str:
    return f"{value / GB:.1f} GB"


def command_plan_model(args: argparse.Namespace) -> None:
    key, model = selected_configured_model(args.model)
    local = model_path(model)
    downloaded = model_is_downloaded(model)
    memory = psutil.virtual_memory()
    simulated_memory = args.memory_gib is not None
    total_memory_bytes = (
        int(args.memory_gib * GIB)
        if simulated_memory
        else int(memory.total)
    )
    if total_memory_bytes <= 0:
        raise SystemExit("--memory-gib must be greater than zero.")
    available_memory_bytes = (
        total_memory_bytes if simulated_memory else int(memory.available)
    )

    reserve_bytes = (
        int(args.reserve_gib * GIB)
        if args.reserve_gib is not None
        else recommended_memory_reserve_bytes(total_memory_bytes)
    )
    if reserve_bytes < 4 * GIB:
        raise SystemExit("--reserve-gib must be at least 4.")
    if args.context_tokens <= 0:
        raise SystemExit("--context-tokens must be greater than zero.")

    estimated_bytes = int(float(model["estimated_size_gb"]) * GB)
    file_count: int | None = None
    repository_bytes = estimated_bytes
    weight_bytes = estimated_bytes
    size_source = "capability catalog estimate"
    config_ref: str | Path | None = local if (local / "config.json").is_file() else None

    if not args.catalog_only:
        try:
            file_count, repository_bytes, weight_bytes = repository_download_sizes(
                model["repo_id"]
            )
            size_source = "Hugging Face repository metadata"
        except Exception as exc:
            raise SystemExit(
                "Could not inspect the Hugging Face repository. Check network "
                "access or rerun with --catalog-only. "
                f"Details: {exc}"
            ) from exc
        if config_ref is None:
            try:
                config_ref = hf_hub_download(
                    repo_id=model["repo_id"],
                    filename="config.json",
                    cache_dir=CACHE_DIR / "huggingface",
                )
            except Exception:
                config_ref = None

    profile = model_memory_profile(config_ref)
    disk = shutil.disk_usage(USER_PATHS.root)
    plan = build_memory_plan(
        model_weight_bytes=weight_bytes,
        download_bytes=0 if downloaded else repository_bytes,
        total_memory_bytes=total_memory_bytes,
        available_memory_bytes=available_memory_bytes,
        disk_free_bytes=int(disk.free),
        context_tokens=args.context_tokens,
        kv_bytes_per_token=int(profile["kv_bytes_per_token"]),
        reserve_bytes=reserve_bytes,
        model_context_tokens=int(profile["model_context_tokens"]),
    )
    abort_bytes = recommended_abort_threshold_bytes(reserve_bytes)
    plan.update(
        {
            "model_key": key,
            "model_alias": model_label(key, model),
            "repo_id": model["repo_id"],
            "downloaded": downloaded,
            "repository_bytes": repository_bytes,
            "repository_file_count": file_count,
            "size_source": size_source,
            "kv_estimate_source": profile["kv_estimate_source"],
            "simulated_memory": simulated_memory,
            "recommended_abort_bytes": abort_bytes,
        }
    )

    if args.json:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return

    verdict_color = {
        "safe": "green",
        "caution": "yellow",
        "unsafe": "red",
    }[plan["verdict"]]
    table = Table(title=f"Model Fit Plan: {model_label(key, model)}")
    table.add_column("Item")
    table.add_column("Estimate")
    table.add_row("Verdict", f"[{verdict_color}]{plan['verdict'].upper()}[/{verdict_color}]")
    table.add_row("Repository", model["repo_id"])
    table.add_row(
        "Repository download",
        (
            f"{format_gb(repository_bytes)} across {file_count} files"
            if file_count is not None
            else f"{format_gb(repository_bytes)} ({size_source})"
        ),
    )
    table.add_row(
        "Download needed",
        "no; model is already present"
        if downloaded
        else f"yes; allow {format_gb(plan['disk_required_bytes'])} with buffer",
    )
    table.add_row(
        "Unified memory",
        f"{format_gib(total_memory_bytes)}"
        + (" (simulated)" if simulated_memory else " (detected)"),
    )
    if not simulated_memory:
        table.add_row(
            "Available now",
            format_gib(available_memory_bytes),
        )
    table.add_row("Estimated weights", format_gib(weight_bytes))
    table.add_row(
        f"KV cache at {args.context_tokens:,} tokens",
        f"{format_gib(plan['kv_cache_bytes'])} ({profile['kv_estimate_source']})",
    )
    table.add_row(
        "Runtime overhead allowance",
        format_gib(plan["runtime_overhead_bytes"]),
    )
    table.add_row("Safety reserve", format_gib(reserve_bytes))
    table.add_row(
        "Estimated memory with reserve",
        format_gib(plan["recommended_memory_bytes"]),
    )
    table.add_row("Disk free", format_gb(int(disk.free)))
    if profile["model_context_tokens"]:
        table.add_row(
            "Advertised model context",
            f"{int(profile['model_context_tokens']):,} tokens",
        )
    console.print(table)

    console.print("\n[bold]Recommended next steps[/bold]")
    if plan["verdict"] == "unsafe":
        console.print(
            "- [red]Do not load this model with these settings.[/red] "
            "Choose a smaller model or reduce the context target."
        )
        console.print("- Compare alternatives with `uv run mlx-lab models`.")
    elif plan["verdict"] == "caution":
        if not plan["current_memory_fit"] and not simulated_memory:
            console.print(
                "- Close memory-intensive applications before loading the model."
            )
        if not plan["disk_fit"]:
            console.print("- Free disk space before downloading the model.")
    else:
        console.print("- The model fits the selected planning envelope.")
    if args.catalog_only:
        console.print(
            "- Rerun without `--catalog-only` to verify the repository's "
            "current file sizes."
        )
    console.print(
        f"- Download check: `uv run mlx-lab download --model "
        f"{model_label(key, model)} --dry-run`"
    )
    console.print(
        f"- Download: `uv run mlx-lab download --model "
        f"{model_label(key, model)}`"
    )
    console.print(
        "- Load only one large model at a time and use "
        "`scripts/mlx-server unload` when finished."
    )
    console.print(
        f"- Keep the memory guard enabled; this plan recommends about "
        f"{reserve_bytes / GIB:.0f} GiB reserve and "
        f"{abort_bytes / GIB:.0f} GiB emergency-abort headroom."
    )
    console.print(
        "- Treat this as a conservative planning estimate. Validate actual peak "
        "memory in the Control Center before increasing context or output limits."
    )


def command_download(args: argparse.Namespace) -> None:
    key, model = selected_configured_model(args.model)
    local = model_path(model)
    if args.dry_run:
        file_count, total, _ = repository_download_sizes(model["repo_id"])
        console.print(f"[bold]Repo:[/bold] {model['repo_id']}")
        console.print(f"[bold]Files:[/bold] {file_count}")
        console.print(f"[bold]Would download:[/bold] {total / 1e9:.1f} GB")
        console.print(f"[bold]Destination:[/bold] {local}")
        console.print(
            "[bold]Memory plan:[/bold] "
            f"uv run mlx-lab plan-model {model_label(key, model)}"
        )
        return

    console.print(f"[bold]Downloading:[/bold] {model['repo_id']}")
    console.print(f"[bold]Destination:[/bold] {local}")
    snapshot_download(
        repo_id=model["repo_id"],
        local_dir=local,
        cache_dir=CACHE_DIR / "huggingface",
    )
    register_model_path(key, local)
    console.print("[green]Download complete.[/green]")


def command_config(_: argparse.Namespace) -> None:
    config = sync_user_store()
    console.print(f"[bold]Capabilities:[/bold] {CONFIG_PATH}")
    console.print(f"[bold]User configuration:[/bold] {USER_PATHS.config}")
    console.print_json(data=config)


def command_models(args: argparse.Namespace) -> None:
    rows = []
    for key, model in configured_models():
        local = model_path(model)
        downloaded = model_is_downloaded(model)
        aliases = ", ".join(model.get("aliases", [])) or "-"
        ref = str(local) if downloaded else model["repo_id"]
        if args.plain:
            rows.append(
                "\t".join(
                    (
                        key,
                        "local" if downloaded else "hosted",
                        aliases,
                        ref,
                    )
                )
            )
            continue

        rows.append(
            (
                model_label(key, model),
                "yes" if downloaded else "no",
                f"{float(model['estimated_size_gb']):.1f}",
                str(model.get("role") or "-"),
            )
        )

    if args.plain:
        print("\n".join(rows))
        return

    table = Table(title="MLX model capabilities")
    table.add_column("Model")
    table.add_column("Downloaded")
    table.add_column("Estimated GB", justify="right")
    table.add_column("Role")
    for row in rows:
        table.add_row(*row)
    console.print(table)


def command_register_model(args: argparse.Namespace) -> None:
    matched = find_configured_model(args.model)
    if not matched:
        raise SystemExit(f"Unknown model capability: {args.model}")
    key, model = matched
    local = Path(args.local_dir).expanduser().resolve()
    if not model_is_present(local):
        raise SystemExit(
            f"No complete MLX model found at {local}; expected config.json and "
            "*.safetensors files."
        )
    register_model_path(key, local)
    console.print(
        f"[green]Registered {model_label(key, model)}:[/green] {local}"
    )


def command_resolve_model(args: argparse.Namespace) -> None:
    matched = find_configured_model(args.model)
    if not matched:
        value = (
            str(Path(args.model).expanduser())
            if args.model.startswith("~")
            else args.model
        )
        print(value)
        return
    _, model = matched
    print(model_path(model) if model_is_downloaded(model) else model["repo_id"])


def command_path(args: argparse.Namespace) -> None:
    paths = {
        "root": USER_PATHS.root,
        "config": USER_PATHS.config,
        "models": USER_PATHS.models,
        "cache": USER_PATHS.cache,
        "runs": USER_PATHS.runs,
        "logs": USER_PATHS.logs,
        "capabilities": CONFIG_PATH,
        "project": ROOT,
    }
    print(paths[args.name])


def command_api_key(args: argparse.Namespace) -> None:
    settings = control_center_settings()
    if args.if_configured and not settings["api_key_configured"]:
        return
    print(control_center_settings(include_secret=True)["api_key"])


def command_generate(args: argparse.Namespace) -> None:
    model = first_model_config()
    model_ref = args.model or resolved_model_ref(model)
    loaded_model, tokenizer = load_model(model_ref)
    _, stats = generate_with_loaded_model(
        model=loaded_model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temp=args.temp,
        top_p=args.top_p,
        top_k=args.top_k,
        max_kv_size=args.max_kv_size,
        show_output=True,
        system_prompt=args.system_prompt,
    )
    console.print_json(data=stats)


def command_benchmark(args: argparse.Namespace) -> None:
    targets = benchmark_targets(args)
    prompt_path = workspace_path(args.prompts)
    prompts = read_prompts(prompt_path, args.limit)
    RUNS_DIR.mkdir(exist_ok=True)
    if args.all_models:
        run_label = "all-models"
    elif args.model_set:
        run_label = args.model_set.replace("_", "-")
    else:
        run_label = "benchmark"
    run_path = RUNS_DIR / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{run_label}.jsonl"
    report_path = run_path.with_suffix(".html")
    output_dir = (
        run_store_path(args.output_dir) / run_path.stem
        if args.output_dir
        else None
    )
    completed_keys = completed_benchmark_keys() if args.skip_completed else set()

    console.print(f"[bold]Benchmark models:[/bold] {len(targets)}")
    console.print(f"[bold]Prompt file:[/bold] {prompt_path}")
    console.print(f"[bold]Prompts:[/bold] {len(prompts)}")
    console.print(f"[bold]Output:[/bold] {run_path}")
    console.print(f"[bold]HTML report:[/bold] {report_path}")
    if args.model_set:
        console.print(f"[bold]Model set:[/bold] {args.model_set}")
    if args.skip_completed:
        console.print(
            "[bold]Skip completed:[/bold] enabled "
            f"({len(completed_keys)} prior benchmark rows indexed)"
        )
    if output_dir:
        console.print(f"[bold]Output files:[/bold] {output_dir}")

    summaries = []
    records = []
    with run_path.open("w", encoding="utf-8") as file:
        for target in targets:
            model_started = time.perf_counter()
            load_seconds = 0.0
            rows = []

            console.print(f"\n[bold]Benchmark model:[/bold] {target['label']}")
            console.print(f"[bold]Model ref:[/bold] {target['model_ref']}")
            target_prompts = prompts
            if args.skip_completed:
                target_prompts = [
                    item
                    for item in prompts
                    if benchmark_completion_key(
                        model_key=target["key"],
                        prompt_file=prompt_path,
                        prompt_name=item["name"],
                        max_tokens=args.max_tokens,
                        temp=args.temp,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        max_kv_size=args.max_kv_size,
                    )
                    not in completed_keys
                ]
                skipped_count = len(prompts) - len(target_prompts)
                if skipped_count:
                    console.print(
                        "[yellow]Skipping completed prompts:[/yellow] "
                        f"{skipped_count}/{len(prompts)}"
                    )
                if not target_prompts:
                    summaries.append(
                        {
                            "label": target["label"],
                            "load_seconds": 0.0,
                            "total_seconds": 0.0,
                            "prompt_count": 0,
                            "prompt_tokens": 0,
                            "generation_tokens": 0,
                            "prompt_tps": 0.0,
                            "generation_tps": 0.0,
                            "peak_memory_gb": 0.0,
                            "status": "skipped",
                        }
                    )
                    continue

            try:
                load_started = time.perf_counter()
                loaded_model, tokenizer = load_model(target["model_ref"])
                load_seconds = time.perf_counter() - load_started
                console.print(f"[bold]Load time:[/bold] {load_seconds:.2f}s")

                for prompt_index, item in enumerate(target_prompts, start=1):
                    console.print(
                        f"\n[bold]Prompt:[/bold] {item['name']} ({item['category']})"
                    )
                    prompt_started = time.perf_counter()
                    text, stats = generate_with_loaded_model(
                        model=loaded_model,
                        tokenizer=tokenizer,
                        prompt=item["prompt"],
                        max_tokens=args.max_tokens,
                        temp=args.temp,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        max_kv_size=args.max_kv_size,
                        show_output=args.show_output,
                        system_prompt=args.system_prompt,
                    )
                    prompt_seconds = time.perf_counter() - prompt_started
                    stats["wall_seconds"] = prompt_seconds
                    output_path = None
                    if output_dir:
                        output_path = write_output_file(
                            output_dir=output_dir,
                            prompt_index=prompt_index,
                            target=target,
                            item=item,
                            prompt_path=prompt_path,
                            stats=stats,
                            text=text,
                        )
                    record = {
                        "model": target["repo_id"],
                        "model_key": target["key"],
                        "model_label": target["label"],
                        "model_ref": target["model_ref"],
                        "prompt_file": str(prompt_path),
                        "prompt_name": item["name"],
                        "category": item["category"],
                        "max_tokens": args.max_tokens,
                        "temperature": args.temp,
                        "top_p": args.top_p,
                        "top_k": args.top_k,
                        "max_kv_size": args.max_kv_size,
                        "load_seconds": load_seconds,
                        "stats": stats,
                        "output_dir": str(output_dir) if output_dir else "",
                        "output_path": str(output_path) if output_path else "",
                        "output": text,
                    }
                    file.write(json.dumps(record) + "\n")
                    records.append(record)
                    rows.append((item["name"], stats))
            except Exception as exc:
                if not (args.all_models or args.model_set):
                    raise
                console.print(f"[red]Benchmark failed:[/red] {target['label']}: {exc}")
                prompt_count = len(rows)
                prompt_tokens = sum(stats["prompt_tokens"] for _, stats in rows)
                generation_tokens = sum(stats["generation_tokens"] for _, stats in rows)
                prompt_tps = (
                    sum(stats["prompt_tps"] for _, stats in rows) / prompt_count
                    if prompt_count
                    else 0.0
                )
                generation_tps = (
                    sum(stats["generation_tps"] for _, stats in rows) / prompt_count
                    if prompt_count
                    else 0.0
                )
                peak_memory_gb = (
                    max(stats["peak_memory_gb"] for _, stats in rows) if rows else 0.0
                )
                summaries.append(
                    {
                        "label": target["label"],
                        "load_seconds": load_seconds,
                        "total_seconds": time.perf_counter() - model_started,
                        "prompt_count": prompt_count,
                        "prompt_tokens": prompt_tokens,
                        "generation_tokens": generation_tokens,
                        "prompt_tps": prompt_tps,
                        "generation_tps": generation_tps,
                        "peak_memory_gb": peak_memory_gb,
                        "status": "failed",
                    }
                )
                continue
            finally:
                if "loaded_model" in locals():
                    del loaded_model
                if "tokenizer" in locals():
                    del tokenizer
                release_model_memory()

            total_seconds = time.perf_counter() - model_started
            prompt_count = len(rows)
            prompt_tokens = sum(stats["prompt_tokens"] for _, stats in rows)
            generation_tokens = sum(stats["generation_tokens"] for _, stats in rows)
            prompt_tps = (
                sum(stats["prompt_tps"] for _, stats in rows) / prompt_count
                if prompt_count
                else 0.0
            )
            generation_tps = (
                sum(stats["generation_tps"] for _, stats in rows) / prompt_count
                if prompt_count
                else 0.0
            )
            peak_memory_gb = (
                max(stats["peak_memory_gb"] for _, stats in rows) if rows else 0.0
            )

            summaries.append(
                {
                    "label": target["label"],
                    "load_seconds": load_seconds,
                    "total_seconds": total_seconds,
                    "prompt_count": prompt_count,
                    "prompt_tokens": prompt_tokens,
                    "generation_tokens": generation_tokens,
                    "prompt_tps": prompt_tps,
                    "generation_tps": generation_tps,
                    "peak_memory_gb": peak_memory_gb,
                    "status": "ok",
                }
            )

            table = Table(title=f"Benchmark Summary: {target['label']}")
            table.add_column("Prompt")
            table.add_column("Wall s")
            table.add_column("Prompt tok")
            table.add_column("Prompt tok/s")
            table.add_column("Gen tok")
            table.add_column("Gen tok/s")
            table.add_column("Peak GB")
            for name, stats in rows:
                table.add_row(
                    name,
                    f"{stats['wall_seconds']:.2f}",
                    str(stats["prompt_tokens"]),
                    f"{stats['prompt_tps']:.2f}",
                    str(stats["generation_tokens"]),
                    f"{stats['generation_tps']:.2f}",
                    f"{stats['peak_memory_gb']:.2f}",
                )
            console.print(table)

    timing_table = Table(title="Model Comparison: Timing", box=box.SIMPLE)
    timing_table.add_column("Model", overflow="fold")
    timing_table.add_column("Status", no_wrap=True)
    timing_table.add_column("Load s", justify="right", no_wrap=True)
    timing_table.add_column("Total s", justify="right", no_wrap=True)
    timing_table.add_column("Prompts", justify="right", no_wrap=True)
    timing_table.add_column("Peak GB", justify="right", no_wrap=True)
    for summary in summaries:
        timing_table.add_row(
            summary["label"],
            summary["status"],
            f"{summary['load_seconds']:.2f}",
            f"{summary['total_seconds']:.2f}",
            str(summary["prompt_count"]),
            f"{summary['peak_memory_gb']:.2f}",
        )
    console.print(timing_table)

    token_table = Table(title="Model Comparison: Tokens", box=box.SIMPLE)
    token_table.add_column("Model", overflow="fold")
    token_table.add_column("Prompt tok", justify="right", no_wrap=True)
    token_table.add_column("Prompt tok/s", justify="right", no_wrap=True)
    token_table.add_column("Gen tok", justify="right", no_wrap=True)
    token_table.add_column("Gen tok/s", justify="right", no_wrap=True)
    for summary in summaries:
        token_table.add_row(
            summary["label"],
            str(summary["prompt_tokens"]),
            f"{summary['prompt_tps']:.2f}",
            str(summary["generation_tokens"]),
            f"{summary['generation_tps']:.2f}",
        )
    console.print(token_table)
    write_benchmark_html_report(
        report_path=report_path,
        run_path=run_path,
        summaries=summaries,
        records=records,
        args=args,
    )
    console.print(f"[bold]HTML report written:[/bold] {report_path}")


def command_report(args: argparse.Namespace) -> None:
    run_path = run_store_path(args.jsonl_path)
    records = read_benchmark_records(run_path)
    if not records:
        raise SystemExit(f"No benchmark records found in {run_path}")

    report_path = Path(args.output) if args.output else run_path.with_suffix(".html")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    first_record = records[0]
    report_args = argparse.Namespace(
        max_tokens=first_record.get("max_tokens", ""),
        temp=first_record.get("temperature", ""),
        prompts=first_record.get("prompt_file", ""),
        output_dir=first_record.get("output_dir", ""),
    )
    write_benchmark_html_report(
        report_path=report_path,
        run_path=run_path,
        summaries=summarize_benchmark_records(records),
        records=records,
        args=report_args,
    )
    console.print(f"[bold]HTML report written:[/bold] {report_path}")


def command_ui(args: argparse.Namespace) -> None:
    from mlx_lab.control_plane import serve_control_plane

    serve_control_plane(args.host, args.port)


def command_serve_command(_: argparse.Namespace) -> None:
    model = first_model_config()
    local = model_path(model)
    print(
        "uv run mlx_lm.server "
        f"--model {local} "
        "--host 127.0.0.1 "
        "--port 8080"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local MLX large-model lab")
    subparsers = parser.add_subparsers(dest="command", required=True)

    info = subparsers.add_parser("info", help="Show local lab and model status")
    info.set_defaults(func=command_info)

    config = subparsers.add_parser(
        "config",
        help="Show the per-user configuration and immutable capability path",
    )
    config.set_defaults(func=command_config)

    models = subparsers.add_parser(
        "models",
        help="List model capabilities and per-user download status",
    )
    models.add_argument(
        "--plain",
        action="store_true",
        help="Print tab-separated rows for scripts",
    )
    models.set_defaults(func=command_models)

    register_model = subparsers.add_parser(
        "register-model",
        help="Associate a downloaded model directory with a capability",
    )
    register_model.add_argument("model", help="Capability key, alias, or repo id")
    register_model.add_argument("local_dir", help="Downloaded MLX model directory")
    register_model.set_defaults(func=command_register_model)

    resolve_model = subparsers.add_parser(
        "resolve-model",
        help="Resolve a model key or alias to its local path or hosted repo id",
    )
    resolve_model.add_argument("model")
    resolve_model.set_defaults(func=command_resolve_model)

    path = subparsers.add_parser(
        "path",
        help="Print one MLX Lab project or per-user path",
    )
    path.add_argument(
        "name",
        choices=(
            "root",
            "config",
            "models",
            "cache",
            "runs",
            "logs",
            "capabilities",
            "project",
        ),
    )
    path.set_defaults(func=command_path)

    api_key = subparsers.add_parser(
        "api-key",
        help="Print the shared Control Center and Model API key",
    )
    api_key.add_argument(
        "--if-configured",
        action="store_true",
        help="Print nothing instead of creating a key when none is configured",
    )
    api_key.set_defaults(func=command_api_key)

    plan_model = subparsers.add_parser(
        "plan-model",
        help="Estimate whether a configured model fits this Mac safely",
    )
    plan_model.add_argument(
        "model",
        nargs="?",
        default=None,
        help="Capability key, alias, or repo id; defaults to the preferred model",
    )
    plan_model.add_argument(
        "--context-tokens",
        type=int,
        default=8_192,
        help="Total prompt-plus-output context to budget (default: 8192)",
    )
    plan_model.add_argument(
        "--memory-gib",
        type=float,
        default=None,
        help="Simulate a Mac memory size instead of using detected unified memory",
    )
    plan_model.add_argument(
        "--reserve-gib",
        type=float,
        default=None,
        help="Override the adaptive safety reserve (minimum: 4 GiB)",
    )
    plan_model.add_argument(
        "--catalog-only",
        action="store_true",
        help="Use catalog estimates without querying Hugging Face",
    )
    plan_model.add_argument(
        "--json",
        action="store_true",
        help="Print the sizing plan as JSON",
    )
    plan_model.set_defaults(func=command_plan_model)

    download = subparsers.add_parser(
        "download",
        help="Download a configured model",
    )
    download.add_argument(
        "--model",
        default=None,
        help="Capability key, alias, or repo id; defaults to the preferred model",
    )
    download.add_argument(
        "--dry-run",
        action="store_true",
        help="Check file count and size without downloading model weights",
    )
    download.set_defaults(func=command_download)

    generate = subparsers.add_parser("generate", help="Run one prompt")
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--model", default=None, help="Override model repo or local path")
    generate.add_argument("--max-tokens", type=int, default=256)
    generate.add_argument("--temp", type=float, default=0.0)
    generate.add_argument("--top-p", type=float, default=1.0)
    generate.add_argument("--top-k", type=int, default=0)
    generate.add_argument("--max-kv-size", type=int, default=None)
    generate.add_argument("--system-prompt", default=SYSTEM_PROMPT)
    generate.set_defaults(func=command_generate)

    benchmark = subparsers.add_parser("benchmark", help="Run a prompt set")
    benchmark_model = benchmark.add_mutually_exclusive_group()
    benchmark_model.add_argument(
        "--model",
        default=None,
        help="Override model repo or local path",
    )
    benchmark_model.add_argument(
        "--all-models",
        action="store_true",
        help="Benchmark every downloaded model in the capability catalog",
    )
    benchmark_model.add_argument(
        "--model-set",
        choices=sorted(configured_benchmark_sets()),
        default=None,
        help="Benchmark a configured model set from the capability catalog",
    )
    benchmark.add_argument("--limit", type=int, default=None)
    benchmark.add_argument(
        "--prompts",
        default=str(PROMPT_PATH),
        help="Prompt JSONL path",
    )
    benchmark.add_argument(
        "--output-dir",
        default=None,
        help="Write Markdown outputs under this user-run-store directory",
    )
    benchmark.add_argument("--max-tokens", type=int, default=512)
    benchmark.add_argument("--temp", type=float, default=0.0)
    benchmark.add_argument("--top-p", type=float, default=1.0)
    benchmark.add_argument("--top-k", type=int, default=0)
    benchmark.add_argument("--max-kv-size", type=int, default=None)
    benchmark.add_argument("--system-prompt", default=SYSTEM_PROMPT)
    benchmark.add_argument("--show-output", action="store_true")
    benchmark.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip model/prompt/settings rows already present in the user run store",
    )
    benchmark.set_defaults(func=command_benchmark)

    report = subparsers.add_parser(
        "report",
        help="Write an HTML report from a benchmark JSONL file",
    )
    report.add_argument("jsonl_path", help="Benchmark JSONL file in the user run store")
    report.add_argument("--output", default=None, help="Override HTML output path")
    report.set_defaults(func=command_report)

    ui = subparsers.add_parser(
        "ui",
        help="Start the local benchmark control plane",
    )
    ui.add_argument(
        "--host",
        default=None,
        help="Override the configured Control Center bind address",
    )
    ui.add_argument(
        "--port",
        type=int,
        default=None,
        help="Override the configured Control Center port",
    )
    ui.set_defaults(func=command_ui)

    serve_command = subparsers.add_parser(
        "serve-command",
        help="Print the local server command for the primary model",
    )
    serve_command.set_defaults(func=command_serve_command)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    sync_user_store()
    args.func(args)


if __name__ == "__main__":
    main()
