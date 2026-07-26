# MLX Large Model Lab

MLX Large Model Lab is a local-first toolkit for downloading, serving,
benchmarking, and observing MLX language models on Apple Silicon. It adds a
small operational layer around `mlx-lm`:

- an OpenAI-compatible server with stable model aliases
- one-model-at-a-time load, switch, and unload controls
- a local Control Center with a model harness, benchmarks, memory safety, and
  privacy-bounded telemetry
- reproducible JSONL and HTML benchmark reports
- a strict separation between project capabilities and machine-specific state

The project is under active development.

## Requirements

- an Apple Silicon Mac
- Python 3.12
- [`uv`](https://docs.astral.sh/uv/)
- enough disk space and unified memory for the model you select

Model weights are not included in this repository.

## Quick start

```bash
git clone <repository-url>
cd mlx-large-model-lab
uv sync --python 3.12
uv run mlx-lab config
uv run mlx-lab models
```

`mlx-lab config` initializes the per-user store and prints its location. It
does not modify the capability catalog in the checkout.

Download the default model:

```bash
uv run mlx-lab download --dry-run
uv run mlx-lab download
```

Run a prompt:

```bash
uv run mlx-lab generate \
  --prompt "Write a Python CLI argument parser." \
  --max-tokens 512
```

Start the managed OpenAI-compatible server:

```bash
scripts/mlx-server load qwen3-coder-30b-8bit
scripts/mlx-server status
scripts/mlx-server test
```

The default API endpoint is `http://127.0.0.1:48080/v1`.

Start the local Control Center:

```bash
uv run mlx-lab ui
```

Then open `http://127.0.0.1:8765`.

## Configuration model

The repository and the user store have intentionally different responsibilities.

| Layer | Contents | Writable at runtime |
| --- | --- | --- |
| Capability catalog | Model repo ids, aliases, roles, estimated sizes, benchmark sets | No |
| User configuration | Hardware snapshot, preferred default, registered model directories | Yes |
| User data | Model weights, Hugging Face cache, benchmark runs, logs, telemetry, runtime state | Yes |

The immutable catalog lives at `configs/models.json` in a source checkout and
is bundled into the Python wheel. It must never contain a contributor's home
directory or downloaded-model path.

Keep hardware snapshots, local model paths, downloaded weights, logs,
telemetry, and benchmark outputs in the per-user store. Do not commit those
machine-specific artifacts to the repository.

On macOS, the default user store is:

```text
~/Library/Application Support/mlx-lab/
  user-config.json
  models/
  cache/
  runs/
  logs/
```

Set `MLX_LAB_HOME` to use a different root, including an isolated test store:

```bash
MLX_LAB_HOME=/path/to/mlx-lab-data uv run mlx-lab config
```

The first CLI or server command:

1. captures the current hardware in `user-config.json`;
2. creates the user data directories;
3. discovers compatible model folders already present in the user store; and
4. imports paths to complete model downloads found in a legacy checkout
   `models/` directory without copying or moving the weights.

Inspect the effective paths and model mappings:

```bash
uv run mlx-lab config
uv run mlx-lab path config
uv run mlx-lab path models
uv run mlx-lab path runs
uv run mlx-lab models
```

Register a model downloaded elsewhere:

```bash
uv run mlx-lab register-model \
  qwen3-coder-30b-8bit \
  /absolute/path/to/Qwen3-Coder-30B-A3B-Instruct-8bit
```

The directory must contain `config.json` and at least one `*.safetensors`
file.

## Model server

The server helper keeps at most one explicitly loaded model resident:

```bash
scripts/mlx-server list
scripts/mlx-server load qwen3-coder-30b-8bit
scripts/mlx-server current
scripts/mlx-server switch qwen3-coder-next
scripts/mlx-server unload
```

Useful environment overrides include:

```bash
MLX_SERVER_HOST=127.0.0.1
MLX_SERVER_PORT=48080
MLX_SERVER_PROMPT_CACHE_SIZE=4
MLX_SERVER_PROMPT_CACHE_BYTES=8GB
MLX_SERVER_MAX_CONTEXT_TOKENS=128000
MLX_SERVER_MAX_OUTPUT_TOKENS=32000
MLX_SERVER_MEMORY_RESERVE_BYTES=16GB
MLX_SERVER_MEMORY_ABORT_BYTES=8GB
```

The default bind address is loopback. Carefully review authentication and
network exposure before binding the API or Control Center to another interface.

### OpenAI-compatible request

```bash
curl http://127.0.0.1:48080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-coder-30b-8bit",
    "messages": [
      {"role": "user", "content": "Reply with a short hello."}
    ],
    "max_tokens": 128,
    "temperature": 0
  }'
```

The managed wrapper advertises downloaded capability aliases from
`GET /v1/models`. Loading or switching remains an explicit control operation;
an API request cannot silently load a different configured model when model
pinning is enabled.

## Control Center

The local Control Center provides:

- live server, process, unified-memory, prompt-cache, and request status
- load, switch, unload, cancel, cache-clear, and memory-safety controls
- a streamed multi-turn model harness
- benchmark launch, saved runs, aggregate model statistics, and HTML reports
- request, token, model, client, latency, error, tool, and MCP telemetry

Control endpoints are deliberately separate from `/v1`. Server mutations are
limited to a fixed command allowlist and downloaded capability models.

Telemetry retains a bounded JSONL history in the user store. It records token
counts, timings, client identity, offered tool names, structured tool calls,
and an optional bounded latest-user-query preview. It does not retain complete
conversations, responses, reasoning text, or tool arguments.

Disable query previews or change retention before starting the server:

```bash
MLX_SERVER_TELEMETRY_QUERY_PREVIEW=0 scripts/mlx-server restart
MLX_SERVER_TELEMETRY_PREVIEW_CHARS=120 scripts/mlx-server restart
MLX_SERVER_TELEMETRY_MAX_RECORDS=2000 scripts/mlx-server restart
```

## Benchmarks

Benchmark every downloaded capability model with the bundled prompt set:

```bash
uv run mlx-lab benchmark --all-models --limit 2 --max-tokens 1200
```

Benchmark one model or a catalog-defined set:

```bash
uv run mlx-lab benchmark --model qwen3-coder-next
uv run mlx-lab benchmark --model-set cybersecurity_harness
```

Runs and HTML reports are written beneath the per-user `runs/` directory.
Optional Markdown outputs are also relative to that run store:

```bash
uv run mlx-lab benchmark \
  --all-models \
  --output-dir app-security-outputs \
  --skip-completed
```

Generate a report from an existing run:

```bash
uv run mlx-lab report <run-id>-benchmark.jsonl
```

## OpenCode

`opencode.jsonc` defines a local `local-mlx` provider. Load a model before
launching one of the convenience targets:

```bash
scripts/mlx-server load qwen3-coder-30b-8bit
scripts/opencode-mlx primary
```

See [scripts/README.md](scripts/README.md) for all server, startup, and
OpenCode helper commands.

## Development

Run the tests against an isolated user store:

```bash
MLX_LAB_HOME="$(mktemp -d)" \
  .venv/bin/python -m unittest discover -s tests -v
```

Validate shell helpers:

```bash
bash -n scripts/mlx-server scripts/macos-startup scripts/opencode-mlx
```

## License

An OSI-approved project license still needs to be selected by the repository
owner before the repository is announced for general open-source reuse. Model
weights and upstream model artifacts retain their own licenses.
