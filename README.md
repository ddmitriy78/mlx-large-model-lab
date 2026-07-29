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

### 1. Initialize the lab

```bash
git clone <repository-url>
cd mlx-large-model-lab
uv sync --python 3.12
uv run mlx-lab config
uv run mlx-lab models
```

`mlx-lab config` initializes the per-user store and prints its location. It
does not modify the capability catalog in the checkout.

### 2. Check whether the first model fits

Run the planner before downloading model weights:

```bash
uv run mlx-lab plan-model \
  qwen3-coder-30b-8bit \
  --context-tokens 8192
```

The planner reads the Mac's unified memory and current availability, checks
free disk space, gets the current repository size from Hugging Face, estimates
the model's KV cache from `config.json`, and preserves an adaptive safety
reserve. It does not download the model weights.

Use the catalog estimate when offline:

```bash
uv run mlx-lab plan-model qwen3-coder-30b-8bit --catalog-only
```

Use `--memory-gib` to evaluate another Mac before buying or moving a model:

```bash
uv run mlx-lab plan-model qwen3-coder-30b-8bit \
  --memory-gib 36 \
  --catalog-only
```

The override accepts any positive memory size, including 24, 36, 48, 64, and
128 GiB. The result is:

- `SAFE`: memory capacity, current availability, context, and disk fit the
  planning envelope.
- `CAUTION`: total capacity fits, but running applications or disk space need
  attention.
- `UNSAFE`: the model and selected context do not fit with a safety reserve.

If the result is not safe, list the catalog's estimated sizes and roles, then
plan a smaller model:

```bash
uv run mlx-lab models
uv run mlx-lab plan-model <model-alias> --context-tokens 8192
```

#### Example: a 24 GiB Mac

The catalog's `gemma-4-12b-coder-8bit` model is a practical smaller example
for a 24 GiB Mac. Start with a 4,096-token total context:

```bash
uv run mlx-lab plan-model \
  gemma-4-12b-coder-8bit \
  --context-tokens 4096
```

Using current Hugging Face metadata, the repository is approximately 12.7 GB.
The planner estimates approximately 19.3 GiB for weights, KV cache, runtime
overhead, and the 4 GiB safety reserve, leaving about 4.7 GiB of capacity
headroom on a 24 GiB system.

If the live result is `SAFE`, download it:

```bash
uv run mlx-lab download --model gemma-4-12b-coder-8bit --dry-run
uv run mlx-lab download --model gemma-4-12b-coder-8bit
scripts/mlx-server load gemma-4-12b-coder-8bit
```

This model is an experimental, Python-focused coding specialist rather than a
general-purpose default. Review generated code and tool actions carefully. A
busy 24 GiB Mac may report `CAUTION`; close memory-heavy applications and rerun
the planner instead of reducing the 4 GiB reserve.

### 3. Verify and download

Check the exact download again, then download only after the fit plan is
acceptable:

```bash
uv run mlx-lab download --model qwen3-coder-30b-8bit --dry-run
uv run mlx-lab download --model qwen3-coder-30b-8bit
```

Omit `--model` to use the preferred model from the capability catalog.
Interrupted Hugging Face downloads can be resumed by running the same command.

The catalog also includes the experimental RavenX cybersecurity-agent model:

```bash
uv run mlx-lab plan-model ravenx-cyberagent --context-tokens 8192
uv run mlx-lab download --model ravenx-cyberagent --dry-run
uv run mlx-lab download --model ravenx-cyberagent
```

Its repository is approximately 69.3 GB. Treat security findings and generated
commands as untrusted until they are independently reviewed and tested in an
authorized environment.

For lightweight question routing, the catalog includes the specialized
`openclaw-classifier` model. It classifies incoming questions as `simple`,
`tricky`, or `complex`; it is not intended for general chat, coding, or tool
use:

```bash
uv run mlx-lab plan-model openclaw-classifier --context-tokens 4096
uv run mlx-lab download --model openclaw-classifier --dry-run
uv run mlx-lab download --model openclaw-classifier
scripts/mlx-server load openclaw-classifier
```

Use this system prompt in an OpenAI-compatible request:

```text
You are an expert question classifier for the OpenClaw system. Your job is to
analyze incoming questions and classify them as 'simple', 'tricky', or
'complex' based on the cognitive effort and expertise required to answer them.
Always respond with valid JSON only.
```

Send the user content as `Classify this question: <question>` and keep the
output limit small (for example, 128 tokens). Also set
`"chat_template_kwargs": {"enable_thinking": false}`. Without that template
setting, the current MLX server returns the model's analysis in the `reasoning`
field but no JSON `content`. The expected content shape is:

```json
{"class":"simple","confidence":0.99,"reason":"Direct factual question."}
```

The Control Center Model Harness does not currently expose
`chat_template_kwargs`, so use the OpenAI-compatible API directly for this
classifier.

### 4. Run a first prompt

```bash
uv run mlx-lab generate \
  --prompt "Write a Python CLI argument parser." \
  --max-tokens 512
```

### 5. Start the managed services

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

## Model sizing and memory safety

Download size and runtime memory are not the same number. The planner combines:

- current repository and weight-file bytes, or the catalog estimate offline;
- a runtime-overhead allowance of 10% of weight size or 2 GiB, whichever is
  larger;
- KV-cache memory for the requested total prompt-plus-output context;
- an adaptive reserve for macOS and other applications; and
- a 10% or 5 GB disk buffer, whichever is larger.

The default reserve scales with detected unified memory:

| Unified memory | Reserve | Emergency abort threshold |
| ---: | ---: | ---: |
| 24 GiB | 4 GiB | 2 GiB |
| 36 GiB | 6 GiB | 3 GiB |
| 48 GiB | 8 GiB | 4 GiB |
| 64 GiB | 10 GiB | 5 GiB |
| 128 GiB | 16 GiB | 8 GiB |

Other memory sizes use 15% rounded up to the next GiB, with a 4 GiB minimum
and 16 GiB maximum. Override the planning reserve only for a deliberate test:

```bash
uv run mlx-lab plan-model <model-alias> --reserve-gib 8
```

Safety practices:

1. On a 24 GiB Mac, start with a 4,096-token total context and an output cap of
   512 tokens. Start at 8,192 tokens on larger systems, then measure.
2. Keep only one large model loaded. Run `scripts/mlx-server unload` before
   switching workloads or opening another model application.
3. Keep the memory guard enabled. It rejects unsafe context allocations and
   cancels active work before the system reserve is exhausted.
4. Close memory-intensive applications when the planner reports `CAUTION`.
5. Increase context gradually while watching available memory, swap, retained
   caches, and measured peak memory in the Control Center.
6. Treat every result as a conservative estimate, not a guarantee. Architectures,
   cache quantization, prompt reuse, and other applications change actual use.

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
  control-center-credentials.json
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

A retained prompt must fit below `MLX_SERVER_PROMPT_CACHE_BYTES`. If a growing
conversation crosses that byte limit, MLX-LM evicts the entry and the next
agent step must prefill the full conversation again. For the Qwen3 Coder 30B
model on a 128 GiB Mac, a 128,000-token context needs approximately 12.6 GB of
KV cache, so use a 16 GB retained-cache budget:

```bash
MLX_SERVER_PROMPT_CACHE_BYTES=16GB \
  scripts/macos-startup install qwen3-coder-30b-8bit
MLX_SERVER_PROMPT_CACHE_BYTES=16GB \
  scripts/mlx-server restart qwen3-coder-30b-8bit
```

The startup installer persists that value for both login services so later
Control Center model-server restarts keep it. Size this override with the
model fit planner and leave the memory guard enabled on smaller-memory Macs.

The reserve and abort defaults are calculated from detected unified memory as
shown in the sizing table. The values above are explicit high-memory overrides.

The default bind address is loopback. Carefully review authentication and
network exposure before binding the API or Control Center to another interface.
The Control Center Settings tab can require the same Bearer key for the Model
API and Control Center. Print the configured key for local clients with
`uv run mlx-lab api-key`.

### OpenAI-compatible request

```bash
MLX_API_KEY="$(uv run mlx-lab api-key)"
curl http://127.0.0.1:48080/v1/chat/completions \
  -H "Authorization: Bearer $MLX_API_KEY" \
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

- live server, process, unified-memory, prompt-cache, and request status,
  including processed/total prompt tokens and their estimated KV-memory size
- load, switch, unload, cancel, cache-clear, and memory-safety controls
- a streamed multi-turn model harness
- benchmark launch, saved runs, aggregate model statistics, and HTML reports
- request, token, model, client, latency, error, tool, MCP, and agent-loop-risk
  telemetry
- classified error and cancellation details by time, client, address, endpoint,
  model, request, and retained message
- one-click Control Center and MLX model-server restarts
- independent Control Center and Model API protection using one shared key
- local-only or trusted-network Control Center access, with mandatory API keys remotely

Control endpoints are deliberately separate from `/v1`. Server mutations are
limited to a fixed command allowlist and downloaded capability models. Local
access on `127.0.0.1` does not require a key by default:

```bash
uv run mlx-lab ui
```

The Settings tab can change the next-start bind scope and port, protect local
Control Center access, and protect every Model API request on port `48080`.
Both services use the same generated or replacement key. Trusted-network
Control Center access binds to all interfaces and always requires the key.
Because the built-in servers use HTTP, use remote access only on a trusted
network or behind a TLS-terminating reverse proxy.

Saved Control Center access changes apply after a Control Center restart.
Model API authentication changes apply after a Model API restart. A shared-key
replacement requires both protected services to restart. The Settings tab has
explicit restart buttons for both services; restarting the Control Center
reconnects at the saved bind/port, while restarting the MLX server reloads the
selected model and clears active requests and retained prompt cache.

Telemetry retains a bounded JSONL history in the user store for up to 30 days.
The active UTC day stays in the primary log; prior days are automatically
rotated into daily JSONL archives. Housekeeping runs at startup, on the first
request or read after a UTC-day boundary, and during normal compaction. It
deletes records after the retention cutoff. The Recent queries panel uses
server-side pagination with selectable 10, 25, 50, or 100-row pages.

Telemetry records token counts, timings, client identity, offered tool names,
structured tool calls, and an optional bounded latest-user-query preview. It
does not retain complete conversations, responses, reasoning text, or tool
arguments.

Amber token-activity buckets show their failure count and can be selected to
filter the error table. The table classifies common failure types such as
client disconnects, cancellations, safety admission, memory, timeouts,
authentication, and model errors, while preserving the exact retained message
and request ID. This makes failures traceable to their client/address, API
endpoint, model, and query preview.

The Telemetry tab also identifies agent runs that may be looping. The detector
groups requests by client, model, address, and bounded query preview, then
scores sustained tool-call chains, dominant tool patterns, large contexts,
prompt-cache loss, and latency degradation. Warnings are advisory and include
the evidence used; MLX does not automatically cancel an agent based on this
heuristic because long, legitimate coding tasks can resemble loops. Separately,
the generation server stops exact token cycles after a substantial repeated
window, records the cancellation reason, and surfaces it in the model harness.
A final response without a tool call clears the advisory warning for that run.

Disable query previews or change retention before starting the server:

```bash
MLX_SERVER_TELEMETRY_QUERY_PREVIEW=0 scripts/mlx-server restart
MLX_SERVER_TELEMETRY_PREVIEW_CHARS=120 scripts/mlx-server restart
MLX_SERVER_TELEMETRY_RETENTION_DAYS=30 scripts/mlx-server restart
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
