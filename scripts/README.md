# Helper scripts

These checkout helpers manage the local MLX server, macOS login startup, and
OpenCode targets. They resolve models and writable paths through the same
per-user store as `mlx-lab`; they do not read the capability JSON directly.

## `mlx-server`

Start or inspect the OpenAI-compatible server:

```bash
scripts/mlx-server start
scripts/mlx-server status
scripts/mlx-server current
scripts/mlx-server test
scripts/mlx-server logs
```

Explicitly load, switch, or unload a model:

```bash
uv run mlx-lab plan-model qwen3-coder-30b-8bit
scripts/mlx-server load qwen3-coder-30b-8bit
scripts/mlx-server switch qwen3-coder-next
scripts/mlx-server unload
```

Plan the fit before loading a model for the first time. The server's default
memory reserve scales from 4 to 16 GiB based on detected unified memory, and
the emergency threshold defaults to half that reserve. The corresponding
environment variables override those adaptive values.

List capability models with the current user's download status:

```bash
scripts/mlx-server models
scripts/mlx-server list
```

The launcher delegates alias resolution and model discovery to
`python -m mlx_lab.cli`. Logs, PID state, telemetry, and runtime state live
under the user store reported by:

```bash
uv run mlx-lab path logs
```

The primary environment overrides are:

```text
MLX_LAB_HOME
MLX_SERVER_MODEL
MLX_SERVER_HOST
MLX_SERVER_PORT
MLX_SERVER_START_TIMEOUT
MLX_SERVER_RUNTIME_STATE
MLX_SERVER_MODEL_LIST
MLX_SERVER_ADVERTISE_UNDOWNLOADED
MLX_SERVER_PIN_MODEL
MLX_SERVER_PROMPT_CACHE_SIZE
MLX_SERVER_PROMPT_CACHE_BYTES
MLX_SERVER_MEMORY_GUARD
MLX_SERVER_MAX_CONTEXT_TOKENS
MLX_SERVER_MAX_OUTPUT_TOKENS
MLX_SERVER_MEMORY_RESERVE_BYTES
MLX_SERVER_MEMORY_ABORT_BYTES
MLX_SERVER_TELEMETRY_PATH
MLX_SERVER_TELEMETRY_MAX_RECORDS
MLX_SERVER_TELEMETRY_RETENTION_DAYS
MLX_SERVER_TELEMETRY_QUERY_PREVIEW
MLX_SERVER_TELEMETRY_PREVIEW_CHARS
```

The default endpoint is `http://127.0.0.1:48080/v1`.
When Model API protection is enabled in Control Center, these helpers
automatically send the shared key. Print it for other local clients with
`uv run mlx-lab api-key`.

## `macos-startup`

Install per-user LaunchAgents that start the MLX server and Control Center at
login:

```bash
scripts/macos-startup install qwen3-coder-30b-8bit
scripts/macos-startup status
scripts/macos-startup uninstall
```

Environment-backed server settings that must survive Control Center restarts
should be supplied when the services are installed. For example, a 128 GiB Mac
running Qwen3 Coder 30B with a 128,000-token context can retain the full prompt
instead of crossing the default 8 GB cache limit:

```bash
MLX_SERVER_PROMPT_CACHE_BYTES=16GB \
  scripts/macos-startup install qwen3-coder-30b-8bit
```

The installer stores the selected prompt-cache byte budget in both
LaunchAgents. Reinstall the services to change the persistent value.

The optional model is loaded after the next login. The Control Center is kept
alive by `launchd`; the model server continues to use `mlx-server`, so normal
load, switch, and unload commands remain authoritative.

Uninstalling login services does not unload a model already held in unified
memory. Run `scripts/mlx-server unload` separately when desired.

## `opencode-mlx`

Load the desired model first, then launch a configured OpenCode target:

```bash
scripts/mlx-server load qwen3-coder-30b-8bit
scripts/opencode-mlx primary
```

The wrapper reads the shared key from the MLX Lab user store, supplies it to
OpenCode through `MLX_MODEL_API_KEY`, and selects the repository's
`opencode.jsonc` through `OPENCODE_CONFIG` even when OpenCode is launched from
another project. The repository configuration does not contain the secret.

Available targets include:

```text
primary
qwen
qwen-chat
qwen30
qwen30-chat
qwen-coder-next8
qwen-coder-next8-chat
qwen-next80
gemma
gemma-coder
deepseek
deepseek8
qwen-oblit
ravenx
models
```

OpenCode model definitions remain in `opencode.jsonc`. The server is pinned to
the explicitly loaded model by default, so select a matching target after
switching. RavenX is intentionally chat-only: its first OpenCode tool-agent
benchmark repeated 669 `read` calls in one length-capped response. Keep tools
disabled until a bounded tool-call evaluation passes.

## Control API

After `uv run mlx-lab ui`, the local Control Center exposes fixed read and
mutation routes:

```bash
curl http://127.0.0.1:8765/api/mlx-server/help
curl http://127.0.0.1:8765/api/mlx-server/list
curl http://127.0.0.1:8765/api/mlx-server/status
curl http://127.0.0.1:8765/api/model-statistics
curl 'http://127.0.0.1:8765/api/telemetry?window_seconds=86400&limit=100'
curl -X POST http://127.0.0.1:8765/api/mlx-server/load \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-coder-30b-8bit"}'
curl -X POST http://127.0.0.1:8765/api/mlx-server/restart \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-coder-30b-8bit"}'
curl -X POST http://127.0.0.1:8765/api/mlx-server/unload
curl -X POST http://127.0.0.1:8765/api/control-center/restart
```

The control routes accept only known operations and downloaded capability
models. They are not a general command or HTTP proxy.

The telemetry response includes `agent_loop_risks`, an advisory list of
possible long-running agent loops. Each item reports the risk level and score,
consecutive tool steps, context and cache behavior, latency, evidence signals,
and a recommended operator action. Detection uses the existing
privacy-bounded telemetry fields and does not cancel a request based on the
agent-loop score. The separate server-side repetition guard does stop exact
token cycles and reports the reason to streaming clients. The response also
includes `failure_types` and `error_events` so the Control Center can show the
classified failure, client/address, endpoint, model, retained error message,
bounded query preview, and request ID behind every amber chart bucket.
