# Scripts

Helper scripts for running the local MLX model lab.

## `mlx-server`

Starts and manages the local OpenAI-compatible MLX server. By default it serves:

`models/Qwen3-Coder-Next-8bit`

Default endpoint:

```text
http://127.0.0.1:48080/v1
```

The launcher runs the repo wrapper `mlx_lab.mlx_server`, which keeps MLX
generation behavior but makes `GET /v1/models` advertise downloaded models from
`configs/models.json`. Completion requests may use those advertised aliases,
for example `qwen3-coder-next` or `glm-4-5-air`.

## Commands

```bash
scripts/mlx-server start
scripts/mlx-server load qwen3-coder-next-8bit
scripts/mlx-server switch qwen3-coder-next
scripts/mlx-server unload
scripts/mlx-server current
scripts/mlx-server status
scripts/mlx-server test
scripts/mlx-server logs
scripts/mlx-server stop
scripts/mlx-server restart
scripts/mlx-server command
scripts/mlx-server models
scripts/mlx-server list
```

The same list/status/load/switch/unload flow is available over the local
control plane API after starting `uv run mlx-lab ui`:

```bash
curl http://127.0.0.1:8765/api/mlx-server/help
curl http://127.0.0.1:8765/api/mlx-server/list
curl http://127.0.0.1:8765/api/mlx-server/status
curl http://127.0.0.1:8765/api/mlx-server/current
curl -X POST http://127.0.0.1:8765/api/mlx-server/load \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-coder-next-8bit"}'
curl -X POST http://127.0.0.1:8765/api/mlx-server/unload
```

The control API is intentionally separate from the OpenAI-compatible `/v1`
model API. It calls this script with whitelisted commands and only loads
downloaded models registered in `configs/models.json`.

## Typical Flow

From the repo root:

```bash
scripts/mlx-server load qwen3-coder-next-8bit
scripts/mlx-server test
```

Then use the model from opencode:

```bash
scripts/opencode-mlx primary
```

Or use the launcher that chooses the safe opencode mode for each local model:

```bash
scripts/opencode-mlx qwen
scripts/opencode-mlx qwen30
scripts/opencode-mlx qwen-coder-next8
scripts/opencode-mlx primary
scripts/opencode-mlx glm
scripts/opencode-mlx gemma
scripts/opencode-mlx deepseek
scripts/opencode-mlx deepseek8
```

## Logs And PID

When started by the helper, runtime files are written under:

```text
logs/mlx-server.log
logs/mlx-server.pid
```

`logs/` is ignored by git.

## Environment Overrides

Use a different host or port:

```bash
MLX_SERVER_PORT=48081 scripts/mlx-server start
MLX_SERVER_PORT=48081 scripts/mlx-server status
```

Serve the fast fallback Qwen Coder model:

```bash
scripts/mlx-server load qwen3-coder-next
opencode -m local-mlx/qwen3-coder-next
```

Serve the primary Qwen3 Coder Next 8-bit model:

```bash
scripts/mlx-server load qwen3-coder-next-8bit
opencode -m local-mlx/qwen3-coder-next-8bit
```

Serve the downloaded Qwen3 Coder 30B 8-bit model:

```bash
scripts/mlx-server load qwen3-coder-30b-8bit
opencode -m local-mlx/qwen3-coder-30b-8bit
```

Serve another configured local or hosted model:

```bash
scripts/mlx-server models
scripts/mlx-server load qwen3-6-27b-obliterated
scripts/mlx-server load glm-4-5-air
```

GLM is configured as a chat-only opencode agent because it is not reliable as
an opencode tool-calling model. Use the `glm-chat` agent instead of launching
the model directly:

```bash
opencode --agent glm-chat
```

The workspace also defines chat-only agents for the other comparison models:

```bash
opencode --agent qwen-chat
opencode --agent qwen30-chat
opencode --agent qwen-coder-next8-chat
opencode --agent gemma-chat
opencode --agent deepseek-chat
opencode --agent deepseek8-chat
opencode --agent qwen-oblit-chat
```

Use Qwen Coder explicitly for the opencode Build/tool workflow:

```bash
opencode -m local-mlx/qwen3-coder-next
opencode -m local-mlx/qwen3-coder-next-8bit
opencode -m local-mlx/qwen3-coder-30b-8bit
```

`mlx_lm.server` serves one loaded model process at a time. Use
`scripts/mlx-server load <alias>` or `scripts/mlx-server switch <alias>` to
restart the server with exactly one model in memory. Use
`scripts/mlx-server unload` when you are done or before running memory-heavy
benchmarks or LM Studio multimodal work.

If you change the port, also update any clients that point at the server, such as opencode, Continue, or Cline.

Set `MLX_SERVER_MODEL_LIST=runtime` when you need raw upstream `mlx_lm.server`
model discovery instead of the configured workspace registry.
Set `MLX_SERVER_ADVERTISE_UNDOWNLOADED=1` to include configured models that are
not present under `models/` yet.

## Notes

Keep the server bound to `127.0.0.1` for local use. `mlx_lm.server` is intended for local development and basic testing, not as an internet-facing production service.
