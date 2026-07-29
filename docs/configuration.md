# Configuration and storage

MLX Large Model Lab keeps immutable application capabilities separate from
mutable user and machine state.

## Capability catalog

`configs/models.json` is the source catalog. The build includes the same file
as `mlx_lab/data/model_capabilities.json` in the wheel.

Each model capability may define:

- `repo_id`
- `aliases`
- `estimated_size_gb`
- `role`
- `notes`

The top-level catalog also defines `first_model` and optional
`benchmark_sets`. It must not define `local_dir`; validation rejects catalogs
that mix a local path into a capability.

## User configuration

The generated `user-config.json` uses schema version 1:

```json
{
  "schema_version": 1,
  "hardware": {
    "system": "Darwin",
    "release": "example",
    "machine": "arm64",
    "processor": "arm",
    "python_version": "3.12.x",
    "memory_bytes": 38654705664,
    "logical_cpu_count": 12,
    "physical_cpu_count": 8
  },
  "preferences": {
    "default_model": "qwen3_coder_30b_a3b_instruct_8bit"
  },
  "control_center": {
    "policy_version": 2,
    "access_mode": "local",
    "port": 8765,
    "authentication": "none",
    "model_api_authentication": "none"
  },
  "models": {
    "qwen3_coder_30b_a3b_instruct_8bit": {
      "local_dir": "/path/to/Qwen3-Coder-30B-A3B-Instruct-8bit"
    }
  },
  "updated_at": 0
}
```

Hardware is refreshed when a CLI command starts. A model entry records where
that user keeps the corresponding downloaded capability. Download status is
verified from the filesystem rather than trusted as a stale boolean.

## Control Center access

Control Center defaults to local-only access on `127.0.0.1:8765` without API
key authentication. The Settings tab can change the next-start access mode,
port, Control Center authentication mode, Model API authentication mode, and
their shared API key. Network access binds to `0.0.0.0` and always requires
API key authentication; it cannot be combined with unauthenticated access.

When either service's authentication is enabled, the shared API key is stored separately in
`control-center-credentials.json` with owner-only file permissions. It is
never included in `user-config.json` or returned by the settings API. The
terminal that starts `uv run mlx-lab ui` prints the current key for entry in
the browser lock screen when Control Center authentication is active. Browser
storage is scoped to the current tab. Local command-line clients can print the
same key with `uv run mlx-lab api-key`.

Each service snapshots its authentication policy and key when it starts.
Control Center access changes require a Control Center restart; Model API
authentication changes require a Model API restart. Replacing the shared key
requires restarting both protected services. Command-line `--host` and
`--port` arguments remain explicit one-run overrides. The Settings tab exposes
separate restart buttons for both services. Control Center restart reconnects
the page using the saved bind and port; MLX server restart reloads the selected
model and clears active requests and retained prompt cache.

## Pre-download fit planning

`mlx-lab plan-model <model-alias>` uses the live hardware snapshot and current
memory availability without writing sizing results into the capability
catalog. `--memory-gib` simulates another Mac but does not replace the detected
hardware stored in `user-config.json`.

The default runtime reserve is derived from detected unified memory. Explicit
`MLX_SERVER_MEMORY_RESERVE_BYTES` and `MLX_SERVER_MEMORY_ABORT_BYTES` values
override that adaptive policy for deliberate testing.

## Default locations

macOS:

```text
~/Library/Application Support/mlx-lab
```

Linux:

```text
${XDG_DATA_HOME:-~/.local/share}/mlx-lab
```

Windows:

```text
%LOCALAPPDATA%\mlx-lab
```

MLX itself currently targets Apple Silicon; the non-macOS locations make the
storage contract predictable for tooling and future portability.

## Environment overrides

| Variable | Purpose |
| --- | --- |
| `MLX_LAB_HOME` | Override the complete user-store root |
| `MLX_LAB_USER_CONFIG` | Override only the user JSON path |
| `MLX_LAB_CAPABILITIES_PATH` | Use an alternate immutable catalog |
| `MLX_LAB_PROJECT_ROOT` | Override source-checkout discovery and bundled assets |

Server-specific telemetry and runtime-state variables continue to override
their individual files when set.

Telemetry defaults to a 30-day retention window. The server rotates completed
prior UTC days into daily JSONL archives and deletes records after the cutoff.
Set `MLX_SERVER_TELEMETRY_RETENTION_DAYS` before starting the server to use a
different window (1–365 days); `MLX_SERVER_TELEMETRY_MAX_RECORDS` remains an
independent upper bound on retained records.

## Resolution order

For a capability model, the runtime resolves a local directory in this order:

1. the explicit path in `user-config.json`;
2. a complete matching model under the user store `models/` directory;
3. a complete case-insensitive matching folder under a source checkout's
   legacy `models/` directory;
4. the future download destination beneath the user store.

A complete model has `config.json` and at least one `*.safetensors` file.
Legacy discovery records the path during synchronization; it never copies or
moves model weights.

## Contributor rule

Code that needs model configuration must use `mlx_lab.user_store` or the
merged helpers exported by `mlx_lab.cli`. Do not read `configs/models.json`
directly in server or UI code, and do not reintroduce machine paths into the
catalog.
