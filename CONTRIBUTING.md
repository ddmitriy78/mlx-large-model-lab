# Contributing

Thank you for helping improve MLX Large Model Lab.

## Development setup

```bash
git clone git@github.com:ddmitriy78/mlx-large-model-lab.git
cd mlx-large-model-lab
uv sync --python 3.12
```

Use an isolated user store while developing:

```bash
export MLX_LAB_HOME="$(mktemp -d)"
.venv/bin/python -m mlx_lab.cli config
.venv/bin/python -m unittest discover -s tests -v
```

This prevents tests and experiments from changing your normal model registry,
runs, or logs.

## Pull requests

- Keep changes focused and explain the user-visible outcome.
- Add or update tests for behavior changes.
- Update README or architecture documentation when configuration, storage,
  CLI, API, or workflows change.
- Run the Python test suite, shell syntax checks, and `git diff --check`.
- Do not commit model weights, caches, telemetry, logs, generated runs, local
  paths, credentials, or personal machine details.

## Adding a model capability

Edit `configs/models.json` and provide a stable key, upstream repo id, useful
aliases, estimated size, role, and concise notes. Add the key to a benchmark
set only when it belongs in that comparison.

Do not add `local_dir`. Contributors associate their own downloads with a
capability through the per-user store:

```bash
uv run mlx-lab register-model <key-or-alias> /absolute/model/path
```

Validate the catalog and tests:

```bash
.venv/bin/python -m json.tool configs/models.json >/dev/null
.venv/bin/python -m unittest discover -s tests -v
```

## Scope and safety

The server and Control Center are local operational tools. Changes that expose
them beyond loopback, broaden command execution, retain additional prompt
content, or weaken model/memory admission controls need explicit threat-model
discussion in the pull request.
