# Development Audit Log

## Purpose

This log records meaningful changes to the local MLX lab, their validation, and
remaining follow-up work.

### 2026-07-24 — Add MLX request and memory safety controls

Context:
- Prevent long prompts and retained KV caches from making the Mac unresponsive.
- Make active MLX work stoppable without unloading the resident model.

Implementation:
- Added exact post-tokenization context/output admission before prompt prefill.
- Added an available-memory reserve, retained-cache eviction, and emergency
  active-request cancellation.
- Added active request telemetry and cancellation to the control center.
- Added per-entry and clear-all retained prompt-cache controls.
- Added live safety-limit controls with environment-backed restart defaults.

Docs updated:
- `README.md`
- `scripts/README.md`
- `scripts/mlx-server`
- `docs/development-audit-log.md`

Validation:
- Added unit coverage for combined-token admission, memory-reserve admission,
  request cancellation, control routing, and required dashboard controls.
- Ran Python compilation, the full unittest suite, and desktop/mobile browser
  validation.

Follow-up:
- Recalibrate per-token KV-cache estimates if a model architecture reports
  materially different live cache usage.

### 2026-07-12 — Add Gemma 4 12B Coder experiment

Context:
- Add `mlx-community/gemma-4-12b-coder-fable5-composer2.5-8bit` to the local
  model server at `models/Gemma-4-12B-Coder-Fable5-Composer2.5-8bit`.

Implementation:
- Registered server aliases and added the model to the new-model benchmark set.
- Added a chat-only opencode provider entry, agent, and launcher shortcut.
- Kept tool calling disabled pending behavioral and safety benchmarks.
- Applied a local architecture metadata adjustment required by `mlx-lm 0.31.3`.

Docs updated:
- `README.md`
- `docs/development-audit-log.md`

Validation:
- Downloaded and indexed three weight shards totaling 12,652,451,247 bytes.
- Validated the JSON and shell configuration syntax.
- Confirmed server discovery reports `gemma-4-12b-coder-8bit` as local.
- Switched the managed server to the model and completed an OpenAI-compatible
  chat request that returned a correct Python `is_even` implementation.

Follow-up:
- Benchmark Python correctness, broader repository work, security prompts, and
  tool-call behavior before considering promotion beyond chat-only use.

### 2026-07-12 — Enable experimental Gemma tool access

Context:
- Allow the Gemma 4 12B Coder agent to use opencode tools for coding work.

Implementation:
- Removed the one-step ceiling that forced OpenCode's text-only summary prompt.
- Enabled question, shell, file, search, task, web, todo, and skill tools.
- Advertised provider tool-call support while retaining the existing agent name
  and launcher aliases for compatibility.

Docs updated:
- `README.md`
- `configs/models.json`
- `scripts/opencode-mlx`
- `docs/development-audit-log.md`

Validation:
- Validated project JSON, launcher shell syntax, and whitespace checks.
- Confirmed the MLX OpenAI endpoint returns a structured `tool_calls` response.
- Ran the full opencode agent non-interactively; Gemma invoked `bash` once for
  `pwd`, received the result, and returned the correct workspace directory.

Follow-up:
- Treat tool support as experimental until multi-step repository edits and
  safety behavior have been evaluated against the existing harnesses.
