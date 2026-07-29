# Implementation status

## Operational

- Direct local generation through `mlx-lm`.
- Model downloads and registration.
- Opt-in question-complexity routing with the 4-bit
  `openclaw-qwen3.5-2b-classifier`, kept separate from general chat, coding,
  tool-use, and broad comparison sets.
- Pre-download fit planning against detected or simulated unified memory,
  current availability, free disk, model weights, KV cache, and adaptive safety
  reserves.
- Model alias resolution and one-model-at-a-time server management.
- OpenAI-compatible model discovery and chat completions.
- Control Center operations, model harness, benchmarks, statistics, telemetry,
  paginated recent queries, 30-day daily archive/purge housekeeping,
  classified error/cancellation drill-down, agent-loop-risk warnings,
  prompt-cache controls, request cancellation, memory admission, and live
  prompt-ingestion progress in both tokens and estimated KV-memory size.
- Selectable amber failure buckets plus client/address, endpoint, model,
  retained-message, query-preview, and request-ID attribution.
- In-app restart controls for the Control Center and selected MLX model server.
- Control Center local/network bind settings, with optional local
  authentication and mandatory API key authentication for network access;
  saved access changes apply together after a Control Center restart.
- Optional Model API Bearer authentication that reuses the Control Center key,
  including authenticated health checks, model controls, harness requests, and
  OpenCode wrapper requests.
- macOS login services persist the configured retained prompt-cache byte budget
  across Control Center and model-server restarts.
- JSONL benchmark records, Markdown outputs, and HTML reports.
- macOS login startup helpers.

## Configuration boundary

The source capability catalog contains portable model metadata only. The
schema-versioned per-user store owns:

- detected hardware
- the user's preferred default capability
- paths to downloaded models
- model/cache/run/log directories
- Control Center access preferences and permission-restricted credentials
- telemetry and runtime state

Existing complete downloads in a checkout `models/` folder are discovered and
recorded by path. No weights are moved.

## Intentionally simplified

- The repository shell scripts remain the lifecycle entry point for the
  background server; a wheel-only lifecycle command is planned.
- User configuration has schema validation but no cross-version migration
  framework yet.
- Trusted-network Control Center access uses bearer API keys but the built-in
  server does not terminate TLS or provide a multi-user authorization model.
- MCP attribution is name-based because the OpenAI tool schema has no standard
  MCP server identity.
- Agent-loop detection is heuristic and advisory. Privacy-bounded telemetry can
  identify suspicious tool-call chains and cache/latency degradation, but it
  cannot determine whether each edit was semantically necessary.
- Model-fit results are conservative estimates. Actual peak memory still needs
  validation with the loaded architecture, context, cache behavior, and other
  applications running.
- The OpenClaw classifier requires
  `chat_template_kwargs: {"enable_thinking": false}` to return JSON content
  through the MLX OpenAI-compatible server. The current Model Harness does not
  expose that per-request template setting.

## Validation

The current suite covers the Control Center, memory safety, common Mac memory
tiers, model-fit verdicts, telemetry, benchmark aggregation, user-store
initialization, capability validation, and legacy model discovery. Public
release still needs clean-machine and wheel installation verification. The
24 GiB starter example is metadata-verified at a 4,096-token planning context;
actual availability must still be checked on the target Mac.
