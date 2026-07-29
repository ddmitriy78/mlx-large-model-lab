# Development Audit Log

## Purpose

This log records meaningful changes to the local MLX lab, their validation, and
remaining follow-up work.

### 2026-07-28 — Review and sanitize the public enhancement release

Context:
- Review the complete enhancement worktree before publishing it to the public
  repository, with an explicit boundary against contributor paths,
  credentials, screenshots, telemetry identifiers, and machine-specific state.

Implementation:
- Replaced hard-coded contributor model paths with portable server aliases and
  replaced personal LaunchAgent labels with project-owned identifiers.
- Corrected the RavenX chat-only OpenCode step-limit field.
- Moved local-use screenshots into an ignored private review directory and
  removed their public documentation references.
- Generalized retained audit evidence that identified one machine or request,
  added public-text path regression coverage, and made private JSON temporary
  files owner-only at creation.
- Kept the Control Center usable when the managed model server is offline by
  returning its offline status payload as a normal operational response.

Docs updated:
- `docs/development-audit-log.md`
- `docs/mlx-control-center-journey.html`

Validation:
- Passed the complete 58-test suite, Python compilation, shell syntax, JSON
  validation, lockfile and environment checks, package builds, whitespace
  checks, a targeted credential scan, and source/wheel/sdist privacy scans.
- Rebuilt the Control Center from an isolated empty user store and verified
  desktop and mobile browser states without using retained local telemetry.

Follow-up:
- The public Git history predates this sanitized tree. Rewriting already
  published history would require a separate, explicitly approved migration.

### 2026-07-28 — Show prompt-ingestion memory capacity

Context:
- Make active prompt prefill understandable in both token counts and the
  corresponding memory scale.

Implementation:
- Reused the active model's existing per-token KV-memory estimate and the
  Control Center's adaptive byte formatter.
- Added processed and total estimated KV-memory values beside live prompt-token
  progress, automatically formatted as MiB, GiB, or TiB.

Docs updated:
- `README.md`
- `docs/implementation-status.md`
- `docs/development-audit-log.md`

Validation:
- Added focused dashboard structure coverage for the model-specific KV-memory
  calculation and its processed/total presentation.
- Passed Python compilation, embedded JavaScript syntax validation,
  `git diff --check`, and the complete 55-test suite.
- Rendered a populated active request at 1,440 × 900 and 390 × 844; the
  capacity text remained readable with no clipping, overlap, console errors,
  or page-level horizontal overflow.

Follow-up:
- The displayed capacity is an estimate of KV memory, not serialized request
  payload or disk-storage size; actual allocation can vary by MLX/runtime
  behavior.

### 2026-07-28 — Retain long OpenCode prompts across tool steps

Context:
- OpenCode reached an 86,485-token audit conversation and every subsequent
  tool step repeated a multi-minute full prompt prefill.
- Live runtime state showed zero cached sequences with an 8 GB retained-cache
  limit even though the model remained pinned in memory.

Implementation:
- Reused MLX-LM's existing LRU prefix cache and the existing
  `MLX_SERVER_PROMPT_CACHE_BYTES` override instead of adding a new cache.
- Persisted the selected byte budget in both macOS LaunchAgents so a
  Control Center-initiated server restart does not silently return to 8 GB.
- Validated a 16 GB retained cache on a representative 128 GiB profile, enough for the
  Qwen3 Coder 30B model's 128,000-token context estimate while preserving the
  existing 16 GiB safety reserve and 8 GiB emergency threshold.

Docs updated:
- `README.md`
- `scripts/README.md`
- `docs/implementation-status.md`
- `docs/development-audit-log.md`

Validation:
- Before the change, telemetry showed five consecutive 83,688-86,485-token
  requests with zero cached tokens and 141-184 second durations.
- Verified shell syntax, both installed LaunchAgents, and the live 16 GB
  runtime cache limit after restarting the managed services.
- An isolated 86,508-token cold request took 129.762 seconds and retained an
  8,504,279,040-byte assistant entry alongside the existing
  2,365,292,544-byte OpenCode system prefix.
- The identical warm request reused 86,508 cached tokens, processed one new
  prompt token, and completed in 2.488 seconds, approximately 52 times faster.
- Ran `git diff --check`, shell syntax validation, and the complete 54-test
  suite.

Follow-up:
- The first request after a model restart remains a cold prefill; subsequent
  requests reuse the retained prefix while it remains within the byte budget.
- Re-evaluate the override when changing models, context caps, or Mac memory.

### 2026-07-28 — Add the OpenClaw question-routing classifier

Context:
- Add `NeelkanthSingh/openclaw-qwen3.5-2b-classifier` for the narrow question
  routing use case without presenting it as another general-purpose model.

Implementation:
- Added the complete 4-bit MLX repository to the immutable capability catalog
  with concise aliases, its 1.1 GB size estimate, and explicit classifier-only
  guidance.
- Kept the model out of the general coding and cybersecurity benchmark sets.
- Documented the managed download/load workflow, required classifier system
  prompt, small output budget, and expected JSON response shape.
- Reused the existing Hugging Face downloader, per-user model registry,
  managed server, and Model Harness without model-specific runtime code or new
  dependencies.

Docs updated:
- `README.md`
- `docs/implementation-status.md`
- `docs/development-audit-log.md`

Validation:
- Validated the catalog JSON, whitespace, the focused catalog regression test,
  and isolated CLI alias discovery.
- Verified live Hugging Face metadata reports eight files and 1,079,471,282
  repository bytes, including 1,059,404,951 weight bytes.
- The 4,096-token fit plan reports `SAFE` on a simulated 128 GiB profile, with
  approximately 3.41 GB of model working memory before the 16 GiB safety
  reserve.
- Downloaded and registered the complete model in the per-user store and
  confirmed its `qwen3_5` architecture with `mlx-lm 0.31.3`.
- Ran three live OpenAI-compatible requests with thinking disabled. The model
  returned valid JSON and correctly classified representative questions as
  `simple`, `tricky`, and `complex`.
- Confirmed that omitting `chat_template_kwargs.enable_thinking=false` returns
  reasoning without JSON content; documented the required request setting and
  current Model Harness limitation.
- Ran the complete 54-test suite and `git diff --check`, then restored the
  previously active Qwen3 Coder 30B server and confirmed it is ready.

Follow-up:
- Add a labeled routing evaluation set only if this classifier will influence
  automated production routing decisions.

### 2026-07-27 — Stop exact output-repetition loops

Context:
- The RavenX model still entered a token-level repetition loop in the Control
  Center harness after its separate OpenCode tool access had been disabled.
- The harness request for `hi` reached its configured 1,200-token ceiling while
  repeating the same sentence instead of reaching a natural stop.

Implementation:
- Added an incremental server-side token repetition guard that stops generation
  after at least 96 output tokens when an exact cycle covers at least 64 tokens
  and repeats at least four times.
- Reused the existing request context stop, cancellation telemetry, SSE
  extension, and harness transcript rendering.
- Added an inline harness notice and explicit status when repeated output is
  stopped automatically.

Validation:
- Passed Python compilation, whitespace checks, and the complete 51-test suite.
- Restarted the RavenX model server and Control Center, then confirmed the
  original `hi` harness request completed naturally at 211 of 1,200 tokens.
- Sent a deliberate exact-cycle prompt through the harness. The server stopped
  it with a `repetition` signal after a four-token pattern repeated 16 times,
  retained the reason in telemetry, and left no active request behind.
- Verified the inline notice, final status, transcript layout, and lack of
  horizontal overflow in rendered desktop and 390-pixel mobile views.

Configuration:
- `MLX_SERVER_REPETITION_GUARD=0` disables the guard.
- The minimum generated tokens, repeated window, minimum repetitions, and
  maximum pattern length can be overridden with
  `MLX_SERVER_REPETITION_MIN_TOKENS`,
  `MLX_SERVER_REPETITION_WINDOW_TOKENS`,
  `MLX_SERVER_REPETITION_MIN_REPEATS`, and
  `MLX_SERVER_REPETITION_MAX_PATTERN_TOKENS`.

### 2026-07-27 — Contain RavenX OpenCode tool-call runaway

Context:
- Diagnose a RavenX OpenCode admission failure without retaining session timing.
- Preserve the 128,000-token safety cap instead of admitting a multi-million
  token request.

Evidence:
- The failing request contained 678
  messages, 8,495,796 input characters, 2,654,966 prompt tokens, and a
  32,000-token output reserve.
- The preceding length-capped response emitted 669 `read` calls in one model
  step, repeatedly cycling over the same 21 files 31–32 times.
- OpenCode executed those calls before beginning its next model iteration, so
  an agent step limit alone does not contain this failure mode.

Implementation:
- Added a dedicated RavenX OpenCode model with tools disabled, a 4,096-token
  output ceiling, and a one-step chat-only agent.
- Added the `ravenx` launcher target and made the wrapper select the repository
  OpenCode configuration through `OPENCODE_CONFIG` from any project directory.
- Routed OpenCode background/title requests through a tool-free active-model
  alias with a 256-token output ceiling, so they cannot target a different
  pinned model or monopolize generation while RavenX is loaded.
- Reclassified RavenX as a chat comparison pending a bounded tool-call pass.

Validation:
- Validated the JSON and shell syntax, whitespace, 13 focused tests, and the
  complete 49-test suite.
- Confirmed from a different project directory that OpenCode resolves
  `ravenx-chat` with one step, every built-in tool disabled, wildcard permission
  denied, a 128,000-token context limit, and a 4,096-token output limit.
- The first live smoke exposed a separate background title request that emitted
  4,460 tokens under the global 32,000-token limit and blocked the chat queue;
  the dedicated active-model alias now caps that path at 256 tokens.
- Repeated the live OpenCode smoke from the affected project directory. It
  completed in 18 seconds with the exact response `ravenx chat safe`; telemetry
  showed no offered tools, a 256-token background request, a 4,096-token chat
  ceiling, and no active requests left behind.

Follow-up:
- Do not promote RavenX to a tool-capable agent until a test proves that one
  response cannot emit or execute an unbounded batch of tool calls.

### 2026-07-26 — Add RavenX cybersecurity-agent capability

Context:
- Add the requested Hugging Face MLX repository to the local model catalog and
  managed downloader.

Implementation:
- Added the 69.3 GB RavenX Qwen3.6 35B-A3B MoE repository, concise aliases,
  experimental security-agent guidance, and cybersecurity harness membership.
- Reused the existing Hugging Face downloader and per-user model registry
  without adding dependencies or model-specific runtime code.

Docs updated:
- `README.md`
- `docs/development-audit-log.md`

Validation:
- Verified live Hugging Face metadata reports 21 files and 69,341,421,696
  repository bytes, including 69,321,320,169 weight bytes.
- Downloaded all 14 SafeTensor shards and registered the completed per-user
  model directory under the `ravenx-cyberagent` alias.
- Confirmed the local model config uses `qwen3_5_moe`, which is supported by
  the installed `mlx-lm 0.31.3`.
- Ran JSON validation, Python compilation, whitespace checks, 13 focused
  catalog/sizing/user-store tests, and the complete 49-test suite.
- The 8,192-token fit plan passes total 128 GiB capacity and disk checks but
  reports `CAUTION` until enough currently used unified memory is released.

Follow-up:
- Benchmark output quality, tool-call behavior, and peak unified-memory use
  before relying on the model for security work.

### 2026-07-26 — Make telemetry failures actionable and add service restarts

Context:
- Replace ambiguous amber token buckets with enough detail to identify what
  failed, where it originated, and which request produced it.
- Make the two required configuration/recovery restarts available directly in
  the Control Center.

Implementation:
- Classified cancellations, client disconnects, safety admission failures,
  memory errors, timeouts, authentication errors, model errors, and otherwise
  unknown request errors from the retained status/message.
- Added a responsive Errors & cancellations panel with type/origin summaries
  and time, client/address, endpoint, model, message, bounded query preview,
  and request ID for each event.
- Made amber chart buckets show failure counts and filter the detail table when
  selected; added per-client and per-model failure totals.
- Enriched older live server payloads from retained history so the new view
  works before the already-running model server is restarted.
- Added explicit Control Center and MLX model-server restart controls, fixed
  restart routes, confirmation copy, progress state, and saved-port reconnect.

Docs updated:
- `README.md`
- `PLANS.md`
- `scripts/README.md`
- `docs/configuration.md`
- `docs/implementation-status.md`
- `docs/development-audit-log.md`

Validation:
- Added aggregation, older-runtime compatibility, UI wiring, and service
  restart tests.
- Ran Python compilation, embedded JavaScript syntax validation,
  `git diff --check`, and the complete 48-test suite.
- Verified sanitized retained telemetry classifies both reported Broken pipe
  events as Client disconnected from OpenCode at `/v1/chat/completions`.
- Inspected populated 1440×1000 and 390×844 renders with no page-level
  horizontal overflow; verified amber-bucket filtering and both restart
  controls at desktop and mobile widths.

Follow-up:
- Extend the classifier only when retained production messages reveal a
  distinct actionable failure category; unknown messages remain Request error.

### 2026-07-26 — Paginate and expire Recent queries telemetry

Context:
- Make the Telemetry tab practical when retained query previews exceed a single
  browser-sized table.
- Enforce the requested 30-day data lifecycle with automatic archival and
  deletion instead of relying only on a record-count cap.

Implementation:
- Added server-side Recent queries pagination with 10, 25, 50, and 100-row
  page sizes, page metadata, bounded navigation, and responsive controls.
- Kept summaries, timelines, and failure analysis scoped to the full selected
  time range while returning only the requested query page.
- Rotate prior UTC days from the active telemetry file into daily JSONL
  archives and delete records after the retention cutoff during startup,
  day-boundary reads/writes, and compaction.
- Preserved the existing maximum-record cap and privacy-bounded query previews;
  no new dependency or conversation content was added.

Docs updated:
- `README.md`
- `PLANS.md`
- `scripts/README.md`
- `docs/configuration.md`
- `docs/implementation-status.md`
- `docs/development-audit-log.md`

Validation:
- Added focused pagination and archive/purge lifecycle tests.
- Ran Python compilation, embedded JavaScript syntax validation,
  `git diff --check`, and the complete 49-test suite.
- Exercised page 1 and page 2 with 32 synthetic records: rows and navigation
  states matched the API metadata.
- Inspected populated 1440×1000 desktop and 390×844 mobile renders with no
  console errors, pagination overflow, or page-level horizontal overflow.

Follow-up:
- Revisit the record-count cap if installations routinely generate more than
  5,000 telemetry records inside the 30-day time window.

### 2026-07-26 — Protect the Model API with the shared Control Center key

Context:
- Require authenticated access to the OpenAI-compatible Model API on port
  `48080` without creating or synchronizing a second credential.
- Keep Control Center model operations, the browser harness, lifecycle scripts,
  and OpenCode working after protection is enabled.

Implementation:
- Added an independent Model API authentication toggle to Control Center
  settings while reusing the existing owner-only shared credential.
- Added constant-time Bearer and `X-API-Key` validation around Model API and
  MLX internal control requests.
- Authenticated Control Center-to-MLX requests, health checks, test requests,
  macOS service checks, and the OpenCode wrapper with the shared key.
- Replaced the committed OpenCode placeholder key with environment
  substitution and added `mlx-lab api-key` for local client setup.
- Recorded the active Model API policy in runtime state so the Settings tab can
  distinguish saved and active protection.

Docs updated:
- `README.md`
- `SECURITY.md`
- `scripts/README.md`
- `docs/configuration.md`
- `docs/implementation-status.md`
- `docs/development-audit-log.md`

Validation:
- Ran Python compilation, shell syntax checks, `git diff --check`, and the
  complete 41-test suite.
- Inspected 1440×1000 desktop and 390×844 mobile Settings renders with no
  clipping or page-level horizontal overflow.
- Enabled protection on the live server and confirmed unauthenticated `/v1`
  and internal control requests return `401`, while shared-key requests return
  `200`.
- Confirmed Control Center status and OpenCode configuration load the same key
  without exposing it in the repository.

Follow-up:
- Put a TLS-terminating proxy and stronger identity/authorization controls in
  front of either service before any internet-facing deployment.

### 2026-07-26 — Detect likely agent loops in MLX telemetry

Context:
- Make long-running tool agents observable when they continue after useful work
  appears complete or cross a prompt-cache cliff.
- Preserve operator control because legitimate repository work can also require
  many tool steps.

Implementation:
- Reused the existing privacy-bounded request telemetry to group likely agent
  runs by client, address, model, and bounded query preview.
- Added advisory risk scoring for sustained tool-call chains, dominant tool
  patterns, large contexts, prompt-cache loss, and latency degradation.
- Added high, elevated, and watch warning cards to the Telemetry tab with the
  exact evidence and recommended operator action.
- Corrected the aggregate cache-hit percentage to use uncached plus cached
  prompt tokens as its denominator.
- Clear a run's warning after a completed response without a tool call, and
  never cancel work automatically.

Docs updated:
- `README.md`
- `PLANS.md`
- `scripts/README.md`
- `docs/implementation-status.md`
- `docs/development-audit-log.md`

Validation:
- Added focused aggregation tests for a high-risk active loop with cache
  thrashing and for warning clearance after an agent's final response.
- Verified the detector against retained OpenCode telemetry from the reported
  incident; it classified the run as high risk and identified the cache and
  latency cliff.
- Ran Python compilation, embedded JavaScript syntax validation, shell syntax
  checks, `git diff --check`, and the complete 36-test suite.
- Inspected populated 1440×1000 desktop and 390×844 mobile Playwright renders;
  both showed the high-risk incident with no console errors, alert overflow, or
  page-level horizontal overflow.

Follow-up:
- Calibrate thresholds against more agent clients and long legitimate coding
  sessions before considering opt-in notifications or automatic intervention.

### 2026-07-26 — Add first-model fit planning and adaptive memory guidance

Context:
- Give first-time users a safe path from an empty install to one downloaded
  model without assuming every Apple Silicon Mac has the same unified memory.
- Make download size, model weight memory, KV-cache growth, runtime overhead,
  disk headroom, and system reserve visible before a large download.

Implementation:
- Added `mlx-lab plan-model` with detected-memory planning and a
  `--memory-gib` simulator for any Mac memory tier.
- Reused Hugging Face dry-run metadata for current repository and weight sizes,
  with an offline capability-catalog fallback.
- Reused model configuration fields for KV-cache estimates and the same
  estimate in server request admission.
- Added safe, caution, and unsafe verdicts covering memory capacity, current
  availability, context limits, and disk space.
- Added estimated size and role columns to the human-readable model catalog.
- Added configured-model selection to `mlx-lab download --model`.
- Made the server reserve adaptive: 15% rounded up, bounded from 4 to 16 GiB,
  with an emergency threshold at half the reserve.
- Documented `gemma-4-12b-coder-8bit` as a cautious 24 GiB starter example,
  including its experimental, Python-focused scope.

Docs updated:
- `README.md`
- `PLANS.md`
- `scripts/README.md`
- `docs/configuration.md`
- `docs/implementation-status.md`
- `docs/development-audit-log.md`

Validation:
- Added sizing coverage for 24, 36, 48, 64, and 128 GiB Macs, model-config
  KV-cache math, fit verdicts, context overflow, and new CLI arguments.
- Ran Python compilation, the complete 28-test suite, shell syntax validation,
  catalog-only CLI smoke plans for 24 and 128 GiB profiles, and an exact
  Hugging Face metadata plan for a 36 GiB profile.
- Built the source distribution and wheel and verified the sizing module,
  capability catalog, and starter prompts are packaged.
- Verified the 24 GiB example against current Hugging Face metadata at a
  4,096-token context: 12.7 GB repository size, 19.3 GiB planned memory with
  reserve, and approximately 4.7 GiB capacity headroom.

Follow-up:
- Add model-fit comparison and smaller-model recommendations to the Control
  Center.
- Validate planning estimates against measured peaks for more architectures and
  cache-quantization settings.

### 2026-07-26 — Separate capabilities from per-user installation state

Context:
- Prepare the repository for community use without committing one operator's
  hardware, downloaded-model paths, logs, or benchmark data as application
  capabilities.
- Preserve existing large model downloads when adopting the new layout.

Implementation:
- Converted `configs/models.json` into an immutable capability catalog with no
  local directories and packaged it into wheels.
- Added a schema-versioned per-user store for hardware, preferred default,
  registered model paths, weights, cache, runs, logs, telemetry, and runtime
  state.
- Added automatic discovery and registration of complete models in the user
  store and legacy checkout, without copying weights.
- Routed the CLI, MLX wrapper, shell launcher, Control Center, benchmark
  artifacts, telemetry, and macOS startup logs through the shared resolver.
- Added config, path, model-list, model-resolution, and model-registration CLI
  commands.
- Replaced machine-specific onboarding with public quick-start, contributor,
  security, configuration, implementation-status, and roadmap documents.

Docs updated:
- `README.md`
- `CONTRIBUTING.md`
- `SECURITY.md`
- `PLANS.md`
- `scripts/README.md`
- `docs/configuration.md`
- `docs/implementation-status.md`
- `docs/development-audit-log.md`

Validation:
- Added user-store coverage for hardware persistence, legacy download
  discovery, user-store defaults, and catalog rejection of local paths.
- Ran Python compilation, the complete 22-test suite, all shell syntax checks,
  CLI config/model smoke tests, and `git diff --check`.
- Built the source distribution and wheel, verified the capability catalog and
  bundled prompt contents, and exercised catalog resolution from an unpacked
  wheel with no source checkout.
- Initialized the real per-user store, restarted the stale pre-migration
  server and Control Center processes, and confirmed model discovery, a
  20-token exact-response smoke test, and both local health APIs.

Follow-up:
- Select an OSI-approved license before announcing the repository for general
  reuse.
- Add macOS CI and validate a clean wheel install on a second Apple Silicon
  machine.
- Package the server lifecycle helper as a wheel entry point.

### 2026-07-25 — Promote Qwen3 Coder 30B 8-bit to the default

Context:
- Make `qwen3-coder-30b-8bit` the default local coding model across the lab,
  OpenCode, the managed MLX server, and macOS login startup.

Implementation:
- Changed the registry primary model to the existing downloaded Qwen3 Coder
  30B A3B Instruct 8-bit entry and updated the model roles.
- Changed the OpenCode default and `scripts/opencode-mlx primary` target.
- Changed the MLX server and macOS startup fallback models.
- Reinstalled the per-user LaunchAgents with the 30B model and switched the
  active local server to it.

Docs updated:
- `README.md`
- `scripts/README.md`
- `docs/development-audit-log.md`

Validation:
- Validated both JSON files with `jq` and all three launcher scripts with
  `bash -n`.
- Confirmed the default server command and `mlx-lab info` resolve the downloaded
  Qwen3 Coder 30B 8-bit model.
- Ran the complete 15-test suite and `git diff --check`.
- Linted and inspected the installed LaunchAgent, then completed a live
  OpenAI-compatible smoke test that returned `local mlx server ok`.

Follow-up:
- None required.

### 2026-07-25 — Improve Model Harness submission and run metrics

Context:
- Make the Model Harness keyboard flow immediate and expose the token and speed
  evidence needed to evaluate each response.

Implementation:
- Changed the message composer so Enter submits the prompt, while Shift+Enter
  inserts a new line and IME composition remains uninterrupted.
- Added an inline shortcut hint and updated the ready status to match the new
  keyboard behavior.
- Reused the server's streamed OpenAI `usage` payload for exact input and
  output token counts.
- Calculated output tokens per second over the browser-observed streamed
  generation interval and displayed all three metrics on each completed
  assistant message and in the completion status.

Docs updated:
- `README.md`
- `docs/development-audit-log.md`

Validation:
- Added regression coverage for Enter submission, Shift+Enter discoverability,
  usage-field wiring, the stream timer, and the per-message metrics display.
- Ran Python compilation, embedded JavaScript syntax validation,
  `git diff --check`, and the complete 15-test suite.
- Used the live Model Harness to verify Shift+Enter preserves a multiline draft
  and Enter submits a 32-token-capped local prompt. The response reported 30
  input tokens, 2 output tokens, and 1.6 output tok/s.
- Inspected 1440×1000 desktop and 390×844 mobile layouts with no page-level
  horizontal overflow and no browser warnings or errors.

Follow-up:
- Use saved benchmark generation rates for controlled model comparisons; the
  harness rate intentionally includes browser streaming and proxy overhead.

### 2026-07-25 — Add MLX usage telemetry

Context:
- Make request volume, token use, model activity, queries, tool calls, and MCP
  activity observable from the Control Center.
- Cover every OpenAI-compatible client using the managed MLX server instead of
  measuring only requests sent from the browser harness.

Implementation:
- Added request instrumentation at the MLX API-handler and generation layers
  for live status, exact prompt/cache/output token counts, duration, errors,
  client identity, offered tools, emitted structured tool calls, and bounded
  latest-user-query previews.
- Added a compacted JSONL history with a 5,000-record default retention limit
  and an internal live telemetry endpoint.
- Added the read-only `/api/telemetry` route with retained-history fallback
  when the model server is offline.
- Added a responsive Telemetry tab with time-range controls, active-request
  state, usage summary cards, token timeline, client/model/tool breakdowns,
  recent query previews, and explicit MCP attribution limits.
- Avoided storing full conversations, model output, reasoning, and tool
  arguments. Query previews can be disabled through an environment setting.
- Resolve request aliases such as `default_model` to the canonical resident
  model after lazy loading, before live or completed telemetry is published.

Docs updated:
- `README.md`
- `scripts/README.md`
- `docs/development-audit-log.md`

Validation:
- Added unit coverage for persistence, token/tool/MCP aggregation, offline
  fallback, and required Control Center controls.
- Ran Python compilation, JavaScript syntax validation, the 16-test unittest
  suite, a live local API fallback check, and populated desktop/mobile
  Playwright renders with no console errors or page overflow.

Follow-up:
- MCP attribution remains name-based because OpenAI-compatible tool schemas do
  not include a standard MCP server identity.

### 2026-07-25 — Consolidate benchmark controls and statistics

Context:
- Keep all benchmark-related actions and evidence in one Control Center
  workflow instead of splitting them between Operations and Model Statistics.

Implementation:
- Added a dedicated Benchmarks tab alongside Operations and Model Harness.
- Moved the benchmark form, latest comparison, saved-runs table, and console
  out of the Operations tab.
- Placed the existing aggregate benchmark summary, speed comparison, coverage
  notes, and complete model statistics table in the same Benchmarks tab.
- Refreshes run history and aggregate model statistics when a benchmark
  completes successfully.
- Renamed the per-run model field to Benchmark target so it is distinct from
  the managed runtime model.
- Standardized benchmark selectors, comparisons, fastest-model summaries,
  speed bars, and statistics rows on the full model artifact name so they can
  be matched directly with Telemetry and other Control Center views.

Docs updated:
- `README.md`
- `scripts/README.md`
- `docs/development-audit-log.md`

Validation:
- Updated the Control Center structure test to verify that both the benchmark
  form and statistics table live inside the Benchmarks panel.
- Added focused coverage for canonical benchmark names in state options and
  retained-run comparisons.
- Ran Python compilation, embedded JavaScript syntax validation, the complete
  19-test suite, and desktop/mobile browser checks against the live Control
  Center with no console errors or page-level horizontal overflow.

Follow-up:
- Consider preserving the last selected tab across page reloads if operators
  routinely move between the harness and benchmark views.

### 2026-07-24 — Add aggregate model statistics

Context:
- Make availability and benchmark coverage visible for every model in the
  Control Center instead of requiring manual JSONL report comparison.

Implementation:
- Added a read-only `/api/model-statistics` route that matches saved benchmark
  records to current registry identities and aggregates model size,
  availability, sample/run counts, prompt and generation rates, peak memory,
  and latest benchmark time.
- Kept configured models with no benchmark samples in the payload and counted
  unmatched historical rows separately to avoid misleading attribution.
- Added a responsive Model Statistics tab with coverage summary cards,
  generation-speed comparison bars, refresh control, and a complete model
  table.

Docs updated:
- `README.md`
- `scripts/README.md`
- `docs/development-audit-log.md`

Validation:
- Added unit coverage for current-key and legacy-path identity matching,
  aggregate calculations, and zero-sample configured models.
- Ran Python compilation, JavaScript syntax validation, the full unittest
  suite, live API checks, and desktop/mobile browser checks.

Follow-up:
- Gemma 4 12B Coder currently has no saved benchmark samples; run the standard
  prompt sets before using its performance metrics for model selection.

### 2026-07-24 — Add the Control Center model harness

Context:
- Make the loaded local model directly usable from the Control Center instead
  of requiring a separate API client or OpenCode session.

Implementation:
- Added a fixed `/api/harness/chat` proxy to the active MLX
  `/v1/chat/completions` endpoint.
- Added strict message, temperature, and output-token validation without
  allowing browser requests to select or implicitly load another model.
- Added a dedicated Model Harness tab with a responsive multi-turn conversation
  UI, streamed output, editable system prompt, generation settings, reasoning
  display, cancellation, and clear-chat controls.
- Reused the existing active-request telemetry, safety limits, and cancellation
  route.

Docs updated:
- `README.md`
- `scripts/README.md`
- `docs/development-audit-log.md`

Validation:
- Added unit coverage for active-model enforcement, streaming request shape,
  validation failures, and required browser controls.
- Ran Python compilation, JavaScript syntax validation, the full unittest
  suite, a live streaming completion, and desktop/mobile browser checks.

Follow-up:
- Consider opt-in conversation export and reusable prompt presets if repeated
  harness workflows need persistence beyond the current browser tab.

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
