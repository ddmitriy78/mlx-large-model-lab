# MLX Large Model Lab

This lab uses Qwen3 Coder Next 8-bit as the primary local model on Apple Silicon:

`mlx-community/Qwen3-Coder-Next-8bit`

It is the default because it is coding-focused and produced the best balance of
quality and speed for local defensive cybersecurity review, secure code changes,
and benchmark learning. The expected download is about 84.7 GB before cache
overhead. Keep the 4-bit model as the fast fallback when latency matters.

## Current Local Setup

Read this section first in future Codex sessions. It captures the machine,
runtime, local paths, aliases, and client wiring used by this workspace.

### Machine And Runtime

- Workspace: `/Users/dima/AI_ML/mlx_01`
- Machine: Apple Silicon MacBook Pro with 128 GB unified memory
- Python runtime: Python 3.12 managed by `uv`
- Main Python package command: `uv run mlx-lab`
- MLX server package command: `uv run mlx_lm.server`
- Hugging Face CLI command: `uv run hf`
- Dependencies are declared in `pyproject.toml`: `mlx-lm`,
  `huggingface-hub[hf-xet]`, `psutil`, and `rich`

Initialize or refresh the environment:

```bash
cd /Users/dima/AI_ML/mlx_01
uv sync --python 3.12
uv run mlx-lab info
```

Authenticate Hugging Face when downloads need higher rate limits or gated model
access:

```bash
uv run hf auth login
```

### Source Of Truth Files

- `configs/models.json`: model registry, aliases, local directories, and sizes
- `scripts/mlx-server`: one-model-at-a-time OpenAI-compatible MLX server helper
- `scripts/opencode-mlx`: launcher for safe opencode modes per local model
- `opencode.jsonc`: opencode `local-mlx` provider and local model ids
- `/Users/dima/.config/opencode/opencode.jsonc`: global opencode config used
  when launching opencode from other project directories
- `models/`: downloaded MLX model folders
- `.hf-cache/`: Hugging Face cache used by this lab
- `logs/mlx-server.log`: MLX server log
- `logs/mlx-server.pid`: managed server PID

### Configured Local Models

Use `scripts/mlx-server models` to see the current state. These aliases are
currently configured:

| Alias | Local directory | Common use |
| --- | --- | --- |
| `qwen3-coder-next-8bit` | `models/Qwen3-Coder-Next-8bit` | Primary opencode, coding, and security-review model |
| `qwen3-coder-next` | `models/Qwen3-Coder-Next-4bit` | Fast fallback when latency matters |
| `gemma-4-31b-it-optiq` | `models/gemma-4-31B-it-OptiQ-4bit` | Multimodal threat modeling and diagram analysis |
| `gemma-4-12b-coder-8bit` | `models/Gemma-4-12B-Coder-Fable5-Composer2.5-8bit` | Experimental lightweight Python coding and opencode tool use |
| `qwen3-coder-30b-8bit` | `models/Qwen3-Coder-30B-A3B-Instruct-8bit` | New 8-bit Qwen coding model |
| `qwen3-next-80b-8bit` | `models/Qwen3-Next-80B-A3B-Instruct-8bit` | Large MoE instruction comparison |
| `deepseek-r1-qwen-32b` | `models/DeepSeek-R1-Distill-Qwen-32B-4bit` | Reasoning comparison |
| `deepseek-r1-qwen-32b-8bit` | `models/DeepSeek-R1-Distill-Qwen-32B-MLX-8Bit` | 8-bit reasoning comparison |
| `qwen3-6-27b-obliterated` | `models/Qwen3.6-27B-OBLITERATED-MLX-4bit` | Experimental long-context comparison |

### Common Commands

Daily MLX server flow:

```bash
# See configured model aliases.
scripts/mlx-server list

# Load the primary everyday model into memory.
scripts/mlx-server load qwen3-coder-next-8bit

# Check what is running and which model is loaded.
scripts/mlx-server current

# Use opencode against the loaded server.
scripts/opencode-mlx primary

# Unload the model and free unified memory when done.
scripts/mlx-server unload
```

Switch to the fast fallback when latency matters:

```bash
scripts/mlx-server switch qwen3-coder-next
scripts/opencode-mlx qwen
```

Switch back to the primary 8-bit model:

```bash
scripts/mlx-server switch qwen3-coder-next-8bit
scripts/opencode-mlx primary
```

Run a prompt through the lab without starting the server:

```bash
uv run mlx-lab generate \
  --model qwen3-coder-30b-8bit \
  --max-tokens 1200 \
  --prompt "Write a Python CLI arg parser."
```

Benchmark all downloaded configured models:

```bash
uv run mlx-lab benchmark --all-models --limit 2 --max-tokens 1200
```

Start the OpenAI-compatible MLX server for a specific model:

```bash
scripts/mlx-server start
scripts/mlx-server status
scripts/mlx-server test
```

`start` uses the default primary model, `qwen3-coder-next-8bit`. Prefer
`load` or `switch` when you want to name the model explicitly:

```bash
scripts/mlx-server load qwen3-coder-next-8bit
scripts/mlx-server load qwen3-coder-next
scripts/mlx-server load gemma-4-31b-it-optiq
```

The server launcher advertises downloaded configured model aliases through
`GET /v1/models`, so discovery-based harnesses can use model ids like
`qwen3-coder-next`, `qwen3-coder-next-8bit`, `qwen3-coder-30b-8bit`,
`gemma-4-31b-it-optiq` or `deepseek-r1-qwen-32b-8bit`
directly.

Stop or inspect the server:

```bash
scripts/mlx-server current
scripts/mlx-server logs
scripts/mlx-server unload
```

Print the raw server command that will run:

```bash
scripts/mlx-server command qwen3-coder-30b-8bit
```

### MLX Server Memory

The MLX server process holds the currently loaded model weights in unified
memory. This laptop can run the 60 GB+ and 85 GB-class models, but it should
not try to keep several large models resident at once.

Use these commands for predictable memory behavior:

```bash
# Load or switch to one model. This restarts the server process.
scripts/mlx-server load qwen3-coder-next-8bit
scripts/mlx-server switch gemma-4-31b-it-optiq

# Show the active server process and loaded model path.
scripts/mlx-server current

# Stop the server and release model memory.
scripts/mlx-server unload
```

The server advertises all configured downloaded model aliases at `/v1/models`,
but for memory hygiene treat `scripts/mlx-server load <alias>` as the point
where a model is intentionally brought into memory. Use `unload` before long
benchmarks, LM Studio multimodal work, or switching to another heavy model.

### opencode Setup

opencode is configured in `opencode.jsonc` with provider `local-mlx`:

- Base URL: `http://127.0.0.1:48080/v1`
- API key placeholder: `local-mlx`
- Default model: `local-mlx/qwen3-coder-next-8bit`
- Fast fallback model: `local-mlx/qwen3-coder-next`
- Qwen3 Coder 30B 8-bit model id:
  `local-mlx/qwen3-coder-30b-8bit`
- Qwen3 Coder Next 8-bit model id:
  `local-mlx/qwen3-coder-next-8bit`
- Qwen3 Next 80B A3B 8-bit chat-only model id:
  `local-mlx/qwen3-next-80b-8bit`

For cross-project launches such as running
`../mlx_01/scripts/opencode-mlx qwen-coder-next8` from another repository, keep
the matching `local-mlx` model entries in
`/Users/dima/.config/opencode/opencode.jsonc` as well.

Serve the model first, then launch opencode:

```bash
scripts/mlx-server load qwen3-coder-next-8bit
scripts/opencode-mlx primary
```

Use the lightweight experimental Gemma coding specialist:

```bash
scripts/mlx-server load gemma-4-12b-coder-8bit
scripts/opencode-mlx gemma-coder
```

This model has experimental opencode tool access. Its fine-tune is focused on
Python and algorithmic tasks and is intentionally less safety-aligned than the
primary model, so review file edits and shell commands carefully. The upstream
MLX snapshot currently labels the architecture
`gemma4_unified`; with `mlx-lm 0.31.3`, the downloaded `config.json` is locally
adjusted to `gemma4` before use.

Use chat-only mode when you do not want opencode tool calls:

```bash
scripts/opencode-mlx qwen-chat
```

Use the Qwen3 Next 80B A3B model as a chat-only opencode agent:

```bash
scripts/mlx-server load qwen3-next-80b-8bit
scripts/opencode-mlx qwen-next80
```

Use the Qwen3 Coder Next 8-bit model as an opencode Build/tool agent:

```bash
scripts/mlx-server load qwen3-coder-next-8bit
scripts/opencode-mlx qwen-coder-next8
```

Direct opencode commands also work:

```bash
opencode -m local-mlx/qwen3-coder-next
opencode -m local-mlx/qwen3-coder-next-8bit
opencode --agent qwen30-chat
opencode --agent qwen-next80-chat
```

### LM Studio Setup

LM Studio is configured to use:

```text
/Users/dima/.lmstudio/models
```

This lab exposes model folders to LM Studio under:

```text
/Users/dima/.lmstudio/models/mlx_01
```

The LM Studio entries are symlinks back to this workspace's `models/`
directories. The Qwen3 Coder 30B 8-bit link is:

```bash
ln -sfn /Users/dima/AI_ML/mlx_01/models/Qwen3-Coder-30B-A3B-Instruct-8bit \
  /Users/dima/.lmstudio/models/mlx_01/Qwen3-Coder-30B-A3B-Instruct-8bit
```

The Qwen3 Coder Next 8-bit link is:

```bash
ln -sfn /Users/dima/AI_ML/mlx_01/models/Qwen3-Coder-Next-8bit \
  /Users/dima/.lmstudio/models/mlx_01/Qwen3-Coder-Next-8bit
```

The Qwen3 Next 80B A3B 8-bit link is:

```bash
ln -sfn /Users/dima/AI_ML/mlx_01/models/Qwen3-Next-80B-A3B-Instruct-8bit \
  /Users/dima/.lmstudio/models/mlx_01/Qwen3-Next-80B-A3B-Instruct-8bit
```

LM Studio's CLI exists at:

```text
/Users/dima/.lmstudio/bin/lms
```

It may not be on `PATH`. Use the full path if needed:

```bash
/Users/dima/.lmstudio/bin/lms ls
```

If `lms ls` times out while starting the daemon, reopen LM Studio and refresh
My Models. The filesystem symlink is still the important integration point.

### Adding A New MLX Model

Use this checklist when adding another Hugging Face MLX model to the lab:

1. Download the model into `models/`:

   ```bash
   uv run hf download <publisher/model> \
     --local-dir models/<model-folder>
   ```

2. Add the model to `configs/models.json` with:

   - `repo_id`
   - one or more short `aliases`
   - `local_dir`
   - `estimated_size_gb`
   - `role`
   - `notes`

3. Confirm the lab sees it:

   ```bash
   scripts/mlx-server models
   ```

4. Smoke test generation:

   ```bash
   uv run mlx-lab generate \
     --model <alias> \
     --max-tokens 256 \
     --prompt "Reply with a short checklist for testing a local model."
   ```

5. Add opencode support in `opencode.jsonc`:

   - add a provider model under `provider.local-mlx.models`
   - add a chat-only agent under `agent` if the model should not call tools
   - keep `tool_call: true` only for models being tested as opencode Build/tool
     agents

6. Add a shortcut in `scripts/opencode-mlx` when the model will be used often.

7. Link it into LM Studio:

   ```bash
   ln -sfn /Users/dima/AI_ML/mlx_01/models/<model-folder> \
     /Users/dima/.lmstudio/models/mlx_01/<model-folder>
   ```

8. Validate the setup:

   ```bash
   jq empty configs/models.json
   jq empty opencode.jsonc
   scripts/mlx-server models
   scripts/opencode-mlx --help
   ```

### README vs Skill

This setup should live in the README because it is specific to this machine,
workspace path, local model aliases, opencode config, and LM Studio symlink
layout. A Codex skill would make sense later only if the same workflow needs to
be reused across multiple MLX workspaces or if model onboarding should become a
repeatable automated procedure.

## 1. Create The Environment

Use Python 3.12 through `uv` so MLX packages stay isolated from the system Python:

```bash
uv sync --python 3.12
```

Check the lab configuration:

```bash
uv run mlx-lab info
```

## 2. Download The Primary Model

Optional: authenticate with Hugging Face for better rate limits:

```bash
uv run hf auth login
```

Check the download size without downloading weights:

```bash
uv run mlx-lab download --dry-run
```

Then download the primary model into `models/Qwen3-Coder-Next-8bit`:

```bash
uv run mlx-lab download
```

If the download is interrupted, run the same command again. Hugging Face will resume already-fetched files.

## 2.1 Download Additional 8-bit Models

This machine has 128 GB unified memory, so 30B-40B 8-bit MLX models are
practical, and 70B-80B 8-bit MLX models are viable when context length is kept
modest. The main pressure points are model weights, KV cache, and other apps
sharing unified memory.

Use the Hugging Face CLI for extra models that are not the primary configured
download:

```bash
uv run hf auth login
```

```bash
uv run hf download mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit \
  --local-dir models/Qwen3-Coder-30B-A3B-Instruct-8bit
```

Run the same command again if the download is interrupted; Hugging Face resumes
files that are already present.

Recommended 8-bit MLX models for this Mac:

| Use case | Model | Expected download | Local directory |
| --- | --- | ---: | --- |
| Coding first pick | `mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit` | 32.4 GB | `models/Qwen3-Coder-30B-A3B-Instruct-8bit` |
| Newer Qwen3.6 general/agentic model | `mlx-community/Qwen3.6-35B-A3B-8bit` | 37.7 GB | `models/Qwen3.6-35B-A3B-8bit` |
| General reasoning/chat baseline | `mlx-community/Qwen3-32B-8bit` | 34.8 GB | `models/Qwen3-32B-8bit` |
| Larger MoE experiment | `mlx-community/Qwen3-Next-80B-A3B-Instruct-8bit` | 84.7 GB | `models/Qwen3-Next-80B-A3B-Instruct-8bit` |
| Experimental uncensored/community tune | `mlx-community/Qwen3.6-40B-Claude-4.6-Opus-Deckard-Heretic-Uncensored-Thinking-8bit` | 41.5 GB | `models/Qwen3.6-40B-Claude-4.6-Opus-Deckard-Heretic-Uncensored-Thinking-8bit` |

Download any of them by changing the repo id and destination:

```bash
uv run hf download mlx-community/Qwen3.6-35B-A3B-8bit \
  --local-dir models/Qwen3.6-35B-A3B-8bit
```

```bash
uv run hf download mlx-community/Qwen3-32B-8bit \
  --local-dir models/Qwen3-32B-8bit
```

```bash
uv run hf download mlx-community/Qwen3-Next-80B-A3B-Instruct-8bit \
  --local-dir models/Qwen3-Next-80B-A3B-Instruct-8bit
```

```bash
uv run hf download mlx-community/Qwen3.6-40B-Claude-4.6-Opus-Deckard-Heretic-Uncensored-Thinking-8bit \
  --local-dir models/Qwen3.6-40B-Claude-4.6-Opus-Deckard-Heretic-Uncensored-Thinking-8bit
```

After download, run a text prompt by pointing `--model` at the local directory:

```bash
uv run mlx-lab generate \
  --model models/Qwen3-Coder-30B-A3B-Instruct-8bit \
  --max-tokens 1200 \
  --prompt "Write a Python function that validates an email address without using a regex. Explain the tradeoffs."
```

Or start an OpenAI-compatible server for that model:

```bash
uv run mlx_lm.server \
  --model models/Qwen3-Coder-30B-A3B-Instruct-8bit \
  --host 127.0.0.1 \
  --port 48080
```

The downloaded Qwen3 Coder 30B 8-bit model is also configured in
`configs/models.json`, so these shorter aliases work throughout the lab:

- `qwen3-coder-30b-8bit`
- `qwen3-coder-30b`
- `qwen3-coder-30b-a3b`

The downloaded Qwen3 Next 80B A3B 8-bit model is also configured in
`configs/models.json`, so these aliases work throughout the lab:

- `qwen3-next-80b-8bit`
- `qwen3-next-80b`
- `qwen3-next-80b-a3b`

Use the alias for generation:

```bash
uv run mlx-lab generate \
  --model qwen3-coder-30b-8bit \
  --max-tokens 1200 \
  --prompt "Write a Python function that validates an email address without using a regex. Explain the tradeoffs."
```

Serve it through the helper script:

```bash
scripts/mlx-server load qwen3-coder-30b-8bit
scripts/mlx-server test
```

Use it from opencode as a Build/tool model:

```bash
scripts/opencode-mlx qwen30
```

Or use the chat-only opencode agent:

```bash
scripts/opencode-mlx qwen30-chat
```

For the Qwen3 Next 80B A3B model, use the same server helper with the larger
model alias and keep initial prompts short while checking memory pressure:

```bash
scripts/mlx-server load qwen3-next-80b-8bit
scripts/mlx-server test
scripts/opencode-mlx qwen-next80
```

LM Studio looks for downloaded models under its configured models directory,
which is currently `/Users/dima/.lmstudio/models`. This lab uses the
`mlx_01` group there. This model is linked into LM Studio with:

```bash
ln -sfn /Users/dima/AI_ML/mlx_01/models/Qwen3-Coder-30B-A3B-Instruct-8bit \
  /Users/dima/.lmstudio/models/mlx_01/Qwen3-Coder-30B-A3B-Instruct-8bit
```

The Qwen3 Next 80B A3B 8-bit model is linked with:

```bash
ln -sfn /Users/dima/AI_ML/mlx_01/models/Qwen3-Next-80B-A3B-Instruct-8bit \
  /Users/dima/.lmstudio/models/mlx_01/Qwen3-Next-80B-A3B-Instruct-8bit
```

Then reopen LM Studio or refresh My Models and look under `mlx_01`.

Gemma 4 is retained as the multimodal specialist for diagram/image analysis and
threat-modeling workflows. The text benchmark helpers and opencode agents use
`mlx-lm`, so treat those as text-only paths. For image input, use LM Studio's
multimodal UI/API or install and use `mlx-vlm`; `mlx-vlm` is not currently part
of this repo's Python environment.

## 3. Run One Prompt

Start with deterministic output and a short generation:

```bash
uv run mlx-lab generate \
  --max-tokens 256 \
  --prompt "Write a Python function that validates an email address without using a regex. Explain the tradeoffs."
```

```bash
uv run mlx-lab generate \
  --max-tokens 1200 \
  --prompt "Write a Python function that validates an email address without using a regex. Explain the tradeoffs."
```

Use the fast fallback explicitly when you want lower latency:

```bash
uv run mlx-lab generate \
  --model models/Qwen3-Coder-Next-4bit \
  --max-tokens 1200 \
  --prompt "Write a Python function that validates an email address without using a regex. Explain the tradeoffs."
```

```bash
uv run mlx-lab generate \
  --model models/DeepSeek-R1-Distill-Qwen-32B-MLX-8Bit \
  --max-tokens 3200 \
  --prompt "Return only the final answer. Start with a Python code block. Write a function that validates an email address without using regex, then explain the tradeoffs in 3 short bullets."
```

What to watch:

- first-token latency, because the model has to load and prefill the prompt
- generation tokens/sec
- peak memory reported by MLX
- whether the answer is useful enough for daily coding

## 4. Run Benchmarks

Start with two prompts:

```bash
uv run mlx-lab benchmark --limit 2 --max-tokens 384
```

Then run the full first-model prompt set:

```bash
uv run mlx-lab benchmark --max-tokens 512
```

Compare every downloaded model in `configs/models.json` with the same prompt set:

```bash
uv run mlx-lab benchmark --all-models --limit 2 --max-tokens 1200
```

Compare only newly added models that still need the app-security workflow run:

```bash
uv run mlx-lab benchmark \
  --model-set new_models \
  --prompts prompts/app_security_workflow.jsonl \
  --max-tokens 1200 \
  --skip-completed \
  --output-dir runs/control-plane-outputs
```

Compare only the 60 GB+ local models, while skipping rows already present in
`runs/*.jsonl` with the same model, prompt file, prompt name, and generation
settings:

```bash
uv run mlx-lab benchmark \
  --model-set large_models \
  --prompts prompts/app_security_workflow.jsonl \
  --max-tokens 1200 \
  --skip-completed \
  --output-dir runs/control-plane-outputs
```

For the cybersecurity harness decision, use:

```bash
uv run mlx-lab benchmark \
  --model-set cybersecurity_harness \
  --prompts prompts/cybersecurity_harness.jsonl \
  --max-tokens 1600 \
  --skip-completed \
  --output-dir runs/cybersecurity-harness-outputs
```

### Prompt Sets

Benchmarks read JSONL prompt files. Each line must include:

- `name`: stable prompt identifier used in reports and output filenames
- `category`: grouping label such as `coding`, `security_review`, or `bug_fix`
- `prompt`: the actual prompt text sent to the model

Available prompt sets:

- `prompts/first_model.jsonl`: seven short coding/security smoke-test prompts
- `prompts/app_security_workflow.jsonl`: four longer app-build, security-review, bug-fix, and release-hardening prompts
- `prompts/cybersecurity_harness.jsonl`: defensive security triage, secure-code-fix, detection, and harness-evaluation prompts

Use a specific prompt set with `--prompts`:

```bash
uv run mlx-lab benchmark \
  --all-models \
  --prompts prompts/app_security_workflow.jsonl \
  --max-tokens 4000
```

### Output Files

For longer tests, save each model/prompt answer as a Markdown file:

```bash
uv run mlx-lab benchmark \
  --all-models \
  --prompts prompts/app_security_workflow.jsonl \
  --output-dir runs/app-security-outputs \
  --max-tokens 4000
```

The command writes those Markdown files under a run-specific subdirectory inside
the directory you pass to `--output-dir`, for example:

```text
runs/app-security-outputs/
  20260610-203000-all-models/
    qwen3-coder-next/
      01-create-notes-api.md
      02-security-review-notes-api.md
```

### Reports

Every benchmark writes:

- raw JSONL into `runs/`
- a readable HTML report into `runs/`
- optional Markdown output files when `--output-dir` is set

Use the JSONL for programmatic analysis. Use the HTML report to compare:

- model load time
- total run time
- prompt and generation token throughput
- peak memory
- finish reasons
- output-file links
- full model outputs

Create or recreate an HTML report from an existing benchmark JSONL file without
rerunning the models:

```bash
uv run mlx-lab report runs/20260610-195339-all-models.jsonl
```

## 5. Use The Local Control Plane

Start the local web control plane:

```bash
uv run mlx-lab ui
```

Then open `http://127.0.0.1:8765`.

To start both the MLX server and Control Center automatically when you log in
to macOS, install the per-user LaunchAgents:

```bash
scripts/macos-startup install qwen3-coder-next-8bit
scripts/macos-startup status
```

Pass a different configured model alias to change the model loaded after the
next login. Remove the login services with `scripts/macos-startup uninstall`.
Uninstalling leaves the currently loaded model running; unload it separately
with `scripts/mlx-server unload` if needed.

The local-only page now acts as an MLX control center and benchmark console. It
lets you:

- see the actual runtime model rather than only the process startup command
- monitor unified-memory use, available memory, wired memory, swap, and process RSS
- monitor active requests with exact prompt progress and cancel them without restarting
- reject requests before prefill when context, output, or estimated KV-cache memory is unsafe
- automatically clear retained cache and cancel active work at low-memory thresholds
- inspect retained prompt/KV-cache entries and clear one entry or all entries
- load, restart, switch, or unload one downloaded model
- see the managed PID, HTTP health, model role, estimated weight size, and cache limit
- choose a prompt set
- run one downloaded model or all downloaded models
- set `max_tokens` and an optional prompt limit
- set a Markdown output directory
- start a benchmark run
- watch benchmark console output
- view recent runs
- open generated HTML reports and JSONL files
- compare generation speed with a simple bar chart

The managed server is pinned to the model selected with `scripts/mlx-server`
or the control center. Requests naming another configured model are rejected
instead of silently replacing the resident model. Retained prompt caches are
limited to four entries and 8 GB by default so long OpenCode sessions cannot
accumulate many full-context KV-cache copies.

The server also keeps a 16 GB admission reserve and cancels active work if
available memory falls below 8 GB. Admission uses the tokenizer's exact prompt
count, reserves the requested output tokens, enforces the detected model context
window, and estimates the active KV-cache allocation before prefill begins.
These limits can be edited live in the control center. Live edits last until the
managed MLX process restarts; environment variables set the next process's
defaults.

Override those safeguards only when intentionally testing:

```bash
MLX_SERVER_PIN_MODEL=0 scripts/mlx-server load qwen3-coder-30b-8bit
MLX_SERVER_PROMPT_CACHE_SIZE=6 \
MLX_SERVER_PROMPT_CACHE_BYTES=12GB \
  scripts/mlx-server load qwen3-coder-30b-8bit

MLX_SERVER_MAX_CONTEXT_TOKENS=128000 \
MLX_SERVER_MAX_OUTPUT_TOKENS=32000 \
MLX_SERVER_MEMORY_RESERVE_BYTES=20GB \
MLX_SERVER_MEMORY_ABORT_BYTES=10GB \
  scripts/mlx-server load qwen3-coder-30b-8bit
```

The control plane also exposes a small local API for model memory management:

```bash
curl http://127.0.0.1:8765/api/mlx-server/help
curl http://127.0.0.1:8765/api/mlx-server/list
curl http://127.0.0.1:8765/api/mlx-server/status
curl http://127.0.0.1:8765/api/mlx-server/current
curl -X POST http://127.0.0.1:8765/api/mlx-server/load \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-coder-next-8bit"}'
curl -X POST http://127.0.0.1:8765/api/mlx-server/switch \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-coder-next"}'
curl -X POST http://127.0.0.1:8765/api/mlx-server/cancel \
  -H 'Content-Type: application/json' \
  -d '{}'
curl -X POST http://127.0.0.1:8765/api/mlx-server/cache/clear \
  -H 'Content-Type: application/json' \
  -d '{}'
curl -X POST http://127.0.0.1:8765/api/mlx-server/safety \
  -H 'Content-Type: application/json' \
  -d '{"enabled":true,"max_context_tokens":128000,"max_output_tokens":32000,"memory_reserve_bytes":17179869184,"memory_abort_bytes":8589934592}'
curl -X POST http://127.0.0.1:8765/api/mlx-server/unload
```

The OpenAI-compatible model server remains on `http://127.0.0.1:48080/v1`.
The control API runs on the control plane, calls the existing
`scripts/mlx-server` helper, and only accepts downloaded models already
registered in `configs/models.json`.

The control plane does not expose arbitrary shell execution. It only starts the
whitelisted `mlx-lab benchmark` command with form options and the whitelisted
`scripts/mlx-server` commands for list/status/load/switch/unload. The additional
runtime routes are limited to cancellation, safety-policy updates, and retained
cache eviction. The status response includes runtime model identity,
prompt-cache entries, active requests, admission policy, recent prefill
activity, and system/process memory statistics.

## 6. Serve The Model Locally

After generation works, start the OpenAI-compatible local server with the helper script:

```bash
scripts/mlx-server load qwen3-coder-next-8bit
scripts/mlx-server status
scripts/mlx-server test
```

Stop or restart it with:

```bash
scripts/mlx-server unload
scripts/mlx-server switch qwen3-coder-next-8bit
```

Tail logs with:

```bash
scripts/mlx-server logs
```

The raw server command is:

```bash
.venv/bin/python -m mlx_lab.mlx_server \
  --model models/Qwen3-Coder-Next-8bit \
  --host 127.0.0.1 \
  --port 48080
```

Test it from another terminal:

```bash
curl http://127.0.0.1:48080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-coder-next-8bit",
    "messages": [
      {"role": "user", "content": "Create a checklist for reviewing a Python CLI tool for command injection risks."}
    ],
    "max_tokens": 300,
    "temperature": 0
  }'
```

## Learning Notes

Use this order before adding more models:

1. Use Qwen3 Coder Next 4-bit for fast text/code harness iteration.
2. Use Qwen3 Coder Next 8-bit for higher-quality text review when latency is acceptable.
3. Keep Gemma 4 for multimodal threat modeling, diagram review, and image analysis.
4. Learn how prompt length changes prefill time and memory.
5. Try `--max-kv-size 4096` if memory is tight during long-context tests.
