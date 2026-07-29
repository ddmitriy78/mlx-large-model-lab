from __future__ import annotations

import hmac
import json
import logging
import os
import re
import threading
import time
import urllib.parse
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Any

import psutil
from mlx_lm import server as mlx_server

from .cli import configured_models, model_is_downloaded, model_label, model_path
from .model_sizing import (
    GIB,
    model_memory_profile,
    recommended_abort_threshold_bytes,
    recommended_memory_reserve_bytes,
)
from .telemetry import TelemetryStore, mcp_server_for_tool
from .user_store import control_center_settings, sync_user_store, user_paths


TRUTHY = {"1", "true", "yes", "on", "all"}
RUNTIME_STATE_LOCK = threading.Lock()
PROMPT_CACHE_MAX_BYTES: int | None = None
WHITESPACE_RE = re.compile(r"\s+")
MODEL_API_ACCESS: dict[str, Any] = {"authentication": "none"}


class TokenRepetitionGuard:
    """Detect exact repeated token cycles after enough generated output."""

    def __init__(
        self,
        *,
        min_generated_tokens: int = 96,
        min_repeated_tokens: int = 64,
        min_repeats: int = 4,
        max_pattern_tokens: int = 128,
    ) -> None:
        self.min_generated_tokens = max(1, min_generated_tokens)
        self.min_repeated_tokens = max(1, min_repeated_tokens)
        self.min_repeats = max(2, min_repeats)
        self.max_pattern_tokens = max(1, max_pattern_tokens)
        retained = max(
            self.min_generated_tokens,
            self.min_repeated_tokens + self.max_pattern_tokens,
            self.min_repeats * self.max_pattern_tokens,
        )
        self.tokens: deque[int] = deque(maxlen=retained)
        self.generated_tokens = 0

    def observe(self, token: int) -> tuple[int, int] | None:
        self.tokens.append(int(token))
        self.generated_tokens += 1
        if self.generated_tokens < self.min_generated_tokens:
            return None

        tokens = list(self.tokens)
        max_pattern = min(
            self.max_pattern_tokens,
            len(tokens) // self.min_repeats,
        )
        for pattern_tokens in range(1, max_pattern + 1):
            repeats = max(
                self.min_repeats,
                (self.min_repeated_tokens + pattern_tokens - 1)
                // pattern_tokens,
            )
            repeated_tokens = pattern_tokens * repeats
            if len(tokens) < repeated_tokens:
                continue
            pattern = tokens[-pattern_tokens:]
            if tokens[-repeated_tokens:] == pattern * repeats:
                return pattern_tokens, repeats
        return None


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        logging.warning("Invalid %s; using %s.", name, default)
        return default


def _env_size(name: str, default: str) -> int:
    value = os.environ.get(name, default)
    try:
        return int(mlx_server._parse_size(value))
    except (KeyError, TypeError, ValueError):
        logging.warning("Invalid %s; using %s.", name, default)
        return int(mlx_server._parse_size(default))


def _model_api_access_snapshot() -> dict[str, Any]:
    settings = control_center_settings()
    access = {
        "authentication": settings["model_api_authentication"],
    }
    if access["authentication"] == "api_key":
        access["api_key"] = control_center_settings(include_secret=True)["api_key"]
    return access


def valid_model_api_key(
    candidate: str,
    access: dict[str, Any] | None = None,
) -> bool:
    active_access = access or MODEL_API_ACCESS
    if active_access.get("authentication") != "api_key":
        return True
    secret = str(active_access.get("api_key") or "")
    return bool(candidate) and bool(secret) and hmac.compare_digest(candidate, secret)


def _telemetry_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""
    parts: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if isinstance(text, str):
            parts.append(text)
    return "\n".join(parts)


def _telemetry_tool_name(tool: Any) -> str:
    if not isinstance(tool, dict):
        return ""
    function = tool.get("function")
    if isinstance(function, dict):
        return str(function.get("name") or "")[:160]
    return str(tool.get("name") or "")[:160]


def _telemetry_client(handler: Any) -> str:
    explicit = str(handler.headers.get("X-MLX-Client") or "").strip()
    if explicit:
        return explicit[:80]
    user_agent = str(handler.headers.get("User-Agent") or "").strip()
    lower_agent = user_agent.casefold()
    if "opencode" in lower_agent:
        return "OpenCode"
    if "curl" in lower_agent:
        return "curl"
    if "python" in lower_agent:
        return "Python client"
    if "mozilla" in lower_agent:
        return "Browser client"
    return user_agent[:80] or "Unknown client"


def _telemetry_request_record(
    handler: Any,
    request: Any,
    request_id: str,
) -> dict[str, Any]:
    request_type = str(getattr(request, "request_type", "unknown"))
    messages = getattr(request, "messages", []) or []
    tools = getattr(request, "tools", []) or []
    input_texts: list[str] = []
    query = ""
    tool_results = 0

    if request_type == "chat" and isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            text = _telemetry_text(message.get("content"))
            if text:
                input_texts.append(text)
            if message.get("role") == "user" and text:
                query = text
            if message.get("role") == "tool":
                tool_results += 1
    else:
        query = _telemetry_text(getattr(request, "prompt", ""))
        if query:
            input_texts.append(query)

    offered_tools = [
        name for name in (_telemetry_tool_name(tool) for tool in tools) if name
    ]
    preview_enabled = os.environ.get(
        "MLX_SERVER_TELEMETRY_QUERY_PREVIEW",
        "1",
    ).strip().lower() in TRUTHY
    preview_limit = max(
        0,
        min(_env_int("MLX_SERVER_TELEMETRY_PREVIEW_CHARS", 240), 2_000),
    )
    query_preview = (
        WHITESPACE_RE.sub(" ", query).strip()[:preview_limit]
        if preview_enabled and preview_limit
        else ""
    )
    model = str(SAFETY.model_profile.get("model") or handler.requested_model)
    client_address = getattr(handler, "client_address", ("", 0))
    return {
        "id": request_id,
        "started_at": time.time(),
        "status": "active",
        "request_type": request_type,
        "endpoint": str(getattr(handler, "path", "")),
        "model": model,
        "client": _telemetry_client(handler),
        "client_address": str(client_address[0] if client_address else ""),
        "stream": bool(getattr(handler, "stream", False)),
        "message_count": len(messages) if isinstance(messages, list) else 0,
        "input_characters": sum(len(text) for text in input_texts),
        "query_preview": query_preview,
        "max_output_tokens": int(getattr(handler, "max_tokens", 0) or 0),
        "prompt_tokens": 0,
        "cached_prompt_tokens": 0,
        "output_tokens": 0,
        "offered_tools": offered_tools,
        "offered_tool_count": len(offered_tools),
        "mcp_servers": sorted(
            {
                server
                for server in (mcp_server_for_tool(name) for name in offered_tools)
                if server
            }
        ),
        "tool_results": tool_results,
        "tool_calls": [],
        "tool_call_count": 0,
        "mcp_tool_call_count": 0,
    }


def _model_safety_profile(model_ref: Any) -> dict[str, Any]:
    profile = model_memory_profile(
        str(model_ref or ""),
        fallback_kv_bytes_per_token=_env_size(
            "MLX_SERVER_KV_BYTES_PER_TOKEN",
            "131072",
        ),
    )
    profile["model"] = _canonical_model_ref(model_ref)
    return profile


class SafetyController:
    POLICY_KEYS = {
        "enabled",
        "max_context_tokens",
        "max_output_tokens",
        "memory_reserve_bytes",
        "memory_abort_bytes",
    }

    def __init__(self) -> None:
        self.lock = threading.RLock()
        recommended_reserve = recommended_memory_reserve_bytes(
            int(psutil.virtual_memory().total)
        )
        memory_reserve = _env_size(
            "MLX_SERVER_MEMORY_RESERVE_BYTES",
            str(recommended_reserve),
        )
        memory_abort = _env_size(
            "MLX_SERVER_MEMORY_ABORT_BYTES",
            str(recommended_abort_threshold_bytes(memory_reserve)),
        )
        self.policy: dict[str, Any] = {
            "enabled": os.environ.get(
                "MLX_SERVER_MEMORY_GUARD", "1"
            ).strip().lower()
            in TRUTHY,
            "max_context_tokens": _env_int(
                "MLX_SERVER_MAX_CONTEXT_TOKENS", 128_000
            ),
            "max_output_tokens": _env_int(
                "MLX_SERVER_MAX_OUTPUT_TOKENS", 32_000
            ),
            "memory_reserve_bytes": memory_reserve,
            "memory_abort_bytes": memory_abort,
        }
        self.model_profile: dict[str, Any] = {
            "model": "",
            "model_context_tokens": 0,
            "kv_bytes_per_token": _env_size(
                "MLX_SERVER_KV_BYTES_PER_TOKEN", "131072"
            ),
        }
        self.requests: dict[str, dict[str, Any]] = {}
        self.request_ids: dict[int, str] = {}
        self.last_event: dict[str, Any] = {}

    def configure_model(self, model_ref: Any) -> None:
        with self.lock:
            self.model_profile = _model_safety_profile(model_ref)
            self.last_event = {
                "type": "model_profile",
                "message": "Safety profile updated for the loaded model.",
                "at": time.time(),
            }
        self.publish()

    def update_policy(self, updates: dict[str, Any]) -> dict[str, Any]:
        unknown = set(updates) - self.POLICY_KEYS
        if unknown:
            raise ValueError(f"Unknown safety setting(s): {', '.join(sorted(unknown))}")

        with self.lock:
            candidate = dict(self.policy)
            if "enabled" in updates:
                if not isinstance(updates["enabled"], bool):
                    raise ValueError("enabled must be a boolean.")
                candidate["enabled"] = updates["enabled"]

            for key in self.POLICY_KEYS - {"enabled"}:
                if key not in updates:
                    continue
                value = updates[key]
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError(f"{key} must be an integer.")
                candidate[key] = value

            if not 1_024 <= candidate["max_context_tokens"] <= 1_048_576:
                raise ValueError("max_context_tokens must be between 1024 and 1048576.")
            if not 1 <= candidate["max_output_tokens"] <= 131_072:
                raise ValueError("max_output_tokens must be between 1 and 131072.")
            if candidate["memory_reserve_bytes"] < 4 * GIB:
                raise ValueError("memory_reserve_bytes must be at least 4 GiB.")
            if candidate["memory_abort_bytes"] < 2 * GIB:
                raise ValueError("memory_abort_bytes must be at least 2 GiB.")
            if (
                candidate["memory_abort_bytes"]
                >= candidate["memory_reserve_bytes"]
            ):
                raise ValueError(
                    "memory_abort_bytes must be lower than memory_reserve_bytes."
                )

            self.policy = candidate
            self.last_event = {
                "type": "policy_updated",
                "message": "Safety policy updated from the control center.",
                "at": time.time(),
            }
            snapshot = self.snapshot()
        self.publish()
        return snapshot

    def register(
        self,
        request_id: str,
        request: Any,
        args: Any,
        response_queue: Any,
    ) -> None:
        now = time.time()
        with self.lock:
            self.request_ids[id(request)] = request_id
            self.requests[request_id] = {
                "id": request_id,
                "request": request,
                "response_queue": response_queue,
                "context": None,
                "model": str(args.model.model),
                "status": "queued",
                "started_at": now,
                "prompt_tokens": 0,
                "prompt_processed": 0,
                "cached_prompt_tokens": 0,
                "output_tokens": 0,
                "max_output_tokens": int(args.max_tokens),
                "estimated_kv_bytes": 0,
                "cancel_requested": False,
                "cancel_reason": "",
                "error": "",
            }
        self.publish()

    def _request_for(self, request: Any) -> dict[str, Any] | None:
        request_id = self.request_ids.get(id(request))
        return self.requests.get(request_id) if request_id else None

    def admit(self, request: Any, prompt_tokens: int, output_tokens: int) -> None:
        memory = psutil.virtual_memory()
        with self.lock:
            policy = dict(self.policy)
            profile = dict(self.model_profile)
            tracked = self._request_for(request)
            model_context = int(profile.get("model_context_tokens") or 0)
            configured_context = int(policy["max_context_tokens"])
            effective_context = (
                min(configured_context, model_context)
                if model_context
                else configured_context
            )
            total_tokens = prompt_tokens + output_tokens
            kv_bytes_per_token = int(profile["kv_bytes_per_token"])
            estimated_kv_bytes = total_tokens * kv_bytes_per_token
            available_budget = max(
                0, int(memory.available) - int(policy["memory_reserve_bytes"])
            )

            if tracked is not None:
                tracked.update(
                    status="admitting",
                    prompt_tokens=prompt_tokens,
                    max_output_tokens=output_tokens,
                    estimated_kv_bytes=estimated_kv_bytes,
                )

            rejection = ""
            if output_tokens > int(policy["max_output_tokens"]):
                rejection = (
                    f"Requested output cap {output_tokens:,} exceeds the server cap "
                    f"of {int(policy['max_output_tokens']):,} tokens."
                )
            elif total_tokens > effective_context:
                rejection = (
                    f"Prompt plus output reserve is {total_tokens:,} tokens, above "
                    f"the effective context cap of {effective_context:,}."
                )
            elif policy["enabled"] and estimated_kv_bytes > available_budget:
                rejection = (
                    f"Estimated active KV cache is "
                    f"{estimated_kv_bytes / GIB:.1f} GiB, but only "
                    f"{available_budget / GIB:.1f} GiB is available after the "
                    f"{int(policy['memory_reserve_bytes']) / GIB:.0f} GiB reserve."
                )

            if rejection:
                if tracked is not None:
                    tracked["status"] = "rejected"
                self.last_event = {
                    "type": "admission_rejected",
                    "message": rejection,
                    "at": time.time(),
                }
                raise ValueError(f"MLX safety admission rejected the request: {rejection}")

            if tracked is not None:
                tracked["status"] = "prefill"

    def attach_context(self, request_id: str, context: Any) -> None:
        response_queue = None
        with self.lock:
            tracked = self.requests.get(request_id)
            if tracked is None:
                return
            tracked["context"] = context
            runtime_model = str(self.model_profile.get("model") or "")
            if runtime_model:
                tracked["model"] = runtime_model
            tracked["cached_prompt_tokens"] = max(
                0,
                int(getattr(context, "prompt_cache_count", 0) or 0),
            )
            if tracked["cancel_requested"]:
                context.stop()
                response_queue = tracked["response_queue"]
        if response_queue is not None:
            response_queue.put(None)

    def update_progress(self, request_id: str, processed: int, total: int) -> None:
        with self.lock:
            tracked = self.requests.get(request_id)
            if tracked is None:
                return
            tracked["status"] = "prefill"
            tracked["prompt_processed"] = int(processed)
            tracked["prompt_tokens"] = int(total)
        self.publish()

    def generated_token(self, request_id: str) -> None:
        with self.lock:
            tracked = self.requests.get(request_id)
            if tracked is None:
                return
            tracked["status"] = "generating"
            tracked["prompt_processed"] = tracked["prompt_tokens"]
            tracked["output_tokens"] += 1

    def mark_error(self, request_id: str, error: Exception) -> None:
        with self.lock:
            tracked = self.requests.get(request_id)
            if tracked is not None:
                tracked["status"] = "rejected"
                tracked["error"] = str(error)
            self.last_event = {
                "type": "request_error",
                "message": str(error),
                "at": time.time(),
            }
        self.publish()

    def record_event(self, event_type: str, message: str, **details: Any) -> None:
        with self.lock:
            self.last_event = {
                "type": event_type,
                "message": message,
                "at": time.time(),
                **details,
            }
        self.publish()

    def finish(self, request_id: str) -> None:
        with self.lock:
            tracked = self.requests.pop(request_id, None)
            if tracked is None:
                return
            request = tracked.get("request")
            if request is not None:
                self.request_ids.pop(id(request), None)
            if not tracked["cancel_requested"] and tracked["status"] != "rejected":
                self.last_event = {
                    "type": "request_completed",
                    "message": f"Request {request_id} completed.",
                    "at": time.time(),
                }
        self.publish()

    def request_snapshot(self, request_id: str) -> dict[str, Any]:
        with self.lock:
            tracked = self.requests.get(request_id) or {}
            return {
                key: value
                for key, value in tracked.items()
                if key not in {"request", "response_queue", "context"}
            }

    def cancel(self, request_id: str | None, reason: str) -> list[str]:
        cancelled: list[str] = []
        queues: list[Any] = []
        with self.lock:
            targets = (
                [self.requests.get(request_id)]
                if request_id
                else list(self.requests.values())
            )
            for tracked in targets:
                if tracked is None or tracked["cancel_requested"]:
                    continue
                tracked["cancel_requested"] = True
                tracked["cancel_reason"] = reason
                tracked["status"] = "cancelling"
                context = tracked.get("context")
                if context is not None:
                    context.stop()
                    queues.append(tracked["response_queue"])
                cancelled.append(tracked["id"])
            if cancelled:
                self.last_event = {
                    "type": "request_cancelled",
                    "message": reason,
                    "request_ids": cancelled,
                    "at": time.time(),
                }

        # Unblock HTTP response writers after the MLX contexts have been stopped.
        for response_queue in queues:
            response_queue.put(None)
        if cancelled:
            self.publish()
        return cancelled

    def snapshot(self) -> dict[str, Any]:
        memory = psutil.virtual_memory()
        with self.lock:
            policy = dict(self.policy)
            profile = dict(self.model_profile)
            model_context = int(profile.get("model_context_tokens") or 0)
            effective_context = (
                min(int(policy["max_context_tokens"]), model_context)
                if model_context
                else int(policy["max_context_tokens"])
            )
            dynamic_token_budget = max(
                0,
                (
                    int(memory.available) - int(policy["memory_reserve_bytes"])
                )
                // max(1, int(profile["kv_bytes_per_token"])),
            )
            requests = []
            for tracked in self.requests.values():
                requests.append(
                    {
                        key: value
                        for key, value in tracked.items()
                        if key not in {"request", "response_queue", "context"}
                    }
                )
            return {
                "policy": policy,
                "profile": {
                    **profile,
                    "effective_context_tokens": effective_context,
                },
                "available_bytes": int(memory.available),
                "dynamic_token_budget": dynamic_token_budget,
                "active_requests": requests,
                "last_event": dict(self.last_event),
            }

    def publish(self) -> None:
        _write_runtime_state(safety=self.snapshot())


SAFETY = SafetyController()
TELEMETRY = TelemetryStore(
    max_history=_env_int("MLX_SERVER_TELEMETRY_MAX_RECORDS", 5_000)
)


def _runtime_state_path() -> Path:
    configured = os.environ.get("MLX_SERVER_RUNTIME_STATE", "")
    if configured:
        return Path(configured).expanduser()
    return user_paths().logs / "mlx-server-runtime.json"


def _write_runtime_state(**updates: Any) -> None:
    path = _runtime_state_path()
    with RUNTIME_STATE_LOCK:
        try:
            state: dict[str, Any] = {}
            if path.is_file():
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    state.update(loaded)
            state.update(updates)
            state["pid"] = os.getpid()
            state["updated_at"] = time.time()
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(
                json.dumps(state, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary.replace(path)
        except (OSError, ValueError, TypeError) as exc:
            logging.warning("Unable to update MLX runtime state: %s", exc)


def _canonical_model_ref(value: Any) -> str:
    text = str(value or "")
    if not text:
        return ""
    path = Path(text).expanduser()
    if path.exists():
        return str(path.resolve())
    return text


def _pin_model_enabled() -> bool:
    value = os.environ.get("MLX_SERVER_PIN_MODEL", "1")
    return value.strip().lower() in TRUTHY


def _cache_state(prompt_cache: Any) -> dict[str, Any]:
    state = {
        "sequences": len(prompt_cache),
        "bytes": prompt_cache.nbytes,
        "max_sequences": prompt_cache.max_size,
        "max_bytes": (
            None
            if prompt_cache.max_bytes >= (1 << 63)
            else prompt_cache.max_bytes
        ),
        "by_type": prompt_cache.stats_by_type(),
    }
    if hasattr(prompt_cache, "snapshot_entries"):
        state["entries"] = prompt_cache.snapshot_entries()
    return state


def _include_undownloaded() -> bool:
    value = os.environ.get("MLX_SERVER_ADVERTISE_UNDOWNLOADED", "")
    return value.strip().lower() in TRUTHY


def _model_ref(model: dict[str, Any]) -> str:
    if model_is_downloaded(model):
        return str(model_path(model))
    return model["repo_id"]


def _model_aliases() -> dict[str, str]:
    aliases: dict[str, str] = {}
    for key, model in configured_models():
        local = model_path(model)
        ref = _model_ref(model)
        candidates = {
            key,
            model["repo_id"],
            model["local_dir"],
            local.name,
            str(local),
            *model.get("aliases", []),
        }
        for candidate in candidates:
            aliases[candidate] = ref
    return aliases


def _advertised_models(created: int) -> list[dict[str, Any]]:
    include_undownloaded = _include_undownloaded()
    seen: set[str] = set()
    models: list[dict[str, Any]] = []

    for key, model in configured_models():
        if not include_undownloaded and not model_is_downloaded(model):
            continue

        model_id = model_label(key, model)
        if model_id in seen:
            continue
        seen.add(model_id)
        models.append({"id": model_id, "object": "model", "created": created})

    return models


def _patch_model_provider() -> None:
    if getattr(mlx_server.ModelProvider, "_mlx_lab_patched", False):
        return

    original_init = mlx_server.ModelProvider.__init__
    original_load = mlx_server.ModelProvider.load

    def patched_init(self, cli_args):
        global PROMPT_CACHE_MAX_BYTES
        original_init(self, cli_args)
        aliases = _model_aliases()
        self._model_map.update(aliases)
        self._draft_model_map.update(aliases)
        PROMPT_CACHE_MAX_BYTES = cli_args.prompt_cache_bytes
        _write_runtime_state(
            status="starting",
            startup_model=_canonical_model_ref(cli_args.model),
            runtime_model="",
            pinned_model=_pin_model_enabled(),
            prompt_cache={
                "sequences": 0,
                "bytes": 0,
                "max_sequences": cli_args.prompt_cache_size,
                "max_bytes": cli_args.prompt_cache_bytes,
                "by_type": {},
            },
            started_at=time.time(),
            last_error="",
        )

    def patched_load(self, model_path, adapter_path=None, draft_model_path=None):
        requested_model = self._model_map.get(model_path, model_path)
        startup_model = self._model_map.get("default_model")
        if (
            _pin_model_enabled()
            and _canonical_model_ref(requested_model)
            != _canonical_model_ref(startup_model)
        ):
            message = (
                f"MLX server is pinned to {startup_model}. "
                "Use scripts/mlx-server load <model> to switch safely."
            )
            _write_runtime_state(
                last_error=message,
                last_rejected_model=_canonical_model_ref(requested_model),
            )
            raise ValueError(message)

        result = original_load(self, model_path, adapter_path, draft_model_path)
        runtime_model = self.model_key[0] if self.model_key else requested_model
        _write_runtime_state(
            status="ready",
            runtime_model=_canonical_model_ref(runtime_model),
            loaded_at=time.time(),
            last_error="",
        )
        SAFETY.configure_model(runtime_model)
        return result

    mlx_server.ModelProvider.__init__ = patched_init
    mlx_server.ModelProvider.load = patched_load
    mlx_server.ModelProvider._mlx_lab_patched = True


def _patch_prompt_cache() -> None:
    if getattr(mlx_server.LRUPromptCache, "_mlx_lab_patched", False):
        return

    original_cache = mlx_server.LRUPromptCache

    class BoundedPromptCache(original_cache):
        _mlx_lab_patched = True

        def __init__(self, max_size: int = 10, max_bytes: int = 1 << 63):
            self._mlx_lab_lock = threading.RLock()
            configured_max = (
                PROMPT_CACHE_MAX_BYTES
                if PROMPT_CACHE_MAX_BYTES is not None
                else max_bytes
            )
            super().__init__(max_size=max_size, max_bytes=configured_max)
            _write_runtime_state(prompt_cache=_cache_state(self))

        def __len__(self) -> int:
            with self._mlx_lab_lock:
                return super().__len__()

        @property
        def nbytes(self) -> int:
            with self._mlx_lab_lock:
                return super().nbytes

        def fetch_nearest_cache(self, *args: Any, **kwargs: Any) -> Any:
            with self._mlx_lab_lock:
                return super().fetch_nearest_cache(*args, **kwargs)

        def insert_cache(self, *args: Any, **kwargs: Any) -> None:
            with self._mlx_lab_lock:
                super().insert_cache(*args, **kwargs)
                state = _cache_state(self)
            _write_runtime_state(prompt_cache=state)

        def trim_to(self, *args: Any, **kwargs: Any) -> None:
            with self._mlx_lab_lock:
                super().trim_to(*args, **kwargs)
                state = _cache_state(self)
            _write_runtime_state(prompt_cache=state)

        def stats_by_type(self) -> dict[str, Any]:
            with self._mlx_lab_lock:
                return super().stats_by_type()

        @staticmethod
        def _entry_id(model: Any, tokens: list[int]) -> str:
            value = hash((repr(model), tuple(tokens))) & ((1 << 64) - 1)
            return f"cache-{value:016x}"

        def snapshot_entries(self) -> list[dict[str, Any]]:
            entries: list[dict[str, Any]] = []
            with self._mlx_lab_lock:
                rank = 0
                for cache_type in self._lru._ordering:
                    for model, tokens in self._lru._lrus[cache_type]:
                        entry = self._trie.get(model, tokens)
                        model_ref = model[0] if isinstance(model, tuple) else model
                        entries.append(
                            {
                                "id": self._entry_id(model, tokens),
                                "model": _canonical_model_ref(model_ref),
                                "cache_type": cache_type,
                                "tokens": len(tokens),
                                "bytes": int(entry.nbytes),
                                "eviction_rank": rank,
                            }
                        )
                        rank += 1
            return entries

        def clear_entry(self, entry_id: str) -> bool:
            removed = False
            with self._mlx_lab_lock:
                for cache_type in self._lru._ordering:
                    for model, tokens in list(self._lru._lrus[cache_type]):
                        if self._entry_id(model, tokens) != entry_id:
                            continue
                        entry = self._trie.pop(model, tokens)
                        self._lru.remove(model, tokens)
                        self._n_bytes -= entry.nbytes
                        self._n_bytes_by_type[entry.cache_type] -= entry.nbytes
                        removed = True
                        break
                    if removed:
                        break
                state = _cache_state(self)
            _write_runtime_state(prompt_cache=state)
            return removed

        def clear_all(self) -> int:
            with self._mlx_lab_lock:
                removed = len(self)
                super().trim_to(n_sequences=0)
                state = _cache_state(self)
            _write_runtime_state(prompt_cache=state)
            return removed

    mlx_server.LRUPromptCache = BoundedPromptCache


REQUEST_LOCAL = threading.local()


def _patch_generation_safety() -> None:
    if getattr(mlx_server.ResponseGenerator, "_mlx_lab_safety_patched", False):
        return

    original_init = mlx_server.ResponseGenerator.__init__
    original_stop_and_join = mlx_server.ResponseGenerator.stop_and_join
    original_tokenize = mlx_server.ResponseGenerator._tokenize
    original_handle_completion = mlx_server.APIHandler.handle_completion
    original_generate_response = mlx_server.APIHandler.generate_response

    def monitor_memory(self) -> None:
        while not self._stop:
            time.sleep(0.5)
            with SAFETY.lock:
                enabled = bool(SAFETY.policy["enabled"])
                has_requests = bool(SAFETY.requests)
                reserve_bytes = int(SAFETY.policy["memory_reserve_bytes"])
                abort_bytes = int(SAFETY.policy["memory_abort_bytes"])
            if not enabled or not has_requests:
                continue

            available = int(psutil.virtual_memory().available)
            if available < reserve_bytes and len(self.prompt_cache):
                removed = self.prompt_cache.clear_all()
                if removed:
                    SAFETY.record_event(
                        "cache_auto_cleared",
                        (
                            f"Memory guard cleared {removed} retained cache "
                            f"entr{'y' if removed == 1 else 'ies'}."
                        ),
                        available_bytes=available,
                    )

            if available < abort_bytes:
                SAFETY.cancel(
                    None,
                    (
                        f"Memory guard cancelled active requests because available "
                        f"memory fell below {abort_bytes / GIB:.0f} GiB."
                    ),
                )

    def patched_init(self, model_provider, prompt_cache):
        original_init(self, model_provider, prompt_cache)
        self._mlx_lab_monitor_thread = threading.Thread(
            target=monitor_memory,
            args=(self,),
            daemon=True,
            name="mlx-memory-guard",
        )
        self._mlx_lab_monitor_thread.start()

    def patched_stop_and_join(self):
        original_stop_and_join(self)
        monitor = getattr(self, "_mlx_lab_monitor_thread", None)
        if monitor is not None:
            monitor.join(timeout=2)

    def patched_tokenize(self, tokenizer, request, args):
        prompt, segments, segment_types, initial_state = original_tokenize(
            self, tokenizer, request, args
        )
        SAFETY.admit(request, len(prompt), int(args.max_tokens))
        return prompt, segments, segment_types, initial_state

    def patched_generate(
        self,
        request,
        generation_args,
        progress_callback=None,
    ):
        response_queue = mlx_server.Queue()
        request_id = str(
            getattr(REQUEST_LOCAL, "request_id", "")
            or f"mlx-{time.time_ns()}"
        )
        SAFETY.register(request_id, request, generation_args, response_queue)
        self.requests.put((response_queue, request, generation_args))
        repetition_guard = (
            TokenRepetitionGuard(
                min_generated_tokens=max(
                    1,
                    _env_int("MLX_SERVER_REPETITION_MIN_TOKENS", 96),
                ),
                min_repeated_tokens=max(
                    1,
                    _env_int("MLX_SERVER_REPETITION_WINDOW_TOKENS", 64),
                ),
                min_repeats=max(
                    2,
                    _env_int("MLX_SERVER_REPETITION_MIN_REPEATS", 4),
                ),
                max_pattern_tokens=max(
                    1,
                    _env_int("MLX_SERVER_REPETITION_MAX_PATTERN_TOKENS", 128),
                ),
            )
            if os.environ.get("MLX_SERVER_REPETITION_GUARD", "1")
            .strip()
            .lower()
            in TRUTHY
            else None
        )

        def inner():
            while True:
                response = response_queue.get()
                if response is None:
                    break
                if isinstance(response, Exception):
                    SAFETY.mark_error(request_id, response)
                    raise response
                if isinstance(response, tuple):
                    SAFETY.update_progress(request_id, *response)
                    if progress_callback is not None:
                        progress_callback(*response)
                    continue
                SAFETY.generated_token(request_id)
                match = (
                    repetition_guard.observe(response.token)
                    if repetition_guard is not None
                    else None
                )
                if match is not None:
                    pattern_tokens, repeats = match
                    SAFETY.cancel(
                        request_id,
                        (
                            "Repetition guard stopped generation after an exact "
                            f"{pattern_tokens}-token pattern repeated {repeats} "
                            "times."
                        ),
                    )
                    yield replace(response, finish_reason="stop")
                    return
                yield response

        context = response_queue.get()
        if isinstance(context, Exception):
            SAFETY.mark_error(request_id, context)
            raise context
        SAFETY.attach_context(request_id, context)
        request_state = SAFETY.request_snapshot(request_id)
        TELEMETRY.update(
            request_id,
            model=str(
                request_state.get("model")
                or generation_args.model.model
            ),
            cached_prompt_tokens=max(
                0,
                int(getattr(context, "prompt_cache_count", 0) or 0),
            ),
        )
        return context, mlx_server._process_control_tokens(context, inner())

    def patched_generate_response(self, *args, **kwargs):
        response = original_generate_response(self, *args, **kwargs)
        request_id = str(getattr(self, "request_id", ""))
        if request_id:
            request_state = SAFETY.request_snapshot(request_id)
            cancel_reason = str(request_state.get("cancel_reason") or "")
            if cancel_reason.startswith("Repetition guard stopped"):
                response["mlx_stop_reason"] = "repetition"
                response["mlx_stop_message"] = cancel_reason
            TELEMETRY.record_response(request_id, response)
        return response

    def patched_handle_completion(self, request, stop_words):
        request_id = str(getattr(self, "request_id", "") or f"mlx-{time.time_ns()}")
        previous_request_id = getattr(REQUEST_LOCAL, "request_id", None)
        REQUEST_LOCAL.request_id = request_id
        TELEMETRY.start(_telemetry_request_record(self, request, request_id))
        request_error: Exception | None = None
        try:
            return original_handle_completion(self, request, stop_words)
        except Exception as exc:
            request_error = exc
            raise
        finally:
            request_state = SAFETY.request_snapshot(request_id)
            if request_error is not None and request_state.get("status") != "rejected":
                request_state["status"] = "rejected"
                request_state["error"] = str(request_error)
            TELEMETRY.finish(request_id, request_state=request_state)
            SAFETY.finish(request_id)
            if previous_request_id is None:
                try:
                    del REQUEST_LOCAL.request_id
                except AttributeError:
                    pass
            else:
                REQUEST_LOCAL.request_id = previous_request_id

    mlx_server.ResponseGenerator.__init__ = patched_init
    mlx_server.ResponseGenerator.stop_and_join = patched_stop_and_join
    mlx_server.ResponseGenerator._tokenize = patched_tokenize
    mlx_server.ResponseGenerator.generate = patched_generate
    mlx_server.ResponseGenerator._mlx_lab_safety_patched = True
    mlx_server.APIHandler.generate_response = patched_generate_response
    mlx_server.APIHandler.handle_completion = patched_handle_completion


def _patch_control_endpoints() -> None:
    if getattr(mlx_server.APIHandler, "_mlx_lab_control_patched", False):
        return

    original_do_get = mlx_server.APIHandler.do_GET
    original_do_post = mlx_server.APIHandler.do_POST

    def send_json(handler, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        handler._set_completion_headers(status)
        handler.send_header("Content-Length", str(len(body)))
        handler.send_header("Cache-Control", "no-store")
        handler.end_headers()
        handler.wfile.write(body)
        handler.wfile.flush()

    def read_body(handler) -> dict[str, Any]:
        content_length = int(handler.headers.get("Content-Length", "0"))
        body = json.loads(handler.rfile.read(content_length) or b"{}")
        if not isinstance(body, dict):
            raise ValueError("JSON body must be an object.")
        return body

    def patched_do_get(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/mlx-lab/telemetry":
            query = urllib.parse.parse_qs(parsed.query)
            try:
                window_seconds = int(query.get("window_seconds", ["86400"])[0])
                limit = int(query.get("limit", ["100"])[0])
                request_page = int(query.get("request_page", ["1"])[0])
                request_page_size = int(
                    query.get("request_page_size", [str(min(limit, 100))])[0]
                )
                if not 0 <= window_seconds <= 31_536_000:
                    raise ValueError(
                        "window_seconds must be between 0 and 31536000."
                    )
                if not 1 <= limit <= 500:
                    raise ValueError("limit must be between 1 and 500.")
                if not 1 <= request_page <= 1_000_000:
                    raise ValueError("request_page must be between 1 and 1000000.")
                if not 1 <= request_page_size <= 100:
                    raise ValueError("request_page_size must be between 1 and 100.")
            except (TypeError, ValueError) as exc:
                send_json(self, {"error": str(exc)}, status=400)
                return
            for request_state in SAFETY.snapshot().get("active_requests", []):
                TELEMETRY.update(
                    str(request_state.get("id") or ""),
                    status=str(request_state.get("status") or "active"),
                    model=str(request_state.get("model") or ""),
                    prompt_tokens=int(request_state.get("prompt_tokens") or 0),
                    cached_prompt_tokens=int(
                        request_state.get("cached_prompt_tokens") or 0
                    ),
                    output_tokens=int(request_state.get("output_tokens") or 0),
                )
            send_json(
                self,
                TELEMETRY.snapshot(
                    window_seconds=window_seconds,
                    limit=limit,
                    request_page=request_page,
                    request_page_size=request_page_size,
                ),
            )
            return
        if parsed.path == "/mlx-lab/status":
            send_json(
                self,
                {
                    "ok": True,
                    "authentication": {
                        "required": (
                            MODEL_API_ACCESS.get("authentication") == "api_key"
                        ),
                        "scheme": "bearer",
                        "shared_with_control_center": True,
                    },
                    "safety": SAFETY.snapshot(),
                    "telemetry": {
                        "active": TELEMETRY.active_snapshot(),
                    },
                    "prompt_cache": _cache_state(
                        self.response_generator.prompt_cache
                    ),
                },
            )
            return
        return original_do_get(self)

    def patched_do_post(self):
        parsed = urllib.parse.urlparse(self.path)
        if not parsed.path.startswith("/mlx-lab/"):
            return original_do_post(self)

        try:
            body = read_body(self)
            if parsed.path == "/mlx-lab/requests/cancel":
                request_id = str(body.get("request_id") or "") or None
                cancelled = SAFETY.cancel(
                    request_id,
                    "Request cancelled from the MLX Control Center.",
                )
                if request_id and not cancelled:
                    raise ValueError("The request is no longer active.")
                payload = {
                    "ok": True,
                    "cancelled_request_ids": cancelled,
                    "safety": SAFETY.snapshot(),
                }
            elif parsed.path == "/mlx-lab/safety":
                payload = {
                    "ok": True,
                    "safety": SAFETY.update_policy(body),
                }
            elif parsed.path == "/mlx-lab/cache/clear":
                prompt_cache = self.response_generator.prompt_cache
                entry_id = str(body.get("entry_id") or "")
                if entry_id:
                    if not prompt_cache.clear_entry(entry_id):
                        raise ValueError("The cache entry no longer exists.")
                    removed = 1
                else:
                    removed = prompt_cache.clear_all()
                SAFETY.record_event(
                    "cache_cleared",
                    (
                        f"Cleared {removed} retained cache "
                        f"entr{'y' if removed == 1 else 'ies'} from the control center."
                    ),
                    removed_entries=removed,
                )
                payload = {
                    "ok": True,
                    "removed_entries": removed,
                    "prompt_cache": _cache_state(prompt_cache),
                }
            else:
                send_json(self, {"error": "Not found"}, status=404)
                return
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            send_json(self, {"error": str(exc)}, status=400)
            return

        send_json(self, payload)

    mlx_server.APIHandler.do_GET = patched_do_get
    mlx_server.APIHandler.do_POST = patched_do_post
    mlx_server.APIHandler._mlx_lab_control_patched = True


def _patch_api_key_authentication(access: dict[str, Any]) -> None:
    if getattr(mlx_server.APIHandler, "_mlx_lab_auth_patched", False):
        return

    original_do_get = mlx_server.APIHandler.do_GET
    original_do_post = mlx_server.APIHandler.do_POST

    def supplied_api_key(handler) -> str:
        authorization = str(handler.headers.get("Authorization") or "")
        scheme, _, value = authorization.partition(" ")
        if scheme.casefold() == "bearer":
            return value.strip()
        return str(handler.headers.get("X-API-Key") or "").strip()

    def require_authentication(handler) -> bool:
        if valid_model_api_key(supplied_api_key(handler), access):
            return True
        payload = json.dumps(
            {"error": "A valid MLX Lab API key is required."}
        ).encode("utf-8")
        handler._set_completion_headers(401)
        handler.send_header(
            "WWW-Authenticate",
            'Bearer realm="MLX Model API"',
        )
        handler.send_header("Cache-Control", "no-store")
        handler.send_header("Content-Length", str(len(payload)))
        handler.end_headers()
        handler.wfile.write(payload)
        handler.wfile.flush()
        return False

    def authenticated_do_get(self):
        if require_authentication(self):
            return original_do_get(self)
        return None

    def authenticated_do_post(self):
        if require_authentication(self):
            return original_do_post(self)
        return None

    mlx_server.APIHandler.do_GET = authenticated_do_get
    mlx_server.APIHandler.do_POST = authenticated_do_post
    mlx_server.APIHandler._mlx_lab_auth_patched = True


def _patch_models_endpoint() -> None:
    if getattr(mlx_server.APIHandler, "_mlx_lab_patched", False):
        return

    original_handle_models_request = mlx_server.APIHandler.handle_models_request

    def patched_handle_models_request(self):
        model_list_mode = os.environ.get("MLX_SERVER_MODEL_LIST", "configured")
        if model_list_mode.strip().lower() in {"runtime", "mlx"}:
            return original_handle_models_request(self)

        created = getattr(self, "created", int(time.time()))
        models = _advertised_models(created)
        if _pin_model_enabled():
            startup_model = _canonical_model_ref(
                self.response_generator.model_provider.cli_args.model
            )
            aliases = _model_aliases()
            models = [
                model
                for model in models
                if _canonical_model_ref(aliases.get(model["id"], model["id"]))
                == startup_model
            ]

        parsed = urllib.parse.urlparse(self.path)
        parts = parsed.path.split("/")
        if len(parts) > 3:
            requested_id = urllib.parse.unquote("/".join(parts[3:]))
            aliases = _model_aliases()
            requested_ref = aliases.get(requested_id)
            models = [
                model
                for model in models
                if model["id"] == requested_id
                or (requested_ref is not None and aliases.get(model["id"]) == requested_ref)
            ]

        response = {"object": "list", "data": models}
        self._set_completion_headers(200)
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())
        self.wfile.flush()

    mlx_server.APIHandler.handle_models_request = patched_handle_models_request
    mlx_server.APIHandler._mlx_lab_patched = True


def main() -> None:
    global MODEL_API_ACCESS
    sync_user_store()
    MODEL_API_ACCESS = _model_api_access_snapshot()
    _write_runtime_state(
        model_api_authentication=MODEL_API_ACCESS["authentication"],
    )
    _patch_model_provider()
    _patch_prompt_cache()
    _patch_generation_safety()
    _patch_models_endpoint()
    _patch_control_endpoints()
    _patch_api_key_authentication(MODEL_API_ACCESS)
    mlx_server.main()


if __name__ == "__main__":
    main()
