from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


GIB = 1 << 30
GB = 1_000_000_000
DEFAULT_KV_BYTES_PER_TOKEN = 131_072
DEFAULT_MEMORY_RESERVE_BYTES = 16 * GIB
MINIMUM_MEMORY_RESERVE_BYTES = 4 * GIB


def recommended_memory_reserve_bytes(total_memory_bytes: int) -> int:
    scaled_gib = math.ceil((total_memory_bytes / GIB) * 0.15)
    return min(
        DEFAULT_MEMORY_RESERVE_BYTES,
        max(MINIMUM_MEMORY_RESERVE_BYTES, scaled_gib * GIB),
    )


def recommended_abort_threshold_bytes(reserve_bytes: int) -> int:
    return max(2 * GIB, reserve_bytes // 2)


def _positive_int(value: Any) -> int:
    try:
        parsed = int(value or 0)
    except (TypeError, ValueError):
        return 0
    return max(0, parsed)


def read_model_config(model_ref: str | Path | None) -> dict[str, Any]:
    if not model_ref:
        return {}
    path = Path(model_ref).expanduser()
    config_path = path / "config.json" if path.is_dir() else path
    if config_path.name != "config.json" or not config_path.is_file():
        return {}
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return {}
    if not isinstance(config, dict):
        return {}
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        return {**config, **text_config}
    return config


def model_memory_profile(
    model_ref: str | Path | None,
    *,
    fallback_kv_bytes_per_token: int = DEFAULT_KV_BYTES_PER_TOKEN,
) -> dict[str, Any]:
    config = read_model_config(model_ref)
    layers = _positive_int(config.get("num_hidden_layers"))
    attention_heads = _positive_int(config.get("num_attention_heads"))
    kv_heads = _positive_int(
        config.get("num_key_value_heads") or attention_heads
    )
    head_dim = _positive_int(config.get("head_dim"))
    if not head_dim:
        hidden_size = _positive_int(config.get("hidden_size"))
        head_dim = hidden_size // attention_heads if attention_heads else 0

    configured_kv_bytes = (
        2 * layers * kv_heads * head_dim * 2
        if layers and kv_heads and head_dim
        else 0
    )
    kv_bytes_per_token = configured_kv_bytes or max(
        1, int(fallback_kv_bytes_per_token)
    )
    return {
        "model": str(model_ref or ""),
        "model_context_tokens": _positive_int(
            config.get("max_position_embeddings")
            or config.get("model_max_length")
        ),
        "kv_bytes_per_token": kv_bytes_per_token,
        "kv_estimate_source": (
            "model config" if configured_kv_bytes else "conservative fallback"
        ),
    }


def build_memory_plan(
    *,
    model_weight_bytes: int,
    download_bytes: int,
    total_memory_bytes: int,
    available_memory_bytes: int,
    disk_free_bytes: int,
    context_tokens: int,
    kv_bytes_per_token: int,
    reserve_bytes: int = DEFAULT_MEMORY_RESERVE_BYTES,
    model_context_tokens: int = 0,
) -> dict[str, Any]:
    if model_weight_bytes <= 0:
        raise ValueError("model_weight_bytes must be greater than zero")
    if context_tokens <= 0:
        raise ValueError("context_tokens must be greater than zero")
    if reserve_bytes < MINIMUM_MEMORY_RESERVE_BYTES:
        raise ValueError("reserve_bytes must be at least 4 GiB")

    runtime_overhead_bytes = max(
        2 * GIB,
        math.ceil(model_weight_bytes * 0.10),
    )
    kv_cache_bytes = context_tokens * max(1, kv_bytes_per_token)
    working_memory_bytes = (
        model_weight_bytes + runtime_overhead_bytes + kv_cache_bytes
    )
    recommended_memory_bytes = working_memory_bytes + reserve_bytes
    disk_buffer_bytes = (
        max(5 * GB, math.ceil(download_bytes * 0.10))
        if download_bytes > 0
        else 0
    )
    disk_required_bytes = download_bytes + disk_buffer_bytes

    context_fit = (
        not model_context_tokens or context_tokens <= model_context_tokens
    )
    capacity_fit = recommended_memory_bytes <= total_memory_bytes
    current_memory_fit = recommended_memory_bytes <= available_memory_bytes
    disk_fit = disk_required_bytes <= disk_free_bytes

    if not context_fit or not capacity_fit:
        verdict = "unsafe"
    elif not current_memory_fit or not disk_fit:
        verdict = "caution"
    else:
        verdict = "safe"

    return {
        "verdict": verdict,
        "model_weight_bytes": int(model_weight_bytes),
        "download_bytes": int(download_bytes),
        "runtime_overhead_bytes": int(runtime_overhead_bytes),
        "kv_cache_bytes": int(kv_cache_bytes),
        "working_memory_bytes": int(working_memory_bytes),
        "reserve_bytes": int(reserve_bytes),
        "recommended_memory_bytes": int(recommended_memory_bytes),
        "total_memory_bytes": int(total_memory_bytes),
        "available_memory_bytes": int(available_memory_bytes),
        "memory_headroom_bytes": int(
            total_memory_bytes - recommended_memory_bytes
        ),
        "available_headroom_bytes": int(
            available_memory_bytes - recommended_memory_bytes
        ),
        "disk_free_bytes": int(disk_free_bytes),
        "disk_buffer_bytes": int(disk_buffer_bytes),
        "disk_required_bytes": int(disk_required_bytes),
        "context_tokens": int(context_tokens),
        "model_context_tokens": int(model_context_tokens),
        "kv_bytes_per_token": int(kv_bytes_per_token),
        "context_fit": context_fit,
        "capacity_fit": capacity_fit,
        "current_memory_fit": current_memory_fit,
        "disk_fit": disk_fit,
    }
