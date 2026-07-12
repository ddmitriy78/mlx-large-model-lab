from __future__ import annotations

import json
import os
import time
import urllib.parse
from typing import Any

from mlx_lm import server as mlx_server

from .cli import configured_models, model_is_downloaded, model_label, model_path


TRUTHY = {"1", "true", "yes", "on", "all"}


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

    def patched_init(self, cli_args):
        original_init(self, cli_args)
        aliases = _model_aliases()
        self._model_map.update(aliases)
        self._draft_model_map.update(aliases)

    mlx_server.ModelProvider.__init__ = patched_init
    mlx_server.ModelProvider._mlx_lab_patched = True


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
    _patch_model_provider()
    _patch_models_endpoint()
    mlx_server.main()


if __name__ == "__main__":
    main()
