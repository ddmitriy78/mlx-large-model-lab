from __future__ import annotations

import json
import os
import platform
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil


APP_NAME = "mlx-lab"
USER_CONFIG_SCHEMA_VERSION = 1
CONTROL_CENTER_ACCESS_MODES = {"local", "network"}
CONTROL_CENTER_AUTHENTICATION_MODES = {"api_key", "none"}
MODEL_API_AUTHENTICATION_MODES = {"api_key", "none"}
DEFAULT_CONTROL_CENTER_PORT = 8765
CONTROL_CENTER_POLICY_VERSION = 2


def _source_checkout_root() -> Path | None:
    source_root = Path(__file__).resolve().parents[2]
    if (
        (source_root / "pyproject.toml").is_file()
        and (source_root / "configs" / "models.json").is_file()
    ):
        return source_root
    return None


@dataclass(frozen=True)
class UserPaths:
    root: Path
    config: Path
    credentials: Path
    models: Path
    cache: Path
    runs: Path
    logs: Path

    def create(self) -> None:
        for path in (self.root, self.models, self.cache, self.runs, self.logs):
            path.mkdir(parents=True, exist_ok=True)


def project_root() -> Path:
    override = os.environ.get("MLX_LAB_PROJECT_ROOT")
    if override:
        return Path(override).expanduser().resolve()

    source_root = _source_checkout_root()
    if source_root is not None:
        return source_root
    return Path.cwd().resolve()


def capabilities_path() -> Path:
    override = os.environ.get("MLX_LAB_CAPABILITIES_PATH")
    if override:
        return Path(override).expanduser().resolve()

    project_override = os.environ.get("MLX_LAB_PROJECT_ROOT")
    if project_override:
        project_path = (
            Path(project_override).expanduser().resolve()
            / "configs"
            / "models.json"
        )
        if project_path.is_file():
            return project_path

    source_root = _source_checkout_root()
    if source_root is not None:
        return source_root / "configs" / "models.json"

    packaged_path = Path(__file__).resolve().parent / "data" / "model_capabilities.json"
    if packaged_path.is_file():
        return packaged_path

    raise FileNotFoundError(
        "The MLX Lab model capability catalog is missing. Reinstall the package "
        "or set MLX_LAB_CAPABILITIES_PATH."
    )


def user_paths() -> UserPaths:
    override = os.environ.get("MLX_LAB_HOME")
    if override:
        root = Path(override).expanduser().resolve()
    elif platform.system() == "Darwin":
        root = Path.home() / "Library" / "Application Support" / APP_NAME
    elif platform.system() == "Windows":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        root = root / APP_NAME
    else:
        data_home = Path(
            os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")
        )
        root = data_home / APP_NAME

    config_override = os.environ.get("MLX_LAB_USER_CONFIG")
    config = (
        Path(config_override).expanduser().resolve()
        if config_override
        else root / "user-config.json"
    )
    return UserPaths(
        root=root,
        config=config,
        credentials=root / "control-center-credentials.json",
        models=root / "models",
        cache=root / "cache",
        runs=root / "runs",
        logs=root / "logs",
    )


def load_capabilities() -> dict[str, Any]:
    path = capabilities_path()
    with path.open("r", encoding="utf-8") as file:
        capabilities = json.load(file)

    models = capabilities.get("models")
    first_model = capabilities.get("first_model")
    if not isinstance(models, dict) or not models:
        raise ValueError(f"Capability catalog has no models: {path}")
    if first_model not in models:
        raise ValueError(f"Capability catalog has an invalid first_model: {path}")
    for key, model in models.items():
        if not isinstance(model, dict) or not model.get("repo_id"):
            raise ValueError(f"Capability {key!r} is missing repo_id: {path}")
        if "local_dir" in model:
            raise ValueError(
                f"Capability {key!r} contains local_dir. Local model paths belong "
                "in the user store."
            )
    return capabilities


def hardware_snapshot() -> dict[str, Any]:
    memory = psutil.virtual_memory()
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "memory_bytes": int(memory.total),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
    }


def _default_user_config(capabilities: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": USER_CONFIG_SCHEMA_VERSION,
        "hardware": hardware_snapshot(),
        "preferences": {"default_model": capabilities["first_model"]},
        "control_center": {
            "policy_version": CONTROL_CENTER_POLICY_VERSION,
            "access_mode": "local",
            "port": DEFAULT_CONTROL_CENTER_PORT,
            "authentication": "none",
            "model_api_authentication": "none",
        },
        "models": {},
    }


def _normalized_control_center_settings(
    settings: Any,
) -> dict[str, Any]:
    source = settings if isinstance(settings, dict) else {}
    access_mode = str(source.get("access_mode") or "local")
    if access_mode not in CONTROL_CENTER_ACCESS_MODES:
        access_mode = "local"

    try:
        policy_version = int(source.get("policy_version", 0))
    except (TypeError, ValueError):
        policy_version = 0
    authentication = str(source.get("authentication") or "none")
    if authentication not in CONTROL_CENTER_AUTHENTICATION_MODES:
        authentication = "none"
    if policy_version < CONTROL_CENTER_POLICY_VERSION and access_mode == "local":
        authentication = "none"
    if access_mode == "network":
        authentication = "api_key"

    model_api_authentication = str(
        source.get("model_api_authentication") or "none"
    )
    if model_api_authentication not in MODEL_API_AUTHENTICATION_MODES:
        model_api_authentication = "none"

    try:
        port = int(source.get("port", DEFAULT_CONTROL_CENTER_PORT))
    except (TypeError, ValueError):
        port = DEFAULT_CONTROL_CENTER_PORT
    if not 1 <= port <= 65_535:
        port = DEFAULT_CONTROL_CENTER_PORT

    return {
        "policy_version": CONTROL_CENTER_POLICY_VERSION,
        "access_mode": access_mode,
        "bind_host": "127.0.0.1" if access_mode == "local" else "0.0.0.0",
        "port": port,
        "authentication": authentication,
        "model_api_authentication": model_api_authentication,
    }


def load_user_config(
    capabilities: dict[str, Any] | None = None,
) -> dict[str, Any]:
    catalog = capabilities or load_capabilities()
    path = user_paths().config
    if not path.is_file():
        return _default_user_config(catalog)

    with path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    if not isinstance(config, dict):
        raise ValueError(f"User configuration must be a JSON object: {path}")
    if config.get("schema_version") != USER_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported user configuration schema at {path}. Expected "
            f"{USER_CONFIG_SCHEMA_VERSION}."
        )

    for field in ("hardware", "preferences", "models"):
        if not isinstance(config.get(field), dict):
            config[field] = {}
    return config


def model_directory_name(model: dict[str, Any]) -> str:
    return str(model["repo_id"]).rstrip("/").rsplit("/", 1)[-1]


def model_is_present(path: Path) -> bool:
    return (path / "config.json").is_file() and any(path.glob("*.safetensors"))


def _case_insensitive_child(parent: Path, name: str) -> Path | None:
    if not parent.is_dir():
        return None
    expected = name.casefold()
    try:
        for child in parent.iterdir():
            if child.name.casefold() == expected:
                return child
    except OSError:
        return None
    return None


def configured_model_path(
    key: str,
    model: dict[str, Any],
    user_config: dict[str, Any] | None = None,
) -> Path:
    paths = user_paths()
    config = user_config if user_config is not None else load_user_config()
    configured = config.get("models", {}).get(key, {})
    if isinstance(configured, dict) and configured.get("local_dir"):
        return Path(str(configured["local_dir"])).expanduser()

    directory_name = model_directory_name(model)
    managed_path = paths.models / directory_name
    if model_is_present(managed_path):
        return managed_path

    legacy_root = (
        project_root()
        if os.environ.get("MLX_LAB_PROJECT_ROOT")
        else _source_checkout_root()
    )
    if legacy_root is not None:
        legacy_path = _case_insensitive_child(
            legacy_root / "models",
            directory_name,
        )
        if legacy_path is not None and model_is_present(legacy_path):
            return legacy_path
    return managed_path


def merged_config() -> dict[str, Any]:
    capabilities = load_capabilities()
    user_config = load_user_config(capabilities)
    merged_models: dict[str, dict[str, Any]] = {}
    for key, capability in capabilities["models"].items():
        model = dict(capability)
        model["local_dir"] = str(
            configured_model_path(key, capability, user_config).resolve()
        )
        merged_models[key] = model

    preferences = user_config.get("preferences", {})
    default_model = preferences.get("default_model")
    if default_model not in merged_models:
        default_model = capabilities["first_model"]
    return {
        **capabilities,
        "first_model": default_model,
        "models": merged_models,
    }


def _write_private_json(path: Path, data: dict[str, Any]) -> None:
    paths = user_paths()
    paths.create()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    )
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, sort_keys=True)
            file.write("\n")
        temporary.replace(path)
        path.chmod(0o600)
    finally:
        temporary.unlink(missing_ok=True)


def _write_user_config(config: dict[str, Any]) -> None:
    _write_private_json(user_paths().config, config)


def _load_or_create_control_center_api_key() -> str:
    path = user_paths().credentials
    if path.is_file():
        with path.open("r", encoding="utf-8") as file:
            credentials = json.load(file)
        api_key = str(credentials.get("api_key") or "")
        if len(api_key) >= 24:
            return api_key

    api_key = secrets.token_urlsafe(32)
    _write_private_json(path, {"api_key": api_key})
    return api_key


def control_center_settings(*, include_secret: bool = False) -> dict[str, Any]:
    config = load_user_config()
    settings = _normalized_control_center_settings(config.get("control_center"))
    if include_secret:
        settings["api_key"] = _load_or_create_control_center_api_key()
    else:
        settings["api_key_configured"] = user_paths().credentials.is_file()
    return settings


def update_control_center_settings(updates: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "access_mode",
        "port",
        "authentication",
        "model_api_authentication",
        "api_key",
    }
    unknown = set(updates) - allowed
    if unknown:
        raise ValueError(
            f"Unknown Control Center setting(s): {', '.join(sorted(unknown))}"
        )

    config = sync_user_store()
    current = _normalized_control_center_settings(config.get("control_center"))
    candidate = {
        "access_mode": updates.get("access_mode", current["access_mode"]),
        "port": updates.get("port", current["port"]),
        "authentication": updates.get(
            "authentication",
            current["authentication"],
        ),
        "model_api_authentication": updates.get(
            "model_api_authentication",
            current["model_api_authentication"],
        ),
    }
    access_mode = str(candidate["access_mode"])
    if access_mode not in CONTROL_CENTER_ACCESS_MODES:
        raise ValueError("access_mode must be local or network.")
    authentication = str(candidate["authentication"])
    if authentication not in CONTROL_CENTER_AUTHENTICATION_MODES:
        raise ValueError("authentication must be api_key or none.")
    if access_mode == "network" and authentication != "api_key":
        raise ValueError("API key authentication is required for network access.")
    model_api_authentication = str(candidate["model_api_authentication"])
    if model_api_authentication not in MODEL_API_AUTHENTICATION_MODES:
        raise ValueError("model_api_authentication must be api_key or none.")
    if isinstance(candidate["port"], bool):
        raise ValueError("port must be an integer between 1 and 65535.")
    try:
        port = int(candidate["port"])
    except (TypeError, ValueError) as exc:
        raise ValueError("port must be an integer between 1 and 65535.") from exc
    if not 1 <= port <= 65_535:
        raise ValueError("port must be an integer between 1 and 65535.")
    api_key = None
    if "api_key" in updates:
        api_key = str(updates["api_key"]).strip()
        if len(api_key) < 24:
            raise ValueError("API keys must contain at least 24 characters.")

    config["control_center"] = {
        "policy_version": CONTROL_CENTER_POLICY_VERSION,
        "access_mode": access_mode,
        "port": port,
        "authentication": authentication,
        "model_api_authentication": model_api_authentication,
    }
    config["updated_at"] = int(time.time())
    _write_user_config(config)

    if api_key is not None:
        _write_private_json(user_paths().credentials, {"api_key": api_key})
    elif (
        authentication == "api_key"
        or model_api_authentication == "api_key"
    ):
        _load_or_create_control_center_api_key()

    return control_center_settings()


def sync_user_store() -> dict[str, Any]:
    capabilities = load_capabilities()
    config = load_user_config(capabilities)
    config["schema_version"] = USER_CONFIG_SCHEMA_VERSION
    config["hardware"] = hardware_snapshot()
    config.setdefault("preferences", {})
    if config["preferences"].get("default_model") not in capabilities["models"]:
        config["preferences"]["default_model"] = capabilities["first_model"]
    control_center = _normalized_control_center_settings(
        config.get("control_center")
    )
    config["control_center"] = {
        "policy_version": CONTROL_CENTER_POLICY_VERSION,
        "access_mode": control_center["access_mode"],
        "port": control_center["port"],
        "authentication": control_center["authentication"],
        "model_api_authentication": control_center[
            "model_api_authentication"
        ],
    }
    if (
        control_center["authentication"] == "api_key"
        or control_center["model_api_authentication"] == "api_key"
    ):
        _load_or_create_control_center_api_key()

    configured_models = config.setdefault("models", {})
    for key, capability in capabilities["models"].items():
        path = configured_model_path(key, capability, config)
        existing = configured_models.get(key)
        if model_is_present(path) or (
            isinstance(existing, dict) and existing.get("local_dir")
        ):
            configured_models[key] = {"local_dir": str(path.resolve())}

    config["updated_at"] = int(time.time())
    _write_user_config(config)
    return config


def register_model_path(key: str, path: Path) -> dict[str, Any]:
    capabilities = load_capabilities()
    if key not in capabilities["models"]:
        raise KeyError(f"Unknown model capability: {key}")
    resolved = path.expanduser().resolve()
    if not model_is_present(resolved):
        raise ValueError(
            f"No complete MLX model found at {resolved}; expected config.json and "
            "*.safetensors files."
        )
    config = sync_user_store()
    config.setdefault("models", {})[key] = {
        "local_dir": str(resolved)
    }
    config["updated_at"] = int(time.time())
    _write_user_config(config)
    return config
