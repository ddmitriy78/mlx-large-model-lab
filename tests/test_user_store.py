from __future__ import annotations

import json
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mlx_lab import user_store


class UserStoreTests(unittest.TestCase):
    def write_capabilities(self, root: Path, *, local_dir: bool = False) -> Path:
        model = {
            "repo_id": "example/Community-Model-4bit",
            "aliases": ["community-model"],
            "estimated_size_gb": 4.0,
            "role": "Test model",
        }
        if local_dir:
            model["local_dir"] = "models/Community-Model-4bit"
        path = root / "capabilities.json"
        path.write_text(
            json.dumps(
                {
                    "first_model": "community_model",
                    "models": {"community_model": model},
                    "benchmark_sets": {},
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_sync_records_hardware_and_discovers_legacy_download(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "project"
            model_path = project / "models" / "community-model-4BIT"
            model_path.mkdir(parents=True)
            (model_path / "config.json").write_text("{}", encoding="utf-8")
            (model_path / "weights.safetensors").write_bytes(b"test")
            capabilities = self.write_capabilities(root)
            home = root / "user-store"

            with mock.patch.dict(
                "os.environ",
                {
                    "MLX_LAB_HOME": str(home),
                    "MLX_LAB_PROJECT_ROOT": str(project),
                    "MLX_LAB_CAPABILITIES_PATH": str(capabilities),
                },
                clear=False,
            ):
                config = user_store.sync_user_store()
                merged = user_store.merged_config()

            self.assertEqual(config["schema_version"], 1)
            self.assertGreater(config["hardware"]["memory_bytes"], 0)
            self.assertEqual(
                config["models"]["community_model"]["local_dir"],
                str(model_path.resolve()),
            )
            self.assertEqual(
                merged["models"]["community_model"]["local_dir"],
                str(model_path.resolve()),
            )
            self.assertTrue((home / "user-config.json").is_file())
            self.assertEqual(
                stat.S_IMODE((home / "user-config.json").stat().st_mode),
                0o600,
            )
            self.assertTrue((home / "models").is_dir())
            self.assertTrue((home / "runs").is_dir())
            self.assertTrue((home / "logs").is_dir())

    def test_new_model_defaults_to_user_store_not_project(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "project"
            project.mkdir()
            capabilities = self.write_capabilities(root)
            home = root / "user-store"

            with mock.patch.dict(
                "os.environ",
                {
                    "MLX_LAB_HOME": str(home),
                    "MLX_LAB_PROJECT_ROOT": str(project),
                    "MLX_LAB_CAPABILITIES_PATH": str(capabilities),
                },
                clear=False,
            ):
                merged = user_store.merged_config()

            self.assertEqual(
                merged["models"]["community_model"]["local_dir"],
                str((home / "models" / "Community-Model-4bit").resolve()),
            )
            self.assertFalse((home / "user-config.json").exists())

    def test_capability_catalog_rejects_user_local_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capabilities = self.write_capabilities(root, local_dir=True)
            with mock.patch.dict(
                "os.environ",
                {"MLX_LAB_CAPABILITIES_PATH": str(capabilities)},
                clear=False,
            ):
                with self.assertRaisesRegex(ValueError, "user store"):
                    user_store.load_capabilities()

    def test_project_catalog_includes_specialized_openclaw_classifier(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        capabilities = project_root / "configs" / "models.json"

        with mock.patch.dict(
            "os.environ",
            {"MLX_LAB_CAPABILITIES_PATH": str(capabilities)},
            clear=False,
        ):
            catalog = user_store.load_capabilities()

        model = catalog["models"]["openclaw_qwen3_5_2b_classifier"]
        self.assertEqual(
            model["repo_id"],
            "NeelkanthSingh/openclaw-qwen3.5-2b-classifier",
        )
        self.assertIn("openclaw-classifier", model["aliases"])
        self.assertEqual(model["estimated_size_gb"], 1.1)
        self.assertIn("routing classifier", model["role"])

    def test_control_center_defaults_to_local_access_without_authentication(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "project"
            project.mkdir()
            capabilities = self.write_capabilities(root)
            home = root / "user-store"

            with mock.patch.dict(
                "os.environ",
                {
                    "MLX_LAB_HOME": str(home),
                    "MLX_LAB_PROJECT_ROOT": str(project),
                    "MLX_LAB_CAPABILITIES_PATH": str(capabilities),
                },
                clear=False,
            ):
                config = user_store.sync_user_store()
                settings = user_store.control_center_settings()

            self.assertEqual(
                config["control_center"],
                {
                    "policy_version": 2,
                    "access_mode": "local",
                    "port": 8765,
                    "authentication": "none",
                    "model_api_authentication": "none",
                },
            )
            self.assertNotIn("api_key", config["control_center"])
            self.assertEqual(settings["bind_host"], "127.0.0.1")
            self.assertEqual(settings["authentication"], "none")
            self.assertEqual(settings["model_api_authentication"], "none")
            self.assertFalse(settings["api_key_configured"])
            credential_path = home / "control-center-credentials.json"
            self.assertFalse(credential_path.exists())

    def test_network_access_requires_api_key_authentication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "project"
            project.mkdir()
            capabilities = self.write_capabilities(root)
            home = root / "user-store"

            with mock.patch.dict(
                "os.environ",
                {
                    "MLX_LAB_HOME": str(home),
                    "MLX_LAB_PROJECT_ROOT": str(project),
                    "MLX_LAB_CAPABILITIES_PATH": str(capabilities),
                },
                clear=False,
            ):
                user_store.sync_user_store()
                with self.assertRaisesRegex(ValueError, "required"):
                    user_store.update_control_center_settings(
                        {
                            "access_mode": "network",
                            "authentication": "none",
                        }
                    )
                updated = user_store.update_control_center_settings(
                    {
                        "access_mode": "network",
                        "port": 9876,
                        "authentication": "api_key",
                        "api_key": "a-secure-replacement-key-123",
                    }
                )
                secret = user_store.control_center_settings(include_secret=True)

            self.assertEqual(updated["bind_host"], "0.0.0.0")
            self.assertEqual(updated["port"], 9876)
            self.assertEqual(secret["api_key"], "a-secure-replacement-key-123")
            credential_path = home / "control-center-credentials.json"
            self.assertEqual(
                stat.S_IMODE(credential_path.stat().st_mode),
                0o600,
            )

    def test_model_api_uses_the_shared_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "project"
            project.mkdir()
            capabilities = self.write_capabilities(root)
            home = root / "user-store"

            with mock.patch.dict(
                "os.environ",
                {
                    "MLX_LAB_HOME": str(home),
                    "MLX_LAB_PROJECT_ROOT": str(project),
                    "MLX_LAB_CAPABILITIES_PATH": str(capabilities),
                },
                clear=False,
            ):
                updated = user_store.update_control_center_settings(
                    {
                        "model_api_authentication": "api_key",
                        "api_key": "one-shared-security-key-123",
                    }
                )
                secret = user_store.control_center_settings(include_secret=True)

            self.assertEqual(updated["authentication"], "none")
            self.assertEqual(updated["model_api_authentication"], "api_key")
            self.assertEqual(secret["api_key"], "one-shared-security-key-123")

    def test_legacy_local_api_key_default_migrates_to_no_authentication(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "project"
            project.mkdir()
            capabilities = self.write_capabilities(root)
            home = root / "user-store"
            home.mkdir()
            (home / "user-config.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "hardware": {},
                        "preferences": {"default_model": "community_model"},
                        "control_center": {
                            "access_mode": "local",
                            "port": 8765,
                            "authentication": "api_key",
                        },
                        "models": {},
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch.dict(
                "os.environ",
                {
                    "MLX_LAB_HOME": str(home),
                    "MLX_LAB_PROJECT_ROOT": str(project),
                    "MLX_LAB_CAPABILITIES_PATH": str(capabilities),
                },
                clear=False,
            ):
                config = user_store.sync_user_store()

            self.assertEqual(config["control_center"]["policy_version"], 2)
            self.assertEqual(config["control_center"]["authentication"], "none")


if __name__ == "__main__":
    unittest.main()
