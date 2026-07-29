from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mlx_lab import cli
from mlx_lab.model_sizing import (
    GB,
    GIB,
    build_memory_plan,
    model_memory_profile,
    recommended_memory_reserve_bytes,
)


class ModelSizingTests(unittest.TestCase):
    def test_adaptive_reserve_covers_common_mac_memory_tiers(self) -> None:
        expected = {
            24: 4,
            36: 6,
            48: 8,
            64: 10,
            128: 16,
        }
        for memory_gib, reserve_gib in expected.items():
            with self.subTest(memory_gib=memory_gib):
                self.assertEqual(
                    recommended_memory_reserve_bytes(memory_gib * GIB),
                    reserve_gib * GIB,
                )

    def test_profile_uses_text_model_config_for_kv_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "text_config": {
                            "num_hidden_layers": 32,
                            "num_attention_heads": 32,
                            "num_key_value_heads": 8,
                            "hidden_size": 4096,
                            "max_position_embeddings": 32_768,
                        }
                    }
                ),
                encoding="utf-8",
            )

            profile = model_memory_profile(config_path)

        self.assertEqual(profile["kv_bytes_per_token"], 131_072)
        self.assertEqual(profile["model_context_tokens"], 32_768)
        self.assertEqual(profile["kv_estimate_source"], "model config")

    def test_plan_reports_safe_caution_and_unsafe_envelopes(self) -> None:
        common = {
            "model_weight_bytes": 30 * GIB,
            "download_bytes": 32 * GB,
            "disk_free_bytes": 100 * GB,
            "context_tokens": 8_192,
            "kv_bytes_per_token": 131_072,
        }
        safe = build_memory_plan(
            **common,
            total_memory_bytes=64 * GIB,
            available_memory_bytes=55 * GIB,
            reserve_bytes=10 * GIB,
        )
        caution = build_memory_plan(
            **common,
            total_memory_bytes=64 * GIB,
            available_memory_bytes=40 * GIB,
            reserve_bytes=10 * GIB,
        )
        unsafe = build_memory_plan(
            **common,
            total_memory_bytes=36 * GIB,
            available_memory_bytes=36 * GIB,
            reserve_bytes=6 * GIB,
        )

        self.assertEqual(safe["verdict"], "safe")
        self.assertEqual(caution["verdict"], "caution")
        self.assertFalse(caution["current_memory_fit"])
        self.assertEqual(unsafe["verdict"], "unsafe")
        self.assertFalse(unsafe["capacity_fit"])

    def test_plan_rejects_context_beyond_model_limit(self) -> None:
        plan = build_memory_plan(
            model_weight_bytes=4 * GIB,
            download_bytes=5 * GB,
            total_memory_bytes=64 * GIB,
            available_memory_bytes=64 * GIB,
            disk_free_bytes=100 * GB,
            context_tokens=65_536,
            kv_bytes_per_token=32_768,
            reserve_bytes=10 * GIB,
            model_context_tokens=32_768,
        )

        self.assertEqual(plan["verdict"], "unsafe")
        self.assertFalse(plan["context_fit"])

    def test_downloaded_model_does_not_require_extra_disk_buffer(self) -> None:
        plan = build_memory_plan(
            model_weight_bytes=4 * GIB,
            download_bytes=0,
            total_memory_bytes=24 * GIB,
            available_memory_bytes=24 * GIB,
            disk_free_bytes=0,
            context_tokens=8_192,
            kv_bytes_per_token=32_768,
            reserve_bytes=4 * GIB,
        )

        self.assertEqual(plan["disk_required_bytes"], 0)
        self.assertTrue(plan["disk_fit"])

    def test_cli_accepts_model_planning_and_selected_download(self) -> None:
        parser = cli.build_parser()

        plan_args = parser.parse_args(
            [
                "plan-model",
                "qwen3-coder-30b-8bit",
                "--memory-gib",
                "36",
                "--catalog-only",
            ]
        )
        download_args = parser.parse_args(
            [
                "download",
                "--model",
                "qwen3-coder-30b-8bit",
                "--dry-run",
            ]
        )

        self.assertEqual(plan_args.memory_gib, 36.0)
        self.assertTrue(plan_args.catalog_only)
        self.assertEqual(download_args.model, "qwen3-coder-30b-8bit")
        self.assertTrue(download_args.dry_run)


if __name__ == "__main__":
    unittest.main()
