from __future__ import annotations

import importlib
import queue
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from mlx_lab import control_plane


class ControlPlaneTests(unittest.TestCase):
    @staticmethod
    def load_safety_module():
        fake_mlx_lm = types.ModuleType("mlx_lm")

        def parse_size(value: str) -> int:
            text = value.strip().upper()
            multipliers = {
                "KB": 1_000,
                "MB": 1_000_000,
                "GB": 1_000_000_000,
            }
            for suffix, multiplier in multipliers.items():
                if text.endswith(suffix):
                    return int(float(text.removesuffix(suffix)) * multiplier)
            return int(text)

        fake_mlx_lm.server = SimpleNamespace(_parse_size=parse_size)
        with mock.patch.dict(sys.modules, {"mlx_lm": fake_mlx_lm}):
            sys.modules.pop("mlx_lab.mlx_server", None)
            return importlib.import_module("mlx_lab.mlx_server")

    def test_parse_mlx_server_current_includes_runtime_fields(self) -> None:
        output = "\n".join(
            [
                "HTTP ready: http://127.0.0.1:48080",
                "Managed PID: 123",
                "Listener PID(s) on port 48080: 123",
                "Loaded model: /models/runtime",
                "Startup model: /models/startup",
                "Prompt cache: 2 sequence(s), 4096 bytes",
                "Prompt cache limit: 8000000000 bytes",
                "Command: python -m mlx_lab.mlx_server",
            ]
        )

        status = control_plane.parse_mlx_server_current(output)

        self.assertTrue(status["http_ready"])
        self.assertEqual(status["managed_pid"], "123")
        self.assertEqual(status["listener_pids"], ["123"])
        self.assertEqual(status["loaded_model"], "/models/runtime")
        self.assertEqual(status["startup_model"], "/models/startup")
        self.assertEqual(status["prompt_cache"], "2 sequence(s), 4096 bytes")

    def test_read_log_activity_detects_active_prefill(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log_path = Path(temporary) / "mlx-server.log"
            log_path.write_text(
                "\n".join(
                    [
                        "2026-07-23 18:00:00 - INFO - Starting httpd at 127.0.0.1",
                        "2026-07-23 18:00:01 - INFO - Prompt Cache: 1 sequences, 1.25 GB",
                        "2026-07-23 18:00:02 - INFO - Prompt processing progress: 4096/8192",
                    ]
                ),
                encoding="utf-8",
            )
            with mock.patch.object(control_plane, "MLX_SERVER_LOG", log_path):
                activity = control_plane.read_log_activity()

        self.assertTrue(activity["active"])
        self.assertEqual(activity["prompt_current"], 4096)
        self.assertEqual(activity["prompt_total"], 8192)
        self.assertEqual(activity["prompt_percent"], 50.0)
        self.assertEqual(activity["cache_sequences"], 1)
        self.assertEqual(activity["cache_gb"], 1.25)

    def test_system_memory_payload_classifies_pressure(self) -> None:
        memory = SimpleNamespace(
            total=1000,
            available=100,
            free=50,
            active=500,
            inactive=100,
            wired=300,
            cached=25,
        )
        swap = SimpleNamespace(used=200, percent=20.0)
        with (
            mock.patch.object(control_plane.psutil, "virtual_memory", return_value=memory),
            mock.patch.object(control_plane.psutil, "swap_memory", return_value=swap),
            mock.patch.object(control_plane.os, "getloadavg", return_value=(1.0, 2.0, 3.0)),
        ):
            payload = control_plane.system_memory_payload(None)

        self.assertEqual(payload["pressure"], "critical")
        self.assertEqual(payload["used"], 900)
        self.assertEqual(payload["used_percent"], 90.0)
        self.assertEqual(payload["swap_used"], 200)

    def test_status_payload_prefers_runtime_model(self) -> None:
        command_result = {
            "returncode": 0,
            "timed_out": False,
            "stdout": "\n".join(
                [
                    "HTTP ready: http://127.0.0.1:48080",
                    "Managed PID: 321",
                    "Listener PID(s) on port 48080: 321",
                    "Loaded model: /models/startup",
                ]
            ),
            "stderr": "",
            "command": [],
        }
        runtime = {"pid": 321, "runtime_model": "/models/runtime"}
        with (
            mock.patch.object(
                control_plane,
                "run_mlx_server_command",
                return_value=command_result,
            ),
            mock.patch.object(control_plane, "read_runtime_state", return_value=runtime),
            mock.patch.object(
                control_plane,
                "configured_model_for_ref",
                return_value={"key": "runtime"},
            ),
            mock.patch.object(
                control_plane,
                "system_memory_payload",
                return_value={"pressure": "healthy"},
            ),
            mock.patch.object(
                control_plane,
                "read_log_activity",
                return_value={"active": False},
            ),
        ):
            payload = control_plane.mlx_server_status_payload()

        self.assertEqual(payload["status"]["loaded_model"], "/models/runtime")
        self.assertEqual(payload["model"]["key"], "runtime")

    def test_control_center_contains_operational_controls(self) -> None:
        html = control_plane.CONTROL_PLANE_HTML
        for element_id in (
            "loadedModelMetric",
            "memoryMetric",
            "cacheMetric",
            "requestPanel",
            "cancelRequestButton",
            "guardEnabled",
            "contextCap",
            "outputCap",
            "reserveGib",
            "abortGib",
            "cacheRows",
            "clearCacheButton",
            "loadServerButton",
            "restartServerButton",
            "unloadServerButton",
        ):
            self.assertIn(f'id="{element_id}"', html)

    def test_context_admission_rejects_combined_token_overflow(self) -> None:
        safety_module = self.load_safety_module()
        controller = safety_module.SafetyController()
        controller.publish = mock.Mock()
        controller.policy.update(
            max_context_tokens=10_000,
            max_output_tokens=4_000,
            memory_reserve_bytes=4 * safety_module.GIB,
            memory_abort_bytes=2 * safety_module.GIB,
        )
        controller.model_profile.update(
            model_context_tokens=20_000,
            kv_bytes_per_token=1_024,
        )
        request = object()
        args = SimpleNamespace(
            model=SimpleNamespace(model="test-model"),
            max_tokens=2_000,
        )
        controller.register("request-1", request, args, queue.Queue())

        memory = SimpleNamespace(available=100 * safety_module.GIB)
        with mock.patch.object(
            safety_module.psutil,
            "virtual_memory",
            return_value=memory,
        ):
            with self.assertRaisesRegex(ValueError, "effective context cap"):
                controller.admit(request, prompt_tokens=9_000, output_tokens=2_000)

    def test_memory_admission_preserves_configured_reserve(self) -> None:
        safety_module = self.load_safety_module()
        controller = safety_module.SafetyController()
        controller.publish = mock.Mock()
        controller.policy.update(
            max_context_tokens=100_000,
            max_output_tokens=10_000,
            memory_reserve_bytes=16 * safety_module.GIB,
            memory_abort_bytes=8 * safety_module.GIB,
        )
        controller.model_profile.update(
            model_context_tokens=100_000,
            kv_bytes_per_token=1 * safety_module.GIB,
        )
        request = object()
        args = SimpleNamespace(
            model=SimpleNamespace(model="test-model"),
            max_tokens=1,
        )
        controller.register("request-2", request, args, queue.Queue())

        memory = SimpleNamespace(available=16 * safety_module.GIB)
        with mock.patch.object(
            safety_module.psutil,
            "virtual_memory",
            return_value=memory,
        ):
            with self.assertRaisesRegex(ValueError, "available after"):
                controller.admit(request, prompt_tokens=1, output_tokens=1)

    def test_cancel_stops_context_and_unblocks_response(self) -> None:
        safety_module = self.load_safety_module()
        controller = safety_module.SafetyController()
        controller.publish = mock.Mock()
        request = object()
        response_queue: queue.Queue[object] = queue.Queue()
        args = SimpleNamespace(
            model=SimpleNamespace(model="test-model"),
            max_tokens=100,
        )
        context = SimpleNamespace(stop=mock.Mock())
        controller.register("request-3", request, args, response_queue)
        controller.attach_context("request-3", context)

        cancelled = controller.cancel("request-3", "test cancellation")

        self.assertEqual(cancelled, ["request-3"])
        context.stop.assert_called_once_with()
        self.assertIsNone(response_queue.get_nowait())

    def test_runtime_control_request_uses_whitelisted_internal_path(self) -> None:
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = b'{"ok": true}'
        with mock.patch.object(
            control_plane.urllib.request,
            "urlopen",
            return_value=response,
        ) as urlopen:
            result = control_plane.mlx_internal_request(
                "cancel",
                method="POST",
                payload={"request_id": "request-4"},
            )

        self.assertTrue(result["ok"])
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:48080/mlx-lab/requests/cancel")
        self.assertEqual(request.method, "POST")


if __name__ == "__main__":
    unittest.main()
