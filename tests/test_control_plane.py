from __future__ import annotations

import importlib
import io
import json
import queue
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from mlx_lab import control_plane
from mlx_lab.telemetry import TelemetryStore, aggregate_telemetry


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

    def test_benchmark_progress_tracks_current_model_and_prompt(self) -> None:
        progress = control_plane.benchmark_progress(
            [
                "Benchmark models: 3",
                "Prompts: 4",
                "Benchmark model: Qwen Coder",
                "Load time: 1.25s",
                "Prompt: repository-review (coding)",
                "Prompt: test-generation (quality)",
            ],
            running=True,
            started_at=100.0,
            finished_at=None,
            returncode=None,
        )

        self.assertEqual(progress["state"], "running")
        self.assertEqual(progress["stage"], "Running prompt")
        self.assertEqual(progress["model_current"], 1)
        self.assertEqual(progress["model_total"], 3)
        self.assertEqual(progress["prompt_current"], 2)
        self.assertEqual(progress["prompt_total"], 4)
        self.assertEqual(progress["current_model"], "Qwen Coder")
        self.assertEqual(progress["current_prompt"], "test-generation")
        self.assertEqual(progress["current_category"], "quality")
        self.assertEqual(progress["completed_steps"], 1)
        self.assertEqual(progress["total_steps"], 12)
        self.assertEqual(progress["percent"], 8.3)

    def test_benchmark_progress_marks_successful_job_complete(self) -> None:
        progress = control_plane.benchmark_progress(
            [
                "Benchmark models: 1",
                "Prompts: 2",
                "Benchmark model: Qwen Coder",
                "Prompt: repository-review (coding)",
                "Prompt: test-generation (quality)",
                "HTML report written: runs/example.html",
            ],
            running=False,
            started_at=100.0,
            finished_at=112.5,
            returncode=0,
        )

        self.assertEqual(progress["state"], "complete")
        self.assertEqual(progress["stage"], "Benchmark complete")
        self.assertEqual(progress["completed_steps"], 2)
        self.assertEqual(progress["percent"], 100.0)
        self.assertEqual(progress["elapsed_seconds"], 12.5)

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

    def test_offline_model_server_status_is_a_normal_response(self) -> None:
        payload = {"ok": False, "status": {"http_ready": False}}
        handler = SimpleNamespace(
            path="/api/mlx-server/status",
            require_authentication=lambda: True,
            send_json=mock.Mock(),
        )
        with mock.patch.object(
            control_plane,
            "mlx_server_status_payload",
            return_value=payload,
        ):
            control_plane.ControlPlaneHandler.do_GET(handler)

        handler.send_json.assert_called_once_with(payload)

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
            "operationsTabButton",
            "telemetryTabButton",
            "benchmarkTabButton",
            "harnessTabButton",
            "operationsTab",
            "telemetryTab",
            "telemetryLoopAlerts",
            "benchmarkTab",
            "benchmarkProgressStage",
            "benchmarkProgressStatus",
            "benchmarkProgressPercent",
            "benchmarkProgressSteps",
            "benchmarkProgressTrack",
            "benchmarkProgressFill",
            "benchmarkProgressModelCount",
            "benchmarkProgressModel",
            "benchmarkProgressPromptCount",
            "benchmarkProgressPrompt",
            "benchmarkProgressElapsed",
            "benchmarkProgressActivity",
            "consoleLineCount",
            "followConsoleOutput",
            "logOutput",
            "harnessTab",
            "telemetryRequests",
            "telemetryInputTokens",
            "telemetryOutputTokens",
            "telemetryToolCalls",
            "telemetryMcpCalls",
            "telemetryChart",
            "telemetryFailurePanel",
            "telemetryFailureCount",
            "telemetryFailureSummary",
            "telemetryFailureRows",
            "telemetryFailureClearFilter",
            "telemetryModelRows",
            "telemetryModelToggle",
            "telemetryToolRows",
            "telemetryToolToggle",
            "telemetryRequestRows",
            "telemetryRetentionPolicy",
            "telemetryQueryCount",
            "telemetryQueryPageSize",
            "telemetryQueryPrevious",
            "telemetryQueryPageStatus",
            "telemetryQueryNext",
            "telemetryWindow",
            "refreshTelemetryButton",
            "statisticsRows",
            "statisticsSpeedBars",
            "refreshStatisticsButton",
            "harnessTranscript",
            "harnessSystemPrompt",
            "harnessInput",
            "harnessSendButton",
            "harnessCancelButton",
            "harnessClearButton",
            "settingsTabButton",
            "settingsTab",
            "accessSettingsForm",
            "localAccessMode",
            "networkAccessMode",
            "controlCenterPort",
            "apiKeyAuthentication",
            "modelApiKeyAuthentication",
            "replacementApiKey",
            "authGate",
            "authForm",
            "restartControlCenterButton",
            "restartMlxServerButton",
            "serviceRestartStatus",
        ):
            self.assertIn(f'id="{element_id}"', html)

    def test_active_prompt_progress_shows_estimated_kv_memory(self) -> None:
        html = control_plane.CONTROL_PLANE_HTML

        self.assertIn(
            "const kvBytesPerToken = "
            "Number(safety.profile?.kv_bytes_per_token || 0);",
            html,
        )
        self.assertIn(
            "formatBytes(promptProcessed * kvBytesPerToken)",
            html,
        )
        self.assertIn(
            "formatBytes(promptTotal * kvBytesPerToken)",
            html,
        )
        self.assertIn("KV memory", html)
        self.assertIn('id="harnessTab" class="harness-section tab-panel"', html)
        self.assertIn('aria-selected="false" aria-controls="harnessTab"', html)
        self.assertIn('aria-selected="false" aria-controls="telemetryTab"', html)
        self.assertIn('aria-selected="false" aria-controls="benchmarkTab"', html)
        self.assertIn('aria-selected="false" aria-controls="settingsTab"', html)
        self.assertIn("<main inert>", html)
        self.assertIn("mainContent.inert = false", html)
        benchmark_markup = html.split('id="benchmarkTab"', 1)[1].split(
            'id="harnessTab"', 1
        )[0]
        self.assertIn('id="benchmarkForm"', benchmark_markup)
        self.assertIn('id="statisticsRows"', benchmark_markup)
        self.assertIn("item.full_name || item.label", html)
        self.assertIn("model.full_name || model.label", html)
        self.assertIn("fastest?.full_name || fastest?.label", html)
        self.assertIn('id="bars" class="latest-comparison-bars"', html)
        self.assertIn('class="latest-comparison-model"', html)
        self.assertIn("<small>tok/s</small>", html)
        self.assertNotIn('id="statisticsTabButton"', html)

    def test_control_center_api_key_validation_is_constant_time_backed(self) -> None:
        with mock.patch.object(
            control_plane,
            "control_center_settings",
            return_value={
                "authentication": "api_key",
                "api_key": "expected-api-key",
            },
        ):
            self.assertTrue(
                control_plane.valid_control_center_api_key("expected-api-key")
            )
            self.assertFalse(control_plane.valid_control_center_api_key("wrong"))

        with mock.patch.object(
            control_plane,
            "control_center_settings",
            return_value={
                "authentication": "none",
                "api_key": "unused",
            },
        ):
            self.assertTrue(control_plane.valid_control_center_api_key(""))

        active = {
            "authentication": "api_key",
            "api_key": "active-api-key",
        }
        self.assertTrue(
            control_plane.valid_control_center_api_key(
                "active-api-key",
                active,
            )
        )
        self.assertFalse(
            control_plane.valid_control_center_api_key(
                "configured-but-not-active",
                active,
            )
        )

    def test_access_payload_reports_pending_restart(self) -> None:
        with mock.patch.object(
            control_plane,
            "control_center_settings",
            return_value={
                "access_mode": "network",
                "bind_host": "0.0.0.0",
                "port": 9876,
                "authentication": "api_key",
                "api_key_configured": True,
            },
        ):
            payload = control_plane.control_center_access_payload(
                active_host="127.0.0.1",
                active_port=8765,
                active_authentication="none",
            )

        self.assertFalse(payload["authentication_required"])
        self.assertTrue(payload["configured_authentication_required"])
        self.assertEqual(payload["active_authentication"], "none")
        self.assertTrue(payload["restart_required"])

    def test_access_payload_reports_model_api_restart(self) -> None:
        with mock.patch.object(
            control_plane,
            "control_center_settings",
            return_value={
                "access_mode": "local",
                "bind_host": "127.0.0.1",
                "port": 8765,
                "authentication": "none",
                "model_api_authentication": "api_key",
                "api_key_configured": True,
            },
        ):
            payload = control_plane.control_center_access_payload(
                active_host="127.0.0.1",
                active_port=8765,
                active_authentication="none",
                active_model_api_authentication="none",
            )

        self.assertFalse(payload["control_center_restart_required"])
        self.assertTrue(payload["model_api_restart_required"])
        self.assertTrue(payload["restart_required"])

    def test_access_payload_reports_api_key_rotation_as_pending(self) -> None:
        public_settings = {
            "access_mode": "network",
            "bind_host": "0.0.0.0",
            "port": 8765,
            "authentication": "api_key",
            "api_key_configured": True,
        }
        secret_settings = {
            **public_settings,
            "api_key": "replacement-api-key",
        }
        with mock.patch.object(
            control_plane,
            "control_center_settings",
            side_effect=[public_settings, secret_settings],
        ):
            payload = control_plane.control_center_access_payload(
                active_host="0.0.0.0",
                active_port=8765,
                active_authentication="api_key",
                active_api_key="active-api-key",
            )

        self.assertTrue(payload["authentication_required"])
        self.assertTrue(payload["restart_required"])

    def test_benchmark_models_include_canonical_full_names(self) -> None:
        configured = [
            (
                "qwen_model",
                {
                    "repo_id": "mlx-community/Qwen-Model-Full-8bit",
                    "aliases": ["qwen-short"],
                    "local_dir": (
                        "/models--mlx-community--Qwen-Model-Full-8bit/"
                        "snapshots/6d3c664dc8539a711783391484fd6784c51fd8fa"
                    ),
                },
            )
        ]

        with (
            mock.patch.object(
                control_plane,
                "configured_models",
                return_value=configured,
            ),
            mock.patch.object(
                control_plane,
                "model_is_downloaded",
                return_value=True,
            ),
        ):
            option = control_plane.model_options()[0]

        self.assertEqual(option["label"], "qwen-short")
        self.assertEqual(option["full_name"], "Qwen-Model-Full-8bit")
        self.assertNotIn("6d3c664", option["full_name"])

    def test_recent_benchmark_summaries_use_canonical_full_names(self) -> None:
        configured = [
            (
                "qwen_model",
                {
                    "repo_id": "mlx-community/Qwen-Model-Full-8bit",
                    "aliases": ["qwen-short"],
                    "local_dir": (
                        "/models--mlx-community--Qwen-Model-Full-8bit/"
                        "snapshots/6d3c664dc8539a711783391484fd6784c51fd8fa"
                    ),
                },
            )
        ]
        record = {
            "model_label": "qwen-short",
            "stats": {
                "prompt_tokens": 10,
                "prompt_tps": 100,
                "generation_tokens": 20,
                "generation_tps": 40,
                "peak_memory_gb": 12,
            },
        }

        with tempfile.TemporaryDirectory() as temporary:
            runs_dir = Path(temporary)
            (runs_dir / "benchmark.jsonl").write_text(
                json.dumps(record) + "\n",
                encoding="utf-8",
            )
            with (
                mock.patch.object(control_plane, "RUNS_DIR", runs_dir),
                mock.patch.object(
                    control_plane,
                    "configured_models",
                    return_value=configured,
                ),
                mock.patch.object(
                    control_plane,
                    "artifact_href",
                    side_effect=lambda path: f"/artifact/{path.name}",
                ),
            ):
                runs = control_plane.recent_runs()

        self.assertEqual(
            runs[0]["summaries"][0]["full_name"],
            "Qwen-Model-Full-8bit",
        )

    def test_telemetry_activity_tables_show_five_rows_before_expanding(self) -> None:
        html = control_plane.CONTROL_PLANE_HTML

        self.assertIn(
            ".telemetry-collapsible:not(.expanded) tr:nth-child(n + 6)",
            html,
        )
        self.assertIn(
            'aria-controls="telemetryModelRows" aria-expanded="false" hidden',
            html,
        )
        self.assertIn(
            'aria-controls="telemetryToolRows" aria-expanded="false" hidden',
            html,
        )
        self.assertIn("const hasAdditionalItems = itemCount > 5;", html)
        self.assertIn('classList.toggle("expanded");', html)
        self.assertNotIn("tools.slice(0, 40)", html)

    def test_recent_queries_show_retention_policy_and_page_controls(self) -> None:
        html = control_plane.CONTROL_PLANE_HTML

        self.assertIn(
            "30-day retention · daily archive · automatic deletion",
            html,
        )
        self.assertIn("request_page=${encodeURIComponent(page)}", html)
        self.assertIn("request_page_size=${encodeURIComponent(pageSize)}", html)
        self.assertIn('telemetryQueryPrevious.addEventListener("click"', html)
        self.assertIn('telemetryQueryNext.addEventListener("click"', html)
        self.assertIn('telemetryQueryPageSize.addEventListener("change"', html)

    def test_telemetry_failures_are_explained_and_linked_from_chart(self) -> None:
        html = control_plane.CONTROL_PLANE_HTML

        self.assertIn("Failure type, originating client, API endpoint", html)
        self.assertIn('data-failure-bucket-start="${bucketStart}"', html)
        self.assertIn("renderTelemetryFailures();", html)
        self.assertIn("record.failure_type", html)
        self.assertIn("record.client_address", html)
        self.assertIn("record.endpoint", html)
        self.assertIn("record.error", html)

    def test_service_restart_controls_use_explicit_routes(self) -> None:
        html = control_plane.CONTROL_PLANE_HTML

        self.assertIn("/api/control-center/restart", html)
        self.assertIn("/api/mlx-server/restart", html)
        self.assertIn("Restart Control Center", html)
        self.assertIn("Restart MLX Server", html)
        self.assertIn("window.location.assign(nextUrl.toString())", html)
        self.assertIn("restart", control_plane.MLX_SERVER_MUTATING_COMMANDS)
        self.assertEqual(
            control_plane.control_center_restart_command(),
            [sys.executable, "-m", "mlx_lab.cli", "ui"],
        )

    def test_control_center_restart_reexecutes_after_response_delay(self) -> None:
        fake_thread = mock.Mock()
        with mock.patch.object(
            control_plane.threading,
            "Thread",
            return_value=fake_thread,
        ) as thread_constructor:
            control_plane.schedule_control_center_restart(delay_seconds=0.5)

        fake_thread.start.assert_called_once_with()
        restart = thread_constructor.call_args.kwargs["target"]
        command = control_plane.control_center_restart_command()
        with (
            mock.patch.object(control_plane.time, "sleep") as sleep,
            mock.patch.object(control_plane.os, "execv") as execv,
        ):
            restart()

        sleep.assert_called_once_with(0.5)
        execv.assert_called_once_with(command[0], command)

    def test_harness_enter_submission_and_usage_metrics_are_wired(self) -> None:
        html = control_plane.CONTROL_PLANE_HTML

        self.assertIn(
            'if (event.key === "Enter" && !event.shiftKey && !event.isComposing)',
            html,
        )
        self.assertIn("harnessForm.requestSubmit();", html)
        self.assertIn("Enter to send · Shift+Enter for a new line", html)
        self.assertIn("message.usage.prompt_tokens", html)
        self.assertIn("message.usage.completion_tokens", html)
        self.assertIn(
            "assistant.tokensPerSecond = completionTokens / generationSeconds;",
            html,
        )
        self.assertIn('metrics.className = "harness-message-metrics";', html)

    def test_telemetry_store_aggregates_tokens_tools_and_mcp(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "telemetry.jsonl"
            store = TelemetryStore(path)
            store.start(
                {
                    "id": "request-telemetry",
                    "started_at": 100.0,
                    "status": "active",
                    "request_type": "chat",
                    "endpoint": "/v1/chat/completions",
                    "model": "models/example",
                    "client": "OpenCode",
                    "query_preview": "Inspect the repository",
                    "offered_tools": ["bash", "mcp__github__search_code"],
                    "tool_calls": [],
                    "tool_call_count": 0,
                    "mcp_tool_call_count": 0,
                }
            )
            store.record_response(
                "request-telemetry",
                {
                    "usage": {
                        "prompt_tokens": 120,
                        "completion_tokens": 30,
                        "prompt_tokens_details": {"cached_tokens": 40},
                    },
                    "choices": [
                        {
                            "message": {
                                "tool_calls": [
                                    {
                                        "function": {
                                            "name": "mcp__github__search_code",
                                            "arguments": '{"query":"telemetry"}',
                                        }
                                    }
                                ]
                            }
                        }
                    ],
                },
            )
            with mock.patch("mlx_lab.telemetry.time.time", return_value=103.0):
                completed = store.finish(
                    "request-telemetry",
                    request_state={
                        "prompt_tokens": 120,
                        "cached_prompt_tokens": 40,
                        "output_tokens": 30,
                        "status": "generating",
                        "cancel_requested": False,
                    },
                )
                payload = store.snapshot(window_seconds=0)
            telemetry_file_created = path.is_file()

        self.assertEqual(completed["status"], "completed")
        self.assertEqual(payload["summary"]["requests"], 1)
        self.assertEqual(payload["summary"]["prompt_tokens"], 120)
        self.assertEqual(payload["summary"]["output_tokens"], 30)
        self.assertEqual(payload["summary"]["cached_prompt_tokens"], 40)
        self.assertEqual(payload["summary"]["cache_hit_percent"], 25.0)
        self.assertEqual(payload["summary"]["tool_calls"], 1)
        self.assertEqual(payload["summary"]["mcp_tool_calls"], 1)
        self.assertEqual(payload["models"][0]["requests"], 1)
        self.assertEqual(payload["tools"][0]["mcp_server"], "github")
        self.assertEqual(payload["requests"][0]["query_preview"], "Inspect the repository")
        self.assertTrue(telemetry_file_created)

    def test_telemetry_recent_queries_are_paginated_newest_first(self) -> None:
        history = [
            {
                "id": f"request-{index}",
                "started_at": 100.0 + index,
                "completed_at": 100.5 + index,
                "status": "completed",
            }
            for index in range(5)
        ]

        payload = aggregate_telemetry(
            history,
            window_seconds=0,
            request_page=2,
            request_page_size=2,
            now=110.0,
        )

        self.assertEqual(
            [record["id"] for record in payload["requests"]],
            ["request-2", "request-1"],
        )
        self.assertEqual(
            payload["request_pagination"],
            {
                "page": 2,
                "page_size": 2,
                "total_items": 5,
                "total_pages": 3,
                "has_previous": True,
                "has_next": True,
                "first_item": 3,
                "last_item": 4,
            },
        )

    def test_telemetry_archives_daily_and_deletes_records_after_30_days(self) -> None:
        day = 86_400
        now = (50 * day) + 3_600
        records = [
            {
                "id": "current",
                "started_at": now - 60,
                "completed_at": now - 30,
                "status": "completed",
            },
            {
                "id": "archived",
                "started_at": now - (2 * day),
                "completed_at": now - (2 * day) + 30,
                "status": "completed",
            },
            {
                "id": "expired",
                "started_at": now - (31 * day),
                "completed_at": now - (31 * day) + 30,
                "status": "completed",
            },
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "telemetry.jsonl"
            path.write_text(
                "".join(f"{json.dumps(record)}\n" for record in records),
                encoding="utf-8",
            )
            archive_dir = path.parent / "telemetry-archive"
            archive_dir.mkdir()
            stale_archive = archive_dir / "stale.jsonl"
            stale_archive.write_text(
                f"{json.dumps(records[-1])}\n",
                encoding="utf-8",
            )

            with mock.patch("mlx_lab.telemetry.time.time", return_value=now):
                store = TelemetryStore(path, retention_days=30)
                payload = store.snapshot(window_seconds=0)

            active_records = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            archive_paths = list(archive_dir.glob("*.jsonl"))
            archived_records = [
                json.loads(line)
                for archive_path in archive_paths
                for line in archive_path.read_text(encoding="utf-8").splitlines()
            ]
            stale_archive_deleted = not stale_archive.exists()

        self.assertEqual([record["id"] for record in active_records], ["current"])
        self.assertEqual([record["id"] for record in archived_records], ["archived"])
        self.assertTrue(stale_archive_deleted)
        self.assertEqual(
            {record["id"] for record in payload["requests"]},
            {"current", "archived"},
        )
        self.assertEqual(payload["retention"]["days"], 30)
        self.assertTrue(payload["retention"]["automatic_deletion"])

    def test_telemetry_classifies_and_locates_failures(self) -> None:
        history = [
            {
                "id": "broken-pipe",
                "started_at": 100.0,
                "completed_at": 101.0,
                "status": "error",
                "request_type": "chat",
                "endpoint": "/v1/chat/completions",
                "model": "models/example",
                "client": "OpenCode",
                "client_address": "127.0.0.1",
                "error": "[Errno 32] Broken pipe",
            },
            {
                "id": "cancelled",
                "started_at": 102.0,
                "completed_at": 103.0,
                "status": "cancelled",
                "request_type": "chat",
                "endpoint": "/v1/chat/completions",
                "model": "models/example",
                "client": "Browser client",
                "client_address": "127.0.0.1",
                "error": "Cancelled from the Control Center",
            },
            {
                "id": "safety",
                "started_at": 104.0,
                "completed_at": 105.0,
                "status": "error",
                "request_type": "completion",
                "endpoint": "/v1/completions",
                "model": "models/example",
                "client": "Python client",
                "client_address": "127.0.0.1",
                "error": "MLX safety admission rejected the request: effective context cap exceeded.",
            },
        ]

        payload = aggregate_telemetry(
            history,
            window_seconds=0,
            now=106.0,
        )

        self.assertEqual(payload["summary"]["errors"], 3)
        self.assertEqual(
            {
                item["code"]: item["count"]
                for item in payload["failure_types"]
            },
            {
                "cancelled": 1,
                "client_disconnected": 1,
                "safety_admission": 1,
            },
        )
        self.assertEqual(
            {item["client"]: item["errors"] for item in payload["sources"]},
            {
                "Browser client": 1,
                "OpenCode": 1,
                "Python client": 1,
            },
        )
        events = {item["id"]: item for item in payload["error_events"]}
        self.assertEqual(
            events["broken-pipe"]["failure_type"],
            "Client disconnected",
        )
        self.assertEqual(
            events["broken-pipe"]["endpoint"],
            "/v1/chat/completions",
        )
        self.assertEqual(events["cancelled"]["failure_type"], "Cancelled")
        self.assertEqual(events["safety"]["failure_type"], "Safety admission")

    def test_telemetry_detects_active_agent_loop_and_cache_thrashing(self) -> None:
        history = []
        for index in range(35):
            cache_hit = index < 31
            completed_at = 110.0 + (index * 15)
            history.append(
                {
                    "id": f"request-{index}",
                    "started_at": completed_at - (12 if cache_hit else 240),
                    "completed_at": completed_at,
                    "status": "completed",
                    "request_type": "chat",
                    "model": "models/example",
                    "client": "OpenCode",
                    "client_address": "127.0.0.1",
                    "query_preview": "Implement the extension",
                    "message_count": 25 + (index * 2),
                    "prompt_tokens": 250 if cache_hit else 80_000 + index,
                    "cached_prompt_tokens": 60_000 if cache_hit else 0,
                    "output_tokens": 150,
                    "duration_seconds": 12 if cache_hit else 240,
                    "offered_tools": ["edit", "bash"],
                    "offered_tool_count": 2,
                    "tool_calls": [{"name": "edit"}],
                    "tool_call_count": 1,
                    "mcp_tool_call_count": 0,
                }
            )
        active = [
            {
                "id": "active-request",
                "started_at": 640.0,
                "status": "active",
                "request_type": "chat",
                "model": "models/example",
                "client": "OpenCode",
                "client_address": "127.0.0.1",
                "query_preview": "Implement the extension",
                "message_count": 95,
                "offered_tools": ["edit", "bash"],
                "offered_tool_count": 2,
            }
        ]

        payload = aggregate_telemetry(
            history,
            active=active,
            window_seconds=0,
            now=650.0,
        )

        self.assertEqual(payload["summary"]["agent_loop_risks"], 1)
        self.assertEqual(payload["summary"]["high_agent_loop_risks"], 1)
        risk = payload["agent_loop_risks"][0]
        self.assertEqual(risk["level"], "high")
        self.assertTrue(risk["active"])
        self.assertEqual(risk["consecutive_tool_steps"], 35)
        self.assertEqual(risk["dominant_tool"], "edit")
        self.assertEqual(risk["cache_hit_percent"], 0.0)
        self.assertEqual(
            {signal["code"] for signal in risk["signals"]},
            {
                "sustained_tool_chain",
                "repeated_tool_pattern",
                "prompt_cache_thrashing",
                "latency_degradation",
                "large_agent_context",
            },
        )

    def test_telemetry_clears_loop_risk_after_agent_finishes(self) -> None:
        history = []
        for index in range(10):
            history.append(
                {
                    "id": f"request-{index}",
                    "started_at": 100.0 + index,
                    "completed_at": 100.5 + index,
                    "status": "completed",
                    "request_type": "chat",
                    "model": "models/example",
                    "client": "OpenCode",
                    "client_address": "127.0.0.1",
                    "query_preview": "Inspect the repository",
                    "message_count": 5 + (index * 2),
                    "prompt_tokens": 1_000 + index,
                    "cached_prompt_tokens": 0,
                    "offered_tools": ["read"],
                    "offered_tool_count": 1,
                    "tool_calls": [{"name": "read"}],
                    "tool_call_count": 1,
                }
            )
        history.append(
            {
                "id": "final-response",
                "started_at": 111.0,
                "completed_at": 111.5,
                "status": "completed",
                "request_type": "chat",
                "model": "models/example",
                "client": "OpenCode",
                "client_address": "127.0.0.1",
                "query_preview": "Inspect the repository",
                "message_count": 25,
                "prompt_tokens": 1_100,
                "cached_prompt_tokens": 0,
                "offered_tools": ["read"],
                "offered_tool_count": 1,
                "tool_calls": [],
                "tool_call_count": 0,
            }
        )

        payload = aggregate_telemetry(
            history,
            window_seconds=0,
            now=112.0,
        )

        self.assertEqual(payload["summary"]["agent_loop_risks"], 0)
        self.assertEqual(payload["agent_loop_risks"], [])

    def test_telemetry_payload_falls_back_to_retained_history(self) -> None:
        record = {
            "id": "request-history",
            "started_at": 100.0,
            "completed_at": 101.0,
            "status": "completed",
            "model": "models/example",
            "client": "curl",
            "prompt_tokens": 10,
            "cached_prompt_tokens": 0,
            "output_tokens": 5,
            "tool_call_count": 0,
            "mcp_tool_call_count": 0,
            "offered_tools": [],
            "tool_calls": [],
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "telemetry.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            with (
                mock.patch.object(control_plane, "MLX_SERVER_TELEMETRY", path),
                mock.patch.object(
                    control_plane,
                    "mlx_internal_request",
                    side_effect=RuntimeError("server offline"),
                ),
                mock.patch("mlx_lab.telemetry.time.time", return_value=102.0),
            ):
                payload = control_plane.telemetry_payload(window_seconds=0)

        self.assertFalse(payload["runtime_available"])
        self.assertEqual(payload["summary"]["requests"], 1)
        self.assertEqual(payload["summary"]["total_tokens"], 15)

    def test_telemetry_enriches_an_older_live_server_payload(self) -> None:
        record = {
            "id": "older-runtime-error",
            "started_at": 100.0,
            "completed_at": 101.0,
            "status": "error",
            "request_type": "chat",
            "endpoint": "/v1/chat/completions",
            "model": "models/example",
            "client": "OpenCode",
            "client_address": "127.0.0.1",
            "error": "[Errno 32] Broken pipe",
        }
        older_payload = aggregate_telemetry(
            [record],
            window_seconds=0,
            now=102.0,
        )
        older_payload.pop("failure_types")
        older_payload.pop("error_events")
        for source in older_payload["sources"]:
            source.pop("errors")

        with (
            mock.patch.object(
                control_plane,
                "mlx_internal_request",
                return_value=older_payload,
            ),
            mock.patch.object(
                control_plane,
                "load_telemetry_history",
                return_value=[record],
            ),
            mock.patch("mlx_lab.telemetry.time.time", return_value=102.0),
        ):
            payload = control_plane.telemetry_payload(window_seconds=0)

        self.assertTrue(payload["runtime_available"])
        self.assertEqual(payload["sources"][0]["errors"], 1)
        self.assertEqual(
            payload["failure_types"][0]["label"],
            "Client disconnected",
        )
        self.assertEqual(
            payload["error_events"][0]["endpoint"],
            "/v1/chat/completions",
        )

    def test_model_statistics_include_every_configured_model(self) -> None:
        configured = [
            (
                "model_a",
                {
                    "repo_id": "example/model-a",
                    "aliases": ["model-a"],
                    "local_dir": "models/model-a",
                    "estimated_size_gb": 12.5,
                    "role": "Fast model",
                    "notes": "First model",
                },
            ),
            (
                "model_b",
                {
                    "repo_id": "example/model-b",
                    "aliases": ["model-b"],
                    "local_dir": "models/model-b",
                    "estimated_size_gb": 20.0,
                    "role": "Comparison model",
                    "notes": "Second model",
                },
            ),
        ]
        records = [
            {
                "model_key": "model_a",
                "model_label": "model-a",
                "prompt_name": "first",
                "category": "coding",
                "stats": {
                    "prompt_tokens": 10,
                    "prompt_tps": 100,
                    "generation_tokens": 20,
                    "generation_tps": 40,
                    "peak_memory_gb": 13,
                    "wall_seconds": 0.5,
                },
            },
            {
                "model_ref": "models/model-a",
                "prompt_name": "second",
                "category": "reasoning",
                "stats": {
                    "prompt_tokens": 30,
                    "prompt_tps": 200,
                    "generation_tokens": 40,
                    "generation_tps": 60,
                    "peak_memory_gb": 14,
                    "wall_seconds": 1.5,
                },
            },
        ]

        with tempfile.TemporaryDirectory() as temporary:
            runs_dir = Path(temporary)
            (runs_dir / "benchmark.jsonl").write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )
            with (
                mock.patch.object(control_plane, "RUNS_DIR", runs_dir),
                mock.patch.object(
                    control_plane,
                    "configured_models",
                    return_value=configured,
                ),
                mock.patch.object(
                    control_plane,
                    "model_is_downloaded",
                    return_value=True,
                ),
            ):
                payload = control_plane.model_statistics_payload()

        self.assertEqual(payload["summary"]["total_models"], 2)
        self.assertEqual(payload["summary"]["downloaded_models"], 2)
        self.assertEqual(payload["summary"]["benchmarked_models"], 1)
        self.assertEqual(payload["summary"]["total_records"], 2)
        self.assertEqual(payload["summary"]["unmatched_records"], 0)
        self.assertEqual(payload["summary"]["fastest_model"]["key"], "model_a")

        by_key = {item["key"]: item for item in payload["models"]}
        self.assertEqual(by_key["model_a"]["full_name"], "model-a")
        self.assertEqual(by_key["model_a"]["benchmark"]["record_count"], 2)
        self.assertEqual(
            by_key["model_a"]["benchmark"]["average_generation_tps"],
            50.0,
        )
        self.assertEqual(by_key["model_a"]["benchmark"]["peak_memory_gb"], 14.0)
        self.assertEqual(by_key["model_b"]["benchmark"]["record_count"], 0)
        self.assertIsNone(
            by_key["model_b"]["benchmark"]["average_generation_tps"]
        )

    def test_harness_chat_request_targets_active_model_and_streams(self) -> None:
        status = {
            "status": {
                "http_ready": True,
                "loaded_model": "/models/active",
            },
            "control": {
                "safety": {
                    "policy": {
                        "max_output_tokens": 4096,
                    }
                }
            },
        }
        with (
            mock.patch.object(
                control_plane,
                "mlx_server_status_payload",
                return_value=status,
            ),
            mock.patch.object(
                control_plane,
                "model_api_authorization_headers",
                return_value={"authorization": "Bearer shared-key"},
            ),
        ):
            request = control_plane.build_harness_chat_request(
                {
                    "model": "ignored-client-model",
                    "messages": [
                        {"role": "system", "content": "Answer directly."},
                        {"role": "user", "content": "Hello"},
                    ],
                    "max_tokens": 512,
                    "temperature": 0.3,
                }
            )

        body = json.loads(request.data)
        self.assertEqual(
            request.full_url,
            "http://127.0.0.1:48080/v1/chat/completions",
        )
        self.assertEqual(body["model"], "/models/active")
        self.assertEqual(body["max_tokens"], 512)
        self.assertEqual(body["temperature"], 0.3)
        self.assertTrue(body["stream"])
        self.assertTrue(body["stream_options"]["include_usage"])
        self.assertEqual(request.headers["Authorization"], "Bearer shared-key")

    def test_harness_chat_validation_rejects_invalid_messages(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported role"):
            control_plane.validate_harness_chat_payload(
                {
                    "messages": [
                        {"role": "tool", "content": "not allowed"},
                    ]
                },
                active_model="/models/active",
            )

        with self.assertRaisesRegex(ValueError, "between 1 and 100"):
            control_plane.validate_harness_chat_payload(
                {
                    "messages": [
                        {"role": "user", "content": "Hello"},
                    ],
                    "max_tokens": 101,
                },
                active_model="/models/active",
                max_output_tokens=100,
            )

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

    def test_model_api_key_validation_uses_the_active_shared_key(self) -> None:
        safety_module = self.load_safety_module()
        access = {
            "authentication": "api_key",
            "api_key": "one-shared-security-key",
        }

        self.assertTrue(
            safety_module.valid_model_api_key(
                "one-shared-security-key",
                access,
            )
        )
        self.assertFalse(
            safety_module.valid_model_api_key("wrong-security-key", access)
        )
        self.assertTrue(
            safety_module.valid_model_api_key(
                "",
                {"authentication": "none"},
            )
        )

    def test_model_api_handler_rejects_missing_key(self) -> None:
        safety_module = self.load_safety_module()

        class FakeHandler:
            def __init__(self, authorization: str = "") -> None:
                self.headers = {"Authorization": authorization}
                self.wfile = io.BytesIO()
                self.status = None
                self.called = False

            def _set_completion_headers(self, status: int) -> None:
                self.status = status

            def send_header(self, *_args) -> None:
                return

            def end_headers(self) -> None:
                return

            def do_GET(self) -> None:
                self.called = True

            def do_POST(self) -> None:
                self.called = True

        safety_module.mlx_server.APIHandler = FakeHandler
        safety_module._patch_api_key_authentication(
            {
                "authentication": "api_key",
                "api_key": "one-shared-security-key",
            }
        )

        unauthorized = FakeHandler()
        unauthorized.do_GET()
        self.assertEqual(unauthorized.status, 401)
        self.assertFalse(unauthorized.called)

        authorized = FakeHandler("Bearer one-shared-security-key")
        authorized.do_POST()
        self.assertTrue(authorized.called)

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

    def test_repetition_guard_detects_an_exact_token_cycle(self) -> None:
        safety_module = self.load_safety_module()
        guard = safety_module.TokenRepetitionGuard(
            min_generated_tokens=12,
            min_repeated_tokens=9,
            min_repeats=3,
            max_pattern_tokens=4,
        )

        matches = [
            guard.observe(token)
            for token in [9, 8, 7, 1, 2, 3, 1, 2, 3, 1, 2, 3]
        ]

        self.assertEqual(matches[-1], (3, 3))

    def test_repetition_guard_ignores_a_short_repeated_suffix(self) -> None:
        safety_module = self.load_safety_module()
        guard = safety_module.TokenRepetitionGuard(
            min_generated_tokens=8,
            min_repeated_tokens=8,
            min_repeats=4,
            max_pattern_tokens=4,
        )

        matches = [
            guard.observe(token)
            for token in [1, 2, 3, 4, 5, 6, 6, 6, 6]
        ]

        self.assertTrue(all(match is None for match in matches))

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

    def test_attached_context_uses_loaded_model_instead_of_default_alias(self) -> None:
        safety_module = self.load_safety_module()
        controller = safety_module.SafetyController()
        controller.publish = mock.Mock()
        controller.model_profile["model"] = "/models/qwen3-30b-8bit"
        request = object()
        args = SimpleNamespace(
            model=SimpleNamespace(model="default_model"),
            max_tokens=100,
        )
        controller.register("request-model", request, args, queue.Queue())

        controller.attach_context(
            "request-model",
            SimpleNamespace(stop=mock.Mock(), prompt_cache_count=12),
        )

        snapshot = controller.request_snapshot("request-model")
        self.assertEqual(snapshot["model"], "/models/qwen3-30b-8bit")
        self.assertEqual(snapshot["cached_prompt_tokens"], 12)

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

    def test_model_api_authorization_uses_shared_key(self) -> None:
        public = {"api_key_configured": True}
        secret = {**public, "api_key": "one-shared-security-key"}
        with mock.patch.object(
            control_plane,
            "control_center_settings",
            side_effect=[public, secret],
        ):
            headers = control_plane.model_api_authorization_headers()

        self.assertEqual(
            headers,
            {"authorization": "Bearer one-shared-security-key"},
        )


if __name__ == "__main__":
    unittest.main()
