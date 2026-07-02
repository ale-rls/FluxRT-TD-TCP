import asyncio
import importlib.util
import sys
import types
import unittest
from pathlib import Path


def load_server_module():
    class FakeRouter:
        def __init__(self):
            self.routes = []

        def add_get(self, path, handler):
            self.routes.append(("GET", path, handler))

        def add_post(self, path, handler):
            self.routes.append(("POST", path, handler))

    class FakeApplication(dict):
        def __init__(self):
            super().__init__()
            self.cleanup_ctx = []
            self.router = FakeRouter()

    aiohttp = types.ModuleType("aiohttp")
    aiohttp.web = types.SimpleNamespace(
        Request=object,
        Application=FakeApplication,
        WebSocketResponse=object,
        json_response=lambda data, **kwargs: {"data": data, **kwargs},
    )
    aiohttp.WSMsgType = types.SimpleNamespace(BINARY=1, TEXT=2, ERROR=3)

    numpy = types.ModuleType("numpy")
    numpy.ndarray = object

    sys.modules.setdefault("aiohttp", aiohttp)
    cv2 = types.ModuleType("cv2")
    cv2.IMWRITE_JPEG_QUALITY = 1
    sys.modules.setdefault("cv2", cv2)
    sys.modules.setdefault("numpy", numpy)

    path = Path(__file__).with_name("server-tcp.py")
    spec = importlib.util.spec_from_file_location("server_tcp", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


server_tcp = load_server_module()


class FakeRunner:
    def __init__(self, config_path, use_int8=False, jpeg_config=None):
        self.config_path = config_path
        self.use_int8 = use_int8
        self.jpeg_config = jpeg_config
        self.resolution = {"width": 512, "height": 512}
        self.config = {
            "default_steps": 2,
            "enable_spatial_cache": False,
            "interpolation_exp": 1,
            "target_fps": None,
            "resolution": self.resolution,
        }
        self.prompts = []

    def set_prompt(self, prompt):
        self.prompts.append(prompt)


class FakeJsonRequest:
    def __init__(self, payload):
        self.payload = payload

    async def json(self):
        return self.payload


class StageTimingWindowTest(unittest.TestCase):
    def test_snapshot_reports_mean_p95_count_and_resets(self):
        window = server_tcp.StageTimingWindow(["decode"])
        for ms in (1.0, 2.0, 100.0):
            window.observe_seconds("decode", ms / 1000.0)

        snapshot = window.snapshot()

        self.assertEqual(snapshot["decode"]["count"], 3)
        self.assertAlmostEqual(snapshot["decode"]["mean_ms"], 103.0 / 3.0)
        self.assertEqual(snapshot["decode"]["p95_ms"], 100.0)
        self.assertEqual(window.snapshot()["decode"]["count"], 0)

    def test_observe_many_allows_runtime_stage_names(self):
        window = server_tcp.StageTimingWindow([])
        window.observe_many({"send": 0.004})

        snapshot = window.snapshot()

        self.assertEqual(snapshot["send"]["count"], 1)
        self.assertEqual(snapshot["send"]["mean_ms"], 4.0)
        self.assertEqual(snapshot["send"]["p95_ms"], 4.0)

    def test_rate_per_second_uses_elapsed_window(self):
        self.assertEqual(server_tcp.rate_per_second(125, 5.0), 25.0)
        self.assertEqual(server_tcp.rate_per_second(125, 0.0), 0.0)


class WorkConfigTest(unittest.TestCase):
    def test_default_work_config_preserves_existing_output_and_uncapped_input(self):
        config = server_tcp.build_work_config(environ={})

        self.assertEqual(config.preset, "default")
        self.assertEqual(config.output_fps, 25.0)
        self.assertEqual(config.input_fps, 0.0)
        self.assertEqual(config.output_interval, 1.0 / 25.0)
        self.assertEqual(config.input_interval, 0.0)

    def test_light_preset_caps_input_and_output_work(self):
        config = server_tcp.build_work_config(preset="light", environ={})

        self.assertEqual(config.preset, "light")
        self.assertEqual(config.output_fps, 15.0)
        self.assertEqual(config.input_fps, 15.0)
        self.assertEqual(config.input_interval, 1.0 / 15.0)

    def test_env_and_cli_overrides_take_precedence(self):
        config = server_tcp.build_work_config(
            preset="light",
            output_fps=12.0,
            input_fps=8.0,
            environ={
                "FLUXRT_WORK_PRESET": "low",
                "FLUXRT_OUTPUT_FPS": "20",
                "FLUXRT_INPUT_FPS": "10",
            },
        )

        self.assertEqual(config.preset, "light")
        self.assertEqual(config.output_fps, 12.0)
        self.assertEqual(config.input_fps, 8.0)

    def test_invalid_work_config_is_rejected(self):
        with self.assertRaises(ValueError):
            server_tcp.build_work_config(preset="fast", environ={})

        with self.assertRaises(ValueError):
            server_tcp.build_work_config(output_fps=0, environ={})

        with self.assertRaises(ValueError):
            server_tcp.build_work_config(input_fps=-1, environ={})

    def test_non_finite_work_config_values_are_rejected(self):
        for value in ("nan", "inf", "-inf"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    server_tcp.build_work_config(
                        environ={"FLUXRT_OUTPUT_FPS": value}
                    )
                with self.assertRaises(ValueError):
                    server_tcp.build_work_config(environ={"FLUXRT_INPUT_FPS": value})
                with self.assertRaises(ValueError):
                    server_tcp.build_work_config(
                        output_fps=float(value), environ={}
                    )
                with self.assertRaises(ValueError):
                    server_tcp.build_work_config(input_fps=float(value), environ={})


class ClientTextMessageTest(unittest.TestCase):
    def test_json_object_is_returned(self):
        self.assertEqual(
            server_tcp.parse_client_text_message('{"prompt": "a forest"}'),
            {"prompt": "a forest"},
        )

    def test_invalid_json_is_rejected(self):
        self.assertIsNone(server_tcp.parse_client_text_message("not json"))

    def test_non_object_json_is_rejected(self):
        # These previously raised TypeError in the receiver, which tore
        # down the entire streaming session.
        for raw in ("123", "true", "null", '"prompt"', "[1, 2]"):
            with self.subTest(raw=raw):
                self.assertIsNone(server_tcp.parse_client_text_message(raw))


class PromptConfigTest(unittest.TestCase):
    def test_source_preserving_prompt_wraps_short_style_prompt(self):
        prompt = server_tcp.build_effective_prompt("a lego character")

        self.assertTrue(prompt.startswith("Turn this camera image into a lego character."))
        self.assertIn("Preserve the source camera feed", prompt)
        self.assertIn("Do not replace the scene", prompt)

    def test_source_preserving_prompt_keeps_existing_instruction_shape(self):
        prompt = server_tcp.build_effective_prompt("Turn this image into oil painting.")

        self.assertTrue(prompt.startswith("Turn this image into oil painting."))
        self.assertIn("Preserve the source camera feed", prompt)

    def test_raw_prompt_mode_leaves_prompt_unchanged(self):
        prompt = server_tcp.build_effective_prompt(
            "a lego character", source_preserve=False
        )

        self.assertEqual(prompt, "a lego character")


class JpegConfigTest(unittest.TestCase):
    def test_default_jpeg_config_preserves_existing_output_quality(self):
        config = server_tcp.build_jpeg_config(environ={})

        self.assertEqual(config.output_quality, 70)
        self.assertEqual(
            config.output_encode_params,
            [server_tcp.cv2.IMWRITE_JPEG_QUALITY, 70],
        )

    def test_env_and_cli_output_quality_are_supported(self):
        config = server_tcp.build_jpeg_config(
            output_quality=55,
            environ={"FLUXRT_OUTPUT_JPEG_QUALITY": "60"},
        )

        self.assertEqual(config.output_quality, 55)

        env_config = server_tcp.build_jpeg_config(
            environ={"FLUXRT_OUTPUT_JPEG_QUALITY": "60"}
        )
        self.assertEqual(env_config.output_quality, 60)

    def test_invalid_jpeg_quality_is_rejected(self):
        for value in (0, 101, "low"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    server_tcp.build_jpeg_config(output_quality=value, environ={})

        with self.assertRaises(ValueError):
            server_tcp.build_jpeg_config(
                environ={"FLUXRT_OUTPUT_JPEG_QUALITY": "101"}
            )


class FluxRTServerLifecycleTest(unittest.TestCase):
    def make_server(self):
        return server_tcp.FluxRTServer(
            "config.json",
            use_int8=True,
            work_config=server_tcp.WorkConfig(
                preset="light", output_fps=15.0, input_fps=15.0
            ),
            jpeg_config=server_tcp.JpegConfig(output_quality=55),
            runner_factory=FakeRunner,
        )

    def test_server_owns_runner_and_config(self):
        server = self.make_server()

        self.assertIsInstance(server.runner, FakeRunner)
        self.assertEqual(server.runner.config_path, "config.json")
        self.assertTrue(server.runner.use_int8)
        self.assertIs(server.runner.jpeg_config, server.jpeg_config)
        self.assertEqual(server.work_config.preset, "light")
        self.assertEqual(server.jpeg_config.output_quality, 55)

    def test_executor_lifecycle_is_explicit_and_idempotent(self):
        server = self.make_server()

        with self.assertRaises(RuntimeError):
            server.require_executor()

        executor = server.start_executor()
        self.assertIs(server.executor, executor)
        self.assertIs(server.require_executor(), executor)
        self.assertIs(server.start_executor(), executor)

        server.shutdown_executor()
        self.assertIsNone(server.executor)
        with self.assertRaises(RuntimeError):
            server.require_executor()

        server.shutdown_executor()

    def test_create_app_registers_server_lifecycle_and_public_routes(self):
        server = self.make_server()

        app = server.create_app()

        self.assertIs(app["server"], server)
        self.assertEqual(app.cleanup_ctx, [server.executor_context])
        self.assertEqual(
            [(method, path) for method, path, _handler in app.router.routes],
            [("GET", "/ws"), ("GET", "/status"), ("POST", "/prompt")],
        )

    def test_status_response_preserves_public_fields(self):
        server = self.make_server()

        response = asyncio.run(server.handle_status(request=None))

        self.assertEqual(
            response["data"],
            {
                "resolution": {"width": 512, "height": 512},
                "transport": "websocket-tcp",
                "work": {
                    "preset": "light",
                    "output_fps": 15.0,
                    "input_fps": 15.0,
                },
                "jpeg": {
                    "output_quality": 55,
                },
                "fluxrt": {
                    "default_steps": 2,
                    "enable_spatial_cache": False,
                    "interpolation_exp": 1,
                    "target_fps": None,
                    "resolution": {"width": 512, "height": 512},
                },
                "prompt": {
                    "source_preserve": True,
                },
            },
        )

    def test_prompt_handler_uses_owned_executor_and_runner(self):
        server = self.make_server()
        server.start_executor()
        try:
            response = asyncio.run(
                server.handle_prompt(FakeJsonRequest({"prompt": "new prompt"}))
            )
        finally:
            server.shutdown_executor()

        self.assertEqual(response["data"]["ok"], True)
        self.assertEqual(response["data"]["prompt"], "new prompt")
        self.assertEqual(response["data"]["source_preserve"], True)
        self.assertEqual(
            response["data"]["effective_prompt"],
            server_tcp.build_effective_prompt("new prompt"),
        )
        self.assertEqual(
            server.runner.prompts,
            [server_tcp.build_effective_prompt("new prompt")],
        )


if __name__ == "__main__":
    unittest.main()
