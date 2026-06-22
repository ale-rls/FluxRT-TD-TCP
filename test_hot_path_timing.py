import importlib.util
import sys
import types
import unittest
from pathlib import Path


def load_server_module():
    aiohttp = types.ModuleType("aiohttp")
    aiohttp.web = types.SimpleNamespace(
        Request=object,
        Application=object,
        WebSocketResponse=object,
        json_response=lambda *args, **kwargs: None,
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


if __name__ == "__main__":
    unittest.main()
