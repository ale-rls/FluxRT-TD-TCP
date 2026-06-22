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
    sys.modules.setdefault("cv2", types.ModuleType("cv2"))
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


if __name__ == "__main__":
    unittest.main()
