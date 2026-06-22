import unittest

import benchmark_ws


class BenchmarkWsHelpersTest(unittest.TestCase):
    def test_summarize_ms_reports_latency_distribution(self):
        summary = benchmark_ws.summarize_ms([0.001, 0.002, 0.100])

        self.assertEqual(summary["count"], 3)
        self.assertAlmostEqual(summary["mean_ms"], 103.0 / 3.0)
        self.assertEqual(summary["p50_ms"], 2.0)
        self.assertEqual(summary["p95_ms"], 100.0)
        self.assertEqual(summary["max_ms"], 100.0)

    def test_recorder_summary_reports_cadence_and_unmatched_sends(self):
        recorder = benchmark_ws.BenchmarkRecorder()
        recorder.mark_sent(10.0, 100)
        recorder.mark_sent(10.5, 100)
        recorder.mark_received(10.75, 200)

        summary = recorder.summary(requested_duration=1.0, elapsed=1.0)

        self.assertEqual(summary["frames_sent"], 2)
        self.assertEqual(summary["frames_received"], 1)
        self.assertEqual(summary["send_fps"], 2.0)
        self.assertEqual(summary["receive_fps"], 1.0)
        self.assertEqual(summary["unmatched_sent_frames"], 1)
        self.assertEqual(summary["latest_send_age_ms"]["count"], 1)
        self.assertEqual(summary["latest_send_age_ms"]["mean_ms"], 250.0)

    def test_parse_args_validates_benchmark_inputs(self):
        args = benchmark_ws.parse_args(
            [
                "ws://127.0.0.1:8080/ws",
                "--width",
                "640",
                "--height",
                "360",
                "--fps",
                "30",
                "--duration",
                "5",
                "--quality",
                "75",
            ]
        )

        self.assertEqual(args.width, 640)
        self.assertEqual(args.height, 360)
        self.assertEqual(args.fps, 30)
        self.assertEqual(args.duration, 5)
        self.assertEqual(args.quality, 75)

        with self.assertRaises(SystemExit):
            benchmark_ws.parse_args(["ws://127.0.0.1:8080/ws", "--fps", "0"])


if __name__ == "__main__":
    unittest.main()
