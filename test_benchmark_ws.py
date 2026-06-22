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

    def test_recorder_summary_reports_cadence_and_count_delta(self):
        recorder = benchmark_ws.BenchmarkRecorder()
        recorder.start_send_window(10.0)
        recorder.start_receive_window(10.0)
        recorder.mark_sent(10.0, 100)
        recorder.mark_sent(10.5, 100)
        recorder.mark_received(10.75, 200)
        recorder.end_send_window(11.0)
        recorder.end_receive_window(11.0)

        summary = recorder.summary(requested_duration=1.0, elapsed=3.0)

        self.assertEqual(summary["frames_sent"], 2)
        self.assertEqual(summary["frames_received"], 1)
        self.assertEqual(summary["frames_received_during_send"], 1)
        self.assertEqual(summary["send_window_s"], 1.0)
        self.assertEqual(summary["receive_window_s"], 1.0)
        self.assertEqual(summary["send_fps"], 2.0)
        self.assertEqual(summary["receive_fps"], 1.0)
        self.assertEqual(summary["sent_minus_received_frames"], 1)
        self.assertEqual(summary["latest_send_age_ms"]["count"], 1)
        self.assertEqual(summary["latest_send_age_ms"]["mean_ms"], 250.0)

    def test_recorder_fps_ignores_connection_delay_and_drain(self):
        recorder = benchmark_ws.BenchmarkRecorder()
        recorder.start_send_window(5.0)
        recorder.start_receive_window(5.0)
        for offset in (0.0, 0.25, 0.50, 0.75):
            recorder.mark_sent(5.0 + offset, 100)
            recorder.mark_received(5.0 + offset + 0.10, 200)
        recorder.end_send_window(6.0)
        recorder.end_receive_window(6.0)

        summary = recorder.summary(requested_duration=1.0, elapsed=8.0)

        self.assertEqual(summary["elapsed_s"], 8.0)
        self.assertEqual(summary["send_window_s"], 1.0)
        self.assertEqual(summary["receive_window_s"], 1.0)
        self.assertEqual(summary["send_fps"], 4.0)
        self.assertEqual(summary["receive_fps"], 4.0)

    def test_recorder_receive_fps_excludes_drain_responses(self):
        recorder = benchmark_ws.BenchmarkRecorder()
        recorder.start_send_window(20.0)
        recorder.start_receive_window(20.0)
        recorder.mark_sent(20.0, 100)
        recorder.mark_received(20.5, 200)
        recorder.end_send_window(21.0)
        recorder.end_receive_window(21.0)
        recorder.mark_received(22.0, 200)

        summary = recorder.summary(requested_duration=1.0, elapsed=3.0)

        self.assertEqual(summary["frames_received"], 2)
        self.assertEqual(summary["frames_received_during_send"], 1)
        self.assertEqual(summary["receive_fps"], 1.0)
        self.assertEqual(summary["sent_minus_received_frames"], -1)

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
