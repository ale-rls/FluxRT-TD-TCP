"""
Pre-TouchDesigner websocket benchmark client for server-tcp.py.

It sends synthetic JPEG frames to the existing /ws websocket endpoint and
measures client-observed send/receive cadence plus response timing. The server
protocol does not tag frames and FluxRT input/output loops are decoupled, so
latency samples are "age since latest client send when a binary response
arrived", not a strict per-input model round trip.
"""

import argparse
import asyncio
import json
import math
import sys
import time
from dataclasses import dataclass, field
from io import BytesIO

BENCHMARK_PRESETS = {
    "baseline": {
        "width": 512,
        "height": 512,
        "fps": 25.0,
        "quality": 70,
    },
    "light": {
        "width": 512,
        "height": 512,
        "fps": 15.0,
        "quality": 70,
    },
    "low": {
        "width": 512,
        "height": 512,
        "fps": 10.0,
        "quality": 70,
    },
}


def percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    index = max(0, math.ceil(len(sorted_values) * pct) - 1)
    return sorted_values[index]


def rate_per_second(count: int, elapsed_seconds: float) -> float:
    if elapsed_seconds <= 0:
        return 0.0
    return count / elapsed_seconds


def average_bytes(total_bytes: int, count: int) -> float:
    if count <= 0:
        return 0.0
    return total_bytes / count


def require_finite(value: float, name: str) -> float:
    if not math.isfinite(value):
        raise SystemExit(f"{name} must be finite")
    return value


def summarize_ms(samples_seconds: list[float]) -> dict[str, float | int]:
    if not samples_seconds:
        return {
            "count": 0,
            "mean_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "max_ms": 0.0,
        }
    samples_ms = sorted(value * 1000.0 for value in samples_seconds)
    return {
        "count": len(samples_ms),
        "mean_ms": sum(samples_ms) / len(samples_ms),
        "p50_ms": percentile(samples_ms, 0.50),
        "p95_ms": percentile(samples_ms, 0.95),
        "max_ms": samples_ms[-1],
    }


@dataclass
class BenchmarkRecorder:
    sent: int = 0
    received: int = 0
    received_during_send: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    first_event_at: float | None = None
    last_event_at: float | None = None
    latest_send_at: float | None = None
    latest_send_age_samples: list[float] = field(default_factory=list)
    send_started_at: float | None = None
    send_ended_at: float | None = None
    receive_started_at: float | None = None
    receive_ended_at: float | None = None
    latest_send_age_active_samples: list[float] = field(default_factory=list)
    latest_send_age_drain_samples: list[float] = field(default_factory=list)

    def start_send_window(self, now: float):
        self.send_started_at = now

    def end_send_window(self, now: float):
        self.send_ended_at = now

    def start_receive_window(self, now: float):
        self.receive_started_at = now

    def end_receive_window(self, now: float):
        self.receive_ended_at = now

    def mark_sent(self, now: float, size: int):
        self._mark_event(now)
        self.sent += 1
        self.bytes_sent += size
        self.latest_send_at = now

    def mark_received(self, now: float, size: int):
        self._mark_event(now)
        self.received += 1
        during_send = self.send_ended_at is None
        if during_send:
            self.received_during_send += 1
        self.bytes_received += size
        if self.latest_send_at is not None:
            age = now - self.latest_send_at
            self.latest_send_age_samples.append(age)
            if during_send:
                self.latest_send_age_active_samples.append(age)
            else:
                self.latest_send_age_drain_samples.append(age)

    def _mark_event(self, now: float):
        if self.first_event_at is None:
            self.first_event_at = now
        self.last_event_at = now

    def send_window_seconds(self) -> float:
        if self.send_started_at is None or self.send_ended_at is None:
            return 0.0
        return max(0.0, self.send_ended_at - self.send_started_at)

    def receive_window_seconds(self) -> float:
        if self.receive_started_at is None or self.receive_ended_at is None:
            return 0.0
        return max(0.0, self.receive_ended_at - self.receive_started_at)

    def summary(self, requested_duration: float, elapsed: float) -> dict:
        send_window_s = self.send_window_seconds()
        receive_window_s = self.receive_window_seconds()
        return {
            "requested_duration_s": requested_duration,
            "elapsed_s": elapsed,
            "send_window_s": send_window_s,
            "receive_window_s": receive_window_s,
            "frames_sent": self.sent,
            "frames_received": self.received,
            "frames_received_during_send": self.received_during_send,
            "send_fps": rate_per_second(self.sent, send_window_s),
            "receive_fps": rate_per_second(
                self.received_during_send, receive_window_s
            ),
            "bytes_sent": self.bytes_sent,
            "bytes_received": self.bytes_received,
            "avg_sent_frame_bytes": average_bytes(self.bytes_sent, self.sent),
            "avg_received_frame_bytes": average_bytes(
                self.bytes_received, self.received
            ),
            "sent_minus_received_frames": self.sent - self.received,
            "latency_kind": "latest_send_age_active_ms",
            "latest_send_age_active_ms": summarize_ms(
                self.latest_send_age_active_samples
            ),
            "latest_send_age_drain_ms": summarize_ms(
                self.latest_send_age_drain_samples
            ),
            "latest_send_age_ms": summarize_ms(self.latest_send_age_samples),
            "protocol_note": (
                "Frames are untagged and server input/output loops are "
                "decoupled; active latency is age since the latest client send "
                "when a binary response arrived during the send window, not a "
                "strict per-input model RTT."
            ),
        }


def make_synthetic_jpeg(width: int, height: int, quality: int) -> bytes:
    try:
        return _make_jpeg_cv2(width, height, quality)
    except ImportError:
        return _make_jpeg_pillow(width, height, quality)


def _make_jpeg_cv2(width: int, height: int, quality: int) -> bytes:
    import cv2
    import numpy as np

    x = np.linspace(0, 255, width, dtype=np.uint8)
    y = np.linspace(0, 255, height, dtype=np.uint8)
    xv = np.tile(x, (height, 1))
    yv = np.tile(y[:, None], (1, width))
    frame = np.dstack((xv, yv, ((xv.astype(np.uint16) + yv) // 2).astype(np.uint8)))
    ok, encoded = cv2.imencode(
        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    )
    if not ok:
        raise RuntimeError("OpenCV failed to encode the synthetic JPEG frame")
    return encoded.tobytes()


def _make_jpeg_pillow(width: int, height: int, quality: int) -> bytes:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "benchmark_ws.py needs OpenCV+NumPy or Pillow to generate JPEG "
            "frames. Install the server deps, for example: "
            "pip install aiohttp opencv-python-headless numpy"
        ) from exc

    image = Image.new("RGB", (width, height))
    pixels = image.load()
    for y in range(height):
        for x in range(width):
            pixels[x, y] = (
                int(255 * x / max(1, width - 1)),
                int(255 * y / max(1, height - 1)),
                int(255 * (x + y) / max(1, width + height - 2)),
            )
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=int(quality))
    return buf.getvalue()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark the FluxRT websocket path before opening TouchDesigner"
    )
    parser.add_argument("url", help="WebSocket URL, e.g. wss://.../ws")
    parser.add_argument(
        "--benchmark-preset",
        choices=sorted(BENCHMARK_PRESETS),
        default="baseline",
        help="Client frame preset. Explicit --width/--height/--fps/--quality override it.",
    )
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--quality", type=int, default=None)
    parser.add_argument(
        "--receive-drain",
        type=float,
        default=2.0,
        help="Seconds to keep reading responses after sending stops",
    )
    parser.add_argument(
        "--max-message-mb",
        type=int,
        default=20,
        help="Maximum websocket message size in MiB",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of the text summary",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    preset = BENCHMARK_PRESETS[args.benchmark_preset]
    if args.width is None:
        args.width = preset["width"]
    if args.height is None:
        args.height = preset["height"]
    if args.fps is None:
        args.fps = preset["fps"]
    if args.quality is None:
        args.quality = preset["quality"]

    require_finite(float(args.fps), "--fps")
    require_finite(float(args.duration), "--duration")
    require_finite(float(args.receive_drain), "--receive-drain")
    require_finite(float(args.max_message_mb), "--max-message-mb")

    if args.width <= 0 or args.height <= 0:
        raise SystemExit("--width and --height must be positive")
    if args.fps <= 0:
        raise SystemExit("--fps must be positive")
    if args.duration <= 0:
        raise SystemExit("--duration must be positive")
    if not 1 <= args.quality <= 100:
        raise SystemExit("--quality must be between 1 and 100")
    if args.receive_drain < 0:
        raise SystemExit("--receive-drain must be non-negative")
    return args


async def run_benchmark(args: argparse.Namespace) -> dict:
    try:
        import aiohttp
    except ImportError as exc:
        raise RuntimeError(
            "benchmark_ws.py requires aiohttp for websocket transport. "
            "Install it with: pip install aiohttp"
        ) from exc

    jpeg = make_synthetic_jpeg(args.width, args.height, args.quality)
    recorder = BenchmarkRecorder()

    timeout = aiohttp.ClientTimeout(total=args.duration + args.receive_drain + 30.0)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.ws_connect(
            args.url, max_msg_size=args.max_message_mb * 1024 * 1024
        ) as ws:
            benchmark_started_at = time.perf_counter()
            recorder.start_receive_window(benchmark_started_at)

            async def send_loop():
                interval = 1.0 / args.fps
                send_started_at = benchmark_started_at
                recorder.start_send_window(send_started_at)
                next_send = send_started_at
                stop_at = send_started_at + args.duration
                try:
                    while True:
                        now = time.perf_counter()
                        if now >= stop_at:
                            break
                        if now < next_send:
                            await asyncio.sleep(next_send - now)
                            now = time.perf_counter()
                            if now >= stop_at:
                                break
                        await ws.send_bytes(jpeg)
                        recorder.mark_sent(time.perf_counter(), len(jpeg))
                        next_send = max(next_send + interval, time.perf_counter())
                finally:
                    recorder.end_send_window(time.perf_counter())
                    recorder.end_receive_window(recorder.send_ended_at)

            async def receive_loop():
                stop_at = benchmark_started_at + args.duration + args.receive_drain
                while True:
                    remaining = stop_at - time.perf_counter()
                    if remaining <= 0:
                        break
                    try:
                        msg = await ws.receive(timeout=remaining)
                    except asyncio.TimeoutError:
                        break
                    if msg.type == aiohttp.WSMsgType.BINARY:
                        recorder.mark_received(time.perf_counter(), len(msg.data))
                    elif msg.type in (
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.CLOSE,
                        aiohttp.WSMsgType.CLOSING,
                    ):
                        break
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        raise RuntimeError(f"websocket error: {ws.exception()}")

            await asyncio.gather(send_loop(), receive_loop())

    elapsed = time.perf_counter() - benchmark_started_at
    summary = recorder.summary(args.duration, elapsed)
    summary.update(
        {
            "benchmark_preset": args.benchmark_preset,
            "frame_width": args.width,
            "frame_height": args.height,
            "requested_send_fps": args.fps,
            "jpeg_quality": args.quality,
            "input_jpeg_bytes": len(jpeg),
        }
    )
    return summary


def print_text_summary(summary: dict):
    active_latency = summary["latest_send_age_active_ms"]
    drain_latency = summary["latest_send_age_drain_ms"]
    combined_latency = summary["latest_send_age_ms"]
    print(
        "preset={benchmark_preset} frame={frame_width}x{frame_height} "
        "requested_send_fps={requested_send_fps:.2f} quality={jpeg_quality}".format(
            **summary
        )
    )
    print(
        "elapsed={elapsed_s:.2f}s requested={requested_duration_s:.2f}s "
        "send_window={send_window_s:.2f}s receive_window={receive_window_s:.2f}s "
        "sent={frames_sent} received={frames_received} "
        "received_during_send={frames_received_during_send} "
        "send_fps={send_fps:.2f} receive_fps={receive_fps:.2f} "
        "sent_minus_received={sent_minus_received_frames}".format(**summary)
    )
    print(
        "bytes sent={bytes_sent} received={bytes_received} "
        "input_jpeg={input_jpeg_bytes} "
        "avg_sent_frame={avg_sent_frame_bytes:.0f} "
        "avg_received_frame={avg_received_frame_bytes:.0f}".format(**summary)
    )
    print(
        "latest_send_age_active_ms count={count} mean={mean_ms:.2f} "
        "p50={p50_ms:.2f} p95={p95_ms:.2f} max={max_ms:.2f}".format(
            **active_latency
        )
    )
    print(
        "latest_send_age_drain_ms count={count} mean={mean_ms:.2f} "
        "p50={p50_ms:.2f} p95={p95_ms:.2f} max={max_ms:.2f}".format(
            **drain_latency
        )
    )
    print(
        "latest_send_age_combined_ms count={count} mean={mean_ms:.2f} "
        "p50={p50_ms:.2f} p95={p95_ms:.2f} max={max_ms:.2f}".format(
            **combined_latency
        )
    )
    print(summary["protocol_note"])


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = asyncio.run(run_benchmark(args))
    except Exception as exc:
        print(f"benchmark failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print_text_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
