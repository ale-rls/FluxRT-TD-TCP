"""
FluxRT TCP/WebSocket server.

Alternative to server.py (WHIP/WHEP over WebRTC, which needs UDP that
RunPod Pods don't support: https://docs.runpod.io/pods/networking — Pods
are TCP-only). This version carries frames over a plain WebSocket
instead, which is just TCP, so it works unmodified on a RunPod Pod.

Built to test ONE thing first: live frame cadence and server-observable
latency over this TCP path, with FluxRT in the loop, before deciding
whether the full two-tier WHIP/WHEP-over-udp-over-tcp architecture is
worth building.

Protocol (deliberately minimal):
  - client connects to ws://<host>:<port>/ws
  - client sends binary WebSocket messages, each one a single JPEG-
    encoded frame
  - server writes input frames into FluxRT's input tensor and
    independently reads the latest output tensor as a binary WebSocket
    message (also JPEG)
  - client measures send/receive cadence and latest-send-age timing;
    exact per-input RTT is not available because frames are untagged and
    FluxRT input/output loops are decoupled

No WHIP/WHEP, no aiortc, no SDP — none of that machinery is needed or
used here. This is intentionally the simplest thing that could possibly
measure "how slow is FluxRT-over-TCP, really."

Run:
    python server-tcp.py --config configs/stream_processor_config.json --port 8080

Benchmark against it with benchmark_ws.py before opening TouchDesigner.
"""

import argparse
import asyncio
import contextlib
import json
import logging
import math
import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np
from aiohttp import web, WSMsgType

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("fluxrt-tcp-server")

DEFAULT_OUTPUT_JPEG_QUALITY = 70  # cv2 expects 0-100; matches the 0.7
                                  # quality factor the original Daydream TD
                                  # extension used for its local MJPEG relay.
JPEG_QUALITY = DEFAULT_OUTPUT_JPEG_QUALITY  # legacy module-level name
OUTPUT_FPS = 25    # match FluxRT demo display pacing and the TD relay SEND_FPS

WORK_PRESETS = {
    "default": {
        "output_fps": OUTPUT_FPS,
        "input_fps": 0.0,
    },
    "light": {
        "output_fps": 15.0,
        "input_fps": 15.0,
    },
    "low": {
        "output_fps": 10.0,
        "input_fps": 10.0,
    },
}

HOT_PATH_STAGES = (
    "input_decode",
    "input_crop_copy",
    "output_read",
    "output_encode",
    "send",
)

SOURCE_PRESERVE_SUFFIX = (
    "Preserve the source camera feed: same subject, pose, silhouette, camera "
    "angle, framing, composition, background layout, and lighting direction. "
    "Do not replace the scene with an unrelated image."
)
SOURCE_PROMPT_STARTS = (
    "turn ",
    "transform ",
    "convert ",
    "make ",
    "restyle ",
    "preserve ",
    "repeat ",
)


@dataclass(frozen=True)
class WorkConfig:
    preset: str
    output_fps: float
    input_fps: float

    @property
    def output_interval(self) -> float:
        return 1.0 / self.output_fps

    @property
    def input_interval(self) -> float:
        if self.input_fps <= 0:
            return 0.0
        return 1.0 / self.input_fps


@dataclass(frozen=True)
class JpegConfig:
    output_quality: int

    @property
    def output_encode_params(self) -> list[int]:
        return [cv2.IMWRITE_JPEG_QUALITY, self.output_quality]


def build_effective_prompt(prompt: str, *, source_preserve: bool = True) -> str:
    prompt = prompt.strip()
    if not source_preserve:
        return prompt

    lower = prompt.lower()
    if lower.startswith(SOURCE_PROMPT_STARTS) or "this image" in lower:
        base = prompt.rstrip(".")
    else:
        base = f"Turn this camera image into {prompt}"
    return f"{base}. {SOURCE_PRESERVE_SUFFIX}"


def jpeg_quality(value: str | int, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not 1 <= parsed <= 100:
        raise ValueError(f"{name} must be between 1 and 100")
    return parsed


def positive_float(value: str, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def non_negative_float(value: str, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative")
    return parsed


def build_jpeg_config(
    *,
    output_quality: int | None = None,
    environ: dict[str, str] | None = None,
) -> JpegConfig:
    env = os.environ if environ is None else environ
    resolved_output_quality = DEFAULT_OUTPUT_JPEG_QUALITY

    if env.get("FLUXRT_OUTPUT_JPEG_QUALITY"):
        resolved_output_quality = jpeg_quality(
            env["FLUXRT_OUTPUT_JPEG_QUALITY"], "FLUXRT_OUTPUT_JPEG_QUALITY"
        )

    if output_quality is not None:
        resolved_output_quality = jpeg_quality(output_quality, "--output-jpeg-quality")

    return JpegConfig(output_quality=resolved_output_quality)


def build_work_config(
    *,
    preset: str | None = None,
    output_fps: float | None = None,
    input_fps: float | None = None,
    environ: dict[str, str] | None = None,
) -> WorkConfig:
    env = os.environ if environ is None else environ
    selected_preset = (
        preset or env.get("FLUXRT_WORK_PRESET") or "default"
    ).strip().lower()
    if selected_preset not in WORK_PRESETS:
        choices = ", ".join(sorted(WORK_PRESETS))
        raise ValueError(f"unknown work preset {selected_preset!r}; choose one of: {choices}")

    preset_values = WORK_PRESETS[selected_preset]
    resolved_output_fps = float(preset_values["output_fps"])
    resolved_input_fps = float(preset_values["input_fps"])

    if env.get("FLUXRT_OUTPUT_FPS"):
        resolved_output_fps = positive_float(env["FLUXRT_OUTPUT_FPS"], "FLUXRT_OUTPUT_FPS")
    if env.get("FLUXRT_INPUT_FPS"):
        resolved_input_fps = non_negative_float(env["FLUXRT_INPUT_FPS"], "FLUXRT_INPUT_FPS")

    if output_fps is not None:
        resolved_output_fps = positive_float(str(output_fps), "--output-fps")
    if input_fps is not None:
        resolved_input_fps = non_negative_float(str(input_fps), "--input-fps")

    return WorkConfig(
        preset=selected_preset,
        output_fps=resolved_output_fps,
        input_fps=resolved_input_fps,
    )


class StageTimingWindow:
    """Small bounded-by-interval timing accumulator for periodic summaries."""

    def __init__(self, stages=HOT_PATH_STAGES):
        self._samples_ms = {stage: [] for stage in stages}

    def observe_seconds(self, stage: str, elapsed_seconds: float):
        self._samples_ms.setdefault(stage, []).append(elapsed_seconds * 1000.0)

    def observe_many(self, timings: dict[str, float]):
        for stage, elapsed_seconds in timings.items():
            self.observe_seconds(stage, elapsed_seconds)

    def snapshot(self) -> dict[str, dict[str, float | int]]:
        summary = {}
        for stage, samples in self._samples_ms.items():
            summary[stage] = self._summarize(samples)
            samples.clear()
        return summary

    @staticmethod
    def _summarize(samples: list[float]) -> dict[str, float | int]:
        if not samples:
            return {"count": 0, "mean_ms": 0.0, "p95_ms": 0.0}
        ordered = sorted(samples)
        p95_index = max(0, math.ceil(len(ordered) * 0.95) - 1)
        return {
            "count": len(samples),
            "mean_ms": sum(samples) / len(samples),
            "p95_ms": ordered[p95_index],
        }


def rate_per_second(count: int, elapsed_seconds: float) -> float:
    if elapsed_seconds <= 0:
        return 0.0
    return count / elapsed_seconds


def parse_client_text_message(raw: str) -> dict | None:
    """Parse a client text frame into a message dict, or None if unusable.

    Clients only ever send JSON objects (e.g. {"prompt": ...}). Anything
    else — invalid JSON, or valid JSON that is not an object (`123`,
    `"prompt"`, `null`, ...) — must be rejected here rather than left to
    raise in the receiver: an uncaught receiver error stops every worker
    task and closes the whole streaming session.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        log.warning(f"Ignoring non-JSON text message: {raw!r}")
        return None
    if not isinstance(data, dict):
        log.warning(f"Ignoring non-object text message: {raw!r}")
        return None
    return data


class FluxRTRunner:
    """Same wrapper as in server.py — kept identical on purpose so any
    latency difference we measure is attributable to transport, not to
    a different FluxRT integration."""

    def __init__(
        self,
        config_path: str,
        use_int8: bool = False,
        jpeg_config: JpegConfig | None = None,
    ):
        self._lock = threading.Lock()
        self.jpeg_config = jpeg_config or build_jpeg_config()
        self._output_jpeg_encode_params = self.jpeg_config.output_encode_params
        from fluxrt import StreamProcessor
        self.processor = StreamProcessor(config_path)
        if use_int8:
            self.processor.enable_quantization()
        self.processor.start()
        self.config = getattr(self.processor, "config", {})

        log.info("Waiting for FluxRT subprocess pipeline to warm up...")
        while not self.processor.is_ready():
            time.sleep(0.1)

        self.input_tensor = self.processor.get_input_tensor()
        self.output_tensor = self.processor.get_output_tensor()
        self.resolution = self.processor.get_resolution()
        log.info(f"FluxRT ready at {self.resolution}")

    def set_prompt(self, prompt: str):
        self.processor.set_prompt(prompt)
        log.info(f"Prompt updated: {prompt!r}")

    # NOTE: input and output are DECOUPLED, matching run_cv2_demo.py.
    # copy_from() and to_numpy() are not a request/response pair — they
    # read/write shared-memory tensors that the background model
    # subprocess consumes/produces at its own (~8fps) pace, independent
    # of how fast we write input or read output. So we expose them as
    # two separate methods driven by two independent loops, instead of
    # one coupled process() call. Coupling them was what fed RIFE
    # sparse, far-apart frames and caused the glitchy interpolation.

    def write_input(self, frame_bgr: np.ndarray):
        from fluxrt.utils import crop_maximal_rectangle
        frame_bgr = crop_maximal_rectangle(
            frame_bgr, self.resolution["height"], self.resolution["width"]
        )
        # Latest-wins: just overwrite the input tensor, same as the demo
        # overwriting it every loop iteration. No queue, no waiting.
        self.input_tensor.copy_from(frame_bgr)

    def read_output(self) -> np.ndarray:
        # Whatever's currently in the output tensor — the freshest
        # available model+RIFE result. Not tied to any specific input.
        return self.output_tensor.to_numpy()

    def write_input_jpeg(self, jpeg_bytes: bytes) -> bool:
        ok, _timings = self.write_input_jpeg_timed(jpeg_bytes)
        return ok

    def write_input_jpeg_timed(self, jpeg_bytes: bytes) -> tuple[bool, dict[str, float]]:
        timings = {}
        t0 = time.perf_counter()
        np_arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        timings["input_decode"] = time.perf_counter() - t0
        if frame_bgr is None:
            return False, timings

        t0 = time.perf_counter()
        self.write_input(frame_bgr)
        timings["input_crop_copy"] = time.perf_counter() - t0
        return True, timings

    def read_output_jpeg(self) -> bytes | None:
        out, _timings = self.read_output_jpeg_timed()
        return out

    def read_output_jpeg_timed(self) -> tuple[bytes | None, dict[str, float]]:
        timings = {}
        t0 = time.perf_counter()
        processed = self.read_output()
        timings["output_read"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        ok, encoded = cv2.imencode(
            ".jpg", processed, self._output_jpeg_encode_params
        )
        timings["output_encode"] = time.perf_counter() - t0
        if not ok:
            return None, timings
        return encoded.tobytes(), timings


class FluxRTWebSocketSession:
    """Owns per-client WebSocket state and worker task lifecycle."""

    def __init__(self, request: web.Request, server: "FluxRTServer"):
        self.request = request
        self.server = server
        self.runner = server.runner
        self.executor = server.require_executor()
        self.work_config = server.work_config
        self.jpeg_config = server.jpeg_config
        self.loop = asyncio.get_event_loop()

        self.ws = web.WebSocketResponse(max_msg_size=20 * 1024 * 1024)
        self.latest_input: bytes | None = None
        self.latest_output: bytes | None = None
        self.got_input = asyncio.Event()
        self.got_output = asyncio.Event()
        self.stop_event = asyncio.Event()
        self.send_lock = asyncio.Lock()
        self.timing_window = StageTimingWindow()
        self.stats = {
            "rx": 0,
            "rx_bytes": 0,
            "input_overwritten": 0,
            "input_written": 0,
            "decode_failed": 0,
            "output_encoded": 0,
            "output_encoded_bytes": 0,
            "output_overwritten": 0,
            "sent": 0,
            "sent_bytes": 0,
        }

    async def run(self):
        await self.ws.prepare(self.request)
        log.info(
            "WebSocket client connected preset=%s output_fps=%.2f "
            "input_fps=%s output_jpeg_quality=%d",
            self.work_config.preset,
            self.work_config.output_fps,
            (
                "uncapped"
                if self.work_config.input_fps <= 0
                else f"{self.work_config.input_fps:.2f}"
            ),
            self.jpeg_config.output_quality,
        )

        tasks = [
            asyncio.create_task(self.receiver()),
            asyncio.create_task(self.input_worker()),
            asyncio.create_task(self.output_worker()),
            asyncio.create_task(self.sender()),
            asyncio.create_task(self.stats_loop()),
        ]
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            self.stop_all()
            await self.ws.close()
            await asyncio.gather(*tasks, return_exceptions=True)

        log.info("WebSocket client disconnected")
        return self.ws

    def stop_all(self):
        self.stop_event.set()
        self.got_input.set()
        self.got_output.set()

    async def receiver(self):
        try:
            async for msg in self.ws:
                if msg.type == WSMsgType.BINARY:
                    if self.latest_input is not None:
                        self.stats["input_overwritten"] += 1
                    self.latest_input = msg.data
                    self.stats["rx"] += 1
                    self.stats["rx_bytes"] += len(msg.data)
                    self.got_input.set()

                elif msg.type == WSMsgType.TEXT:
                    data = parse_client_text_message(msg.data)
                    if data is None:
                        continue

                    if "prompt" in data:
                        effective_prompt = await self.loop.run_in_executor(
                            self.executor, self.server.set_prompt, data["prompt"]
                        )
                        async with self.send_lock:
                            await self.ws.send_str(
                                json.dumps(
                                    {
                                        "ok": True,
                                        "prompt": data["prompt"],
                                        "effective_prompt": effective_prompt,
                                        "source_preserve": (
                                            self.server.source_preserve_prompts
                                        ),
                                    }
                                )
                            )

                elif msg.type == WSMsgType.ERROR:
                    log.error(f"WebSocket closed with exception {self.ws.exception()}")
                    break
        except (ConnectionResetError, RuntimeError):
            pass
        except Exception:
            log.exception("receiver error")
        finally:
            self.stop_all()

    async def input_worker(self):
        # Consumes only the newest input JPEG and writes FluxRT's input tensor
        # from the worker pool, keeping decode/crop/copy off the event loop.
        last_input_started_at: float | None = None
        input_interval = self.work_config.input_interval
        try:
            while not self.stop_event.is_set():
                await self.got_input.wait()
                self.got_input.clear()
                if self.stop_event.is_set():
                    break
                if input_interval and last_input_started_at is not None:
                    delay = max(
                        0.0,
                        last_input_started_at + input_interval - time.perf_counter(),
                    )
                    if delay:
                        with contextlib.suppress(asyncio.TimeoutError):
                            await asyncio.wait_for(
                                self.stop_event.wait(), timeout=delay
                            )
                    if self.stop_event.is_set():
                        break
                buf = self.latest_input
                self.latest_input = None
                if buf is None:
                    continue
                last_input_started_at = time.perf_counter()
                ok, timings = await self.loop.run_in_executor(
                    self.executor, self.runner.write_input_jpeg_timed, buf
                )
                self.timing_window.observe_many(timings)
                if ok:
                    self.stats["input_written"] += 1
                else:
                    self.stats["decode_failed"] += 1
        except Exception:
            log.exception("input_worker error")
        finally:
            self.stop_all()

    async def output_worker(self):
        # Independent of input: on a steady timer, read whatever's currently
        # in the output tensor and encode it into a latest-only output slot.
        output_interval = self.work_config.output_interval
        try:
            while not self.stop_event.is_set():
                t0 = time.perf_counter()
                out, timings = await self.loop.run_in_executor(
                    self.executor, self.runner.read_output_jpeg_timed
                )
                self.timing_window.observe_many(timings)
                if out is not None:
                    if self.latest_output is not None:
                        self.stats["output_overwritten"] += 1
                    self.latest_output = out
                    self.stats["output_encoded"] += 1
                    self.stats["output_encoded_bytes"] += len(out)
                    self.got_output.set()
                elapsed = time.perf_counter() - t0
                delay = max(0.0, output_interval - elapsed)
                if delay:
                    with contextlib.suppress(asyncio.TimeoutError):
                        await asyncio.wait_for(self.stop_event.wait(), timeout=delay)
        except Exception:
            log.exception("output_worker error")
        finally:
            self.stop_all()

    async def sender(self):
        # Sends only the freshest encoded output available after each network
        # backpressure wait. Older outputs are overwritten by output_worker.
        try:
            while not self.stop_event.is_set():
                await self.got_output.wait()
                self.got_output.clear()
                if self.stop_event.is_set():
                    break
                out = self.latest_output
                self.latest_output = None
                if out is None:
                    continue
                async with self.send_lock:
                    send_t0 = time.perf_counter()
                    await self.ws.send_bytes(out)
                    self.timing_window.observe_seconds(
                        "send", time.perf_counter() - send_t0
                    )
                self.stats["sent"] += 1
                self.stats["sent_bytes"] += len(out)
        except (ConnectionResetError, RuntimeError):
            pass
        except Exception:
            log.exception("sender error")
        finally:
            self.stop_all()

    async def stats_loop(self):
        last = self.stats.copy()
        last_report_t = time.perf_counter()
        try:
            while not self.stop_event.is_set():
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(self.stop_event.wait(), timeout=5.0)
                if self.stop_event.is_set():
                    break
                now = time.perf_counter()
                elapsed = now - last_report_t
                last_report_t = now
                delta = {k: self.stats[k] - last[k] for k in self.stats}
                last = self.stats.copy()
                self.log_stats_window(elapsed, delta)
        finally:
            self.log_totals()

    def log_stats_window(self, elapsed: float, delta: dict[str, int]):
        timings = self.timing_window.snapshot()
        avg_rx_kb = delta["rx_bytes"] / max(1, delta["rx"]) / 1024.0
        avg_encoded_kb = (
            delta["output_encoded_bytes"] / max(1, delta["output_encoded"]) / 1024.0
        )
        avg_sent_kb = delta["sent_bytes"] / max(1, delta["sent"]) / 1024.0
        log.info(
            "ws stats/5s window=%.2fs "
            "rx=%d rx_fps=%.2f "
            "wrote=%d wrote_fps=%.2f "
            "encoded=%d encoded_fps=%.2f "
            "sent=%d sent_fps=%.2f "
            "avg_kb rx=%.1f encoded=%.1f sent=%.1f "
            "drop_in=%d drop_out=%d bad_decode=%d "
            "hot_ms input_decode=%.2f/%.2f/%d "
            "input_crop_copy=%.2f/%.2f/%d "
            "output_read=%.2f/%.2f/%d "
            "output_encode=%.2f/%.2f/%d "
            "send=%.2f/%.2f/%d",
            elapsed,
            delta["rx"],
            rate_per_second(delta["rx"], elapsed),
            delta["input_written"],
            rate_per_second(delta["input_written"], elapsed),
            delta["output_encoded"],
            rate_per_second(delta["output_encoded"], elapsed),
            delta["sent"],
            rate_per_second(delta["sent"], elapsed),
            avg_rx_kb,
            avg_encoded_kb,
            avg_sent_kb,
            delta["input_overwritten"],
            delta["output_overwritten"],
            delta["decode_failed"],
            timings["input_decode"]["mean_ms"],
            timings["input_decode"]["p95_ms"],
            timings["input_decode"]["count"],
            timings["input_crop_copy"]["mean_ms"],
            timings["input_crop_copy"]["p95_ms"],
            timings["input_crop_copy"]["count"],
            timings["output_read"]["mean_ms"],
            timings["output_read"]["p95_ms"],
            timings["output_read"]["count"],
            timings["output_encode"]["mean_ms"],
            timings["output_encode"]["p95_ms"],
            timings["output_encode"]["count"],
            timings["send"]["mean_ms"],
            timings["send"]["p95_ms"],
            timings["send"]["count"],
        )

    def log_totals(self):
        log.info(
            "ws totals rx=%d wrote=%d encoded=%d sent=%d "
            "rx_mb=%.2f encoded_mb=%.2f sent_mb=%.2f "
            "drop_in=%d drop_out=%d bad_decode=%d",
            self.stats["rx"],
            self.stats["input_written"],
            self.stats["output_encoded"],
            self.stats["sent"],
            self.stats["rx_bytes"] / 1024.0 / 1024.0,
            self.stats["output_encoded_bytes"] / 1024.0 / 1024.0,
            self.stats["sent_bytes"] / 1024.0 / 1024.0,
            self.stats["input_overwritten"],
            self.stats["output_overwritten"],
            self.stats["decode_failed"],
        )


class FluxRTServer:
    """Owns shared server state and aiohttp lifecycle hooks."""

    def __init__(
        self,
        config_path: str,
        use_int8: bool,
        work_config: WorkConfig | None = None,
        jpeg_config: JpegConfig | None = None,
        runner_factory: Callable[..., FluxRTRunner] = FluxRTRunner,
        source_preserve_prompts: bool = True,
    ):
        self.work_config = work_config or build_work_config()
        self.jpeg_config = jpeg_config or build_jpeg_config()
        self.source_preserve_prompts = source_preserve_prompts
        self.runner = runner_factory(
            config_path, use_int8=use_int8, jpeg_config=self.jpeg_config
        )
        self._executor: ThreadPoolExecutor | None = None
        # There is exactly ONE FluxRT pipeline (one shared input/output
        # tensor pair) and the executor is sized for one session, so only
        # one WebSocket client can stream at a time. See adopt_session().
        self.active_session: "FluxRTWebSocketSession | None" = None

    @property
    def executor(self) -> ThreadPoolExecutor | None:
        return self._executor

    def start_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="fluxrt-ws"
            )
        return self._executor

    def shutdown_executor(self):
        if self._executor is None:
            return
        executor = self._executor
        self._executor = None
        executor.shutdown(wait=True)

    def require_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            raise RuntimeError("FluxRTServer executor is not running")
        return self._executor

    def create_app(self) -> web.Application:
        app = web.Application()
        app["server"] = self
        app.cleanup_ctx.append(self.executor_context)
        app.router.add_get("/ws", self.handle_ws)
        app.router.add_get("/status", self.handle_status)
        app.router.add_post("/prompt", self.handle_prompt)
        return app

    async def executor_context(self, app: web.Application):
        self.start_executor()
        try:
            yield
        finally:
            self.shutdown_executor()

    def adopt_session(self, session) -> None:
        """Make `session` the single active streaming session.

        Concurrent clients would interleave frames into the one shared
        FluxRT input tensor and contend for the two executor workers, so
        a new connection preempts the existing one instead of sharing
        with it. Preempting (rather than rejecting the newcomer) means a
        stale relay page or forgotten browser tab can never lock the
        live operator out — reconnecting always takes over.
        """
        previous = self.active_session
        self.active_session = session
        if previous is not None:
            log.warning(
                "New WebSocket client connected; preempting the previous "
                "streaming session"
            )
            previous.stop_all()

    def release_session(self, session) -> None:
        if self.active_session is session:
            self.active_session = None

    async def handle_ws(self, request: web.Request):
        session = FluxRTWebSocketSession(request, self)
        self.adopt_session(session)
        try:
            return await session.run()
        finally:
            self.release_session(session)

    def set_prompt(self, prompt: str) -> str:
        effective_prompt = build_effective_prompt(
            prompt, source_preserve=self.source_preserve_prompts
        )
        self.runner.set_prompt(effective_prompt)
        return effective_prompt

    async def handle_status(self, request: web.Request):
        return web.json_response(
            {
                "resolution": self.runner.resolution,
                "transport": "websocket-tcp",
                "work": {
                    "preset": self.work_config.preset,
                    "output_fps": self.work_config.output_fps,
                    "input_fps": self.work_config.input_fps,
                },
                "jpeg": {
                    "output_quality": self.jpeg_config.output_quality,
                },
                "fluxrt": {
                    key: self.runner.config.get(key)
                    for key in (
                        "default_steps",
                        "enable_spatial_cache",
                        "interpolation_exp",
                        "target_fps",
                        "resolution",
                    )
                    if key in self.runner.config
                },
                "prompt": {
                    "source_preserve": self.source_preserve_prompts,
                },
            }
        )

    async def handle_prompt(self, request: web.Request):
        """HTTP POST {"prompt": "..."} — used by the TD extension to update
        the prompt out-of-band from the frame WebSocket. (Prompts can also
        arrive as WebSocket text messages via handle_ws; both routes call
        the same runner.set_prompt.)"""
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "invalid json"}, status=400)
        prompt = data.get("prompt")
        if not prompt:
            return web.json_response({"error": "missing 'prompt'"}, status=400)
        loop = asyncio.get_event_loop()
        effective_prompt = await loop.run_in_executor(
            self.require_executor(), self.set_prompt, prompt
        )
        return web.json_response(
            {
                "ok": True,
                "prompt": prompt,
                "effective_prompt": effective_prompt,
                "source_preserve": self.source_preserve_prompts,
            }
        )


def create_app(
    config_path: str,
    use_int8: bool,
    work_config: WorkConfig | None = None,
    jpeg_config: JpegConfig | None = None,
    source_preserve_prompts: bool = True,
) -> web.Application:
    return FluxRTServer(
        config_path,
        use_int8=use_int8,
        work_config=work_config,
        jpeg_config=jpeg_config,
        source_preserve_prompts=source_preserve_prompts,
    ).create_app()


def main():
    parser = argparse.ArgumentParser(description="FluxRT TCP/WebSocket server")
    parser.add_argument(
        "--config",
        default="configs/stream_processor_config.json",
        help="FluxRT StreamProcessor config path",
    )
    parser.add_argument("--int8", action="store_true", help="Enable int8 quantization")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--work-preset",
        choices=sorted(WORK_PRESETS),
        default=None,
        help=(
            "Server work preset. default preserves existing behavior; light/low "
            "cap input writes and output reads for lower per-frame work. "
            "Can also be set with FLUXRT_WORK_PRESET."
        ),
    )
    parser.add_argument(
        "--output-fps",
        type=float,
        default=None,
        help="Server output tensor read/JPEG send cap. Overrides preset and FLUXRT_OUTPUT_FPS.",
    )
    parser.add_argument(
        "--input-fps",
        type=float,
        default=None,
        help=(
            "FluxRT input tensor write cap. 0 means uncapped latest-wins input writes. "
            "Overrides preset and FLUXRT_INPUT_FPS."
        ),
    )
    parser.add_argument(
        "--output-jpeg-quality",
        type=int,
        default=None,
        help=(
            "Server output JPEG quality, 1-100. Default 70; lower values reduce "
            "CPU encode cost and websocket bytes at visual quality cost. Can also "
            "be set with FLUXRT_OUTPUT_JPEG_QUALITY."
        ),
    )
    parser.add_argument(
        "--raw-prompts",
        action="store_true",
        help=(
            "Send prompts to FluxRT exactly as received. By default prompts "
            "are wrapped to preserve the input camera image."
        ),
    )
    args = parser.parse_args()

    try:
        work_config = build_work_config(
            preset=args.work_preset,
            output_fps=args.output_fps,
            input_fps=args.input_fps,
        )
        jpeg_config = build_jpeg_config(output_quality=args.output_jpeg_quality)
    except ValueError as exc:
        parser.error(str(exc))
    log.info(
        "work config preset=%s output_fps=%.2f input_fps=%s output_jpeg_quality=%d",
        work_config.preset,
        work_config.output_fps,
        "uncapped" if work_config.input_fps <= 0 else f"{work_config.input_fps:.2f}",
        jpeg_config.output_quality,
    )

    app = create_app(
        args.config,
        args.int8,
        work_config=work_config,
        jpeg_config=jpeg_config,
        source_preserve_prompts=not args.raw_prompts,
    )
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
