"""
FluxRT TCP/WebSocket server.

Alternative to server.py (WHIP/WHEP over WebRTC, which needs UDP that
RunPod Pods don't support: https://docs.runpod.io/pods/networking — Pods
are TCP-only). This version carries frames over a plain WebSocket
instead, which is just TCP, so it works unmodified on a RunPod Pod.

Built to test ONE thing first: real round-trip frame latency over this
TCP path, with FluxRT in the loop, before deciding whether the full
two-tier WHIP/WHEP-over-udp-over-tcp architecture is worth building.

Protocol (deliberately minimal):
  - client connects to ws://<host>:<port>/ws
  - client sends binary WebSocket messages, each one a single JPEG-
    encoded frame
  - server runs each frame through FluxRT, returns the result as a
    binary WebSocket message (also JPEG)
  - client measures the round-trip time itself

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
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from aiohttp import web, WSMsgType

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("fluxrt-tcp-server")

JPEG_QUALITY = 70  # cv2 expects 0-100, matches the 0.7 quality factor
                   # the original Daydream TD extension used for its own
                   # local MJPEG-over-WebSocket relay (JPEG_QUALITY_STREAM)
OUTPUT_FPS = 25    # match FluxRT demo display pacing and the TD relay SEND_FPS

HOT_PATH_STAGES = (
    "input_decode",
    "input_crop_copy",
    "output_read",
    "output_encode",
    "send",
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


class FluxRTRunner:
    """Same wrapper as in server.py — kept identical on purpose so any
    latency difference we measure is attributable to transport, not to
    a different FluxRT integration."""

    def __init__(self, config_path: str, use_int8: bool = False):
        self._lock = threading.Lock()
        from fluxrt import StreamProcessor
        self.processor = StreamProcessor(config_path)
        if use_int8:
            self.processor.enable_quantization()
        self.processor.start()

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
            ".jpg", processed, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        )
        timings["output_encode"] = time.perf_counter() - t0
        if not ok:
            return None, timings
        return encoded.tobytes(), timings


async def handle_ws(request: web.Request):
    app = request.app
    runner: FluxRTRunner = app["runner"]
    executor: ThreadPoolExecutor = app["executor"]

    ws = web.WebSocketResponse(max_msg_size=20 * 1024 * 1024)  # 20MB, generous
                                                                # headroom over
                                                                # any single
                                                                # JPEG frame
    await ws.prepare(request)
    log.info("WebSocket client connected")

    loop = asyncio.get_event_loop()

    # Output read rate. The model produces ~8fps natively but RIFE fills
    # in between, so the output tensor refreshes faster than that. A
    # separate sender task drains a latest-only slot, so slow network sends
    # drop stale encoded outputs instead of making a send queue.
    output_interval = 1.0 / OUTPUT_FPS

    latest_input = {"buf": None}
    got_input = asyncio.Event()
    latest_output = {"buf": None}
    got_output = asyncio.Event()
    stop_event = asyncio.Event()
    send_lock = asyncio.Lock()
    timing_window = StageTimingWindow()
    stats = {
        "rx": 0,
        "input_overwritten": 0,
        "input_written": 0,
        "decode_failed": 0,
        "output_encoded": 0,
        "output_overwritten": 0,
        "sent": 0,
    }

    def stop_all():
        stop_event.set()
        got_input.set()
        got_output.set()

    async def receiver():
        try:
            async for msg in ws:
                if msg.type == WSMsgType.BINARY:
                    if latest_input["buf"] is not None:
                        stats["input_overwritten"] += 1
                    latest_input["buf"] = msg.data
                    stats["rx"] += 1
                    got_input.set()

                elif msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                    except json.JSONDecodeError:
                        log.warning(f"Ignoring non-JSON text message: {msg.data!r}")
                        continue

                    if "prompt" in data:
                        await loop.run_in_executor(
                            executor, runner.set_prompt, data["prompt"]
                        )
                        async with send_lock:
                            await ws.send_str(json.dumps({"ok": True}))

                elif msg.type == WSMsgType.ERROR:
                    log.error(f"WebSocket closed with exception {ws.exception()}")
                    break
        except (ConnectionResetError, RuntimeError):
            pass
        except Exception:
            log.exception("receiver error")
        finally:
            stop_all()

    async def input_worker():
        # Consumes only the newest input JPEG and writes FluxRT's input tensor
        # from the worker pool, keeping decode/crop/copy off the event loop.
        try:
            while not stop_event.is_set():
                await got_input.wait()
                got_input.clear()
                if stop_event.is_set():
                    break
                buf = latest_input["buf"]
                latest_input["buf"] = None
                if buf is None:
                    continue
                ok, timings = await loop.run_in_executor(
                    executor, runner.write_input_jpeg_timed, buf
                )
                timing_window.observe_many(timings)
                if ok:
                    stats["input_written"] += 1
                else:
                    stats["decode_failed"] += 1
        except Exception:
            log.exception("input_worker error")
        finally:
            stop_all()

    async def output_worker():
        # Independent of input: on a steady timer, read whatever's currently
        # in the output tensor and encode it into a latest-only output slot.
        try:
            while not stop_event.is_set():
                t0 = time.perf_counter()
                out, timings = await loop.run_in_executor(
                    executor, runner.read_output_jpeg_timed
                )
                timing_window.observe_many(timings)
                if out is not None:
                    if latest_output["buf"] is not None:
                        stats["output_overwritten"] += 1
                    latest_output["buf"] = out
                    stats["output_encoded"] += 1
                    got_output.set()
                elapsed = time.perf_counter() - t0
                delay = max(0.0, output_interval - elapsed)
                if delay:
                    with contextlib.suppress(asyncio.TimeoutError):
                        await asyncio.wait_for(stop_event.wait(), timeout=delay)
        except Exception:
            log.exception("output_worker error")
        finally:
            stop_all()

    async def sender():
        # Sends only the freshest encoded output available after each network
        # backpressure wait. Older outputs are overwritten by output_worker.
        try:
            while not stop_event.is_set():
                await got_output.wait()
                got_output.clear()
                if stop_event.is_set():
                    break
                out = latest_output["buf"]
                latest_output["buf"] = None
                if out is None:
                    continue
                async with send_lock:
                    send_t0 = time.perf_counter()
                    await ws.send_bytes(out)
                    timing_window.observe_seconds(
                        "send", time.perf_counter() - send_t0
                    )
                stats["sent"] += 1
        except (ConnectionResetError, RuntimeError):
            pass
        except Exception:
            log.exception("sender error")
        finally:
            stop_all()

    async def stats_loop():
        last = stats.copy()
        last_report_t = time.perf_counter()
        try:
            while not stop_event.is_set():
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(stop_event.wait(), timeout=5.0)
                if stop_event.is_set():
                    break
                now = time.perf_counter()
                elapsed = now - last_report_t
                last_report_t = now
                delta = {k: stats[k] - last[k] for k in stats}
                last = stats.copy()
                timings = timing_window.snapshot()
                log.info(
                    "ws stats/5s window=%.2fs "
                    "rx=%d rx_fps=%.2f "
                    "wrote=%d wrote_fps=%.2f "
                    "encoded=%d encoded_fps=%.2f "
                    "sent=%d sent_fps=%.2f "
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
        finally:
            log.info(
                "ws totals rx=%d wrote=%d encoded=%d sent=%d "
                "drop_in=%d drop_out=%d bad_decode=%d",
                stats["rx"],
                stats["input_written"],
                stats["output_encoded"],
                stats["sent"],
                stats["input_overwritten"],
                stats["output_overwritten"],
                stats["decode_failed"],
            )

    tasks = [
        asyncio.create_task(receiver()),
        asyncio.create_task(input_worker()),
        asyncio.create_task(output_worker()),
        asyncio.create_task(sender()),
        asyncio.create_task(stats_loop()),
    ]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        stop_all()
        await ws.close()
        await asyncio.gather(*tasks, return_exceptions=True)

    log.info("WebSocket client disconnected")
    return ws


async def executor_context(app: web.Application):
    app["executor"] = ThreadPoolExecutor(
        max_workers=2, thread_name_prefix="fluxrt-ws"
    )
    try:
        yield
    finally:
        app["executor"].shutdown(wait=True)


async def handle_status(request: web.Request):
    app = request.app
    return web.json_response(
        {"resolution": app["runner"].resolution, "transport": "websocket-tcp"}
    )


async def handle_prompt(request: web.Request):
    """HTTP POST {"prompt": "..."} — used by the TD extension to update
    the prompt out-of-band from the frame WebSocket. (Prompts can also
    arrive as WebSocket text messages via handle_ws; both routes call
    the same runner.set_prompt.)"""
    app = request.app
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "invalid json"}, status=400)
    prompt = data.get("prompt")
    if not prompt:
        return web.json_response({"error": "missing 'prompt'"}, status=400)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(app["executor"], app["runner"].set_prompt, prompt)
    return web.json_response({"ok": True, "prompt": prompt})


def create_app(config_path: str, use_int8: bool) -> web.Application:
    app = web.Application()
    app["runner"] = FluxRTRunner(config_path, use_int8=use_int8)
    app.cleanup_ctx.append(executor_context)
    app.router.add_get("/ws", handle_ws)
    app.router.add_get("/status", handle_status)
    app.router.add_post("/prompt", handle_prompt)
    return app


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
    args = parser.parse_args()

    app = create_app(args.config, args.int8)
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
