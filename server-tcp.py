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

Test against it with test_latency_client.py (sends a local video file or
webcam frames, measures round-trip per frame, prints stats).
"""

import argparse
import asyncio
import logging
import threading
import time

import cv2
import numpy as np
from aiohttp import web, WSMsgType

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("fluxrt-tcp-server")

JPEG_QUALITY = 70  # cv2 expects 0-100, matches the 0.7 quality factor
                   # the original Daydream TD extension used for its own
                   # local MJPEG-over-WebSocket relay (JPEG_QUALITY_STREAM)


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


async def handle_ws(request: web.Request):
    app = request.app
    runner: FluxRTRunner = app["runner"]

    ws = web.WebSocketResponse(max_msg_size=20 * 1024 * 1024)  # 20MB, generous
                                                                # headroom over
                                                                # any single
                                                                # JPEG frame
    await ws.prepare(request)
    log.info("WebSocket client connected")

    loop = asyncio.get_event_loop()

    # Output send rate. The model produces ~8fps natively but RIFE fills
    # in between, so the output tensor refreshes faster than that; we
    # send at a steady cadence regardless of input rate. 25fps matches
    # the demo's cv2.waitKey(1000//25) display pacing.
    OUTPUT_FPS = 25
    output_interval = 1.0 / OUTPUT_FPS

    stop_event = asyncio.Event()

    async def output_loop():
        """Independent of input: on a steady timer, read whatever's
        currently in the output tensor and send it. Never waits for or
        corresponds to any particular input frame."""
        while not stop_event.is_set():
            t0 = time.time()
            try:
                processed = await loop.run_in_executor(None, runner.read_output)
                ok, encoded = cv2.imencode(
                    ".jpg", processed, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
                )
                if ok:
                    await ws.send_bytes(encoded.tobytes())
            except Exception as e:
                log.exception(f"output_loop error: {e}")
                break
            # Pace to OUTPUT_FPS, accounting for time already spent.
            elapsed = time.time() - t0
            await asyncio.sleep(max(0, output_interval - elapsed))

    output_task = asyncio.ensure_future(output_loop())

    try:
        # Receive loop: write each incoming frame straight to input
        # (latest-wins). Does NOT send anything back — output is the
        # other loop's job entirely.
        async for msg in ws:
            if msg.type == WSMsgType.BINARY:
                np_arr = np.frombuffer(msg.data, dtype=np.uint8)
                frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if frame_bgr is None:
                    continue
                try:
                    await loop.run_in_executor(None, runner.write_input, frame_bgr)
                except Exception as e:
                    log.exception(f"write_input error: {e}")
                    continue

            elif msg.type == WSMsgType.TEXT:
                import json
                try:
                    data = json.loads(msg.data)
                    if "prompt" in data:
                        runner.set_prompt(data["prompt"])
                        await ws.send_str(json.dumps({"ok": True}))
                except json.JSONDecodeError:
                    log.warning(f"Ignoring non-JSON text message: {msg.data!r}")

            elif msg.type == WSMsgType.ERROR:
                log.error(f"WebSocket closed with exception {ws.exception()}")
    finally:
        stop_event.set()
        await output_task

    log.info("WebSocket client disconnected")
    return ws


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
    app["runner"].set_prompt(prompt)
    return web.json_response({"ok": True, "prompt": prompt})


def create_app(config_path: str, use_int8: bool) -> web.Application:
    app = web.Application()
    app["runner"] = FluxRTRunner(config_path, use_int8=use_int8)
    app.router.add_get("/ws", handle_ws)
    app.router.add_get("/status", handle_status)
    app.router.add_post("/prompt", handle_prompt)
    return app
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
