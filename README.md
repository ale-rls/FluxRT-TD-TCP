# FluxRT ↔ TouchDesigner (TCP/WebSocket backend)

Self-hosted real-time AI video backend for **TouchDesigner**, using
[FluxRT](https://github.com/tensorforger/FluxRT) (FLUX.2-klein-4B + RIFE
interpolation) on a rented GPU. TouchDesigner streams a live video feed to the
GPU box, the model stylizes it in real time, and the result streams back for
live display — built for live performance / theater visuals.

Frames travel over a plain **WebSocket (TCP)**, not WebRTC — see
[Why WebSocket and not WHIP/WHEP](#why-websocket-and-not-whipwhep) below. This
makes it work on TCP-only GPU hosts (e.g. RunPod Pods) and even through an SSH /
VS Code tunnel.

> **Deploying on Modal instead of a RunPod Pod?** See **[MODAL.md](MODAL.md)** —
> the same `server-tcp.py`, packaged as a Modal app on a warm GPU and served
> over `wss://…modal.run`. The Modal config defaults to EU web routing/GPU
> placement for Berlin/Europe latency tests and also documents an optional
> temporary Modal Tunnel path.

---

## Architecture

```
┌─────────────────┐     JPEG frames over WebSocket (TCP)     ┌──────────────────┐
│  TouchDesigner  │  ────────────────────────────────────▶   │  GPU box         │
│                 │                                           │  server-tcp.py   │
│  webcam/video   │     input frames (steady ~25fps)          │       │          │
│      │          │                                           │       ▼          │
│      ▼          │                                           │   FluxRT model   │
│  relay HTML     │  ◀────────────────────────────────────   │   (~8fps native, │
│  in web_render  │     processed frames (steady ~25fps)      │    RIFE → smooth)│
└─────────────────┘                                           └──────────────────┘
```

Two **independent** loops over one WebSocket (this matters — see
[Frame cadence](#the-frame-cadence-lesson)):

- TouchDesigner sends input frames at a steady rate; the server writes each
  straight into FluxRT's input tensor (latest-wins).
- The server, on its own timer, reads FluxRT's output tensor and sends it back.

They never wait for each other, mirroring FluxRT's shared-memory design.

---

## Components

| File | Runs on | Purpose |
|------|---------|---------|
| `setup.sh` | GPU box | One-shot VM setup (miniconda, git-lfs, clone + install FluxRT) |
| `server-tcp.py` | GPU box | WebSocket server wrapping FluxRT's `StreamProcessor` |
| `benchmark_ws.py` | Laptop / operator machine | Pre-TouchDesigner websocket benchmark client |
| `td_fluxrt_ext_tcp.py` | TouchDesigner | Extension: lifecycle, serves the relay page, streams input frames |
| `fluxrt_tcp_relay.html` | TouchDesigner (in `web_render`) | Bridges local input frames ↔ remote server, displays output |

---

## Setup

### 1. GPU box

```bash
bash setup.sh
# then restart shell / source ~/.bashrc
# download FluxRT model weights per FluxRT's own README (FLUX.2-klein-4B, RIFE)
```

Copy `server-tcp.py` into the `FluxRT/` directory and run it:

```bash
python server-tcp.py --config configs/stream_processor_config.json --port 8080
# add --int8 to use the quantized model
```

Wait for `FluxRT ready` and `Running on http://0.0.0.0:8080`.

### 2. Expose the port

The server listens on `0.0.0.0:8080`, but the GPU host must also expose it
externally. Options:

- **Production:** create the pod with port 8080 exposed (RunPod's TCP port
  exposure), giving you a public `host:port`.
- **Testing:** SSH tunnel from your TD machine —
  `ssh -p <ssh-port> root@<host> -L 8080:localhost:8080 -i ~/.ssh/<key>` — then
  the server is reachable at `127.0.0.1:8080` locally.

Verify before touching TouchDesigner:

```bash
curl http://127.0.0.1:8080/status   # should return JSON
```

### 3. TouchDesigner

1. Create a Base COMP. Inside it, add child operators named:
   `stream_source` (TOP, your input feed), `web_server` (Web Server DAT),
   `web_render` (Web Render TOP), `frame_timer` (Timer CHOP), and a CHOP Execute
   DAT watching `frame_timer`.
2. Paste `td_fluxrt_ext_tcp.py` into a Text DAT, and paste the contents of
   `fluxrt_tcp_relay.html` into the `RELAY_HTML_TEMPLATE` string at the bottom
   of that file.
3. Set the COMP's **Extension** parameter to
   `op('./<dat_name>').module.FluxRTExt(me)` and promote it as `FluxRTExt`.
4. Wire the callback DATs (see [Wiring](#touchdesigner-wiring)).
5. Set the `Serverurl` parameter to `ws://127.0.0.1:8080/ws` (use `127.0.0.1`,
   **not** `localhost` — see gotchas).
6. Flip **Active** on.

---

## TouchDesigner wiring

The extension does nothing on its own — TD operators must forward events to it.

**`web_server` Callbacks DAT** forwards HTTP + WebSocket events:

```python
def onHTTPRequest(webServerDAT, request, response):
    parent().ext.FluxRTExt.OnHTTPRequest(request, response, 'frame')
    return response
def onWebSocketOpen(webServerDAT, client, uri):
    parent().ext.FluxRTExt.OnWebSocketOpen(client, uri)
def onWebSocketClose(webServerDAT, client):
    parent().ext.FluxRTExt.OnWebSocketClose(client)
```

**CHOP Execute DAT** (watching `frame_timer`, `While On` enabled) drives the
per-frame input send:

```python
def whileOn(channel, sampleIndex, val, prev):
    parent().ext.FluxRTExt.OnTimerPulse()
    return
```

**Parameter Execute DAT** (`Custom` + `Value Change` enabled) forwards
parameter changes:

```python
def onValueChange(par, prev):
    parent().ext.FluxRTExt.OnParameterChange(par)
    return
```

---

## Tuning

In `configs/stream_processor_config.json`:

- **`interpolation_exp`** — RIFE interpolation strength. `2` means 4× frame
  multiplication; high values look smooth in still moments but cause warping
  distortion during motion. **`1` (2×) is a good default** for content with
  movement. *(Requires server restart.)*
- **`target_fps`** — set explicitly (e.g. `25`) to even out interpolated
  playback pacing.
- **`resolution`** — higher costs more per frame; the model is the bottleneck.

In `server-tcp.py` / `fluxrt_tcp_relay.html`:

- **Server work preset** — start the server with
  `--work-preset default|light|low`, or set `FLUXRT_WORK_PRESET`. `default`
  preserves the original 25fps output loop and uncapped latest-wins input
  writes. `light` caps server output reads/encodes and input decode/crop/copy
  work to 15fps; `low` caps both to 10fps.
- **`--output-fps` / `FLUXRT_OUTPUT_FPS`** — overrides the preset's server
  output tensor read/JPEG send cadence.
- **`--input-fps` / `FLUXRT_INPUT_FPS`** — overrides the preset's FluxRT input
  tensor write cadence. `0` keeps the original uncapped latest-wins input
  behavior.
- **`--output-jpeg-quality` / `FLUXRT_OUTPUT_JPEG_QUALITY`** — server output
  JPEG quality, `1-100`, default `70`. Lower values reduce OpenCV encode CPU
  and websocket bytes at the cost of display quality.
- **`SEND_FPS`** (relay) — TouchDesigner relay input send rate, default 25.
  For a tuned show run, keep it close to the server input/output caps you chose
  to avoid oversending frames that will be dropped.
- **`Input JPEG Quality`** (TouchDesigner extension) — local
  `stream_source.saveByteArray('.jpg')` quality, default `0.7`. Input quality
  can usually go lower than output quality because the model transforms it
  anyway; watch `input_decode`, `rx` average KB, and visual stability.

## Runtime stats

`server-tcp.py` logs one WebSocket summary every 5 seconds, with frame counts,
measured FPS/cadence, latest-wins drops, bad JPEG decodes, and hot-path
timings:

```
ws stats/5s window=...s rx=... rx_fps=... wrote=... wrote_fps=...
  encoded=... encoded_fps=... sent=... sent_fps=...
  avg_kb rx=... encoded=... sent=...
  drop_in=... drop_out=... bad_decode=...
  hot_ms input_decode=mean/p95/count
  input_crop_copy=mean/p95/count output_read=mean/p95/count
  output_encode=mean/p95/count send=mean/p95/count
```

For benchmarking, compare `rx_fps` with `wrote_fps` to see receive pressure
versus accepted FluxRT input cadence. Compare `encoded_fps` with `sent_fps`
plus `drop_out` to spot network backpressure, and compare `avg_kb` with
`output_encode` and `send` timings during JPEG quality sweeps. The `hot_ms`
fields are `mean/p95/count` latencies in milliseconds for the
server-observable stages: input JPEG decode, input crop/copy into FluxRT,
output tensor read, JPEG encode, and websocket send wait.

## Pre-TouchDesigner benchmark

Before opening TouchDesigner, run a real backend and drive the same websocket
API with synthetic JPEG frames:

```bash
python3 benchmark_ws.py ws://127.0.0.1:8080/ws \
  --benchmark-preset baseline --duration 30
```

Then restart the server with a lower-work preset and run the matching client
preset:

```bash
python server-tcp.py --config configs/stream_processor_config.json \
  --port 8080 --work-preset light

python3 benchmark_ws.py ws://127.0.0.1:8080/ws \
  --benchmark-preset light --duration 30
```

For Modal, use the `wss://.../ws` URL printed by `modal serve`, `modal deploy`,
or the optional tunnel. The benchmark script itself does not need FluxRT weights
locally; weights are only needed by the server it connects to. It uses `aiohttp`
for websocket transport and OpenCV+NumPy, or Pillow, to generate JPEG frames.
The `baseline`, `light`, and `low` benchmark presets all keep the real
TouchDesigner relay geometry, 512x512. To test smaller input frames as a
separate JPEG/network/decode experiment, pass explicit `--width` and `--height`
overrides and label that run separately from cadence tuning.

Client output reports elapsed time, active send/receive windows, frames
sent/received, send/receive FPS for the active send window, byte totals,
average frame bytes, the simple `sent_minus_received` count delta, and
`latest_send_age_ms` mean/p50/p95/max. Connection setup and post-send receive
drain do not dilute the headline FPS numbers. Because the server protocol
intentionally does not tag frames and FluxRT input/output loops are decoupled,
that latency is the age since the latest client send when a binary response
arrived, not a strict per-input model round trip. Pair it with the server's
`ws stats/5s` `hot_ms` fields to separate network pressure, server
CPU/shared-memory costs, and FluxRT output cadence across runs. Compare
baseline vs. `light` before opening TouchDesigner:

- Client: `send_fps`, `receive_fps`, and `latest_send_age_ms` p50/p95.
- Server: `rx_fps` vs. `wrote_fps`, `encoded_fps` vs. `sent_fps`, `drop_in`,
  `drop_out`, `avg_kb`, and the `hot_ms` stage means/p95s.

To measure JPEG CPU/quality tradeoffs, keep cadence fixed and run a small
quality sweep. Example:

```bash
python server-tcp.py --config configs/stream_processor_config.json \
  --port 8080 --work-preset light --output-jpeg-quality 70
python3 benchmark_ws.py ws://127.0.0.1:8080/ws \
  --benchmark-preset light --quality 70 --duration 30

python server-tcp.py --config configs/stream_processor_config.json \
  --port 8080 --work-preset light --output-jpeg-quality 55
python3 benchmark_ws.py ws://127.0.0.1:8080/ws \
  --benchmark-preset light --quality 55 --duration 30
```

For each run, record the benchmark's `input_jpeg`, average sent/received frame
bytes, `receive_fps`, and `latest_send_age_ms` p95, then pair it with server
`avg_kb`, `input_decode`, `output_encode`, `sent_fps`, and `send` p95. In
TouchDesigner, match the client-side part of the run by setting **Input JPEG
Quality** to the same `quality / 100` value, for example `0.55`.

---
