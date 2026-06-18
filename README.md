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

- **`OUTPUT_FPS`** (server) and **`SEND_FPS`** (relay) — steady in/out rates,
  default 25. Keep them aligned with each other and roughly with RIFE's output
  rate to avoid resampling stutter.
- **`JPEG_QUALITY`** — input quality can go lower (the model transforms it
  anyway); keep output quality higher since that's what's displayed.

---