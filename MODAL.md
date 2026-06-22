# Deploying on Modal

A drop-in alternative to the RunPod Pod path. The GPU backend is identical —
the same `server-tcp.py`, the same FluxRT `StreamProcessor`, the same two
decoupled frame loops — but instead of provisioning a VM and exposing a TCP
port, it runs as a [Modal](https://modal.com) app on a warm GPU, served over
public **HTTPS/WSS**.

What changes vs. RunPod:

| | RunPod Pod | Modal |
|---|---|---|
| Provisioning | `setup.sh` on a bare VM | `modal_app.py` builds the image |
| Weights | clone into the Pod each time | cloned once into a Modal **Volume** |
| Serving | `python server-tcp.py` + expose port 8080 | `@modal.web_server` proxies WSS, or optional Modal Tunnel |
| Client URL | `ws://host:port/ws` (often via SSH tunnel) | `wss://…modal.run/ws` (TLS, no tunnel) |
| Idle cost | pay while the Pod exists | scale to zero; pay per-second while warm |

**GPU: A100-80GB, full precision** — the closest stable Modal analog to the
RTX 5090 that worked best on RunPod (big HBM2e bandwidth helps per-frame
latency; 80 GB fits the full model). Change it via the `GPU` / `USE_INT8`
constants at the top of `modal_app.py`.

**Low-latency EU default:** `modal_app.py` sets both the GPU container region
and Modal Web Function routing region to `eu-west` for Berlin/Europe clients:

- `GPU_REGION = "eu-west"` keeps the FluxRT container in Europe.
- `WEB_ROUTING_REGION = "eu-west"` keeps Modal web endpoint traffic out of the
  default `us-east` routing path.

Set `GPU_REGION = None` to let Modal choose the cheapest/most available GPU
region. Keep `WEB_ROUTING_REGION = "eu-west"` for the normal EU web endpoint
unless measurements say otherwise. Regional containers have Modal's regional
pricing multiplier; routing-region support is currently a Modal beta.

## Prerequisites

```bash
pip install modal
modal token new        # one-time browser auth
```

Run all `modal` commands **from the repo root**, so `modal_app.py` can find
`server-tcp.py`.

## 1. Download the weights (one-time)

Pulls the public FluxRT model repos into a persistent Modal Volume named
`fluxrt-weights`, so cold starts don't re-download multi-GB weights:

```bash
modal run modal_app.py::download_weights
```

Re-running is safe (existing models are skipped). Re-run after setting
`USE_INT8 = True` to also fetch the int8 weights.

## 2. Run it: persistent routed endpoint

**Dev / rehearsal** — ephemeral, stays warm while the command runs, hot-reloads
on edit:

```bash
modal serve modal_app.py
```

**Production** — persistent endpoint that survives after you disconnect:

```bash
modal deploy modal_app.py
```

Either prints a URL like:

```
https://<workspace>--fluxrt-tcp-serve.modal.run
```

The first request triggers a cold start: container boot + model load, ~1–3 min.
Watch the logs for FluxRT's `ready` line. Verify before touching TouchDesigner:

```bash
curl https://<workspace>--fluxrt-tcp-serve.modal.run/status   # -> JSON
```

Before opening TouchDesigner, drive the websocket endpoint with synthetic JPEG
frames from your laptop:

```bash
python3 benchmark_ws.py wss://<workspace>--fluxrt-tcp-serve.modal.run/ws \
  --benchmark-preset baseline --duration 30
```

The script reports client-observed send/receive FPS over the active send
window, plus `latest_send_age_ms` mean/p50/p95/max. Connection setup and
post-send receive drain are shown in elapsed time but do not dilute the headline
FPS. Since the server protocol does not tag frames and FluxRT input/output loops
are decoupled, this is not a strict per-input model RTT; compare it with the
backend's 5-second `ws stats/5s` `hot_ms` summaries to separate websocket
pressure from server decode/crop/read/encode/send costs. Run the same command
against the optional tunnel URL to compare `modal.run` and `modal.host` before
opening TouchDesigner.

For a lower-work comparison, set `SERVER_WORK_PRESET = "light"` in
`modal_app.py`, then `modal serve modal_app.py` or `modal deploy modal_app.py`
again and rerun:

```bash
python3 benchmark_ws.py wss://<workspace>--fluxrt-tcp-serve.modal.run/ws \
  --benchmark-preset light --duration 30
```

Compare each run's client `send_fps`, `receive_fps`, and `latest_send_age_ms`
p50/p95 with the server `ws stats/5s` line: `rx_fps` vs. `wrote_fps`,
`encoded_fps` vs. `sent_fps`, `drop_in`, `drop_out`, `avg_kb`, and `hot_ms`
means/p95s. Use the lighter preset for TouchDesigner only if the benchmark
improves latency or steadiness enough to justify the lower send/display
cadence. All benchmark presets keep the real TouchDesigner relay geometry,
512x512; use explicit `--width` and `--height` only for a separate
JPEG/network/decode experiment.

For JPEG CPU/byte sweeps, keep cadence fixed and vary only quality. Set
`SERVER_OUTPUT_JPEG_QUALITY = 70`, `60`, or `55` in `modal_app.py`, redeploy,
then run the benchmark with the matching input quality:

```bash
python3 benchmark_ws.py wss://<workspace>--fluxrt-tcp-serve.modal.run/ws \
  --benchmark-preset light --quality 55 --duration 30
```

The benchmark prints generated input JPEG bytes and average received output
frame bytes. Pair those with Modal logs for server `avg_kb`,
`input_decode`, `output_encode`, `sent_fps`, and `send` p95. In
TouchDesigner, set **Input JPEG Quality** to `quality / 100`, for example
`0.55`, when repeating the same run through the relay.

Modal only allows `routing_region` to be set when a Function is first created.
If you already deployed this app before the `eu-west` routing change and Modal
rejects the redeploy, create a fresh Function/App name before deploying the
routed endpoint.

## 3. Point TouchDesigner at it

Set the extension's **`Serverurl`** parameter to the **`wss://`** form of the
URL, with the `/ws` path:

```
wss://<workspace>--fluxrt-tcp-serve.modal.run/ws
```

That's the only client-side change. The relay HTML already uses whatever scheme
you give it, and the extension's prompt POST already rewrites `wss://` →
`https://` for `/prompt`. No SSH tunnel, no port exposure — Modal terminates TLS
for you. (Use `wss://`, **not** `ws://`; Modal endpoints are HTTPS-only.)

## Optional: direct Modal Tunnel

For rehearsals or latency tests, Modal Tunnels expose the same container port
through a direct TLS tunnel instead of the Web Function proxy path:

```bash
modal run modal_app.py::serve_tunnel
```

After the model starts, the logs print:

```
[tunnel] HTTPS status/prompt base: https://<random>.modal.host
[tunnel] TouchDesigner Serverurl: wss://<random>.modal.host/ws
```

Copy the printed `wss://.../ws` URL into TouchDesigner. The `/status` and
`/prompt` endpoints use the printed HTTPS base URL, and the TouchDesigner
extension derives that automatically from the WebSocket URL. This tunnel is
temporary: it exists only while `modal run modal_app.py::serve_tunnel` is
running, and the generated URL changes each time. Use `modal deploy` for a
stable show URL; use the tunnel when its measured websocket cadence and
latest-send-age timing beat the normal Modal endpoint enough to justify the
temporary URL.

## Pre-warming for a live show

By default the app **scales to zero** when idle (`min_containers=0`) — no GPU
cost between sessions, but a cold start on the next connection. For a
performance you don't want a cold start mid-show:

- Set `min_containers = 1` in `modal_app.py` and `modal deploy` before the show
  to keep one GPU hot (~$2.50/hr while it's up), then set it back to `0` and
  redeploy afterwards; **or**
- just keep `modal serve modal_app.py` running for the whole session — the
  container stays warm as long as that command is alive.

`scaledown_window` is 20 min, so brief disconnects (soundcheck → show) won't
drop the warm container.

## Notes & caveats

- **Latency.** For Berlin/Europe clients, the default Modal path now uses
  `routing_region="eu-west"` and `region="eu-west"` to avoid the previous
  default `us-east` routing path. This whole stack was built to *measure*
  websocket cadence, latest-send-age timing, and server hot-path costs —
  compare the deployed `modal.run` URL with the optional `modal.host` tunnel
  before committing to it for a show. A pre-warmed `min_containers=1` container
  removes cold-start variance but does not change network transit time.
- **Runtime stats.** The backend logs one `ws stats/5s` line per connected
  client. Use `rx_fps` vs. `wrote_fps` to compare receive pressure with
  accepted FluxRT input cadence, and `encoded_fps` vs. `sent_fps` plus
  `drop_out` to spot websocket backpressure. The `avg_kb` values show average
  incoming, encoded, and sent frame sizes for the same window. `hot_ms` fields
  are `mean/p95/count` latencies in milliseconds for `input_decode`,
  `input_crop_copy`, `output_read`, `output_encode`, and websocket `send`,
  making Modal `modal.run` vs. `modal.host` tunnel runs directly comparable.
- **WebSocket lifetime.** A connection is capped by the function `timeout` (set
  to 6 h). If it's recycled, the relay auto-reconnects within ~1 s.
- **Tuning.** Server-side cadence presets live near the top of `modal_app.py`:
  `SERVER_WORK_PRESET = "default"`, `"light"`, or `"low"`, with optional
  `SERVER_OUTPUT_FPS` and `SERVER_INPUT_FPS` overrides. Server output JPEG
  quality is `SERVER_OUTPUT_JPEG_QUALITY`, default `70`. FluxRT model tuning
  (`interpolation_exp`, `target_fps`, `resolution`) still lives in
  `configs/stream_processor_config.json` inside the FluxRT clone. To override
  FluxRT's own config, edit a local copy and add it to the image
  (`.add_local_file(...)` in `modal_app.py`), then redeploy.
- **No HF token needed** — all FluxRT model repos are public. If one ever goes
  gated, add a `modal.Secret` with `HF_TOKEN` and pass it to `download_weights`.
- **CUDA image tag.** If `nvidia/cuda:12.8.0-devel-ubuntu22.04` fails to pull,
  substitute any valid CUDA 12.8.x devel tag for Ubuntu 22.04.
