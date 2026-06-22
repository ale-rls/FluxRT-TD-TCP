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
stable show URL; use the tunnel when its measured round trip beats the normal
Modal endpoint enough to justify the temporary URL.

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
  round-trip frame latency — compare the deployed `modal.run` URL with the
  optional `modal.host` tunnel before committing to it for a show. A pre-warmed
  `min_containers=1` container removes cold-start variance but does not change
  network round-trip time.
- **WebSocket lifetime.** A connection is capped by the function `timeout` (set
  to 6 h). If it's recycled, the relay auto-reconnects within ~1 s.
- **Tuning** (`interpolation_exp`, `target_fps`, `resolution`) lives in
  `configs/stream_processor_config.json` inside the FluxRT clone. To override it,
  edit a local copy and add it to the image (`.add_local_file(...)` in
  `modal_app.py`), then redeploy.
- **No HF token needed** — all FluxRT model repos are public. If one ever goes
  gated, add a `modal.Secret` with `HF_TOKEN` and pass it to `download_weights`.
- **CUDA image tag.** If `nvidia/cuda:12.8.0-devel-ubuntu22.04` fails to pull,
  substitute any valid CUDA 12.8.x devel tag for Ubuntu 22.04.
