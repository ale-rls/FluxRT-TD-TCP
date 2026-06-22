"""
modal_app.py — deploy the FluxRT TCP/WebSocket backend on Modal.

Alternative to the RunPod Pod path (setup.sh + a long-lived VM running
server-tcp.py). Instead of provisioning a bare GPU box and exposing a TCP
port, this packages the *exact same* server-tcp.py into a Modal container,
runs it on a warm GPU, and lets Modal serve it over public HTTPS/WSS.

Nothing about the server changes: server-tcp.py runs verbatim, the two
decoupled loops and FluxRT's shared-memory model are untouched. Only the
*deployment* changes — VM -> Modal container, raw ws://host:port ->
wss://...modal.run/ws (Modal terminates TLS for you), and the RunPod
"expose port 8080 / SSH tunnel" step disappears.

Why this maps cleanly onto Modal:
  - aiohttp isn't ASGI, so we use @modal.web_server(PORT): Modal runs an
    arbitrary server process in the container and proxies HTTPS/WSS to that
    port. WebSockets are supported on web_server endpoints.
  - FluxRT model weights are multi-GB, so we keep them in a modal.Volume:
    downloaded ONCE (download_weights) and reused on every cold start
    instead of re-pulled on each boot. At serve() time we symlink them into
    FluxRT's expected on-disk layout (FluxRT/<model-dir>).
  - A WebSocket connection counts as a single Modal "input", so we set
    @modal.concurrent to let the long-lived frame socket and the /status and
    /prompt HTTP calls share one container instead of fanning out.

Usage (see MODAL.md for the full walkthrough). Run from the repo root:
    pip install modal && modal token new
    modal run   modal_app.py::download_weights     # one-time, fills the Volume
    modal serve modal_app.py                        # dev: warm while running
    modal deploy modal_app.py                       # persistent routed endpoint
    modal run   modal_app.py::serve_tunnel          # optional direct tunnel
Then point TouchDesigner's `Serverurl` at  wss://<...>.modal.run/ws .
"""

import modal

# --- What to run on -----------------------------------------------------
# Chosen tier: A100-80GB, full precision (see MODAL.md). The A100's large
# HBM2e memory bandwidth helps diffusion per-frame latency, and 80GB fits
# the full FLUX.2-klein-4B comfortably. Swap GPU / flip USE_INT8 to change:
#   "L40S"        full   ~$1.95/hr   Ada, 48GB
#   "A100-80GB"   full   ~$2.50/hr   <- current
#   "H100"        full   ~$3.95/hr   Hopper, fastest
#   "A10"         int8   ~$1.10/hr   budget (set USE_INT8 = True)
GPU = "A100-80GB"
USE_INT8 = False

# Server-side work tuning. "default" preserves the original 25fps output loop
# and uncapped latest-wins input writes. "light" and "low" reduce server
# decode/crop/read/encode work for benchmark runs before opening TouchDesigner.
SERVER_WORK_PRESET = "default"
SERVER_OUTPUT_FPS = None
SERVER_INPUT_FPS = None

# Region for the GPU container. A narrow region has Modal's higher regional
# pricing multiplier but avoids a US GPU hop for Berlin/EU clients. Set to None
# to let Modal choose the cheapest/most available region.
GPU_REGION = "eu-west"

# Region for Modal Function input/output routing. Modal defaults this to
# us-east; eu-west keeps the public WebSocket edge in Europe. Modal currently
# allows routing_region changes only on a newly-created Function, so if an
# existing deployment rejects this, deploy under a new app/function name.
WEB_ROUTING_REGION = "eu-west"

APP_NAME = "fluxrt-tcp"
PORT = 8080

FLUXRT_DIR = "/root/FluxRT"
WEIGHTS_DIR = "/weights"

# FluxRT weight repos (all public on HuggingFace — no HF token needed).
# Cloned once into the Volume by download_weights(), then symlinked into the
# FluxRT working dir at serve() time so FluxRT sees its expected layout.
MODELS = {
    "FLUX.2-klein-4B": "https://huggingface.co/black-forest-labs/FLUX.2-klein-4B",
    "RIFE-safetensors": "https://huggingface.co/TensorForger/RIFE-safetensors",
}
if USE_INT8:
    MODELS["FLUX.2-klein-4B-int8"] = "https://huggingface.co/aydin99/FLUX.2-klein-4B-int8"

# --- Container image ----------------------------------------------------
# Mirrors setup.sh / FluxRT's README (Python 3.12, CUDA 12.8, torch cu128),
# but built as a reproducible Modal image instead of a conda env on a bare
# VM. CUDA *devel* base so any FluxRT CUDA extensions can compile.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install(
        "git", "git-lfs", "ffmpeg", "libgl1", "libglib2.0-0",
        "build-essential", "wget", "ca-certificates",
    )
    .run_commands("git lfs install")
    # torch first (heavy, rarely-changing layer), built against CUDA 12.8.
    .pip_install(
        "torch", "torchvision",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    # clone + install FluxRT itself. Keep the cu128 index here too, so if
    # requirements.txt re-pins torch it still resolves to the CUDA build.
    .run_commands(
        f"git clone https://github.com/tensorforger/FluxRT {FLUXRT_DIR}",
        f"cd {FLUXRT_DIR} && pip install --extra-index-url "
        "https://download.pytorch.org/whl/cu128 -r requirements.txt",
        f"cd {FLUXRT_DIR} && pip install -e .",
    )
    # server-tcp.py's own deps (FluxRT may already pull these; harmless).
    .pip_install("aiohttp", "opencv-python-headless", "numpy")
    # The WebSocket server, copied straight from this repo. Added last and
    # mounted at runtime so editing it doesn't rebuild the heavy layers.
    .add_local_file("server-tcp.py", f"{FLUXRT_DIR}/server-tcp.py")
)

app = modal.App(APP_NAME)

# Persistent store for the multi-GB model weights.
weights = modal.Volume.from_name("fluxrt-weights", create_if_missing=True)


@app.function(image=image, volumes={WEIGHTS_DIR: weights}, timeout=60 * 60 * 4)
def download_weights():
    """One-time: clone the FluxRT model repos into the Volume.

        modal run modal_app.py::download_weights

    Safe to re-run — existing model dirs are skipped. Re-run after flipping
    USE_INT8 to pull the int8 weights too.
    """
    import os
    import subprocess

    subprocess.run(["git", "lfs", "install"], check=True)
    for name, url in MODELS.items():
        dest = os.path.join(WEIGHTS_DIR, name)
        if os.path.isdir(dest):
            print(f"[weights] {name} already present, skipping")
            continue
        print(f"[weights] cloning {name} ...")
        subprocess.run(["git", "clone", url, dest], check=True)
    weights.commit()
    print("[weights] done")


@app.function(
    image=image,
    gpu=GPU,
    region=GPU_REGION,
    routing_region=WEB_ROUTING_REGION,
    volumes={WEIGHTS_DIR: weights},
    # A WS connection is one input that lives for the whole show; give it
    # plenty of room. The relay auto-reconnects if a connection is recycled.
    timeout=60 * 60 * 6,
    # Stay warm for 20 min (Modal's max) after the last client disconnects,
    # so a soundcheck -> show gap doesn't trigger a cold reload.
    scaledown_window=60 * 20,
    # 0 = scale to zero when idle (no standing cost). Set to 1 before a live
    # show to keep a GPU pre-warmed and avoid a cold start mid-performance.
    min_containers=0,
    # Hard ceiling: never run more than ONE A100 container, no matter how many
    # inputs/connections arrive. With max_inputs=100 above, a single container
    # already serves everything; this guarantees a runaway can't fan out into a
    # fleet of GPUs (and a fleet of bills).
    max_containers=1,
)
@modal.concurrent(max_inputs=100)  # one WS = one input; let WS + HTTP coexist
@modal.web_server(PORT, startup_timeout=60 * 10)  # model load can take minutes
def serve():
    """Launch server-tcp.py inside the container; Modal proxies HTTPS/WSS to
    PORT. Non-blocking (Popen) so Modal can detect the port once FluxRT has
    finished warming up."""
    _start_fluxrt_server()


@app.function(
    image=image,
    gpu=GPU,
    region=GPU_REGION,
    volumes={WEIGHTS_DIR: weights},
    timeout=60 * 60 * 6,
    scaledown_window=60 * 20,
    max_containers=1,
)
def serve_tunnel():
    """Launch server-tcp.py behind a Modal Tunnel for a direct low-latency
    rehearsal path. Run with:

        modal run modal_app.py::serve_tunnel

    Copy the printed wss://.../ws URL into TouchDesigner. The tunnel exists
    only while this function is running.
    """
    with modal.forward(PORT) as tunnel:
        ws_url = tunnel.url.replace("https://", "wss://", 1) + "/ws"
        print(f"[tunnel] HTTPS status/prompt base: {tunnel.url}")
        print(f"[tunnel] TouchDesigner Serverurl: {ws_url}")
        process = _start_fluxrt_server()
        process.wait()


def _start_fluxrt_server():
    """Symlink weights and start the FluxRT aiohttp server."""
    import os
    import subprocess

    # Make the persisted weights look like a local `git clone` inside FluxRT.
    for name in MODELS:
        src = os.path.join(WEIGHTS_DIR, name)
        dst = os.path.join(FLUXRT_DIR, name)
        if os.path.isdir(src) and not os.path.lexists(dst):
            os.symlink(src, dst)

    cmd = [
        "python", "server-tcp.py",
        "--config", "configs/stream_processor_config.json",
        "--port", str(PORT),
        "--work-preset", SERVER_WORK_PRESET,
    ]
    if SERVER_OUTPUT_FPS is not None:
        cmd.extend(["--output-fps", str(SERVER_OUTPUT_FPS)])
    if SERVER_INPUT_FPS is not None:
        cmd.extend(["--input-fps", str(SERVER_INPUT_FPS)])
    if USE_INT8:
        cmd.append("--int8")
    # server-tcp.py binds 0.0.0.0 by default — required for web_server.
    return subprocess.Popen(cmd, cwd=FLUXRT_DIR)
