"""
td_fluxrt_ext_tcp.py

TCP/WebSocket variant of the FluxRT TD extension. Same lifecycle and
display structure as the WHIP/WHEP version (td_fluxrt_ext.py) — Active
toggle drives Start/Stop, web_server pushes input frames locally,
web_render displays the result — but the actual frame round-trip to the
GPU server goes over a plain WebSocket (handled entirely inside the
relay HTML, fluxrt_tcp_relay.html), which works on RunPod's TCP-only
networking.

What's the SAME as the WHIP/WHEP version:
  - web_server DAT pushes stream_source frames to the relay page locally
    (unchanged input mechanism)
  - web_render TOP loads the relay page and captures its output canvas
    for display (unchanged display mechanism)
  - Active -> Start/Stop, frame_timer streams input frames

What's DIFFERENT / REMOVED:
  - no web_server_sdp, no WHIP/WHEP SDP proxy, no aiortc — the relay page
    talks plain WebSocket to the FluxRT server directly
  - the relay HTML is fluxrt_tcp_relay.html, not the WebRTC one
  - prompt updates go through window.sendPrompt() in the relay page (or
    you can leave prompt handling to the relay's own socket); see
    OnParameterChange below

REQUIRED CHILD OPERATORS:
  - stream_source : TOP, the webcam/video input feed
  - web_server    : Web Server DAT, serves the relay page + streams input
                    frames over its local WebSocket (same as original)
  - web_render    : Web Render TOP, displays the relay page's output
  - frame_timer   : drives OnTimerPulse to push input frames
  - text_overlay  : optional Text TOP for status

Set the relay HTML: paste fluxrt_tcp_relay.html's contents into the
RELAY_HTML_TEMPLATE string at the bottom of this file (same pattern as
the original — kept as a separate paste step so the large HTML block
isn't error-prone to hand-edit inline).
"""

import json
import socket
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

VERSION = "0.1.0-fluxrt-tcp"

JPEG_QUALITY = 0.7  # matches JPEG_QUALITY_STREAM from the original relay

PARAM_DEFAULTS = {
    'Serverurl': 'ws://localhost:8080/ws',  # full ws:// URL to server-tcp.py
    'Prompt': 'a person standing in a forest, cinematic lighting',
    'Inputjpegquality': JPEG_QUALITY,
    'Active': False,
}


class ParameterManager:
    """Same trimmed param panel as the WHIP/WHEP version."""

    def __init__(self, owner_comp):
        self.ownerComp = owner_comp

    def _get(self, name, default=None):
        if hasattr(self.ownerComp.par, name):
            return getattr(self.ownerComp.par, name).eval()
        return default if default is not None else PARAM_DEFAULTS.get(name)

    def _get_bool(self, name, default=None):
        val = self._get(name, default)
        return bool(val) if val is not None else default

    def _get_float(self, name, default=None):
        val = self._get(name, default)
        if val is None:
            return default
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    @property
    def Serverurl(self):
        return self._get('Serverurl', PARAM_DEFAULTS['Serverurl'])

    @property
    def Prompt(self):
        return self._get('Prompt', PARAM_DEFAULTS['Prompt'])

    @property
    def Active(self):
        return self._get_bool('Active', False)

    @property
    def Inputjpegquality(self):
        value = self._get_float(
            'Inputjpegquality', PARAM_DEFAULTS['Inputjpegquality']
        )
        return min(1.0, max(0.05, value))

    def _get_page(self, name):
        for p in self.ownerComp.customPages:
            if p.name == name:
                return p
        return None

    def setup(self):
        page = self._get_page('FluxRT')
        if not page:
            self.create_all()
            return
        self._ensure_missing_params(page)

    def _ensure_missing_params(self, page):
        if not hasattr(self.ownerComp.par, 'Version'):
            p = page.appendStr('Version', label='Version')[0]
            p.default = p.val = VERSION
            p.readOnly = True
        if not hasattr(self.ownerComp.par, 'Serverurl'):
            p = page.appendStr('Serverurl', label='Server WS URL')[0]
            p.default = p.val = PARAM_DEFAULTS['Serverurl']
        if not hasattr(self.ownerComp.par, 'Prompt'):
            p = page.appendStr('Prompt', label='Prompt')[0]
            p.default = p.val = PARAM_DEFAULTS['Prompt']
        if not hasattr(self.ownerComp.par, 'Inputjpegquality'):
            p = page.appendFloat('Inputjpegquality', label='Input JPEG Quality')[0]
            p.default = p.val = PARAM_DEFAULTS['Inputjpegquality']
            p.normMin = 0.05
            p.normMax = 1.0
        if not hasattr(self.ownerComp.par, 'Active'):
            p = page.appendToggle('Active', label='Active')[0]
            p.default = p.val = False

    def create_all(self):
        page = self.ownerComp.appendCustomPage('FluxRT')
        p = page.appendStr('Version', label='Version')[0]
        p.default = p.val = VERSION
        p.readOnly = True
        page.appendHeader('Connection')
        p = page.appendStr('Serverurl', label='Server WS URL')[0]
        p.default = p.val = PARAM_DEFAULTS['Serverurl']
        page.appendHeader('Controls')
        p = page.appendStr('Prompt', label='Prompt')[0]
        p.default = p.val = PARAM_DEFAULTS['Prompt']
        p = page.appendFloat('Inputjpegquality', label='Input JPEG Quality')[0]
        p.default = p.val = PARAM_DEFAULTS['Inputjpegquality']
        p.normMin = 0.05
        p.normMax = 1.0
        p = page.appendToggle('Active', label='Active')[0]
        p.default = p.val = False

    def update_states(self, connected):
        par = self.ownerComp.par
        for par_name in ['Prompt', 'Active', 'Serverurl', 'Inputjpegquality']:
            if hasattr(par, par_name):
                getattr(par, par_name).enable = True

    def setup_param_exec(self):
        param_exec = self.ownerComp.op('param_exec')
        if param_exec and hasattr(param_exec.par, 'pars'):
            param_exec.par.pars = 'Active Prompt Serverurl Inputjpegquality'


class FrameServer:
    """Serves the relay HTML and streams stream_source frames to it over
    a local WebSocket — same role as the original web_server handler, but
    trimmed (no SDP proxy routes, since there's no WebRTC here)."""

    def __init__(self, ext):
        self.ext = ext

    def handle(self, request, response, server_type='frame'):
        path = request.get('uri', '/').split('?')[0]
        method = request.get('method', 'GET')
        if path == '/relay.html' and method == 'GET':
            response['statusCode'] = 200
            response['statusReason'] = 'OK'
            response['content-type'] = 'text/html; charset=utf-8'
            response['data'] = self.ext._get_relay_html()
        else:
            response['statusCode'] = 404
            response['data'] = b'Not Found'


class FluxRTExt:
    """TCP/WebSocket variant. Lifecycle identical to the WHIP/WHEP
    version; transport handled by the relay HTML."""

    def __init__(self, ownerComp):
        self.ownerComp = ownerComp
        self.params = ParameterManager(ownerComp)
        self.frame_server = FrameServer(self)

        self.state = "IDLE"
        self.mjpeg_port = self._allocate_port()
        self.ws_clients = set()
        self._ws_lock = threading.Lock()
        self._relay_html_cache = None

        self._input_frame_count = 0
        self._input_fps = 0.0
        self._fps_last_calc_time = time.time()

        self.params.setup()
        if hasattr(self.ownerComp.par, 'Active'):
            self.ownerComp.par.Active.val = False
        self.params.update_states(True)
        self.params.setup_param_exec()
        self._startServers()
        self._warmupWebRender()

        print(f"FluxRTExt (TCP/WebSocket relay) v{VERSION} initialized")

    def _allocate_port(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
        s.close()
        return port

    def _set_state(self, new_state):
        self.state = new_state

    def Start(self):
        stream_source = self.ownerComp.op('stream_source')
        if not stream_source or stream_source.width == 0 or stream_source.height == 0:
            print("FluxRT Error: No input connected to stream_source.")
            self._set_state("ERROR")
            return

        web_render = self.ownerComp.op('web_render')
        if web_render:
            web_render.par.url = f"http://localhost:{self.mjpeg_port}/relay.html"

        frame_timer = self.ownerComp.op('frame_timer')
        if frame_timer:
            frame_timer.par.active = 1

        self._set_state("STREAMING")
        self.UpdateStatusText("Streaming")

        # Apply the current prompt now that we're streaming. Without this the
        # initial prompt is never sent (OnParameterChange only fires on a
        # *change*), so the model runs with its built-in default until you edit
        # the field. NOTE: live prompt edits still require the Parameter Execute
        # DAT wired to OnParameterChange (see README wiring).
        self._send_prompt(self.params.Prompt)

    def Stop(self):
        frame_timer = self.ownerComp.op('frame_timer')
        if frame_timer:
            frame_timer.par.active = 0
        web_render = self.ownerComp.op('web_render')
        if web_render:
            web_render.par.url = 'about:blank'
        with self._ws_lock:
            self.ws_clients.clear()
        self._input_fps = 0.0
        self._input_frame_count = 0
        self._set_state("IDLE")
        self.UpdateStatusText("Idle")

    def _warmupWebRender(self):
        web_render = self.ownerComp.op('web_render')
        if web_render:
            web_render.par.url = 'about:blank'
            web_render.par.active = 1

    def _startServers(self):
        web_server = self.ownerComp.op('web_server')
        if web_server:
            web_server.par.active = 0
            web_server.par.port = self.mjpeg_port
            web_server.par.active = 1
            print(f"FluxRT: Frame server started on port {self.mjpeg_port}")
        else:
            print("FluxRT Error: web_server DAT not found")

    def UpdateStatusText(self, text):
        text_op = self.ownerComp.op('text_overlay')
        if text_op:
            text_op.par.text = f"FluxRT (TCP)\n{text}"

    def OnParameterChange(self, par):
        if par.name == "Active":
            if par.eval():
                self.Start()
            else:
                self.Stop()
        elif par.name == "Prompt":
            if self.state == "STREAMING":
                self._send_prompt(par.eval())
        elif par.name == "Serverurl":
            self.InvalidateRelayCache()
            if self.state == "STREAMING":
                web_render = self.ownerComp.op('web_render')
                if web_render:
                    web_render.par.url = 'about:blank'
                    web_render.par.url = f"http://localhost:{self.mjpeg_port}/relay.html"
                self._send_prompt(self.params.Prompt)

    def _normalized_urls(self, server_url):
        """Return (ws_url, prompt_url) from either a WSS URL or HTTPS base."""
        raw = (server_url or '').strip()
        if not raw:
            raw = PARAM_DEFAULTS['Serverurl']
        if '://' not in raw:
            raw = 'wss://' + raw

        parsed = urllib.parse.urlparse(raw)
        scheme = parsed.scheme.lower()
        if scheme in ('http', 'https'):
            ws_scheme = 'wss' if scheme == 'https' else 'ws'
            http_scheme = scheme
        elif scheme in ('ws', 'wss'):
            ws_scheme = scheme
            http_scheme = 'https' if scheme == 'wss' else 'http'
        else:
            ws_scheme = 'wss'
            http_scheme = 'https'

        path = parsed.path.rstrip('/')
        if not path:
            ws_path = '/ws'
        elif path.endswith('/ws'):
            ws_path = path
        elif path.endswith('/prompt') or path.endswith('/status'):
            ws_path = path.rsplit('/', 1)[0] + '/ws'
        else:
            ws_path = path + '/ws'

        base_path = ws_path.rsplit('/ws', 1)[0] or ''
        ws_url = urllib.parse.urlunparse(
            (ws_scheme, parsed.netloc, ws_path, '', parsed.query, '')
        )
        prompt_url = urllib.parse.urlunparse(
            (http_scheme, parsed.netloc, base_path + '/prompt', '', '', '')
        )
        return ws_url, prompt_url

    def _send_prompt(self, prompt):
        """POST the prompt directly to the server's /prompt endpoint.
        Sidesteps the Web Render TOP entirely (this build has no
        Javascript parameter to inject window.sendPrompt() through).

        IMPORTANT: TD's operator/parameter API is NOT thread-safe, so we
        resolve Serverurl here on the main thread and pass the plain
        string into the worker. The thread does only the urllib call,
        which is safe off-thread. (Reading self.params.Serverurl inside
        the thread raises a cryptic SystemError.)"""
        import threading

        server_url = self.params.Serverurl  # main-thread read
        _ws_url, prompt_url = self._normalized_urls(server_url)

        def _post():
            import ssl
            import urllib.error
            import urllib.request
            import urllib.parse
            import json as _json
            try:
                req = urllib.request.Request(
                    prompt_url,
                    data=_json.dumps({'prompt': prompt}).encode(),
                    headers={'Content-Type': 'application/json'},
                    method='POST',
                )
                try:
                    urllib.request.urlopen(req, timeout=30)
                except urllib.error.URLError as cert_error:
                    # Retry unverified ONLY for certificate verification
                    # failures against Modal hosts (TD's bundled Python
                    # can lack their CA chain). Any other failure — HTTP
                    # 4xx/5xx, timeout, connection refused — must
                    # propagate instead of triggering a duplicate,
                    # TLS-unverified POST of the same prompt.
                    parsed = urllib.parse.urlparse(prompt_url)
                    modal_host = parsed.hostname and (
                        parsed.hostname.endswith('.modal.run') or
                        parsed.hostname.endswith('.modal.host')
                    )
                    cert_failure = isinstance(
                        getattr(cert_error, 'reason', None),
                        ssl.SSLCertVerificationError,
                    )
                    if (
                        parsed.scheme != 'https'
                        or not modal_host
                        or not cert_failure
                    ):
                        raise
                    urllib.request.urlopen(
                        req, timeout=30,
                        context=ssl._create_unverified_context(),
                    )
                    print(
                        "FluxRT: prompt HTTPS certificate verification "
                        f"failed in TD, retried unverified: {cert_error}"
                    )
                print(f"FluxRT: prompt updated -> {prompt!r}")
            except Exception as e:
                print(f"FluxRT: prompt update failed: {e}")

        threading.Thread(target=_post, daemon=True).start()

    def OnTimerPulse(self):
        # Push the current input frame to any connected relay page(s) over
        # the local WebSocket — same mechanism as the original relay's
        # input path.
        self._input_frame_count += 1
        now = time.time()
        elapsed = now - self._fps_last_calc_time
        if elapsed >= 1.0:
            self._input_fps = self._input_frame_count / elapsed
            self._input_frame_count = 0
            self._fps_last_calc_time = now

        with self._ws_lock:
            if self.state != "STREAMING" or not self.ws_clients:
                return
            clients = list(self.ws_clients)

        stream_source = self.ownerComp.op('stream_source')
        web_server = self.ownerComp.op('web_server')
        if not stream_source or not web_server:
            return

        try:
            jpeg = stream_source.saveByteArray(
                '.jpg', quality=self.params.Inputjpegquality
            )
            dead = []
            for c in clients:
                try:
                    web_server.webSocketSendBinary(c, jpeg)
                except Exception:
                    dead.append(c)
            if dead:
                with self._ws_lock:
                    for c in dead:
                        self.ws_clients.discard(c)
        except Exception:
            pass

    def OnWebSocketOpen(self, client, uri):
        with self._ws_lock:
            self.ws_clients.add(client)

    def OnWebSocketClose(self, client):
        with self._ws_lock:
            self.ws_clients.discard(client)

    def OnHTTPRequest(self, request, response, server_type='frame'):
        self.frame_server.handle(request, response, server_type)

    def _get_relay_html(self):
        if self._relay_html_cache is None:
            remote_url, _prompt_url = self._normalized_urls(self.params.Serverurl)
            html = RELAY_HTML_TEMPLATE.replace('{{LOCAL_WS_PORT}}', str(self.mjpeg_port))
            html = html.replace('{{REMOTE_WS_URL}}', remote_url)
            self._relay_html_cache = html.encode('utf-8')
        return self._relay_html_cache

    def InvalidateRelayCache(self):
        """Call if Serverurl changes and you want the next page load to
        pick it up (or just re-Start)."""
        self._relay_html_cache = None

    def Destroy(self):
        pass

# Paste the full contents of fluxrt_tcp_relay.html here, between the
# triple quotes. The {{LOCAL_WS_PORT}} and {{REMOTE_WS_URL}} placeholders
# inside it are filled in by _get_relay_html() above.
RELAY_HTML_TEMPLATE = '''<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8" />
<title>FluxRT TCP Relay</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #000; width: 512px; height: 512px; overflow: hidden; }
  #output-canvas { width: 512px; height: 512px; display: block; }
  #input-canvas { display: none; }
  #status {
    position: absolute; inset: 0; z-index: 101;
    display: flex; align-items: center; justify-content: center;
    pointer-events: none; transition: opacity 0.3s ease-out;
  }
  #status.hidden { opacity: 0; }
  #status-text {
    color: rgba(255,255,255,0.9);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    font-size: 22px; font-weight: 500; text-align: center;
    text-shadow: 0 2px 6px rgba(0,0,0,0.5);
  }
</style>
</head>
<body>
  <canvas id="output-canvas" width="512" height="512"></canvas>
  <canvas id="input-canvas" width="512" height="512"></canvas>
  <div id="status"><div id="status-text">Connecting...</div></div>

<script>
/*
 * FluxRT TCP relay page.
 *
 * Replaces the original WHIP/WHEP (WebRTC) relay with plain WebSocket
 * frame I/O, so it works over RunPod's TCP-only networking.
 *
 * Two WebSocket connections, same bridging role the original relay had:
 *   1. LOCAL (to TD): receives input frames (JPEG) that TD's web_server
 *      DAT pushes from stream_source — identical mechanism to the
 *      original relay's input path. Painted onto #input-canvas.
 *   2. REMOTE (to FluxRT server): the input frames are forwarded here as
 *      binary JPEG; processed JPEG frames come back and are painted onto
 *      #output-canvas, which web_render captures for display in TD.
 *
 * Self-throttling: a new frame is only forwarded to the remote server
 * once the previous processed frame has come back (FRAME_IN_FLIGHT flag).
 * Same reasoning as the TD-Python version — never flood the link, and
 * keep input/output frames unambiguously paired.
 *
 * Template variables filled in by the extension before serving:
 *   {{LOCAL_WS_PORT}} - TD's local web_server port (input frames in)
 *   {{REMOTE_WS_URL}} - full ws:// URL of the FluxRT server's /ws endpoint
 */

const LOCAL_WS_URL = "ws://" + window.location.hostname + ":{{LOCAL_WS_PORT}}/ws";
const REMOTE_WS_URL = "{{REMOTE_WS_URL}}";

const statusEl = document.getElementById("status");
const statusText = document.getElementById("status-text");
function setStatus(msg) {
  console.log("[FluxRT relay]", msg);
  statusText.textContent = msg;
}
function hideStatus() { statusEl.classList.add("hidden"); }

// --- Output canvas: paint processed frames coming back from server ---
// Reuses the original relay's bitmaprenderer fast-path for painting
// JPEG frames with minimal overhead.
const outCanvas = document.getElementById("output-canvas");
let outCtx, useBitmapRenderer;
(function initOutputCanvas() {
  useBitmapRenderer = !!outCanvas.getContext("bitmaprenderer");
  outCtx = useBitmapRenderer
    ? outCanvas.getContext("bitmaprenderer")
    : outCanvas.getContext("2d");
  if (!useBitmapRenderer) {
    outCtx.fillStyle = "#000";
    outCtx.fillRect(0, 0, 512, 512);
  }
})();

let paintPending = null;
let paintBusy = false;
function paintProcessedFrame(arrayBuffer) {
  paintPending = arrayBuffer;
  drainPaint();
}
function drainPaint() {
  if (!paintPending || paintBusy) return;
  const buf = paintPending;
  paintPending = null;
  paintBusy = true;
  createImageBitmap(new Blob([buf], { type: "image/jpeg" }))
    .then((bmp) => {
      if (useBitmapRenderer) {
        outCtx.transferFromImageBitmap(bmp);
      } else {
        outCtx.drawImage(bmp, 0, 0, 512, 512);
        bmp.close();
      }
    })
    .catch(() => {})
    .finally(() => {
      paintBusy = false;
      if (paintPending) drainPaint();
    });
}

// --- Input canvas: latest input frame received from TD locally ---
const inCanvas = document.getElementById("input-canvas");
const inCtx = inCanvas.getContext("2d");
let latestInputJpeg = null; // ArrayBuffer of the most recent input frame

// --- Remote connection (to FluxRT server) ---
// DECOUPLED design (matches server's two independent loops and FluxRT's
// shared-memory model): we send input frames on a steady timer, and
// separately paint whatever processed frames arrive. We do NOT wait for
// a processed frame before sending the next input — coupling those was
// what fed the model sparse, far-apart frames and made RIFE glitch.
let remoteWs = null;

// Send input to the server at this rate, independent of output. Matches
// the server's OUTPUT_FPS and the demo's ~25fps input cadence. The model
// consumes the latest input at its own pace; oversending just means it
// always has a fresh frame, undersending would starve it.
const SEND_FPS = 25;
let lastSendTime = 0;

function connectRemote() {
  setStatus("Connecting to FluxRT server...");
  remoteWs = new WebSocket(REMOTE_WS_URL);
  remoteWs.binaryType = "arraybuffer";

  remoteWs.onopen = () => {
    setStatus("Connected, streaming...");
    setTimeout(hideStatus, 500);
    pumpFrames();
  };
  remoteWs.onmessage = (ev) => {
    if (ev.data instanceof ArrayBuffer) {
      // A processed frame arrived — just paint it. No coupling to sends.
      paintProcessedFrame(ev.data);
    }
    // Text messages (e.g. prompt-update acks) ignored.
  };
  remoteWs.onclose = () => {
    setStatus("Disconnected, reconnecting...");
    statusEl.classList.remove("hidden");
    setTimeout(connectRemote, 1000);
  };
  remoteWs.onerror = () => {
    setStatus("Connection error");
  };
}

// Send the latest input frame, but ONLY once the socket has drained the
// previous one (bufferedAmount === 0). On a fast link this runs at the full
// SEND_FPS; on a slower/proxied link (e.g. through Modal's wss edge) it
// automatically backs off and skips stale frames — latestInputJpeg is always
// the newest — so the send queue, and therefore end-to-end latency, can't
// pile up. Without this guard a fixed 25fps over a link that can't keep up
// makes the browser buffer frames for seconds: classic bufferbloat.
function pumpFrames() {
  const now = performance.now();
  if (
    remoteWs &&
    remoteWs.readyState === WebSocket.OPEN &&
    latestInputJpeg &&
    remoteWs.bufferedAmount === 0 &&
    now - lastSendTime >= 1000 / SEND_FPS
  ) {
    lastSendTime = now;
    remoteWs.send(latestInputJpeg);
  }
  requestAnimationFrame(pumpFrames);
}

// Allow the extension to push prompt updates through the same socket.
window.sendPrompt = function (promptText) {
  if (remoteWs && remoteWs.readyState === WebSocket.OPEN) {
    remoteWs.send(JSON.stringify({ prompt: promptText }));
  }
};

// --- Local connection (from TD): receive input frames ---
function connectLocal() {
  const localWs = new WebSocket(LOCAL_WS_URL);
  localWs.binaryType = "arraybuffer";
  localWs.onopen = () => console.log("[FluxRT relay] local input WS connected");
  localWs.onmessage = (ev) => {
    if (ev.data instanceof ArrayBuffer) {
      // Newest input frame from TD. Keep only the latest — if FluxRT is
      // behind, we skip stale input frames rather than queueing them.
      latestInputJpeg = ev.data;
    }
  };
  localWs.onclose = () => {
    console.log("[FluxRT relay] local input WS closed, reconnecting...");
    setTimeout(connectLocal, 1000);
  };
}

// Start both connections.
connectLocal();
connectRemote();
</script>
</body>
</html>
'''
