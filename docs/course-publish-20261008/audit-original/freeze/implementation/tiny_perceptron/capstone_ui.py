"""A loopback-only playground for the real small-world capstone checkpoint.

The interface never substitutes a reference answer for generation. Images and
tones are real tensors made by the same generators used by the teaching model.
"""

import ipaddress
import json
import logging
import math
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from tiny_perceptron.capstone import load_capstone, modality_tensors, prompt_ids, run_assistant
from tiny_perceptron.multimodal import tone

MAX_REQUEST_BYTES = 8192
STYLE = "短"
LOGGER = logging.getLogger(__name__)


def request_row(data, *, context_length=None, require_prompt=True):
    """Validate browser fields without inferring intent or reference answers."""
    allowed = {"prompt", "calculator_available", "style", "image", "audio"}
    if not isinstance(data, dict) or set(data) - allowed:
        raise ValueError("請使用已提供的問題、計算器、風格、圖片與聲音欄位。")
    prompt = data.get("prompt", "")
    if not isinstance(prompt, str) or (require_prompt and not prompt.strip()):
        raise ValueError("請先輸入問題，或按一個範例。")
    if len(prompt.encode("utf-8")) > 512:
        raise ValueError("問題太長，請縮短後再試。")
    available = data.get("calculator_available", True)
    if type(available) is not bool:
        raise ValueError("計算器開關必須是開啟或關閉。")
    if data.get("style", STYLE) != STYLE:
        raise ValueError("這個權重只示範短回答風格，尚未訓練其他風格。")
    image = data.get("image")
    if image is not None:
        if not isinstance(image, dict) or set(image) != {"color", "shape"}:
            raise ValueError("圖片請選擇顏色與形狀。")
        if image["color"] not in ("red", "green", "blue") or image["shape"] not in ("square", "circle"):
            raise ValueError("圖片只支援紅、綠、藍，以及正方形、圓形。")
        image = dict(image, offset=0, variant=1)
    audio = data.get("audio")
    if audio is not None:
        if not isinstance(audio, dict) or set(audio) != {"frequency"}:
            raise ValueError("聲音請提供純音頻率。")
        frequency = audio["frequency"]
        if type(frequency) not in (int, float) or not math.isfinite(frequency) or not 200 <= frequency <= 1200:
            raise ValueError("純音頻率必須介於 200 與 1200 Hz。")
        # These fields reconstruct the requested waveform; the generated model
        # answer, not this bookkeeping label, decides its low/high response.
        pitch = "low" if frequency < 660 else "high"
        audio = {"pitch": pitch, "variation": 1, "delta": frequency - (440 if pitch == "low" else 880)}
    row = {
        "user": prompt.strip(),
        "system": f"計算器={'開' if available else '關'}；風格={STYLE}。",
        "available": available,
        "image": image,
        "audio": audio,
    }
    if require_prompt and context_length is not None and len(prompt_ids(row)) >= context_length:
        raise ValueError("這個小模型的問題長度有限，請縮短文字後再試。")
    return row


def input_preview(row):
    """Expose the actual RGB and waveform inputs, never their expected answers."""
    image, _ = modality_tensors(row)
    pixels = None if image is None else image.permute(1, 2, 0).mul(255).round().clamp(0, 255).int().tolist()
    waveform = None
    if row["audio"] is not None:
        spec = row["audio"]
        frequency = (440 if spec["pitch"] == "low" else 880) + spec["delta"]
        waveform = tone(frequency, seconds=0.04).tolist()
    return {"image_pixels": pixels, "audio_waveform": waveform, "sample_rate": 16000, "duration_seconds": 0.04}


class PlaygroundServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, model, *, stage=None):
        self.model = model
        self.stage = stage
        self.inference_lock = threading.Lock()
        super().__init__(address, PlaygroundHandler)


class PlaygroundHandler(BaseHTTPRequestHandler):
    """Same-origin JSON routes; no remote files, shell commands or tool plugins."""

    def _reply(self, status, value, content_type="application/json; charset=utf-8"):
        body = (
            value.encode("utf-8") if isinstance(value, str) else json.dumps(value, ensure_ascii=False).encode("utf-8")
        )
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.end_headers()
        self.wfile.write(body)

    def _local_request(self):
        # Prevent a foreign browser origin or a rebinding hostname from invoking
        # the local model. The default server never listens on a public address.
        try:
            authority = urlsplit("http://" + self.headers.get("Host", ""))
            if (
                authority.hostname not in (self.server.server_address[0], "localhost")
                or authority.port != self.server.server_port
                or authority.username is not None
                or authority.path
                or authority.query
                or authority.fragment
            ):
                return False
            origin = self.headers.get("Origin")
            if origin is not None:
                source = urlsplit(origin)
                if (source.scheme, source.hostname, source.port) != ("http", authority.hostname, authority.port):
                    return False
            return True
        except ValueError:
            return False

    def do_GET(self):
        if not self._local_request():
            self._reply(403, {"error": "請從本機啟動時顯示的網址操作。"})
        elif self.path in ("/", "/index.html"):
            self._reply(200, PAGE, "text/html; charset=utf-8")
        elif self.path == "/api/info":
            self._reply(200, {"stage": self.server.stage, **self.server.model.description()})
        elif self.path == "/favicon.ico":
            self._reply(204, "")
        else:
            self._reply(404, {"error": "找不到這個頁面。"})

    def do_POST(self):
        if not self._local_request():
            self._reply(403, {"error": "請從本機啟動時顯示的網址操作。"})
            return
        if self.path not in ("/api/chat", "/api/preview"):
            self._reply(404, {"error": "找不到這個操作。"})
            return
        if self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
            self._reply(415, {"error": "請使用 JSON 傳送問題。"})
            return
        try:
            length = int(self.headers.get("Content-Length", "-1"))
        except ValueError:
            length = -1
        if not 0 < length <= MAX_REQUEST_BYTES:
            self._reply(413, {"error": "請縮短問題後再試。"})
            return
        try:
            data = json.loads(self.rfile.read(length).decode("utf-8"))
            row = request_row(
                data,
                context_length=self.server.model.config.max_length,
                require_prompt=self.path == "/api/chat",
            )
        except (ValueError, UnicodeError) as error:
            self._reply(400, {"error": str(error)})
            return
        try:
            if self.path == "/api/preview":
                self._reply(200, input_preview(row))
            else:
                with self.server.inference_lock:
                    record = run_assistant(self.server.model, row)
                self._reply(200, record)
        except Exception:
            LOGGER.exception("Capstone playground inference failed")
            self._reply(500, {"error": "推論未完成，請查看啟動伺服器的終端機。"})


def create_server(model, host="127.0.0.1", port=8765, *, stage=None):
    """Return an unstarted server; callers own shutdown and server_close."""
    address = "127.0.0.1" if host == "localhost" else host
    try:
        local = ipaddress.ip_address(address)
    except ValueError as error:
        raise ValueError("學生操作介面只接受本機 loopback 位址。") from error
    if not local.is_loopback:
        raise ValueError("學生操作介面只接受本機 loopback 位址。")
    if type(port) is not int or not 0 <= port <= 65535:
        raise ValueError("port 必須是 0 到 65535 的整數。")
    server_class = PlaygroundServer
    if local.version == 6:

        class IPv6PlaygroundServer(PlaygroundServer):
            address_family = socket.AF_INET6

        server_class = IPv6PlaygroundServer
    return server_class((address, port), model.eval(), stage=stage)


def serve(checkpoint, host="127.0.0.1", port=8765):
    """Load a CPU checkpoint and serve until Ctrl+C; never publish a port."""
    model, payload = load_capstone(checkpoint, device="cpu")
    server = create_server(model, host, port, stage=payload["stage"])
    address = server.server_address[0]
    address = f"[{address}]" if ":" in address else address
    print(f"小世界助理已啟動：http://{address}:{server.server_port}/", flush=True)
    print("只在本機提供服務；按 Ctrl+C 結束。", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


PAGE = r"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>小世界助理｜小小感知機</title>
<style>
:root{color-scheme:light;--ink:#233043;--muted:#566278;--blue:#195cb2;--line:#d9e2ed}
*{box-sizing:border-box}body{margin:0;background:#f3f6fa;color:var(--ink);font:17px/1.65 system-ui,sans-serif}
main{max-width:1120px;margin:auto;padding:32px 24px 60px}h1{font-size:2rem;margin:0 0 8px}h2{font-size:1.2rem;margin:0 0 16px}
p{margin:8px 0 16px}.muted,small{color:var(--muted)}.grid{display:grid;grid-template-columns:minmax(0,1.2fr) minmax(0,1fr);gap:24px}
.card{padding:24px;background:white;border:1px solid var(--line);border-radius:16px;min-width:0}.banner{margin-bottom:24px}
label{display:block;font-weight:600;margin:14px 0 6px}textarea,select,input[type=number]{width:100%;font:inherit;padding:10px 12px;border:1px solid #a9b7c9;border-radius:8px;background:white;color:var(--ink)}
textarea{min-height:110px;resize:vertical}button{font:inherit;cursor:pointer;min-height:44px;padding:8px 14px;border-radius:8px;border:1px solid var(--line);color:var(--ink);background:#f7f9fc}
button:focus-visible,input:focus-visible,select:focus-visible,textarea:focus-visible{outline:3px solid #74b1ff;outline-offset:2px}button:hover{background:#eaf1fb}button:disabled{opacity:.55;cursor:wait}
.examples{display:flex;flex-wrap:wrap;gap:8px}.examples button{font-size:.92rem}.row{display:flex;gap:12px;align-items:center}.row>*{flex:1;min-width:0}.check{font-weight:400;display:flex;align-items:center;gap:10px}.check input{width:20px;height:20px}
.submit{width:100%;margin-top:20px;background:var(--blue);color:white;border:none;font-weight:600}.submit:hover{background:#124889}
.error{color:#a4232d}.badge{display:inline-block;background:#e7effa;color:#194b86;border-radius:6px;padding:2px 8px;font-size:.88rem}
canvas{display:block;max-width:100%;border-radius:8px;background:#111;margin:8px 0}#image-preview{width:160px;height:160px;image-rendering:pixelated}#audio-preview{width:100%;height:88px}
.answer{font-size:1.4rem;white-space:pre-wrap;overflow-wrap:anywhere;padding:16px;border-radius:10px;background:#edf4ff;min-height:70px}
.result-row{margin:16px 0}.result-row strong{display:block}code,pre{font:14px/1.65 ui-monospace,monospace;overflow-wrap:anywhere}code{white-space:pre-wrap}pre{white-space:pre-wrap;background:#f5f7fa;padding:14px;border-radius:8px}details{margin-top:20px}summary{cursor:pointer}
@media(max-width:760px){main{padding:24px 14px 40px}.grid{grid-template-columns:1fr}.card{padding:18px}h1{font-size:1.7rem}.row{gap:8px}}
</style>
</head>
<body><main>
<h1>小世界助理</h1>
<p class="muted">把教材最後的模型真的用一次：輸入問題，觀察它選擇回答、求助，還是呼叫計算器。</p>
<div class="card banner"><strong>這個模型只練習一個小世界。</strong>
<p>它學過短問句、簡單加法、合成圖形與純音。它不是通用聊天模型，也沒有學過照片或自然語音。換個問法可能答錯；這也是值得觀察的結果。</p>
<small id="model-info">正在讀取模型資訊……</small></div>
<div class="grid">
<section class="card"><h2>1. 準備問題與輸入</h2>
<div class="examples" aria-label="問題範例">
<button type="button" data-example="calc">加法</button><button type="button" data-example="concept">解釋加法</button>
<button type="button" data-example="missing">缺少資訊</button><button type="button" data-example="copy">照抄數字</button>
<button type="button" data-example="context">讀一段資料</button><button type="button" data-example="image">看圖</button>
<button type="button" data-example="audio">聽聲音</button><button type="button" data-example="joint">圖與聲音</button>
</div>
<form id="chat-form">
<label for="prompt">你想問什麼？</label><textarea id="prompt" required maxlength="160">1+2等於多少？</textarea>
<small>每次送出是一次獨立提問，不會保留上次對話。請先試範例，再改幾個字比較。</small>
<label class="check"><input id="calculator" type="checkbox" checked>允許使用計算器</label>
<label for="style">回答風格</label><select id="style"><option value="短">短回答（這個權重已訓練的風格）</option></select>
<label class="check"><input id="use-image" type="checkbox">加入合成圖片</label>
<div id="image-controls" hidden><div class="row"><div><label for="color">顏色</label><select id="color"><option value="red">紅</option><option value="green">綠</option><option value="blue">藍</option></select></div>
<div><label for="shape">形狀</label><select id="shape"><option value="square">正方形</option><option value="circle">圓形</option></select></div></div>
<canvas id="image-preview" width="16" height="16" aria-label="實際送入模型的合成 RGB 圖片"></canvas><small>實際送入模型的是這張 16 × 16 RGB 圖片。</small></div>
<label class="check"><input id="use-audio" type="checkbox">加入合成純音</label>
<div id="audio-controls" hidden><label for="frequency">頻率（Hz）</label><input id="frequency" type="number" min="200" max="1200" step="1" value="440">
<canvas id="audio-preview" width="480" height="88" aria-label="實際送入模型的純音波形"></canvas>
<button id="play-tone" type="button">播放這段 0.04 秒純音</button>
<p><small>440 與 880 Hz 是方便開始的例子。任意頻率、邊界附近的音高不保證答對。</small></p></div>
<button class="submit" id="submit" type="submit">送出，讓模型回答</button>
<p id="status" role="status" aria-live="polite"></p>
</form></section>
<section class="card" aria-live="polite"><h2>2. 看模型實際做了什麼</h2>
<div class="result-row"><strong>最後回答</strong><div class="answer" id="answer">送出問題後，回答會出現在這裡。</div></div>
<div class="result-row"><strong>模型選擇的動作</strong><span id="action" class="badge">尚未提問</span></div>
<div class="result-row"><strong>模型原始輸出</strong><code id="raw">—</code></div>
<div class="result-row"><strong>計算器是否真的執行？</strong><span id="tool">尚未提問</span></div>
<p class="muted">工具成功後，回傳值會交給同一個模型再回答。若模型沒有完成合法動作，畫面會直接顯示失敗，不會代它填入正確答案。</p>
<details><summary>展開教學用的生成與工具紀錄</summary><pre id="trace">尚無紀錄。</pre></details>
</section></div>
</main>
<script>
'use strict';
const $ = id => document.getElementById(id);
let waveform = null, previewVersion = 0;
const examples = {
  calc: ['1+2等於多少？', false, false], concept: ['用1和2說明加法。', false, false],
  missing: ['每個商品3元。總價多少？', false, false], copy: ['照抄數字7，只要答案。', false, false],
  context: ['已讀資料：盒7在書櫃。盒7在哪？', false, false],
  image: ['圖片是什麼顏色？', true, false], audio: ['聲音是高還是低？', false, true],
  joint: ['圖片顏色與聲音高低？', true, true]
};
function fields() {
  return {prompt: $('prompt').value, calculator_available: $('calculator').checked, style: $('style').value,
    image: $('use-image').checked ? {color: $('color').value, shape: $('shape').value} : null,
    audio: $('use-audio').checked ? {frequency: Number($('frequency').value)} : null};
}
async function post(path, data) {
  const response = await fetch(path, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(data)});
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || '操作未完成，請查看終端機。');
  return result;
}
async function preview() {
  $('image-controls').hidden = !$('use-image').checked;
  $('audio-controls').hidden = !$('use-audio').checked;
  const version = ++previewVersion;
  waveform = null;
  $('play-tone').disabled = true;
  if (!$('use-image').checked && !$('use-audio').checked) return;
  try {
    const result = await post('/api/preview', {...fields(), prompt: ''});
    if (version !== previewVersion) return;
    if (result.image_pixels) {
      const context = $('image-preview').getContext('2d');
      const pixels = context.createImageData(16, 16);
      result.image_pixels.flat().forEach((rgb, i) => pixels.data.set([...rgb, 255], i * 4));
      context.putImageData(pixels, 0, 0);
    }
    waveform = result.audio_waveform;
    $('play-tone').disabled = !waveform;
    if (waveform) {
      const context = $('audio-preview').getContext('2d');
      context.clearRect(0, 0, 480, 88); context.strokeStyle = '#85c7ff'; context.beginPath();
      waveform.forEach((value, i) => {const x = i / (waveform.length - 1) * 480, y = 44 - value * 72;
        if (i === 0) context.moveTo(x, y); else context.lineTo(x, y);}); context.stroke();
    }
  } catch (error) {if (version === previewVersion) {$('status').textContent = error.message; $('status').className = 'error';}}
}
document.querySelectorAll('[data-example]').forEach(button => button.addEventListener('click', () => {
  const [prompt, image, audio] = examples[button.dataset.example];
  $('prompt').value = prompt; $('use-image').checked = image; $('use-audio').checked = audio;
  $('status').textContent = ''; preview(); $('prompt').focus();
}));
['use-image', 'use-audio', 'color', 'shape', 'frequency'].forEach(id => $(id).addEventListener('change', preview));
$('play-tone').addEventListener('click', async () => {
  if (!waveform) return;
  try {
    const Audio = window.AudioContext || window.webkitAudioContext;
    const context = new Audio();
    const buffer = context.createBuffer(1, waveform.length, 16000);
    buffer.getChannelData(0).set(waveform);
    const source = context.createBufferSource(); source.buffer = buffer;
    source.connect(context.destination); source.onended = () => context.close();
    await context.resume(); source.start();
  } catch (error) {$('status').textContent = '瀏覽器無法播放聲音：' + error.message; $('status').className = 'error';}
});
$('chat-form').addEventListener('submit', async event => {
  event.preventDefault(); $('submit').disabled = true; $('status').className = '';
  $('status').textContent = '模型正在生成……';
  try {
    const result = await post('/api/chat', fields());
    const action = result.parsed_action.status;
    const names = {direct: '直接回答', ask: '求助／補充資訊', tool: '要求工具', invalid: '未完成合法動作'};
    $('action').textContent = names[action] || action;
    $('raw').textContent = result.action_trace.raw || '（空輸出）';
    $('answer').textContent = result.answer === null ? '模型沒有產生可顯示的最後回答。請展開紀錄觀察原因。' : result.answer;
    if (!result.runtime) $('tool').textContent = '沒有執行計算器。';
    else if (result.runtime.status === 'ok') $('tool').textContent = '已執行，工具回傳：' + result.runtime.result;
    else $('tool').textContent = '沒有成功執行：' + (result.runtime.reason || result.runtime.status);
    $('trace').textContent = JSON.stringify(result, null, 2);
    $('status').textContent = '已完成這次提問。';
  } catch (error) {$('status').textContent = error.message; $('status').className = 'error';}
  finally {$('submit').disabled = false;}
});
fetch('/api/info').then(response => response.json()).then(info => {
  $('model-info').textContent = '目前載入：' + (info.stage || '未標示階段') + '；總參數 ' + info.parameters.toLocaleString() + '。所有回答由這個本機權重生成。';
}).catch(() => {$('model-info').textContent = '模型資訊讀取失敗，請查看終端機。';});
</script></body></html>
"""
