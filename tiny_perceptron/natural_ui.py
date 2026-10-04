"""Local uploads, editable speech transcripts, and the actual practical chat core.

The HTTP tests exercise this interface with an injected runner. They are not
evidence of the pretrained model's visual, transcription, or chat capability.
"""

import base64
import binascii
import io
import ipaddress
import json
import logging
import secrets
import socket
import tempfile
import threading
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from PIL import Image

from tiny_perceptron import natural_assistant as assistant

MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_REQUEST_BYTES = 12 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000
MAX_SESSIONS = 32
MAX_ASSETS_PER_SESSION = 12
MAX_TURNS = 16
MAX_PROMPT_BYTES = 8192
LOGGER = logging.getLogger(__name__)
IMAGE_TYPES = {"PNG": (".png", "image/png"), "JPEG": (".jpg", "image/jpeg"), "WEBP": (".webp", "image/webp")}
AUDIO_TYPES = {".wav": "audio/wav", ".flac": "audio/flac", ".mp3": "audio/mpeg", ".ogg": "audio/ogg"}


def upload_contents(data):
    """Decode a bounded browser upload and verify the actual image / audio."""
    filename, kind, encoded = data.get("filename"), data.get("kind"), data.get("base64")
    if (
        not isinstance(filename, str)
        or not filename
        or len(filename) > 200
        or any(character in filename for character in ("/", "\\", "\x00"))
        or not isinstance(encoded, str)
        or kind not in {"image", "audio"}
    ):
        raise ValueError("請選擇一個圖片檔或語音檔。")
    if len(encoded) > 4 * ((MAX_UPLOAD_BYTES + 2) // 3):
        raise ValueError("每個檔案最多 8 MB，請先縮小檔案。")
    try:
        content = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("檔案傳送不完整，請重新選擇。") from error
    if not 0 < len(content) <= MAX_UPLOAD_BYTES:
        raise ValueError("請選擇非空白、最多 8 MB 的檔案。")
    suffix = Path(filename).suffix.lower()
    if kind == "image":
        if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
            raise ValueError("圖片請使用 PNG、JPEG 或 WebP。")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(content)) as image:
                    if image.format not in IMAGE_TYPES or image.width * image.height > MAX_IMAGE_PIXELS:
                        raise ValueError("圖片最多 1600 萬像素，請縮小後再試。")
                    if getattr(image, "n_frames", 1) != 1:
                        raise ValueError("請選擇一張靜態圖片。")
                    extension, media_type = IMAGE_TYPES[image.format]
                    image.verify()
                with Image.open(io.BytesIO(content)) as image:
                    image.load()
        except (OSError, SyntaxError, Image.DecompressionBombWarning, Image.DecompressionBombError) as error:
            raise ValueError("無法讀取圖片，請重新儲存為 PNG 或 JPEG。") from error
    else:
        if suffix not in AUDIO_TYPES:
            raise ValueError("語音請使用 WAV、FLAC、MP3 或 OGG。")
        import soundfile as sf

        try:
            information = sf.info(io.BytesIO(content))
        except (RuntimeError, OSError, ValueError) as error:
            raise ValueError("無法讀取聲音檔，請重新儲存為 WAV。") from error
        if (
            information.frames <= 0
            or not 0 < information.samplerate <= 384000
            or not 0 < information.channels <= 8
            or information.frames / information.samplerate > 30
        ):
            raise ValueError("請選擇非空白、最長 30 秒的語音。")
        extension, media_type = suffix, AUDIO_TYPES[suffix]
    return content, extension, media_type


class NaturalServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, model, processor, options, data_root, *, asr=None):
        self.model, self.processor, self.options = model, processor, options
        self.data_root = Path(data_root).resolve()
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.upload_directory = tempfile.TemporaryDirectory(prefix=".natural-ui-", dir=self.data_root)
        self.sessions = {}
        self.inference_lock = threading.Lock()
        self.asr = asr
        try:
            super().__init__(address, NaturalHandler)
        except BaseException:
            self.upload_directory.cleanup()
            raise

    def server_close(self):
        super().server_close()
        self.upload_directory.cleanup()

    def session(self, identifier):
        if not isinstance(identifier, str) or identifier not in self.sessions:
            raise ValueError("這段對話已結束，請重新整理頁面。")
        return self.sessions[identifier]

    def asset(self, session, identifier, kind=None):
        if not isinstance(identifier, str) or identifier not in session["assets"]:
            raise ValueError("找不到這段對話的檔案，請重新選擇。")
        asset = session["assets"][identifier]
        if kind is not None and asset["kind"] != kind:
            raise ValueError("檔案種類與操作不相符，請重新選擇。")
        assistant.asset_path(asset["path"], self.data_root)
        return asset

    def operation(self, route, data):
        """Run with the same lock for mutation, ASR and model generation."""
        if route == "/api/session":
            if data:
                raise ValueError("建立對話不需要其他欄位。")
            if len(self.sessions) >= MAX_SESSIONS:
                raise ValueError("開啟的頁面太多，請重新啟動介面。")
            identifier = secrets.token_urlsafe(24)
            self.sessions[identifier] = {"history": [], "assets": {}, "transcriptions": {}}
            return {"session": identifier}
        session = self.session(data.get("session"))
        allowed = {
            "/api/upload": {"session", "filename", "kind", "base64"},
            "/api/transcribe": {"session", "audio"},
            "/api/chat": {"session", "prompt", "image", "speech"},
            "/api/reset": {"session"},
        }[route]
        if set(data) - allowed:
            raise ValueError("請使用頁面提供的輸入欄位。")
        if route == "/api/reset":
            for asset in session["assets"].values():
                assistant.asset_path(asset["path"], self.data_root).unlink()
            session.update(history=[], assets={}, transcriptions={})
            return {"reset": True}
        if route == "/api/upload":
            if len(session["assets"]) >= MAX_ASSETS_PER_SESSION:
                raise ValueError("這段對話已收到 12 個檔案，請開始新對話。")
            content, extension, media_type = upload_contents(data)
            identifier = secrets.token_hex(16)
            path = Path(self.upload_directory.name) / (identifier + extension)
            path.write_bytes(content)
            session["assets"][identifier] = {
                "path": str(path.relative_to(self.data_root)),
                "kind": data["kind"],
                "media_type": media_type,
            }
            return {"asset": identifier, "kind": data["kind"], "bytes": len(content)}
        if route == "/api/transcribe":
            identifier = data.get("audio")
            asset = self.asset(session, identifier, "audio")
            if self.asr is None:
                self.asr = assistant.load_asr(self.options)
            record = assistant.transcribe(*self.asr, assistant.asset_path(asset["path"], self.data_root))
            session["transcriptions"][identifier] = record
            return {"asr": record, "speech": identifier}
        prompt = data.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt.encode("utf-8")) > MAX_PROMPT_BYTES:
            raise ValueError("請輸入一個問題；太長的問題請分段詢問。")
        if len(session["history"]) >= 2 * MAX_TURNS:
            raise ValueError("這段對話已達 16 回合，請開始新對話。")
        image = self.asset(session, data["image"], "image") if data.get("image") is not None else None
        speech = data.get("speech")
        if speech is not None and (not isinstance(speech, str) or speech not in session["transcriptions"]):
            raise ValueError("請先辨識語音，再確認要送出的文字。")
        prompt = prompt.strip()
        row = {
            "id": "interactive",
            "user": prompt,
            "image": image["path"] if image else None,
            "history": session["history"],
        }
        result = assistant.generate(self.model, self.processor, row, self.data_root, self.options)
        prediction = result["prediction"]
        if not isinstance(prediction, str):
            raise RuntimeError("The model runner must return its actual decoded text")
        content = []
        if image:
            content.append({"type": "image", "image": image["path"]})
        content.append({"type": "text", "text": prompt})
        session["history"].extend(
            [
                {"role": "user", "content": content},
                {"role": "assistant", "content": [{"type": "text", "text": prediction}]},
            ]
        )
        asr = None
        if speech is not None:
            asr = {
                **session["transcriptions"][speech],
                "submitted_text": prompt,
                "corrected": prompt != session["transcriptions"][speech]["transcript"],
            }
        return {"user": prompt, "prediction": prediction, "image": data.get("image"), "asr": asr}


class NaturalHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Asset query strings contain a private local session identifier.
        LOGGER.debug("Natural UI %s %s", self.command, urlsplit(self.path).path)

    def _reply(self, status, value, content_type="application/json; charset=utf-8"):
        if isinstance(value, bytes):
            body = value
        elif isinstance(value, str):
            body = value.encode("utf-8")
        else:
            body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(body)

    def _local_request(self):
        try:
            authority = urlsplit("http://" + self.headers.get("Host", ""))
            if (
                authority.hostname not in (self.server.server_address[0], "localhost")
                or authority.port != self.server.server_port
                or authority.username is not None
                or authority.password is not None
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
            self._reply(403, {"error": "請從啟動時顯示的本機網址操作。"})
            return
        parsed = urlsplit(self.path)
        if parsed.path in ("/", "/index.html"):
            self._reply(200, PAGE, "text/html; charset=utf-8")
        elif parsed.path == "/favicon.ico":
            self._reply(204, b"")
        elif parsed.path == "/api/asset":
            try:
                query = parse_qs(parsed.query, strict_parsing=True)
                if set(query) != {"session", "asset"} or any(len(value) != 1 for value in query.values()):
                    raise ValueError("請重新選擇檔案。")
                with self.server.inference_lock:
                    session = self.server.session(query["session"][0])
                    asset = self.server.asset(session, query["asset"][0])
                    content = assistant.asset_path(asset["path"], self.server.data_root).read_bytes()
                self._reply(200, content, asset["media_type"])
            except ValueError as error:
                self._reply(400, {"error": str(error)})
        else:
            self._reply(404, {"error": "找不到這個頁面。"})

    def do_POST(self):
        if not self._local_request():
            self._reply(403, {"error": "請從啟動時顯示的本機網址操作。"})
            return
        if self.path not in {"/api/session", "/api/upload", "/api/chat", "/api/transcribe", "/api/reset"}:
            self._reply(404, {"error": "找不到這個操作。"})
            return
        if self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
            self._reply(415, {"error": "請使用頁面提供的表單。"})
            return
        try:
            length = int(self.headers.get("Content-Length", "-1"))
        except ValueError:
            length = -1
        if not 0 < length <= MAX_REQUEST_BYTES:
            self._reply(413, {"error": "傳送內容太大，請選擇最多 8 MB 的檔案。"})
            return
        try:
            data = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(data, dict):
                raise ValueError("請使用頁面提供的輸入欄位。")
            with self.server.inference_lock:
                result = self.server.operation(self.path, data)
        except (ValueError, UnicodeError) as error:
            self._reply(400, {"error": str(error)})
            return
        except Exception:
            LOGGER.exception("Natural assistant operation failed")
            self._reply(500, {"error": "這次未完成，請查看啟動介面的終端機；原本的對話仍保留。"})
            return
        self._reply(200, result)


def create_server(model, processor, options, data_root, host="127.0.0.1", port=8766, *, asr=None):
    """Create an unstarted loopback server; the caller owns its lifecycle."""
    address = "127.0.0.1" if host == "localhost" else host
    try:
        local = ipaddress.ip_address(address)
    except ValueError as error:
        raise ValueError("聊天介面只接受本機 loopback 位址。") from error
    if not local.is_loopback:
        raise ValueError("聊天介面只接受本機 loopback 位址。")
    if type(port) is not int or not 0 <= port <= 65535:
        raise ValueError("port 必須是 0 到 65535 的整數。")
    server_class = NaturalServer
    if local.version == 6:

        class IPv6NaturalServer(NaturalServer):
            address_family = socket.AF_INET6

        server_class = IPv6NaturalServer
    return server_class((address, port), model, processor, options, data_root, asr=asr)


def serve(options, host="127.0.0.1", port=8766):
    """Use the actual selected base / adapter and load speech recognition on demand."""
    model, processor = assistant.load_core(options, adapter=options.adapter)
    root = options.data_root or Path(options.output) / "ui-data"
    server = create_server(model, processor, options, root, host, port)
    address = server.server_address[0]
    address = f"[{address}]" if ":" in address else address
    print(f"照片與語音助理已啟動：http://{address}:{server.server_port}/", flush=True)
    print("只在本機提供服務；按 Ctrl+C 結束，結束時會刪除這次上傳的檔案。", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


PAGE = r"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>照片與語音助理｜小小感知機</title>
<style>
:root{color-scheme:light;--ink:#253144;--muted:#52627a;--line:#d7e0ec;--blue:#1558a7}
*{box-sizing:border-box}body{margin:0;background:#f2f5fa;color:var(--ink);font:17px/1.65 system-ui,sans-serif}
main{max-width:1120px;margin:auto;padding:28px 24px 50px}h1{font-size:2rem;margin:0 0 8px}h2{font-size:1.2rem;margin:0 0 12px}p{margin:8px 0 16px}
.muted,small{color:var(--muted)}.grid{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.15fr);gap:22px}
.card{min-width:0;background:white;border:1px solid var(--line);padding:22px;border-radius:16px}
label{display:block;font-weight:600;margin:14px 0 6px}input,textarea,button{font:inherit;max-width:100%}
input[type=file]{width:100%;font-size:.95rem}textarea{width:100%;min-height:120px;resize:vertical;border:1px solid #a3b1c4;border-radius:8px;padding:10px;color:var(--ink)}
button{cursor:pointer;padding:9px 14px;min-height:44px;border:1px solid var(--line);border-radius:8px;background:#f3f6fc;color:var(--ink)}
button:disabled{opacity:.55;cursor:wait}button:focus-visible,input:focus-visible,textarea:focus-visible{outline:3px solid #6da9f2;outline-offset:2px}
.send{background:var(--blue);color:white;border:0;width:100%;margin-top:14px}.topline{display:flex;gap:12px;align-items:center;justify-content:space-between;flex-wrap:wrap}
.preview{display:block;width:100%;max-height:420px;object-fit:contain;background:#f1f4f9;border-radius:8px;margin:10px 0}.preview[hidden],audio[hidden]{display:none}
audio{display:block;width:100%;margin:10px 0}.hint{font-size:.93rem;margin:6px 0 10px}.error{color:#a0232d}#status{min-height:1.7em;margin:10px 0}
#conversation{display:flex;flex-direction:column;gap:14px}.turn{padding:14px 16px;border-radius:10px;background:#edf4ff;overflow-wrap:anywhere;white-space:pre-wrap}
.turn.user{background:#f2f4f8}.turn strong{display:block;font-size:.9rem;color:var(--muted);margin-bottom:5px}.turn img{display:block;max-width:100%;max-height:320px;object-fit:contain;margin:8px 0;border-radius:6px}
.transcript{white-space:pre-wrap;overflow-wrap:anywhere;padding:10px;background:#f2f4f8;border-radius:6px;font-size:.95rem}
@media(max-width:760px){main{padding:22px 14px 40px}h1{font-size:1.65rem}.grid{grid-template-columns:1fr}.card{padding:18px}}
</style></head><body><main>
<h1>照片與語音助理</h1>
<p class="muted">可以打字，也可以先把中文語音轉成文字。照片會跟著對話保留，下一句可以繼續問同一張圖。</p>
<div class="grid"><section class="card" aria-labelledby="input-title"><h2 id="input-title">想聊什麼？</h2>
<label for="image">照片（可不選）</label><input id="image" type="file" accept="image/png,image/jpeg,image/webp">
<p class="hint muted">PNG、JPEG 或 WebP，每個檔案最多 8 MB。</p><img id="preview" class="preview" alt="這次選擇的照片" hidden>
<button id="remove-image" type="button" hidden>移除這次照片</button>
<label for="audio">中文語音（可不選）</label><input id="audio" type="file" accept=".wav,.flac,.mp3,.ogg">
<p class="hint muted">最多 30 秒、8 MB。先按「辨識語音」，檢查文字，再送給助理。</p>
<audio id="audio-preview" controls hidden></audio><button id="transcribe" type="button">辨識語音</button>
<div id="original-transcript" class="transcript" hidden></div>
<label for="prompt">要送出的文字</label><textarea id="prompt" placeholder="例如：照片裡的人在做什麼？"></textarea>
<p class="hint muted">語音辨識可能聽錯，你可以直接在這裡更正。助理會讀這份文字。</p>
<button id="send" class="send" type="button">送出問題</button>
<p id="status" role="status" aria-live="polite"></p>
</section><section class="card" aria-labelledby="conversation-title">
<div class="topline"><h2 id="conversation-title">對話</h2><button id="reset" type="button">開始新對話</button></div>
<p id="empty" class="muted">助理的實際回答會出現在這裡。看錯、聽錯或回答錯時，也會保留原文。</p>
<div id="conversation" aria-live="polite"></div>
</section></div></main>
<script>
const $ = id => document.getElementById(id);
let session = null, imageAsset = null, audioAsset = null, speechAsset = null, imageUrl = null, audioUrl = null, pending = false;
const controls = ['send','transcribe','reset','image','audio','prompt','remove-image'];
function status(message, error=false){$('status').textContent=message;$('status').classList.toggle('error',error);}
function busy(value){pending=value;controls.forEach(id=>$(id).disabled=value || !session);}
async function api(path, data){
 const response = await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
 const result = await response.json();if(!response.ok)throw new Error(result.error || '操作未完成，請重試。');return result;
}
function readFile(file){return new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(String(reader.result).split(',',2)[1]);reader.onerror=()=>reject(new Error('無法讀取檔案。'));reader.readAsDataURL(file);});}
async function upload(file,kind){
 if(!file)throw new Error(kind==='audio'?'請先選擇一段語音。':'請先選擇一張照片。');
 if(!file.size || file.size>8*1024*1024)throw new Error('請選擇非空白、最多 8 MB 的檔案。');
 return (await api('/api/upload',{session,kind,filename:file.name,base64:await readFile(file)})).asset;
}
function assetUrl(asset){return '/api/asset?session='+encodeURIComponent(session)+'&asset='+encodeURIComponent(asset);}
function addTurn(role,text,image=null,asr=null){
 $('empty').hidden=true;const turn=document.createElement('div');turn.className='turn '+role;
 const title=document.createElement('strong');title.textContent=role==='user'?'你':'助理';turn.appendChild(title);
 if(image){const preview=document.createElement('img');preview.src=assetUrl(image);preview.alt='這一回合的照片';turn.appendChild(preview);}
 const body=document.createElement('div');body.textContent=text;turn.appendChild(body);
 if(asr){const note=document.createElement('small');note.textContent='語音原稿：'+asr.transcript+(asr.corrected?'（送出前已更正）':'');turn.appendChild(note);}
 $('conversation').appendChild(turn);
}
function clearImage(){if(imageUrl)URL.revokeObjectURL(imageUrl);imageUrl=null;imageAsset=null;$('image').value='';$('preview').hidden=true;$('preview').removeAttribute('src');$('remove-image').hidden=true;}
function clearAudio(){if(audioUrl)URL.revokeObjectURL(audioUrl);audioUrl=null;audioAsset=null;speechAsset=null;$('audio').value='';$('audio-preview').hidden=true;$('audio-preview').removeAttribute('src');$('original-transcript').hidden=true;$('original-transcript').textContent='';}
$('image').addEventListener('change',()=>{
 if(imageUrl)URL.revokeObjectURL(imageUrl);imageAsset=null;const file=$('image').files[0];
 if(!file){clearImage();return;}imageUrl=URL.createObjectURL(file);$('preview').src=imageUrl;$('preview').hidden=false;$('remove-image').hidden=false;
});
$('remove-image').addEventListener('click',clearImage);
$('audio').addEventListener('change',()=>{
 if(audioUrl)URL.revokeObjectURL(audioUrl);audioAsset=null;speechAsset=null;$('original-transcript').hidden=true;
 const file=$('audio').files[0];if(!file){clearAudio();return;}audioUrl=URL.createObjectURL(file);$('audio-preview').src=audioUrl;$('audio-preview').hidden=false;
});
$('transcribe').addEventListener('click',async()=>{
 if(pending)return;busy(true);status('正在辨識語音，第一次可能需要稍等。');
 try{if(!audioAsset)audioAsset=await upload($('audio').files[0],'audio');const result=await api('/api/transcribe',{session,audio:audioAsset});speechAsset=result.speech;
  $('prompt').value=result.asr.transcript;$('original-transcript').textContent='辨識原稿：'+result.asr.transcript;$('original-transcript').hidden=false;status('請檢查上面的文字，再按「送出問題」。');
 }catch(error){status(error.message,true);}finally{busy(false);}
});
$('send').addEventListener('click',async()=>{
 if(pending)return;const prompt=$('prompt').value.trim();if(!prompt){status('請先輸入問題，或辨識一段語音。',true);return;}
 busy(true);status('助理正在回答…');
 try{if($('image').files[0] && !imageAsset)imageAsset=await upload($('image').files[0],'image');
  const result=await api('/api/chat',{session,prompt,image:imageAsset,speech:speechAsset});
  addTurn('user',result.user,result.image,result.asr);addTurn('assistant',result.prediction);$('prompt').value='';clearImage();clearAudio();status('可以繼續詢問。');
  $('conversation').lastElementChild.scrollIntoView({block:'nearest'});
 }catch(error){status(error.message,true);}finally{busy(false);}
});
$('reset').addEventListener('click',async()=>{
 if(pending)return;busy(true);try{await api('/api/reset',{session});$('conversation').replaceChildren();$('empty').hidden=false;$('prompt').value='';clearImage();clearAudio();status('已開始新對話。');}
 catch(error){status(error.message,true);}finally{busy(false);}
});
(async()=>{busy(true);try{session=(await api('/api/session',{})).session;status('可以開始輸入。');}catch(error){status(error.message,true);}finally{busy(false);}})();
</script></body></html>
"""
