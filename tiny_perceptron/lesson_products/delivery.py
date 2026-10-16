"""固定HF revision、SHA驗證下載，以及只綁本機的有限HTTP示範。"""

import hashlib
import json
import re
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote

from .inference import Product, auto_solve, check_manifest


def download_product(repo, revision, prefix, manifest_sha256, destination, product):
    if (
        not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo)
        or not re.fullmatch(r"[a-f0-9]{40}", revision)
        or not re.fullmatch(r"[a-f0-9]{64}", manifest_sha256)
        or any(p in ("", ".", "..") for p in prefix.split("/"))
    ):
        raise ValueError("下載需要確切HF repo、40字revision、相對prefix與64字manifest SHA")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    base = f"https://huggingface.co/{repo}/resolve/{revision}/{quote(prefix, safe='/')}"

    def fetch(name, digest, size=None):
        path = destination / name
        if path.is_symlink():
            raise ValueError("拒絕覆寫符號連結")
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == digest:
            return path.read_bytes()
        limit = 5 * 1024 * 1024 if size is None else size
        if not 0 < limit <= 5 * 1024 * 1024:
            raise ValueError("超過本成品固定下載上限")
        request = urllib.request.Request(base + "/" + quote(name), headers={"User-Agent": "tiny-perceptron-course/1"})
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read(limit + 1)
        if len(raw) > limit or (size is not None and len(raw) != size) or hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("匿名下載大小／SHA不符：" + name)
        temporary = path.with_name(path.name + ".download")
        if temporary.is_symlink():
            raise ValueError("拒絕下載至符號連結")
        temporary.write_bytes(raw)
        temporary.replace(path)
        return raw

    manifest = json.loads(fetch("artifact-manifest.json", manifest_sha256))
    if manifest.get("product") != product or manifest.get("version") != 1:
        raise ValueError("下載成品種類不符")
    for name, expected in manifest["files"].items():
        if Path(name).name != name or name in (".", "..", "artifact-manifest.json"):
            raise ValueError("下載檔名需在成品目錄內")
        fetch(name, expected["sha256"], expected["bytes"])
    check_manifest(destination, product)
    return {
        "repo": repo,
        "revision": revision,
        "prefix": prefix,
        "product": product,
        "manifest_sha256": manifest_sha256,
        "anonymous": True,
        "files_verified": len(manifest["files"]),
    }


def serve(b_directory, c_directory, port=8765, device="cpu"):
    b, c = Product(b_directory, "b", device), Product(c_directory, "c", device)

    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, value):
            raw = (json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n").encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            self.respond(
                200 if self.path == "/health" else 404,
                {"status": "ready", "products": ["b", "c"]}
                if self.path == "/health"
                else {"error": "unknown endpoint"},
            )

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 1 <= length <= 16384:
                    raise ValueError("請提供1–16384 bytes的JSON request")
                request = json.loads(self.rfile.read(length))
                if not isinstance(request, dict):
                    raise ValueError("request需為JSON object")
                if self.path == "/solve":
                    value = b.solve(**request)
                elif self.path == "/judge":
                    value = c.judge(**request)
                elif self.path == "/auto":
                    value = auto_solve(b, c, **request)
                else:
                    self.respond(404, {"error": "unknown endpoint"})
                    return
                self.respond(200, value)
            except (ValueError, TypeError, KeyError) as error:
                self.respond(400, {"error": str(error)})

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"有限成品API：http://127.0.0.1:{port}；POST /solve /judge /auto，GET /health", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
