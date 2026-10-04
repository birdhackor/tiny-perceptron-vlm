"""Temporary CPU-only fresh source bootstrap; run from the reserved GHA checkout."""
import base64
import hashlib
import http.client
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit

REPO = "birdhackor/tiny-perceptron-vlm"
BASE = "https://github.com/" + REPO + ".git/info/lfs"
MANIFEST = "docs/natural-assistant/v4/manifest.json"
MANIFEST_SHA = "0c660490eb78bd82a8e092c2658646a6bae59c70058b6f5c2c944d138f732f60"
SCRIPTS = ["scripts/" + name + ".py" for name in (
    "build_natural_v4_assets", "prepare_natural_v4_ocr", "replay_natural_v4_voice", "modal_natural"
)]
LICENSE = "docs/natural-assistant-v4/research/chat-speech.Apache-2.0-LICENSE"
BATCH = "natural-v4"
OPERATION = "source-data-bootstrap"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def settings():
    return {
        "steps": 100, "seed": 42, "max_seconds": 1620, "max_pixels": 524288,
        "learning_rate": 0.0001, "asr_variant": "turbo",
        "asr_model": "openai/whisper-large-v3-turbo",
        "asr_revision": "41f01f3fe87f28c78e2fbf8b568835947dd65ed9",
        "checkpoint_steps": [], "adapter_checkpoint": "", "adapter_checkpoints": [],
    }


def runtime_sha():
    return hashlib.sha256(json.dumps(settings(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError("Refusing an LFS credential redirect")


def action_url(action, kind, oid):
    href = action.get("href")
    if not isinstance(href, str) or any(ord(c) <= 32 or ord(c) >= 127 for c in href):
        raise ValueError("Invalid object action URL")
    parsed = urlsplit(href)
    host = parsed.hostname or ""
    if parsed.scheme != "https" or parsed.netloc != host or parsed.fragment or "\\" in parsed.path:
        raise ValueError("Object action is not a canonical HTTPS URL")
    if kind == "verify":
        if host != "lfs.github.com" or parsed.path != "/" + REPO + "/objects/" + oid + "/verify":
            raise ValueError("Verify action does not name the exact frozen GitHub LFS object")
    elif kind == "upload":
        if not host.endswith(".amazonaws.com") or not (host.startswith(("s3.", "s3-")) or ".s3." in host or ".s3-" in host) or parsed.path in {"", "/"}:
            raise ValueError("Upload action is not an S3 object PUT")
    return parsed


def validated_actions(batch, objects, github_token=""):
    if batch.get("transfer", "basic") != "basic" or batch.get("hash_algo", "sha256") != "sha256":
        raise ValueError("Unexpected LFS transfer")
    items = batch.get("objects", [])
    if len(items) != 3 or len({item.get("oid") for item in items}) != 3:
        raise ValueError("Need exactly three unique frozen LFS objects")
    by_oid = {item["oid"]: item for item in items}
    selected = []
    for expected in objects:
        item = by_oid.get(expected["oid"], {})
        if item.get("size") != expected["size"] or item.get("error"):
            raise ValueError("LFS object does not match the frozen manifest")
        actions = item.get("actions", {})
        chosen = {**expected, "upload": actions.get("upload"), "verify": actions.get("verify")}
        serialized = json.dumps(chosen)
        if github_token and any(secret in serialized for secret in (
            github_token, base64.b64encode(("x-access-token:" + github_token).encode()).decode()
        )):
            raise ValueError("Refusing to send the GitHub token to Modal")
        for kind in ("upload", "verify"):
            action = chosen[kind]
            if action is None:
                continue
            action_url(action, kind, expected["oid"])
            headers = action.get("header", {})
            if not isinstance(headers, dict):
                raise ValueError("Invalid object action headers")
            for name, value in headers.items():
                if not isinstance(name, str) or not isinstance(value, str) or "\r" in value or "\n" in value:
                    raise ValueError("Invalid object action header")
                if name.lower() in {"cookie", "set-cookie", "proxy-authorization"}:
                    raise ValueError("Refusing reusable cookie or proxy credentials")
                if name.lower() == "authorization":
                    v4 = re.fullmatch(r"AWS4-HMAC-SHA256 +Credential=[A-Za-z0-9]+/[0-9]{8}/[a-z0-9-]+/s3/aws4_request, *SignedHeaders=([a-z0-9-]+(?:;[a-z0-9-]+)*), *Signature=[0-9a-fA-F]{64}", value)
                    v2 = re.fullmatch(r"AWS +[A-Za-z0-9]+:[A-Za-z0-9+/]{27}=", value)
                    if kind == "upload" and not (v4 and "host" in v4.group(1).split(";") or v2):
                        raise ValueError("Upload authentication is not an S3 request signature")
                    if kind == "verify" and not value.startswith("RemoteAuth "):
                        raise ValueError("Verify authentication is not the provided object-scoped RemoteAuth")
        selected.append(chosen)
    return selected


def batch_request(operation, objects, github_token):
    basic = "Basic " + base64.b64encode(("x-access-token:" + github_token).encode()).decode()
    request = urllib.request.Request(BASE + "/objects/batch", data=json.dumps({"operation": operation, "transfers": ["basic"], "objects": objects}).encode(), headers={"Authorization": basic, "Accept": "application/vnd.git-lfs+json", "Content-Type": "application/vnd.git-lfs+json"})
    try:
        with urllib.request.build_opener(NoRedirect()).open(request, timeout=60) as response:
            return json.loads(response.read(1024 * 1024))
    except urllib.error.HTTPError as error:
        raise RuntimeError("GitHub LFS batch HTTP " + str(error.code)) from None
    except urllib.error.URLError:
        raise RuntimeError("GitHub LFS batch transport failure") from None


def transport(action, kind, oid, body, size, timeout):
    parsed = action_url(action, kind, oid)
    headers = dict(action.get("header", {}))
    headers["Content-Length"] = str(size)
    if kind == "verify":
        headers.setdefault("Accept", "application/vnd.git-lfs+json")
        headers.setdefault("Content-Type", "application/vnd.git-lfs+json")
    connection = http.client.HTTPSConnection(parsed.hostname, timeout=timeout)
    try:
        path = parsed.path + ("?" + parsed.query if parsed.query else "")
        connection.request("PUT" if kind == "upload" else "POST", path, body=body, headers=headers, encode_chunked=False)
        response = connection.getresponse()
        status = response.status
        response.read(4096)
        if not 200 <= status < 300:
            raise RuntimeError("Object " + kind + " failed with HTTP " + str(status))
    except (OSError, http.client.HTTPException):
        raise RuntimeError("Object " + kind + " transport failed") from None
    finally:
        connection.close()


def source_gate(root, revision, run_id, expected_runtime_sha):
    lock = json.loads((root / "source-lock.json").read_bytes())
    if not re.fullmatch(r"[0-9a-f]{40}", revision) or lock.get("revision") != revision or expected_runtime_sha != runtime_sha():
        raise ValueError("Worker revision or runtime contract differs from the reservation")
    if lock.get("manifest_sha256") != MANIFEST_SHA or digest(root / MANIFEST) != MANIFEST_SHA:
        raise ValueError("Worker manifest differs from the frozen source snapshot")
    if not set(SCRIPTS + [MANIFEST, LICENSE]).issubset(lock.get("files", {})):
        raise ValueError("Source lock is missing required runtime, builder, manifest or license files")
    for relative, sha in lock["files"].items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or digest(root / path) != sha:
            raise ValueError("Worker source file differs from the immutable Git snapshot")
    os.environ["NATURAL_MODAL_PHASE"] = "control"
    os.environ["NATURAL_MANIFEST"] = MANIFEST
    spec = importlib.util.spec_from_file_location("bootstrap_existing_natural_control", root / "scripts/modal_natural.py")
    control = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(control)
    options = {**settings(), "runtime_options": settings(), "runtime_options_sha256": expected_runtime_sha}
    if control.verify_runtime_contract("prepare", options) != expected_runtime_sha:
        raise ValueError("Existing control runtime contract rejected the fixed bootstrap options")
    control.volume.reload()
    control.require_reservation(run_id, BATCH, "prepare", revision, MANIFEST_SHA, runtime_sha=expected_runtime_sha)
    return control


def source_worker(run_id, revision, expected_runtime_sha):
    root = Path("/app")
    source_gate(root, revision, run_id, expected_runtime_sha)
    deadline = time.monotonic() + 1620
    manifest = json.loads((root / MANIFEST).read_bytes())
    objects = [{"oid": a["sha256"], "size": a["bytes"]} for a in manifest["archives"]]
    # Keep capabilities in process memory and remove them before public-source subprocesses.
    secret = os.environ.pop("NATURAL_LFS_ACTIONS")
    capabilities = json.loads(secret)
    actions = validated_actions({"objects": [{**item, "actions": {"upload": item["upload"], "verify": item["verify"]}} for item in capabilities]}, objects)
    del secret
    with tempfile.TemporaryDirectory(prefix="natural-v4-source-") as temporary:
        work = Path(temporary)
        command = [sys.executable, str(root / "scripts/build_natural_v4_assets.py"), "--reconstruct", "--verify-manifest", str(root / MANIFEST), "--manifest", str(work / "manifest.json"), "--data-root", str(work / "data"), "--assets", str(work / "assets")]
        subprocess.run(command, check=True, timeout=max(1, deadline - time.monotonic()), cwd=root)
        if digest(work / "manifest.json") != MANIFEST_SHA:
            raise ValueError("Fresh source reconstruction changed the frozen manifest")
        receipts = []
        for archive, item in zip(manifest["archives"], actions, strict=True):
            path = work / "assets" / Path(archive["path"]).name
            if path.stat().st_size != item["size"] or digest(path) != item["oid"]:
                raise ValueError("Fresh source archive differs from the frozen object")
            if time.monotonic() >= deadline:
                raise RuntimeError("Source bootstrap exceeded the reserved runtime")
            if item["upload"]:
                with path.open("rb") as stream:
                    transport(item["upload"], "upload", item["oid"], stream, item["size"], min(120, max(1, deadline - time.monotonic())))
            if item["verify"]:
                body = json.dumps({"oid": item["oid"], "size": item["size"]}).encode()
                transport(item["verify"], "verify", item["oid"], body, len(body), min(60, max(1, deadline - time.monotonic())))
            receipts.append({"oid": item["oid"], "bytes": item["size"], "fresh_source_sha256_verified": True, "upload_performed": bool(item["upload"]), "verify_performed": bool(item["verify"])})
    return {"operation": OPERATION, "status": "completed", "run_id": run_id, "source_git_revision": revision, "manifest_sha256": MANIFEST_SHA, "runtime_options_sha256": expected_runtime_sha, "gpu_started": False, "ordinary_model_prepare_completed": False, "objects": receipts}


def verify_downloads(objects, github_token):
    batch = batch_request("download", objects, github_token)
    items = batch.get("objects", [])
    if len(items) != 3 or len({item.get("oid") for item in items}) != 3:
        raise ValueError("Download batch does not contain exactly the frozen objects")
    by_oid = {item["oid"]: item for item in items}
    receipts = []
    opener = urllib.request.build_opener(NoRedirect())
    for expected in objects:
        item = by_oid.get(expected["oid"], {})
        if item.get("size") != expected["size"] or item.get("error"):
            raise ValueError("Uploaded LFS object is unavailable")
        action = item["actions"]["download"]
        parsed = urlsplit(action["href"])
        if parsed.scheme != "https" or parsed.username or parsed.password or parsed.fragment or parsed.port not in (None, 443):
            raise ValueError("Invalid HTTPS LFS download action")
        request = urllib.request.Request(action["href"], headers=action.get("header", {}))
        count, sha = 0, hashlib.sha256()
        try:
            with opener.open(request, timeout=120) as response:
                while raw := response.read(min(1024 * 1024, expected["size"] + 1 - count)):
                    count += len(raw)
                    if count > expected["size"]:
                        raise ValueError("LFS object exceeds the frozen byte count")
                    sha.update(raw)
        except urllib.error.HTTPError as error:
            raise RuntimeError("LFS download HTTP " + str(error.code)) from None
        except urllib.error.URLError:
            raise RuntimeError("LFS download transport failure") from None
        if count != expected["size"] or sha.hexdigest() != expected["oid"]:
            raise ValueError("LFS object bytes differ from the frozen manifest")
        receipts.append({**expected, "lfs_download_sha256_verified": True})
    return receipts


def main():
    import modal

    root = Path.cwd()
    revision, run_id = os.environ["GITHUB_SHA"], os.environ["NATURAL_BOOTSTRAP_RUN_ID"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision) or not re.fullmatch(r"gha-[A-Za-z0-9_.-]{1,90}", run_id):
        raise ValueError("Need the actual GHA revision and fresh run identifier")
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root).decode().strip() != revision:
        raise ValueError("Checkout is not the exact reserved Git revision")
    reservation = json.loads((root / "outputs/natural-assistant/reservation.json").read_bytes())["entry"]
    for key, value in {"run_id": run_id, "batch_id": BATCH, "stage": "prepare", "revision": revision, "manifest_sha256": MANIFEST_SHA, "runtime_options_sha256": runtime_sha(), "status": "reserved"}.items():
        if reservation.get(key) != value:
            raise ValueError("Missing exact existing prepare reservation before CPU image build")
    files = sorted(set(SCRIPTS + [MANIFEST, LICENSE] + [p.relative_to(root).as_posix() for p in (root / "docs/natural-assistant/v4/data").rglob("*") if p.is_file()]))
    lock = {"revision": revision, "manifest_sha256": MANIFEST_SHA, "files": {}}
    for relative in files:
        committed = subprocess.check_output(["git", "show", revision + ":" + relative], cwd=root)
        if (root / relative).read_bytes() != committed:
            raise ValueError("Source input differs from its immutable Git blob")
        lock["files"][relative] = hashlib.sha256(committed).hexdigest()
    if lock["files"][MANIFEST] != MANIFEST_SHA:
        raise ValueError("Immutable manifest is not the authorized frozen snapshot")
    manifest = json.loads((root / MANIFEST).read_bytes())
    objects = [{"oid": a["sha256"], "size": a["bytes"]} for a in manifest["archives"]]
    token = os.environ["GITHUB_TOKEN"]
    if not token:
        raise ValueError("GHA GitHub token is unavailable")
    actions = validated_actions(batch_request("upload", objects, token), objects, token)
    secret = modal.Secret.from_dict({"NATURAL_LFS_ACTIONS": json.dumps(actions)})
    volume = modal.Volume.from_name("tiny-perceptron-course", create_if_missing=False)
    image = modal.Image.debian_slim(python_version="3.13").apt_install("libsndfile1").pip_install("Pillow==12.3.0", "h5py==3.16.0", "numpy==2.5.3", "soundfile==0.14.0").env({"NATURAL_MODAL_PHASE": "control", "NATURAL_MANIFEST": MANIFEST, "PYTHONPATH": "/app", "OMP_NUM_THREADS": "2"}).workdir("/app")
    for relative in files:
        image = image.add_local_file(root / relative, "/app/" + relative, copy=True)
    output = root / "outputs/natural-v4/modal-source"
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / "source-lock.json"
    lock_path.write_text(json.dumps(lock, sort_keys=True) + "\n")
    image = image.add_local_file(lock_path, "/app/source-lock.json", copy=True)
    app = modal.App("tiny-perceptron-natural-v4-source-bootstrap")
    remote = app.function(image=image, cpu=(2, 2), memory=(8192, 8192), timeout=1800, retries=0, max_containers=1, scaledown_window=2, volumes={"/course": volume}, secrets=[secret], serialized=True)(source_worker)
    with app.run():
        receipt = remote.remote(run_id, revision, runtime_sha())
    (output / "modal-source-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    downloads = verify_downloads(objects, token)
    receipt.update(lfs_upload_completed_and_download_verified=True, gha_lfs_downloads=downloads)
    (output / "upload-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"operation": OPERATION, "status": "completed", "manifest_sha256": MANIFEST_SHA, "gpu_started": False, "lfs_download_objects_verified": len(downloads)}))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Do not stringify remote exceptions: their internal frames may hold capabilities.
        print(json.dumps({"operation": OPERATION, "status": "failed", "exception_type": type(error).__name__, "gpu_started": False}), flush=True)
        raise SystemExit(1) from None
