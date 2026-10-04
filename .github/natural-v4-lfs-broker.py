import base64
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

REPO = "birdhackor/tiny-perceptron-vlm"
BASE = "https://github.com/" + REPO + ".git/info/lfs"
MANIFEST = Path("docs/natural-assistant/v4/manifest.json")
FROZEN_MANIFEST_SHA256 = "0c660490eb78bd82a8e092c2658646a6bae59c70058b6f5c2c944d138f732f60"
OUTPUT = Path("outputs/natural-v4/lfs-broker")
PENDING = Path("outputs/natural-v4/lfs-internal/pending.json")
OUTPUT.mkdir(parents=True, exist_ok=True)
manifest_bytes = MANIFEST.read_bytes()
manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
if manifest_sha != FROZEN_MANIFEST_SHA256:
    raise RuntimeError("Manifest differs from the three frozen source bundles")
manifest = json.loads(manifest_bytes)
objects = [{"oid": a["sha256"], "size": a["bytes"]} for a in manifest["archives"]]
if len(objects) != 3 or len({item["oid"] for item in objects}) != 3 or any(
    not isinstance(item["oid"], str) or not re.fullmatch(r"[0-9a-f]{64}", item["oid"])
    or type(item["size"]) is not int or item["size"] <= 0 for item in objects
):
    raise RuntimeError("Manifest must describe exactly three unique SHA256 objects")
token = os.environ["GITHUB_TOKEN"]
if not token:
    raise RuntimeError("GITHUB_TOKEN must be present in the Actions runner")
auth = "Basic " + base64.b64encode(("x-access-token:" + token).encode()).decode()


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError("Refusing an HTTP redirect for an LFS action")


opener = urllib.request.build_opener(NoRedirect())


def validate_repo_url(url, allow_query=False):
    if not isinstance(url, str) or any(ord(c) <= 32 or ord(c) >= 127 for c in url):
        raise ValueError("Invalid repository verification endpoint")
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https" or parsed.netloc != "github.com"
        or parsed.fragment or "#" in url
        or (not allow_query and (parsed.query or "?" in url))
        or token in parsed.path or auth in parsed.path or auth.removeprefix("Basic ") in parsed.path
        or not parsed.path.startswith("/" + REPO + ".git/info/lfs/")
        or "%" in parsed.path or "\\" in parsed.path
        or any(segment in (".", "..", "") for segment in parsed.path.split("/")[1:])
    ):
        raise ValueError("Refusing an endpoint outside the canonical repository LFS path")
    return parsed


class LFSHTTPError(RuntimeError):
    def __init__(self, status):
        self.status = status
        super().__init__("LFS action failed with HTTP " + str(status))


def safe_open(req, timeout):
    try:
        return opener.open(req, timeout=timeout)
    except urllib.error.HTTPError as error:
        # HTTPError includes the request URL; do not expose a presigned capability.
        raise LFSHTTPError(error.code) from None
    except urllib.error.URLError:
        raise RuntimeError("LFS action failed at the transport layer") from None


def request(url, body, action_headers=None, allow_query=False, timeout=60):
    validate_repo_url(url, allow_query=allow_query)
    payload = json.dumps(body).encode()
    headers = {"Authorization": auth, "Accept": "application/vnd.git-lfs+json", "Content-Type": "application/vnd.git-lfs+json"}
    if action_headers is not None:
        if not isinstance(action_headers, dict) or any(not isinstance(name, str) or not isinstance(value, str) or "\r" in value or "\n" in value for name, value in action_headers.items()):
            raise RuntimeError("Malformed private LFS action headers")
        headers.update(action_headers)
    req = urllib.request.Request(url, data=payload, headers=headers)
    with safe_open(req, timeout=timeout) as response:
        raw = response.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            raise RuntimeError("LFS JSON response exceeds the three-object response limit")
        return json.loads(raw) if raw else {}


def exact_batch(batch):
    if batch.get("transfer", "basic") != "basic" or batch.get("hash_algo", "sha256") != "sha256":
        raise RuntimeError("LFS batch selected an unexpected transfer or hash algorithm")
    items = batch.get("objects")
    if not isinstance(items, list) or len(items) != 3:
        raise RuntimeError("LFS batch did not return exactly three objects")
    by_oid = {}
    for item in items:
        if not isinstance(item, dict) or item.get("oid") in by_oid:
            raise RuntimeError("LFS batch contains a malformed or duplicate object")
        by_oid[item.get("oid")] = item
    ordered = []
    for expected in objects:
        item = by_oid.get(expected["oid"], {})
        if item.get("size") != expected["size"] or type(item.get("size")) is not int or item.get("error"):
            raise RuntimeError("LFS batch did not authorize the exact three frozen objects")
        ordered.append(item)
    return ordered


def validate_upload(action, expected):
    if not isinstance(action, dict):
        raise RuntimeError("Malformed upload action")
    exported = json.dumps(action)
    if token in exported or auth in exported or base64.b64encode(("x-access-token:" + token).encode()).decode() in exported:
        raise RuntimeError("Refusing to export the GitHub token in an upload action")
    href = action.get("href")
    if not isinstance(href, str) or any(ord(c) <= 32 or ord(c) >= 127 for c in href):
        raise RuntimeError("Malformed upload URL")
    parsed = urlsplit(href)
    host = parsed.hostname or ""
    # The broker only delegates GitHub's HTTPS S3 object PUT capability.
    if (
        parsed.scheme != "https" or parsed.netloc != host or parsed.fragment
        or not host.endswith(".amazonaws.com")
        or not (host.startswith(("s3.", "s3-")) or ".s3." in host or ".s3-" in host)
        or not parsed.path or parsed.path == "/" or "\\" in parsed.path
    ):
        raise RuntimeError("Upload transport is not an HTTPS S3 object destination; hostname=" + host)
    headers = action.get("header", {})
    if not isinstance(headers, dict):
        raise RuntimeError("Malformed upload headers")
    for name, value in headers.items():
        if not isinstance(name, str) or not re.fullmatch(r"[!#$%&'()*+.^_`|~0-9A-Za-z-]+", name) or not isinstance(value, str) or "\r" in value or "\n" in value:
            raise RuntimeError("Malformed upload header")
        if name.lower() in {"proxy-authorization", "cookie", "set-cookie"}:
            raise RuntimeError("Refusing to export a proxy or cookie authentication header")
        if name.lower() == "authorization":
            # AWS documents both forms as request signatures bound to the method
            # and S3 object resource. Neither contains the AWS secret signing key.
            v4 = re.fullmatch(
                r"AWS4-HMAC-SHA256 +Credential=[A-Za-z0-9]+/[0-9]{8}/[a-z0-9-]+/s3/aws4_request, *"
                r"SignedHeaders=([a-z0-9-]+(?:;[a-z0-9-]+)*), *Signature=[0-9a-fA-F]{64}", value
            )
            v2 = re.fullmatch(r"AWS +[A-Za-z0-9]+:[A-Za-z0-9+/]{27}=", value)
            if not (v4 and "host" in v4.group(1).split(";") or v2):
                scheme = value.split(" ", 1)[0]
                diagnostic = {
                    "upload_hostname": host,
                    "auth_scheme": scheme if scheme in {"AWS4-HMAC-SHA256", "AWS", "Basic", "Bearer"} else "unrecognized",
                    "header_names": sorted(headers),
                    "query_keys": sorted({key for key, _ in parse_qsl(parsed.query, keep_blank_values=True) if token not in key and auth.removeprefix("Basic ") not in key}),
                }
                raise RuntimeError("Refusing a non-S3 request-signature authentication header; safe_shape=" + json.dumps(diagnostic))
        if name.lower() == "content-length" and value != str(expected["size"]):
            raise RuntimeError("Upload header disagrees with frozen byte length")
        if name.lower() == "host" and value != host:
            raise RuntimeError("Upload Host header disagrees with the S3 destination")
    return action


phase = os.environ.get("BROKER_PHASE", "issue")
if phase not in {"issue", "await"}:
    raise RuntimeError("BROKER_PHASE must be issue or await")


if phase == "issue":
    batch = request(BASE + "/objects/batch", {"operation": "upload", "transfers": ["basic"], "objects": objects})
    selected = []
    pending = []
    for expected, item in zip(objects, exact_batch(batch), strict=True):
        if item.get("oid") != expected["oid"] or item.get("size") != expected["size"] or item.get("error"):
            raise RuntimeError("LFS batch did not authorize the exact three frozen objects")
        actions = item.get("actions", {})
        upload = actions.get("upload")
        if upload:
            upload = validate_upload(upload, expected)
        selected.append({**expected, "upload": upload})
        verify = actions.get("verify")
        if verify:
            try:
                validate_repo_url(verify["href"], allow_query=True)
            except ValueError:
                parsed = urlsplit(verify["href"])
                diagnostic = {"verify_hostname": parsed.hostname, "verify_path": parsed.path if token not in parsed.path and auth.removeprefix("Basic ") not in parsed.path else "redacted", "query_keys": sorted({key for key, _ in parse_qsl(parsed.query, keep_blank_values=True) if token not in key and auth.removeprefix("Basic ") not in key})}
                raise RuntimeError("Private verify action is outside the expected repository; safe_shape=" + json.dumps(diagnostic)) from None
        pending.append({**expected, "verify": verify})
    private = json.dumps({"manifest_sha256": manifest_sha, "objects": selected}).encode()
    key, nonce = os.urandom(32), os.urandom(12)
    public = serialization.load_pem_public_key(Path(".github/natural-v4-transfer-public.pem").read_bytes())
    wrapped = public.encrypt(key, padding.OAEP(mgf=padding.MGF1(hashes.SHA256()), algorithm=hashes.SHA256(), label=None))
    encrypted = AESGCM(key).encrypt(nonce, private, manifest_sha.encode())
    encoded = {"manifest_sha256": manifest_sha, "wrapped_key": base64.b64encode(wrapped).decode(), "nonce": base64.b64encode(nonce).decode(), "ciphertext": base64.b64encode(encrypted).decode()}
    (OUTPUT / "encrypted-transfer.json").write_text(json.dumps(encoded) + "\n")
    PENDING.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(PENDING, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump({"manifest_sha256": manifest_sha, "objects": pending}, stream)
    os.chmod(PENDING, 0o600)
    (OUTPUT / "verify-locators.json").unlink(missing_ok=True)
    print(json.dumps({"status": "issued_encrypted_object_scoped_uploads", "object_count": len(selected), "private_verification_action_count": sum(bool(item["verify"]) for item in pending), "manifest_sha256": manifest_sha, "no_github_token_exported": True}))
else:
    pending = json.loads(PENDING.read_bytes())
    if pending.get("manifest_sha256") != manifest_sha:
        raise RuntimeError("Private pending actions differ from the frozen manifest")
    waiting = exact_batch({"objects": pending["objects"]})
    waiting = [item for item in waiting if item.get("verify")]
    deadline = time.monotonic() + 600
    while waiting:
        remaining = []
        for item in waiting:
            if time.monotonic() >= deadline:
                raise RuntimeError("Timed out waiting for the three local LFS object uploads")
            action = item["verify"]
            try:
                request(action["href"], {"oid": item["oid"], "size": item["size"]}, action.get("header", {}), allow_query=True, timeout=min(60, max(1, deadline - time.monotonic())))
            except LFSHTTPError as error:
                if error.status not in {404, 409, 422}:
                    raise
                remaining.append(item)
        waiting = remaining
        if waiting:
            print(json.dumps({"status": "awaiting_local_object_uploads", "remaining_object_count": len(waiting)}), flush=True)
            if time.monotonic() >= deadline:
                raise RuntimeError("Timed out waiting for the three local LFS object uploads")
            time.sleep(min(10, deadline - time.monotonic()))
    batch = request(BASE + "/objects/batch", {"operation": "download", "transfers": ["basic"], "objects": objects})
    receipts = []
    for expected, item in zip(objects, exact_batch(batch), strict=True):
        if item.get("oid") != expected["oid"] or item.get("size") != expected["size"] or item.get("error"):
            raise RuntimeError("Uploaded object is unavailable from LFS")
        action = item["actions"]["download"]
        parsed = urlsplit(action["href"])
        if parsed.scheme != "https" or parsed.username or parsed.password or parsed.fragment or parsed.port not in (None, 443):
            raise RuntimeError("LFS download action must use canonical HTTPS")
        headers = action.get("header", {})
        if parsed.hostname == "github.com":
            validate_repo_url(action["href"], allow_query=True)
        elif parsed.hostname and (parsed.hostname.endswith(".amazonaws.com") or parsed.hostname.endswith(".githubusercontent.com")):
            if token in json.dumps(action) or auth in json.dumps(action) or auth.removeprefix("Basic ") in json.dumps(action):
                raise RuntimeError("Refusing to send the GitHub token to object storage")
        else:
            raise RuntimeError("LFS download host is outside GitHub object storage")
        req = urllib.request.Request(action["href"], headers=headers)
        digest, count = hashlib.sha256(), 0
        with safe_open(req, timeout=120) as response:
            while raw := response.read(min(1024 * 1024, expected["size"] + 1 - count)):
                count += len(raw)
                if count > expected["size"]:
                    raise RuntimeError("LFS download exceeds frozen byte length")
                digest.update(raw)
        if digest.hexdigest() != expected["oid"] or count != expected["size"]:
            raise RuntimeError("Uploaded LFS bytes differ from frozen manifest")
        receipts.append({**expected, "download_sha256_verified": True})
    receipt = {"manifest_sha256": manifest_sha, "git_revision": os.environ["GITHUB_SHA"], "objects": receipts, "lfs_upload_completed_and_download_verified": True, "source_replay_scope": "locally verified fixed-source snapshots; GHA fresh Wikimedia replay returned HTTP429", "gpu_started": False}
    (OUTPUT / "upload-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    PENDING.unlink()
    print(json.dumps(receipt))
