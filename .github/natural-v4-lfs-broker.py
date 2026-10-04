import base64
import hashlib
import json
import os
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

REPO = "birdhackor/tiny-perceptron-vlm"
BASE = "https://github.com/" + REPO + ".git/info/lfs"
MANIFEST = Path("docs/natural-assistant/v4/manifest.json")
OUTPUT = Path("outputs/natural-v4/lfs-broker")
OUTPUT.mkdir(parents=True, exist_ok=True)
manifest = json.loads(MANIFEST.read_bytes())
manifest_sha = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
objects = [{"oid": a["sha256"], "size": a["bytes"]} for a in manifest["archives"]]
assert len(objects) == 3
auth = "Basic " + base64.b64encode(("x-access-token:" + os.environ["GITHUB_TOKEN"]).encode()).decode()


def request(url, body):
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname != "github.com" or parsed.query or not parsed.path.startswith("/" + REPO + ".git/info/lfs/"):
        raise ValueError("Refusing a verification endpoint outside the expected repository")
    payload = json.dumps(body).encode()
    req = urllib.request.Request(url, data=payload, headers={"Authorization": auth, "Accept": "application/vnd.git-lfs+json", "Content-Type": "application/vnd.git-lfs+json"})
    with urllib.request.urlopen(req, timeout=60) as response:
        raw = response.read()
        return json.loads(raw) if raw else {}


if os.environ.get("BROKER_PHASE", "issue") == "issue":
    batch = request(BASE + "/objects/batch", {"operation": "upload", "transfers": ["basic"], "objects": objects})
    selected = []
    locators = []
    for expected, item in zip(objects, batch["objects"], strict=True):
        if item.get("oid") != expected["oid"] or item.get("size") != expected["size"] or item.get("error"):
            raise RuntimeError("LFS batch did not authorize the exact three frozen objects")
        actions = item.get("actions", {})
        upload = actions.get("upload")
        if upload:
            host = urlsplit(upload["href"]).hostname or ""
            if urlsplit(upload["href"]).scheme != "https" or not host.endswith(".amazonaws.com"):
                raise RuntimeError("Upload transport is not an object-scoped HTTPS S3 destination: " + host)
        selected.append({**expected, "upload": upload})
        if actions.get("verify"):
            verify = actions["verify"]["href"]
            parsed = urlsplit(verify)
            if parsed.scheme != "https" or parsed.hostname != "github.com" or parsed.query or not parsed.path.startswith("/" + REPO + ".git/info/lfs/"):
                raise RuntimeError("Verification locator must contain no credential or query")
            locators.append({**expected, "href": verify})
    private = json.dumps({"manifest_sha256": manifest_sha, "objects": selected}).encode()
    key, nonce = os.urandom(32), os.urandom(12)
    public = serialization.load_pem_public_key(Path(".github/natural-v4-transfer-public.pem").read_bytes())
    wrapped = public.encrypt(key, padding.OAEP(mgf=padding.MGF1(hashes.SHA256()), algorithm=hashes.SHA256(), label=None))
    encrypted = AESGCM(key).encrypt(nonce, private, manifest_sha.encode())
    encoded = {"manifest_sha256": manifest_sha, "wrapped_key": base64.b64encode(wrapped).decode(), "nonce": base64.b64encode(nonce).decode(), "ciphertext": base64.b64encode(encrypted).decode()}
    (OUTPUT / "encrypted-transfer.json").write_text(json.dumps(encoded) + "\n")
    (OUTPUT / "verify-locators.json").write_text(json.dumps(locators) + "\n")
    print(json.dumps({"status": "issued_encrypted_object_scoped_uploads", "object_count": len(selected), "verification_locator_count": len(locators), "manifest_sha256": manifest_sha, "no_github_token_exported": True}))
else:
    locators = json.loads(os.environ["VERIFY_LOCATORS"])
    expected_by_oid = {item["oid"]: item for item in objects}
    for item in locators:
        if {"oid": item["oid"], "size": item["size"]} != expected_by_oid[item["oid"]]:
            raise RuntimeError("Verification does not match the frozen object")
        request(item["href"], {"oid": item["oid"], "size": item["size"]})
    batch = request(BASE + "/objects/batch", {"operation": "download", "transfers": ["basic"], "objects": objects})
    receipts = []
    for expected, item in zip(objects, batch["objects"], strict=True):
        if item.get("oid") != expected["oid"] or item.get("size") != expected["size"] or item.get("error"):
            raise RuntimeError("Uploaded object is unavailable from LFS")
        action = item["actions"]["download"]
        req = urllib.request.Request(action["href"], headers=action.get("header", {}))
        digest, count = hashlib.sha256(), 0
        with urllib.request.urlopen(req, timeout=120) as response:
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
    print(json.dumps(receipt))
