"""Metadata helpers for manually supplied assessments; no source truth or judgments inferred."""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = "docs/technical-reviews/artifacts/p7_technical_a/"

def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()

def original(identifier, title, url, version, inspection_note):
    return {"id": identifier, "kind": "official_docs", "title": title, "verified": True, "url": url, "version": version, "authority_reason": "發布此契約的官方專案原始文件或原始碼", "checked_original": True, "inspection_note": inspection_note, "accessed_on": "2026-10-07"}

def repository(identifier, path, note):
    return {"id": identifier, "kind": "repository_code", "title": path, "verified": True, "path": path, "sha256": digest(path), "version": "3c0f3254d4762f5ed87e77bd55e5eaf9be2ec248 plus actual file SHA", "inspection_note": note}

def execution_source(identifier, artifact_id, title):
    return {"id": identifier, "kind": "execution", "title": title, "verified": True, "artifact_id": artifact_id}

def snapshot(identifier, path, description):
    return {"id": identifier, "kind": "source_snapshot", "path": path, "sha256": digest(path), "description": description}

def execution(identifier, path, command, result, environment, description):
    return {"id": identifier, "kind": "execution", "path": path, "sha256": digest(path), "command": command, "result": result, "environment": environment, "description": description}

def evidence(source_id, locator, supports):
    return {"source_id": source_id, "locator": locator, "supports": supports}

def claim(identifier, kind, statement, location, scope, references, artifact_ids, verification=None, status="verified"):
    result = {"id": identifier, "kind": kind, "statement": statement, "location": location, "scope": scope, "status": status, "evidence": references, "artifact_ids": artifact_ids}
    if verification is not None:
        result["verification"] = verification
    return result

def check(status, details, claim_ids):
    return {"status": status, "details": details, "claim_ids": claim_ids}

def visual_checks(page_id):
    result = []
    seen = set()
    for path in sorted((ROOT / BASE / "checkpoints").glob("*.json")):
        checkpoint = json.loads(path.read_text())
        if checkpoint["page_id"] != page_id:
            continue
        for item in checkpoint.get("visual_checks", []):
            key = (item["figure"], item["source_sha256"])
            if key not in seen:
                result.append(item)
                seen.add(key)
    return result

def placement(receipt_path, details):
    receipt = json.loads((ROOT / receipt_path).read_text())
    return {"status": "verified", "required": True, "source_sha256": receipt["source_sha256"], "details": details, "observation": receipt["observation"], "receipt": {"path": receipt_path, "sha256": digest(receipt_path)}, "artifacts": receipt["artifacts"]}

def save(record):
    subprocess.run([str(ROOT / ".venv/bin/python"), str(ROOT / BASE / "save_page.py")], cwd=ROOT, input=json.dumps(record, ensure_ascii=False), text=True, check=True)
