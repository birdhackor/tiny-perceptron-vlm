"""Opaque preservation before this owner's narrow 2026-10-06 callback."""
from pathlib import Path
import hashlib
import json
import platform
import sys

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent
source = ROOT / 'docs/technical-reviews/1.14.json'
raw = source.read_bytes()  # Do not parse or print the prior report body here.
digest = hashlib.sha256(raw).hexdigest()
target = OUT / ('own-prior-report-' + digest + '.raw.json')
if target.exists():
    assert target.read_bytes() == raw
else:
    target.write_bytes(raw)
receipt = {'command': '.venv/bin/python docs/technical-reviews/artifacts/phase4-1_14-independent/callback-20261006/archive_prior.py', 'exit_code': 0, 'prior_source': source.relative_to(ROOT).as_posix(), 'prior_opaque_file': target.relative_to(ROOT).as_posix(), 'prior_sha256': digest, 'archive_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'environment': {'python': platform.python_version(), 'python_executable': sys.executable, 'platform': platform.platform(), 'device': 'no model execution'}}
(OUT / 'archive.receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(receipt, ensure_ascii=False))
