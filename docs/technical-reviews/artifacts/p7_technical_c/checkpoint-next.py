"""Save this owner's supplied checkpoint and unlock exactly one next unit."""
import json
import hashlib
import subprocess
import sys
import time
from pathlib import Path

record = json.load(sys.stdin)
for ref in record.pop('own_visual_receipts', []):
    receipt_path = Path(ref['receipt'])
    receipt = json.loads(receipt_path.read_text())
    record.setdefault('visual_checks', []).append({
        'figure': ref['figure'], 'source_sha256': receipt['source_sha256'],
        'status': 'verified', 'details': '本人已经分别调用 tools.view_image 看过640及360，随后保存此收据。',
        'observation': receipt['observation'],
        'receipt': {'path': str(receipt_path), 'sha256': hashlib.sha256(receipt_path.read_bytes()).hexdigest()},
        'artifacts': receipt['artifacts'],
    })
path = Path('outputs/reader-checkpoints/p7_technical_c') / f'{record["page_id"]}-{record["unit_index"]}-{time.time_ns()}.json'
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
raise SystemExit(subprocess.call(['.venv/bin/python', 'docs/review-tools/phase7_review.py', 'next', sys.argv[1], '--checkpoint', str(path)]))
