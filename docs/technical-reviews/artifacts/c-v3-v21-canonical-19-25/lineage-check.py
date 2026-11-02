"""Read only the hash-bound parent raw manual observations, not review reports."""
import hashlib
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path('/workspace/work/omni-book-integrated-v1')
OUT = ROOT / 'docs/technical-reviews/artifacts/c-v3-v21-canonical-19-25'
path = Path('/workspace/work/c-product-v3-20261101/outputs/c3-development/indexed-aux96-r47-v4/qualification/product-checks.json')
started = datetime.now(UTC).isoformat()
raw = path.read_bytes()
digest = hashlib.sha256(raw).hexdigest()
assert digest == json.loads((ROOT / 'docs/decision-product/v3/metadata/lineage.json').read_text())['parent_manual_failure_sha256']
rows = json.loads(raw)['observations']
errors = [row for row in rows if row['prediction']['model_choice'] != row['expected']]
by_kind = Counter(row['kind'] for row in errors)
info = [row for row in errors if row['kind'] == 'info']
assert info and all(row['permission'] is None and row['ready'] is None for row in info)
receipt = {
    'reviewer_task': '/root/c_v3_technical_coord/canonical_19_25',
    'started_at': started, 'completed_at': datetime.now(UTC).isoformat(),
    'command': 'UV_PROJECT_ENVIRONMENT=/workspace/tiny-perceptron-vlm/.venv PYTHONPATH=/workspace/work/omni-book-integrated-v1 UV_OFFLINE=1 uv run --extra cpu --no-sync python docs/technical-reviews/artifacts/c-v3-v21-canonical-19-25/lineage-check.py',
    'environment': {'python': sys.version.split()[0], 'device': 'cpu', 'operation': 'JSON/raw-observation arithmetic only'},
    'raw_parent': {'path': str(path), 'bytes': len(raw), 'sha256': digest, 'matches_fixed_lineage_reference': True},
    'retained_observations': len(rows), 'raw_model_choice_errors': len(errors),
    'error_counts_by_kind': dict(by_kind),
    'info_errors': {'count': len(info), 'all_have_both_fields_unknown': True, 'task_counts': dict(Counter(row['task'] for row in info))},
    'scope': 'Retained raw parent engineering checks show omitted local unknown-field states. Combined with inspected facts[0]-targeting versus target_views code and same-8000-family hashes, this supports the development narrative. It does not isolate causal gains of retargeting from concurrent data/representation/training changes. No old review report was read and no inference/training/final was run.',
}
with (OUT / 'lineage-execution.json').open('x', encoding='utf-8') as handle:
    json.dump(receipt, handle, ensure_ascii=False, indent=2)
    handle.write('\n')
print(json.dumps({'path': str(OUT / 'lineage-execution.json'), 'raw_model_choice_errors': len(errors), 'errors_by_kind': dict(by_kind)}, ensure_ascii=False))
