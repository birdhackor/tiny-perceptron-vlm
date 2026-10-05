from pathlib import Path
import hashlib
import json
import sys
ROOT = Path(__file__).resolve().parents[4]
PROOF = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from scripts.course_experiments.common import split_records, records_sha256
from tiny_perceptron.data import load_jsonl
records = load_jsonl(PROOF / 'raw/tinystories-train-512.jsonl')
for record in records:
    record['family'] = record.get('text_sha256', records_sha256([{'text': record['text']}]))
data = split_records(records, 42)
serialized = (json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
raw = json.loads((PROOF / 'raw/moe.json').read_bytes())
expected = next(item for item in raw['artifacts'] if item['path'] == 'dataset.json')
observed = hashlib.sha256(serialized).hexdigest()
assert len(serialized) == expected['bytes'] and observed == expected['sha256']
print('RECONSTRUCTED_DATASET', len(serialized), observed, 'exact raw /artifacts/1 identity; in-memory serialization only, no model execution')
