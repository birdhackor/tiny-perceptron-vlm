"""Recount released historical predictions; this does not rerun the models."""
import gzip
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path('/workspace/tiny-perceptron-vlm')
HERE = Path(__file__).resolve().parent
raw = Path('/workspace/work/omni-technical-readme-en-v1/supplement/product-predictions.jsonl.gz')
counts, correct = Counter(), Counter()
with gzip.open(raw, 'rt') as stream:
    for line in stream:
        row = json.loads(line)
        if row['product'] == 'b' and row['split'] == 'test':
            key = 'b:test:depth5:' + row['task']
            counts[key] += 1
            correct[key] += row['answers_by_depth']['5'] == row['gold']
        elif row['product'] == 'c':
            key = 'c:' + row['split'] + ':' + row['question']
            counts[key] += 1
            predicted = row.get('selected')
            if predicted is None:
                predicted = row['candidates'][max(range(len(row['raw_scores'])), key=lambda i: row['raw_scores'][i])]
            correct[key] += predicted == row['gold']
freeze = ROOT / 'docs/lesson-products/results/settings-freeze-v1.json'
test = json.loads((ROOT / 'docs/lesson-products/results/settings-freeze-v1.test-started.json').read_bytes())
report = json.loads((ROOT / 'docs/lesson-products/results/final-report-v1.json').read_bytes())
result = {
    'created_at': datetime.now(UTC).isoformat(),
    'method': 'Stream the pinned public gzip. Compare B depth-five answer and C candidate argmax to gold; positive scalar temperature preserves argmax.',
    'raw_path': str(raw), 'raw_sha256': hashlib.sha256(raw.read_bytes()).hexdigest(),
    'counts': {key: {'correct':correct[key], 'count':value} for key,value in counts.items()},
    'actual_freeze_sha256': hashlib.sha256(freeze.read_bytes()).hexdigest(),
    'test_start_freeze_sha256': test['settings_freeze_sha256'],
    'report_freeze_sha256': report['settings_freeze_sha256'],
    'test_started_at': test['recorded_at'], 'report_recorded_at': report['recorded_at'],
    'denominators': {'b_test_families_boxes':1478, 'b_test_families_sum':10, 'b_depth':5, 'b_train_seed':7421, 'c_train_seed':9041},
    'scope': 'Historical published fixed grammar/family data only; no GPU or new-model inference/training. Wording-shift rows are reported separately from fixed test.',
}
p = HERE / 'raw-counts-summary.json'
with p.open('x') as stream:
    json.dump(result, stream, ensure_ascii=False, indent=2)
    stream.write('\n')
print(json.dumps(result, ensure_ascii=False))
