import json, tarfile, hashlib, sys, math
from pathlib import Path
from scripts.course_experiments.text import _deduplicate_text
from scripts.course_experiments.common import split_records, text_examples
from tiny_perceptron.data import ByteTokenizer

b = Path('docs/technical-reviews/artifacts/p7_technical_a')
raw = Path('docs/course-experiments/results/real_text.json')
r = json.loads(raw.read_text())
q = r['results']['runs']['tinystories']
archive = Path('outputs/phase7-source-cache/p7_technical_a/tinystories-v1.tar.gz')
assert hashlib.sha256(archive.read_bytes()).hexdigest() == r['assets'][0]['archive_sha256']
with tarfile.open(archive) as tf:
    name = next(x for x in tf.getnames() if x.endswith('tinystories-train-512.jsonl'))
    data = tf.extractfile(name).read()
assert hashlib.sha256(data).hexdigest() == next(x['sha256'] for x in r['assets'][0]['files'] if x['path'].endswith('tinystories-train-512.jsonl'))
records = [json.loads(x) for x in data.splitlines()]
parts = split_records(_deduplicate_text(records), seed=r['seed'])
checks = {}
for split, rows in parts.items():
    serial = ''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in rows).encode()
    sha = hashlib.sha256(serial).hexdigest()
    assert sha == q['data'][split]['sha256']
    examples = text_examples(rows, mode='text', max_length=128)
    targets = sum(len(y) for x, y in examples)
    checks[split] = {'records': len(rows), 'sha256': sha, 'effective_targets': targets, 'chunks': len(examples), 'utf8_bytes': sum(len(x['text'].encode()) for x in rows)}
    if split == 'validation':
        assert targets == 42453 and len(rows) == 51
nll = {stage: q[stage]['validation']['nll_sum'] / q[stage]['validation']['effective_tokens'] for stage in ['before', 'after']}
assert all(abs(nll[k] - q[k]['validation']['nll']) < 1e-12 for k in nll)
samples = {stage: {side: len(q[stage][side]['samples']) for side in ['validation', 'test']} for stage in ['before', 'after']}
assert all(x == 8 for v in samples.values() for x in v.values())
ex = q['after']['validation']['samples'][1]
assert ByteTokenizer().decode(ex['generated_ids']) == ex['generated']
assert len(ex['generated_ids']) == 32
f = json.load(open('docs/course-experiments/results/text_foundation.json'))['results']
fv = json.loads((b / 'foundation-reconstructed-validation.jsonl').read_text().strip())
assert fv['text'] == 'color=blue;shape=circle;side=left.'
manual = {}
for label, prediction in [('TF', [1, 2, 3, 4]), ('free', [1, 2, 0, 0]), ('repair_last_only', [1, 2, 0, 4])]:
    manual[label] = {'token_accuracy': sum(a == z for a, z in zip([1, 2, 3, 4], prediction, strict=True)) / 4, 'exact': prediction == [1, 2, 3, 4]}
try:
    list(zip([1, 2, 3, 4], [1, 2], strict=True))
    raise AssertionError
except ValueError as e:
    strict_error = str(e)
source_checks = {}
for path in ['scripts/course_experiments/text.py', 'scripts/course_experiments/common.py', 'tiny_perceptron/model.py', 'tiny_perceptron/data.py']:
    sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    assert sha == r['code_sha256'][path]
    source_checks[path] = sha
out = {'command': 'PYTHONPATH=. .venv/bin/python docs/technical-reviews/artifacts/p7_technical_a/check-5-8.py', 'python': sys.version, 'device': 'CPU no model training', 'manual': manual, 'strict_length_error': strict_error, 'same_top_probability_nll': {str(p): -math.log(p) for p in [.6, .9]}, 'raw_path': str(raw), 'raw_sha256': hashlib.sha256(raw.read_bytes()).hexdigest(), 'archive_original_url': 'https://media.githubusercontent.com/media/birdhackor/tiny-perceptron-vlm/5581462ef01959636425eb7795ae3153142dbfeb/assets/training/tinystories-v1.tar.gz', 'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(), 'dataset_file_sha256': hashlib.sha256(data).hexdigest(), 'reconstructed_splits': checks, 'validation_nll': nll, 'story_sample': ex, 'sample_counts': samples, 'toy_validation_text': fv['text'], 'toy_before_nll': f['before']['validation']['nll'], 'toy_after_nll': f['after']['validation']['nll'], 'toy_after_sample': f['after']['validation']['samples'][0], 'original_source_shas_match_current': source_checks, 'not_retrained': 'No800-step stories or600-step toy retraining; no old checkpoints replayed; saved nll_sum/count and generation only', 'initial_failed_check': 'First stdin attempt compared git LFS pointer bytes to archive SHA; next tar attempt failed not gzip. Downloaded original revision LFS media into own ignored cache and verified object SHA before reconstruction. No curriculum issue.'}
p = b / '5-8-original-checks.json'
assert not p.exists()
p.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(out, ensure_ascii=False, indent=2))
