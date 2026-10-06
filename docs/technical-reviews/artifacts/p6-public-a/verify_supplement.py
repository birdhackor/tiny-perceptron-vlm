"""Read original metadata and archive headers only; no inference or training."""
from pathlib import Path
import collections, hashlib, io, json, subprocess, tarfile, wave
import pyarrow.parquet as pq

ROOT = Path.cwd()
OUT = ROOT / 'docs/technical-reviews/artifacts/p6-public-a'
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    return json.loads(Path(p).read_bytes())
result = {'course_execution': [], 'pilot_media': {}, 'export_model_config': {}}
plan = load('docs/course-experiments/plan.json')
for experiment in plan['sequence']:
    p = Path(experiment['evidence'])
    d = load(p)
    keys = ['experiment_id', 'revision', 'device', 'torch_version', 'python_version', 'gpu', 'status']
    result['course_execution'].append({'path': str(p), 'sha256': sha(p),
        'inspected_pointers': ['/' + k for k in keys], **{k: d[k] for k in keys}})
manifest = load('assets/training/manifest.json')
git = Path(subprocess.check_output(['git', 'rev-parse', '--git-common-dir'], text=True).strip())
for asset in manifest['assets']:
    if asset['id'] not in ['fashion-mnist', 'fsdd']:
        continue
    oid = asset['archive_sha256']
    archive = Path(asset['archive'])
    if archive.stat().st_size != asset['archive_bytes']:
        archive = git / 'lfs/objects' / oid[:2] / oid[2:4] / oid
    assert sha(archive) == oid
    with tarfile.open(archive, 'r:gz') as tar:
        if asset['id'] == 'fashion-mnist':
            p = 'vision-initial/upstream/fashion-mnist-train.parquet'
            content = tar.extractfile(p).read()
            rows = pq.read_metadata(io.BytesIO(content)).num_rows
            records = len(tar.extractfile('vision-initial/train.jsonl').read().splitlines())
            assert rows == 60000 and records == 50
            result['pilot_media']['fashion-mnist'] = {'archive_sha256': oid,
                'member': p, 'member_sha256': hashlib.sha256(content).hexdigest(),
                'parquet_row_count': rows, 'pilot_train_record_count': records}
        else:
            counts = {}
            for split in ['train', 'validation', 'test']:
                counts[split] = len(tar.extractfile(f'fsdd-initial/{split}.jsonl').read().splitlines())
            sample_rates = collections.Counter()
            for m in tar:
                if m.isfile() and m.name.endswith('.wav'):
                    with wave.open(io.BytesIO(tar.extractfile(m).read()), 'rb') as w:
                        sample_rates[w.getframerate()] += 1
            assert counts['validation'] == counts['test'] == 20
            assert dict(sample_rates) == {8000: 60}
            result['pilot_media']['fsdd'] = {'archive_sha256': oid,
                'record_counts': counts, 'wav_sample_rates': dict(sample_rates)}
for arch in ['moe', 'dense']:
    p = next((OUT / 'originals').glob(f'*__{arch}-joint__model-config.json'))
    d = load(p)
    keys = [k for k in d if any(t in k for t in ['expert', 'top', 'architecture'])]
    result['export_model_config'][arch] = {'path': str(p.relative_to(ROOT)),
        'sha256': sha(p), 'inspected_pointers': ['/' + k for k in keys],
        'fields': {k: d[k] for k in keys}}
(OUT / 'supplemental-measurements.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'course_environment_counts': {str(k): v for k, v in collections.Counter(
    (x['device'], x['gpu'], x['torch_version']) for x in result['course_execution']).items()},
    'pilot_media': result['pilot_media'], 'export_model_config': result['export_model_config']},
    ensure_ascii=False, indent=2, default=str))
