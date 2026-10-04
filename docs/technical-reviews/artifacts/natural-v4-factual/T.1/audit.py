"""Independent bounded T.1 audit. No fetching, training, or raw public copies."""
import hashlib
import io
import json
import platform
import random
import subprocess
import sys
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from scripts.course_experiments.modalities import _resample_8_to_16
from scripts.prepare_data import generate_records
from tiny_perceptron.modal_data import modal_example

torch.set_num_threads(1)
def sha(b):
    return hashlib.sha256(b).hexdigest()
def load(path):
    return json.loads((ROOT / path).read_text())
def rows(path):
    return [json.loads(line) for line in (ROOT / path).read_text().splitlines() if line]
def independent_split(records):
    groups = defaultdict(list)
    for row in records:
        groups[row['family']].append(row)
    keys = sorted(groups)
    random.Random(42).shuffle(keys)
    a, b = int(.8 * len(keys)), int(.9 * len(keys))
    return {name: [r for key in side for r in groups[key]] for name, side in
            [('train', keys[:a]), ('validation', keys[a:b]), ('test', keys[b:])]}

out = {'environment': {'python': platform.python_version(), 'torch': torch.__version__,
       'numpy': np.__version__, 'soundfile': sf.__version__, 'device': 'cpu',
       'cuda_available': torch.cuda.is_available(), 'platform': platform.platform()},
       'scope': 'Own CPU source/data/numeric audit of existing originals and existing CUDA records; no training or inference replication.'}
cmd = [str(ROOT / '.venv/bin/python'), 'scripts/fetch_training_assets.py', '--list']
run = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
out['list_execution'] = {'command': ' '.join(cmd), 'exit_code': run.returncode,
                         'stdout': run.stdout, 'stderr': run.stderr}
assert run.returncode == 0
assets = load('assets/training/manifest.json')['assets']
out['archives'] = []
for asset in assets:
    archive = ROOT / asset['archive']
    assert archive.stat().st_size == asset['archive_bytes']
    assert sha(archive.read_bytes()) == asset['archive_sha256']
    expected = {r['path']: r for r in asset['files']}
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        assert {m.name for m in members} == set(expected)
        for member in members:
            data = tar.extractfile(member).read()
            assert len(data) == expected[member.name]['bytes']
            assert sha(data) == expected[member.name]['sha256']
    metadata = load(asset['source_metadata'])
    out['archives'].append({'id': asset['id'], 'archive_sha256': asset['archive_sha256'],
        'files_verified': len(expected), 'version': asset['version'], 'license': asset['license'],
        'source_locator': metadata.get('source_url', metadata.get('source_repository', metadata.get('upstream_repository', metadata.get('source', {}).get('repository') if isinstance(metadata.get('source'),dict) else metadata.get('dataset')))),
        'source_version_recorded': any(k in metadata for k in ['source_revision', 'source_sha', 'source_repository_revision', 'revision', 'hf_revision']) or bool(metadata.get('source', {}).get('revision') if isinstance(metadata.get('source'),dict) else None)})

text_report = load('docs/course-experiments/results/real_text.json')
out['text'] = {}
for name, path in [('tinystories', 'data/training/text-initial/tinystories-train-512.jsonl'),
                   ('chinese-poetry', 'data/training/text-initial/chinese-classical-train-365.jsonl')]:
    raw = rows(path)
    if name == 'tinystories':
        prefix = (ROOT / 'data/training/text-initial/tinystories-train-prefix-complete.txt').read_text()
        complete = [p.strip() for p in prefix.split('<|endoftext|>')[:-1]]
        assert len(raw) == len(complete) == 512
        assert [r['text'] for r in raw] == complete
        assert all(r['source_split'] == 'train' for r in raw)
        completeness = '512 delimiter-terminated original stories match every text field; no partial tail.'
    else:
        poems = load('data/training/text-initial/tang300-source.json')
        rendered = ['\n'.join([p['title'], p['author'], *p['paragraphs']]) for p in poems]
        unique = list(dict.fromkeys(rendered))
        assert len(poems) == 366 and len(unique) == 365
        assert [r['text'] for r in raw] == unique
        assert all(r['source_split'] == 'unpartitioned' for r in raw)
        completeness = '366 original complete title/author/paragraph records; one duplicate removed; 365 full text records match.'
    records, seen = [], set()
    for row in raw:
        key = ' '.join(row['text'].split())
        if key not in seen:
            records.append({**row, 'family': sha(key.encode())})
            seen.add(key)
    parts = independent_split(records)
    original = text_report['results']['runs'][name]
    stats = {}
    for split, part in parts.items():
        serialized = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in part).encode()
        assert sha(serialized) == original['data'][split]['sha256']
        assert len(part) == original['data'][split]['records']
        stats[split] = {'records': len(part), 'families': len({r['family'] for r in part}),
                        'sha256': sha(serialized)}
    sets = [{r['family'] for r in part} for part in parts.values()]
    assert not any(a & b for i, a in enumerate(sets) for b in sets[i+1:])
    assert original['training']['steps'] == 800 and len(original['training']['history']) > 1
    out['text'][name] = {'source_count': len(raw), 'completeness': completeness,
       'splits': stats, 'original_updates': 800, 'original_seed': text_report['seed'],
       'source_revision': raw[0]['source_revision'], 'license': raw[0]['license'],
       'near_duplicate_limit': 'Only complete-content whitespace-normalized fingerprints; no near-duplicate clustering.'}

generated = generate_records('attributes-sft')
selected = [r for r in generated if r['messages'][0]['content'] in [
 'color=red;shape=square;pitch=low;shape?', 'color=red;shape=square;pitch=low;describe',
 'color=blue;shape=circle;pitch=high;shape?']]
assert len(selected) == 3
assert selected[0]['family'] == selected[1]['family'] == 'red:square:low'
assert [r['messages'][1]['content'] for r in selected] == ['square','square','circle']
out['attributes'] = {'rows': selected, 'records': len(generated),
                     'families': len({r['family'] for r in generated})}

modal_report = load('docs/course-experiments/results/real_modal.json')
out['modal_runtime'] = {k: modal_report[k] for k in ['revision','device','seed','torch_version',
            'python_version','gpu','timing_scope','step_scale','status']}
fashion = rows('data/training/vision-initial/train.jsonl')
source = pq.read_table(ROOT / 'data/training/vision-initial/upstream/fashion-mnist-train.parquet')
assert source.num_rows == 60000
labels = source.column('label').to_pylist()
first_five, counts = [], Counter()
for index, label in enumerate(labels):
    if counts[label] < 5:
        first_five.append(index)
        counts[label] += 1
assert [r['source_row'] for r in fashion] == first_five
groups = defaultdict(list)
for r in fashion:
    assert r['source_split'] == 'train'
    img = ROOT / 'data/training/vision-initial' / r['image']
    assert sha(img.read_bytes()) == r['sha256']
    original_image = source.column('image')[r['source_row']].as_py()['bytes']
    with Image.open(img) as a, Image.open(io.BytesIO(original_image)) as b:
        assert a.mode == 'L' and a.size == (28,28)
        assert np.array_equal(np.asarray(a), np.asarray(b))
    groups[r['label']].append(r)
fashion_parts = {s: [] for s in ['train','validation','test']}
for label, group in groups.items():
    for index, row in enumerate(group):
        side = 'train' if index < 3 else 'validation' if index == 3 else 'test'
        fashion_parts[side].append(row)
for s, part in fashion_parts.items():
    report_rows = modal_report['results']['fashion-mnist']['data']['splits'][s]['records']
    assert [Path(r['image']).name for r in part] == [r['family'] for r in report_rows]
out['fashion'] = {'source_rows':60000, 'pilot_rows':len(fashion), 'per_label':dict(Counter(r['label'] for r in fashion)),
                 'custom_splits':{s:{'records':len(p),'per_label':dict(Counter(r['label'] for r in p))} for s,p in fashion_parts.items()}}

fsdd = modal_report['results']['fsdd']
conversions = {r['source']:r for r in fsdd['resampling']}
out['fsdd'] = {'splits':{}, 'headers_verified':0, 'resampling_records_verified':0}
example = None
for side, speaker in [('train','jackson'),('validation','nicolas'),('test','theo')]:
    part = rows(f'data/training/fsdd-initial/{side}.jsonl')
    assert len(part) == 20 and {r['speaker'] for r in part} == {speaker}
    assert Counter(r['digit_label'] for r in part) == Counter({i:2 for i in range(10)})
    assert {r['recording_index'] for r in part} == {5,6}
    out['fsdd']['splits'][side] = {'speaker':speaker,'recordings':len(part),'digits':dict(Counter(r['digit_label'] for r in part)),'original_indices':[5,6]}
    for row in part:
        path = ROOT / 'data/training/fsdd-initial' / row['path']
        info = sf.info(path)
        assert info.samplerate == 8000 and info.channels == 1 and info.subtype == 'PCM_16'
        assert sha(path.read_bytes()) == row['sha256']
        conv = conversions[path.name]
        assert conv['source_sha256'] == row['sha256'] and conv['samples_before'] == info.frames
        assert conv['source_rate'] == 8000 and conv['target_rate'] == 16000 and conv['samples_after'] == info.frames*2
        out['fsdd']['headers_verified'] += 1
        out['fsdd']['resampling_records_verified'] += 1
        if example is None:
            values, rate = sf.read(path, dtype='float32')
            times = np.arange(len(values)*2)/2
            indices = np.floor(times).astype(int)[:,None] + np.arange(-15,17)[None,:]
            distance = times[:,None] - indices
            valid = (indices>=0)&(indices<len(values))
            kernel = np.sinc(distance)*np.where(abs(distance)<16,.5+.5*np.cos(np.pi*distance/16),0)*valid
            derived = ((kernel*values[np.clip(indices,0,len(values)-1)]).sum(axis=1)/kernel.sum(axis=1)).astype('float32')
            repository_result = _resample_8_to_16(values)
            diff = float(np.max(abs(derived-repository_result)))
            assert diff <= 1e-7
            buf = io.BytesIO(); sf.write(buf, repository_result,16000,format='WAV',subtype='FLOAT')
            fresh_sha = sha(buf.getvalue())
            try:
                modal_example({'audio':row['path'],'question':'digit?','answer':'0'},path.parent.parent,'audio')
            except ValueError as error:
                rejected = str(error)
            else:
                raise AssertionError('8kHz input was not rejected')
            example={'file':path.name,'before_samples':len(values),'after_samples':len(derived),
                     'before_seconds':len(values)/8000,'after_seconds':len(derived)/16000,
                     'independent_numpy_vs_torch_max_difference':diff,'fresh_derived_wav_sha256':fresh_sha,
                     'original_record_derived_wav_sha256':conv['derived_sha256'],
                     'full_file_sha_matches_original_run':fresh_sha == conv['derived_sha256'],
                     'raw_training_entry_rejection':rejected}
out['fsdd']['actual_resampling_probe'] = example
out['scores'] = {}
for asset in ['fashion-mnist','fsdd']:
    run = modal_report['results'][asset]
    score = {}
    for side in ['validation','test']:
        data = run['data']['splits'][side]['records']
        samples = run[side]['samples']
        assert len(samples) == len(data)
        assert [r['target'] for r in samples] == [r['answer'] for r in data]
        assert [r['family'] for r in samples] == [r['family'] for r in data]
        recomputed = sum(s['generated'] == s['target'] and not s['generation_error'] and s['invalid_special_tokens'] == 0 for s in samples)
        assert recomputed == run[side]['correct']
        assert recomputed/len(samples) == run[side]['exact_match']
        score[side] = {'correct':recomputed,'denominator':len(samples),'rate':recomputed/len(samples),
                       'pairs':[[r['target'],r['generated']] for r in samples]}
    out['scores'][asset] = {'updates':run['training']['steps'],'seed':modal_report['seed'],
        'effective_training_targets':run['training']['effective_targets'],'scores':score}
assert out['scores']['fashion-mnist']['scores']['test']['correct'] == 3
assert out['scores']['fsdd']['scores']['test']['correct'] == 3
print(json.dumps(out,ensure_ascii=False,indent=2))
