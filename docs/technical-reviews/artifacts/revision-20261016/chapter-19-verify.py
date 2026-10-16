import ast
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = next(path for path in Path(__file__).resolve().parents if (path / 'docs/technical-review-guide.md').is_file())
OUT = ROOT / 'docs/technical-reviews/artifacts/revision-20261016/chapter-19-check.json'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    return json.loads((ROOT / path).read_text())

def fetch_json(url):
    with urlopen(Request(url, headers={'User-Agent': 'chapter-19-independent-technical-review'}), timeout=30) as response:
        raw = response.read()
        return json.loads(raw), {'url': url, 'http_status': response.status, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}

result = {
    'reviewer_task': '/root/tech_20261016_chapter_19',
    'accessed_at': datetime.now(timezone.utc).isoformat(),
    'command': '.venv/bin/python docs/technical-reviews/artifacts/revision-20261016/chapter-19-verify.py',
    'environment': {'python': platform.python_version(), 'system': platform.platform(), 'device': 'CPU for JSON/hash/HTTP provenance inspection; original CUDA training was not rerun'},
    'scope': 'Only chapter-19 introduction: historical release, from-random provenance, and explicitly listed data sources. No new training or quality/latency experiment.',
}
manifest = read('docs/selftrained/v2-manifest.json')
manifest_sha = sha(ROOT / 'docs/selftrained/v2-manifest.json')
assert manifest['initialization'] == 'random'
archived = read('docs/selftrained/results/training-raw/index.json')
stages = ['moe-pretrain', 'moe-sft', 'moe-vision', 'moe-ocr', 'moe-audio', 'moe-joint', 'moe-weighted', 'moe-native']
rows = []
previous = None
for stage in stages:
    base = 'docs/selftrained/results/training-raw/' + stage
    receipts = {}
    for name in ['receipt.json', 'raw/execution.json', 'raw/train-receipt.json']:
        path = base + '/' + name
        expected = next(x for x in archived['files'] if x['copy'] == path)
        actual = sha(ROOT / path)
        assert actual == expected['sha256']
        receipts[name] = {'path': path, 'sha256': actual}
    receipt = read(base + '/receipt.json')
    execution = read(base + '/raw/execution.json')
    trained = read(base + '/raw/train-receipt.json')
    files = {x['path']: x for x in receipt['files']}
    assert execution['status'] == 'completed' and execution['returncode'] == 0
    assert execution['manifest_sha256'] == manifest_sha
    assert trained['completed_requested_steps'] and not trained['interrupted']
    assert trained['origin']['kind'] == 'all-neural-weights-random'
    assert trained['architecture'] == 'moe'
    assert receipts['raw/execution.json']['sha256'] == files['execution.json']['sha256']
    assert receipts['raw/train-receipt.json']['sha256'] == files['train-receipt.json']['sha256']
    init = execution['job'].get('init_checkpoint')
    if previous is None:
        assert init is None and '--init-checkpoint' not in execution['command'] and '--resume' not in execution['command']
    else:
        assert init['run_id'] == previous['run_id'] and init['sha256'] == previous['best_sha256']
    row = {
        'stage': stage, 'run_id': execution['run_id'], 'training_revision': execution['revision'],
        'historical_device': execution['command'][execution['command'].index('--device') + 1],
        'historical_command': execution['command'],
        'status': execution['status'], 'returncode': execution['returncode'],
        'origin': {'kind': trained['origin']['kind'], 'seed': trained['origin']['seed']},
        'init_checkpoint': init,
        'steps': trained['steps'], 'batch_size': execution['job']['batch_size'],
        'sample_draws': trained['comparison']['sample_draws'],
        'tokens': trained['tokens'], 'target_tokens': trained['target_tokens'],
        'train_records': trained['train_records'], 'validation_records': trained['validation_records'],
        'trainable_parameters': trained['trainable_parameters'],
        'best_sha256': files['best.pt']['sha256'],
        'model_safetensors': files['model.safetensors'], 'receipts': receipts,
    }
    rows.append(row)
    previous = row
result['historical_training'] = {'manifest_sha256': manifest_sha, 'stages': rows, 'all_eight_source_links_and_archive_hashes_match': True, 'original_training_rerun': False}

public = read('docs/selftrained/results/v2-final-public-results.json')
moe = public['architectures']['moe']
assert moe['selected_source']['sha256'] == rows[-1]['best_sha256']
assert moe['safe_weights']['sha256'] == rows[-1]['model_safetensors']['sha256']
assert moe['selected_source']['run_id'] == rows[-1]['run_id']
revision = public['hf_revision']
url = f'https://huggingface.co/api/models/{public["hf_repo"]}/revision/{revision}?blobs=true'
hub, response = fetch_json(url)
assert hub['sha'] == revision and hub['private'] is False
siblings = [x for x in hub['siblings'] if x['rfilename'].startswith('selftrained/v2/moe-joint/')]
weight = next(x for x in siblings if x['rfilename'].endswith('/model.safetensors'))
assert weight['lfs']['sha256'] == moe['safe_weights']['sha256']
assert weight['size'] == moe['safe_weights']['bytes']
result['publication'] = {'request': response, 'revision': revision, 'private': hub['private'], 'moe_joint_files': siblings, 'historical_selected_export_matches_public_lfs': True}

producer_paths = ['scripts/selftrained/prepare_vision_ocr.py', 'scripts/selftrained/prepare_voice.py']
constants = {}
for path in producer_paths:
    tree = ast.parse((ROOT / path).read_text())
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {'FASHION_REVISION', 'NOTO_REVISION', 'REVISION'}:
                    values[target.id] = ast.literal_eval(node.value)
    constants[path] = {'sha256': sha(ROOT / path), 'source_revisions': values}
notice = (ROOT / 'docs/selftrained/DATA-NOTICE.md').read_text()
v2notice = (ROOT / 'docs/selftrained/V2-DATA-NOTICE.md').read_text()
for text in ['Fashion-MNIST', 'Noto Sans/Serif', 'MInDS-14']:
    assert text in notice
for item in constants.values():
    for revision_id in item['source_revisions'].values():
        assert revision_id in notice
assert 'No external per-example teacher-model API or mature neural weights initialize the assistant.' in v2notice
assert all(any(x['path'].startswith(prefix) for x in manifest['assets']) for prefix in ['images/vision/', 'images/ocr/', 'audio/'])
result['data_source_listing'] = {'notice_path': 'docs/selftrained/DATA-NOTICE.md', 'notice_sha256': sha(ROOT / 'docs/selftrained/DATA-NOTICE.md'), 'v2_notice_path': 'docs/selftrained/V2-DATA-NOTICE.md', 'v2_notice_sha256': sha(ROOT / 'docs/selftrained/V2-DATA-NOTICE.md'), 'producer_constants': constants, 'three_media_families_in_frozen_manifest': True, 'scope': 'Confirms explicit identified source versions and recipe, not a fresh byte-level regeneration or licensing audit.'}

# Compare actual historical constructor implementations with the inspected local code.
code_rows = []
for revision_id in [rows[0]['training_revision'], rows[-1]['training_revision']]:
    path = 'tiny_perceptron/selftrained/model.py'
    proc = subprocess.run(['git', 'show', revision_id + ':' + path], cwd=ROOT, capture_output=True, check=True)
    tree = ast.parse(proc.stdout)
    constructors = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in ['LimitedAssistant', 'VisionEncoder', 'OCREncoder', 'AudioEncoder']:
            init_node = next(x for x in node.body if isinstance(x, ast.FunctionDef) and x.name == '__init__')
            constructors[node.name] = ast.unparse(init_node)
    assert len(constructors) == 4
    assert all('torch.load' not in value and 'from_pretrained' not in value for value in constructors.values())
    code_rows.append({'revision': revision_id, 'path': path, 'blob_sha256': hashlib.sha256(proc.stdout).hexdigest(), 'constructors': constructors})
result['historical_constructors'] = code_rows
result['result'] = 'PASS: original training receipts form a from-random project lineage to the anonymous public V2 MoE export; data sources are explicitly pinned. No quality generalization or B/C completion is inferred.'
OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'artifact': str(OUT.relative_to(ROOT)), 'sha256': sha(OUT), 'result': result['result'], 'stages': len(rows), 'public_files': len(siblings)}, ensure_ascii=False))
