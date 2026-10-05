"""Freeze personally inspected 6.4 inputs and execute the original fence on CPU."""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path('/workspace/tiny-perceptron-vlm')
OUT = ROOT / 'docs/technical-reviews/artifacts/phase4-6_4-independent'
REV = 'a253d1262bf5f361f9ac4e19232ae752f0ecc7a3'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def save(rel, raw):
    path = OUT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': len(raw), 'sha256': digest(raw)}


def run(argv, name, timeout=60):
    completed = subprocess.run(argv, cwd=ROOT, capture_output=True, timeout=timeout)
    save(name + '.stdout.txt', completed.stdout)
    save(name + '.stderr.txt', completed.stderr)
    receipt = {'argv': argv, 'cwd': str(ROOT), 'exit_code': completed.returncode, 'timeout_seconds': timeout}
    save(name + '.receipt.json', (json.dumps(receipt, indent=2) + '\n').encode())
    return completed


manifest = {'purpose': 'Fresh independent 6.4 factual review; no legacy review judgments read', 'snapshots': [], 'historical_code': [], 'inputs': []}
for rel in ['course/chapters/06.md', 'course/training.md', 'pyproject.toml', 'docs/review-tools/factual-reviewer-instructions.md', 'docs/review-tools/section_facts.py', 'scripts/check_technical_reviews.py', '.agents/skills/clear-tutorial/references/review-protocol.md', 'tiny_perceptron/tokenization.py']:
    manifest['snapshots'].append({'original_path': rel, **save('inputs/' + rel, (ROOT / rel).read_bytes())})
result_raw = (ROOT / 'docs/course-experiments/results/tokenizer.json').read_bytes()
result = json.loads(result_raw)
manifest['result'] = save('inputs/tokenizer-result.json', result_raw)
manifest['historical_revision'] = result['revision']
assert result['revision'] == REV
for rel, expected in result['code_sha256'].items():
    completed = run(['git', 'show', REV + ':' + rel], 'git-show/' + rel.replace('/', '__'))
    assert completed.returncode == 0
    item = {'original_path': rel, 'expected_sha256': expected, **save('historical/' + rel, completed.stdout)}
    item['matches_recorded_hash'] = item['sha256'] == expected
    manifest['historical_code'].append(item)
    assert item['matches_recorded_hash'], item
for asset in result['assets']:
    for entry in asset['files']:
        if entry['path'].endswith(('tinystories-train-512.jsonl', 'chinese-classical-train-365.jsonl')):
            rel = 'data/training/' + entry['path']
            item = {'original_path': rel, 'expected_sha256': entry['sha256'], **save('inputs/assets/' + Path(rel).name, (ROOT / rel).read_bytes())}
            item['matches_recorded_hash'] = item['sha256'] == entry['sha256']
            manifest['inputs'].append(item)
            assert item['matches_recorded_hash'], item
for entry in result['artifacts']:
    if entry['path'].endswith('.pt') or entry['path'] == 'current-run.json':
        continue
    rel = 'outputs/text-behavior-interface-check/tokenizer/' + entry['path']
    item = {'original_path': rel, 'expected_sha256': entry['sha256'], **save('inputs/experiment/' + entry['path'], (ROOT / rel).read_bytes())}
    item['matches_recorded_hash'] = item['sha256'] == entry['sha256']
    manifest['inputs'].append(item)
    assert item['matches_recorded_hash'], item
temporary = Path('/tmp/phase4-6_4-original-fence')
assert not temporary.exists(), 'unique extraction output is required'
execution = run([str(ROOT / '.venv/bin/python'), 'docs/review-tools/section_facts.py', 'course/chapters/06.md#6.4', '--output', str(temporary), '--execute', '--timeout', '45'], 'extract-execute', 60)
assert execution.returncode == 0
for path in temporary.iterdir():
    if path.is_file():
        save('original/' + path.name, path.read_bytes())
metadata = json.loads((temporary / 'extraction.json').read_text())
manifest['original_source_sha256'] = metadata['source_sha256']
manifest['figure_sha256'] = metadata['figure_sha256']
manifest['figure_render_applicability'] = 'not applicable: 6.4 contains no figure, SVG reference, image, or spatial diagram'
save('input-provenance.json', (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode())
print(json.dumps({'historical_files': len(manifest['historical_code']), 'matched_inputs': len(manifest['inputs']), 'source_sha256': metadata['source_sha256'], 'fences': len(metadata['python_fences']), 'figures': len(metadata['svg_references'])}, ensure_ascii=False))
