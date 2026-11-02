"""Reviewer-owned current README visual-builder probes; mutations only in Git-external copies."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys

ROOT = Path('/workspace/tiny-perceptron-vlm')
WORK = Path('/workspace/work/omni-technical-readme-v1/current-fix14')
OUT = ROOT / 'docs/technical-reviews/artifacts/omni-technical-readme-v1/current-c75-probe-v1.json'
WORK.mkdir(exist_ok=False)
COPY = WORK / 'source'
(COPY / 'scripts').mkdir(parents=True)
shutil.copy2(ROOT / 'scripts/build_visuals.py', COPY / 'scripts/build_visuals.py')
shutil.copytree(ROOT / 'course/figures', COPY / 'course/figures')
pub = Path('docs/lesson-products/results/publication-spec-v1.json')
(COPY / pub).parent.mkdir(parents=True)
shutil.copy2(ROOT / pub, COPY / pub)
sha = lambda raw: hashlib.sha256(raw).hexdigest()
def inventory():
    return {p.name: {'sha256': sha(p.read_bytes()), 'bytes': p.stat().st_size}
            for p in sorted((COPY / 'course/figures').glob('*.svg'))}
def aggregate(value):
    return sha(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode())
records = []
def run(label, should_pass, expected_error=None):
    command = [sys.executable, 'scripts/build_visuals.py']
    p = subprocess.run(command, cwd=COPY, capture_output=True, text=True)
    raw = (p.stdout + p.stderr).encode()
    path = WORK / (label + '.txt'); path.write_bytes(raw)
    r = {'label': label, 'command': 'python scripts/build_visuals.py', 'argv': command,
         'cwd': str(COPY), 'exit_code': p.returncode, 'stdout': p.stdout,
         'last_error': p.stderr.splitlines()[-1] if p.stderr else None,
         'raw_path': str(path), 'raw_sha256': sha(raw), 'raw_bytes': len(raw)}
    assert (p.returncode == 0) == should_pass, r
    if expected_error:
        assert expected_error in p.stderr, r
    records.append(r)

before = inventory()
assert len(before) == 215
run('current-clean', True)
after = inventory()
assert before == after, [p for p in before if before[p] != after.get(p)]
index = json.loads((COPY / 'course/figures/index.json').read_text())
assert len(index) == 215
plot_name = 'looped-mechanism-measured.svg'
plot_entry = index[Path(plot_name).stem]
assert plot_entry['source'] == 'generated_result'
assert plot_entry['generator'] == 'scripts/lesson_products/plot_results.py'
assert plot_entry['result_manifest'] == pub.as_posix()
publication = json.loads((COPY / pub).read_text())
assert before[plot_name] == publication['files'][plot_name]

metadata_path = COPY / 'course/figures/source-metadata.json'
metadata_raw = metadata_path.read_bytes()
metadata = json.loads(metadata_raw)
metadata_checks = []
for name, entry in metadata['figures'].items():
    path = ROOT / 'course/figures' / name
    assert before[name] == {'sha256': entry['sha256'], 'bytes': entry['bytes']}
    source_path = ROOT / entry['description_source']
    text = source_path.read_text()
    matches = re.findall(r'!\[([^\]]*)\]\([^)]*/' + re.escape(name) + r'\)', text)
    assert entry['description'] in matches, (name, matches, entry['description'])
    assert index[path.stem]['metadata_source'] == 'source-metadata.json'
    assert index[path.stem]['title'] == entry['title']
    assert index[path.stem]['caption'] == entry['description']
    metadata_checks.append({'figure': name, 'source': entry['description_source'],
                            'source_sha256': sha(source_path.read_bytes()),
                            'matched_alt_sha256': sha(entry['description'].encode())})

unknown = COPY / 'course/figures/zz-review-unknown.svg'
unknown.write_text('<svg xmlns="http://www.w3.org/2000/svg"><rect width="1" height="1"/></svg>')
run('unknown-untitled-rejected', False, '手寫圖解需要標題與描述或對應bytes的明示metadata：zz-review-unknown.svg')
unknown.unlink()

name = next(iter(metadata['figures']))
entry = metadata['figures'].pop(name)
metadata_path.write_text(json.dumps(metadata, ensure_ascii=False))
run('registered-entry-removed-rejected', False, '手寫圖解需要標題與描述或對應bytes的明示metadata：' + name)
metadata_path.write_bytes(metadata_raw)
known = COPY / 'course/figures' / name
known_raw = known.read_bytes(); known.write_bytes(known_raw + b'\n')
run('registered-figure-drift-rejected', False, '手寫圖解需要標題與描述或對應bytes的明示metadata：' + name)
known.write_bytes(known_raw)

plot_path = COPY / 'course/figures' / plot_name
plot_raw = plot_path.read_bytes(); plot_path.write_bytes(plot_raw + b'\n')
run('published-result-drift-rejected', False, '實測圖與固定發布清單不同：' + plot_name)
plot_path.write_bytes(plot_raw)
pub_raw = (COPY / pub).read_bytes()
modified = json.loads(pub_raw); modified['files'][plot_name]['bytes'] += 1
(COPY / pub).write_text(json.dumps(modified))
run('publication-byte-count-mismatch-rejected', False, '實測圖與固定發布清單不同：' + plot_name)
(COPY / pub).write_bytes(pub_raw)
run('restored-clean', True)
assert inventory() == before

result = {
    'created_at': datetime.now(timezone.utc).isoformat(),
    'reviewer_task': '/root/omni_technical_coordinator/technical_readme',
    'environment': {'python': platform.python_version(), 'device': 'cpu', 'os': platform.platform()},
    'source_inputs': {p: sha((ROOT / p).read_bytes()) for p in ['scripts/build_visuals.py',
          'course/figures/source-metadata.json', pub.as_posix(), 'scripts/lesson_products/plot_results.py']},
    'snapshot_strategy': 'Exact current script, complete current figure directory and the actual publication spec copied to an isolated Git-external tree; no source substitutions or author execution results.',
    'svg_count_before': len(before), 'svg_count_after': len(after), 'changed_svg_count': 0,
    'before_inventory_sha256': aggregate(before), 'after_inventory_sha256': aggregate(after),
    'verified_original_svgs': before,
    'published_plot_index_entry': plot_entry,
    'published_plot_bound': publication['files'][plot_name],
    'external_metadata_entries': len(metadata_checks), 'metadata_alt_checks': metadata_checks,
    'records': records,
    'limitations': 'This verifies the README maintenance command and preservation/provenance checks. It does not regenerate the measured plot or re-review every scientific figure or sign the broader tool inventory.'
}
with OUT.open('x') as stream:
    stream.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'output': str(OUT), 'sha256': sha(OUT.read_bytes()), 'svg_count': len(before),
                  'changed': 0, 'metadata_entries': len(metadata_checks),
                  'cases': [{'label': r['label'], 'exit_code': r['exit_code'], 'last_error': r['last_error']} for r in records]}, ensure_ascii=False, indent=2))
