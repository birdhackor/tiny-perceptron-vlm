"""原 R.3 owner 的知情依賴確認；不跑 Notebook、模型或正式測試。"""

import ast
import hashlib
import json
import platform
import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.reading_time import lesson_slices


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def flatten(node):
    if isinstance(node, str):
        yield node
    elif isinstance(node, list):
        for value in node:
            yield from flatten(value)
    elif isinstance(node, dict):
        for value in node.values():
            yield from flatten(value)


prior_path = 'docs/course-revision-20261016/reviews/technical-R.3-original.json'
prior = json.loads((ROOT / prior_path).read_text())
old_receipt = json.loads((ROOT / prior['artifacts'][1]['path']).read_text())
for artifact in prior['artifacts']:
    assert sha((ROOT / artifact['path']).read_bytes()) == artifact['sha256']

body = lesson_slices((ROOT / 'course/README.md').read_bytes().decode())['R.3']
assert sha(body.encode()) == prior['source_sha256']
freeze_path = 'docs/course-revision-20261016/freeze-15.json'
freeze = json.loads((ROOT / freeze_path).read_text())
page = next(page for page in freeze['pages'] if page['page_id'] == 'course')
assert sha((ROOT / page['snapshot']).read_bytes()) == page['source_sha256']
assert sha((ROOT / page['source']).read_bytes()) == page['source_sha256']
assert lesson_slices((ROOT / page['snapshot']).read_bytes().decode())['R.3'] == body

old_dependencies = dict(old_receipt['inputs_sha256'])
old_dependencies.update({'course/' + link['target']: link['sha256'] for link in old_receipt['links']})
dependencies = []
for path, old_hash in old_dependencies.items():
    current_hash = sha((ROOT / path).read_bytes())
    dependencies.append({'path': path, 'prior_sha256': old_hash, 'current_sha256': current_hash, 'changed': current_hash != old_hash})

links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', body)
assert len(links) == 26
assert all((ROOT / 'course' / target).is_file() for _, target in links)
nav = list(flatten(tomllib.loads((ROOT / 'zensical.toml').read_text())['project']['nav']))
sequence = old_receipt['navigation_sequence_verified']
positions = [nav.index(target) for target in sequence]
assert positions == sorted(positions)
assert [target for target in nav if re.fullmatch(r'14\.\d+\.md', target)] == old_receipt['chapter14_navigation']

index = json.loads((ROOT / 'course/lesson-index.json').read_text())
catalog = (ROOT / 'course/lessons.md').read_text()
chapter_sections = {}
for path in (ROOT / 'course/chapters').glob('*.md'):
    for identifier in lesson_slices(path.read_bytes().decode()):
        chapter_sections[identifier] = path.relative_to(ROOT).as_posix()
assert len(index) == len(chapter_sections) == 312
assert {item['id'] for item in index} == set(chapter_sections)
for item in index:
    assert item['source'] == chapter_sections[item['id']]
    assert f"[閱讀](../{item['source']}#{item['id']})" in catalog
    assert f"[開啟](../{item['notebook']})" in catalog
    assert (ROOT / item['notebook']).is_file()

old_freeze = json.loads((ROOT / 'docs/course-revision-20261016/freeze-08.json').read_text())
changed_chapter_identities = []
for name in ('07', '09', '14', '19'):
    source = f'course/chapters/{name}.md'
    pages = [p for p in old_freeze['pages'] if p['source'] == source]
    intro = next(p for p in pages if p['selector'] == 'before-first-section')
    ordered = [intro] + [p for p in pages if p['selector'].startswith('section:')]
    old_raw = b''.join((ROOT / p['snapshot']).read_bytes() for p in ordered)
    assert sha(old_raw) == old_dependencies[source]
    headings = lambda text: re.findall(r'^#{1,2} .+$', text, re.M)
    assert headings(old_raw.decode()) == headings((ROOT / source).read_text())
    changed_chapter_identities.append({'source': source, 'original_full_file_reconstructed_and_hash_matched': True, 'chapter_and_section_headings_unchanged': True})

old_export = subprocess.check_output(['git', 'show', old_receipt['historical_revision'] + ':scripts/export_course.py'], cwd=ROOT).decode()
assert sha(old_export.encode()) == old_dependencies['scripts/export_course.py']
def function_dump(text, name):
    return ast.dump(next(node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef) and node.name == name), include_attributes=False)
assert function_dump(old_export, 'reading_markdown') == function_dump((ROOT / 'scripts/export_course.py').read_text(), 'reading_markdown')

receipt = {
    'reviewer_task': '/root/tech_20261016_r_3', 'current_context': 'informed',
    'scope': 'R.3 unchanged subsection and its navigation dependencies only; not a whole course-page or empirical-result review.',
    'prior_report': {'path': prior_path, 'sha256': sha((ROOT / prior_path).read_bytes())},
    'prior_artifact_identity_verified': prior['artifacts'],
    'freeze': {'path': freeze_path, 'sha256': sha((ROOT / freeze_path).read_bytes()), 'course_snapshot': page['snapshot']},
    'source': 'course/README.md#R.3', 'source_sha256': sha(body.encode()), 'figure_sha256': {},
    'section_matches_original_and_current_freeze': True,
    'dependencies': dependencies,
    'changed_chapter_identity_checks': changed_chapter_identities,
    'current_navigation_sequence_verified': sequence,
    'local_guide_links_verified': len(links), 'catalog_chapter_sections_and_notebook_paths_verified': len(index),
    'export_reading_markdown_ast_unchanged': True,
    'command': '.venv/bin/python docs/technical-reviews/artifacts/revision-20261016/R.3-backfill-callback-01-check.py',
    'environment': {'python': platform.python_version(), 'platform': platform.platform(), 'device': 'CPU; static file and navigation checks only'},
    'result': 'PASS: original evidence identities, unchanged R.3 bytes, current freeze-15, 26 local guide links, C/chapter-14 route and 312 catalog/Notebook paths verified; changed chapter headings and link-conversion function remain consistent.',
    'not_rerun': ['Original R.3-navigation-check.py (would overwrite historical evidence)', 'Notebook execution', 'Model inference or training', 'Formal tests', 'Browser or hosted website verification'],
}
output = Path(__file__).with_name('R.3-backfill-callback-01-dependencies.json')
with output.open('x') as stream:
    json.dump(receipt, stream, ensure_ascii=False, indent=2)
    stream.write('\n')
print(receipt['result'])
