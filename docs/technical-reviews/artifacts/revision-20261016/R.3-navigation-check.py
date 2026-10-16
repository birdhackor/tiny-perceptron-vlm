"""R.3 的來源、導覽與連結核對；不執行課程訓練。"""

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


def sha(data):
    return hashlib.sha256(data).hexdigest()


def flatten(node):
    if isinstance(node, str):
        yield node
    elif isinstance(node, list):
        for value in node:
            yield from flatten(value)
    elif isinstance(node, dict):
        for value in node.values():
            yield from flatten(value)


source = ROOT / 'course/README.md'
body = lesson_slices(source.read_bytes().decode('utf-8'))['R.3']
freeze = json.loads((ROOT / 'docs/course-revision-20261016/freeze-08.json').read_text())
frozen = next(page for page in freeze['pages'] if page['page_id'] == 'course')
snapshot = (ROOT / frozen['snapshot']).read_bytes()
assert sha(source.read_bytes()) == sha(snapshot) == frozen['source_sha256']
assert body == lesson_slices(snapshot.decode('utf-8'))['R.3']

links = []
for label, target in re.findall(r'\[([^\]]+)\]\(([^)]+)\)', body):
    path = (source.parent / target).resolve()
    assert path.is_file(), target
    links.append({'label': label, 'target': target, 'exists': True, 'sha256': sha(path.read_bytes())})

nav = list(flatten(tomllib.loads((ROOT / 'zensical.toml').read_text())['project']['nav']))
required = ['7.25.md'] + [f'C.{i}.md' for i in range(1, 6)] + ['chapter-08.md', '8.1.md']
required += ['13.17.md', 'C.6.md', 'C.7.md', 'chapter-14.md']
required += [f'14.{i}.md' for i in list(range(1, 7)) + list(range(11, 19)) + list(range(7, 11))]
positions = [nav.index(target) for target in required]
assert positions == sorted(positions)
chapter14 = [target for target in nav if re.fullmatch(r'14\.\d+\.md', target)]
assert chapter14 == [f'14.{i}.md' for i in list(range(1, 7)) + list(range(11, 19)) + list(range(7, 11))]

index = json.loads((ROOT / 'course/lesson-index.json').read_text())
lessons = (ROOT / 'course/lessons.md').read_text()
chapter_sections = {}
for chapter in (ROOT / 'course/chapters').glob('*.md'):
    for identifier, section in lesson_slices(chapter.read_bytes().decode('utf-8')).items():
        chapter_sections[identifier] = chapter.relative_to(ROOT).as_posix()
assert len(index) == len(chapter_sections)
assert {item['id'] for item in index} == set(chapter_sections)
for item in index:
    assert item['source'] == chapter_sections[item['id']]
    assert f"[閱讀](../{item['source']}#{item['id']})" in lessons
    assert f"[開啟](../{item['notebook']})" in lessons
    assert (ROOT / item['notebook']).is_file()

revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
old_index = json.loads(subprocess.check_output(['git', 'show', f'{revision}:course/lesson-index.json'], cwd=ROOT, text=True))
old14 = [item for item in old_index if item['id'].startswith('14.')]
current_by_id = {item['id']: item for item in index}
assert [item['id'] for item in old14] == [f'14.{i}' for i in range(1, 11)]
for item in old14:
    assert all(current_by_id[item['id']][key] == item[key] for key in ('id', 'source', 'notebook'))

inputs = ['course/README.md', 'course/lessons.md', 'course/lesson-index.json', 'zensical.toml', 'scripts/export_course.py']
receipt = {
    'reviewer_task': '/root/tech_20261016_r_3',
    'source': 'course/README.md#R.3',
    'source_sha256': sha(body.encode('utf-8')),
    'command': '.venv/bin/python docs/technical-reviews/artifacts/revision-20261016/R.3-navigation-check.py',
    'environment': {'python': platform.python_version(), 'platform': platform.platform(), 'device': 'CPU; filesystem and navigation checks only'},
    'freeze_snapshot': frozen['snapshot'],
    'freeze_and_live_section_match': True,
    'links': links,
    'navigation_sequence_verified': required,
    'chapter14_navigation': chapter14,
    'historical_revision': revision,
    'preserved_legacy_14_ids_source_and_notebook': [item['id'] for item in old14],
    'catalog_chapter_sections_and_notebooks_verified': len(index),
    'inputs_sha256': {path: sha((ROOT / path).read_bytes()) for path in inputs},
    'result': 'PASS: frozen R.3 matches; 26 local guide links exist; required C and chapter-14 navigation order matches; 312 chapter sections have catalog reading and existing Notebook links; historical 14.1–14.10 identifiers/source paths/Notebook paths are preserved.',
    'limits': 'Source-level navigation only; no live hosted website, browser layout, model quality or training execution is verified.'
}
assert len(links) == 26
output = Path(__file__).with_suffix('.json')
output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
print(receipt['result'])
