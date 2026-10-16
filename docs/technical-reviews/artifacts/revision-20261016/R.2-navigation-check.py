"""R.2 技術審閱的連結收據；不建置網站、不執行教材或訓練。"""
import hashlib
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.export_course import DOCUMENTS, reading_markdown
from scripts.reading_time import lesson_slices

source = ROOT / 'course/README.md'
body = lesson_slices(source.read_bytes().decode('utf-8'))['R.2']
freeze = json.loads((ROOT / 'docs/course-revision-20261016/freeze-08.json').read_text())
frozen_page = next(page for page in freeze['pages'] if page['page_id'] == 'course')
frozen_bytes = (ROOT / frozen_page['snapshot']).read_bytes()
assert hashlib.sha256(frozen_bytes).hexdigest() == frozen_page['source_sha256']
assert body == lesson_slices(frozen_bytes.decode('utf-8'))['R.2']

index = json.loads((ROOT / 'course/lesson-index.json').read_text())
targets = {(ROOT / path).resolve(): name + '.md' for name, path in DOCUMENTS.items()}
lesson_targets = {}
for item in index:
    chapter = (ROOT / item['source']).resolve()
    targets[chapter] = 'chapter-' + chapter.stem + '.md'
    lesson_targets.setdefault(chapter, {})[item['id']] = item['id'] + '.md'
links = []
for label, target in re.findall(r'\[([^\]]+)\]\(([^)]+)\)', body):
    path_text, _, fragment = target.partition('#')
    path = (source.parent / path_text).resolve()
    assert path.is_file() and path.is_relative_to(ROOT)
    text = path.read_bytes().decode('utf-8')
    if fragment:
        target_body = lesson_slices(text)[fragment]
        assert fragment in lesson_targets[path]
        expected_output = fragment + '.md'
    else:
        target_body = text
        expected_output = targets[path]
    converted = reading_markdown(f'[{label}]({target})', source, targets, lesson_targets, 'review-R.2')
    assert converted.strip() == f'[{label}]({expected_output})'
    links.append({
        'label': label, 'target': target,
        'path': path.relative_to(ROOT).as_posix(),
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'fragment': fragment or None,
        'target_heading': target_body.splitlines()[0],
        'exported_target': expected_output,
        'status': 'resolved',
    })

ranges = {
    '1–4': ['1.4', '1.6', '2.1', '3.1', '4.5'],
    '7.20–7.22': ['7.20', '7.21', '7.22'],
    '7.23–7.25': ['7.23', '7.24', '7.25'],
    '8.14–8.16': ['8.14', '8.15', '8.16'],
    '9.11–9.16': ['9.11', '9.12', '9.13', '9.14', '9.15', '9.16'],
    '14.11–14.18': [f'14.{i}' for i in range(11, 19)],
    'C.1–C.5': [f'C.{i}' for i in range(1, 6)],
}
locations = {item['id']: (ROOT / item['source']) for item in index}
range_results = []
for label, section_ids in ranges.items():
    headings = []
    for section_id in section_ids:
        headings.append(lesson_slices(locations[section_id].read_text())[section_id].splitlines()[0])
    range_results.append({'label': label, 'checked_headings': headings, 'status': 'present'})

receipt = {
    'reviewer_task': '/root/tech_20261016_r_2',
    'lesson_id': 'R.2',
    'command': '.venv/bin/python docs/technical-reviews/artifacts/revision-20261016/R.2-navigation-check.py',
    'environment': {'python': platform.python_version(), 'platform': platform.platform(), 'device': 'CPU; static Markdown/export check only'},
    'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    'source_sha256': hashlib.sha256(body.encode()).hexdigest(),
    'freeze_snapshot': frozen_page['snapshot'],
    'freeze_snapshot_matches': True,
    'links_count': len(links),
    'links': links,
    'ranges': range_results,
    'export_source_sha256': hashlib.sha256((ROOT / 'scripts/export_course.py').read_bytes()).hexdigest(),
    'index_sha256': hashlib.sha256((ROOT / 'course/lesson-index.json').read_bytes()).hexdigest(),
    'result': 'All local paths, explicit section IDs, named ranges, and in-memory exported targets resolved. No model/training/benchmark execution performed.',
}
out = ROOT / 'docs/technical-reviews/artifacts/revision-20261016/R.2-navigation-check.json'
out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'links_count': len(links), 'ranges_count': len(range_results), 'result': receipt['result'], 'receipt': str(out.relative_to(ROOT))}, ensure_ascii=False))
