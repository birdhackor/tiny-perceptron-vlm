"""R.2 原 owner 知情回填複查：保存身份、依賴差異及靜態連結核對。"""
import ast
import difflib
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
from scripts.reading_time import chapter_introduction, figure_hashes, lesson_slices

TASK = '/root/tech_20261016_r_2'
BASE = 'docs/technical-reviews/artifacts/revision-20261016/'
ORIGINAL_REPORT = 'docs/course-revision-20261016/reviews/technical-R.2-original.json'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def file_identity(path):
    raw = (ROOT / path).read_bytes()
    return {'path': path, 'sha256': digest(raw), 'bytes': len(raw)}


original = json.loads((ROOT / ORIGINAL_REPORT).read_text())
old_receipt = json.loads((ROOT / (BASE + 'R.2-navigation-check.json')).read_text())
assert original['reviewer_task'] == TASK
old_artifacts = []
for artifact in original['artifacts']:
    identity = file_identity(artifact['path'])
    assert identity['sha256'] == artifact['sha256']
    identity['matches_original_report'] = True
    old_artifacts.append(identity)

source = ROOT / 'course/README.md'
body = lesson_slices(source.read_bytes().decode('utf-8'))['R.2']
assert digest(body.encode()) == original['source_sha256']
assert figure_hashes(ROOT, source, body) == original['figure_sha256'] == {}
current_freeze_path = 'docs/course-revision-20261016/freeze-15.json'
current_freeze = json.loads((ROOT / current_freeze_path).read_text())
page = next(p for p in current_freeze['pages'] if p['page_id'] == 'course')
frozen_bytes = (ROOT / page['snapshot']).read_bytes()
assert digest(frozen_bytes) == page['source_sha256']
assert lesson_slices(frozen_bytes.decode())['R.2'] == body
assert source.read_bytes() == frozen_bytes

old_paths = {item['path']: item['sha256'] for item in old_receipt['links']}
old_paths['scripts/export_course.py'] = old_receipt['export_source_sha256']
old_paths['course/lesson-index.json'] = old_receipt['index_sha256']
dependencies = []
for path, old_sha in sorted(old_paths.items()):
    item = file_identity(path)
    item['original_sha256'] = old_sha
    item['changed'] = old_sha != item['sha256']
    dependencies.append(item)

freeze08 = json.loads((ROOT / 'docs/course-revision-20261016/freeze-08.json').read_text())
old_pages = {p['page_id']: p for p in freeze08['pages']}
section_ids = [
    '2.1', '3.1', '4.5', 'chapter-13',
    '7.11', '7.17', '7.18', '7.19', '7.20', '7.21', '7.22', '7.23', '7.24', '7.25',
    '9.11', '9.12', '9.13', '9.14', '9.15', '9.16',
    '14.7', *[f'14.{i}' for i in range(11, 19)], '19.13', '19.17',
    'chapter-07', 'chapter-09', 'chapter-14', 'chapter-19',
]
section_comparisons = []
for section_id in section_ids:
    old_page = old_pages[section_id]
    old_bytes = (ROOT / old_page['snapshot']).read_bytes()
    assert digest(old_bytes) == old_page['source_sha256']
    current_text = (ROOT / old_page['source']).read_bytes().decode()
    current_section = chapter_introduction(current_text) if section_id.startswith('chapter-') else lesson_slices(current_text)[section_id]
    section_comparisons.append({
        'id': section_id, 'path': old_page['source'],
        'original_snapshot': old_page['snapshot'], 'original_sha256': digest(old_bytes),
        'current_sha256': digest(current_section.encode()),
        'changed': old_bytes.decode() != current_section,
        'diff': ''.join(difflib.unified_diff(old_bytes.decode().splitlines(True), current_section.splitlines(True), fromfile='freeze08/' + section_id, tofile='current/' + section_id, n=3)),
    })

old_export = subprocess.check_output(['git', 'show', old_receipt['git_head'] + ':scripts/export_course.py'], cwd=ROOT)
assert digest(old_export) == old_receipt['export_source_sha256']
current_export = (ROOT / 'scripts/export_course.py').read_bytes()

def function_tree(raw):
    return ast.dump(next(node for node in ast.parse(raw) .body if isinstance(node, ast.FunctionDef) and node.name == 'reading_markdown'), include_attributes=False)

assert function_tree(old_export) == function_tree(current_export)
export_diff = ''.join(difflib.unified_diff(old_export.decode().splitlines(True), current_export.decode().splitlines(True), fromfile='original/scripts/export_course.py', tofile='current/scripts/export_course.py', n=3))

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
    if fragment:
        target_body = lesson_slices(path.read_text())[fragment]
        expected_output = lesson_targets[path][fragment]
    else:
        target_body = path.read_text()
        expected_output = targets[path]
    converted = reading_markdown(f'[{label}]({target})', source, targets, lesson_targets, 'informed-R.2-callback')
    assert converted.strip() == f'[{label}]({expected_output})'
    links.append({'label': label, 'target': target, 'heading': target_body.splitlines()[0], 'exported_target': expected_output, 'status': 'resolved'})
assert len(links) == len(old_receipt['links']) == 27

ranges = []
locations = {item['id']: ROOT / item['source'] for item in index}
for old_range in old_receipt['ranges']:
    headings = []
    for old_heading in old_range['checked_headings']:
        section_id = old_heading.split()[1]
        heading = lesson_slices(locations[section_id].read_text())[section_id].splitlines()[0]
        assert heading == old_heading
        headings.append(heading)
    ranges.append({'label': old_range['label'], 'headings_identical': True, 'headings': headings})

manual_findings = [
    {'dependencies': ['course/chapters/07.md', 'course/chapters/09.md'], 'finding': '已讀所涉小節相對freeze-08的差異：7.25、9.14–9.16僅數字字面值與空白排版，其他R.2所涉小節及章導言相同；兩階段、長度、MTP、effort與機率判斷的路線名稱和用途不變。'},
    {'dependencies': ['course/chapters/14.md'], 'finding': '已讀14.12–14.17差異；14.15新增兩盒任務與實測曲線，14.16新增非單調實測解讀，其他是排版。14.11及14.7、14.18未變；新增內容仍屬R.2所指變動輪數與非單調評估路線。R.2沒有重述這些實測數字，故本複查只確認題意與導覽適用性，不審核或重跑新實測。'},
    {'dependencies': ['course/chapters/19.md'], 'finding': '已讀章導言差異及完整當前19.13、19.17。19.13由待回填改為已訓練產品，仍是A/B/C用途總覽；19.17補上C判資訊及已凍結規則Auto、品質成本取捨，仍回答R.2的Auto與合作問題。R.2沒有宣稱Auto必勝或報告測分，導覽仍成立；本次不獨立認證新訓練與最後測試數字。'},
    {'dependencies': ['scripts/export_course.py', 'course/lesson-index.json'], 'finding': '已讀導出程式差異：新增MathJax資產與C章導言分組。reading_markdown函式AST與原執行版本相同，索引檔hash未變；已另作27個當前靜態轉換核對。未執行prepare_assets、網站建置或瀏覽器版面檢查。'},
    {'dependencies': ['course/chapters/02.md', 'course/chapters/03.md', 'course/chapters/04.md', 'course/chapters/13.md'], 'finding': '原必要材料2.1、3.1、4.5及第13章導言逐段hash與仍有效的freeze-08快照相同。其餘未變目的地整檔hash與原收據相同，延續原已讀題意核對，不聲稱重新讀完所有目的地全文。'},
]
receipt = {
    'reviewer_task': TASK, 'reviewer_context_at_task_start': 'fresh', 'current_context': 'informed',
    'scope': '原owner的R.2單節知情回填複查；保留原報告和舊執行證據。',
    'command': '.venv/bin/python docs/technical-reviews/artifacts/revision-20261016/R.2-backfill-callback-01-check.py',
    'environment': {'python': platform.python_version(), 'platform': platform.platform(), 'device': 'CPU; hashes, snapshot diffs, AST comparison and static link conversion only'},
    'original_report': file_identity(ORIGINAL_REPORT), 'original_artifact_identities': old_artifacts,
    'current_freeze': file_identity(current_freeze_path), 'current_course_snapshot': file_identity(page['snapshot']),
    'current_source': file_identity('course/README.md'), 'section_sha256': digest(body.encode()),
    'section_matches_original_and_freeze15': True, 'figure_sha256': {},
    'dependency_files': dependencies,
    'additional_inspected_file_identities': [file_identity('course/chapters/' + chapter + '.md') for chapter in ['02', '03', '04', '13']],
    'section_dependency_comparisons': section_comparisons,
    'export_reading_markdown_ast_unchanged': True, 'export_diff': export_diff,
    'links': links, 'ranges': ranges, 'manual_findings': manual_findings,
    'not_rerun': ['原教材的數學/模型/訓練例子', '正式測試、GPU或Modal工作', 'HF寫入、權重下載或推論', '全頁或全書技術重審', '網站建置及瀏覽器視覺檢查'],
    'result': 'R.2 unchanged; original owner/evidence identities verified; changed necessary dependencies inspected for navigation impact; all 27 current links and 7 named ranges remain valid. Original pass can continue for this single section.',
}
out = ROOT / (BASE + 'R.2-backfill-callback-01-check.json')
with out.open('x', encoding='utf-8') as handle:
    json.dump(receipt, handle, ensure_ascii=False, indent=2)
    handle.write('\n')
print(json.dumps({'receipt': str(out.relative_to(ROOT)), 'links': len(links), 'ranges': len(ranges), 'changed_dependency_files': [item['path'] for item in dependencies if item['changed']], 'result': receipt['result']}, ensure_ascii=False))
