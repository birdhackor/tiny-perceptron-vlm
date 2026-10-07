"""Own source-only CPU checks; synthetic minutes do not constitute reading estimates."""
import ast
import hashlib
import json
import sys
from html.parser import HTMLParser
from pathlib import Path

R = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(R))
from scripts import reading_time

def sha(path):
    return hashlib.sha256((R / path).read_bytes()).hexdigest()

old_path = 'docs/course-revision-20261007-phase7/reviews/freeze-04/reports/technical/f/p7_technical_f-recheck-20261007T055405.json'
manifest_path = 'docs/course-revision-20261007-phase7/reviews/freeze-06/manifest.json'
old = json.loads((R / old_path).read_text())
manifest = json.loads((R / manifest_path).read_text())
current = {p['page_id']: p for p in manifest['pages']}
assert len(old['pages']) == 54
assert all(p['source_sha256'] == current[p['page_id']]['source_sha256'] and p['figures_sha256'] == current[p['page_id']]['figures_sha256'] for p in old['pages'])
for p in old['pages']:
    frozen = current[p['page_id']]
    assert sha(frozen['snapshot']) == frozen['source_sha256']
    for path, digest in frozen['figures_sha256'].items():
        assert sha(path) == digest
stale = []
for p in old['pages']:
    for source in p.get('sources', []):
        if source['kind'] == 'repository_code' and sha(source['path']) != source['sha256']:
            stale.append({'page_id': p['page_id'], 'source_id': source['id'], 'path': source['path'], 'old_sha256': source['sha256'], 'current_sha256': sha(source['path'])})
assert len(stale) == 1 and stale[0]['page_id'] == 'publishing' and stale[0]['source_id'] == 'reading-time'
tree = ast.parse((R / 'scripts/reading_time.py').read_text())
spans = {node.name: [node.lineno, node.end_lineno] for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in ('load_estimates', 'load_routes', 'total_range', 'format_range', 'render_page', 'main')}

class Elements(HTMLParser):
    def __init__(self):
        super().__init__()
        self.details = []
    def handle_starttag(self, tag, attrs):
        if tag == 'details':
            self.details.append(dict(attrs))

content = '# 合成標題\n\n原始正文。\n\n```python\nprint(7)\n```\n'
inventory = {'groups': [{'id': 'canonical_course', 'label': '全部教材正文', 'page_ids': ['lesson']}, {'id': 'complete_site', 'label': '全部網站頁面', 'page_ids': ['lesson', 'index', 'course']}]}
estimates = {i: {'minutes_min': 2, 'minutes_max': 5} for i in ('lesson', 'index', 'course')}
routes = [{'id': 'synthetic', 'label': '<合成路線>', 'page_ids': ['lesson', 'lesson']}]
cases = []
for page_id in ('lesson', 'index', 'course'):
    out = reading_time.render_page(content, page_id, estimates, inventory, routes)
    parser = Elements(); parser.feed(out)
    assert out.startswith('# 合成標題\n') and content.partition('\n')[2] in out
    assert out.count('閱讀約 2–5 分鐘 · AI 估計') == 1
    if page_id == 'lesson':
        assert not parser.details and '閱讀總量' not in out
    else:
        assert len(parser.details) == 1 and parser.details[0]['class'] == 'reading-time-method' and 'open' not in parser.details[0]
        assert '<summary>全教材約 2–5 分鐘 · AI 估計</summary>' in out
        assert '全部網站頁面：約 6–15 分鐘' in out and '&lt;合成路線&gt;：約 2–5 分鐘' in out
        details_start, details_end = out.index('<details'), out.index('</details>')
        assert details_start < out.index('reading-time-totals') < details_end
        assert '不是實測平均' in out and '不包含安裝、下載' in out
    cases.append({'page_id': page_id, 'fixture_only': True, 'body_preserved': True, 'details_attributes': parser.details, 'rendered': out})
missing = reading_time.render_page(content, 'index', {}, inventory)
assert '閱讀約' not in missing and 'reading-time-totals' not in missing and '閱讀時間怎麼估？' in missing
assert reading_time.render_page(content, 'lesson', {}, inventory) == content
cases.append({'case': 'missing-estimates', 'fixture_only': True, 'no_fabricated_minutes': True, 'rendered_home': missing, 'lesson_returned_exact_original': True})
print(json.dumps({'scope': 'Source-only callback; no new incremental reading trace or visual viewing. Synthetic render contract only, not measured reading speed or full site build.', 'script_sha256': sha('scripts/reading_time.py'), 'old_report_sha256': sha(old_path), 'manifest_sha256': sha(manifest_path), 'unchanged_primary_page_count': 54, 'actual_snapshot_and_figure_hashes_checked': True, 'only_stale_repository_source': stale, 'ast_function_line_spans': spans, 'render_cases': cases}, ensure_ascii=False))
