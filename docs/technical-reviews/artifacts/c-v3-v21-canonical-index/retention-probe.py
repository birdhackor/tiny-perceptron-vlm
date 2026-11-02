"""Read-only source22 index evidence identity and bounded retention check.

Reads saved Notebooks/HTML only; does not execute a lesson, browser, or model.
"""

import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import re
import subprocess
import sys
import tomllib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from bs4 import BeautifulSoup


ROOT = Path('/tmp/tpb').resolve()
OLD_ROOT = Path('/workspace/work/omni-book-integrated-v1')
ROUND = 'docs/course-revision-20261101-c-v3/'
HERE = 'docs/technical-reviews/artifacts/c-v3-v21-canonical-index/'
PRIOR = 'docs/technical-reviews/artifacts/omni-P5-index/'
PREVIEW = ROOT / 'outputs/c-v3-author-preview-current'
BUNDLE = ROOT / 'outputs/omni-c-v3-kernel-bundle-v22'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def record(path):
    raw = path.read_bytes()
    return {'path': str(path), 'bytes': len(raw), 'sha256': sha(raw)}


def read_ref(relative, expected):
    raw = (ROOT / relative).read_bytes()
    assert sha(raw) == expected, relative
    return json.loads(raw)


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def output_signature(div):
    return {
        'label': div.select_one('.output-label').get_text(),
        'pre': [x.get_text() for x in div.find_all('pre')],
        'images': [sha(x['src'].encode()) for x in div.find_all('img')],
    }


def nav_entries(soup):
    return [(a.get_text(' ', strip=True), a.get('href', '').removeprefix('./'))
            for a in soup.select('nav.md-nav--primary a[href]')]


def flatten(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for child in value:
            yield from flatten(child)
    elif isinstance(value, dict):
        for child in value.values():
            yield from flatten(child)


started_at = datetime.now(timezone.utc).isoformat()
setup = module('index_retention_native_setup', 'scripts/omni_revision/review_setup.py')
exporter = module('scripts.index_retention_exporter', 'scripts/export_course.py')
freeze = read_ref(ROUND+'source-freeze-v22.json', '0f4e9d63bfcb2243234c17c62242ff79cd4ab640a21d1f313f94c99946a93a59')
previous_freeze = read_ref(ROUND+'source-freeze-v21.json', 'd9100e058cae5bbd6427c8f449d7540c8d5a7e9682ed69b4dc0eda87ee4c9cea')
pages = {p['page_id']: p for p in freeze['pages']}
previous_pages = {p['page_id']: p for p in previous_freeze['pages']}
original = read_ref(ROUND+'technical-v21/canonical_index/original-v1.json', '3f3d49d6d66d1789c8829146f9afd173ad8695a9b7deb42bf048eacc9febd4c5')
received = read_ref(ROUND+'technical-coordination-v21/receipts/canonical-index-original-v1.json', 'da580dbfe06494c8711e206e5292120f84d2126de1e9de152a775f5e3cbfe65e')
raw_refs = {
    PRIOR+'inspection-notes-v1.json': '96b8ba140e9aa4fd7ab4e4b7118135b2ec6df9ed1f1843a859ea6938c0663bc4',
    PRIOR+'browser-audit-v2.json': 'a0a33e2612325fa62a0b4efe9f3312ae3abccf4a8319a4f714036456fdef7b16',
    PRIOR+'bounded-update-v1.json': '854c96a136be539a2928d0133fe58c6b734ebb603110735a948f997f1f7acbc5',
    PRIOR+'current-static-audit-v1.json': '3f31103d48811548d83f6c1ab9d724ab3b5b2d131741bbf4cb4520b21763a28b',
}
raw = {relative: read_ref(relative, expected) for relative, expected in raw_refs.items()}
inspection = raw[PRIOR+'inspection-notes-v1.json']
browser = raw[PRIOR+'browser-audit-v2.json']
update = raw[PRIOR+'bounded-update-v1.json']
old_static = raw[PRIOR+'current-static-audit-v1.json']
old_report = read_ref('docs/course-revision-20261029-omni/reviews/P5-actual/technical-index/technical-current-v1.json', '1d51606948b288b89b0c63b63ec8c27f0ad1752ac32c2e7e3581ae6d1efdda75')
old_freeze = read_ref('docs/course-revision-20261029-omni/reviews/P5-actual/source-freeze-v8.json', '8df31b044f819c10ba0db5db9ae65f36bd39134c017a646e902b7376e78cbcaa')
old_pages = {p['page_id']: p for p in old_freeze['pages']}
old_home = setup.frozen_body(old_pages['index'])
home = setup.frozen_body(pages['index'])
index = json.loads((ROOT/'course/lesson-index.json').read_bytes())
assert home == exporter.home_introduction(index, True)
assert old_home.replace('全部 331 個小節', '全部 332 個小節') == home
assert pages['index'] == previous_pages['index']
assert sha((ROOT/'scripts/export_course.py').read_bytes()) == old_static['exporter']['sha256']

paper_path = Path(inspection['paper']['full_text_outside_git'])
assert record(paper_path)['sha256'] == inspection['paper']['sha256']
paper_page = subprocess.run(['pdftotext', '-f', '2', '-l', '2', str(paper_path), '-'],
                            capture_output=True, text=True, check=True).stdout
assert 'combinations of the last n' in paper_page and 'words:' in paper_page
paper_excerpt = 'n-gram models construct tables of conditional probabilities for the next word ... combinations of the last n − 1 words'

screenshots = []
for item in browser['screenshots']:
    actual = record(Path(item['path_outside_git']))
    assert actual['sha256'] == item['sha256'] and actual['bytes'] == item['bytes']
    screenshots.append(actual)

prior_preview = OLD_ROOT/'outputs/omni-P5-preview-v9'
old_home_record = record(prior_preview/'index.html')
assert old_home_record['sha256'] == update['preview_home_records'][1]['sha256']
old_soup = BeautifulSoup((prior_preview/'index.html').read_bytes(), 'html.parser')
current_soup = BeautifulSoup((PREVIEW/'index.html').read_bytes(), 'html.parser')
assets = []
for item in update['browser_assets']:
    actual = record(PREVIEW/item['relative'].removeprefix('./'))
    assert actual['sha256'] == item['new']['sha256']
    assets.append({'relative': item['relative'], 'sha256': actual['sha256'], 'bytes': actual['bytes'], 'identical_to_original_runtime_asset': True})
header_identical = str(old_soup.select_one('header.md-header')) == str(current_soup.select_one('header.md-header'))
config_identical = old_soup.select_one('#__config').get_text() == current_soup.select_one('#__config').get_text()
menu_identical = str(old_soup.select_one('label[for="__drawer"].md-header__button')) == str(current_soup.select_one('label[for="__drawer"].md-header__button'))
drawer_toggle_identical = str(old_soup.select_one('input#__drawer')) == str(current_soup.select_one('input#__drawer'))
assert header_identical and config_identical and menu_identical and drawer_toggle_identical
old_nav = nav_entries(old_soup)
new_nav = nav_entries(current_soup)
nav_added = [entry for entry in new_nav if entry not in old_nav]
nav_removed = [entry for entry in old_nav if entry not in new_nav]
assert not nav_removed
assert nav_added == [('19.25 怎樣把判斷器接成可用的裝置服務？', '19.25.html')]
assert [entry for entry in new_nav if entry not in nav_added] == old_nav
settings = tomllib.loads((ROOT/'zensical.toml').read_text())['project']
nav_counts = Counter(flatten(settings['nav']))
assert len(index) == len({item['id'] for item in index}) == 332
assert all(nav_counts[item['id']+'.md'] == 1 for item in index)
assert all(sum(href == item['id']+'.html' for _, href in new_nav) == 1 for item in index)

old_search = json.loads((prior_preview/'search.json').read_bytes())
new_search = json.loads((PREVIEW/'search.json').read_bytes())
old_probe_items = [item for item in old_search['items'] if item.get('location') == '1.1.html']
new_probe_items = [item for item in new_search['items'] if item.get('location') == '1.1.html']
assert len(old_probe_items) == len(new_probe_items) == 1
old_probe, new_probe = old_probe_items[0], new_probe_items[0]
assert old_probe['title'] == new_probe['title']
assert 'vocabulary' in new_probe.get('text', '') and 'vocabulary' not in new_probe['title']
assert '文字怎麼變成可還原的編號' in new_probe['title']
search_data_equal = old_probe == new_probe
assert all(any(item.get('location') == lesson['id']+'.html' for item in new_search['items']) for lesson in index)

kernel = read_ref(ROUND+'evidence/kernel-refresh-source22-v1.json', 'c490237cce379d5e2d47bb4d4fc4103ddd78ce48779f08e1fff51784e620d3ed')
equivalence = read_ref(ROUND+'evidence/format-only-source22-equivalence-v1.json', '16a9e24b2a20b62662e060f82fdcd1808f9995c2312a6f6b7bbacaafb2740f33')
assert kernel['source_freeze']['sha256'] == sha((ROOT/(ROUND+'source-freeze-v22.json')).read_bytes())
fresh = {row['lesson']: row for row in kernel['actual_executions']}
assert set(fresh) == {'19.25'} and kernel['retained_current_count'] == 331 and kernel['bundle_count'] == 332
build_info = json.loads((PREVIEW/'build-info.json').read_bytes())
assert build_info['revision'] == 'worktree-c-v3-source22-review'
assert build_info['lessons'] == 332 and build_info['executed_cpu_outputs'] is True
assert build_info['builder_version'] == importlib.metadata.version('zensical') == old_static['build_info']['builder_version']
rows = []
retained_count = teaching_code_cells = saved_output_blocks = total_code_cells = 0
for item in index:
    lesson_id = item['id']
    source_path = ROOT/item['notebook']
    copied_path = PREVIEW/item['notebook']
    executed_path = BUNDLE/Path(item['notebook']).relative_to('notebooks')
    source_sha = sha(source_path.read_bytes())
    assert source_sha == pages[lesson_id]['notebook_sha256'] == sha(copied_path.read_bytes())
    source = json.loads(source_path.read_bytes())
    checked = exporter.checked_notebook(source, executed_path)
    executed = json.loads(executed_path.read_bytes())
    executed_sha = sha(executed_path.read_bytes())
    if lesson_id in fresh:
        assert source_sha == fresh[lesson_id]['source_sha256'] and executed_sha == fresh[lesson_id]['executed_sha256']
        origin = 'one fresh source22 kernel; original actor /root, original times retained'
    else:
        retained_path = Path(kernel['retained_bundle'])/Path(item['notebook']).relative_to('notebooks')
        assert executed_sha == sha(retained_path.read_bytes())
        assert pages[lesson_id]['notebook_sha256'] == previous_pages[lesson_id]['notebook_sha256']
        retained_count += 1
        origin = 'source-identical executed bytes retained from v21; no kernel rerun by reviewer'
    expected = []
    lesson_code_count = 0
    for cell in checked['cells']:
        if cell['cell_type'] != 'code':
            continue
        total_code_cells += 1
        if cell.get('metadata', {}).get('course_setup') or 'course_figure' in cell.get('metadata', {}):
            continue
        teaching_code_cells += 1
        lesson_code_count += 1
        rendered = exporter.output_html(cell.get('outputs', []))
        if rendered:
            expected.append(output_signature(BeautifulSoup(rendered, 'html.parser').div))
    page_path = PREVIEW/(lesson_id+'.html')
    soup = BeautifulSoup(page_path.read_bytes(), 'html.parser')
    article = soup.select_one('article.md-content__inner')
    actual = [output_signature(div) for div in article.select('div.output')]
    assert expected == actual, lesson_id+' output signature'
    downloads = article.select('.actions a[download]')
    colab = [a for a in article.select('.actions a[href]') if a['href'].startswith('https://colab.research.google.com/')]
    assert len(downloads) == 1 and downloads[0]['href'] == item['notebook'] and len(colab) == 1
    saved_output_blocks += len(actual)
    rows.append({'id': lesson_id, 'source_notebook_sha256': source_sha, 'executed_sha256': executed_sha,
                 'rendered_html_sha256': sha(page_path.read_bytes()), 'teaching_code_cells': lesson_code_count,
                 'saved_rendered_output_blocks': len(actual), 'output_signature_sha256': sha(json.dumps(actual, sort_keys=True).encode()),
                 'source_cell_type_text_and_execution_outputs_checked': True, 'saved_and_rendered_output_signatures_equal': True,
                 'downloaded_notebook_matches_source22': True, 'download_and_colab_footer_checked': True, 'origin': origin})
assert retained_count == 331

result = {
    'reviewer_task': '/root/c_v3_technical_coord/canonical_index',
    'started_at': started_at, 'created_at': datetime.now(timezone.utc).isoformat(),
    'environment': {'python': sys.version.split()[0], 'platform': platform.platform(), 'zensical': importlib.metadata.version('zensical'), 'device': 'CPU; read-only JSON/HTML/hash inspection'},
    'scope': 'Source22 index only. No new research reproduction, browser run, kernel run, training, scientific final, installation, site rebuild, deployment or Colab runtime.',
    'source_freeze': {'path': ROUND+'source-freeze-v22.json', 'sha256': sha((ROOT/(ROUND+'source-freeze-v22.json')).read_bytes())},
    'original_sha256_unchanged': sha((ROOT/(ROUND+'technical-v21/canonical_index/original-v1.json')).read_bytes()),
    'native_frozen_body_equals_current_selector': True, 'index_unchanged_v21_to_v22': True,
    'old_home_to_current_only_count_331_to_332': True,
    'retained_original_evidence': [{'path': relative, 'sha256': expected, 'original_owner': raw[relative].get('reviewer_task'), 'original_created_at': raw[relative].get('created_at')} for relative, expected in raw_refs.items()],
    'paper': {'path_outside_git': str(paper_path), 'sha256': inspection['paper']['sha256'], 'bytes': inspection['paper']['bytes'], 'original_accessed_at': inspection['paper']['accessed_at'], 'original_owner': inspection['reviewer_task'], 'limited_reinspection': '2026-11-01, printed p1138/PDF page2; pdftotext read from same SHA-pinned original PDF', 'locator': inspection['paper']['locator'], 'short_excerpt': paper_excerpt, 'scope': inspection['derivation']['limits']},
    'screenshots_hash_checked': screenshots,
    'screenshots_manually_reinspected': ['mobile-index-v2.png', 'mobile-expanded-chapter-v2.png', 'desktop-body-search-v2.png'],
    'actual_preview': {'path': str(PREVIEW), 'build_info': build_info, 'index_html': record(PREVIEW/'index.html'), 'search_json': record(PREVIEW/'search.json')},
    'ui_assets_identical_to_original_observed_runtime': assets,
    'header_dom_identical': header_identical, 'menu_label_dom_identical': menu_identical,
    'drawer_toggle_dom_identical': drawer_toggle_identical, 'search_runtime_config_identical': config_identical,
    'navigation_delta': {'added': nav_added, 'removed': nav_removed, 'old_order_and_labels_retained': True, 'current_lesson_count': len(index), 'all_332_source_nav_and_rendered_primary_nav_entries_once': True},
    'search_1_1_entire_item_equal': search_data_equal,
    'search_probe_identity': {'old_title': old_probe['title'], 'current_title': new_probe['title'], 'current_title_query_present': True, 'body_only_query': 'vocabulary', 'present_in_current_body_not_title': True, 'old_probe_text_sha256': sha(old_probe.get('text', '').encode()), 'current_probe_text_sha256': sha(new_probe.get('text', '').encode()), 'all_332_search_pages_present': True},
    'browser_observation_retained': {'original_owner': browser['reviewer_task'], 'original_started_at': browser['started_at'], 'original_browser_version': browser['environment']['browser_version'], 'original_viewports': '1440x1000 desktop and 390x844 mobile', 'queries': [{'query': row['query'], 'expected_target_found': row['expected_target_found']} for row in browser['search_probes']], 'mobile_menu_position': browser['mobile_menu_position'], 'drawer_before': browser['mobile_drawer_before'], 'drawer_after': browser['mobile_drawer_after'], 'automated_exact_relative_href_selector_false': browser['mobile_first_lesson_link_visible_after_expand'], 'visual_resolution': 'SHA-matched expanded screenshot manually shows 1.1 in first chapter; do not treat false exact-href selector as visual absence.', 'not_a_fresh_browser_run': True},
    'output_bundle': {'path': str(BUNDLE), 'source22_kernel_receipt': {'path': ROUND+'evidence/kernel-refresh-source22-v1.json', 'sha256': sha((ROOT/(ROUND+'evidence/kernel-refresh-source22-v1.json')).read_bytes())}, 'original_kernel_actor': kernel['actor'], 'original_receipt_created_at': kernel['created_at'], 'fresh_original_executions': kernel['actual_executions'], 'source_identical_retained_v21_executed_files': retained_count, 'current_source_notebooks': len(rows), 'total_code_cells_including_setup_figures': total_code_cells, 'teaching_code_cells': teaching_code_cells, 'saved_rendered_output_blocks': saved_output_blocks, 'all_332_current_source_execution_and_render_matches': True},
    'rows': rows,
    'limits': ['Old browser/paper evidence retains its owner/date/scope; no old pass verdict is evidence.', 'Local source22 preview only; no public deployment or all-phone-browser claim.', 'Saved execution records, matching cell text/types/counts/no errors, and HTML signatures establish display/provenance identity, not all-book mathematical or scientific correctness.', 'Reading-time values and C3 product/research claims are outside this index review.'],
}
with (ROOT/(HERE+'retention-proof-v1.json')).open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({'result': HERE+'retention-proof-v1.json', 'sha256': sha((ROOT/(HERE+'retention-proof-v1.json')).read_bytes()), 'lessons': len(rows), 'teaching_code_cells': teaching_code_cells, 'saved_rendered_output_blocks': saved_output_blocks, 'search_item_identical': search_data_equal, 'navigation_delta': nav_added}, ensure_ascii=False))
