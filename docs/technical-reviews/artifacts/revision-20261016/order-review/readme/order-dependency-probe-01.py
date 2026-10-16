"""Actual original-owner informed README order-dependency check; no model or remote work."""
from datetime import datetime, UTC
import difflib
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import tomllib

ROOT = Path(__file__).resolve().parents[6]
OUT = Path(__file__).parent
PRIOR = ROOT/'docs/course-revision-20261016/reviews/technical-readme-backfill-callback-03.json'
PREVIOUS_COMMIT = 'f8188f88d205ca92a40fd53312d29e31ec17e2e7'
def sha(raw):
    return hashlib.sha256(raw).hexdigest()
def file_sha(path):
    return sha((ROOT/path).read_bytes())
def old_bytes(path):
    return subprocess.check_output(['git', 'show', PREVIOUS_COMMIT+':'+path])
def sections(text):
    matches = list(re.finditer(r'^## ([A-Z\d]+\.\d+) .+$', text, re.M))
    return text[:matches[0].start()], {m[1]: text[m.start(): matches[i+1].start() if i+1 < len(matches) else len(text)] for i, m in enumerate(matches)}
def norm(value):
    if isinstance(value, dict):
        return {k: norm(v) for k,v in value.items()}
    if isinstance(value, list):
        return sorted((norm(v) for v in value), key=lambda v: json.dumps(v, sort_keys=True, ensure_ascii=False))
    return value
def leaves(value):
    if isinstance(value, list):
        return [leaf for v in value for leaf in leaves(v)]
    if isinstance(value, dict):
        return [leaf for k,v in value.items() for leaf in ([(k,v)] if isinstance(v,str) else leaves(v))]
    return []
prior = json.loads(PRIOR.read_bytes())
assert sha(PRIOR.read_bytes()) == '571427aa4b2c971b27ded747e468759e1964d3b753c96eb1ec61f33dc5bddc10'
freeze = json.loads((ROOT/'docs/course-revision-20261016/freeze-18.json').read_bytes())
readme_page = next(p for p in freeze['pages'] if p['page_id'] == 'readme')
assert file_sha('README.md') == prior['source_sha256'] == readme_page['source_sha256']
assert (ROOT/'README.md').read_bytes() == (ROOT/readme_page['snapshot']).read_bytes()
changed = []
identities = []
for s in prior['sources']:
    if s['kind'] == 'repository_code':
        actual = file_sha(s['path'])
        identities.append({'id': s['id'], 'path': s['path'], 'previous_sha256': s['sha256'], 'current_sha256': actual, 'unchanged': actual == s['sha256']})
        if actual != s['sha256']:
            changed.append(s['id'])
assert changed == ['s_index', 's_site', 's_route', 's_ch19'], changed
diffs = {}
for s in prior['sources']:
    if s['id'] in changed:
        before = old_bytes(s['path'])
        assert sha(before) == s['sha256']
        diffs[s['path']] = ''.join(difflib.unified_diff(before.decode().splitlines(True), (ROOT/s['path']).read_text().splitlines(True), fromfile=PREVIOUS_COMMIT+':'+s['path'], tofile='current:'+s['path']))
old_index = json.loads(old_bytes('course/lesson-index.json'))
index = json.loads((ROOT/'course/lesson-index.json').read_bytes())
assert len(index) == len(old_index) == 312
assert len({x['id'] for x in index}) == 312
assert {x['id']:x for x in index} == {x['id']:x for x in old_index}
missing = [x['notebook'] for x in index if not (ROOT/x['notebook']).is_file()]
assert not missing
config = tomllib.loads((ROOT/'zensical.toml').read_text())
old_config = tomllib.loads(old_bytes('zensical.toml').decode())
nav = config['project'].pop('nav')
old_nav = old_config['project'].pop('nav')
assert config == old_config
assert norm(nav) == norm(old_nav)
nav_leaves = leaves(nav)
old_leaves = leaves(old_nav)
nav_ids = [path[:-3] for _,path in nav_leaves if re.fullmatch(r'[A-Z\d]+\.\d+\.md',path)]
old_nav_ids = [path[:-3] for _,path in old_leaves if re.fullmatch(r'[A-Z\d]+\.\d+\.md',path)]
index_ids = [x['id'] for x in index]
old_index_ids = [x['id'] for x in old_index]
assert len(nav_ids) == len(set(nav_ids)) == 312 and set(nav_ids) == set(index_ids)
assert [x for x in nav_ids if not x.startswith('C.')] == [x for x in index_ids if not x.startswith('C.')]
assert [x for x in old_nav_ids if not x.startswith('C.')] == [x for x in old_index_ids if not x.startswith('C.')]
c_positions = {'current_index':{x:index_ids.index(x) for x in index_ids if x.startswith('C.')}, 'current_nav':{x:nav_ids.index(x) for x in nav_ids if x.startswith('C.')}, 'prior_index':{x:old_index_ids.index(x) for x in old_index_ids if x.startswith('C.')}, 'prior_nav':{x:old_nav_ids.index(x) for x in old_nav_ids if x.startswith('C.')}}
assert c_positions['current_index'] == c_positions['prior_index'] and c_positions['current_nav'] == c_positions['prior_nav']
chapter_orders = {}
for chapter,count in [('14',18),('19',20)]:
    expected = [f'{chapter}.{n}' for n in range(1,count+1)]
    from_index = [x['id'] for x in index if x['id'].startswith(chapter+'.')]
    from_nav = [x for x in nav_ids if x.startswith(chapter+'.')]
    text = (ROOT/f'course/chapters/{chapter}.md').read_text()
    intro,parts = sections(text)
    assert list(parts) == from_index == from_nav == expected
    chapter_orders[chapter] = {'source_order': list(parts), 'index_order': from_index, 'nav_order': from_nav, 'current_intro': intro, 'source_sha256': file_sha(f'course/chapters/{chapter}.md')}
old_intro,old_parts = sections(old_bytes('course/chapters/19.md').decode())
intro,parts = sections((ROOT/'course/chapters/19.md').read_text())
assert set(parts) == set(old_parts)
section_changes = [key for key in parts if parts[key] != old_parts[key]]
assert section_changes == ['19.12'], section_changes
old_ending = '接著從[19.14](19.md#19.14)讀共同切分，再完成Looped解題器、機率判斷器與Auto的路線。'
new_ending = '接著在[19.13](19.md#19.13)對照三個成品的用途，再從[19.14](19.md#19.14)讀共同切分，完成Looped解題器、機率判斷器與Auto的路線。'
assert parts['19.12'].replace(new_ending, old_ending, 1) == old_parts['19.12']
reconstructed = old_intro + ''.join(old_parts['19.12'] if key == '19.12' else parts[key] for key in old_parts)
assert sha(reconstructed.encode()) == sha(old_bytes('course/chapters/19.md'))
assert parts['19.13'] == old_parts['19.13']
assert parts['19.19'] == old_parts['19.19'] and parts['19.20'] == old_parts['19.20']
route_before = old_bytes('course/README.md').decode()
route_after = (ROOT/'course/README.md').read_text()
route_intro,route_parts = sections(route_after)
old_route_intro,old_route_parts = sections(route_before)
assert route_intro == old_route_intro
assert [key for key in route_parts if route_parts[key] != old_route_parts[key]] == ['R.3']
assert '第14章依序讀14.1–14.6' in route_parts['R.3']
assert '第19章先沿19.1–19.12' in route_parts['R.3']
for needle in ('必修[7.23–7.25]', '再讀[C.1–C.5]', '必修[9.11–9.16]', '必修[14.11–14.18]'):
    assert needle in route_after
assert nav_ids[nav_ids.index('7.25')+1:nav_ids.index('8.1')] == [f'C.{i}' for i in range(1,6)]
assert nav_ids[nav_ids.index('13.17')+1:nav_ids.index('14.1')] == ['C.6','C.7']
chapter_files = sorted((ROOT/'course/chapters').glob('*.md'))
chapter_count = sum(len(re.findall(r'^## [A-Z\d]+\.\d+ ', p.read_text(), re.M)) for p in chapter_files)
front = {name: len(re.findall(r'^## [A-Z\d]+\.\d+ ', (ROOT/'course'/name).read_text(), re.M)) for name in ['README.md','first-steps.md','training.md','glossary.md']}
assert chapter_count == 312 and chapter_count+sum(front.values()) == 338
assert len([p for p in chapter_files if p.stem.isdecimal()]) == 20
assert len(chapter_files) == 23
artifacts = []
for a in prior['artifacts']:
    actual = file_sha(a['path'])
    assert actual == a['sha256'], a['path']
    artifacts.append({'id': a['id'], 'path': a['path'], 'sha256': actual, 'unchanged': True})
for path,digest in prior['figure_sha256'].items():
    assert file_sha(path) == digest
build = subprocess.run(['uv','run','--frozen','--extra','cpu','python','scripts/build_course.py','--check'], cwd=ROOT, capture_output=True, text=True)
assert build.returncode == 0, build.stderr
readme_links = []
for path,anchor in [('course/chapters/07.md','7.23'),('course/chapters/14.md','14.11'),('course/chapters/09.md','9.11'),('course/chapters/19.md','19.13'),('course/chapters/19.md','19.20')]:
    assert re.search(r'^## '+re.escape(anchor)+r' ',(ROOT/path).read_text(),re.M)
    readme_links.append({'path':path,'anchor':anchor,'heading_exists':True})
result = {'at': datetime.now(UTC).isoformat(), 'reviewer_task': '/root/tech_transport_readme_20261016', 'current_context': 'informed', 'scope': 'README current whole-page order-dependency applicability; retain original44 claims and own03 evidence. This is not a fresh blind review.', 'command': 'uv run --frozen --extra cpu python '+Path(__file__).relative_to(ROOT).as_posix(), 'environment': {'python':platform.python_version(),'device':'CPU; stdlib source/structure/hash checks, build_course --check only; no model execution'}, 'prior': {'path': PRIOR.relative_to(ROOT).as_posix(),'sha256':sha(PRIOR.read_bytes())}, 'freeze': {'path':'docs/course-revision-20261016/freeze-18.json','sha256':file_sha('docs/course-revision-20261016/freeze-18.json')}, 'readme_sha256':file_sha('README.md'), 'changed_source_ids':changed, 'source_identities':identities, 'unchanged_prior_artifact_identities':artifacts, 'diffs':diffs, 'chapter_orders':chapter_orders, 'index_check': {'count':len(index),'record_content_identical_by_id':True,'missing_notebooks':missing,'nav_and_index_same_id_set':True,'full_nav_lesson_order_equals_index':False,'non_C_nav_lesson_order_equals_index':True,'C_positions':c_positions,'C_order_difference_unchanged_from_prior03':True,'all_non_nav_configuration_identical':True,'nav_group_membership_and_metadata_identical':True}, 'chapter19_check': {'only_changed_section_content':'19.12 final transition sentence','19.13_moved_bytes_identical':True,'all_other19_section_bytes_identical':True,'reconstructed_previous_whole_sha256':sha(reconstructed.encode()),'matches_prior03_whole_pin':True,'personally_read_current_intro_19_12_19_13':True}, 'route_check': {'personally_read_whole_current_route':True,'only_changed_section':'R.3','C1_5_after7_and_C6_7_after13':True,'mandatory_effort_looped_jev_preserved':True,'readme_anchor_checks':readme_links}, 'counts': {'chapter_source_files':23,'numeric_chapters':20,'chapter_sections':chapter_count,'notebook_records':len(index),'front_sections':front,'total_numbered_sections':chapter_count+sum(front.values()),'canonical_pages':len(freeze['pages'])}, 'build_check': {'command':'uv run --frozen --extra cpu python scripts/build_course.py --check','exit_code':build.returncode,'stdout':build.stdout,'stderr':build.stderr}, 'judgment': 'All44 README claims remain applicable. Changed index/config only reorder unchanged records. New R.3 sequence agrees with current source/index/nav within14/19; mandatory effort/Jev/Looped/C navigation placements remain.19.13 A/B/C semantics and19.19/19.20 operation/results bytes are identical;19.12 adds navigation only, preserving A failure/limited-scope conclusions. README itself and SVG are unchanged. No necessary README correction.', 'resolved_probe_observation':'Initial probe incorrectly required full inventory order to equal site reading order. Index listsC at the end, while site nav insertsC after7/13. Actual old/current positions are identical, and README promises the reading route rather than full inventory order. No current claim depends on those two lists having identical global order; numeric14/19 order and identical inventory were verified separately.', 'issues': [], 'not_rerun':'No HF32 download, models/inference, raw score recount, formaltest, notebook kernel, GPU/training, paid work, remote write, browser UI or reading-time certification. Own03 CSS/UI evidence is retained after unchanged source identity, not rerun. No other reviewer report or author problem list read.'}
path = OUT/'order-dependency-results-01.json'
assert not path.exists()
path.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({'changed_source_ids':changed,'counts':result['counts'],'chapter19_changed_sections':section_changes,'build':build.stdout.strip(),'issues':[]},ensure_ascii=False))
