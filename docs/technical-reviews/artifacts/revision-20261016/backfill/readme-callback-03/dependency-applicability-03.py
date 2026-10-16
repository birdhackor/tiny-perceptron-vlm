"""Same-owner informed CSS-only continuity callback. Preserve all earlier reports and receipts."""
import copy
from datetime import datetime, UTC
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import subprocess

ROOT = Path(__file__).resolve().parents[6]
OUT = Path(__file__).parent
BASE = OUT.relative_to(ROOT).as_posix()+'/'
PRIOR = 'docs/course-revision-20261016/reviews/technical-readme-backfill-callback-02.json'
TARGET = 'docs/course-revision-20261016/reviews/technical-readme-backfill-callback-03.json'
TASK = '/root/tech_transport_readme_20261016'
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):
    return json.loads(Path(path).read_bytes())
def save(path, value):
    assert not Path(path).exists(), 'Preserve existing records: '+str(path)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')

preserved = {
    'docs/course-revision-20261016/reviews/technical-readme-transport-original-01.json': 'e468fca8c6faec3b680d345688b4734b835206b81c18821b3dd172f4ba701c80',
    'docs/course-revision-20261016/reviews/technical-readme-transport-original-02.json': '262804487a60271d249fd76579f2a69eac046b8ffeb12d0f323335b312fd3d78',
    'docs/course-revision-20261016/reviews/technical-readme-backfill-callback-01.json': '4e3d9080205588b53e8c8ca326677ffd5c5ea3d5e75514f54700e117c74cc351',
    PRIOR: '82e1241c5097486366ceb0621ef56cb0651f0c0ff1c77a12d5761fbc91f4342f',
}
for path, digest in preserved.items():
    assert sha(ROOT/path) == digest, path
prior = load(ROOT/PRIOR)
ui = load(OUT/'css-ui-results-03.json')
followup = load(OUT/'css-ui-followup-results-03.json')
assert ui['all_checks_passed'] and followup['passed']
assert sha(ROOT/'README.md') == prior['source_sha256']
source_identities = []
changed = []
for source in prior['sources']:
    if source['kind'] == 'repository_code':
        actual = sha(ROOT/source['path'])
        row = {'id': source['id'], 'path': source['path'], 'previous_sha256': source['sha256'], 'current_sha256': actual, 'unchanged': actual == source['sha256']}
        source_identities.append(row)
        if not row['unchanged']:
            changed.append(source['id'])
assert changed == ['s_css'], changed
artifact_identities = []
for artifact in prior['artifacts']:
    actual = sha(ROOT/artifact['path'])
    assert actual == artifact['sha256'], artifact['path']
    artifact_identities.append({'id': artifact['id'], 'path': artifact['path'], 'sha256': actual, 'unchanged': True})
for path, digest in prior['figure_sha256'].items():
    assert sha(ROOT/path) == digest
old_css = subprocess.run(['git', 'show', 'HEAD:course/web/course.css'], capture_output=True, check=True).stdout
assert hashlib.sha256(old_css).hexdigest() == next(s['sha256'] for s in prior['sources'] if s['id'] == 's_css')
assert ui['served_css']['sha256'] == sha(ROOT/'course/web/course.css')
for path, digest in ui['checked_files'].items():
    assert sha(ROOT/path) == digest
checked_files = [{'path': 'README.md', 'sha256': sha(ROOT/'README.md'), 'inspection': 'Current entire README personally reread; unchanged from callback02/source16.'}]
checked_files.extend({'path': row['path'], 'sha256': row['current_sha256'], 'inspection': 'Current CSS full text and exact diff personally read, then actual UI checked.' if row['id'] == 's_css' else 'Byte identity checked; reuse my original inspection/execution, not a new full reread.'} for row in source_identities)
checked_files.extend({'path': path, 'sha256': digest, 'inspection': 'Unchanged SVG identity; original personal render evidence retained; visible within current page.'} for path, digest in prior['figure_sha256'].items())
limits = '本輪只做CSS dependency continuity intake：本人重讀README整頁與CSS全文／diff，核其餘已pin來源及primary artifact同byte，並在preview17實測必要UI。未重跑formaltest、匿名HF32檔、CPU模型推論、raw分數重算、312個Notebook、GPU訓練、付費或遠端寫入。未驗全部21個capstone route頁、所有瀏覽器／viewport、縮放與減少動畫互動；後兩者實作bytes未變，沿用本人舊檢視。'
judgments = [
    {'change': 'Only added max-width59.984375em rules moving mobile TOC into header and reserving title space.', 'original_source': 'Personally read entire current course/web/course.css and exact diff against callback02-identical HEAD bytes.', 'actual': '390x844: fixed sidebar/button x237,y6 before and after900px scroll; title right231 with48px margin, button begins237; popup below48px header and inside right edge; TOC anchor navigates to README installation section.', 'effect': 'Presentation-only change leaves README text, numbers, figure labels and44 substantive claims unchanged. c11 navigation support is directly checked anew; c12 zoom/reduced-motion selectors remain identical.', 'affected_claim_ids': ['c11', 'c12'], 'necessary_readme_issue': False},
    {'change': 'Hide the TOC when search or drawer is checked.', 'original_source': 'Current CSS sibling selectors and actual checkbox/computed visibility in browser.', 'actual': 'Search opens with TOC hidden; actual2928 query finds README; selecting that result closes search and restores visible TOC. Drawer opens with TOC hidden; viewport click360,500 outside drawer closes it and restores TOC, also visually checked after500ms.', 'effect': 'New mobile rules do not obstruct tested search/drawer operations. No claim of Escape-close support: it did not close search with either prior or current CSS; preserved as unchanged operation limitation.', 'affected_claim_ids': ['c11'], 'necessary_readme_issue': False},
    {'change': 'Desktop remains outside the new media block.', 'original_source': 'Current max-width media query and actual prior-CSS-only browser control.', 'actual': '1280x900 header/title/sidebar/button/popup geometry and media state exactly equal between current CSS and prior CSS served into otherwise identical preview.959px matches mobile;960px does not.', 'effect': 'No tested desktop layout regression. This is not all-device or all-route certification.', 'affected_claim_ids': ['c11', 'c12'], 'necessary_readme_issue': False},
]
result = {'at': datetime.now(UTC).isoformat(), 'reviewer_task': TASK, 'current_context': 'informed', 'phase': 'continuity-phase', 'command': 'uv run --frozen --extra cpu python '+BASE+'dependency-applicability-03.py', 'scope': 'Same-owner narrow stylesheet applicability callback03; full44-claim report retained with explicit inherited evidence provenance.', 'prior_report': {'path': PRIOR, 'sha256': sha(ROOT/PRIOR)}, 'source_freeze_reference': 'docs/course-revision-20261016/freeze-16.json', 'readme_sha256': sha(ROOT/'README.md'), 'chapter19_sha256': sha(ROOT/'course/chapters/19.md'), 'source_identities': source_identities, 'artifact_identities': artifact_identities, 'checked_files': checked_files, 'css_diff': ui['css_diff'], 'new_ui_checks': [{'name': c['name'], 'passed': c['passed']} for c in ui['checks']], 'judgments': judgments, 'all44_claims_applicable': True, 'issues': [], 'limits': limits, 'preserved_reports': preserved, 'environment': {'python': platform.python_version(), 'device': 'CPU; local source/hash checks and direct validator only; UI execution recorded separately'}, 'independence': 'No other technical report or author fix plan read. Parent later notified another UI owner had finished; that notification was not used as evidence and its report was not read. My CSS source judgment and actual browser receipts were recorded independently.'}
result_path = OUT/'dependency-applicability-results-03.json'
save(result_path, result)
r = copy.deepcopy(prior)
r.update(recorded_at=datetime.now(UTC).isoformat(), review_round='original-owner-informed-narrow-stylesheet-callback-03', current_context='informed', source_freeze_reference='docs/course-revision-20261016/freeze-16.json', continuity_phase='continuity-phase', checked_files=checked_files)
r['previous_callback'] = {'path': PRIOR, 'sha256': sha(ROOT/PRIOR), 'preserved_bytes': True, 'scope_at_that_time': 'Original-owner callback02:44 README claims retained after personally checking19.13 dependency changes. Current03 is narrower CSS-only intake.'}
for source in r['sources']:
    if source['id'] == 's_css':
        source['callback02_sha256'] = source['sha256']
        source['callback02_inspection_note'] = source['inspection_note']
        source['sha256'] = sha(ROOT/source['path'])
        source['version'] = '2026-10-16 current worktree CSS; preview17 exact served bytes; original-owner callback03'
        source['inspection_note'] = 'Personally read complete current CSS and exact8-line mobile block insertion against prior02-identical HEAD CSS. Existing SVG, zoom and reduced-motion rules are byte-identical. Actual390px TOC placement, popup, title slot, search/drawer hidden/restored states and1280px prior/current CSS comparison were checked; source and screenshot hashes in own03 receipts.'
for artifact in r['artifacts']:
    artifact['description'] = '承接本人02及更早原始證據；本次核bytes一致，未重新執行。原用途：'+artifact['description']
new_artifacts = [
    ('a_dependency03', 'execution', 'dependency-applicability-results-03.json', '本人本輪來源身分、CSS差異與44主張適用性判斷。', result['command'], 'Only s_css differs; README/whole19/SVG and all remaining repository sources/17 primary artifacts match02.44 claims remain applicable; no new necessary README issue.'),
    ('a_css_ui03', 'execution', 'css-ui-results-03.json', '實際preview17的13項CSS/UI檢查；截圖留ignored work並記SHA。', 'uv run --frozen --extra cpu python '+BASE+'css-ui-probe-03.py', '13/13 checks pass with systemChromium; actual search result navigation and overlay click restore TOC; desktop prior/current CSS geometry identical.'),
    ('a_css_followup03', 'execution', 'css-ui-followup-results-03.json', '實際舊／新CSS對照Escape操作限制，並查看抽屜關閉500ms後頁面。', 'uv run --frozen --extra cpu python '+BASE+'css-ui-followup-03.py', 'Escape leaves search open under both prior/current CSS; this unchanged operation limit is preserved. Drawer closes and TOC visible after500ms; settled screenshot personally viewed.'),
    ('a_css_visual03', 'source_snapshot', 'personal-visual-inspection-03.json', '本人親看必要手機／桌面截圖的記錄與SHA；不含PNG payload。', None, None),
    ('a_css_attempt01', 'source_snapshot', 'probe-attempt-01.json', '保留第一次探測selector相對座標錯誤與原碼／截圖SHA。', None, None),
    ('a_css_attempt02', 'source_snapshot', 'probe-attempt-02-results.json', '保留第二次探測的未通過關閉操作假設及實際state；不是刪除失敗再宣稱全通。', None, None),
    ('a_css_attempt02_note', 'source_snapshot', 'probe-attempt-02-note.json', '本人說明Escape未關閉和scrollbar外點擊假設，後續用真正UI操作核清。', None, None),
    ('code_dependency03', 'code', 'dependency-applicability-03.py', '本次身分檢查、完整報告建構與direct validator程式。', None, None),
    ('code_css_ui03', 'code', 'css-ui-probe-03.py', '本次實際Chromium UI探測原碼；無模型操作。', None, None),
    ('code_css_followup03', 'code', 'css-ui-followup-03.py', '本次CSS控制組及settled drawer檢查原碼。', None, None),
]
for identifier, kind, name, description, command, observed in new_artifacts:
    entry = {'id': identifier, 'kind': kind, 'path': BASE+name, 'sha256': sha(OUT/name), 'description': description}
    if kind == 'execution':
        entry.update(command=command, result=observed, environment=result['environment'] if identifier == 'a_dependency03' else ui['environment'])
    r['artifacts'].append(entry)
for claim in r['claims']:
    claim['callback02_revalidation'] = claim.pop('callback_revalidation')
    claim['callback_revalidation'] = {'round': '03', 'current_context': 'informed', 'scope': 'Narrow CSS dependency applicability only.', 'status': 'verified', 'dependency_relevance': 'Current CSS full text/diff and actual necessary UI independently checked; no relevant presentation conflict.' if claim['id'] in ('c11', 'c12') else 'README text/source16, required implementation and primary evidence unchanged; new CSS block changes only mobile TOC presentation and does not change this claim.', 'execution_provenance': 'Retain my02/01/original technical evidence; no new model, score, formaltest or anonymousHF32 verification.'}
    claim['artifact_ids'].append('a_dependency03')
    if claim['id'] in ('c11', 'c12'):
        claim['callback02_scope'] = claim['scope']
        claim['artifact_ids'].extend(['a_css_ui03', 'a_css_followup03', 'a_css_visual03'])
    if claim['id'] == 'c11':
        claim['scope'] = '原設定／HTML證據沿用；本輪在preview17另實測390px TOC、搜尋README及抽屜操作和1280px CSS控制組。未作所有路徑／裝置或palette／copy的完整互動驗收。Escape未關閉搜尋在舊／新CSS一致，不宣稱該快捷操作受支援。'
        claim['verification']['observed'] += ' Callback03另實際通過13項窄CSS/UI checks；搜尋2928後選README結果會關閉面板且恢復TOC，抽屜外側click360,500亦可關閉恢復，桌面幾何不變。'
    if claim['id'] == 'c12':
        claim['scope'] = '原輪次已親看CairoSVG、核zoom/reduced-motion原碼；本輪SVG與相關規則SHA／bytes未變，並實際在Chromium頁面看到圖解。只核新TOC樣式的影響，沒有新增點擊放大或系統reduced-motion實機驗收。原輪次Chromium失敗紀錄仍保留，本輪用既有systemChromium完成窄UI。'
r['callback02_dependency_audit'] = r.pop('callback_dependency_audit')
r['callback_dependency_audit'] = {'round': '03', 'phase': 'continuity-phase', 'artifact_id': 'a_dependency03', 'scope': result['scope'], 'judgments': judgments, 'changed_repository_source_ids': changed, 'all_other_identities_unchanged': True, 'readme_sha256': result['readme_sha256'], 'chapter19_current_sha256': result['chapter19_sha256'], 'css_previous_sha256': hashlib.sha256(old_css).hexdigest(), 'css_current_sha256': sha(ROOT/'course/web/course.css'), 'actual_ui_checks': len(ui['checks']), 'actual_ui_checks_passed': True, 'limits': limits}
r['callback02_scope_and_independence'] = r.pop('scope_and_independence')
r['scope_and_independence'] = {'scope': '同一原owner的informed CSS-only callback03，承接本人02完整44claims，不是新盲審或把44claims全部重新執行。', 'role': 'reviewer_context=fresh只保留初始task角色；current_context=informed。', 'prior_report_preserved': r['previous_callback'], 'source_materials': result['independence'], 'new_work': 'Read current whole README and CSS/diff; checked all other pinned source/primary receipt identities; actual local preview17 Chromium UI and prior-CSS-only controls; direct validator.', 'not_rerun': limits, 'not_human_testing': True}
r['checks']['factual_accuracy']['details'] = '本人02與更早原始查證的44claims保留；本輪README整頁實讀且SHA相同，所有實作／primary receipts除CSS皆同byte。親讀新CSS並實際核必要UI，對44claims及轉介無新必要問題。'
r['checks']['numeric_verification']['details'] = '本輪沒有重算模型分數或重跑formaltest。原數字、資料、實作與收據SHA均相同；CSS只改呈現。新UI幾何為實際Chromium測量：390px x237,y6，title margin48px；1280px幾何與舊CSS完全相同；959/960 media邊界吻合。'
r['checks']['figure_consistency']['details'] = 'README唯一SVG的SHA與02相同；原親看CairoSVG證據保留。本輪在實際README頁面親看圖解與手機收合／展開TOC，未見新TOC遮住正文行尾；未重測SVG zoom/reduced-motion互動。'
r['checks']['source_verification']['details'] = '本人實讀當前README、CSS全文及與02-identical HEAD CSS的exact差異，核preview17實際serves目前CSS SHA。其餘所有repository sources與17份原primary artifacts核SHA一致；整19仍c41d949d，沿用本人02已親讀19.13的判斷，未僅由root換hash。未讀其他人的technical report或fix plan。'
r['checks']['limitations']['details'] = limits+' 搜尋Escape在新舊CSS均未關閉；本輪以真實搜尋結果導航及overlay點擊完成關閉，未擅稱Escape正常。原Actions client failure與remote completed分層、有限能力等判斷原樣保留。'
r['issues'] = []
target = ROOT/TARGET
save(target, r)
spec = importlib.util.spec_from_file_location('readme_cb03_validator', ROOT/'scripts/check_technical_reviews.py')
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)
errors = validator._validate(ROOT, ROOT/'README.md', 'readme', (ROOT/'README.md').read_text(), r, {TASK: ['readme']}, set())
validation_path = OUT/'direct-validation-03.json'
save(validation_path, {'at': datetime.now(UTC).isoformat(), 'command': result['command'], 'report': {'path': TARGET, 'sha256': sha(target)}, 'source_sha256': sha(ROOT/'README.md'), 'current_context': 'informed', 'scope': 'Direct _validate on current whole README/full44-claim report. New execution only narrow CSS/UI; old report02 preserved.', 'errors': errors})
assert not errors, errors
for path, digest in preserved.items():
    assert sha(ROOT/path) == digest
seal_path = OUT/'owner-callback-seal-03.json'
save(seal_path, {'schema_version': 1, 'sealed_at': datetime.now(UTC).isoformat(), 'reviewer_task': TASK, 'lesson_id': 'readme', 'role': 'technical', 'current_context': 'informed', 'phase': 'continuity-phase', 'round_type': 'same-owner narrow stylesheet applicability callback03; no new blind review', 'report': {'path': TARGET, 'sha256': sha(target)}, 'source': {'path': 'README.md', 'sha256': sha(ROOT/'README.md')}, 'chapter19': {'path': 'course/chapters/19.md', 'sha256': sha(ROOT/'course/chapters/19.md'), 'unchanged_from_callback02': True}, 'stylesheet': {'path': 'course/web/course.css', 'sha256': sha(ROOT/'course/web/course.css'), 'personally_read_full_text_and_diff': True, 'preview17_exact_served_bytes': True}, 'verdict': r['verdict'], 'claim_count': len(r['claims']), 'claims_provenance': 'All44 inherited from my02 and underlying original evidence; new narrow CSS applicability/UI verification only.', 'previous_callback': r['previous_callback'], 'validation': {'path': validation_path.relative_to(ROOT).as_posix(), 'sha256': sha(validation_path), 'errors': errors}, 'new_artifacts': {a['path']: a['sha256'] for a in r['artifacts'] if a['id'] in {x[0] for x in new_artifacts}}, 'preserved_reports': preserved, 'limits': limits, 'operation_limits_preserved': 'Search Escape unchanged under old/currentCSS; actual search-result selection and outside-drawer click checked.', 'independence': result['independence']})
print(json.dumps({'report': TARGET, 'report_sha256': sha(target), 'seal': seal_path.relative_to(ROOT).as_posix(), 'seal_sha256': sha(seal_path), 'validation_sha256': sha(validation_path), 'verdict': r['verdict'], 'claims': len(r['claims']), 'ui_checks': len(ui['checks']), 'errors': errors}, ensure_ascii=False))
