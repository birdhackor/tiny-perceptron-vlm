"""Seal my completed narrow informed README order callback; never rewrite earlier reports."""
import copy
from datetime import datetime, UTC
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[6]
OUT = Path(__file__).parent
BASE = OUT.relative_to(ROOT).as_posix()+'/'
PRIOR = 'docs/course-revision-20261016/reviews/technical-readme-backfill-callback-03.json'
TARGET = 'docs/course-revision-20261016/reviews/technical-order-readme-callback-01.json'
TASK = '/root/tech_transport_readme_20261016'
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):
    return json.loads(Path(path).read_bytes())
def save(path, value):
    assert not Path(path).exists(), 'Preserve prior bytes: '+str(path)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
assert sha(ROOT/PRIOR) == '571427aa4b2c971b27ded747e468759e1964d3b753c96eb1ec61f33dc5bddc10'
prior = load(ROOT/PRIOR)
evidence = load(OUT/'order-dependency-results-01.json')
assert evidence['issues'] == []
assert sha(ROOT/'README.md') == prior['source_sha256'] == evidence['readme_sha256']
for row in evidence['source_identities']:
    assert sha(ROOT/row['path']) == row['current_sha256']
for row in evidence['unchanged_prior_artifact_identities']:
    assert sha(ROOT/row['path']) == row['sha256']
assert sha(ROOT/evidence['freeze']['path']) == evidence['freeze']['sha256']
r = copy.deepcopy(prior)
r.update(recorded_at=datetime.now(UTC).isoformat(), review_round='original-owner-informed-order-callback-01', current_context='informed', source_freeze_reference='docs/course-revision-20261016/freeze-18.json', continuity_phase='order-correction technical callback')
r['previous_callback'] = {'path': PRIOR, 'sha256': sha(ROOT/PRIOR), 'preserved_bytes': True, 'scope_at_that_time': 'Same-owner narrow stylesheet callback03 retaining my44 original substantive README claims.'}
r['checked_files'] = [{'path':'README.md','sha256':sha(ROOT/'README.md'),'inspection':'Current whole-file source identity equals own03 and freeze18 snapshot; text unchanged.'}]+[{'path':row['path'],'sha256':row['current_sha256'],'inspection':'Changed source actually read and structurally compared in this round.' if row['id'] in evidence['changed_source_ids'] else 'Exact unchanged bytes; reuse my original inspection/execution rather than claim a new full reread.'} for row in evidence['source_identities']]
r['checked_files'].extend([{'path':'course/chapters/14.md','sha256':evidence['chapter_orders']['14']['source_sha256'],'inspection':'Personally read current chapter introduction/headings; checked section IDs against actual14 index/nav. Route context only, not technical re-review of14 methods.'}, {'path':evidence['freeze']['path'],'sha256':evidence['freeze']['sha256'],'inspection':'Current readme whole-file source/figure binding and canonical-page inventory used; no reviewer verdicts used.'}])
notes = {
    's_index':'Actually parsed all312 current records and compared every record field byID against bytes pinned by own03: identical ID/title/source/notebook records; only14/19 order changes. All312 Notebook paths exist and build_course --check passes. Chapter14/19 orders agree with current source and nav. C is at inventory end while nav inserts it after7/13; actual old/current positions are identical, so full global order equality is not claimed.',
    's_site':'Personally read changed14/19 nav and current theme/features/palettes/MathJax configuration. Parsed whole TOML and proved every non-nav field identical to own03; normalized nav group membership/labels/paths also identical. Current14/19 leaf order is numeric and matches source/index. No new UI, MathJax, theme or copy interaction run.',
    's_route':'Personally read complete current R.1–R.4 and exact diff against own03 source bytes. Only R.3 paragraph changes:14.1–14.6 then14.7–14.10 then14.11–14.18;19.1–19.12 then19.13–19.20. Actual source/index/nav agree; mandatory effort/Jev/Looped and C after7/13 remain. R.2/R.4 task-based19.13 jump entries remain valid; no model capability statement changed.',
    's_ch19':'Personally read current introduction, full19.12 and full19.13 plus exact whole-file diff. Parsed every numbered section: only19.12 final transition sentence changes,19.13 moves with identical bytes, all other sections including19.19/19.20 identical. Rebuilt the whole prior file with old intro/order/19.12 transition and obtained exactly own03 c41d949d... SHA. New19.12→19.13→19.14 links preserve A/B/C and mature-extension separation; no scores, trained model, data or release pointer changed.'
}
for source in r['sources']:
    if source['id'] in evidence['changed_source_ids']:
        source['backfill03_sha256'] = source['sha256']
        source['backfill03_inspection_note'] = source['inspection_note']
        source['sha256'] = sha(ROOT/source['path'])
        source['version'] = 'Current2026-10-16 worktree bytes, freeze18 order-correction original-owner informed callback01'
        source['inspection_note'] = notes[source['id']]
r['sources'].append({'id':'s_order14','kind':'repository_code','title':'Current chapter14 reading-order context','path':'course/chapters/14.md','sha256':sha(ROOT/'course/chapters/14.md'),'version':'Current2026-10-16 freeze18 worktree','verified':True,'inspection_note':'Personally read current introduction and numbered headings. Actual source section IDs14.1–14.18, current inventory and nav are numeric and agree with updated R.3. This source is used only for route context, not a new technical judgment on chapter14 methods.'})
for artifact in r['artifacts']:
    artifact['description'] = '承接本人backfill03及更早原始證據；本輪核bytes不變，未重跑原執行。原用途：'+artifact['description']
new = [
    {'id':'a_order01','kind':'execution','path':BASE+'order-dependency-results-01.json','sha256':sha(OUT/'order-dependency-results-01.json'),'description':'本人本輪四個變動來源的實讀／結構比較、原whole19重建、312/338/350分母核對及其餘pins身分。','command':evidence['command'],'result':'Only4 expected pins changed.312 unchanged index records;14/19 numeric order agrees between source/index/nav; all non-nav TOML fields unchanged;19.13 bytes identical,19.12 transition-only; rebuilt old whole19 exactly.312 Notebook sync passes;312 chapter lessons+26 other numbered sections=338;350 canonical pages. No necessary README issue.','environment':evidence['environment']},
    {'id':'code_order01','kind':'code','path':BASE+'order-dependency-probe-01.py','sha256':sha(OUT/'order-dependency-probe-01.py'),'description':'本次真正執行的來源／索引／導覽／檔案與Notebook同步檢查原碼；不執行模型。'},
    {'id':'a_order_attempt01','kind':'source_snapshot','path':BASE+'order-probe-attempt-01.json','sha256':sha(OUT/'order-probe-attempt-01.json'),'description':'保留初次過度假定全域index與nav順序相等的失敗探測，後由old/current對照核清C放置差異原已存在。'},
    {'id':'code_order_seal01','kind':'code','path':Path(__file__).relative_to(ROOT).as_posix(),'sha256':sha(Path(__file__)),'description':'本人本次完整44claim callback建構、磁碟重讀direct validator與封存程式。'}
]
r['artifacts'].extend(new)
for claim in r['claims']:
    claim['backfill03_revalidation'] = claim.pop('callback_revalidation')
    claim['callback_revalidation'] = {'round':'order01','current_context':'informed','scope':'Narrow order-dependency applicability; own03 substantive evidence retained.','status':'verified','dependency_relevance':'Actually checked changed index/nav/route/chapter19 source and effect on this claim.' if claim['id'] in {'c02','c03','c11','c13','c26','c36'} else 'Current README source/figure and necessary implementation/primary evidence remain identical; changed dependencies only reorder records/sections or navigation text and do not alter this claim.','execution_provenance':'New work: structural/hash/count checks and build_course --check. No model, score, formaltest, HF download, kernel or browser rerun.'}
    claim['artifact_ids'].append('a_order01')
    if claim['id'] == 'c02':
        claim['verification']['observed'] += ' Order01新核：312個逐課Notebook索引紀錄逐ID內容不變，所有路徑存在，build_course --check實際回報312個小節同步通過。'
    if claim['id'] == 'c03':
        claim['scope'] = '338只指所有編號小節：312個章內逐課Notebook小節，加R4節、W7節、訓練11節、名詞4節，共26；不是338份Notebook。另含未編號頁後canonical頁數350。'
        claim['verification']['observed'] += ' Order01實數：23個章來源共312節；閱讀指南4、暖身7、訓練11、名詞4，合計338；freeze18 canonical共350頁。'
    if claim['id'] == 'c11':
        claim['backfill03_scope'] = claim['scope']
        claim['scope'] = 'backfill03已實測preview17的390px TOC/search/drawer與1280px CSS控制組；本次只核TOML非nav設定不變，以及14/19導覽排序。未重跑UI、palette/copy或所有路徑／裝置；原Escape操作限制保留。'
    if claim['id'] == 'c12':
        claim['backfill03_scope'] = claim['scope']
        claim['scope'] = '原CairoSVG與backfill03實際Chromium圖解／CSS證據保留，本次SVG/CSS/JS相關bytes不變，沒有重渲染或UI、zoom、reduced-motion實機重跑。'
    if claim['id'] == 'c26':
        for ref in claim['evidence']:
            if ref['source_id'] == 's_ch19':
                ref['supports'] = 'Current19.13 retains A/B/C scope and completed B/C backfill; relocation and19.12 transition do not change those facts.'
        claim['evidence'].append({'source_id':'s_order14','locator':'current chapter introduction and14.1–14.18 numbered heading sequence','supports':'Updated R.3 reading order agrees with actual chapter14 section order; no method claim re-review.'})
        claim['scope'] += ' Order01另核第14/19章改成節號順讀；R.2/R.4的任務式19.13選路不等於全章順讀。'
r['backfill03_dependency_audit'] = r.pop('callback_dependency_audit')
r['callback_dependency_audit'] = {'round':'order01','scope':evidence['scope'],'artifact_id':'a_order01','changed_repository_source_ids':evidence['changed_source_ids'],'all_other_source_and_prior_artifact_identities_unchanged':True,'source_freeze_reference':r['source_freeze_reference'],'readme_sha256':r['source_sha256'],'counts_with_denominators':evidence['counts'],'judgment':evidence['judgment'],'probe_observation':evidence['resolved_probe_observation'],'not_rerun':evidence['not_rerun']}
r['backfill03_scope_and_independence'] = r.pop('scope_and_independence')
r['scope_and_independence'] = {'scope':'README original-owner informed narrow order callback; current whole-page verdict with44 substantive claims inherited from own03 and earlier primary checks. Not a new blind review.','role':'reviewer_context=fresh describes initial original task only; current_context=informed.','prior_report_preserved':r['previous_callback'],'source_materials':'Only current assigned README/necessary primary dependency sources, own03/earlier evidence, freeze18 source binding and immutable prior source bytes. No other reviewer report, current reader/continuity score or author problem list read.','new_work':'Personally read four changed source diffs, complete current course README, current19 introduction/full19.12/full19.13, chapter14 introduction/headings and current relevant config. Actually compared all312 index records, nav/source orders, whole19 reconstruction, current counts/paths and all other pinned sources/artifacts; ran build_course --check.','not_rerun':evidence['not_rerun'],'not_human_testing':True}
r['checks']['factual_accuracy']['details'] = '44個README主張原實質證據承接本人03；本輪四個dependency實際查讀及結構對照，只有排序／導覽改動。19.13用途、19.19固定操作、19.20成績均同bytes，19.12承接不改A能力失敗或成熟延伸界線；無必要README修訂。'
r['checks']['numeric_verification']['details'] = '本輪實數分開：312個章內課節＝312份逐課Notebook；另閱讀指南4＋暖身7＋訓練11＋名詞4＝26個編號小節，總計338；含未編號頁共350 canonical pages。312不是338，338不稱Notebook數。原模型分數／權重／raw資料不變且未重算。'
r['checks']['figure_consistency']['details'] = 'README與唯一SVG SHA仍和本人03及freeze18相同；本輪改動未更換圖、圖示數字或方向，沿用本人原渲染／視圖證據，未冒稱新看圖或UI。'
r['checks']['source_verification']['details'] = '本人真正核4個changed pins後更新當前SHA：index全312紀錄逐ID完全相同；TOML只nav排序，非nav設定完全相同；route只R.3；whole19以舊intro/order/19.12導覽句重建，SHA精確符合03，19.13內容搬移不變。其餘原sources與27個prior artifacts逐檔SHA相同，沿用本人原始查證。'
r['checks']['limitations']['details'] = evidence['not_rerun']+' 初probe過度要求庫存清單與閱讀導覽全序一致已保留並核清：C的index末尾/nav插入位置在prior03與現在相同，README只宣稱閱讀安排。這不是用舊pass代替新依賴實讀。'
r['issues'] = []
save(ROOT/TARGET, r)
# Validate the actual saved bytes, not only an in-memory proposed report.
saved = load(ROOT/TARGET)
spec = importlib.util.spec_from_file_location('readme_order01_validator', ROOT/'scripts/check_technical_reviews.py')
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)
errors = validator._validate(ROOT,ROOT/'README.md','readme',(ROOT/'README.md').read_text(),saved,{TASK:['readme']},set())
validation = {'at':datetime.now(UTC).isoformat(),'command':'uv run --frozen --extra cpu python '+Path(__file__).relative_to(ROOT).as_posix(),'report':{'path':TARGET,'sha256':sha(ROOT/TARGET)},'source_sha256':sha(ROOT/'README.md'),'current_context':'informed','scope':'Direct original _validate after disk reread on current whole README/full44claims; narrow order callback only.','errors':errors}
save(OUT/'direct-validation-01.json',validation)
assert not errors, errors
assert sha(ROOT/PRIOR) == r['previous_callback']['sha256']
seal = {'schema_version':1,'sealed_at':datetime.now(UTC).isoformat(),'reviewer_task':TASK,'lesson_id':'readme','role':'technical','current_context':'informed','round_type':'same-owner narrow order-dependency callback01; not fresh blind review','report':{'path':TARGET,'sha256':sha(ROOT/TARGET)},'source':{'path':'README.md','sha256':sha(ROOT/'README.md')},'source_freeze':evidence['freeze'],'verdict':r['verdict'],'claim_count':len(r['claims']),'claims_provenance':'44 original substantive claims retained from own03; newly checked4 dependencies plus counts/Notebook source sync; no model/formaltest/HF rerun.','changed_sources':{s['path']:s['sha256'] for s in r['sources'] if s['id'] in evidence['changed_source_ids']},'previous_callback':r['previous_callback'],'validation':{'path':BASE+'direct-validation-01.json','sha256':sha(OUT/'direct-validation-01.json'),'errors':errors},'new_artifacts':{a['path']:a['sha256'] for a in new},'limits':evidence['not_rerun'],'independence':'No other reader/technical/continuity reports or author known-issue list read.'}
save(OUT/'owner-order-seal-01.json',seal)
print(json.dumps({'report':TARGET,'report_sha256':sha(ROOT/TARGET),'seal_sha256':sha(OUT/'owner-order-seal-01.json'),'validation_sha256':sha(OUT/'direct-validation-01.json'),'verdict':r['verdict'],'claims':len(r['claims']),'errors':errors},ensure_ascii=False))
