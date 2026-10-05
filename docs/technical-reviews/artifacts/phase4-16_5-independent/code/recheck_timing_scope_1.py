"""Actual reviewer follow-up: saved current section, original timing fields and AST."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parents[1]
REL = OUT.relative_to(ROOT).as_posix()
TASK = '/root/phase4_factual_coordinator/factual_16_5'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
spec = importlib.util.spec_from_file_location('facts', ROOT/'docs/review-tools/section_facts.py')
facts = importlib.util.module_from_spec(spec); spec.loader.exec_module(facts)
body, whole, first_line = facts.original_section(ROOT/'course/chapters/16.md', '16.5')
assert body == (OUT/'input/section-recheck-1.md').read_bytes()
assert hashlib.sha256(body).hexdigest() == 'f2c178ae8ecf5a7327a5678e48edc424ad898812e4ef74888ca3c4adb329cc76'
(OUT/'input/chapter-16-recheck-1.md').write_bytes(whole)
oldbody = (OUT/'input/section.md').read_bytes()
oldwhole = (OUT/'input/chapter-16-frozen.md').read_bytes()
assert oldwhole.count(oldbody) == 1
assert whole == oldwhole.replace(oldbody, body, 1)
oldfence = facts.fences(oldbody, first_line)[0]['raw']
newfence = facts.fences(body, first_line)[0]['raw']
assert oldfence == newfence == (OUT/'input/fence-1.py').read_bytes()
figure = ROOT/'course/figures/rewrite-16-packed-mask.svg'
assert figure.read_bytes() == (OUT/'input/rewrite-16-packed-mask.svg').read_bytes()
original_report = json.loads((OUT/'initial-review.json').read_text())
assert original_report['reviewer_task'] == TASK and original_report['verdict'] == 'revise'
for s in original_report['sources']:
    if s['kind'] == 'repository_code':
        assert sha(ROOT/s['path']) == s['sha256']
for a in original_report['artifacts']:
    assert sha(ROOT/a['path']) == a['sha256']
for lesson, path in [('3.6','course/chapters/03.md'),('3.7','course/chapters/03.md'),
                     ('7.3','course/chapters/07.md'),('7.7','course/chapters/07.md'),
                     ('5.10','course/chapters/05.md')]:
    current, _, _ = facts.original_section(ROOT/path, lesson)
    assert current == (OUT/'input/dependencies'/(lesson+'.md')).read_bytes()

# Read only the original timing measurements and necessary provenance.
raw_path=OUT/'input/efficiency-original.json'
raw=json.loads(raw_path.read_text())
method_path=OUT/'code/architecture-at-raw-revision.py'
assert sha(method_path) == raw['code_sha256']['scripts/course_experiments/architecture.py']
tree=ast.parse(method_path.read_text())
method=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_packing_updates')
sync=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_sync')
median=next(n for n in ast.walk(method) if isinstance(n,ast.Call)
            and isinstance(n.func,ast.Attribute) and n.func.attr=='median')
warm_slice=median.args[0].values[0]
assert isinstance(warm_slice,ast.Subscript) and warm_slice.value.id=='latencies'
assert warm_slice.slice.lower.value==3 and warm_slice.slice.upper is None
assert raw['device']=='cuda' and raw['gpu']=='NVIDIA L4'
entries=[]
for name in ('padded','packed'):
    u=raw['results']['packing']['actual_updates'][name]
    assert u['steps']==u['optimizer_updates']==40
    assert u['effective_tokens']==845
    latency_samples=u['steps']-warm_slice.slice.lower.value
    assert latency_samples==37
    ms=u['warm_step_median_seconds']*1000
    assert round(ms,3)=={'padded':10.799,'packed':11.191}[name]
    entries.append({'path':name,'completed_updates':40,'effective_targets':845,
                    'excluded_latency_samples':3,'median_latency_samples':37,
                    'stored_warm_step_median_seconds':u['warm_step_median_seconds'],
                    'converted_median_ms':ms,'printed_ms':round(ms,3)})
quote='略去各路徑前3次暖身後，餘37次模型更新的計時中位數，PAD為10.799毫秒、packing為11.191毫秒，這輪沒有加速。每次計時包括模型前向計算、代價計算、反向計算、梯度裁剪、參數更新及末尾的GPU同步；抽樣、裝填與清空梯度不計入。'
assert quote in body.decode()
receipt={
 'id':'recheck_timing_scope_1','reviewer_task':TASK,'performed_on':'2026-10-05',
 'source':'course/chapters/16.md#16.5','source_sha256':hashlib.sha256(body).hexdigest(),
 'actual_section_snapshot':REL+'/input/section-recheck-1.md',
 'actual_full_chapter_snapshot':{'path':REL+'/input/chapter-16-recheck-1.md','sha256':hashlib.sha256(whole).hexdigest(),
   'meaning':'Full chapter bytes at this actual recheck; original frozen full-chapter bytes and hash remain unchanged.'},
 'reading_scope':'本人重新讀新版完整16.5所有正文與選讀段；必要前文3.6,3.7,7.3,7.7,5.10與原fence未變。非16.1 reviewer，未審導言。',
 'changed_text_personally_read':quote,
 'scientific_source_inspection':[
  {'source_path':REL+'/code/architecture-at-raw-revision.py','sha256':sha(method_path),
   'read_locators':['_sync lines40-44','_packing_updates lines731-751','_packing_updates summary lines773-787'],
   'own_findings':'selected/_packing_batch/zero_grad occur before began=perf_counter; forward, loss/count, backward, clip, optimizer.step, final CUDA sync occur before latencies.append. median(latencies[3:]) discards only timing samples, not the first3 parameter updates.'},
  {'source_path':REL+'/input/efficiency-original.json','sha256':sha(raw_path),
   'pointers':['/device','/gpu','/code_sha256/scripts~1course_experiments~1architecture.py',
    '/results/packing/actual_updates/{padded,packed}/{steps,optimizer_updates,effective_tokens,warm_step_median_seconds}'],
   'own_findings':entries}],
 'figure_reinspection':{'source_path':figure.relative_to(ROOT).as_posix(),'sha256':sha(figure),
   'render_snapshot':REL+'/execution/packed-mask-inkscape.png','render_sha256':sha(OUT/'execution/packed-mask-inkscape.png'),
   'actual_view':True,'tool':'view_image','finding':'本人再看同一實render圖；Query列、Key欄及A/B block下三角0/1與新版原fence吻合；原SVG bytes未變，未假稱重新渲染或整頁browser檢查。'},
 'other_claims_personally_reassessed':{
  'packing_independence':'全文重讀，PAD計算/隔離/EOS機制未變；原權威定位與有界EOS反例仍支持。',
  'toy_padding':'8總格/4内容/4PAD的算例與分母未變。',
  'mask_api':'原fence bytes、helper bytes、四列和exercise未變；本人再對原圖。',
  'boundary_targets_positions':'targets/position/segment段未變；原helper契約與零更新CPU forward/backward證據仍支持。',
  'logits_gradients':'定義未變；官方CrossEntropyLoss/autograd原定位仍支持，未當作AdamW步長完整公式。',
  'raw_probe':'7+7、14、scalar及scope未變；原測量與dataset原件SHA未變，無GPU重測。',
  'raw_updates':'40更新含前3次及845目標未變；原seed抽樣分母與weight scalar證據仍支持。',
  'timing_scope':'新版37樣本、10.799/11.191ms及每個included/excluded步驟逐項吻合原method；只支持本次warm model update計時比較。',
  'sft_objective':'問答role loss与assistant-text續寫區分未變，原方法與短render_chat執行仍支持。',
  'heldout_quality':'原15題/三model paths的sample exact重算與circle→low未變，未重新生成成績。'},
 'history':{'original_issue_id':'timing_scope_missing','original_report_path':REL+'/initial-review.json',
  'original_report_sha256':sha(OUT/'initial-review.json'),'original_source_sha256':hashlib.sha256(oldbody).hexdigest(),
  'resolved_after_actual_recheck':True},
 'remaining_limits':'沒有原40次完整latency vector，不重算全40次median；現稿已精確說37次warm update區間。未GPU重訓、未weights下載或模型重評；圖核查為直接SVG render而非響應式網頁。',
 'verdict':'pass','unresolved_substantive_issues':[]}
(OUT/'recheck-timing-scope-1.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print('RECHECK_SOURCE_SHA256',receipt['source_sha256'])
print('RECHECK_MEASUREMENTS',json.dumps(entries,sort_keys=True))
print('OTHER_CLAIMS_REASSESSED',','.join(receipt['other_claims_personally_reassessed']))
print('RECEIPT_ID',receipt['id'])
print('RECEIPT_SHA256',sha(OUT/'recheck-timing-scope-1.json'))
print('ACTUAL_RECHECK_PASSED')
