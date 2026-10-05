"""This reviewer's own support callback after personally reading 12.8 and 12.9."""
import hashlib
import importlib.util
import json
import pathlib

BASE=pathlib.Path(__file__).resolve().parent
ROOT=BASE.parents[4]
OWNER='/root/phase4_factual_coordinator/factual_12_8'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def rel(path):return path.relative_to(ROOT).as_posix()
report_path=ROOT/'docs/technical-reviews/12.8.json'
report=json.loads(report_path.read_bytes())
assert report['reviewer_task']==OWNER
before_sha=sha(report_path)
inputs=json.loads((BASE/'input-receipt.json').read_bytes())
assert before_sha==inputs['initial_pass']['sha256']

spec=importlib.util.spec_from_file_location('sf',ROOT/'docs/review-tools/section_facts.py')
sf=importlib.util.module_from_spec(spec);spec.loader.exec_module(sf)
old_background,_,old_line=sf.original_section(BASE.parent/'inputs/course/chapters/12.md','12.9')
(BASE/'12.9.initial-frozen.md').write_bytes(old_background)
current_background=(BASE/'12.9.current.md').read_bytes()
current_own=(BASE/'12.8.current.md').read_bytes()
assert sha(BASE/'12.8.current.md')==report['source_sha256']
assert b'frequency' not in current_own  # Only Chinese task prose and actual import/arguments in this section.
fences=sf.fences(current_own,inputs['12.8']['first_line'])
assert len(fences)==1
assert hashlib.sha256(fences[0]['raw']).hexdigest()=='5d76f94f04c33e1867dc4881c64ecfcc2c59e51294ba0de5f6c9ea27420f4ad1'
unchanged={}
for path in ['tiny_perceptron/multimodal.py','scripts/course_experiments/modalities.py','docs/course-experiments/results/encoders.json','course/figures/rewrite-12-frame-features.svg']:
 snapshot=BASE.parent/'inputs'/path
 digest=sha(ROOT/path)
 assert digest==sha(snapshot)
 unchanged[path]=dict(current_sha256=digest,initial_snapshot=rel(snapshot),unchanged=True)

claim_assessments=[
 dict(claim_id='batch_axis',depends_on_changed_12_9_assertion=False,decision='retained',supports='12.8原fence、tone/AudioEncoder、官方None indexing與本人CPU實跑支持；不引用12.9錯例解讀。'),
 dict(claim_id='logmel_frames',depends_on_changed_12_9_assertion=False,decision='retained',supports='0.1秒/1600samples/16帶/11框由12.8與不變helper、STFT官方公式及本人原數值支持。'),
 dict(claim_id='internal_pipeline',depends_on_changed_12_9_assertion=False,decision='retained',supports='完整AudioEncoder契約與本人逐層CPU檢查、SVG支持；12.9仍使用11框特徵接頭，背景沒有否定此形狀。'),
 dict(claim_id='time_positions',depends_on_changed_12_9_assertion=False,decision='retained',supports='共享逐框層與平均的差別來自12.8實作/官方API/本人反轉變體；不以12.9錯例歸因推論時間辨識能力。'),
 dict(claim_id='untrained_scope',depends_on_changed_12_9_assertion=False,decision='retained',supports='12.8仍為隨機初值forward、不更新參數；12.9新全文仍明記backward沒有optimizer step。這裡只核銜接與範圍，不重新驗12.9模型能力。'),
 dict(claim_id='width_bands',depends_on_changed_12_9_assertion=False,decision='retained',supports='只依12.8練習、固定實作及本人width12/bands32變體；與12.9歷史錯例段無依賴。'),
 dict(claim_id='separate_tasks',depends_on_changed_12_9_assertion=False,decision='retained',supports='單音threshold與真人需求任務區別由raw encoders規約及PolyAI原卡支持；12.9仍先以440Hz/300Hz規則示範梯度，未成中文能力驗收。'),
 dict(claim_id='historical_link',depends_on_changed_12_9_assertion=False,decision='retained',supports='12.8連結encoders.json，原raw pointers和方法未改；12.9修改audio.json結果解讀不是本claim的測量資料。')
]
assert {x['claim_id'] for x in claim_assessments}=={x['id'] for x in report['claims']}
receipt=dict(schema_version=1,kind='own_section_support_callback',reviewer_task=OWNER,
 accessed_on='2026-10-05',lesson_id='12.8',verdict='pass',
 initial_pass_opaque_backup=inputs['initial_pass'],
 current_own_section=inputs['12.8'],
 background_before=dict(source='course/chapters/12.md#12.9',section_sha256=hashlib.sha256(old_background).hexdigest(),snapshot=rel(BASE/'12.9.initial-frozen.md'),first_line=old_line,meaning='Extracted from initial personally-read frozen chapter; not current source.'),
 background_now=inputs['12.9'],
 actual_read_scope=dict(own='All current 12.8 bytes, including figure reference/fence/details.',background='All new 12.9 bytes, including fence, result paragraph and details.',
   own_locator=f"course/chapters/12.md lines {inputs['12.8']['first_line']}–{inputs['12.8']['first_line']+len(current_own.splitlines())-1}",
   background_locator=f"course/chapters/12.md lines {inputs['12.9']['first_line']}–{inputs['12.9']['first_line']+len(current_background.splitlines())-1}"),
 personally_observed_change=dict(
  before='既有單音訓練另測14題，生成答案對11/14，錯誤集中在邊界與時長變化。',
  after='既有單音訓練另測14題，生成答案對11/14；三個錯例的頻率都是290或300Hz。資料把振幅與時長一起改動，無法分辨兩者各自的影響。',
  assessment='Changed historical error attribution does not support any of my 12.8 claims. I do not carry either attribution into 12.8, and do not independently review the truth of all 12.9 empirical claims in this callback.'),
 claim_assessments=claim_assessments,
 transition_assessment='12.8下一節先檢查接到文字核心後的梯度通路；新版12.9仍包含這個示範及沒有optimizer step的明確範圍，所以教材銜接相符。',
 unchanged_dependencies=unchanged,
 retained_evidence='Initial CPU executions, official original-source inspection and personally viewed SVG are retained because section/fence/code/JSON/SVG bytes are unchanged. No unnecessary CPU/GPU/model work was rerun.',
 frozen_input_policy=dict(initial_frozen_complete_chapter=report['read_scope']['frozen_complete_chapter'],callback_frozen_complete_chapter=inputs['new_whole_frozen_input']),
 independence='No 12.9 technical report, other reviewer verdict, author extra notes or answers were read. This is my own claim-by-claim dependency assessment.',
 limitations=report['limitations'])
(BASE/'support-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')

def add_artifact(identifier,path,kind,description):
 report['artifacts'].append(dict(id=identifier,path=rel(path),sha256=sha(path),kind=kind,description=description))
add_artifact('support_callback_12_9',BASE/'support-receipt.json','derivation','本人親讀新12.9及完整12.8後的逐claim依賴/支持核對；保留初PASS與真frozen input版本。')
add_artifact('support_callback_own_source',BASE/'12.8.current.md','source_snapshot','本次callback親讀的完整12.8，原sourceSHA未變。')
add_artifact('support_callback_new_background',BASE/'12.9.current.md','source_snapshot','本次callback親讀新版完整12.9，SHA0930ed...；不引用其technical report。')
add_artifact('support_callback_old_background',BASE/'12.9.initial-frozen.md','source_snapshot','最初本人親讀背景的原bytes，從當次frozen whole-input取得，不冒稱current。')
add_artifact('support_callback_frozen_chapter',BASE/'chapter.current-frozen.md','source_snapshot','本次callback新完整章節frozen input；原初讀frozen快照與SHA仍保留。')
add_artifact('support_callback_inputs',BASE/'input-receipt.json','source_snapshot','初PASS opaque備份的真SHA/path，親讀新版兩小節locator/hash與新whole-input定位。')
add_artifact('support_callback_code',BASE/'callback_review.py','code','本人callback版本/hash核對和完整重寫report的實際程式；沒有重跑模型/訓練。')
for claim in report['claims']:
 assert claim['status']=='verified'
 claim['artifact_ids'].append('support_callback_12_9')
report.setdefault('recheck_history',[]).append(dict(rechecked_on='2026-10-05',reviewer_task=OWNER,trigger='Previously personally-read background12.9 changed; own12.8 unchanged.',before_report_sha256=before_sha,opaque_prior_report=inputs['initial_pass'],background_source='course/chapters/12.md#12.9',background_source_sha256=inputs['12.9']['section_sha256'],artifact_id='support_callback_12_9',result='pass; all eight own claims personally reassessed and retained; no claim depends on changed error attribution.'))
report['read_scope']['support_callbacks']=[dict(artifact_id='support_callback_12_9',actual_read_scope=receipt['actual_read_scope'],background_source_sha256=inputs['12.9']['section_sha256'],callback_frozen_complete_chapter=inputs['new_whole_frozen_input'])]
report['last_rechecked_on']='2026-10-05'
report['issues'].append(dict(id='changed_background_12_9',status='resolved',original_question='親讀過的12.9背景歷史錯例解讀改動，需本人確認12.8支持範圍。',resolution='本人親讀新版12.9全部與12.8全部，逐八claim確認不依赖改動的歷史錯因歸因；下一節梯度示範銜接仍相符。原證據bytes未改，保留初PASS opaque與frozen input，formal receipt為support_callback_12_9。'))
report['checks']['factual_accuracy']['details']+=' 本人callback親讀新版12.9全部與12.8全部，逐八claim依賴核對仍一致；未承接12.9舊/新錯因歸因。'
report['checks']['limitations']['details']+=' 新12.9只作本節support callback，不將其全部實測主張當作已驗；本人先前verified未改證據不重跑。'
report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(reviewer_task=OWNER,verdict=report['verdict'],report_path=rel(report_path),report_sha256=sha(report_path),formal_artifact_id='support_callback_12_9',receipt_path=rel(BASE/'support-receipt.json'),receipt_sha256=sha(BASE/'support-receipt.json'),initial_pass_opaque_backup=inputs['initial_pass']),ensure_ascii=False,indent=2))
