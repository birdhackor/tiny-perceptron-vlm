"""Same original owner records actual current reinspection, then runs the single-section checker."""
from pathlib import Path
from datetime import datetime, UTC
import hashlib
import json
import re
import copy
import subprocess
import sys

ROOT=Path('/workspace/tiny-perceptron-vlm')
OUT=ROOT/'docs/technical-reviews/artifacts/phase4-7_4-independent/reinspection-20261006'
REPORT=ROOT/'docs/technical-reviews/7.4.json'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d): p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
comparison=json.loads((OUT/'input-comparison.json').read_text())
priorpath=ROOT/comparison['prior_report']['path']
priorraw=priorpath.read_bytes()
assert sha(priorpath)==comparison['prior_report']['sha256']
assert REPORT.read_bytes()==priorraw
report=json.loads(priorraw)
assert report['reviewer_task']=='/root/phase4_factual_coordinator/factual_7_4'
raw=(ROOT/'course/chapters/07.md').read_bytes()
headers=list(re.finditer(rb'(?m)^## [^\r\n]+',raw))
i=next(i for i,x in enumerate(headers) if x[0].startswith(b'## 7.4 '))
body=raw[headers[i].start():headers[i+1].start()]
assert body==(OUT/'current-section.md').read_bytes()
assert sha(OUT/'current-section.md')==comparison['current_section']['sha256']
assert comparison['changed_substantive_claim_ids']==[]
for item in comparison['current_code_comparison']:
    assert sha(ROOT/item['path'])==item['current_sha256']==item['own_frozen_sha256']
for a in report['artifacts']: assert sha(ROOT/a['path'])==a['sha256']
for p,digest in report['figure_sha256'].items(): assert sha(ROOT/p)==digest
for name in ['desktop','mobile']:
    assert (OUT/(name+'.png')).is_file()
    assert '接著只把答案A改成B' in (OUT/(name+'.body.txt')).read_text()
(OUT/'prior-report.opaque.json').write_bytes(priorraw)

claim_assessments=[
 {'claim_id':'autoregressive_first_answer','changed':False,
  'actual_reinspection':'Current開場與原A5→Y4/EOS6→Y5說明親讀；assistant格猜下一項的語義未改。Current7.3也明說答案與計分標記一起配到前一格。原Transformer p2§3/p3§3.1再次讀其autoregressive/offset/禁止未來內容段；只支持一般causal原理，repo特定角色序列仍由原data.py與已核同SHA現檔支援。',
  'reused_source_ids':['transformer','data','attention','bounded_run'],'reused_artifact_ids':['paper_pdf','paper_text','bounded_result','inspection']},
 {'claim_id':'ids_positions_and_figure','changed':False,
  'actual_reinspection':'親讀current全節與SVG，逐行重新看目前rendered PNG及桌面/手機預覽：位置0..5，X=[1,3,89,2,4,73]，Y前四IGNORE後73,2；索引/輸入ID/目標ID三欄不混淆。新渲染沒有新箭頭或量綱。原stdout兩行再次讀，Q81+8/A65+8及原位置5→4、6→5整數推導仍完全支持現文。',
  'reused_source_ids':['data','arithmetic','original_run'],'reused_artifact_ids':['original_stdout','bounded_result','svg_snapshot']},
 {'claim_id':'fence_api_contract','changed':False,
  'actual_reinspection':'原fence與current fence逐byte相同，SHA03b6d9f185c51ed9d159647bd6b6fdcd2564dda45b3218ccfa1accb5f2c0351c；原nonzero與item官方2.9快照再次讀所用契約。仍為一維非IGNORE索引首格4與一元素item，兩assert檢查輸入assistant4/答案73；没有训练或参数更新主张。原実跑stdout/receipt已重核hash，沿用2026-10-05实跑，不声称本轮重跑。',
  'reused_source_ids':['torch_nonzero','torch_item','original_run'],'reused_artifact_ids':['original_fence','original_stdout','original_execution','original_env','docs_nonzero','docs_item']},
 {'claim_id':'mask_alignment_no_double_shift','changed':False,
  'actual_reinspection':'Current mask段親讀未變：有效数正确不等于位置正确，targets需配前一格，TinyLM/loss不能再shift。data/model/attention/build bootstrap皆与本人冻结源码SHA相同；再次核原same-shape/same-count2反例：正确first4，错误targets[:-1]first5读A猜A；原synthetic logits梯度只4、5。原官方cross_entropy快照再次读输入/target和ignore契约；仅本课API约定，未扩展为所有外部模型通用约定。',
  'reused_source_ids':['data','model','torch_cross_entropy','bounded_run'],'reused_artifact_ids':['bounded_code','bounded_result','bounded_receipt','docs_cross_entropy']},
 {'claim_id':'eos_mask_and_valid_denominator','changed':False,
  'actual_reinspection':'Current EOS/ignored输入内容親讀未變；现图仍是4 assistant→A与5 A→EOS。原bounded JSON本轮再次读：有效分母2、sum2ln264与meanln264，单位nats及nats/token、容差1e-12只用于辅助数值契约，不是课程模型测量。原causal检查X4可见0..4/不可见5，Q→R影响assistant输出而A→B不影响0..4，仍支持被忽略前文可读，未声称任意读未来或模型学会答案。',
  'reused_source_ids':['torch_cross_entropy','data','model','attention','bounded_run'],'reused_artifact_ids':['bounded_result','bounded_receipt','docs_cross_entropy','inspection']},
 {'claim_id':'qq_b_exercise','changed':False,
  'actual_reinspection':'本节唯一改动是练习段「接着」→「接著」。QQ/A及QQ/B的内容、先后顺序、位置数字及ID均未改；本人重新读current段与原bounded JSON：QQ/A first5 X4 Y73，QQ/B first5 X4 Y74，B66+8=74仍精确，无近似/容差改变或新的学习成效主张。',
  'reused_source_ids':['data','bounded_run','arithmetic'],'reused_artifact_ids':['bounded_code','bounded_result','inspection']},
]
receipt={
 'kind':'actual_same_owner_technical_reinspection','reviewer_task':report['reviewer_task'],
 'completed_reading_at_utc':datetime.now(UTC).isoformat(),
 'owner_identity':'Same original independent technical owner; followup reinspection, not a newly claimed fresh reviewer.',
 'prior_report':comparison['prior_report'],'current_source':'course/chapters/07.md#7.4',
 'current_source_sha256':comparison['current_section']['sha256'],
 'current_source_snapshot':comparison['current_section']['path'],
 'own_original_frozen_source_sha256':comparison['own_prior_section']['sha256'],
 'exact_raw_difference':comparison['exact_change'],
 'changed_substantive_claim_ids':[],
 'actual_read_scope':['Current 7.4 entire original UTF-8 section including its fence, exercise and details',
   'Current necessary preceding 7.3 entire section; target-mask/front-cell explanation is consistent with 7.4',
   'Own original frozen 7.4, own prior canonical report and own underlying source and execution evidence',
   'Current SVG text and own frozen SVG fingerprint; new Inkscape PNG personally viewed with view_image',
   'Current provided 7.4 preview actually rendered by Chromium/Playwright at1280x800 and390x844; both full-page screenshots personally viewed',
   'Original paper snapshot p2§3/p3§3.1 and original official nonzero/item/cross_entropy API snapshots, limited to the already relevant support',
   'All38 own prior registered artifacts rehashed; relevant current data/model/attention/modern/build/checker code fingerprints matched',
   'Current factual-reviewer instructions read; current pyproject diff is only Ruff extend-exclude docs/course-revision-20261005, no runtime dependency change'],
 'visual_observations':'Current six-row diagram matches numeric position/ID claims on raw PNG and both preview widths. Green positions4/5 mark supervised A/EOS; first4 blue-gray target rows are ignored yet causal-prefix inputs remain available. Browser image loads complete at640x750 natural size; diagram labels readable, no new technical inconsistency. Mobile code/output blocks use narrow horizontally scrollable presentation; body has no horizontal overflow. This is technical consistency reinspection, not a new blind reader session.',
 'claim_assessments':claim_assessments,
 'explicit_evidence_reuse':'All six material claims and their support scopes are unchanged. Original fence and bounded CPU results remain 2026-10-05 executions, personally re-read/rehashed now. Original official API pages remain PyTorch2.9 source access2026-10-05; original execution/runtime2.14.1+cpu separately recorded. No new PDF/source acquisition needed.',
 'new_model_or_fence_execution':False,
 'why_no_new_cpu_probe':'Only spelling changed; original fence, computational code, bootstrap, SVG, primary source snapshots and numerical execution outputs have identical SHA-256. Existing evidence supports precisely the same original/QQ/B, mask, denominator and causal claims. Re-running computation would not examine a changed claim.',
 'new_actual_actions':['Full firsthand current section/context reading','Own frozen-input raw diff','Source/code/evidence fingerprint and claim-support comparison','New SVG rendering and personal view','New desktop/mobile preview rendering and personal views','Opaque own prior-report preservation','Owner canonical update and separate actual single-section checker after this receipt'],
 'excluded_actions':['No other reviewers reports or author repair explanations read','No GPU/train/checkpoint eval/dataset prep/model-data downloads','No entire pipeline rerun or unrelated paper search','No course text/SVG edits or commit'],
 'unresolved_questions':[], 'verdict':'pass',
 'fingerprint_evidence_path':str((OUT/'input-comparison.json').relative_to(ROOT)),
 'fingerprint_evidence_sha256':sha(OUT/'input-comparison.json'),
 'preview_render_receipt_path':str((OUT/'preview-render-receipt.json').relative_to(ROOT)),
 'preview_render_receipt_sha256':sha(OUT/'preview-render-receipt.json')}
write(OUT/'actual-reinspection-receipt.json',receipt)

report['source_sha256']=comparison['current_section']['sha256']
report['read_scope']='Original technical owner actual reinspection: current7.4 whole section/fence/SVG, current necessary7.3, own frozen source diff, own original primary sources and execution results, current code/SVG/evidence hashes, new diagram and desktop/mobile preview renders personally viewed. Own prior report preserved; no other reports or author repair explanations read.'
report['reinspection']={'owner_task':report['reviewer_task'],'context':'same_original_technical_owner_followup',
 'performed_on':'2026-10-06','prior_report':comparison['prior_report'],'changed_substantive_claim_ids':[],
 'receipt_artifact_id':'same_owner_reinspection_20261006','receipt_path':str((OUT/'actual-reinspection-receipt.json').relative_to(ROOT)),
 'receipt_sha256':sha(OUT/'actual-reinspection-receipt.json'),'model_or_fence_rerun':False,
 'evidence_reuse':'Original unchanged2026-10-05 CPU and primary source evidence personally re-read and fingerprints rechecked; preserved historical execution dates/results.'}
for a in report['artifacts']:
    if a['id'] in ['section','extract','inspection']:
        a['description']='Own original2026-10-05 frozen evidence, preserved unchanged; current source and reinspection are separately recorded. '+a['description']
def artifact(id,name,kind,description):
    p=OUT/name
    report['artifacts'].append({'id':id,'path':str(p.relative_to(ROOT)),'sha256':sha(p),'kind':kind,'description':description})
artifact('same_owner_reinspection_20261006','actual-reinspection-receipt.json','derivation','原技术owner本人current全文/图/必要前文重读、逐claim支持复查、明确沿用原自身实跑、new actual visual views和结论；非只更新source SHA。')
artifact('current_section_20261006','current-section.md','source_snapshot','本次完整current7.4原始UTF8 bytes，canonical source_sha对应此副本，未替换旧冻结输入。')
artifact('current_context_7_3_20261006','current-context-7.3.md','source_snapshot','本次本人亲读的当前必要7.3全文；监督标记和答案配前一预测格与7.4一致。')
artifact('current_source_diff_20261006','current-vs-own-frozen.diff','derivation','本人对自身冻结输入算得精确差异，仅接着→接著。')
artifact('current_inputs_compared_20261006','input-comparison.json','derivation','相关现码/原证据38项/SVG/fence指纹核查、原history SHA及环境；计算代码未改。')
artifact('current_svg_20261006','current-figure.svg','source_snapshot','目前被引用SVG原始bytes副本，与初审指纹同一版。')
artifact('current_figure_render_20261006','current-figure.png','figure_render','原owner本轮新Inkscape render并view_image实际查看；六列表位置与输入/目标ID核对。')
artifact('current_desktop_preview_20261006','desktop.png','figure_render','Current预览页面1280x800 full-page screenshot，本人以original detail实际查看。')
artifact('current_mobile_preview_20261006','mobile.png','figure_render','Current预览页面390x844 full-page screenshot，本人以original detail实际查看。')
artifact('current_preview_render_receipt_20261006','preview-render-receipt.json','source_snapshot','本轮Chromium/Playwright实际HTTP200、尺寸、图加载、viewport以及HTML/PNG指纹。')
artifact('prior_owner_report_opaque_20261006','prior-report.opaque.json','source_snapshot','自己的前canonical report完整opaque原bytes，另有history同SHA；不覆写或伪造历史。')
for name in ['prepare_reinspection.py','render_preview.py','update_review.py']:
    artifact('reinspection_code_'+name.replace('.','_'),name,'code','本轮真实reinspection比较/渲染/报告保存与后续单节checker原码。')
for name in ['current-factual-reviewer-instructions.md','current-pyproject.toml','pyproject-vs-own-frozen.diff','desktop.html','mobile.html','desktop.body.txt','mobile.body.txt','prepare.stdout.txt','prepare.stderr.txt','preview.stdout.txt','preview.stderr.txt','figure-render.stdout.txt','figure-render.stderr.txt']:
    artifact('reinspection_snapshot_'+name.replace('.','_'),name,'source_snapshot','本轮当前方法/精确输入或实际渲染命令输出，保留本次fingerprint和查核范围。')

report['sources'].append({'id':'same_owner_current_reinspection','kind':'derivation','title':'Same original technical owner current7.4 support-scope comparison','verified':True,
 'details':'Personally reread current7.4 and necessary7.3, compare own original frozen input: exact only spelling接着→接著; all six original substantial claims/support unchanged. Rehash all38 own evidence plus computational code/SVG/fence and read relevant original source paragraphs/results; actual new diagram/desktop/mobile views. See same_owner_reinspection_20261006 artifact with per-claim records and explicit original historical-execution reuse.'})
for s in report['sources']:
    if s['kind'] in ['paper','official_docs','repository_code','execution']:
        s['same_owner_reinspection_note']='2026-10-06: unchanged original source or computational/CPU execution evidence personally re-read/rehashed; supports same current claim scope. Original source version, access date and original execution date retained; no new model run or source download claimed.'
for claim in report['claims']:
    claim['artifact_ids'].append('same_owner_reinspection_20261006')
    claim['evidence'].append({'source_id':'same_owner_current_reinspection','locator':'actual-reinspection-receipt.json claim_assessments/'+claim['id'],
                              'supports':'This owner personally checked the current version against own frozen input and the unchanged claim-specific source/CPU evidence; spelling edit introduces no changed substantive claim or extended support scope.'})
    claim['current_reinspection_status']='verified_by_same_original_owner'
    if 'verification' in claim:
        claim['verification']['details']+=' 2026-10-06 same-owner reinspection: original2026-10-05 execution result personally re-read and hash-verified; code/fence/claim unchanged, not rerun or reported as a fresh measurement.'
    if claim['id'] in ['ids_positions_and_figure','eos_mask_and_valid_denominator']:
        claim['artifact_ids'] += ['current_figure_render_20261006','current_desktop_preview_20261006','current_mobile_preview_20261006']
for name,item in report['checks'].items():
    item['details']+=' Current same-owner reinspection2026-10-06: exact spelling-only edit接着→接著, six substantive claim scopes unchanged; original relevant evidence hash-verified and explicitly reused. Actual full current-section/context reading and new diagram/desktop/mobile render+views recorded in same_owner_reinspection_20261006.'
report['verdict']='pass'
assert not report['issues']
write(REPORT,report)
command=[str(ROOT/'.venv/bin/python'),'scripts/check_technical_reviews.py','--lesson','7.4']
checked=subprocess.run(command,cwd=ROOT,capture_output=True,check=False)
(OUT/'checker.stdout.txt').write_bytes(checked.stdout)
(OUT/'checker.stderr.txt').write_bytes(checked.stderr)
checker_receipt={'kind':'actual_single_section_checker_after_same_owner_reinspection','completed_at_utc':datetime.now(UTC).isoformat(),
 'command_argv':command,'cwd':str(ROOT),'actual_exit_code':checked.returncode,
 'stdout':checked.stdout.decode(),'stderr':checked.stderr.decode(),'checker_sha256':sha(ROOT/'scripts/check_technical_reviews.py'),
 'canonical_report_path':str(REPORT.relative_to(ROOT)),'canonical_report_sha256':sha(REPORT),
 'prior_history_path':str(priorpath.relative_to(ROOT)),'prior_history_sha256':sha(priorpath),
 'source_sha256':report['source_sha256'],'receipt_artifact_id':'same_owner_reinspection_20261006',
 'actual_reinspection_receipt_path':str((OUT/'actual-reinspection-receipt.json').relative_to(ROOT)),
 'actual_reinspection_receipt_sha256':sha(OUT/'actual-reinspection-receipt.json'),
 'actual_scope':'7.4 current whole section and SVG plus necessary current7.3; own-original evidence and exact diff support review; new desktop/mobile/diagram views; no new model run.',
 'unresolved_questions':[], 'verdict':report['verdict']}
write(OUT/'checker-receipt.json',checker_receipt)
manifest={'canonical_report_sha256':sha(REPORT),'prior_history_sha256':sha(priorpath),
 'artifact_files':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p),'bytes':p.stat().st_size}
                   for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='hash-manifest.json']}
write(OUT/'hash-manifest.json',manifest)
print(json.dumps(checker_receipt,ensure_ascii=False,indent=2))
sys.exit(checked.returncode)
