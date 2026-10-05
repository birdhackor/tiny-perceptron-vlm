import difflib,hashlib,importlib.metadata,importlib.util,json,platform,sys
from pathlib import Path

ROOT=Path('/workspace/tiny-perceptron-vlm')
BASE=ROOT/'docs/technical-reviews/artifacts/phase4-16_8-independent-fresh'
OUT=BASE/'reinspection-20261005-current'
OWNER='/root/phase4_factual_coordinator/factual_16_8'
EXPECTED='011a997d566251699645c41441b3a68c502d911214578ec9bfcab72e53a92dea'
digest=lambda raw:hashlib.sha256(raw).hexdigest()
rel=lambda path:path.relative_to(ROOT).as_posix()
prior_raw=(OUT/'prior-report.opaque.json').read_bytes()
prior=json.loads(prior_raw)
assert prior['reviewer_task']==OWNER
spec=importlib.util.spec_from_file_location('facts',ROOT/'docs/review-tools/section_facts.py')
facts=importlib.util.module_from_spec(spec);spec.loader.exec_module(facts)
current,whole,line=facts.original_section(ROOT/'course/chapters/16.md','16.8')
assert digest(current)==EXPECTED
assert current==(OUT/'current-section.md').read_bytes()
old=(BASE/'inputs/section.md').read_bytes()
assert digest(old)==prior['source_sha256']
assert old.replace('兩条路'.encode(),'兩條路'.encode(),1)==current
fence=facts.fences(current,line)
assert len(fence)==1 and digest(fence[0]['raw'])=='9e4f9ce917aea61f81457b5a6c96458f52cff7d9bc271c2724f7a89ef21f1be5'
retained=[]
for artifact in prior['artifacts']:
 observed=digest((ROOT/artifact['path']).read_bytes())
 assert observed==artifact['sha256'],artifact['id']
 retained.append({'id':artifact['id'],'path':artifact['path'],'sha256':observed,'fingerprint_unchanged':True})
current_contracts=[]
for path,snapshot in [
 ('course/figures/rewrite-16-causal-mask.svg','inputs/course__figures__rewrite-16-causal-mask.svg'),
 ('docs/course-experiments/results/efficiency.json','inputs/efficiency.json'),
 ('docs/course-experiments/results/flash_probe.json','inputs/flash_probe.json'),
 ('tiny_perceptron/attention.py','inputs/efficiency__tiny_perceptron__attention.py'),
 ('tiny_perceptron/data.py','inputs/efficiency__tiny_perceptron__data.py'),
 ('tiny_perceptron/model.py','inputs/efficiency__tiny_perceptron__model.py'),
 ('scripts/check_technical_reviews.py','inputs/scripts__check_technical_reviews.py'),
 ('docs/review-tools/section_facts.py','inputs/docs__review-tools__section_facts.py')]:
 raw=(ROOT/path).read_bytes();assert raw==(BASE/snapshot).read_bytes()
 current_contracts.append({'current_path':path,'retained_own_snapshot':rel(BASE/snapshot),'sha256':digest(raw),'unchanged':True})
context=json.loads((OUT/'current-context-16.4-scope.json').read_bytes())
ctx,_,_=facts.original_section(ROOT/'course/chapters/16.md','16.4')
assert digest(ctx)==context['whole_section_sha256']
assert digest((OUT/'current-context-16.4-selected.md').read_bytes())==context['selected_slice_sha256']
environment={'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'torch_installed':importlib.metadata.version('torch'),'device':'CPU fingerprint/content audit; no tensor kernels/model/training executed'}
receipt_id='a_reinspection_20261005_current'
receipt={
 'schema_version':1,'kind':'same_owner_substantive_reinspection_receipt','reviewer_task':OWNER,
 'canonical_artifact_id':receipt_id,'canonical_artifact_path':rel(OUT/'receipt.json'),
 'source':'course/chapters/16.md#16.8','source_sha256':EXPECTED,'source_first_line':line,
 'current_whole_frozen':{'path':rel(OUT/'current-whole-frozen.md'),'sha256':digest(whole),'meaning':'reinspection-time full raw snapshot, not lesson identity'},
 'intro':{'applicable':False,'reason':'16.8不是章節第一節；本次沒有導言依賴，未讀章導言也不捏造intro SHA。'},
 'figure_sha256':prior['figure_sha256'],
 'opaque_history':{'prior_report_path':rel(OUT/'prior-report.opaque.json'),'prior_report_sha256':digest(prior_raw),'prior_source_sha256':prior['source_sha256'],'prior_issue_history':'完整保存於own prior-report.opaque.json /issues（原為[]）；未抹去或覆寫前報告。','prior_proof_manifest':{'path':rel(OUT/'prior-artifact-manifest.opaque.json'),'sha256':digest((OUT/'prior-artifact-manifest.opaque.json').read_bytes())}},
 'actual_read_scope':{'current_lesson':'本人親讀完整現稿，包括開場/QKV軸/圖說/fence後解釋/練習及完整選讀，非只patch fingerprint。','current_context':context,'current_method':'完整讀最新factual-reviewer-instructions.md，保存該版bytes。','current_external_reread':'自己的官方functional.py固定commit：6394–6398 GQA共享repeat，以及6473–6479 backend/整除限制；沒有重fetch papers或官方全頁。','other_prior_context':'先前3.4/5.17/16.7讀取作初次背景，不冒稱本次重讀其他全節。','render_scope':'SVG完整SHA與已親看PNG不變，沿用原本人Inkscape render+view證據；無關網頁/全頁與SVG沒有重渲染。'},
 'change_assessment':{'byte_transform':'old.replace(UTF8("兩条路"), UTF8("兩條路"), 1) == current, exact bytes','actual_changes':['兩条路→兩條路字形修正'],'new_or_changed_substantive_claims':[],'unchanged_claim_ids':[c['id'] for c in prior['claims']],'judgement':'真完整重讀與byte diff確認14項概念/軟體/實測/數值主張及支持範圍未變；當前16.4必要GQA切片與原官方支持一致，不增加KV快取/其他成績驗收。','new_questions_requiring_tensor_execution':False,'residual_dependencies':[]},
 'retained_own_proofs':retained,
 'current_contract_fingerprints':current_contracts,
 'retained_authoritative_locators':[
  {'source_id':'s_sdpa','url':'https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/torch/nn/functional.py','version':'5c4886908584029761b579af026dcfb627c84070','locator':'6367–6536，原本人查核；本次另親讀GQA6394–6398及6473–6479','snapshot_path':rel(BASE/'sources/functional-5c488690.py'),'sha256':digest((BASE/'sources/functional-5c488690.py').read_bytes())},
  {'source_id':'s_math','url':'https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/torch/_torch_docs.py','version':'同commit','locator':'allclose833–863/matmul7907–7975，沿用原本人查核','snapshot_path':rel(BASE/'sources/torch-docs-5c488690.py'),'sha256':digest((BASE/'sources/torch-docs-5c488690.py').read_bytes())},
  {'source_id':'s_attention_paper','url':'https://arxiv.org/pdf/1706.03762v7','version':'arxiv v7','locator':'§3.2.1 Eq(1),p4及§3.2.3,p5，沿用原本人親核','snapshot_path':rel(BASE/'sources/attention-1706.03762v7.pdf'),'sha256':digest((BASE/'sources/attention-1706.03762v7.pdf').read_bytes())},
  {'source_id':'s_flash_paper','url':'https://arxiv.org/pdf/2205.14135v2','version':'arxiv v2','locator':'Abstract,p1/Introduction及Fig1圖說,p2，沿用原本人親核','snapshot_path':rel(BASE/'sources/flashattention-2205.14135v2.pdf'),'sha256':digest((BASE/'sources/flashattention-2205.14135v2.pdf').read_bytes())},
  {'source_id':'s_autograd','url':'https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/torch/autograd/__init__.py','version':'同commit','locator':'grad434–480，沿用原本人親核','snapshot_path':rel(BASE/'sources/autograd-5c488690.py'),'sha256':digest((BASE/'sources/autograd-5c488690.py').read_bytes())},
 ],
 'actual_command':'.venv/bin/python '+rel(OUT/'reinspect_current.py'),
 'code':{'path':rel(OUT/'reinspect_current.py'),'sha256':digest(Path(__file__).read_bytes())},
 'environment':environment,
 'executions':'只做原bytes對照、精確fence/圖/code/data/主源/全部本人proof fingerprints；無需重跑未變fence、CPU變體、GPU測量或training。',
 'retained_limitations':prior['limitations'],'verdict':'pass','unresolved_issues':[],
}
(OUT/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
def add(identifier,name,kind,description):
 path=OUT/name
 artifact={'id':identifier,'path':rel(path),'sha256':digest(path.read_bytes()),'kind':kind,'description':description}
 if kind=='execution':artifact.update(command=receipt['actual_command'],result='完整重讀+精確byte diff只有兩条→兩條字形；14項實質claims、原fence/SVG/current code/raw與全部原本人proof SHA不變；current16.4必要slice核對通過；不需額外tensor工作。',environment=environment)
 prior['artifacts'].append(artifact)
 return artifact
summary={'source_sha256':EXPECTED,'prior_report_sha256':digest(prior_raw),'retained_artifacts_checked':len(retained),'changed_substantive_claims':0,'receipt_artifact_id':receipt_id,'receipt_path':rel(OUT/'receipt.json'),'receipt_sha256':digest((OUT/'receipt.json').read_bytes()),'verdict':'pass'}
(OUT/'stdout.txt').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False,indent=2))
add(receipt_id,'receipt.json','execution','原技術owner完整當前節及必要currentcontext實讀、變動主張辨識、本人證據沿用與精確fingerprints的新receipt。')
for identifier,name,kind,description in [
 ('a_current_section_reinspection','current-section.md','source_snapshot','當前16.8完整原UTF8 bytes，source identity011a997d...'),
 ('a_current_whole_reinspection','current-whole-frozen.md','source_snapshot','本次重讀時整章frozen raw bytes，明示不是current lesson identity。'),
 ('a_prior_report_reinspection','prior-report.opaque.json','source_snapshot','先opaque保存own前完整report及問題歷史，SHA37680f...；沒有覆寫歷史。'),
 ('a_prior_proof_manifest_reinspection','prior-artifact-manifest.opaque.json','source_snapshot','own前proof SHA清單opaque備份；原proof本身未修改且全部親核SHA。'),
 ('a_reinspection_diff','section-diff.txt','derivation','精確UTF8 diff表明僅两条→两條字形，沒有公式/code/數字/換行變動。'),
 ('a_current_context_16_4','current-context-16.4-selected.md','source_snapshot','只親讀16.4 current V3標題、開場GQA定義及一般分組/MQA限制段。'),
 ('a_current_context_scope_16_4','current-context-16.4-scope.json','source_snapshot','current16.4必要切片/全節hash與本人真正實讀定位；不擴原實驗評審。'),
 ('a_current_factual_method','current-factual-instructions.md','source_snapshot','本次完整讀取的最新factual方法原bytes。'),
 ('a_reinspection_code','reinspect_current.py','code','實際本次指紋/byte diff/範圍驗證與canonical報告更新程式。'),
 ('a_reinspection_stdout','stdout.txt','execution','本次指紋與scope查核實際stdout，含receipt實際ID/path/SHA。'),
 ('a_reinspection_hashes','retained-evidence-hash-checks.json','source_snapshot','前次所有canonical artifact ID/path/SHA的實際重核，原proof均未變。'),
 ('a_reinspection_freeze','freeze.json','source_snapshot','先opaque保存前報與本次raw/current來源的實際指紋記錄。')]:add(identifier,name,kind,description)
prior['source_sha256']=EXPECTED
prior['verdict']='pass'
prior['actual_read_scope']['reinspection']=receipt['actual_read_scope']
prior['checks']['factual_accuracy']['details']+=' 本次原owner完整讀current，byte差只為字形；receipt '+receipt_id+' 判斷所有原14項支持範圍仍適用。'
prior['checks']['source_verification']['details']+=' 本次逐一SHA核原本人canonical證據及current fence/SVG/code/raw；無重新fetch，新增current16.4必要slice與receipt。'
prior['checks']['figure_consistency']['details']+=' 本次SVG與原親看PNG完整SHA相同，沿用本人原render/view，沒有重做未變圖。'
prior['reinspection_history']=[{'same_owner':OWNER,'source_sha256':EXPECTED,'prior_report_sha256':digest(prior_raw),'receipt_artifact_id':receipt_id,'receipt_artifact_path':rel(OUT/'receipt.json'),'receipt_sha256':digest((OUT/'receipt.json').read_bytes()),'current_context':context,'changed_substantive_claims':[],'verdict':'pass','remaining_dependencies':[]}]
prior['independent_summary']+=' 原owner本次完整重讀source011a997d...，僅兩条→兩條字形修正；14項主張無實質變動，精確證據/圖/code/raw指紋一致，當前16.4必要GQA切片與官方支持相同。'
report_path=ROOT/'docs/technical-reviews/16.8.json'
report_path.write_text(json.dumps(prior,ensure_ascii=False,indent=2)+'\n')
confirmed=json.loads(report_path.read_bytes())
assert confirmed['reviewer_task']==OWNER and confirmed['source_sha256']==EXPECTED
assert any(a['id']==receipt_id and a['sha256']==summary['receipt_sha256'] for a in confirmed['artifacts'])
assert confirmed['issues']==[] and confirmed['verdict']=='pass'
summary['report_sha256']=digest(report_path.read_bytes())
(OUT/'submission-before-checker.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
