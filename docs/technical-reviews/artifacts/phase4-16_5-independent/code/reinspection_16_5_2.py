"""Same-owner actual current-section reinspection, with no retraining or scores."""
import hashlib
import importlib.util
import json
import platform
from pathlib import Path

ROOT=Path(__file__).resolve().parents[5]
OUT=Path(__file__).resolve().parents[1]
REL=OUT.relative_to(ROOT).as_posix()
TASK='/root/phase4_factual_coordinator/factual_16_5'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location('facts',ROOT/'docs/review-tools/section_facts.py')
facts=importlib.util.module_from_spec(spec);spec.loader.exec_module(facts)
body,whole,line=facts.original_section(ROOT/'course/chapters/16.md','16.5')
old=(OUT/'input/section-recheck-1.md').read_bytes()
assert body==(OUT/'input/section-recheck-2.md').read_bytes()
assert hashlib.sha256(body).hexdigest()=='0b28e18284ac77b98bb24e8a7a86d11310c8608cbcadebd894a32a79998212e6'
assert old.count('谁'.encode())==1 and body==old.replace('谁'.encode(),'誰'.encode())
prior_path=OUT/'prior-current-review-1-opaque.json'
assert sha(prior_path)=='8900473d1c4ad48c020b9facacafc63fc65a7b2cf8ed8ad160d4b0ad2e050694'
prior=json.loads(prior_path.read_text())
assert prior['reviewer_task']==TASK
checks=[]
for s in prior['sources']:
 if s['kind']=='repository_code':
  observed=sha(ROOT/s['path']);assert observed==s['sha256']
  checks.append({'id':s['id'],'path':s['path'],'sha256':observed,'reuse':'own previously inspected original source, unchanged'})
for a in prior['artifacts']:
 assert sha(ROOT/a['path'])==a['sha256']
contexts=[]
for lesson,path in [('3.6','course/chapters/03.md'),('3.7','course/chapters/03.md'),
                    ('7.3','course/chapters/07.md'),('7.7','course/chapters/07.md'),('5.10','course/chapters/05.md')]:
 current,_,_=facts.original_section(ROOT/path,lesson)
 snapshot=OUT/'input/dependencies'/(lesson+'.md')
 assert current==snapshot.read_bytes()
 contexts.append({'source':path+'#'+lesson,'actual_prior_snapshot':snapshot.relative_to(ROOT).as_posix(),
                  'sha256':hashlib.sha256(current).hexdigest(),'unchanged':True,
                  'scope':'own previously read necessary definition; no new context dependence'})
oldf=facts.fences(old,197)[0]['raw'];newf=facts.fences(body,line)[0]
assert oldf==newf['raw']==(OUT/'input/fence-1.py').read_bytes()
figure=ROOT/'course/figures/rewrite-16-packed-mask.svg'
assert figure.read_bytes()==(OUT/'input/rewrite-16-packed-mask.svg').read_bytes()
snapshot_fingerprint=[]
for a in prior['artifacts']:
 if a['path'].startswith(REL+'/primary/'):
  snapshot_fingerprint.append({'id':a['id'],'path':a['path'],'sha256':a['sha256']})
receipt={
 'id':'reinspection_16_5_2','reviewer_task':TASK,'performed_on':'2026-10-05',
 'source':'course/chapters/16.md#16.5','source_sha256':hashlib.sha256(body).hexdigest(),
 'actual_section_snapshot':REL+'/input/section-recheck-2.md','section_first_line':line,
 'instructions':{'path':'docs/review-tools/factual-reviewer-instructions.md',
                 'sha256':sha(ROOT/'docs/review-tools/factual-reviewer-instructions.md'),'personally_read_latest':True},
 'actual_read_scope':'本人重新讀目前完整16.5正文、原fence、圖說、全部選讀實測與範圍段；與本人前次原section bytes比較，只發現誰字繁體修訂。另親讀current attention_mask lines10-18。',
 'changed_text':{'prior':'先畫出谁能讀誰','current':'先畫出誰能讀誰',
                 'exact_only_change':True,'changed_substantive_claims':[],
                 'own_assessment':'字形修訂不改Query/Key、因果或同segment取交集的意思；完整當前段仍清楚說明B不可讀A。不是僅更新指紋，已讀全節並核對這個含義。'},
 'necessary_context':contexts,
 'not_needed_context':['16.4：本節packing隔離不依賴該鄰節的新prose','16.1導言：非本owner範圍'],
 'intro':{'applicable':False,'inspected':False,'reason':'本owner只審16.5；非第一節，不代審intro，也不把chapter-wide fingerprint當intro已驗證。'},
 'scientific_support_for_changed_wording':[
  {'path':'tiny_perceptron/attention.py','sha256':sha(ROOT/'tiny_perceptron/attention.py'),
   'locator':'attention_mask lines10-18','actual_action':'本人重讀契約：key_position<=query_position，且segments兩端相等；bool表列Query欄Key。'},
  {'source_id':'transformer','version':'arXiv1706.03762v7','locator':'page3 §3.1 decoder mask; page4 §3.2.1 equation1',
   'actual_action':'原方法概念與原權威摘錄未變，精確snapshotSHA核對後復用本人已親读的原證據，不重新fetch論文。'},
  {'source_id':'megatron','version':'core_v0.12.0','locator':'_get_ltor_masks_and_position_ids lines620-694',
   'actual_action':'EOS/position/reset-mask概念未變，復用本人原核查與精確SHA摘錄。'}],
 'reused_local_original_source_fingerprints':checks,
 'reused_primary_snapshot_fingerprints':snapshot_fingerprint,
 'original_measurement_scope':'raw L4效率檔與dataset原件SHA均未變；7+7、845、40、37與15保留題的原數值主張逐段重讀未變，復用本人初審與recheck1的真實計算/方法證據，不借前次pass結論充當來源。',
 'timing_scope_reinspection':'現文仍逐項列前3次暖身排除、餘37次以及計時內外步驟；本人recheck1親核immutable _packing_updates731-751,773-787與raw warm_step_median_seconds的正式receipt完全保留且SHA未變。',
 'figure':{'source_path':figure.relative_to(ROOT).as_posix(),'sha256':sha(figure),
           'actual_render_reused':REL+'/execution/packed-mask-inkscape.png',
           'render_sha256':sha(OUT/'execution/packed-mask-inkscape.png'),
           'scope':'本輪未重渲染；原SVG及本人已實際渲染/親view的PNG精確指紋未變，圖解相關文字與fence未變。'},
 'fence':{'sha256':hashlib.sha256(newf['raw']).hexdigest(),'current_code_first_line':newf['code_line'],
          'scope':'原碼bytes完全未變，復用本人真正執行original fence與CPU短變化的證據，未虛報本輪重跑。'},
 'preview':{'provided_url':'http://127.0.0.1:8765/16.5.html','used_as_inspection_evidence':False,
            'scope':'本輪只有字形變動、無新的畫面技術主張；未聲稱本人browser或root preview目視驗證。圖證據是本人的原render/view，不是根預覽。'},
 'prior_opaque':{'path':prior_path.relative_to(ROOT).as_posix(),'sha256':sha(prior_path),
                 'preservation':'完整own前一report、全部issues與既有proof原樣保留；無刪除或覆寫歷史。'},
 'environment':{'python':platform.python_version(),'device':'cpu','work':'原始bytes/AST契約及指紋核對；沒有模型運算或訓練'},
 'no_new_cpu_needed_reason':'無新算例、分母、單位、圖或原碼；現source與前source只有一個字形差，真正機制無新疑問。',
 'verdict':'pass','unresolved_substantive_issues':[],'pending_dependencies':[]}
receipt_path=OUT/'reinspection-16_5-2.json'
receipt_path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print('SOURCE_SHA256',receipt['source_sha256'])
print('EXACT_ONLY_CHANGE',receipt['changed_text'])
print('INTRO',receipt['intro'])
print('FIGURE_SHA256',receipt['figure']['sha256'])
print('FORMAL_RECEIPT_ID',receipt['id'])
print('FORMAL_RECEIPT_SHA256',sha(receipt_path))
print('REINSPECTION_COMPLETE_NO_NEW_TECHNICAL_CHANGE')
