"""Original owner's actual current reread; archive own report before rewriting."""
import copy
import difflib
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT=Path('/workspace/tiny-perceptron-vlm')
OUT=Path(__file__).parent
BASE=OUT.parent
sys.dont_write_bytecode=True
sha=lambda b:hashlib.sha256(b).hexdigest()
command='PYTHONDONTWRITEBYTECODE=1 .venv/bin/python '+str((OUT/'reinspect.py').relative_to(ROOT))
report_path=ROOT/'docs/technical-reviews/6.1.json'
prior_raw=report_path.read_bytes()
prior_sha=sha(prior_raw)
history=ROOT/'docs/technical-reviews/history'/('phase4-6_1-own-before-current-reinspection-'+prior_sha+'.json')
assert not history.exists(), 'Do not overwrite an existing history record'
history.write_bytes(prior_raw)
(OUT/'own-prior-report.raw.json').write_bytes(prior_raw)
assert history.read_bytes()==prior_raw
prior=json.loads(prior_raw)
assert prior['reviewer_task']=='/root/phase4_factual_coordinator/factual_6_1'
assert prior_sha=='883b517d8908b46a21e1ef22e3f72731c74ac80f888d597d5a893b83454dbe8b'
assert prior['verdict']=='pass'
spec=importlib.util.spec_from_file_location('own_raw_slicer',BASE/'inputs/docs/review-tools/section_facts.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
section,whole,first_line=helper.original_section(ROOT/'course/chapters/06.md','6.1')
intro=whole[:whole.index(b'## 6.1 ')]
old_section=(BASE/'recheck-20261005/section.raw.md').read_bytes()
old_intro=(BASE/'recheck-20261005/chapter-intro.raw.md').read_bytes()
assert sha(old_section)==prior['source_sha256']
assert section.decode()==old_section.decode().replace('较少種類V','較少種類V').replace('序列较長','序列較長')
assert intro==old_intro
assert sha(section)=='aecd7b3bee1470b4e1e15d0b714a38c34eac34eb3de072e8dfee05de272304a6'
fences=helper.fences(section,first_line)
assert len(fences)==1 and fences[0]['language']=='python'
code=fences[0]['raw']
assert code==(BASE/'original/fence-1.py').read_bytes()
assert not re.findall(r'!\[[^\]]*\]\(([^)]+)\)',section.decode())
(OUT/'section.raw.md').write_bytes(section)
(OUT/'chapter-intro.raw.md').write_bytes(intro)
(OUT/'fence-1.raw.py').write_bytes(code)
diff=''.join(difflib.unified_diff(old_section.decode().splitlines(keepends=True),section.decode().splitlines(keepends=True),fromfile='own-previous-6.1',tofile='current-6.1'))
(OUT/'actual-diff.txt').write_text(diff,encoding='utf-8')
(OUT/'factual-reviewer-instructions.raw.md').write_bytes((ROOT/'docs/review-tools/factual-reviewer-instructions.md').read_bytes())
artifact_checks=[]
for item in prior['artifacts']:
 actual=sha((ROOT/item['path']).read_bytes())
 assert actual==item['sha256'],item['path']
 artifact_checks.append({'artifact_id':item['id'],'path':item['path'],'sha256':actual,'unchanged':True})
live_checks=[]
for name in ['tiny_perceptron/data.py','tiny_perceptron/attention.py','scripts/check_technical_reviews.py']:
 live_sha=sha((ROOT/name).read_bytes());frozen_sha=sha((BASE/'inputs'/name).read_bytes())
 assert live_sha==frozen_sha,name
 live_checks.append({'path':name,'live_sha256':live_sha,'own_frozen_sha256':frozen_sha,'unchanged':True})
probe=json.loads((BASE/'probe.json').read_bytes())
lengths={r['text']:(r['codepoints'],r['utf8_bytes']) for r in probe['length_rows']}
assert [lengths[s] for s in ['小小貓','cat','🙂','cat🙂','貓🙂']]==[(3,9),(3,3),(1,4),(4,7),(2,7)]
assert [(r['T'],r['table_entries']) for r in probe['attention']]==[(3,9),(9,81)]
assert probe['poem']['prompt_utf8_bytes']==23
assert len(probe['poem']['generated_ids'])==32
assert probe['poem']['final_partial_byte_count']==2
intro_summary='本章從可見的「小小貓」追到 byte 與模型 ID，先合常見片段再建立可還原未見文字的 BPE，並以字表、文字代價、標記、半字解碼與窗口討論每個位置的含義。'
receipt={
 'reviewer_task':prior['reviewer_task'],'executed_at':datetime.now(UTC).isoformat(),
 'command':command,'cwd':str(ROOT),
 'environment':{'python':sys.version,'device':'cpu','operation':'UTF-8 raw capture, diffs and evidence hash/result audit; no model execution'},
 'prior_history_path':str(history.relative_to(ROOT)),'prior_history_sha256':prior_sha,
 'prior_source_sha256':sha(old_section),'source_sha256':sha(section),
 'intro_sha256':sha(intro),'intro_summary':intro_summary,'intro_unchanged':True,
 'read_scope':['Complete current 6.1 including original fence, exercise, empirical supplement and helper notice',
               'Actual chapter 6 title and entire introductory paragraph before first numbered heading',
               'Own previous frozen 6.1 and introduction for comparison',
               'Current factual-reviewer-instructions.md',
               'Own original GPT-2 paper snapshot §2.2 Input Representation, txt169–231',
               'Own RFC 3629 original snapshot §3, txt175–204',
               'Own initial CPU probe length_rows/attention and named poem counters; no author repair answers'],
 'raw_changed_text':[{'before':'较少種類V','after':'較少種類V','claim_ids':['C5'],'judgment':'Traditional character correction; vocabulary-count V and typical-tradeoff scope unchanged.'},
                     {'before':'中文序列较長','after':'中文序列較長','claim_ids':['C6'],'judgment':'Traditional character correction; byte-sequence unit and UTF-8 scope unchanged.'}],
 'material_changed_claims':[],
 'independent_inspection':'完整重讀後，比較自己的前版原bytes只見兩處较→較。V仍指token種類數而非token長度；較細與較粗是常見取捨，未變成普遍定理。中文序列較長仍指本節固定UTF-8拆byte與同碼點長度cat的比較；重新親讀RFC表核對常用BMP漢字3byte、ASCII1byte，原小小貓9與cat3的自己的CPU結果保持一致。重讀GPT-2 §2.2確認byte基表256與碼點/較粗片段的表示取捨。全文token/decoder有條件還原、注意力T²、練習與詩截停的範圍均未改，原I1仍已解。沒有新增實質claim或未解疑點。',
 'source_locators':[{'source_id':'S7','url':'https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf','version':'Radford et al. OpenAI GPT-2 report, 2019','snapshot':str((BASE/'sources/gpt2-report.pdf').relative_to(ROOT)),
                     'locator':'§2.2 Input Representation, p3; own txt169–231','supports':'C5/C6: byte-level base256, differing character/word/BPE granularity; not a universal numeric best vocabulary.'},
                    {'source_id':'S5','url':'https://www.rfc-editor.org/rfc/rfc3629.txt','version':'RFC 3629, November 2003','snapshot':str((BASE/'sources/rfc3629.txt').relative_to(ROOT)),
                     'locator':'§3 UTF-8 definition/range table, own txt175–204','supports':'C6: common BMP Chinese needs3 UTF-8 bytes while ASCII1; the literal exemplars retain9/3 bytes.'}],
 'code_fence_sha256':sha(code),'code_unchanged':True,'figures':[],
 'visual_scope':'Current 6.1 has no figure references and no changed visual material; no render/view claimed. The parent preview URL is a locator, not technical proof.',
 'reuse_policy':{'original_fence_execution_reused':True,'original_cpu_variants_and_empirical_audit_reused':True,'code_execution_rerun':False,
                'reason':'Unchanged fence and original evidence hashes; current change only two orthographic characters. Current command audits saved evidence, not replaying model training or inference.'},
 'artifact_hash_audit':artifact_checks,'live_implementation_hash_audit':live_checks,
 'original_result_rechecks':{'lengths':lengths,'attention_grids':[(r['T'],r['table_entries']) for r in probe['attention']],
                             'prompt_bytes':23,'new_byte_ids':32,'final_partial_bytes':2},
 'verdict':'pass','unresolved_issues':[]}
receipt_path=OUT/'reinspection-receipt.json'
receipt_path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
report=copy.deepcopy(prior)
report['source_sha256']=sha(section)
report['verdict']='pass'
report['read_scope']=receipt['read_scope']
report['intro_sha256']=sha(intro)
report['intro_summary']=intro_summary
report['intro_source']='Current actual chapter 6 introduction, original UTF-8 bytes before ## 6.1; saved in reinspection-current-20261005/chapter-intro.raw.md.'
report['review_history'].append({'stage':'current_original_owner_reinspection','prior_report_path':str(history.relative_to(ROOT)),
 'prior_report_sha256':prior_sha,'source_sha256':sha(section),'scope':receipt['read_scope'],
 'inspection':receipt['independent_inspection'],'changed_material_claims':[],
 'execution_reuse':receipt['reuse_policy'],'receipt_artifact_id':'FR1'})
new_artifacts=[('FR1','reinspection-receipt.json','execution','原技術owner本次實際複查receipt，記完整親讀scope、raw diff、兩處字形與其claim範圍判斷、原來源locators及真實hash/result audit。'),
               ('FR2','reinspect.py','code','本次實際命令程式；先opaque保存自己的前報告，再保存現稿、核對證據與產生canonical報告，最後另執行checker。'),
               ('FR3','section.raw.md','source_snapshot','本次完整親讀的current6.1原UTF8 bytes。'),
               ('FR4','chapter-intro.raw.md','source_snapshot','本次真正章首導言raw bytes；本人重讀並寫自己的summary。'),
               ('FR5','actual-diff.txt','derivation','對自己previous frozen input產生的實際diff，兩個较→較，其他字句/範圍相同。'),
               ('FR6','own-prior-report.raw.json','source_snapshot','自己的前canonical報告opaque副本，與history逐byte一致；保留原問題/解決過程。'),
               ('FR7','fence-1.raw.py','code','本次現稿原fence；實核與初輪已跑code精確相同，誠實沿用初輪結果。'),
               ('FR8','factual-reviewer-instructions.raw.md','source_snapshot','本次親讀的方法快照，保留當次規約與指紋。')]
for aid,filename,kind,desc in new_artifacts:
 p=OUT/filename
 entry={'id':aid,'path':str(p.relative_to(ROOT)),'sha256':sha(p.read_bytes()),'kind':kind,'description':desc}
 if kind=='execution':entry.update(command=command,environment=receipt['environment'],result='All raw diff/hash/result audit assertions passed. Full current reread found only orthography changes, no unresolved claim; original executions reused, not rerun.')
 report['artifacts'].append(entry)
report['reinspection_receipt']={'artifact_id':'FR1','path':str(receipt_path.relative_to(ROOT)),'sha256':sha(receipt_path.read_bytes())}
for claim in report['claims']:
 claim['artifact_ids']+=['FR1','FR3']
 if claim['id'] in ['C5','C6']:
  claim['scope']+=' 本次較字字形修訂沒有改此主張或單位；完整親讀及原來源重新定位見FR1/FR5。'
  claim['artifact_ids'].append('FR5')
report['sources'].append({'id':'E3','kind':'execution','title':'Original owner current reread raw/hash/result audit','verified':True,'artifact_id':'FR1'})
for source in report['sources']:
 if source['id'] in ['S5','S7']:
  source['inspection_note']+=' 本次current重讀後，重新亲讀FR1記錄的原snapshot定位核C5/C6的單位及範圍；未重新泛抓資料。'
for name in ['factual_accuracy','source_verification','limitations']:
 report['checks'][name]['details']+=' 本次原owner完整重讀current6.1與真章首：實際diff只較字字形，C5/C6來源段親查，無新實質claim或範圍問題；FR1保存inspection。'
report['checks']['numeric_verification']['details']+=' 本次hash核原fence與保存的數值一致，未重跑；較字修訂未動數字或計量單位。'
report['checks']['figure_consistency']={'status':'not_applicable','details':'親讀current6.1並raw解析引用：本節仍無圖，{}；無改圖或需要新render/view的內容，本次沒有冒稱視覺驗證。','claim_ids':[]}
report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
assert report_path.is_file() and json.loads(report_path.read_bytes())['source_sha256']==sha(section)
check_command=[str(ROOT/'.venv/bin/python'),'scripts/check_technical_reviews.py','--lesson','6.1']
check=subprocess.run(check_command,cwd=ROOT,capture_output=True,timeout=20)
(OUT/'checker.stdout.txt').write_bytes(check.stdout)
(OUT/'checker.stderr.txt').write_bytes(check.stderr)
check_receipt={'command_argv':check_command,'exit_code':check.returncode,'cwd':str(ROOT),
 'executed_at':datetime.now(UTC).isoformat(),'environment':receipt['environment'],
 'report_written_before_check':True,'report_sha256':sha(report_path.read_bytes()),'source_sha256':sha(section),
 'prior_history_path':str(history.relative_to(ROOT)),'prior_history_sha256':prior_sha,
 'canonical_receipt_artifact_id':'FR1','canonical_receipt_path':str(receipt_path.relative_to(ROOT)),
 'canonical_receipt_sha256':sha(receipt_path.read_bytes()),
 'checker_sha256':sha((ROOT/'scripts/check_technical_reviews.py').read_bytes()),
 'stdout_sha256':sha(check.stdout),'stderr_sha256':sha(check.stderr)}
(OUT/'checker-receipt.json').write_text(json.dumps(check_receipt,ensure_ascii=False,indent=2)+'\n')
print(check.stdout.decode(),end='')
print(json.dumps(check_receipt,ensure_ascii=False,indent=2))
raise SystemExit(check.returncode)
