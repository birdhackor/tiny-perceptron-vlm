"""Actual second reading receipt: reuse unchanged evidence, never claim rerun."""
import copy
import difflib
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from datetime import datetime, UTC
from pathlib import Path

ROOT=Path('/workspace/tiny-perceptron-vlm')
OUT=Path(__file__).parent
BASE=OUT.parent
REL=OUT.relative_to(ROOT).as_posix()
sha=lambda raw:hashlib.sha256(raw).hexdigest()
history=ROOT/'docs/technical-reviews/history/phase4-6_1-own-initial-revise-777c89bbee1bc92f8ad46ab9b723c2253feb3f61819c0049d2ea8bd979505eb2.json'
history_raw=history.read_bytes()
assert sha(history_raw)=='777c89bbee1bc92f8ad46ab9b723c2253feb3f61819c0049d2ea8bd979505eb2'
initial=json.loads(history_raw)
assert initial['reviewer_task']=='/root/phase4_factual_coordinator/factual_6_1'
assert initial['verdict']=='revise' and initial['claims'][0]['status']=='contradicted'
(OUT/'own-initial-revise.raw.json').write_bytes(history_raw)
# The helper was read during the independent initial review. Audit its hash now,
# then use its original raw-byte slicing functions only; do not execute fences.
helper=ROOT/'docs/review-tools/section_facts.py'
frozen_helper=BASE/'inputs/docs/review-tools/section_facts.py'
assert sha(helper.read_bytes())==sha(frozen_helper.read_bytes())
spec=importlib.util.spec_from_file_location('recheck_section_facts',frozen_helper)
facts=importlib.util.module_from_spec(spec)
spec.loader.exec_module(facts)
section,whole,first_line=facts.original_section(ROOT/'course/chapters/06.md','6.1')
intro=whole[:whole.index(b'## 6.1 ')]
old_section=(BASE/'original/section.md').read_bytes()
old_intro=(BASE/'inputs/chapter-intro.raw.md').read_bytes()
assert intro==old_intro
old_sentence='token是模型一次處理的片段，tokenizer將原文拆片段並轉ID，再能解碼還原。'
new_sentence='token是模型一次處理的片段，tokenizer將原文拆成片段並轉ID；decoder把ID轉回文字，能否完整還原原文取決於字表與前處理規則。'
assert old_section.decode().count(old_sentence)==1
assert section.decode()==old_section.decode().replace(old_sentence,new_sentence)
new_fences=facts.fences(section,first_line)
assert len(new_fences)==1 and new_fences[0]['language']=='python'
new_code=new_fences[0]['raw']
assert new_code==(BASE/'original/fence-1.py').read_bytes()
assert not re.findall(r'!\[[^\]]*\]\(([^)]+)\)',section.decode())
(OUT/'section.raw.md').write_bytes(section)
(OUT/'chapter-intro.raw.md').write_bytes(intro)
(OUT/'fence-1.raw.py').write_bytes(new_code)
(OUT/'section.diff.txt').write_text(''.join(difflib.unified_diff(
 old_section.decode().splitlines(keepends=True),section.decode().splitlines(keepends=True),
 fromfile='initial-6.1.raw.md',tofile='rechecked-6.1.raw.md')),encoding='utf-8')
reused=[]
for artifact in initial['artifacts']:
 p=ROOT/artifact['path']
 actual=sha(p.read_bytes())
 assert actual==artifact['sha256'],p
 reused.append({'artifact_id':artifact['id'],'path':artifact['path'],'sha256':actual,
                'matches_initial_report':True})
probe=json.loads((BASE/'probe.json').read_text())
assert probe['char_unseen_counterexample']=={'input':'貓','ids':[0],'decoded':'<unk>','equal_original':False}
audit={
 'reviewer_task':initial['reviewer_task'],'executed_at':datetime.now(UTC).isoformat(),
 'command':'.venv/bin/python '+REL+'/recheck.py','cwd':str(ROOT),
 'environment':{'python':sys.version,'device':'cpu','operation':'raw-byte/hash audit and checker; no model execution'},
 'actual_read_scope':['Complete current chapter 6 title/introduction','Complete current 6.1, including fence, exercise, T.4 empirical supplement',
                      'Own initial 6.1 raw bytes for diff','HF v4.57.1 tokenizer_summary Subword example txt260–277 and BPE/Byte-level BPE txt337–353',
                      'GPT-2 original report §2.2 txt169–185','Historical ByteTokenizer/CharTokenizer L15–44','Own original probe counterexample and poem result'],
 'initial_source_sha256':sha(old_section),'current_source_sha256':sha(section),
 'intro_sha256':sha(intro),'intro_unchanged':True,'fence_sha256':sha(new_code),'fence_unchanged':True,
 'helper_sha256':sha(helper.read_bytes()),'helper_unchanged':True,
 'figures':[],'only_change':{'old':old_sentence,'new':new_sentence},
 'initial_history_path':str(history.relative_to(ROOT)),'initial_history_sha256':sha(history_raw),
 'reused_artifact_hash_audit':reused,
 'execution_evidence_reused':{'original_fence':True,'probe':True,'executions_rerun_this_recheck':False,
  'reason':'Only prose contract changed; original fence, all numerical examples, empirical input and original artifacts are byte-identical.'},
 'inspection':'親自完整重讀新版小節及章首。新版明確將decoder轉回文字與無損還原分開，以字表與前處理規則限定契約。HF官方未知m→<unk>與uncased I/GPU→i/gpu，及本課已實跑貓→<unk>，現在均可被這個限定解釋。原反例仍真實，原無條件句仍屬錯誤，只是新版已不再作該主張。其餘C2–C8文字和code逐段讀後與原逐claim證據一致，無新增實質未知。',
 'issue_resolution':'I1 resolved after actual full reread; the original contradicted C1 is retained verbatim in own initial history and review_history.'}
(OUT/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
report=copy.deepcopy(initial)
report['source_sha256']=sha(section)
report['verdict']='pass'
report['read_scope']=audit['actual_read_scope']
report['intro_summary']=initial['intro_summary']
report['intro_sha256']=sha(intro)
report['intro_source']='current course/chapters/06.md bytes before ## 6.1; raw recheck snapshot; unchanged from first independent read'
report['review_history']=[{'stage':'initial_independent_review','verdict':'revise',
 'source_sha256':initial['source_sha256'],'report_path':str(history.relative_to(ROOT)),
 'report_sha256':sha(history_raw),'claim_C1':copy.deepcopy(initial['claims'][0]),
 'issue_I1':copy.deepcopy(initial['issues'][0])},
 {'stage':'actual_recheck','source_sha256':sha(section),'scope':audit['actual_read_scope'],
  'inspection':audit['inspection'],'execution_evidence_reused':audit['execution_evidence_reused']}]
report['issues'][0]['status']='resolved'
report['issues'][0]['resolution']=audit['inspection']
report['issues'][0]['rechecked_original_text']=new_sentence
report['issues'][0]['initial_revise_history']=str(history.relative_to(ROOT))
report['claims'][0]['status']='verified'
report['claims'][0]['statement']='tokenizer將原文拆成片段並轉ID；decoder把ID轉回文字，完整還原原文取決於字表與前處理規則。切分單位可是一字、多字或一字的部分bytes。'
report['claims'][0]['scope']='新版為條件契約，沒有宣稱所有tokenizer或所有字表都無損；未知字標記和小寫化等前處理均可丟失原文資訊。原C1矛盾與實跑反例完整保留在review_history與history原檔。'
report['claims'][0]['evidence'].append({'source_id':'S7','locator':'§2.2 Input Representation, original report p3 txt169–185',
 'supports':'lowercasing/tokenization/out-of-vocabulary會限制原文表示；新版前處理與字表限定有原論文支持。'})
report['claims'][0]['evidence'].append({'source_id':'S9','locator':'Subword tokenization uncased BERT example, txt260–277',
 'supports':'官方I have a new GPU!會先lowercase，不能由decode恢復原大小寫；新版沒有保證原文無損。'})
for source in report['sources']:
 if source['id']=='S9':
  source['inspection_note']+=' 本次複查另親讀Subword txt260–277的小寫化示例及再次親讀337–353未知字/byte表段，確認新版兩項條件。'
 if source['id']=='S7':
  source['inspection_note']+=' 本次複查再次親讀§2.2 txt169–185的lowercasing/out-of-vocabulary限制，確認新版前處理限定。'
report['checks']['factual_accuracy']={'status':'pass','details':'完整重讀本節。C1已明說還原條件，未知字和小寫化原反例支持新版；原錯句/contradicted claim保留。其他7組claims逐段與未變原證據對照，無新實質问题。','claim_ids':[c['id'] for c in report['claims']]}
report['checks']['limitations']={'status':'pass','details':'新版補足還原的字表/前處理條件。其餘dense表範圍、常/約、UTF-8 scalar範圍、一筆既有生成的限制仍成立；原CPU執行結果hash皆吻合，誠實沿用而非宣稱本次重跑。','claim_ids':['C1','C4','C5','C6','C7','C8']}
report['checks']['numeric_verification']['details']+=' 本次完整重讀與raw fence hash確認數字/原碼未變；沿用初輪親跑證據，本次未重跑。'
report['checks']['source_verification']['details']+=' 複查親讀未知字與lowercasing的原始定位，全部初輪artifact hash核對相符。'
for index,(filename,kind,description) in enumerate([
 ('recheck.py','code','實際複查命令程式：完整raw section/intro/fence/初輪artifact hash audit，記讀取範圍後產生本版報告及checker receipt。'),
 ('section.raw.md','source_snapshot','本次親讀的新版6.1原始UTF-8 bytes，不正規化換行。'),
 ('chapter-intro.raw.md','source_snapshot','本次親讀章首導言raw bytes；與初輪精確一致。'),
 ('fence-1.raw.py','code','新版原fence bytes；與初輪已執行原fence完全一致，本次未重跑。'),
 ('section.diff.txt','derivation','自己從初輪原bytes與新版原bytes產生的diff，唯一變更為decoder/還原條件句。'),
 ('own-initial-revise.raw.json','source_snapshot','自己的initial revise逐byte副本，保存原contradicted C1/I1與初輪判定。'),
 ('audit.json','execution','本次實際raw/hash核對；保留實際讀取範圍、修句分析及證據沿用真實界線。')],1):
 p=OUT/filename
 artifact={'id':'RC'+str(index),'kind':kind,'path':str(p.relative_to(ROOT)),'sha256':sha(p.read_bytes()),'description':description}
 if kind=='execution':
  artifact.update(command=audit['command'],environment=audit['environment'],
    result='All hash/diff assertions passed; original counterexample preserved, I1 resolved by actual current full reread; no model/code-fence rerun.')
 report['artifacts'].append(artifact)
report['claims'][0]['artifact_ids']+=['RC2','RC5','RC6','RC7']
report_path=ROOT/'docs/technical-reviews/6.1.json'
report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
command=[str(ROOT/'.venv/bin/python'),'scripts/check_technical_reviews.py','--lesson','6.1']
completed=subprocess.run(command,cwd=ROOT,capture_output=True,timeout=20)
(OUT/'checker.stdout.txt').write_bytes(completed.stdout)
(OUT/'checker.stderr.txt').write_bytes(completed.stderr)
receipt={'command_argv':command,'cwd':str(ROOT),'exit_code':completed.returncode,
 'executed_at':datetime.now(UTC).isoformat(),'environment':audit['environment'],
 'source_sha256':sha(section),'report_sha256':sha(report_path.read_bytes()),
 'initial_history_path':str(history.relative_to(ROOT)),'initial_history_sha256':sha(history_raw),
 'evidence_audit_sha256':sha((OUT/'audit.json').read_bytes()),
 'checker_sha256':sha((ROOT/'scripts/check_technical_reviews.py').read_bytes()),
 'stdout_sha256':sha(completed.stdout),'stderr_sha256':sha(completed.stderr)}
(OUT/'checker-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(completed.stdout.decode(),end='')
print(json.dumps(receipt,ensure_ascii=False,indent=2))
raise SystemExit(completed.returncode)
