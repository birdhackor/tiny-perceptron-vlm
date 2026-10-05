import difflib
import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[5]
OUT=Path(__file__).resolve().parent
BASE=OUT.relative_to(ROOT).as_posix()
INITIAL=ROOT/'docs/technical-reviews/history/phase4-6_8-own-initial-revise-5b79bfc0c52000a52097913022aa4232094c0f83b496ddc61595356011dcd626.json'
def sha(raw):return hashlib.sha256(raw).hexdigest()
assert sha(INITIAL.read_bytes())=='5b79bfc0c52000a52097913022aa4232094c0f83b496ddc61595356011dcd626'
report=json.loads(INITIAL.read_bytes())
assert report['verdict']=='revise' and report['issues'][0]['status']=='open'
assert report['reviewer_task']=='/root/phase4_factual_coordinator/factual_6_8'
source=Path('/tmp/phase4-6_8-independent-recheck')
for p in source.iterdir():
 if p.is_file():shutil.copyfile(p,OUT/p.name)
old=(OUT.parent/'original-run/section.md').read_bytes()
new=(OUT/'section.md').read_bytes()
assert sha(new)=='270ca0d9b3408029ec1da29fc3c38d02273fe85d411323835043866cf4b1709e'
assert new==old.replace('這些高度重疊窗口'.encode(),'這些窗口'.encode())
delta=''.join(difflib.unified_diff(old.decode().splitlines(True),new.decode().splitlines(True),fromfile='initial-6.8',tofile='revised-6.8'))
(OUT/'section.diff').write_text(delta)
assert (OUT/'fence-1.py').read_bytes()==(OUT.parent/'original-run/fence-1.py').read_bytes()
meta=json.loads((OUT/'extraction.json').read_text())
old_meta=json.loads((OUT.parent/'original-run/extraction.json').read_text())
assert meta['helper_sha256']==old_meta['helper_sha256']
assert meta['python_fences']==old_meta['python_fences']
assert meta['figure_sha256']==old_meta['figure_sha256']=={}
assert meta['svg_references']==old_meta['svg_references']==[]
art_checks=[]
for a in report['artifacts']:
 p=ROOT/a['path']
 assert sha(p.read_bytes())==a['sha256'],a['path']
 art_checks.append({'path':a['path'],'sha256':a['sha256'],'matched':True})
for name in ('tiny_perceptron/data.py','tiny_perceptron/model.py','scripts/course_experiments/common.py','scripts/course_experiments/text.py','docs/review-tools/section_facts.py','scripts/check_technical_reviews.py'):
 assert (ROOT/name).read_bytes()==(OUT.parent/'inputs/current'/name).read_bytes(),name
probe=json.loads((OUT.parent/'probe.results.json').read_text())
windows=probe['c3_windows']
raw=[w['x']+[w['y'][-1]] for w in windows]
boundary=[sorted(set(a)&set(b)) for a,b in zip(raw,raw[1:])]
events=[p for w in windows for p in zip(w['x'],w['y'])]
assert boundary==[[3],[6]]
assert events==list(zip(range(9),range(1,10))) and len(set(events))==9
assert all(not set(a['x'])&set(b['x']) and not set(a['y'])&set(b['y']) for a,b in zip(windows,windows[1:]))
sys.path.insert(0,str(ROOT))
import torch
import tokenizers
assert torch.version.cuda is None and not torch.cuda.is_available()
env={'python':sys.version,'python_executable':sys.executable,'torch':str(torch.__version__),'tokenizers':tokenizers.__version__,'device':'cpu','cuda_build':str(torch.version.cuda),'cuda_available':str(torch.cuda.is_available()),'platform':platform.platform(),'scope':'Textual full-section rereading, byte/hash/diff checks and saved executed-array recount only. Original fence/probe reused unchanged; no rerun, weights, training, dataset preparation or scoring.'}
(OUT/'environment.json').write_text(json.dumps(env,ensure_ascii=False,indent=2)+'\n')
observation={'initial_source_sha256':sha(old),'current_source_sha256':sha(new),'initial_report_history_path':INITIAL.relative_to(ROOT).as_posix(),'initial_report_sha256':sha(INITIAL.read_bytes()),'section_delta':'Only removed 高度重疊 from 這些高度重疊窗口','whole_section_reread':True,'whole_read_notes':f'{BASE}/whole-read-notes.md','fence_byte_equal':True,'helper_byte_equal':True,'reused_original_execution':True,'reran_original_fence_or_probe':False,'figure_count':0,'boundary_shared_ids':boundary,'shared_count_per_four_ID_window':[1,1],'overlap_denominator_per_window':4,'next_ID_events':events,'unique_next_ID_target_count':9,'duplicate_next_ID_target_count':0,'verified_prior_artifact_count':len(art_checks),'artifact_hash_checks':art_checks,'issue_recheck':'New section supports same-source splitting without calling this boundary-only example highly overlapping; no new substantive issue found.'}
(OUT/'recheck-results.json').write_text(json.dumps(observation,ensure_ascii=False,indent=2)+'\n')
new_artifacts=[]
for i,p in enumerate(sorted(OUT.rglob('*'))):
 if not p.is_file() or p.name.startswith('checker') or p.name in ('report-update.stdout.json','report-update.stderr.txt','checker-run.stdout.json','checker-run.stderr.txt'):
  continue
 kind='code' if p.suffix=='.py' else 'source_snapshot'
 if p.name in ('whole-read-notes.md','section.diff'):kind='derivation'
 a={'id':f'recheck1-{i}','kind':kind,'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p.read_bytes()),'description':f'Actual same-reviewer textual recheck1 evidence: {p.name}; original evidence reused only after hash equality, not rerun.'}
 if p.name=='recheck-results.json':
  a.update(kind='execution',command=f'.venv/bin/python {BASE}/recheck_report.py > {BASE}/report-update.stdout.json 2> {BASE}/report-update.stderr.txt',result='exit0; complete revised section/read notes preserved; exact textual diff verified;60 original artifacts and6 current helper/source files hash-identical; boundary IDs3/6 andnine unique saved executed targets independently recounted; original runs not repeated.',environment={k:str(v) for k,v in env.items()})
 new_artifacts.append(a)
report['artifacts'].extend(new_artifacts)
aid=next(a['id'] for a in new_artifacts if a['path'].endswith('/recheck-results.json'))
notes_id=next(a['id'] for a in new_artifacts if a['path'].endswith('/whole-read-notes.md'))
report['sources'].append({'id':'recheck1','kind':'execution','title':'Same original reviewer actual full-section textual recheck','verified':True,'artifact_id':aid})
c4=next(c for c in report['claims'] if c['id']=='c4')
c4['original_statement']=c4['statement']
c4['original_status']='unverified'
c4['statement']='Revised section permits the displayed shared boundary ID within one training side and requires complete-source/family grouping before assigning windows across training and validation.'
c4['scope']='The displayed windows share only boundaryID3 or6 and nine next-ID labels occur once. Grouping is supported by unseen dependent-source validation; revised section does not label them highly overlapping or infer a numerical leakage threshold. Original unsupported adjective and its initial unresolved status are preserved in issue/history.'
c4['status']='verified'
c4['evidence'].append({'source_id':'recheck1','locator':'recheck-1/section.md complete section; section.diff; whole-read-notes.md; recheck-results.json','supports':'Direct complete rereading and recount confirm changed wording genuinely distinguishes source-group isolation from duplicate supervised targets; no new substantive claim/change.'})
c4['artifact_ids'].extend([aid,notes_id])
issue=report['issues'][0]
issue['initial_status']='open'
issue['status']='resolved'
issue['initial_report_history_path']=INITIAL.relative_to(ROOT).as_posix()
issue['initial_report_history_sha256']=sha(INITIAL.read_bytes())
issue['revised_claim']='不能先從同一文件取這些窗口，再隨機分訓練與驗證'
issue['resolution']='Same original independent reviewer directly reread all of revised6.8 and the original grouped-data/leakage passages, rechecked saved actual arrays and exact diff. Only 高度重疊 removed. The one-ID shared boundary andnine unique next-ID targets remain explicit and unchanged; source grouping guidance stands independently of high-overlap labels. Complete rereading found no new substantive error. Recheck execution/hashes/read notes are saved; original code/probe not rerun, honestly reused after equality checks.'
issue['actual_recheck_artifact_ids']=[aid,notes_id]
report['checks']['factual_accuracy']={'status':'pass','details':'Complete revised6.8 reread by same initial reviewer. Original c4/i1, quantitative counterexample and revise history retained. The new phrase these windows no longer declares boundary-only segments highly overlapping; source/family isolation is independently supported. Other claims/exercises/budgets unchanged and rechecked against hashes/saved execution.','claim_ids':['c1','c2','c4','c5','c6','c8']}
report['checks']['source_verification']['details']+=' Recheck1 reread original scikit-learn group/leakage paragraphs; all authoritative snapshots/hash-matched inputs unchanged.'
report['verdict']='pass'
report['source_sha256']=sha(new)
report['recheck_history']=[{'stage':'initial','verdict':'revise','source_sha256':sha(old),'report_path':INITIAL.relative_to(ROOT).as_posix(),'report_sha256':sha(INITIAL.read_bytes()),'unresolved_issue_id':'i1'},{'stage':'actual-recheck1','verdict':'pass','source_sha256':sha(new),'scope':'Entire section directly reread, source passages reread and saved CPU arrays recounted, all previous inputs/code/authority hashes unchanged; prior executions reused and not rerun.','artifact_ids':[aid,notes_id]}]
report['review_scope']+=' Same reviewer actual complete revised6.8 reread and original source reread in recheck1; prior execution reused only after byte/hash equality.'
(ROOT/'docs/technical-reviews/6.8.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'verdict':report['verdict'],'source_sha256':report['source_sha256'],'report_sha256':sha((ROOT/'docs/technical-reviews/6.8.json').read_bytes()),'history_path':INITIAL.relative_to(ROOT).as_posix(),'history_sha256':sha(INITIAL.read_bytes()),'resolved_issue':'i1','prior_run_reuse':True,'original_fence_rerun':False},ensure_ascii=False,indent=2))
