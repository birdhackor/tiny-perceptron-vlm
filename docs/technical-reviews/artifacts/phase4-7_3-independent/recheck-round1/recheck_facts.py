"""New bounded CPU alignment check plus exact bytes checks for reused earlier proofs."""
import difflib
import hashlib
import json
import platform
import shutil
import sys
from datetime import datetime, UTC
from pathlib import Path

BASE=Path(__file__).resolve().parents[1]
ROOT=BASE.parents[3]
OUT=Path(__file__).parent
sys.path.insert(0,str(ROOT))
import torch
from tiny_perceptron.data import ByteTokenizer, render_chat

torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
initial=json.loads((BASE/'initial-review.json').read_text())
assert sha(BASE/'initial-review.json')=='f5959d637d2a809131856bd91da08ebf5f8c6526e3f7e67c98caae7294ee4c46'
history=ROOT/'docs/technical-reviews/history/phase4-7_3-own-initial-revise-f5959d637d2a809131856bd91da08ebf5f8c6526e3f7e67c98caae7294ee4c46.json'
assert sha(history)==sha(BASE/'initial-review.json')
temp=ROOT/'outputs/phase4-7_3-independent-recheck1'
for name in ['section.md','extraction.json','fence-1.py','bootstrap.py']:
    shutil.copyfile(temp/name,OUT/name)
new=json.loads((OUT/'extraction.json').read_text())
assert new['source_sha256']=='5c36cbb802c83d3e23586457498db50d0763152681d26a359cd20baca0b313c2'
assert new['figure_sha256']=={} and not new['svg_references']
artifacts=[]
for item in initial['artifacts']:
    observed=sha(ROOT/item['path'])
    assert observed==item['sha256'],item['id']
    artifacts.append({'id':item['id'],'path':item['path'],'original_sha256':item['sha256'],'observed_sha256':observed,'same_bytes':True})
contracts=[]
for name in ['tiny_perceptron/data.py','tiny_perceptron/model.py','tiny_perceptron/attention.py','scripts/build_course.py','docs/review-tools/section_facts.py','scripts/check_technical_reviews.py']:
    current_sha=sha(ROOT/name)
    snapshot_sha=sha(BASE/'inputs'/name)
    assert current_sha==snapshot_sha,name
    contracts.append({'path':name,'original_sha256':snapshot_sha,'current_sha256':current_sha,'same_bytes':True})
assert sha(OUT/'fence-1.py')==sha(BASE/'original-run/fence-1.py')
assert sha(OUT/'bootstrap.py')==sha(BASE/'original-run/bootstrap.py')
old_text=(BASE/'original-run/section.md').read_text()
new_text=(OUT/'section.md').read_text()
diff=''.join(difflib.unified_diff(old_text.splitlines(True),new_text.splitlines(True),fromfile='original-7.3-'+initial['source_sha256'],tofile='revised-7.3-'+new['source_sha256']))
(OUT/'section.diff').write_text(diff)
tok=ByteTokenizer()
messages=[{'role':'user','content':'問'},{'role':'assistant','content':'答'}]
x,y=render_chat(messages,tok)
full_ids=[tok.bos_id,tok.user_id]+tok.encode('問')+[tok.eos_id,tok.assistant_id]+tok.encode('答')+[tok.eos_id]
full_targets=[-100]*7+tok.encode('答')+[tok.eos_id]
original_target_indices=[i for i,value in enumerate(full_targets) if value!=-100]
current_effective_indices=(y!=-100).nonzero().flatten().tolist()
assert original_target_indices==[7,8,9,10]
assert current_effective_indices==[6,7,8,9]
assert x.tolist()==full_ids[:-1] and y.tolist()==full_targets[1:]
assert x[6].item()==tok.assistant_id and y[6].item()==tok.encode('答')[0]
reused=json.loads((BASE/'probes/results.json').read_text())
assert x.tolist()==reused['examples']['答']['x'] and y.tolist()==reused['examples']['答']['y']
facts={'kind':'actual_revised_section_facts_and_reuse_validation','observed_utc':datetime.now(UTC).isoformat(),'source':new['source'],'source_sha256':new['source_sha256'],'initial_source_sha256':initial['source_sha256'],'history_path':history.relative_to(ROOT).as_posix(),'history_sha256':sha(history),'new_fence_sha256':sha(OUT/'fence-1.py'),'old_fence_sha256':sha(BASE/'original-run/fence-1.py'),'fence_same_bytes':True,'bootstrap_same_bytes':True,'figure_sha256':{},'contracts':contracts,'original_artifacts_checked':artifacts,'new_cpu_alignment':{'messages':messages,'full_ids':full_ids,'full_targets':full_targets,'original_target_indices':original_target_indices,'current_effective_indices':current_effective_indices,'x':x.tolist(),'y':y.tolist(),'first_predictor_x':x[6].item(),'first_predictor_y':y[6].item(),'effective_count':int((y!=-100).sum()),'mapping':'Original full target index j is supervised at predictor index j-1; both label and mask use targets[1:] exactly once.'},'execution_scope':'This new run executes render_chat and exact-byte/hash comparison only. The prior original fence, variant/empty/multi-turn tests, mean-loss and backward proofs are REUSED after byte equality checks, not claimed rerun. No model forward/backward, optimizer, training, existing weights, data/model downloads or figure rendering in this recheck.','environment':{'python':platform.python_version(),'torch':str(torch.__version__),'torch_git_version':torch.version.git_version,'device':'cpu','cuda_build':str(torch.version.cuda),'threads':str(torch.get_num_threads()),'cwd':str(Path.cwd()),'python_executable':sys.executable}}
(OUT/'facts.json').write_text(json.dumps(facts,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(diff)
print(json.dumps(facts,ensure_ascii=False,indent=2,allow_nan=False))
