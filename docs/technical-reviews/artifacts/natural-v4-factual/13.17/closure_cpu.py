"""Same-owner actual recheck of changed necessary prerequisites, no training rerun."""
from pathlib import Path
import hashlib
import io
import json
import platform
import re
from contextlib import redirect_stdout
import torch

ROOT=Path(__file__).resolve().parents[5]
ART=Path(__file__).resolve().parent
def h(b):return hashlib.sha256(b).hexdigest()
def part(file,ID):
    b=(ROOT/file).read_bytes();ms=list(re.finditer(rb'^## ',b,re.M))
    return next(b[m.start():ms[i+1].start() if i+1<len(ms) else len(b)] for i,m in enumerate(ms)
                if b[m.start():].startswith(('## '+ID+' ').encode()))
original=json.loads((ART/'prerequisite-receipt.json').read_text())
current=[]
for old in original:
    file,ID=old['source'].split('#');b=part(file,ID)
    current.append({'source':old['source'],'sha256':h(b),'bytes':len(b),
                    'changed_from_first_read':h(b)!=old['sha256']})
    if h(b)!=old['sha256']:
        assert ID in ['13.15','13.16']
        assert b==(ART/('closure-current-prerequisite-'+ID+'.md')).read_bytes()
current.append({'source':'course/chapters/07.md#7.17','sha256':h(part('course/chapters/07.md','7.17')),
   'bytes':len(part('course/chapters/07.md','7.17')),'added_necessary_prerequisite':True})
(ART/'closure-prerequisite-receipt.json').write_text(json.dumps(current,ensure_ascii=False,indent=2)+'\n')
new15=part('course/chapters/13.md','13.15').decode();old15=(ART/'prerequisite-13.15.md').read_text()
block=lambda t:re.search(r'```python\n(.*?)\n```',t,re.S).group(1)
assert block(new15)==block(old15)
ns={};stream=io.StringIO();torch.set_num_threads(2)
with redirect_stdout(stream):exec(compile(block(new15),'current13.15-closure','exec'),ns)
stdout=stream.getvalue();print(stdout,end='')
assert ns['action'].item()==0 and ns['terms']['ratio'].item()==1.0
assert not torch.equal(ns['before'],next(ns['policy'].parameters()))
assert not any(p.grad is not None for p in ns['reward_model'].parameters())
assert not any(p.grad is not None for p in ns['reference'].parameters())
assert list(ns['context'].shape)==list(ns['new_logits'].shape)==[1,4]
report=json.loads((ROOT/'docs/technical-reviews/13.17.json').read_text())
assert h(part('course/chapters/13.md','13.17'))==report['source_sha256']==h((ART/'original-section.md').read_bytes())
audit=json.loads((ART/'audit-result.json').read_text())
for file,digest in audit['source_fingerprints'].items():assert h((ROOT/file).read_bytes())==digest
for name in ['ppo','dpo']:
    assert audit['independent_results']['test'][name]['total']==[12,18]
    assert [audit['independent_results']['test'][name]['by_mode'][m] for m in ['number','explain','missing']]==[[6,6],[0,6],[6,6]]
out={'reviewer_task':'/root/v4_review_coordinator/factual_v4_13_17',
 'actual_full_reread':['13.15','13.16','7.17'],
 'environment':{'python':platform.python_version(),'torch':str(torch.__version__),'device':'cpu','threads':str(torch.get_num_threads())},
 'affected_claims':['c1','c11','c12','c13','c15','c16','c18','c22'],
 'changed_scope':'13.15 explicitly calls the finite random-start stage supervised demonstration training borrowing the SFT teaching method;13.16 calls its baseline a card network rather than an SFT model.7.17 explains pretrained model→demonstration continuation. No experiment code/configuration/outcome or13.17 text changed.',
 'authority_rechecked':'InstructGPT2203.02155v1 §3.1 explicitly starts from a pretrained language model, then demonstrations. DPO2305.18290v3 §3 uses initial SFT/ref while the lesson toy substitutes a stated finite demonstration baseline.',
 'current13_15_python_block_identical':True,'current13_15_actual_stdout':stdout,
 'current13_17_sha256':report['source_sha256'],'chapter13_fullfile_sha256':h((ROOT/'course/chapters/13.md').read_bytes()),
 'experiment_code_fingerprints_unchanged':True,'preserved_cpu_results_still_applicable':True,
 'source_and_failed_probes_preserved':True,
 'conclusion':'No new unresolved or contradicted13.17 fact. The finite SFT/checkpoint labels denote its demonstration-teaching analogue, not a pretrained/token LM. Full current prerequisite scope and unchanged actual output were personally rechecked; pass retained.'}
(ART/'closure-result.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(out,ensure_ascii=False,indent=2))
