"""Preserve actual rerun and verify the precise allowed reuse of original proofs."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[6]
RUN=Path(__file__).resolve().parent.parent
ART=RUN.parent
TMP=Path('/tmp/phase4-7_7-recheck-1')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
(RUN/'original-run').mkdir(exist_ok=True)
for name in ('section.md','fence-1.py','bootstrap.py','extraction.json','execution.json','environment.json','stdout.txt','stderr.txt'):
    shutil.copyfile(TMP/name,RUN/'original-run'/name)
old=(ART/'original-run/section.md').read_bytes()
new=(RUN/'original-run/section.md').read_bytes()
assert old.replace(b'assert torch.allclose(base, actual, atol=1e-6)',b'assert torch.allclose(base, actual, atol=1e-6, rtol=0.0)') == new
reuse=[]
for name in ('tiny_perceptron/model.py','tiny_perceptron/attention.py','tiny_perceptron/data.py','tiny_perceptron/modern.py','scripts/build_course.py','docs/review-tools/section_facts.py','course/figures/rewrite-07-07-padding-positions.svg'):
    previous=ART/'inputs'/name; current=ROOT/name
    matches=sha(previous)==sha(current)
    assert matches
    reuse.append({'path':name,'previous_snapshot_sha256':sha(previous),'current_sha256':sha(current),'same_bytes':matches})
(RUN/'reuse-verification.json').write_text(json.dumps({'old_source_sha256':sha(ART/'original-run/section.md'),
    'revised_source_sha256':sha(RUN/'original-run/section.md'),'only_change':'assert adds rtol=0.0; all other section UTF-8 bytes identical',
    'checked_files':reuse,'source_contract_reread':'Official allclose _torch_docs.py lines766–795 and installed2.14.1+cpu __doc__ lines69–98 personally reread; actual model.forward and attention_mask/manual_attention reread.',
    'legal_reuse_scope':'Unchanged concept claims, previous own source snapshots, key/query/denominator/loss mechanism checks, and actual viewed figure rendering reused after checksum checks. Revised fence, exercises and tolerance boundary are NEW executions. No old review verdict borrowed.'},ensure_ascii=False,indent=2)+'\n')
env=os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
argv=[str(ROOT/'.venv/bin/python'),str(RUN/'code/recheck_probe.py')]
p=subprocess.run(argv,cwd=ROOT,env=env,capture_output=True,timeout=45,check=False)
(RUN/'probe-stdout.json').write_bytes(p.stdout); (RUN/'probe-stderr.txt').write_bytes(p.stderr)
receipt={'revised_original_fence_command':'.venv/bin/python docs/review-tools/section_facts.py course/chapters/07.md#7.7 --output /tmp/phase4-7_7-recheck-1 --execute --timeout 45',
    'revised_original_fence_exit_code':json.loads((RUN/'original-run/execution.json').read_text())['exit_code'],
    'probe_command_argv':argv,'probe_exit_code':p.returncode,'cwd':str(ROOT),'timeout_seconds':45,'probe_stdout_sha256':sha(RUN/'probe-stdout.json'),'probe_stderr_sha256':sha(RUN/'probe-stderr.txt')}
(RUN/'commands.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(receipt,ensure_ascii=False,indent=2))
if p.returncode: print(p.stderr.decode()); raise SystemExit(p.returncode)
