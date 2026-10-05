"""Execute current original fence, revised exercise and bounded late-freeze variant."""
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

OUT=Path(__file__).resolve().parent
INITIAL=OUT.parent
ROOT=INITIAL.parents[3]
sha=lambda b:hashlib.sha256(b).hexdigest()
body=(OUT/'section.md').read_bytes()
assert sha(body)=='3b28eb0efe72d07040ea2063a3f2da379230c7f8c56cb9529a80be6d374921d0'
fences=re.findall(rb'```python\n(.*?)```',body,re.S)
assert len(fences)==1
fence=fences[0]
assert fence==(INITIAL/'fence-1.py').read_bytes()
(OUT/'fence-current.py').write_bytes(fence)

diag='print("可訓練總數", sum(p.numel() for p in model.parameters() if p.requires_grad))\nprint("loss.requires_grad", loss.requires_grad)\n'.encode()
early=fence.replace(b'model.image_projector.requires_grad_(True)',b'model.image_projector.requires_grad_(False)',1)
early=early.replace(b'loss.backward()\n',diag+b'loss.backward()\n',1)
(OUT/'exercise-freeze-before-forward.py').write_bytes(early)

# Only the late-freeze variant changes the original terminal gradient diagnostic,
# because a frozen parameter's .grad is None and cannot have .norm() called on it.
late=fence.replace(b'out = model(ids, labels, image=scene())\n',b'out = model(ids, labels, image=scene())\nmodel.image_projector.requires_grad_(False)\n',1)
late=late.replace(b'loss.backward()\n',diag+'loss.backward()\nprint("backward已完成", True)\n'.encode(),1)
late=late.replace('print("接頭收到梯度", model.image_projector.weight.grad.norm().item() > 0)'.encode(),'print("接頭未累積梯度", model.image_projector.weight.grad is None)'.encode(),1)
(OUT/'variant-freeze-after-forward.py').write_bytes(late)

env={**os.environ,'CUDA_VISIBLE_DEVICES':'','HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(ROOT)}
checks=[]
for name,exit_expected in [('fence-current.py',0),('exercise-freeze-before-forward.py',1),('variant-freeze-after-forward.py',0)]:
    stdout_name=name.replace('.py','.stdout.txt')
    stderr_name=name.replace('.py','.stderr.txt')
    command=[str(ROOT/'.venv/bin/python'),str(OUT/name)]
    with (OUT/stdout_name).open('wb') as stdout,(OUT/stderr_name).open('wb') as stderr:
        result=subprocess.run(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr,timeout=30,check=False)
    stdout=(OUT/stdout_name).read_text()
    stderr=(OUT/stderr_name).read_text()
    assert result.returncode==exit_expected
    if name.startswith('exercise-'):
        assert stdout=='可訓練總數 0\nloss.requires_grad False\n'
        assert 'RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn' in stderr
    elif name.startswith('variant-'):
        assert stdout=='可訓練總數 0\nloss.requires_grad True\nbackward已完成 True\n接頭未累積梯度 True\n語言權重未累積梯度 True\n'
        assert stderr==''
    else:
        assert stdout=='接頭收到梯度 True\n語言權重未累積梯度 True\n'
        assert stderr==''
    checks.append({'code':str((OUT/name).relative_to(ROOT)),'code_sha256':sha((OUT/name).read_bytes()),'argv':command,'cwd':str(ROOT),'exit_code':result.returncode,'expected_exit_code':exit_expected,'stdout_path':str((OUT/stdout_name).relative_to(ROOT)),'stdout_sha256':sha((OUT/stdout_name).read_bytes()),'stderr_path':str((OUT/stderr_name).relative_to(ROOT)),'stderr_sha256':sha((OUT/stderr_name).read_bytes()),'stdout':stdout,'expected_failure_matched':name.startswith('exercise-'),'scope':'Original bytes unchanged for fence-current; exercise only replaces True→False and adds the requested pre-backward prints; late variant freezes immediately after out and replaces original gradient norm print with None diagnostic.'})

# Hash verification is not re-execution of the initial evidence.
initial=json.loads((INITIAL/'initial-review.json').read_bytes())
reused=[]
for typ,rows in [('artifact',initial['artifacts']),('repository_source',[s for s in initial['sources'] if s['kind']=='repository_code'])]:
    for row in rows:
        path=ROOT/row['path'];observed=sha(path.read_bytes())
        assert observed==row['sha256'],row['path']
        reused.append({'kind':typ,'id':row['id'],'path':row['path'],'sha256':observed,'same_bytes':True,'rerun_during_recheck':False})
(OUT/'reused-evidence-hash-receipt.json').write_text(json.dumps(reused,ensure_ascii=False,indent=2)+'\n')
info=subprocess.run([str(ROOT/'.venv/bin/python'),'-c','import json,sys,torch; print(json.dumps({"python":sys.version,"torch":str(torch.__version__),"torch_git":torch.version.git_version,"device":"cpu","cuda_build":str(torch.version.cuda),"cuda_available":str(torch.cuda.is_available())}))'],cwd=ROOT,env=env,text=True,capture_output=True,check=True)
environment=json.loads(info.stdout)
assert environment['cuda_build']=='None' and environment['cuda_available']=='False'
receipt={'reviewer_task':'/root/phase4_factual_coordinator/factual_11_3','source_sha256':sha(body),'original_source_sha256':initial['source_sha256'],'date':'2026-10-05','checks':checks,'environment':environment,'nonsecret_runtime_overrides':{k:env[k] for k in ['CUDA_VISIBLE_DEVICES','HF_HUB_OFFLINE','HF_DATASETS_OFFLINE','TRANSFORMERS_OFFLINE','OMP_NUM_THREADS','MKL_NUM_THREADS','PYTHONDONTWRITEBYTECODE','PYTHONPATH']},'scope':'Three separate fresh CPU processes; no optimizer step, training, old model evaluation, model download or .pt saving. Initial historical/scientific/visual evidence reused only after byte-hash verification. Initial screenshots show previous paragraph version, not the new exercise.'}
(OUT/'execution-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(receipt,ensure_ascii=False,indent=2))
