"""Personally run revised raw fence and exact exercise call changes in one model."""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[6]
RUN=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
import torch
torch.set_num_threads(1)
torch.set_default_device('cpu')
assert torch.version.cuda is None and not torch.cuda.is_available()
raw=(RUN/'original-run/fence-1.py').read_bytes()
assert b'assert torch.allclose(base, actual, atol=1e-6, rtol=0.0)' in raw
ns={'__name__':'__main__'}
out=io.StringIO()
with redirect_stdout(out): exec(compile(raw,'revised 7.7 raw original fence','exec'),ns)
model=ns['model']; base=ns['base']
def param_hash(): return hashlib.sha256(b''.join(p.detach().contiguous().numpy().tobytes() for p in model.parameters())).hexdigest()
before=param_hash()
tail=raw[raw.index(b'actual = '):]
variants={
    'remove-valid':tail.replace(b'valid=valid, ',b''),
    'remove-positions':tail.replace(b', positions=positions',b''),
}
results=[]
for name,code in variants.items():
    (RUN/'code'/('exercise-'+name+'.py')).write_bytes(code)
    captured=io.StringIO(); failed=False
    try:
        with redirect_stdout(captured): exec(compile(code,name+'; same model from revised fence','exec'),ns)
    except AssertionError: failed=True
    maximum=(base-ns['actual']).abs().max().item()
    assert failed and maximum > 1e-6
    results.append({'variant':name,'same_model':True,'exact_code_sha256':hashlib.sha256(code).hexdigest(),
        'stdout':captured.getvalue(),'assertion_failed_as_predicted':failed,'max_abs':maximum,
        'current_pure_atol_allclose':torch.allclose(base,ns['actual'],atol=1e-6,rtol=0.0)})

# Zero baseline avoids rounding ambiguity at the exact absolute threshold.
boundary=[]
for d,expect in [(0.0,True),(0.5e-6,True),(1e-6,True),(1.000001e-6,False),(2e-6,False)]:
    a=torch.tensor([0.0],dtype=torch.float64); b=torch.tensor([d],dtype=torch.float64)
    got=torch.allclose(a,b,atol=1e-6,rtol=0.0)
    assert got == expect
    boundary.append({'input':0.0,'other':d,'actual_abs_difference':(a-b).abs().item(),'expected':expect,'observed':got})
a=torch.tensor([1.0],dtype=torch.float64); b=torch.tensor([1.000002],dtype=torch.float64)
counterexample={'actual_abs_difference':(a-b).abs().item(),
    'preserved_old_call_allclose':torch.allclose(a,b,atol=1e-6),
    'revised_call_allclose':torch.allclose(a,b,atol=1e-6,rtol=0.0)}
assert counterexample['preserved_old_call_allclose'] and not counterexample['revised_call_allclose']
after=param_hash()
assert before==after and all(p.grad is None for p in model.parameters())
print(json.dumps({'environment':{'python':sys.version,'torch':str(torch.__version__),'torch_git_version':str(torch.version.git_version),
    'cuda_build':str(torch.version.cuda),'cuda_available':str(torch.cuda.is_available()),'device':'cpu','threads':str(torch.get_num_threads())},
    'revised_fence_sha256':hashlib.sha256(raw).hexdigest(),'revised_fence_stdout':out.getvalue(),
    'exercise_variants':results,'pure_atol_boundary':boundary,'preserved_counterexample_rechecked':counterexample,
    'model_parameters_before_sha256':before,'model_parameters_after_sha256':after,
    'parameters_updated':False,'backward_called':False,'old_proofs_overwritten':False},ensure_ascii=False,indent=2,allow_nan=False))
