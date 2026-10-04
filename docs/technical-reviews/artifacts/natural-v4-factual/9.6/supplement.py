"""Bounded check of independent arithmetic denominator and actual EOS stop API."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
import hashlib
import json
import platform
from types import SimpleNamespace
import torch
from scripts.course_experiments.common import split_records
from scripts.course_experiments.text import arithmetic_records
from tiny_perceptron.model import generate

run = json.loads(Path('docs/course-experiments/results/safety.json').read_text())
splits = split_records(arithmetic_records(), seed=42)
counts = {k:len(v) for k,v in splits.items()}
assert counts == {'train':49,'validation':8,'test':7}
assert all(v['arithmetic']['test']['records'] == 7 for v in run['results']['runs'].values())
path = Path('tiny_perceptron/model.py')
source_sha = hashlib.sha256(path.read_bytes()).hexdigest()
assert source_sha == run['code_sha256'][str(path)]

class FixedOutput(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(max_length=16)
        self.calls = 0
    def forward(self, ids, cache=None):
        self.calls += 1
        logits = torch.full((*ids.shape,264), -10.0)
        logits[:,-1,56 if self.calls==1 else 2] = 10.0
        return {'logits':logits,'cache':None}

model = FixedOutput()
prompt = torch.tensor([[1,3,48,2,4]])
out = generate(model,prompt,max_new_tokens=5,temperature=0.0,eos_id=2,use_cache=False)
assert out.tolist() == [[1,3,48,2,4,56,2]]
assert model.calls == 2
assert model.training
print(json.dumps({'python':platform.python_version(),'torch':torch.__version__,'device':'cpu',
 'arithmetic_split_counts':counts,'original_arithmetic_test_records_each_variant':7,
 'model_source_sha256':source_sha,'single_sequence_EOS_probe':{
  'input':prompt.tolist(),'output':out.tolist(),'requested_max_new_tokens':5,
  'observed_forward_calls':model.calls,'eos_id':2,'temperature':0.0,'use_cache':False,
  'expected_new_tokens':[56,2],'training_mode_restored':model.training},
 'scope':'CPU deterministic stub exercises repository generate stopping behavior; no trained model inference or quality measurement.'},ensure_ascii=False,indent=2))
print('ALL_SUPPLEMENT_ASSERTIONS_PASSED')
