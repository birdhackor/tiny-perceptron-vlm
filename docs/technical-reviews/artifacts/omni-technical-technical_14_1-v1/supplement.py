"""Small no-training checks for CLI parsing and input sensitivity."""
import contextlib
import hashlib
import io
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.course_experiments import run
from tiny_perceptron.modern import rope

stdout = io.StringIO()
stub = {'experiment_id':'modern','elapsed_seconds':0.0,'evidence_status':'mocked_no_training','artifacts':[]}
with patch.object(sys, 'argv', ['run', '--experiment', 'modern', '--device', 'cuda']):
    with patch.object(run, 'execute', return_value=stub) as execute:
        with contextlib.redirect_stdout(stdout):
            run.main()
assert execute.call_count == 1
args, kwargs = execute.call_args
assert args[0:2] == ('modern', 'cuda') and kwargs['step_scale'] == 1.0

def swapped_inputs(x, positions):
    d = x.shape[-1]
    angle = (positions[:, None] * (10000 ** (-torch.arange(0, d, 2) / d))[None, :])[None, None]
    cosine, sine = angle.cos(), angle.sin()
    a, b = x[..., 1::2], x[..., 0::2]
    return torch.stack((a * cosine - b * sine, a * sine + b * cosine), -1).flatten(-2)

torch.manual_seed(141)
ones = torch.ones(1, 1, 16, 8)
check = torch.randn(2, 4, 16, 8)
positions = torch.arange(16)
ones_equal = torch.equal(swapped_inputs(ones, positions), rope(ones, positions))
random_equal = torch.equal(swapped_inputs(check, positions), rope(check, positions))
assert ones_equal and not random_equal
raw_scores = torch.tensor([-0.7071, 0.0])
weights = raw_scores.softmax(0)
assert bool((weights >= 0).all()) and float(weights.sum()) == 1.0
logits = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
labels = torch.tensor([0, 0])
losses = F.cross_entropy(logits, labels, reduction='none')
direct = -logits.log_softmax(-1)[:, 0]
assert torch.equal(losses, direct)
output = {
    'created_at':datetime.now(UTC).isoformat(),
    'environment':{'python':sys.version,'torch':str(torch.__version__),'device':'cpu'},
    'cli':{'argv':['--experiment','modern','--device','cuda'],
           'execute_arguments':[str(x) for x in args], 'execute_kwargs':kwargs,
           'stdout':stdout.getvalue().strip(),
           'scope':'execute replaced with a spy; argument parsing only. No CUDA operations or training.'},
    'swapping_adjacent_input_coordinates':{'all_ones_equal':ones_equal,'random_equal':random_equal,
                                           'scope':'One concrete mistake invisible on all-ones inputs, detected by random nonidentical coordinates.'},
    'softmax':{'raw_scores':raw_scores.tolist(),'weights':weights.tolist(),'sum':float(weights.sum())},
    'nll_definition':{'cross_entropy':losses.tolist(),'negative_log_target_probability':direct.tolist(),
                       'scope':'Definition check on two toy class-index targets; no performance claim.'},
    'status':'passed',
}
path = Path(__file__).with_name('supplement-output.json')
with path.open('x', encoding='utf-8') as stream:
    json.dump(output, stream, ensure_ascii=False, indent=2)
    stream.write('\n')
print(json.dumps({'status':'passed','output':str(path.relative_to(ROOT)),
                  'sha256':hashlib.sha256(path.read_bytes()).hexdigest()},ensure_ascii=False))
