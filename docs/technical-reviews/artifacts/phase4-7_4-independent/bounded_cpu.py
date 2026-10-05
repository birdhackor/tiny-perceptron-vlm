"""Independent bounded alignment, mask, denominator and causality checks. No optimizer or data files."""
from pathlib import Path
import sys
import json
import math
import hashlib

ROOT = Path('/workspace/tiny-perceptron-vlm')
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.data import ByteTokenizer, render_chat, IGNORE
from tiny_perceptron.model import TinyLM, ModelConfig, loss_sum, masked_loss
from tiny_perceptron.attention import attention_mask

torch.set_num_threads(1)
torch.manual_seed(7405)
assert torch.version.cuda is None
tok = ByteTokenizer()
results = {}

def chat(question, answer):
    return render_chat([{'role':'user','content':question}, {'role':'assistant','content':answer}])

for q,a,expected_first,expected_answer in [('Q','A',4,73),('QQ','A',5,73),('QQ','B',5,74)]:
    x,y = chat(q,a)
    # Construct the original IDs and supervision without using render_chat.
    original = [1,3] + [b+8 for b in q.encode('utf-8')] + [2,4] + [b+8 for b in a.encode('utf-8')] + [2]
    targets = [IGNORE] * (3 + len(q.encode('utf-8'))) + [IGNORE] + [b+8 for b in a.encode('utf-8')] + [2]
    assert x.tolist() == original[:-1]
    assert y.tolist() == targets[1:]
    first = (y != IGNORE).nonzero()[0].item()
    assert first == expected_first and x[first].item() == 4 and y[first].item() == expected_answer
    assert y[first+1].item() == 2 and (y!=IGNORE).sum().item() == 2
    assert tok.encode(q) == [b+8 for b in q.encode('utf-8')]
    results[q+'/'+a] = {'original_ids':original,'original_targets':targets,'x':x.tolist(),'y':y.tolist(),
                       'first':first,'first_input_id':x[first].item(),'first_target_id':y[first].item(),
                       'valid_count':2}

x,y = chat('Q','A')
# Equal counts and equal tensor shapes still do not imply equal positions.
# A trailing ignored user message keeps assistant EOS in both wrong and right slices.
multi_x,multi_y = render_chat([{'role':'user','content':'Q'}, {'role':'assistant','content':'A'},
                              {'role':'user','content':'R'}])
raw_multi_ids = [1,3,89,2,4,73,2,3,90,2]
raw_multi_targets = [IGNORE]*5 + [73,2] + [IGNORE]*3
wrong_y = torch.tensor(raw_multi_targets[:-1])
assert multi_x.tolist() == raw_multi_ids[:-1]
assert multi_y.tolist() == raw_multi_targets[1:]
assert wrong_y.shape == multi_y.shape
assert (wrong_y!=IGNORE).sum().item() == (multi_y!=IGNORE).sum().item() == 2
wrong_first = (wrong_y!=IGNORE).nonzero()[0].item()
right_first = (multi_y!=IGNORE).nonzero()[0].item()
assert wrong_first == 5 and multi_x[wrong_first].item() == 73 and right_first == 4
results['unshifted_mask_counterexample'] = {'x':multi_x.tolist(),'correct_y':multi_y.tolist(),
    'wrong_y':wrong_y.tolist(),'both_valid_count':2,'correct_first':right_first,'wrong_first':wrong_first,
    'wrong_first_input':multi_x[wrong_first].item(),'wrong_first_target':wrong_y[wrong_first].item(),
    'construction':'user Q, assistant A, trailing ignored user R; wrong labels targets[:-1], correct targets[1:]'}

# Uniform distribution allows direct arithmetic: sum=2*ln(264), mean=ln(264), no second shift.
logits = torch.zeros(1,len(x),264, dtype=torch.float64, requires_grad=True)
total,count = loss_sum(logits,y[None])
mean = masked_loss(logits,y[None])
expected_mean = math.log(264)
assert count.item() == 2
assert abs(total.item()-2*expected_mean)<1e-12 and abs(mean.item()-expected_mean)<1e-12
mean.backward()
active = (logits.grad.abs().sum(-1)[0] > 0).nonzero().flatten().tolist()
assert active == [4,5]
assert torch.equal(logits.grad[0,:4],torch.zeros_like(logits.grad[0,:4]))
results['loss_alignment_and_denominator'] = {'logits_shape':list(logits.shape),'valid_count':count.item(),
              'sum':total.item(),'mean':mean.item(),'ln264':expected_mean,'tolerance':1e-12,
              'logit_positions_with_gradient':active,'direct_loss_ignored_positions':[0,1,2,3]}

# An untrained TinyLM only probes causal information flow, never answer quality.
model = TinyLM(ModelConfig(width=8, heads=2, layers=1, max_length=16)).eval()
allowed = attention_mask(torch.arange(len(x)),torch.arange(len(x)))[0,0]
assert allowed[4].tolist() == [True,True,True,True,True,False]
with torch.no_grad():
    base = model(x[None])['logits']
    changed_future = x.clone(); changed_future[5] = tok.encode('B')[0]
    future = model(changed_future[None])['logits']
    changed_question = x.clone(); changed_question[2] = tok.encode('R')[0]
    question = model(changed_question[None])['logits']
    prefix = model(x[None,:5])['logits']
future_difference = (base[0,:5]-future[0,:5]).abs().max().item()
prefix_difference = (base[0,:5]-prefix[0]).abs().max().item()
question_difference = (base[0,4]-question[0,4]).abs().max().item()
answer_input_difference = (base[0,5]-future[0,5]).abs().max().item()
assert future_difference == 0.0
assert prefix_difference < 1e-6
assert question_difference > 1e-6 and answer_input_difference > 1e-6
results['causal_visibility'] = {'assistant_visible_input_positions':allowed[4].nonzero().flatten().tolist(),
           'future_A_to_B_max_difference_positions_0_to_4':future_difference,
           'prefix_equivalence_max_difference':prefix_difference,'prefix_tolerance':1e-6,
           'ignored_question_Q_to_R_assistant_logits_max_difference':question_difference,
           'read_answer_position5_logits_max_difference':answer_input_difference,
           'weights_updated':False,'scope':'Random CPU model checks computation dependency only, not learning or accuracy.'}

results['environment'] = {'python':sys.version,'torch':torch.__version__, 'torch_git_version':torch.version.git_version,
    'cuda_build':str(torch.version.cuda),'device':'cpu','threads':torch.get_num_threads(),
    'seed':7405,'grad_mode_used':'synthetic loss backward only; model forward no_grad; no parameter updates'}
results['input_sha256'] = {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
   for p in ['tiny_perceptron/data.py','tiny_perceptron/model.py','tiny_perceptron/attention.py','tiny_perceptron/modern.py']}
print(json.dumps(results,ensure_ascii=False,indent=2))
