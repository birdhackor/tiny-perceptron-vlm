"""Independent, deterministic CPU checks; no optimizer, training, weights or data download."""
import hashlib
import json
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.multimodal import expand_modalities
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.model import TinyLM, ModelConfig, loss_sum, masked_loss
import tiny_perceptron.attention as attention

torch.set_num_threads(1)
torch.manual_seed(42)
assert torch.version.cuda is None and not torch.cuda.is_available()
embedding = nn.Embedding(264, 8)
ids = torch.tensor([1, 5, 4, 73, 2])
labels = torch.tensor([-100, -100, -100, 73, 2])
tok = ByteTokenizer()
assert (tok.bos_id, tok.image_id, tok.assistant_id, tok.eos_id) == (1, 5, 4, 2)
assert b'A'[0] == 65 and tok.encode('A') == [73]
rows = []
for m in (1, 3, 5):
    features = torch.arange(m * 8, dtype=torch.float32).reshape(m, 8) / 10
    x, y = expand_modalities(ids, labels, embedding, {5: features}, {5})
    expanded = torch.cat((embedding(ids[:1]), features, embedding(ids[2:])))
    expanded_targets = [-100] * (m + 2) + [73, 2]
    assert torch.equal(x[0], expanded[:-1])
    assert x.shape == (1, m + 3, 8)
    assert y.tolist() == [expanded_targets[1:]]
    assert torch.equal(x[0, m+1], embedding(ids[2]))
    assert torch.equal(x[0, m+2], embedding(ids[3]))
    assert y[0, m+1].item() == 73 and y[0, m+2].item() == 2
    positions = torch.arange(x.shape[1])
    mask = attention.attention_mask(positions, positions)
    assert mask.shape == (1, 1, m+3, m+3)
    assert mask[0, 0, m+1, :m+2].all() and not mask[0, 0, m+1, m+2]
    rows.append(dict(feature_rows=m, expanded_rows=m+4, input_shape=list(x.shape),
                     expanded_targets=expanded_targets, shifted_targets=y.tolist(),
                     assistant_input_index=m+1, answer_input_index=m+2,
                     positions=positions.tolist(), mask_shape=list(mask.shape),
                     effective_target_count=int((y!=-100).sum())))

torch.manual_seed(42)
model = TinyLM(ModelConfig(width=8, heads=1, max_length=16)).eval()
features = torch.randn(3, 8, requires_grad=True)
x, y = expand_modalities(ids, labels, model.embedding, {5: features}, {5})
positions_read, masks_read = [], []
hook = model.position.register_forward_pre_hook(lambda module,args: positions_read.append(args[0].tolist()))
original_mask = attention.attention_mask
def record_mask(q,k,valid=None,segments=None):
    result=original_mask(q,k,valid,segments)
    masks_read.append(dict(query=q.tolist(),key=k.tolist(),shape=list(result.shape),allowed=result.int().tolist()))
    return result
attention.attention_mask=record_mask
logits = model(embeddings=x)['logits']
hook.remove()
attention.attention_mask=original_mask
logits.retain_grad()
total,count = loss_sum(logits,y)
mean = masked_loss(logits,y)
assert count.item()==2 and torch.equal(mean,total/count)
mean.backward()
assert torch.equal(logits.grad[0,:4], torch.zeros_like(logits.grad[0,:4]))
assert features.grad.norm().item()>0
assert positions_read == [[0,1,2,3,4,5]]
assert masks_read[0]['shape'] == [1,1,6,6]
with torch.no_grad():
    baseline = model(embeddings=x.detach())['logits']
    answer_changed=x.detach().clone();answer_changed[0,5]+=torch.arange(8,dtype=torch.float32)*5
    answer_logits=model(embeddings=answer_changed)['logits']
    image_changed=x.detach().clone();image_changed[0,1]+=torch.arange(8,dtype=torch.float32)*5
    image_logits=model(embeddings=image_changed)['logits']
    future_delta=(baseline[0,4]-answer_logits[0,4]).abs().max().item()
    image_delta=(baseline[0,4]-image_logits[0,4]).abs().max().item()
    answer_self_delta=(baseline[0,5]-answer_logits[0,5]).abs().max().item()
    assert future_delta==0 and image_delta>0 and answer_self_delta>0
result=dict(environment=dict(python=sys.version,torch=str(torch.__version__),torch_git_version=torch.version.git_version,
                 device='cpu',cuda_build=str(torch.version.cuda),threads=torch.get_num_threads()),
            tokenizer=dict(ascii_byte=65,byte_offset=8,A_id=73,vocab_size=264),
            mapping_cases=rows,forward=dict(logits_shape=list(logits.shape),positions=positions_read,masks=masks_read),
            loss=dict(sum=total.item(),effective_target_denominator=count.item(),mean=mean.item(),
                      ignored_logits_gradient_max=logits.grad[0,:4].abs().max().item(),
                      image_feature_gradient_norm=features.grad.norm().item(),
                      image_feature_row_gradient_norms=features.grad.norm(dim=1).tolist()),
            perturbations=dict(future_answer_change_at_assistant_max_delta=future_delta,
                               image_change_at_assistant_max_delta=image_delta,
                               answer_change_at_answer_max_delta=answer_self_delta),
            scope='A single deterministic forward/backward and bounded input variants; no optimizer step, training or performance evaluation.')
print(json.dumps(result,indent=2,allow_nan=False))
