"""B.6 independent bounded CPU check; no model inference scores or saved weights."""
import hashlib
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.data import ByteTokenizer, render_chat, pad_batch
from tiny_perceptron.model import loss_sum, masked_loss
from scripts.course_experiments.common import Context, new_lm, records_sha256, text_examples
from scripts.course_experiments.applications import _prompt_ids
from scripts.course_experiments.tool_choice import build_records, split_records, parse_action, summarize

torch.set_num_threads(1)
assert torch.version.cuda is None
OUT = Path(__file__).resolve().parent
ORIGINAL = OUT.parent / 'original'
raw = json.loads((ORIGINAL / 'results.json').read_bytes())
published = json.loads((ORIGINAL / 'published-tool-choice.json').read_bytes())
dataset = json.loads((ORIGINAL / 'dataset.json').read_bytes())
assert published['results'] == raw
tok = ByteTokenizer()
messages = [
    {'role':'system','content':'精確計算用工具，解釋或照抄直接回答；缺資訊或工具先求助。計算器可用。'},
    {'role':'user','content':'1加2等於多少？'},
    {'role':'assistant','content':'TOOL'},
]
x, y = render_chat(messages, tok)
assert len(x) == len(y) == 132
assert y[y != -100].tolist() == tok.encode('TOOL') + [tok.eos_id]
assert x[(y != -100).nonzero()[0]].item() == tok.assistant_id
assert tok.decode(y[y != -100].tolist()) == 'TOOL'
assert tok.encode(messages[1]['content']) == [b+8 for b in messages[1]['content'].encode('utf8')]
variants = []
for action in ['TOOL', 'DIRECT', 'ASK']:
    variant = [dict(m) for m in messages]
    variant[1]['content'] = '請照抄：é、數字3。'
    variant[2]['content'] = action
    vx, vy = render_chat(variant, tok)
    expected_ids = tok.encode(action) + [tok.eos_id]
    assert vy[vy != -100].tolist() == expected_ids
    # Reconstruct the exact aligned targets independently of render_chat.
    seq, labels = [tok.bos_id], [-100]
    role_ids = {'system':tok.system_id,'user':tok.user_id,'assistant':tok.assistant_id}
    for m in variant:
        content = [b+8 for b in m['content'].encode('utf8')] + [tok.eos_id]
        seq += [role_ids[m['role']]] + content
        labels += [-100] + (content if m['role']=='assistant' else [-100]*len(content))
    assert vx.tolist() == seq[:-1] and vy.tolist() == labels[1:]
    assert tok.encode(variant[1]['content']) == vx[seq.index(tok.user_id)+1:seq.index(tok.user_id)+1+len(tok.encode(variant[1]['content']))].tolist()
    variants.append({'action':action,'effective_targets':len(expected_ids),'prompt_bytes_retained':True})
for bad in [[{'role':'user','content':'沒有答案'}],[{'role':'tool','content':'3'}]]:
    try:
        render_chat(bad, tok)
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid chat accepted')

logits = torch.zeros((1,len(y),tok.vocab_size),dtype=torch.float64,requires_grad=True)
loss, n = loss_sum(logits,y[None])
mean = masked_loss(logits,y[None])
assert int(n)==5 and math.isclose(float(mean.detach()),math.log(264),abs_tol=1e-12)
mean.backward()
assert torch.equal(logits.grad[0,y == -100],torch.zeros_like(logits.grad[0,y == -100]))
changed = logits.detach().clone()
changed[0,y == -100,:] = torch.arange(tok.vocab_size,dtype=torch.float64)
assert float(masked_loss(changed,y[None])) == float(mean.detach())

splits = split_records(build_records(),42)
test_families = {r['family'] for r in splits['test']}
splits['paraphrase_diagnostic'] = [r for r in build_records(paraphrase=True) if r['family'] in test_families]
assert splits == dataset
dataset_audit = {}
for name, rows in dataset.items():
    assert records_sha256(rows) == raw['data'][name]['sha256']
    assert len(rows)==raw['data'][name]['records']
    counts=Counter(r['expected_action'] for r in rows)
    assert dict(counts)==raw['data'][name]['action_counts']
    for r in rows:
        assert r['messages'][0]['content'] in ('計算器可用。','計算器停用。')
        expected = ('TOOL' if r['calculator_available'] else 'ASK') if r['intent']=='numerical_addition' else ('ASK' if r['intent']=='missing_quantity' else 'DIRECT')
        assert r['expected_action'] == r['messages'][-1]['content'] == expected
        assert not r['messages'][1]['content'].startswith(('CALC','COPY'))
    examples = text_examples(rows,mode='sft',max_length=160)
    assert all(len(ix)==len(iy) for ix,iy in examples)
    assert all(int((iy != -100).sum())==len(tok.encode(row['expected_action']))+1 for row,(_,iy) in zip(rows,examples,strict=True))
    dataset_audit[name] = {'records':len(rows),'families':len({r['family'] for r in rows}),'actions':dict(counts),'sha256':records_sha256(rows),'target_tokens_one_pass':sum(int((iy != -100).sum()) for _,iy in examples)}
assert not ({r['family'] for r in dataset['train']} & test_families)
lengths = [len(tok.encode(r['expected_action']))+1 for r in dataset['train']]
sampler = random.Random(42)
effective_tokens = sum(sum(sampler.choices(lengths,k=24)) for _ in range(900))
assert effective_tokens == raw['training']['effective_tokens'] == 121297
assert raw['training']['records_sha256'] == records_sha256(dataset['train'])
assert raw['training']['steps'] == 900

sample_audit = {}
for split in ['validation','test','paraphrase_diagnostic']:
    rows,preds = [],[]
    samples = raw[split]['samples']
    for item in samples:
        row = item['record']; gen = item['generation']; sample = gen['samples'][0]
        assert gen['messages'] == row['messages'][:-1]
        assert gen['input_ids'] == _prompt_ids(row['messages'][:-1],tok)
        assert gen['input_tokens'] == len(gen['input_ids'])
        ids = sample['generated_ids']; eos = bool(ids and ids[-1]==tok.eos_id)
        body = ids[:-1] if eos else ids
        assert sample['eos']==eos and sample['generated']==tok.decode(body)
        assert sample['invalid_special_tokens']==[i for i in body if i<8]
        assert sample['generated_tokens']==len(ids)
        pred = parse_action(sample)
        assert pred==item['chosen_action']
        rows.append(row);preds.append(pred)
    assert rows == dataset[split]
    metrics = summarize(rows,preds)
    assert metrics == raw[split]['metrics']
    sample_audit[split] = {'sample_count':len(samples),'accuracy':metrics['accuracy'],'selected_actions':dict(Counter(preds)),'only_action_targets':True}

ctx=Context('cpu',OUT,OUT,ROOT/'assets',42)
full_config_model = new_lm(ctx,width=64,layers=2,heads=2,max_length=160)
parameter_count = sum(p.numel() for p in full_config_model.parameters())
assert parameter_count==raw['model']['parameters']==raw['training']['trainable_parameters']==143616
assert all(p.requires_grad for p in full_config_model.parameters())
del full_config_model
model = new_lm(ctx,width=16,layers=1,heads=2,max_length=160)
bx,by,valid=pad_batch([(x,y)])
optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.003)
before=[p.detach().clone() for p in model.parameters()]
initial=float(masked_loss(model(bx,valid=valid)['logits'],by).detach())
optimizer.zero_grad(set_to_none=True)
result=model(bx,valid=valid)
objective=masked_loss(result['logits'],by)+.01*result['auxiliary']
objective.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
unchanged_after_backward = all(torch.equal(a,p) for a,p in zip(before,model.parameters(),strict=True))
assert unchanged_after_backward
optimizer.step()
changed_parameters=sum(not torch.equal(a,p) for a,p in zip(before,model.parameters(),strict=True))
final=float(masked_loss(model(bx,valid=valid)['logits'],by).detach())
assert changed_parameters>0 and final<initial

checkpoint=ROOT/'outputs/tool-choice-development/fixed-900/tool-choice.pt'
checkpoint_hash=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
assert checkpoint_hash==raw['frozen_evaluation']['checkpoint_sha256']
script_hash=hashlib.sha256((ROOT/'scripts/course_experiments/tool_choice.py').read_bytes()).hexdigest()
assert script_hash==raw['reproducibility']['script_sha256']
summary = {
    'environment':{'python':sys.version,'torch':str(torch.__version__),'torch_git_version':str(torch.version.git_version),'device':'cpu','threads':torch.get_num_threads(),'cuda_build':str(torch.version.cuda)},
    'original_example':{'input_positions':len(x),'effective_targets':int(n),'target_ids':y[y != -100].tolist(),'decoded':'TOOL','last_target_eos':True},
    'variants':variants,
    'loss_check':{'sum':float(loss.detach()),'effective_denominator':int(n),'mean':float(mean.detach()),'expected_uniform_mean':math.log(264),'tolerance':1e-12,'ignored_logits_gradient_zero':True,'changing_ignored_logits_preserves_loss':True},
    'dataset_audit':dataset_audit,
    'schedule_count_replay':{'steps':900,'batch_size':24,'effective_tokens':effective_tokens,'purpose':'Pure label-count replay, no training'},
    'sample_audit':sample_audit,
    'parameters':{'original_config_count':parameter_count,'all_trainable':True,'historical_checkpoint_sha256':checkpoint_hash,'historical_script_sha256':script_hash,'no_checkpoint_copied':True},
    'bounded_one_step':{'width':16,'layers':1,'batch_records':1,'optimizer_steps':1,'loss_before':initial,'loss_after':final,'changed_parameter_tensors':changed_parameters,'backward_alone_kept_weights':unchanged_after_backward,'no_weights_saved':True,'scope':'Demonstrates supervised update only; no accuracy or generalization score'},
    'json_inspection':{'raw':['/training','/model','/data','/reproducibility','/frozen_evaluation/checkpoint_sha256','/validation/samples','/test/samples','/paraphrase_diagnostic/samples','/validation/metrics','/test/metrics','/paraphrase_diagnostic/metrics'],'dataset':['/train','/validation','/test','/paraphrase_diagnostic'],'excluded':['/limitations','/policy','author result interpretations']},
}
(OUT/'audit-results.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False,indent=2))
