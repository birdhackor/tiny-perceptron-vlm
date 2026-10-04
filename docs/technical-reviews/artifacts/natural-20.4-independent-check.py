"""Independent CPU technical checks; no pretrained weights or GPU are loaded."""
import ast
import hashlib
import json
import math
import platform
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import torch
from torch import nn
import peft
import transformers
from accelerate import init_empty_weights
from transformers import AutoProcessor, Qwen3VLConfig, Qwen3VLForConditionalGeneration
from transformers.loss.loss_utils import ForCausalLMLoss
from peft import LoraConfig, TaskType, get_peft_model
from tiny_perceptron.alignment import LoRALinear
from tiny_perceptron import natural_assistant as a

OUT = ROOT / 'docs/technical-reviews/artifacts'
def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
    return json.loads((ROOT / p).read_text())

result = {'environment': {'python': platform.python_version(), 'torch': str(torch.__version__),
          'transformers': transformers.__version__, 'peft': peft.__version__, 'device': 'CPU; architecture tensors on meta'},
          'scope': 'Independent toy updates, meta architecture count, actual CPU processor mask audit and arithmetic on primary training/validation artifacts. No full model weights, GPU, new generation, test inference or Git.'}

toy = []
for rank in [2, 1]:
    torch.manual_seed(42)
    layer = LoRALinear(nn.Linear(4, 3), rank=rank, alpha=2)
    before = {n: p.detach().clone() for n,p in layer.named_parameters()}
    opt = torch.optim.SGD([p for p in layer.parameters() if p.requires_grad], lr=0.1)
    loss = layer(torch.ones(1, 4)).square().mean()
    loss.backward()
    gradients = {n: None if p.grad is None else float(p.grad.abs().max()) for n,p in layer.named_parameters()}
    opt.step()
    record = {'rank': rank, 'alpha': 2, 'scaling': 2/rank,
              'a_shape': list(layer.a.shape), 'b_shape': list(layer.b.shape),
              'trainable': sum(p.numel() for p in layer.parameters() if p.requires_grad),
              'base_weight_equal': torch.equal(before['base.weight'], layer.base.weight),
              'base_bias_equal': torch.equal(before['base.bias'], layer.base.bias),
              'b_changed': not torch.equal(before['b'], layer.b),
              'gradients': gradients, 'loss': float(loss.detach()),
              'snapshot_has_separate_storage': before['base.weight'].data_ptr() != layer.base.weight.data_ptr()}
    assert record['trainable'] == (14 if rank==2 else 7)
    assert record['base_weight_equal'] and record['base_bias_equal'] and record['b_changed']
    assert gradients['a'] == 0 and gradients['b'] > 0
    toy.append(record)
result['toy'] = toy

upstream = nn.Parameter(torch.ones(1, 4))
fixed = nn.Linear(4, 3).requires_grad_(False)
fixed(upstream).square().mean().backward()
assert upstream.grad is not None and bool((upstream.grad != 0).any())
with torch.no_grad():
    disabled = fixed(upstream)
assert not disabled.requires_grad
result['frozen_downstream_autograd'] = {'fixed_parameter_gradients_none':all(p.grad is None for p in fixed.parameters()),
                                     'upstream_nonzero_gradient':True,'no_grad_output_requires_grad':False}

class LocalCheckpointModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_check = nn.Parameter(torch.ones(3))
    def save_pretrained(self, destination, safe_serialization):
        Path(destination).mkdir(parents=True)
        torch.save(self.state_dict(), Path(destination)/'toy-weights.pt')
checkpoint_model=LocalCheckpointModel()
checkpoint_optimizer=torch.optim.AdamW(checkpoint_model.parameters(),lr=0.0001)
checkpoint_model.lora_check.square().mean().backward(); checkpoint_optimizer.step()
expected_rng=torch.get_rng_state().clone()
with tempfile.TemporaryDirectory(prefix='natural-20.4-checkpoint-') as temporary:
    a.save_checkpoint(checkpoint_model,checkpoint_optimizer,temporary,{'completed_steps':1})
    state=torch.load(Path(temporary)/'adapter/training_state.pt',weights_only=True)
    assert set(state)=={'optimizer','torch_rng','cuda_rng'} and torch.equal(state['torch_rng'],expected_rng)
    assert state['optimizer']['state'] and state['cuda_rng']==[]
    assert json.loads((Path(temporary)/'adapter/training.json').read_text())=={'completed_steps':1}
result['checkpoint_contract']={'optimizer_state_saved':True,'torch_rng_saved_exactly':True,
                               'cuda_rng_saved_empty_on_cpu':True,'record_saved':True,
                               'scope':'Actual project save_checkpoint with tiny local stand-in; full pretrained resume not executed.'}

# Construct the pinned official architecture without materializing 2B values.
config = Qwen3VLConfig.from_dict(read('docs/technical-reviews/artifacts/natural-20.4-qwen-config.txt'))
with init_empty_weights():
    model = Qwen3VLForConditionalGeneration(config)
model.tie_weights()
base_count = sum(p.numel() for p in model.parameters())
model.requires_grad_(False)
model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
                        target_modules=a.LORA_TARGETS, task_type=TaskType.CAUSAL_LM))
shapes = {n:list(p.shape) for n,p in model.named_parameters() if p.requires_grad}
counts = Counter((n.split('.self_attn.')[1].split('.')[0], n.split('.self_attn.')[1].split('.')[1], tuple(s)) for n,s in shapes.items())
actual_trainable = sum(math.prod(s) for s in shapes.values())
assert base_count == 2127532032 and actual_trainable == 1605632 and len(shapes) == 112
assert all('lora_' in n and '.visual.' not in n for n in shapes)
result['architecture'] = {'base_parameters':base_count, 'trainable_parameters':actual_trainable,
                          'trainable_tensor_count':len(shapes), 'meta_only':True,
                          'shape_groups': [{'projection':q, 'branch':b, 'shape':list(s), 'count':v}
                                           for (q,b,s),v in counts.items()]}
del model

# Verify length-weighted accumulation against one concatenated CE objective.
torch.manual_seed(42)
w = nn.Parameter(torch.randn(3, 5))
logits = [torch.randn(1, 4, 3) @ w, torch.randn(1, 7, 3) @ w]
labels = [torch.tensor([[-100, 1, 2, 3]]),torch.tensor([[-100,-100,2,1,3,2,4]])]
lengths = [int((y[:,1:] != -100).sum()) for y in labels]
weighted = sum(ForCausalLMLoss(z,y,5) * n/sum(lengths) for z,y,n in zip(logits,labels,lengths))
g1 = torch.autograd.grad(weighted,w,retain_graph=True)[0]
flat_logits = torch.cat([z[:,:-1].reshape(-1,5) for z in logits])
flat_labels = torch.cat([y[:,1:].reshape(-1) for y in labels])
direct = nn.functional.cross_entropy(flat_logits,flat_labels,ignore_index=-100)
g2 = torch.autograd.grad(direct,w)[0]
assert torch.allclose(weighted,direct,atol=1e-6,rtol=0) and torch.allclose(g1,g2,atol=1e-6,rtol=0)
result['token_weighted_loss'] = {'supervised_tokens':lengths,'weighted_loss':float(weighted.detach()),
                               'concatenated_loss':float(direct.detach()), 'gradient_max_error':float((g1-g2).abs().max()),
                               'tolerance':1e-6}

manifest_path = ROOT / 'docs/natural-assistant/manifest.json'
manifest = json.loads(manifest_path.read_text())
rows = [r for r in manifest['rows'] if r['split']=='train']
processor = AutoProcessor.from_pretrained(a.MODEL_ID,revision=a.MODEL_REVISION,
             cache_dir=ROOT/'outputs/natural-extension/student-base-cache/hf',
             local_files_only=True,min_pixels=65536,max_pixels=524288)
audit = []
for row in rows:
    batch = a.encode_training_row(processor,row,ROOT/'data/natural',2048)
    prompt = a.encode_messages(processor,a.messages_for(row,ROOT/'data/natural'),ROOT/'data/natural',generation_prompt=True)
    n = prompt['input_ids'].shape[-1]
    assert torch.equal(prompt['input_ids'],batch['input_ids'][:,:n])
    assert bool((batch['labels'][:,:n]==-100).all())
    assert torch.equal(batch['labels'][:,n:],batch['input_ids'][:,n:])
    target = batch['labels'][:,1:]; ids = target[target!=-100].tolist()
    assert processor.tokenizer.eos_token_id in ids
    audit.append({'id':row['id'],'input_tokens':batch['input_ids'].shape[-1],
                  'prompt_tokens_masked':n,'shifted_supervised_tokens':len(ids), 'eos_in_suffix':True})
original = read('docs/natural-assistant/evidence/training-mask/result.json')
assert len(audit)==272 and max(r['input_tokens'] for r in audit)==578
assert sum(r['shifted_supervised_tokens'] for r in audit)==4476
assert all(all(r[k]==o[k] for k in ['id','input_tokens','prompt_tokens_masked','shifted_supervised_tokens']) for r,o in zip(audit,original['rows']))
# The function must reject oversize sequences, never silently slice.
oversize_rejected = False
try:
    a.encode_training_row(processor,rows[0],ROOT/'data/natural',1)
except ValueError as exc:
    oversize_rejected = 'refusing silent multimodal truncation' in str(exc)
assert oversize_rejected
result['mask_audit'] = {'row_count':len(audit),'max_input_tokens':max(r['input_tokens'] for r in audit),
                       'max_tokens':2048,'single_pass_supervised_tokens':sum(r['shifted_supervised_tokens'] for r in audit),
                       'all_prompt_ignored':True,'all_suffix_matches':True,'all_eos_supervised':True,
                       'all_rows_match_primary_audit':True,'oversize_rejected':True,
                       'manifest_sha256':digest(manifest_path),'rows':audit}

training = {}
for name in ['train','train-gentle']:
    d = read(f'docs/natural-assistant/evidence/{name}/training.json')
    outer = read(f'docs/natural-assistant/evidence/{name}/result.json')
    assert all(outer[k]==v for k,v in d.items())
    h = d['history']; visits = [id for step in h for id in step['row_ids']]
    assert len(h)==180 and all(len(step['row_ids'])==2 for step in h)
    assert sum(step['supervised_tokens'] for step in h)==5791 and len(visits)==360 and len(set(visits))==272
    assert visits==[a.training_row_at(rows,i,42)['id'] for i in range(360)]
    audit_tokens={r['id']:r['shifted_supervised_tokens'] for r in audit}
    assert all(step['supervised_tokens']==sum(audit_tokens[id] for id in step['row_ids']) for step in h)
    changed=[n for n,v in d['initial_adapter_tensors'].items() if v['sha256_values']!=d['final_adapter_tensors'][n]['sha256_values']]
    assert len(changed)==112 and d['frozen_parameter_samples_initial']==d['frozen_parameter_samples_final']
    assert len(d['frozen_parameter_samples_initial'])==7
    assert set(d['optimizer_parameter_names'])==set(d['trainable_parameter_names'])==set(d['initial_adapter_tensors'])
    assert {n: v['shape'] for n,v in d['initial_adapter_tensors'].items()}==shapes
    assert all(math.isfinite(step['answer_token_loss_before_update']) and math.isfinite(step['gradient_norm_before_clip']) for step in h)
    training[name]={'steps':len(h),'visits':len(visits),'unique':len(set(visits)),
                    'supervised_tokens':sum(step['supervised_tokens'] for step in h),'changed_tensors':len(changed),
                    'frozen_sample_tensors':7,'frozen_sample_values':sum(len(v['flat_indices']) for v in d['frozen_parameter_samples_initial'].values()),
                    'learning_rate':d['learning_rate'],'seed':d['seed'], 'elapsed_seconds':d['elapsed_seconds'],
                    'peak_allocated_bytes':d['peak_cuda_memory_allocated_bytes'], 'peak_gib':d['peak_cuda_memory_allocated_bytes']/1024**3,
                    'all_finite_loss_and_grad_norm':True,'optimizer_matches_lora':True,
                    'fresh_run_no_resume_key':'resume_from' not in d,
                    'first_loss':h[0]['answer_token_loss_before_update'],'last_loss':h[-1]['answer_token_loss_before_update']}
d1=read('docs/natural-assistant/evidence/train/training.json');d2=read('docs/natural-assistant/evidence/train-gentle/training.json')
assert d1['initial_adapter_tensors']==d2['initial_adapter_tensors']
assert [(h['row_ids'],h['supervised_tokens']) for h in d1['history']]==[(h['row_ids'],h['supervised_tokens']) for h in d2['history']]
result['training_artifact_recomputation']=training
result['controlled_recomputation']={'initial_hashes_equal':112,'same_per_step_row_order_and_tokens':True,'learning_rates':[d1['learning_rate'],d2['learning_rate']],
                                  'quality_claim':'No quality conclusion follows from these training artifacts.'}
settings=['model','model_revision','asr_model','asr_revision','versions','device','dtype','gpu_name',
          'attention_implementation','max_pixels','min_pixels','max_tokens','seed','dataset_version','manifest_sha256',
          'asset_sha256','sources','total_parameters','trainable_parameters','lora_rank','lora_targets','requested_steps',
          'gradient_accumulation','optimizer_parameter_names','trainable_parameter_names','optimizer_only_lora']
assert all(d1[k]==d2[k] for k in settings)
result['controlled_recomputation']['identical_configuration_fields']=settings

base=read('docs/natural-assistant/evidence/validation/generations-base.json')
adapt=read('docs/natural-assistant/evidence/validation/generations-adapter.json')
base_by={r['id']:r for r in base}
trunc=[]
for r in adapt:
    if r['task']=='scene':
        no_eos=not any(id in r['eos_token_ids'] for id in r['generated_token_ids'])
        recomputed=no_eos and len(r['generated_token_ids'])==384
        assert r['truncated']==recomputed
        if recomputed:
            b=base_by[r['id']]
            trunc.append({'id':r['id'],'user':r['user'],'adapter_prediction':r['prediction'],'base_prediction':b['prediction'],
                          'generated_tokens':len(r['generated_token_ids']),'no_eos':no_eos,'base_ended_with_eos':b['ended_with_eos']})
assert len(trunc)==5
result['photo_truncation']=trunc
result['short_answer_pairs']=[{'id':r['id'],'user':r['user'],'base':base_by[r['id']]['prediction'],'adapter':r['prediction'],
                               'base_ended_with_eos':base_by[r['id']]['ended_with_eos'],'adapter_ended_with_eos':r['ended_with_eos']}
                               for r in adapt if r['task'] in ['text_chat','text_dialogue']]
source=(ROOT/'tiny_perceptron/natural_assistant.py').read_text()
result['timer_source_order']={
    'starts_after_manifest_audit':source.index('manifest, data_root = load_manifest',source.index('def run_train')) < source.index('started = time.monotonic()',source.index('def run_train')),
    'record_before_final_hashes_and_save':source.index('record["elapsed_seconds"] = time.monotonic() - started',source.index('def run_train')) < source.index('record["final_adapter_tensors"]',source.index('def run_train')),
    'scope':'Model loading, training and checkpoints inside loop; excludes manifest audit, external setup/download/transfer, final tensor checks and final checkpoint/provenance/result writes.'}

body=(ROOT/'course/chapters/20.md').read_text(); start=re.search(r'^## 20\.4 ',body,re.M).start(); end=re.search(r'^## ',body[start+1:],re.M)
section=body[start:start+1+end.start()] if end else body[start:]
(OUT/'natural-20.4-reviewed-section.md').write_text(section)
result['reviewed_source_sha256']=hashlib.sha256(section.encode()).hexdigest()
result['input_sha256']={str(p.relative_to(ROOT)):digest(p) for p in [ROOT/'tiny_perceptron/natural_assistant.py',ROOT/'tiny_perceptron/alignment.py',manifest_path,ROOT/'course/figures/natural_base_adapter.svg']}
(OUT/'natural-20.4-independent-check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['mask_audit','photo_truncation','short_answer_pairs']},ensure_ascii=False,indent=2))
print('mask_audit', len(audit), 'max578', 'all matched; 5 truncation raw outputs saved')
