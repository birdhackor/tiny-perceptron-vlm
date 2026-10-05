"""Bounded structural review: original exercise, parameter accounting, chain, existing record."""
from pathlib import Path
import hashlib
import inspect
import json
import os
import platform
import sys

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import torch
from torch import nn
from torch.nn import functional as F
from tiny_perceptron.model import TinyLM, ModelConfig
from tiny_perceptron.data import ByteTokenizer, SPECIALS

torch.set_num_threads(1)
torch.manual_seed(42)
assert torch.version.cuda is None and not torch.cuda.is_available()
torch.set_default_device('cpu')

def number(module):
    return sum(p.numel() for p in module.parameters())

def audit(model):
    entries=[{'name':name,'shape':list(p.shape),'numel':p.numel(),'requires_grad':p.requires_grad} for name,p in model.named_parameters()]
    outside={name:number(getattr(model,name)) for name in ('embedding','position','final_norm','output')}
    blocks=[]
    for block in model.blocks:
        parts={
          'attention_weights':sum(number(getattr(block.attention,name)) for name in ('q','k','v','out')),
          'ffn_weights':block.ffn.up.weight.numel()+block.ffn.down.weight.numel(),
          'ffn_bias':block.ffn.up.bias.numel()+block.ffn.down.bias.numel(),
          'ln_weight_and_bias':number(block.norm1)+number(block.norm2),
        }
        assert all(getattr(block.attention,name).bias is None for name in ('q','k','v','out'))
        assert block.ffn.gate is None and block.ffn.up.out_features==4*model.config.width
        assert sum(parts.values())==number(block)
        blocks.append({'parts':parts,'total':number(block)})
    assert model.output.bias is None
    assert model.output.weight.data_ptr()!=model.embedding.weight.data_ptr()
    assert sum(outside.values())+sum(b['total'] for b in blocks)==model.description()['parameters']
    parameter_ids=[{id(p) for p in b.parameters()} for b in model.blocks]
    assert all(not left & right for i,left in enumerate(parameter_ids) for right in parameter_ids[i+1:])
    return {'outside':outside,'outside_total':sum(outside.values()),'blocks':blocks,'named_parameters':entries,'description':model.description(),'independent_blocks':True,'untied_output':True}

rows=[]
for width in (8,16):
    totals=[]
    for depth in (1,2,3):
        model=TinyLM(ModelConfig(vocab_size=20,width=width,layers=depth))
        input_trace=[]
        output_trace=[]
        saved=[]
        hooks=[]
        for block in model.blocks:
            hooks.append(block.register_forward_pre_hook(lambda _module,args: input_trace.append(args[0].detach().clone())))
            hooks.append(block.register_forward_hook(lambda _module,_args,result: output_trace.append(result[0].detach().clone())))
        before={k:p.detach().clone() for k,p in model.named_parameters()}
        def pack(tensor):
            saved.append({'shape':list(tensor.shape),'numel':tensor.numel(),'bytes':tensor.numel()*tensor.element_size()})
            return tensor
        with torch.autograd.graph.saved_tensors_hooks(pack,lambda tensor:tensor):
            logits=model(torch.tensor([[1,2]]))['logits']
            loss=F.cross_entropy(logits.reshape(-1,20),torch.tensor([3,4]))
        loss.backward()
        for hook in hooks:
            hook.remove()
        assert list(logits.shape)==[1,2,20]
        assert len(input_trace)==len(output_trace)==depth
        assert all(torch.equal(output_trace[i-1],input_trace[i]) for i in range(1,depth))
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        assert all(torch.equal(before[name],p.detach()) for name,p in model.named_parameters())
        counts=audit(model)
        assert counts['outside_total']==(20*width+128*width+2*width+20*width)
        assert all(b['total']==12*width*width+9*width for b in counts['blocks'])
        count=counts['description']['parameters']
        totals.append(count)
        rows.append({'width':width,'depth':depth,'shape':list(logits.shape),'count':count,'audit':counts,'block_trace_shapes':[list(t.shape) for t in input_trace],'chain_equals_previous_output':True,'saved_for_backward_count':len(saved),'saved_tensor_elements_including_parameter_references':sum(t['numel'] for t in saved),'saved_tensor_shapes':saved,'backward_finite':True,'parameters_unchanged_after_backward':True})
        print(f'width={width} depth={depth} parameters={count} outside={counts["outside_total"]} per_block={[b["total"] for b in counts["blocks"]]} output={list(logits.shape)} chain=OK backward=finite no_optimizer_update=OK')
    assert totals==([2200,3040,3880] if width==8 else [5936,9152,12368])

model=TinyLM(ModelConfig(vocab_size=20,width=8,layers=1))
assert list(model(torch.ones((1,128),dtype=torch.long))['logits'].shape)==[1,128,20]
try:
    model(torch.ones((1,129),dtype=torch.long))
except ValueError as error:
    boundary=str(error)
else:
    raise AssertionError('Expected max_length error')
large=TinyLM(ModelConfig(vocab_size=264,width=64,layers=2,max_length=128))
large_counts=audit(large)
assert large_counts['outside_total']==42112
assert [b['total'] for b in large_counts['blocks']]==[49728,49728]
assert number(large)==141568
record_path=OUT/'inputs/docs/course-experiments/results/text_foundation.json'
record=json.loads(record_path.read_text())
training=record['results']['training']
assert training['parameters']==training['trainable_parameters']==number(large)
assert record['step_scale']==1 and record['evidence_status']=='complete_run'
assert training['steps']==600 and training['records']==record['results']['data']['train']['records']==9
assert record['results']['data']['validation']['records']==1 and record['results']['data']['test']['records']==2
scaling=record['results']['scaling']
assert set(scaling)=={'scaling-w16-n16','scaling-w16-n64','scaling-w32-n16','scaling-w32-n64'}
assert all(row['training']['steps']==150 for row in scaling.values())
assert {row['parameters'] for row in scaling.values()}=={13744,33632}
assert ByteTokenizer.vocab_size==256+len(SPECIALS)==264
assert len(ByteTokenizer().encode('中'))==3 and len(ByteTokenizer().encode('A'))==1
print(f'fixed_record: parameters={training["parameters"]} outside=42112 block=49728 steps=600 records=train9/validation1/test2; width/data scaling points are not a depth comparison; 264=256 byte IDs+8 control IDs')

installed=[]
for name,module in [('linear',torch.nn.modules.linear),('normalization',torch.nn.modules.normalization),('sparse',torch.nn.modules.sparse),('module',torch.nn.modules.module)]:
    path=Path(inspect.getsourcefile(module))
    saved=OUT/'installed'/f'torch-{name}.py'
    saved.parent.mkdir(exist_ok=True)
    raw=path.read_bytes()
    saved.write_bytes(raw)
    remote=OUT/'sources'/f'torch-{name}.py'
    installed.append({'module':module.__name__,'installed_path':str(path),'saved_path':str(saved.relative_to(ROOT)),'sha256':hashlib.sha256(raw).hexdigest(),'matches_official_git_commit_source':raw==remote.read_bytes()})
environment={'python':sys.version,'python_executable':sys.executable,'torch':str(torch.__version__),'torch_git_version':str(torch.version.git_version),'torch_cuda_build':str(torch.version.cuda),'cuda_available':str(torch.cuda.is_available()),'device':'cpu','threads':str(torch.get_num_threads()),'platform':platform.platform(),'cwd':str(Path.cwd()),'offline_requested':{k:os.environ.get(k,'unset') for k in ['CUDA_VISIBLE_DEVICES','HF_HUB_OFFLINE','HF_DATASETS_OFFLINE','TRANSFORMERS_OFFLINE','OMP_NUM_THREADS','MKL_NUM_THREADS']}}
(OUT/'probe-environment.json').write_text(json.dumps(environment,ensure_ascii=False,indent=2)+'\n')
answer={'rows':rows,'width_comparison':{'outside_ratio':2,'attention_weight_ratio':4,'ffn_weight_ratio':4,'ffn_bias_ratio':2,'ln_ratio':2,'per_block_8':840,'per_block_16':3216,'formula':'outside=(2*vocab_size+max_length+2)*width; block=12*width**2+9*width for default nonrotary/untied/one-head/GELU settings'},'max_length_boundary':{'length128':'passed','length129':boundary},'fixed_existing_record':{'input':str(record_path.relative_to(ROOT)),'sha256':hashlib.sha256(record_path.read_bytes()).hexdigest(),'revision':record['revision'],'historical_torch':record['torch_version'],'training':{k:training[k] for k in ['parameters','trainable_parameters','steps','records','effective_tokens','checkpoint']},'outside_total':42112,'per_block':49728,'total':141568,'heldout_records':{'validation':1,'test':2},'one_layer_ablations':'Only width16/32 plus data16/64 scaling rows; no depth1 vs depth2 matched-budget comparison','vocab':{'size':264,'byte_ids':256,'control_ids':8,'one_chinese_character_bytes':3}},'installed_official_source_binding':installed,'scope':'Structural CPU checks, one backward per tiny untrained model, no optimizer step. Saved tensor element sum is diagnostic, includes parameter references and duplicate tensors, and is not peak memory or a runtime benchmark. Existing GPU result was inspected, not rerun.'}
(OUT/'probe-result.json').write_text(json.dumps(answer,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'installed_source_binding':installed,'environment':environment},ensure_ascii=False,indent=2))
