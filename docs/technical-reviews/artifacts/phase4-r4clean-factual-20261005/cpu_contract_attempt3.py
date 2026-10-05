"""Independent bounded contracts, not a trained capability evaluation."""
from pathlib import Path
import ast
from collections import Counter
import hashlib
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.attention import manual_attention
from tiny_perceptron.capstone import CapstoneModel, TOK, prompt_ids
from tiny_perceptron.model import ModelConfig
from scripts.course_experiments.common import split_records

OUT = Path(__file__).resolve().parent
torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()
torch.manual_seed(1729)
observed = {}

# A table lookup, attention mixture and scalar update demonstrate operations only.
embedding = torch.nn.Embedding.from_pretrained(torch.tensor([[2.,0.],[0.,4.]]), freeze=False)
assert torch.equal(embedding(torch.tensor([1,0])), torch.tensor([[0.,4.],[2.,0.]]))
q = torch.tensor([[1.]])
k = torch.tensor([[0.],[0.]])
v = embedding(torch.tensor([0,1]))
mix, weights = manual_attention(q,k,v,torch.tensor([[True,True]]))
assert torch.equal(weights,torch.tensor([[.5,.5]]))
assert torch.equal(mix,torch.tensor([[1.,2.]]))
masked, masked_weights = manual_attention(q,k,v,torch.tensor([[False,True]]))
assert torch.equal(masked,torch.tensor([[0.,4.]]))
assert torch.equal(masked_weights,torch.tensor([[0.,1.]]))
observed['lookup_and_mixing'] = 'Lookup selected rows; equal compatibility mixed both; masked first position selected only second.'
parameter = torch.nn.Parameter(torch.tensor([2.]))
optimizer = torch.optim.AdamW([parameter],lr=.1,weight_decay=0)
before = parameter.detach().clone()
(parameter**2).sum().backward()
assert torch.equal(parameter,before) and torch.equal(parameter.grad,torch.tensor([4.]))
optimizer.step()
assert not torch.equal(parameter,before)
after = parameter.detach().clone()
with torch.no_grad(): (parameter**2).sum()
assert torch.equal(parameter,after)
observed['gradient_update_evaluation'] = 'backward produced gradient without a value update; AdamW.step changed it; no_grad observation did not update it.'

# Existing helper isolates source families; no training loop is run.
records = [{'family': f'f{i}', 'value': j} for i in range(10) for j in range(2)]
splits = split_records(records, seed=42)
families = {name:{row['family'] for row in rows} for name,rows in splits.items()}
assert not any(families[a]&families[b] for a,b in [('train','validation'),('train','test'),('validation','test')])
assert sum(map(len,splits.values()))==len(records)
observed['split_contract'] = {name:len(rows) for name,rows in splits.items()}

# Capture actual inputs passed to the core, without interpreting random logits.
model = CapstoneModel(ModelConfig(vocab_size=264,width=8,heads=2,layers=1,max_length=64))
row = {'system':'', 'user':'Q', 'image':{}, 'audio':{}}
ids = torch.tensor([prompt_ids(row)])
question_start = len(prompt_ids({**row, 'user':''})) - 2
assert ids[0,question_start:question_start+len(TOK.encode(row['user']))].tolist() == TOK.encode(row['user'])
assert ids[0].tolist().index(TOK.image_id) < question_start
assert ids[0].tolist().index(TOK.audio_id) < question_start
images = torch.zeros((1,3,16,16)); features = torch.zeros((1,16))
captured = []
def capture(module, args, kwargs): captured.append(kwargs['embeddings'].detach().clone())
hook = model.language.register_forward_pre_hook(capture,with_kwargs=True)
with torch.no_grad(): model(ids, images=images, audio_features=features)
actual = captured[-1]
image_vector = model.image_projector(torch.nn.functional.adaptive_avg_pool2d(images,(4,4)).flatten(1)).detach()[0]
audio_vector = model.audio_projector(features).detach()[0]
assert torch.equal(actual[0,ids[0]==TOK.image_id],image_vector.unsqueeze(0))
assert torch.equal(actual[0,ids[0]==TOK.audio_id],audio_vector.unsqueeze(0))
plain = ~((ids==TOK.image_id)|(ids==TOK.audio_id))
assert torch.equal(actual[plain],model.language.embedding(ids).detach()[plain])
changed_features=torch.ones_like(features)
with torch.no_grad(): model(ids, images=images, audio_features=changed_features)
assert not torch.equal(captured[0][ids==TOK.audio_id],captured[1][ids==TOK.audio_id])
assert torch.equal(captured[0][ids==TOK.image_id],captured[1][ids==TOK.image_id])
hook.remove()
observed['prefix_contract'] = 'Image/audio markers precede user question; actual projected vectors replace those positions; audio-only input change leaves image vector unchanged. No capability inference from random logits.'

# The mature route contracts load existing upstream weights; never invoke loaders.
source=(ROOT/'tiny_perceptron/natural_assistant.py').read_text()
tree=ast.parse(source)
route_contract={}
for function in ('load_core','load_asr'):
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==function)
    calls=[n for n in ast.walk(node) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='from_pretrained']
    assert len(calls)==2
    route_contract[function]={'from_pretrained_call_lines':[n.lineno for n in calls],'scope':'AST/API contract only; no model loader invoked and no model weights downloaded.'}
observed['mature_route_contract']=route_contract

manifest=json.loads((ROOT/'assets/training/manifest.json').read_text())
observed['asset_license_declarations']=[{k:a[k] for k in ('id','license','source_metadata')} for a in manifest['assets']]
assert any(a['license']!='MIT' for a in manifest['assets'])
assert 'MIT License' in (ROOT/'LICENSE').read_text()
observed['license_scope']='Root MIT grant applies to repository software/documentation; upstream data/font/model terms remain their own. Representative primary source licenses personally inspected separately; this is not an audit of every redistribution obligation.'

# Inspect only raw named input/output pointers; do not read author summary fields.
data=json.loads((OUT/'legacy-data.json').read_text())
results=json.loads((OUT/'legacy-test-joint.json').read_text())
test=data['splits']['test']; generated=results['records']
data_ids={row['id'] for row in test}; result_ids={row['id'] for row in generated}
assert data_ids==result_ids
images=[row['image'] for row in test if row['image'] is not None]
audio=[row['audio'] for row in test if row['audio'] is not None]
assert all(set(spec)<= {'color','shape','offset','variant'} for spec in images)
assert all(set(spec)<= {'pitch','variation','delta'} for spec in audio)
assert {spec['pitch'] for spec in audio} <= {'low','high'}
observed['archived_input_output_contract'] = {'raw_test_records':len(test),'raw_output_records':len(generated),'input_task_counts':dict(Counter(row['task'] for row in test)),'image_fields':sorted(set().union(*(set(x) for x in images))),'audio_fields':sorted(set().union(*(set(x) for x in audio))),'personally_inspected_pointers':['/splits/test/*/id','/splits/test/*/task','/splits/test/*/image','/splits/test/*/audio','/records/*/id'],'scope':'ID and synthetic material schema validation only; no archived accuracy claim or newly generated model score.'}

body=(OUT/'source-R.4.md').read_text()
links=[]
for label,relative in re.findall(r'\[([^\]]+)\]\(([^)]+)\)',body):
    path,sep,fragment=relative.partition('#'); target=(ROOT/'course'/path).resolve()
    assert target.is_file()
    if sep: assert re.search(r'^## '+re.escape(fragment)+r' ',target.read_text(),re.M)
    links.append({'label':label,'target':target.relative_to(ROOT).as_posix(),'fragment':fragment,'exists':True,'inspection':'Existence/anchor only for assets/training/README.md and docs/course-experiments/README.md; their bodies were not read.'})
observed['navigation_targets']=links
training=(ROOT/'course/training.md').read_text()
observed['capstone_operation_locator'] = {'fetch_capstone_in_training_page':'fetch_capstone' in training,'four_stage_capstone_command_in_training_page':'scripts/capstone.py train' in training,'actual_old_capstone_recipe':'course/chapters/19.md#19.11','finding':'training.md contains general/local recipes; the old integrated product download and staged recipe are in 19.11. New product operations remain pending.'}

environment={'python':sys.version,'python_executable':sys.executable,'torch':str(torch.__version__),'torch_git_version':str(torch.version.git_version),'device':'cpu','cuda_build':str(torch.version.cuda),'cuda_available':str(torch.cuda.is_available()),'threads':str(torch.get_num_threads()),'scope':'Only small CPU operation/contracts; no dataset/model-weight download, persistent weights, complete training, GPU use or ability score.'}
(OUT/'cpu-environment.json').write_text(json.dumps(environment,ensure_ascii=False,indent=2)+'\n')
(OUT/'cpu-observations.json').write_text(json.dumps(observed,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(observed,ensure_ascii=False,indent=2))
print('All bounded CPU contract assertions passed.')
