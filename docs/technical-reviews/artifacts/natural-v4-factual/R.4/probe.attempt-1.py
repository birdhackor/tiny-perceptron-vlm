import contextlib
import hashlib
import io
import json
import math
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

import torch
from torch import nn

from tiny_perceptron.alignment import distillation_loss
from tiny_perceptron.capstone import CapstoneModel, TOK, build_dataset, default_config, export_inference, load_capstone, save_capstone
from tiny_perceptron.quantization import QuantizedLinear

torch.set_num_threads(2)
torch.manual_seed(42)
out = {}
inventory = {}
scratch = Path('outputs/natural-v4/factual-research/R.4/probe')
scratch.mkdir(parents=True, exist_ok=True)

def raw(path):
    p = Path(path)
    data = p.read_bytes()
    inventory[path] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    return json.loads(data)

model = CapstoneModel()
dense = CapstoneModel(default_config(dense=True))
assert sum(p.numel() for p in model.parameters()) == 328128
assert model.description()['logical_active_parameters'] == 195776
assert sum(p.numel() for p in dense.parameters()) == 129088
out['architecture'] = {'moe': model.description(), 'dense': dense.description(), 'moe_fp32_weight_bytes': 328128*4}
splits, manifest = build_dataset()
data = raw('docs/course-experiments/capstone-evidence/deployment/data.json')
assert data['splits'] == splits and data['manifest'] == manifest
assert all(not ({r['family'] for r in splits[a]} & {r['family'] for r in splits[b]})
           for a,b in [('train','validation'),('train','test'),('validation','test')])
out['dataset'] = {'counts': manifest['counts'], 'split_unit': manifest['split_unit'], 'test_tasks': dict(Counter(r['task'] for r in splits['test']))}
out['training_records'] = {}
for stage in ['pretrain','sft','joint','dpo']:
    record = raw(f'docs/course-experiments/capstone-evidence/{stage}/train-report.json')
    assert record['schedule_completed'] and record['data_manifest'] == manifest
    assert record['parameters']['parameters'] == 328128
    out['training_records'][stage] = {k: record[k] for k in ['seed','steps','requested_steps','effective_tokens','parent_checkpoint_sha256','objective','seconds']}
assert out['training_records']['pretrain']['parent_checkpoint_sha256'] is None

test_rows = {r['id']: r for r in splits['test']}
out['evaluations'] = {}
evaluations = {}
for label, path in [
    ('joint','docs/course-experiments/capstone-evidence/deployment/test-joint.json'),
    ('joint_int4','docs/course-experiments/capstone-evidence/deployment/test-joint-ptq4.json'),
    ('ce','docs/course-experiments/capstone-evidence/student/test-ce.json'),
    ('kd','docs/course-experiments/capstone-evidence/student/test-kd.json')]:
    record = raw(path)
    correct = 0; action_correct = 0; failures = []
    assert {r['id'] for r in record['records']} == set(test_rows)
    for row in record['records']:
        expected = test_rows[row['id']]['answer']
        trace = row['action_trace']; generated = trace['generated_ids']
        assert TOK.decode(generated) == trace['raw']
        eos = bool(generated and generated[-1] == TOK.eos_id)
        assert eos == trace['eos']
        first_ok = eos and trace['raw'] == expected
        target = str(sum(map(int, expected.split(':')[-1].split('+')))) if expected.startswith('TOOL:') else expected.split(':',1)[1]
        if row['final_trace'] is not None:
            final = row['final_trace']; ids = final['generated_ids']
            assert TOK.decode(ids) == final['raw']
            assert bool(ids and ids[-1] == TOK.eos_id) == final['eos']
            answer = final['raw'][7:] if final['eos'] and final['raw'].startswith('DIRECT:') else None
        else:
            answer = trace['raw'].split(':',1)[1] if eos and re.match(r'^(DIRECT|ASK):.+', trace['raw']) else None
        whole_ok = first_ok and answer == target
        assert first_ok == row['action_correct'] and whole_ok == row['end_to_end_correct']
        action_correct += first_ok; correct += whole_ok
        if not whole_ok and len(failures)<3:
            failures.append({'task':row['task'],'expected':expected,'generated':trace['raw'],'final':answer})
    out['evaluations'][label] = {'count':len(record['records']),'action_correct':action_correct,'end_to_end_correct':correct,'first_failures':failures}
    evaluations[label] = record
assert [out['evaluations'][x]['end_to_end_correct'] for x in ['joint','joint_int4','ce','kd']] == [78,78,62,61]
out['int4_changed_generation_records'] = sum(
    a['action_trace']['generated_ids'] != b['action_trace']['generated_ids'] or
    (a['final_trace'] or {}).get('generated_ids') != (b['final_trace'] or {}).get('generated_ids')
    for a,b in zip(evaluations['joint']['records'],evaluations['joint_int4']['records'],strict=True))
assert out['int4_changed_generation_records']==1
out['student_recipes'] = {}
for label in ['ce','kd']:
    record = raw(f'docs/course-experiments/capstone-evidence/student/{label}/train-report.json')
    assert record['steps']==350 and record['effective_tokens']==145163 and record['parameters']['parameters']==79920
    out['student_recipes'][label] = {k:record[k] for k in ['steps','effective_tokens','objective','teacher_checkpoint_sha256','initialization']}

save_capstone(scratch/'training.pt',model,stage='pretrain',step=0,optimizer=torch.optim.AdamW(model.parameters()))
export_inference(scratch/'training.pt',scratch/'inference.pt')
loaded,payload = load_capstone(scratch/'inference.pt')
omitted = ['optimizer','training_state','torch_rng','python_rng','cuda_rng','reference']
assert all(k not in payload for k in omitted)
assert all(torch.equal(v,loaded.state_dict()[k]) for k,v in model.state_dict().items())
out['inference_export'] = {'keys':sorted(payload),'omitted':omitted,'exact_weight_roundtrip':True,'trained_updates':0}

layer = nn.Linear(3,1,bias=False)
with torch.no_grad():layer.weight.copy_(torch.tensor([[1.,2.,3.]]))
before = layer.weight.detach().clone(); loss = layer(torch.ones(1,3)).square().mean();loss.backward()
assert torch.equal(before,layer.weight)
torch.optim.SGD(layer.parameters(),lr=0.01).step()
assert not torch.equal(before,layer.weight)
out['backward_vs_update'] = {'before':before.tolist(),'after_backward':before.tolist(),'after_step':layer.weight.detach().tolist(),'loss':loss.item(),'shape':[1,3]}
values = torch.tensor([[2.,0.],[0.,4.]])
out['attention'] = (torch.tensor([.9,.1]) @ values).tolist()
assert torch.allclose(torch.tensor(out['attention']),torch.tensor([1.8,.4]))
image_vectors = torch.tensor([[1.,2.,3.],[7.,8.,9.]])
ignored_image = torch.zeros_like(image_vectors)+torch.tensor([3.,4.,5.])
assert image_vectors.shape == ignored_image.shape and torch.equal(ignored_image[0],ignored_image[1])
out['dimension_counterexample'] = {'both_vector_dimensions':3,'different_inputs_same_output':ignored_image.tolist(),'scope':'Compatible dimensions do not prove that an output depends on its image.'}
quant = nn.Linear(8,4); compressed = QuantizedLinear(quant,bits=4)
out['small_quantization'] = {'float_bytes':sum(p.numel()*p.element_size() for p in quant.parameters()),'packed_bytes':compressed.storage_bytes(),'output_max_error':float((quant(torch.ones(1,8))-compressed(torch.ones(1,8))).abs().max().detach())}
assert out['small_quantization']['float_bytes']==144 and out['small_quantization']['packed_bytes']==48
student=torch.tensor([[[.2,.1,-.3]]],requires_grad=True); teacher=torch.tensor([[[1.,-.5,.2]]],requires_grad=True);labels=torch.tensor([[0]])
actual=distillation_loss(student,teacher,labels,alpha=.5,temperature=2)
s=student.detach().flatten().tolist();t=teacher.detach().flatten().tolist()
def sm(xs):
    ex=[math.exp(x) for x in xs];return [v/sum(ex) for v in ex]
p=sm([x/2 for x in t]);q=sm([x/2 for x in s]);ce=-math.log(sm(s)[0]);kl=sum(a*math.log(a/b) for a,b in zip(p,q,strict=True))*4
expected=.5*ce+.5*kl
assert abs(float(actual.detach())-expected)<1e-6
actual.backward();assert teacher.grad is None
out['distillation_math']={'expected':expected,'observed':float(actual.detach()),'teacher_grad':None,'temperature':2,'alpha':.5,'positions':1,'candidates':3}

root='outputs/natural-v4/modal-runs'
training=raw(root+'/train-37217452291/natural-natural-v4-train-37217452291-1/review/training.json')
assert len(training['history'])==training['completed_steps']==2077
assert sum(len(r['row_ids']) for r in training['history'])==training['trained_rows']==4154
assert sum(r['supervised_tokens'] for r in training['history'])==78872
changed=sum(training['initial_adapter_tensors'][k]['sha256'] != training['final_adapter_tensors'][k]['sha256'] for k in training['initial_adapter_tensors'])
assert changed==112
out['natural_training']={k:training[k] for k in ['model','model_revision','asr_model','asr_revision','pretrained_boundary','trainable_parameters','completed_steps','trained_rows','learning_rate','lora_rank','lora_targets','device','dtype','seed']}
out['natural_training'].update(supervised_tokens=78872,changed_adapter_tensors=changed)
generations=raw(root+'/evaluate-37221188153/natural-natural-v4-evaluate-37221188153-1/review/generations-base.json')
assert len(generations)==178
eos=sum(bool(r['generated_token_ids'] and r['generated_token_ids'][-1] in r['eos_token_ids']) for r in generations)
assert eos==171 and all(r['variant']=='base' for r in generations)
assert all(r['task']=='chat' for r in generations if r['truncated'])
out['natural_generation']={'records':len(generations),'raw_eos':eos,'truncated':sum(r['truncated'] for r in generations),'tasks':dict(Counter(r['task'] for r in generations)),'selected_variant':'base','scope':'Original inference outputs counted; no model rerun or semantic regrading.'}
benchmark=raw('docs/course-experiments/capstone-evidence/deployment/mechanism-benchmark.json')
out['original_gpu_resource_receipt']={}
for label in ['moe','dense80']:
    b=benchmark[label];median=statistics.median(b['seconds']);extra=b['cuda_peak_allocated_bytes']-b['cuda_baseline_allocated_bytes']
    assert median==b['median_seconds'] and extra==b['cuda_peak_extra_bytes']
    out['original_gpu_resource_receipt'][label]={'median_ms':median*1000,'extra_bytes':extra,'warmups':b['warmup_iterations'],'measurements':len(b['seconds']),'updates':b['optimizer_updates'],'batch_shape':b['batch_shape'],'device':b['device'],'dtype':b['dtype']}

assets=raw('assets/training/manifest.json')
out['asset_license_boundaries']={a['id']:a['license'] for a in assets['assets']}
out['asset_contains_checkpoint_files']=any(f['path'].endswith(('.pt','.safetensors')) for a in assets['assets'] for f in a['files'])
assert not out['asset_contains_checkpoint_files']
out['prerequisite_code_executions']={}
for id,file in [('2.1','course/chapters/02.md'),('3.1','course/chapters/03.md'),('15.1','course/chapters/15.md')]:
    s=Path(file).read_text();start=s.index('## '+id+' ');end=s.find('\n## ',start+1);section=s[start:end if end>=0 else None]
    stream=io.StringIO()
    with contextlib.redirect_stdout(stream):
        for code in re.findall(r'```python\n(.*?)```',section,re.S):exec(compile(code,file+'#'+id,'exec'),{})
    out['prerequisite_code_executions'][id]=stream.getvalue()

out['environment']={'python':sys.version.split()[0],'torch':torch.__version__,'device':'cpu','cuda_available':str(torch.cuda.is_available()),'threads':str(torch.get_num_threads())}
out['original_input_receipts']=inventory
Path('docs/technical-reviews/artifacts/natural-v4-factual/R.4/probe-results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(out,ensure_ascii=False,indent=2))
