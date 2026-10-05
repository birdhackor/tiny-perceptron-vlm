"""Bounded CPU arithmetic, axes, packing, loss, and saved-record verification."""
import hashlib
import io
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import torch
from torch import nn
from tiny_perceptron.quantization import QuantizedLinear, quantize_symmetric, pack_int4, unpack_int4
from tiny_perceptron.alignment import distillation_loss, distillation_kl
from tiny_perceptron.model import masked_loss
from tiny_perceptron.capstone import calculator_runtime, build_dataset
from scripts.capstone_release import _metadata

assert torch.version.cuda is None
torch.set_num_threads(1)
torch.set_default_device('cpu')
environment = {'python': platform.python_version(), 'torch': str(torch.__version__), 'torch_git_version': str(torch.version.git_version), 'cuda_build': str(torch.version.cuda), 'device': 'cpu', 'threads': str(torch.get_num_threads())}
(OUT / 'verify.environment.json').write_text(json.dumps(environment, indent=2) + '\n')
torch.manual_seed(42)
layer = nn.Linear(8, 4)
x = torch.ones(1, 8)
results = {'environment': environment, 'input': x.tolist(), 'fp32_bytes': sum(p.numel()*p.element_size() for p in layer.parameters()), 'quantization': {}}
assert layer.weight.shape == (4,8) and layer.bias.shape == (4,)
assert layer.weight.dtype == torch.float32 and results['fp32_bytes'] == 144
for bits, expected in [(4,48),(8,64)]:
    compressed=QuantizedLinear(layer,bits=bits)
    q, scale=quantize_symmetric(layer.weight.detach(), bits, per_channel=True)
    with torch.no_grad():
        baseline=layer(x); output=compressed(x)
        restored=q.float()*scale
        manual=x@restored.T+layer.bias
        reference=x@layer.weight.detach().T+layer.bias
    buffers={name:{'shape':list(b.shape),'dtype':str(b.dtype),'numel':b.numel(),'element_size':b.element_size(),'bytes':b.numel()*b.element_size()} for name,b in compressed.named_buffers()}
    assert scale.shape==(4,1) and scale.dtype==torch.float32
    assert buffers['values']['bytes']==(16 if bits==4 else 32)
    assert buffers['scale']['bytes']==16 and buffers['bias']['bytes']==16
    assert compressed.storage_bytes()==expected
    torch.testing.assert_close(manual,output,rtol=0,atol=1e-7)
    torch.testing.assert_close(reference,baseline,rtol=0,atol=1e-7)
    if bits==4: assert torch.equal(unpack_int4(compressed.values,list(q.shape)),q)
    delta=(baseline-output).abs()
    max_error=delta.max().item()
    weight_error=(layer.weight.detach()-restored).abs()
    assert bool((weight_error <= scale/2 + 1e-7).all())
    # x contains eight 1s: each output has eight weight terms, unchanged bias.
    assert bool((delta <= weight_error.sum(-1).unsqueeze(0) + 1e-7).all())
    results['quantization'][str(bits)]={'buffers':buffers,'storage_bytes':compressed.storage_bytes(),'weight_shape':list(layer.weight.shape),'scale':scale.tolist(),'integer_range':[int(q.min()),int(q.max())],'original_output':baseline.tolist(),'quantized_output':output.tolist(),'manual_output':manual.tolist(),'absolute_output_error':delta.tolist(),'max_error':max_error,'max_error_axis':'max over the 1 by 4 output tensor; one input sample, four output coordinates','output_unit':'linear layer activation units, not percentage or rate','rounding_weight_bound_max':float((scale/2).max()),'absolute_weight_error_row_sum':weight_error.sum(-1).tolist()}
assert results['quantization']['8']['max_error'] < results['quantization']['4']['max_error']
integers=torch.tensor([-8,-7,-1,0,1,7,0],dtype=torch.int8)
assert torch.equal(unpack_int4(pack_int4(integers),list(integers.shape)),integers)
results['odd_int4_roundtrip']={'integers':integers.tolist(),'packed':pack_int4(integers).tolist(),'bytes':pack_int4(integers).numel()}

# Independent expression checks KL direction, valid-token denominator, T² once.
student=torch.tensor([[[.3,-.5,.2],[1.,-.2,.4],[.0,.6,-.4]]],requires_grad=True)
teacher=torch.tensor([[[.5,-.3,.1],[.1,.8,-.2],[1.,.2,-.7]]],requires_grad=True)
labels=torch.tensor([[-100,1,0]])
T=2.; valid=labels!=-100
p=(teacher.detach()/T).softmax(-1); logq=(student/T).log_softmax(-1)
raw_kl=(p*(p.log()-logq)).sum(-1)[valid].mean()
manual=.5*masked_loss(student,labels)+.5*T*T*raw_kl
actual=distillation_loss(student,teacher,labels,alpha=.5,temperature=T)
torch.testing.assert_close(actual,manual,atol=1e-7,rtol=0)
actual.backward(); assert teacher.grad is None
results['distillation']={'valid_tokens':int(valid.sum()),'vocabulary':student.shape[-1],'T':T,'alpha':.5,'raw_forward_kl':raw_kl.item(),'scaled_kl':distillation_kl(student,teacher,labels,T).item(),'actual_loss':actual.item(),'independent_expression':manual.item(),'teacher_gradient':None,'method':'sum across vocabulary; mean of two valid answer positions; T² included once'}

def raw(path):return json.loads((OUT/'frozen'/path).read_text())
def check_records(path):
    d=raw(path);rs=d['records'];assert len(rs)==d['count']==90
    assert sum(r['action_correct'] for r in rs)==d['action_correct']
    assert sum(r['end_to_end_correct'] for r in rs)==d['end_to_end_correct']
    assert len({r['id'] for r in rs})==90
    for r in rs:
        assert r['action_correct']==(r['action_trace']['eos'] and r['action_trace']['raw']==r['expected_action'])
        assert r['end_to_end_correct']==(r['action_correct'] and r['answer']==r['expected_final'])
    for task,s in d['by_task'].items():
        subset=[r for r in rs if r['task']==task]
        assert len(subset)==s['count']
        for metric in ('action_correct','end_to_end_correct'):assert sum(r[metric] for r in subset)==s[metric]
    return d
paths={'joint':'docs/course-experiments/capstone-evidence/deployment/test-joint.json','int4':'docs/course-experiments/capstone-evidence/deployment/test-joint-ptq4.json','ce':'docs/course-experiments/capstone-evidence/student/test-ce.json','kd':'docs/course-experiments/capstone-evidence/student/test-kd.json'}
data={name:check_records(path) for name,path in paths.items()}
assert [r['id'] for r in data['joint']['records']]==[r['id'] for r in data['int4']['records']]
results['saved_evaluations']={name:{'count':d['count'],'action_correct':d['action_correct'],'end_to_end_correct':d['end_to_end_correct']} for name,d in data.items()}
for name, action, expected_answer in [('joint','TOOL:calculator:1+0','111'),('int4','TOOL:calculator:1+0','11'),('ce','TOOL:calculator:1+8','9'),('kd','TOOL:calculator:1+8','11')]:
    matches=[(i,r) for i,r in enumerate(data[name]['records']) if r['expected_action']==action];assert len(matches)==1
    i,r=matches[0];assert r['answer']==expected_answer
    results.setdefault('named_records',{})[name]={'pointer':f'/records/{i}','id':r['id'],'expected_action':r['expected_action'],'expected_final':r['expected_final'],'action_raw':r['action_trace']['raw'],'runtime':r['runtime'],'final_raw':r['final_trace']['raw'],'answer':r['answer'],'action_correct':r['action_correct'],'end_to_end_correct':r['end_to_end_correct']}
assert data['joint']['records'][20]['runtime']['result']==data['int4']['records'][20]['runtime']['result']=='1'
assert data['ce']['records'][6]['action_trace']['raw']=='TOOL:calculator:1+8'
assert data['kd']['records'][6]['action_trace']['raw']=='TOOL:calculator:1+9'
assert data['kd']['records'][6]['runtime']['result']=='10'
for name in ['joint','int4','ce','kd']:
    index=20 if name in ['joint','int4'] else 6
    r=data[name]['records'][index]
    assert calculator_runtime(r['parsed_action'])==r['runtime']
dep=raw('docs/course-experiments/results/capstone_deployment.json')
stu=raw('docs/course-experiments/results/capstone_student.json')
splits, manifest=build_dataset(42)
assert manifest==dep['results']['data_manifest']==stu['results']['data_manifest']
families={name:{row['family'] for row in rows} for name,rows in splits.items()}
assert not (families['train'] & families['validation'] or families['train'] & families['test'] or families['validation'] & families['test'])
test_rows={r['id']:r for r in splits['test']}
for d in data.values():
    assert set(test_rows)=={r['id'] for r in d['records']}
    for r in d['records']:assert test_rows[r['id']]['answer']==r['expected_action']
results['dataset_method_check']={'seed':42,'counts':manifest['counts'],'split_unit':manifest['split_unit'],'family_overlap':False,'manifest_matches_both_saved_reports':True,'named_users':{name:test_rows[value['id']]['user'] for name,value in results['named_records'].items()},'scope':'Deterministic local data construction and saved-record matching only; no training or model-score reevaluation.'}
for path in paths.values():
    file_path=Path(path); report=dep if '/deployment/' in path else stu
    receipt=next(a for a in report['artifacts'] if a['path']==file_path.name)
    assert hashlib.sha256((OUT/'frozen'/path).read_bytes()).hexdigest()==receipt['sha256']
assert data['joint']['checkpoint_sha256']==data['int4']['source_checkpoint_sha256']
for mode in ['ce','kd']:
    branch=stu['results']['branches'][mode]
    assert branch['steps']==branch['requested_steps']==350 and branch['schedule_completed']
    assert branch['teacher_checkpoint_sha256']==stu['results']['teacher_checkpoint_sha256']
results['student_provenance']={'teacher_checkpoint_sha256':stu['results']['teacher_checkpoint_sha256'],'initial_student_state_sha256':stu['results']['initial_student_state_sha256'],'seed':stu['seed'],'step_scale':stu['step_scale'],'steps_per_branch':stu['results']['steps_per_branch'],'branches':{mode:{key:stu['results']['branches'][mode][key] for key in ['steps','requested_steps','schedule_completed','parameters','effective_tokens','teacher_checkpoint_sha256']} for mode in ['ce','kd']}}
preference=raw('docs/course-experiments/results/capstone_preference.json')
assert preference['results']['stage']=='dpo'
assert preference['results']['inference_export']['sha256']==stu['results']['teacher_checkpoint_sha256']
results['student_provenance']['teacher_stage']='dpo'
results['code_provenance_matches']={}
for source in ['scripts/course_experiments/capstone.py','scripts/course_experiments/capstone_deployment.py','scripts/course_experiments/capstone_student.py','tiny_perceptron/alignment.py','tiny_perceptron/capstone.py','tiny_perceptron/capstone_quantization.py','tiny_perceptron/quantization.py']:
    h=hashlib.sha256((OUT/'frozen'/source).read_bytes()).hexdigest()
    assert h==dep['code_sha256'][source]==stu['code_sha256'][source]
    results['code_provenance_matches'][source]=h
# Test the actual metadata allowlist without saving any weight file.
saved={'metadata':{'seed':42,'data_manifest':{'counts':{'test':90}},'training_history':[1,2]}}
clean_metadata=_metadata(saved,{})
assert 'training_history' not in clean_metadata and 'data_manifest' not in clean_metadata
assert clean_metadata['seed']==42 and 'dataset_manifest_sha256' in clean_metadata
before=io.BytesIO(); after=io.BytesIO()
torch.save(saved,before);torch.save({'metadata':clean_metadata},after)
assert before.getvalue()!=after.getvalue()
results['metadata_normalization']={'input_keys':list(saved['metadata']),'output_keys':list(clean_metadata),'in_memory_serialized_bytes_differ':True,'scope':'Only metadata dictionaries were serialized in BytesIO; no model, .pt file or public release was generated.'}
(OUT/'verify.results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps(results,ensure_ascii=False,indent=2,allow_nan=False))
print('All bounded CPU arithmetic, packing, loss, denominator and stored-record assertions passed.')
