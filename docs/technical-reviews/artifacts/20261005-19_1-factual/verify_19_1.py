"""Bounded CPU review: parser, protocol, data provenance, recorded outputs; no training or scoring."""
import contextlib
import hashlib
import importlib.util
import inspect
import io
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.capstone import CapstoneModel, TOK, build_dataset, calculator_runtime, modality_tensors, parse_action, prompt_ids, run_assistant
from tiny_perceptron.capstone_ui import input_preview, request_row
from scripts.fetch_capstone import fetch_capstone
from scripts.capstone_release import validate_manifest

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def snapshot(name):
    return OUT / 'repository-snapshots' / name
def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()
write('environment.json', {'python':platform.python_version(),'torch':str(torch.__version__),'torch_git_version':torch.version.git_version,'device':'cpu','cuda_build':str(torch.version.cuda),'threads':str(torch.get_num_threads()),'cwd':str(ROOT),'command':'.venv/bin/python docs/technical-reviews/artifacts/20261005-19_1-factual/verify_19_1.py'})
for item in json.loads((OUT/'snapshot-manifest.json').read_bytes()):
    assert sha(ROOT/item['original_path']) == item['sha256'] == sha(ROOT/item['snapshot_path'])

print('ORIGINAL PYTHON FENCE')
exec(compile((OUT/'fence-1.py').read_bytes(),'course/chapters/19.md#19.1:fence-1','exec'),{})
variants={
    'empty_direct':({'raw':'DIRECT:','eos':True},{'status':'invalid','reason':'malformed_action'}),
    'unterminated':({'raw':'DIRECT:red','eos':False},{'status':'invalid','reason':'unterminated_generation'}),
    'bound':({'raw':'TOOL:calculator:999+0','eos':True},{'status':'tool','name':'calculator','a':999,'b':0}),
    'out_of_bound':({'raw':'TOOL:calculator:1000+0','eos':True},{'status':'invalid','reason':'malformed_action'}),
    'non_allowlisted':({'raw':'TOOL:other:1+2','eos':True},{'status':'tool','name':'other','a':1,'b':2}),
}
checked={}
for name,(trace,expected) in variants.items():
    original=dict(trace); actual=parse_action(trace); assert actual==expected and trace==original;checked[name]=actual
assert calculator_runtime(checked['non_allowlisted'])=={'status':'error','reason':'tool_not_allowlisted'}
assert calculator_runtime(checked['bound'])=={'status':'ok','result':'999'}
assert calculator_runtime(parse_action({'raw':'TOOL:calculator:1+2','eos':True}))=={'status':'ok','result':'3'}
write('parser-variants.json',checked)
print('Parser variants:',json.dumps(checked,ensure_ascii=False))

# The stub supplies actions; only the runtime and protocol are exercised, not model capability.
calls=[]
def stub(model,row,max_new_tokens):
    calls.append(dict(row))
    return {'raw':'TOOL:calculator:1+2' if len(calls)==1 else 'DIRECT:3','eos':True}
row={'user':'1+2等於多少？','system':'計算器=開；風格=短。','available':True,'image':None,'audio':None}
with patch('tiny_perceptron.capstone.generate_trace',stub):
    loop=run_assistant(object(),row)
assert loop['runtime']=={'status':'ok','result':'3'} and loop['answer']=='3' and len(calls)==2
assert calls[1]['user']=='原題：1+2。計算器回報：3。請回答。'
write('protocol-variant.json',{'scope':'stubbed generation; real calculator runtime; not a model score','record':loop,'generation_inputs':calls})

splits,manifest=build_dataset()
assert 'numbers:1:2' in manifest['families']['test']['numbers']
assert all(r['family']!='numbers:1:2' for split in ['train','validation'] for r in splits[split])
assert len('大小上下左右開關入出人口')==len(set('大小上下左右開關入出人口'))==12
model=CapstoneModel(); description=model.description();assert description['parameters']==328128
print('Data split counts:',manifest['counts'],'old model parameters:',description['parameters'])
write('data-and-parameters.json',{'description':description,'data_version':manifest['version'],'split_counts':manifest['counts'],'test_number_families':manifest['families']['test']['numbers'],'reserved_family_descendants':[{k:r[k] for k in ['id','family','task','user','system','answer','available']} for r in splits['test'] if r['family']=='numbers:1:2'],'word_card_dictionary':'大小上下左右開關入出人口','dictionary_size':12,'dictionary_unique_count':12})

# Inspect already recorded raw generations. No model re-evaluation.
test=json.loads(snapshot('docs/course-experiments/capstone-evidence/deployment/test-joint.json').read_bytes())
assert test['count']==len(test['records'])==90
by_id={r['id']:r for r in test['records']}
selected=[]
for task in ['calculator','unavailable']:
    source_row=next(r for r in splits['test'] if r['family']=='numbers:1:2' and r['task']==task and r['user']=='1+2等於多少？')
    record=by_id[source_row['id']]
    assert record['action_trace']['raw']==source_row['answer'] and record['action_trace']['eos'] is True
    assert record['action_trace']['prompt_ids']==prompt_ids(source_row)
    assert record['action_trace']['generated_ids']==TOK.encode(record['action_trace']['raw'])+[TOK.eos_id]
    if task=='calculator':
        assert record['runtime']=={'status':'ok','result':'3'} and record['final_trace']['raw']=='DIRECT:3' and record['final_trace']['eos'] is True and record['answer']=='3'
        assert record['final_trace']['generated_ids']==TOK.encode('DIRECT:3')+[TOK.eos_id]
    else: assert record['runtime'] is None and record['final_trace'] is None
    selected.append({'pointer':'/records/'+str(test['records'].index(record)),'source_row':source_row,'record':record})
image_row=next(r for r in splits['test'] if r['family']=='modalities:green:square' and r['task']=='image_shape' and r['image']=={'color':'green','shape':'square','offset':0,'variant':1})
image_record=by_id[image_row['id']]
assert image_record['action_trace']['raw']=='DIRECT:circle' and image_record['action_trace']['eos'] is True
selected.append({'pointer':'/records/'+str(test['records'].index(image_record)),'source_row':image_row,'record':image_record})
write('selected-recorded-generations.json',{'original_sha256':sha(snapshot('docs/course-experiments/capstone-evidence/deployment/test-joint.json')),'checkpoint_sha256':test['checkpoint_sha256'],'records_total':90,'selected_records':selected,'scope':'original recorded 1+2 available/unavailable and green-square offset0 variant1; no new model scores'})
print('Raw recorded checks:',[(r['pointer'],r['record']['action_trace']['raw']) for r in selected])

# Public-export provenance: inspect existing original export, no new weights or downloads.
export=ROOT/'outputs/integration-runs/v2-deployment/capstone-review/capstone_deployment/gha-37169529991-1/joint.pt'
assert sha(export)=='7476cc488663eac0aff740f39b45054a6ee9d8c252e71fb99f86fd3953fff274'
payload=torch.load(export,map_location='cpu',weights_only=True)
assert payload['metadata']['source_checkpoint_sha256']==test['checkpoint_sha256']
assert payload['config']==asdict(model.config)
fingerprints=[{'name':name,'shape':list(t.shape),'dtype':str(t.dtype),'numel':t.numel(),'element_size_bytes':t.element_size(),'sha256':hashlib.sha256(t.detach().contiguous().numpy().tobytes()).hexdigest()} for name,t in sorted(payload['model'].items())]
assert all(t['dtype']=='torch.float32' and t['element_size_bytes']==4 for t in fingerprints)
assert sum(t['numel'] for t in fingerprints)==328128
public=json.loads(snapshot('docs/course-experiments/capstone-public.json').read_bytes());validate_manifest(public)
assert len(public['models'])==11
joint=next(m for m in public['models'] if m['id']=='joint')
public_joint=next(f for f in joint['files'] if f['output']==joint['checkpoint'])
measure=json.loads(snapshot('docs/course-experiments/student-checks/capstone-public-tensors.json').read_bytes())
assert measure['public_revision']==public['revision']
mrecord=next(r for r in measure['records'] if r['stage']=='joint')
assert mrecord['approved_source_sha256']==sha(export) and mrecord['public_sha256']==public_joint['sha256'] and mrecord['passed'] is True
assert {'config','model','tokenizer','data_version','stage','step'}<=set(mrecord['exact_tensor_and_config_fields'])
program=snapshot('docs/course-experiments/student-checks/capstone-public-tensors-program.txt')
assert sha(program)==measure['verification_program']['sha256']
write('checkpoint-provenance.json',{'scope':'independent existing-export CPU tensor inspection plus raw exact-equality measurement/program; no new model score or weight artifact','source_test_checkpoint_sha256':test['checkpoint_sha256'],'local_export_original_path':str(export.relative_to(ROOT)),'local_export_sha256':sha(export),'source_checkpoint_metadata_pointer':'/metadata/source_checkpoint_sha256','config':payload['config'],'parameters':sum(t['numel'] for t in fingerprints),'tensor_bytes':sum(t['numel']*t['element_size_bytes'] for t in fingerprints),'tensor_fingerprints':fingerprints,'public_revision':public['revision'],'public_joint_checkpoint_sha256':public_joint['sha256'],'raw_measurement_pointer':'/records/'+str(measure['records'].index(mrecord)),'raw_measurement':mrecord,'program_sha256':sha(program),'published_variants':len(public['models'])})
print('Public provenance:',{'test_checkpoint':test['checkpoint_sha256'],'source_export':sha(export),'public_checkpoint':public_joint['sha256'],'parameters':328128,'tensor_bytes':1312512,'published_variants':len(public['models'])})

# Inputs are real tensors; labels are used to construct them, not appended to text.
ui_row=request_row({'prompt':'圖片是什麼形狀？','image':{'color':'green','shape':'square'},'audio':{'frequency':440}})
image,audio=modality_tensors(ui_row);preview=input_preview(ui_row)
assert image.shape==(3,16,16) and audio.shape==(16,)
assert torch.count_nonzero(image[0])==torch.count_nonzero(image[2])==0
assert torch.count_nonzero(image[1])==81
assert len(preview['audio_waveform'])==640 and preview['sample_rate']==16000 and preview['duration_seconds']==0.04
text_ids=prompt_ids(ui_row);assert TOK.image_id in text_ids and TOK.audio_id in text_ids
assert TOK.decode([x for x in text_ids if x>=8])==ui_row['system']+ui_row['user']
other=request_row({'prompt':'圖片是什麼形狀？','image':{'color':'red','shape':'circle'},'audio':{'frequency':880}})
assert prompt_ids(other)==text_ids
image2,audio2=modality_tensors(other);assert not torch.equal(image,image2) and not torch.equal(audio,audio2)
write('ui-input-variant.json',{'prompt_same_across_changed_materials':True,'prompt_decoded':TOK.decode([x for x in text_ids if x>=8]),'image_shape':list(image.shape),'green_nonzero_pixels':81,'image_green_value':float(image[1].max()),'audio_features_shape':list(audio.shape),'waveform_samples':640,'sample_rate_hz':16000,'duration_seconds':0.04,'changed_materials_change_image_and_audio_features':True})
print('UI tensors and text:',{'image':[3,16,16],'audio_features':[16],'waveform_samples':640,'prompt_unchanged_after_material_change':True})

# No download: test early existing-directory and byte-hash failure paths.
blocked=OUT/'fetch-existing';(blocked/'joint').mkdir(parents=True,exist_ok=True)
with patch('huggingface_hub.hf_hub_download',side_effect=AssertionError('must not download')):
    try:fetch_capstone(public,'joint',blocked)
    except ValueError as e:assert 'already exists' in str(e)
    else:raise AssertionError('must refuse overwrite')
fixture=OUT/'invalid-download.bin';fixture.write_bytes(b'not the published bytes')
with patch('huggingface_hub.hf_hub_download',return_value=str(fixture)):
    try:fetch_capstone(public,'joint',OUT/'fetch-hash-test')
    except ValueError as e:assert 'SHA/size' in str(e)
    else:raise AssertionError('must reject wrong bytes')
write('fetch-path-checks.json',{'existing_directory':'refused before hf_hub_download','sha_size_mismatch':'rejected with a mocked local non-model byte fixture','network_downloads':0})

for argv in [[str(ROOT/'.venv/bin/python'),'scripts/fetch_capstone.py','--list'],[str(ROOT/'.venv/bin/python'),'scripts/capstone.py','infer','--help'],[str(ROOT/'.venv/bin/python'),'scripts/capstone.py','serve','--help']]:
    proc=subprocess.run(argv,cwd=ROOT,capture_output=True,text=True,timeout=30);assert proc.returncode==0,proc.stderr
    name='cli-'+('fetch-list' if '--list' in argv else argv[-2])
    (OUT/(name+'-stdout.txt')).write_text(proc.stdout);(OUT/(name+'-stderr.txt')).write_text(proc.stderr)
print('CLI --list and infer/serve parser checks passed; no checkpoint inference/download/server run')
print('All bounded CPU assertions passed; no training, GPU, network downloads, or new model scores.')
