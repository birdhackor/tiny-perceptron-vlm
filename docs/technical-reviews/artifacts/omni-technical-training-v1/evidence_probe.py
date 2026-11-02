"""Read primary evidence and run small deterministic/CLI checks for training.md."""

import copy
import hashlib
import inspect
import json
import math
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from scripts.evaluate import answer_sample, evaluate
from scripts.prepare_data import generate_records
from tiny_perceptron.data import ByteTokenizer, render_chat, split_documents, toy_documents
from tiny_perceptron.modal_data import modal_example
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.tokenization import generation_report
from tiny_perceptron.training import load_checkpoint

REPO = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
SCRATCH = Path('/workspace/work/omni-technical-training-v1')
torch.set_num_threads(2)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def small_run(label, args, allowed=(0,)):
    process = subprocess.run([sys.executable, *map(str, args)], cwd=SCRATCH,
        env=dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2'), text=True,
        capture_output=True, timeout=40)
    row = {'label':label,'command':[sys.executable,*map(str,args)],'cwd':str(SCRATCH),
           'returncode':process.returncode,'stdout':process.stdout,'stderr':process.stderr}
    assert process.returncode in allowed, row
    return row


expected_times = {'simple_models':4.3,'text_foundation':16.0,'real_text':23.9,'tokenizer':16.2,
 'sft':31.7,'sft_ablation':23.5,'style':29.5,'safety':22.2,'lora':24.5,'encoders':10.2,
 'projector':15.9,'vqa':83.8,'dpo':18.7,'posttraining':2.1,'modern':44.6,'moe':65.7,
 'efficiency':32.3,'precision':15.0,'flash_probe':4.5,'quantization':7.8,
 'distillation':90.1,'joint':33.4,'multimodal_distillation':53.0}


def denominator_fields(value, prefix='results'):
    rows = {}
    if isinstance(value, dict):
        for key, item in value.items():
            path = prefix+'/'+key
            if key in ('steps','requested_steps','examples','records','effective_tokens','train_examples',
                       'holdout_examples','train_records','validation_records','test_records','count',
                       'batch_size','warmup_calls','repeats','measured_calls','selected_rows',
                       'student_training_records') and isinstance(item,(int,float,str)):
                rows[path] = item
            elif key not in ('samples','generated_samples','skipped','history','artifacts'):
                rows.update(denominator_fields(item,path))
    return rows


history = {}
for name, expected in expected_times.items():
    path = REPO/'docs/course-experiments/results'/f'{name}.json'
    result = json.loads(path.read_text())
    assert round(result['elapsed_seconds'],1) == expected
    fields = ('experiment_id','revision','device','gpu','seed','torch_version','python_version',
              'elapsed_seconds','timing_scope','evidence_status','step_scale')
    history[name] = {'path':str(path.relative_to(REPO)),'sha256':sha(path),
        'metadata':{key:result.get(key) for key in fields},'rounded_time':expected,
        'denominators':denominator_fields(result['results'])}

modern = json.loads((REPO/'docs/course-experiments/results/modern.json').read_text())['results']
moe = json.loads((REPO/'docs/course-experiments/results/moe.json').read_text())['results']
assert [modern['dataset'][s]['records'] for s in ('train','validation','test')] == [409,51,52]
assert len(modern['variants']) == 6 and len(moe['variants']) == 7
assert set(v['training']['steps'] for v in modern['variants'].values()) == {240}
assert set(v['training']['steps'] for v in moe['variants'].values()) == {180}
assert all(v['model']['config']['width']==64 and v['model']['config']['layers']==2
           and v['model']['config']['heads']==4 for v in modern['variants'].values())

tok = ByteTokenizer()
samples = {'A_ids':tok.encode('A'),'EOS':tok.eos_id,'Chinese_red_utf8_bytes':list('紅'.encode()),
           'hand_generation':generation_report(tok,[73,2]),
           'literal_whitespace_not_removed':answer_sample(tok,tok.encode(' 5')+[2],'5'),
           'matching_without_eos':answer_sample(tok,tok.encode('5'),'5')}
assert samples['A_ids']==[73] and samples['EOS']==2
assert samples['literal_whitespace_not_removed']['exact_match'] is False
assert samples['matching_without_eos']['exact_match'] is True
assert samples['matching_without_eos']['completed_exact_match'] is False
chat = generate_records('attributes-sft')[0]
x,y=render_chat(chat['messages'],tok)
samples['sft_targets'] = {'question':chat['messages'][0]['content'],
    'answer':chat['messages'][1]['content'],'target_ids':y[y!=-100].tolist(),
    'expected_answer_plus_eos':tok.encode(chat['messages'][1]['content'])+[tok.eos_id]}
assert samples['sft_targets']['target_ids'] == samples['sft_targets']['expected_answer_plus_eos']
samples['document_splits'] = {s:len(v) for s,v in split_documents(toy_documents(),42).items()}
for kind in ('toy-text','attributes-sft','style','safety','preference'):
    parts={s:json.loads((SCRATCH/f'data/generated/{kind}/manifest.json').read_text())['splits'][s]
           for s in ('train','validation','test')}
    samples[kind+'_manifest'] = parts
samples['hand_examples']={'smaller_file':60000<80000,'accuracy_not_preserved':3/5<4/5,
    'byte_reduction':80000-60000,'old_accuracy':4/5,'new_accuracy':3/5,
    'two_nll_means':{'0.8':(0.7+0.9)/2,'0.6':(0.5+0.7)/2},
    'same_total_different_correct_sets':{'first':[0,1,2,3],'second':[0,1,2,4]}}
p=torch.nn.Parameter(torch.tensor(2.));q=torch.nn.Parameter(torch.tensor(3.))
before=(p.detach().clone(),q.detach().clone())
(p*p).backward()
samples['backward_without_update']={'used_grad':p.grad.item(),'unused_grad':str(q.grad),
    'weights_unchanged':bool(p==before[0] and q==before[1])}
start=torch.load(SCRATCH/'checkpoints/start.pt',weights_only=True,map_location='cpu')
stage=torch.load(SCRATCH/'checkpoints/text-then-sft.pt',weights_only=True,map_location='cpu')
joint=torch.load(SCRATCH/'checkpoints/joint.pt',weights_only=True,map_location='cpu')
samples['checkpoint_start']={'config':start['config'],'step':start['step'],
    'optimizer_is_none':start['optimizer'] is None,'training_state':start['training_state']}
samples['new_stage']={'step':stage['step'],'optimizer_step_counts':sorted({int(v['step'])
    for v in stage['optimizer']['state'].values()}),'metadata':{k:v for k,v in stage['metadata'].items() if k!='report'}}
samples['joint_checkpoint']={'format_version':joint['format_version'],'step':joint['step'],
    'optimizer_saved':joint['optimizer'] is not None,'rng_fields':sorted(k for k in joint if 'rng' in k),
    'trainable_parameter_prefixes':sorted({k.split('.')[0] for k in joint['trainable_parameters']})}

runs=[]
for experiment in ('simple_models','posttraining','flash_probe'):
    runs.append(small_run('fixed-'+experiment,['-m','scripts.course_experiments.run','--experiment',experiment,
        '--device','cpu','--output',str(SCRATCH/'fixed')]))
flash=json.loads((SCRATCH/'fixed/flash_probe/result.json').read_text())
samples['cpu_flash_scope']={k:flash[k] for k in ('device','evidence_status','experiment_kind','timing_scope')}
assert flash['evidence_status']=='interface_smoke_only'
runs.append(small_run('ocr-materials',[REPO/'scripts/prepare_ocr.py','--output','data/generated/ocr']))
for ablation in ('none','blank','shuffle'):
    runs.append(small_run('ocr-'+ablation,[REPO/'scripts/evaluate_modal.py','checkpoints/vision.modal.pt',
        '--data','data/generated/ocr/validation.jsonl','--ablation',ablation,'--tokens','2','--limit','2',
        '--device','cpu','--output',f'outputs/ocr-{ablation}.json']))
source_records=[json.loads(line) for line in (SCRATCH/'data/generated/ocr/validation.jsonl').read_text().splitlines()]
ocr=json.loads((SCRATCH/'outputs/ocr-shuffle.json').read_text())
assert all(s['target']==source_records[s['row']]['answer'] and s['donor_row']!=s['row'] for s in ocr['samples'])
samples['ocr_shuffle']={'examples':ocr['examples'],'samples':ocr['samples']}
image=SCRATCH/'data/generated/ocr'/source_records[0]['image']
sf.write(SCRATCH/'valid16k.wav',np.zeros(1600,dtype=np.float32),16000)
sf.write(SCRATCH/'invalid8k.wav',np.zeros(800,dtype=np.float32),8000)
audio_checks={}
for name in ('valid16k.wav','invalid8k.wav'):
    try:
        _,_,_,wave=modal_example({'audio':name,'question':'pitch?','answer':'low'},SCRATCH,'audio')
        audio_checks[name]={'accepted':True,'samples':wave.numel()}
    except ValueError as error:
        audio_checks[name]={'accepted':False,'error':str(error)}
samples['audio_input_validation']=audio_checks
assert audio_checks['valid16k.wav']['accepted'] and not audio_checks['invalid8k.wav']['accepted']

original=json.loads((REPO/'docs/course-experiments/results/tools.json').read_text())
tool_checks=[]
for key in ('samples','one_step_cap_samples'):
    for episode in original['results'][key]:
        first=next(e['generation'] for e in episode['trace'] if 'generation' in e)
        messages=copy.deepcopy(first['messages'][:2])
        for e in episode['trace']:
            if 'generation' not in e:continue
            roles={'system':7,'user':3,'assistant':4}
            ids=[1]
            for m in messages:ids += [roles[m['role']]]+tok.encode(m['content'])+[2]
            ids += [4]
            g=e['generation']
            assert ids==g['input_ids'] and len(ids)==g['input_tokens']
            tool_checks.append({'ids_equal':True,'messages_differ':messages!=g['messages']})
            if e.get('executed') and e.get('finite_result'):
                messages += [{'role':'assistant','content':g['samples'][0]['generated']},
                             {'role':'user','content':f"TOOL_RESULT:{e['tool_result']}"}]
samples['historical_tool_snapshot']={'generations':len(tool_checks),
    'messages_mismatches':sum(x['messages_differ'] for x in tool_checks),
    'all_actual_ids_equal':all(x['ids_equal'] for x in tool_checks),
    'source_sha256':sha(REPO/'docs/course-experiments/results/tools.json')}
assert samples['historical_tool_snapshot']['generations']==56
assert samples['historical_tool_snapshot']['messages_mismatches']==36

result={'created_at':datetime.now(timezone.utc).isoformat(),
    'environment':{'python':platform.python_version(),'torch':str(torch.__version__),'device':'cpu'},
    'historical_results':history,'numeric_and_software_checks':samples,'extra_cli_runs':runs,
    'scope':'Historical times are inspection of original stored reports, not a GPU rerun. '
            'Two-step fixture answers are not quality measurements; no full model or new module training.'}
(HERE/'evidence-probe-results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'historical_time_records':len(history),'extra_cli_runs':len(runs),
                  'tool_generations_checked':len(tool_checks),'result_path':str(HERE/'evidence-probe-results.json')}))
