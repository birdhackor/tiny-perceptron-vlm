import copy
import dataclasses
import hashlib
import json
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import huggingface_hub
import numpy as np
import safetensors
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

REPO_ROOT = Path.cwd()
OUT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))
from scripts.selftrained import train, train_local_stage
from tiny_perceptron.selftrained.inference import InferenceAssistant, fetch_public_export, verify_export

torch.set_num_threads(2)
environment = {'python': sys.version.split()[0], 'torch': torch.__version__,
               'safetensors': safetensors.__version__, 'huggingface_hub': huggingface_hub.__version__, 'device': 'cpu'}
CACHE = Path('/tmp/p5-native-public-cpu-smoke-actual/public-model')
REV = '979cdfacc588ad0536f1c64fff96f264571cf054'
PIN = 'f27856892b0c46ca81ba43bfadf7051ed6bd31d848e4868a05147e5799df1e5e'
checks = {}

def rejects(name, function, exception=ValueError):
    try:
        function()
    except exception as error:
        checks[name] = {'rejected': True, 'exception': type(error).__name__, 'message': str(error)}
    else:
        raise AssertionError(name + ' did not reject')

manifest, digest = verify_export(CACHE, manifest_sha256=PIN)
state = load_file(CACHE/'model.safetensors', device='cpu')
assistant = InferenceAssistant(CACHE, '/tmp/p6-19.11-no-assets-present', device='cpu', manifest_sha256=PIN)
assert set(state) == set(assistant.model.state_dict())
assert not any(name.startswith(('optimizer', 'rng', 'sampler')) for name in state)
with safe_open(CACHE/'model.safetensors', framework='pt', device='cpu') as archive:
    checks['complete_neural_state'] = {'tensor_count': len(state), 'keys_match': True,
        'keys': list(state), 'metadata': archive.metadata(), 'manifest_sha256': digest,
        'tensor_sha256': hashlib.sha256((CACHE/'model.safetensors').read_bytes()).hexdigest(),
        'optimizer_rng_sampler_tensor_keys': []}
with (CACHE/'model.safetensors').open('rb') as handle:
    length_bytes = handle.read(8)
    header = handle.read(int.from_bytes(length_bytes,'little'))
(OUT/'safetensors-header.json').write_bytes(header)
for name in ['model-config.json','tokenizer.json','inference-manifest.json']:
    original = CACHE/name
    destination = OUT/('cache-'+name)
    shutil.copyfile(original,destination)
    assert hashlib.sha256(original.read_bytes()).digest() == hashlib.sha256(destination.read_bytes()).digest()

rejects('unpinned_branch_revision', lambda: fetch_public_export('birdhackor/tiny-perceptron-course-models','main','unused'))
rejects('wrong_manifest_pin', lambda: verify_export(CACHE,manifest_sha256='0'*64))
with tempfile.TemporaryDirectory(prefix='p6-19.11-') as tmp:
    root=Path(tmp)
    changed=root/'changed'
    shutil.copytree(CACHE,changed)
    with (changed/'model-config.json').open('ab') as handle:handle.write(b' ')
    rejects('payload_hash_mismatch',lambda:verify_export(changed,manifest_sha256=PIN))
    altered=root/'missing-tensor'
    altered.mkdir()
    for name in ['model-config.json','tokenizer.json','inference-manifest.json']:shutil.copyfile(CACHE/name,altered/name)
    missing=copy.copy(state)
    deleted=next(iter(missing))
    del missing[deleted]
    save_file(missing,str(altered/'model.safetensors'),metadata={'origin':'all-neural-weights-random','schema':'selftrained-random-v1'})
    altered_manifest=json.loads((altered/'inference-manifest.json').read_text())
    altered_manifest['files']['model.safetensors']=hashlib.sha256((altered/'model.safetensors').read_bytes()).hexdigest()
    (altered/'inference-manifest.json').write_text(json.dumps(altered_manifest))
    verify_export(altered)
    rejects('valid_hash_incomplete_model',lambda:InferenceAssistant(altered,root/'no-assets'))
    checks['valid_hash_incomplete_model']['deleted_tensor']=deleted

    destination=root/'actual-public-cli'
    argv=[sys.executable,'scripts/selftrained/chat.py','--model-dir',str(destination),
          '--asset-dir',str(root/'absent-assets'),'--repo','birdhackor/tiny-perceptron-course-models',
          '--revision',REV,'--prefix','selftrained/v2/moe-joint','--manifest-sha256',PIN,
          '--messages','docs/selftrained/examples/v2/text.messages.json','--task','text',
          '--device','cpu','--max-new-tokens','128','--threads','2']
    (OUT/'current-cli-argv.json').write_text(json.dumps({'argv':argv,'cwd':str(REPO_ROOT)},indent=2)+'\n')
    process=subprocess.run(argv,capture_output=True,timeout=90)
    (OUT/'current-cli-stdout.json').write_bytes(process.stdout)
    (OUT/'current-cli-stderr.txt').write_bytes(process.stderr)
    assert process.returncode == 0, process.stderr.decode()
    result=json.loads(process.stdout)
    expected='1. 先檢查網路並重新啟動App。\n2. 仍失敗再詢問官方客服。'
    assert result['answer'] == expected
    assert result['public_source']['authentication'] == 'disabled'
    assert result['public_source']['revision'] == REV
    assert len(result['generations']) == 1 and result['generations'][0]['modality_kinds'] == []
    checks['original_known_text_cli']={'returncode':process.returncode,'answer':result['answer'],
         'generation_count':len(result['generations']),'generated_tokens':len(result['generations'][0]['generated_ids']),
         'generation_status':result['generations'][0]['generation_status'],
         'assets_directory_exists':(root/'absent-assets').exists(),'public_source':result['public_source']}
    short=argv.copy();short[short.index('--max-new-tokens')+1]='1'
    (OUT/'short-cli-argv.json').write_text(json.dumps({'argv':short,'cwd':str(REPO_ROOT)},indent=2)+'\n')
    shorter=subprocess.run(short,capture_output=True,timeout=90)
    (OUT/'short-cli-stdout.json').write_bytes(shorter.stdout)
    (OUT/'short-cli-stderr.txt').write_bytes(shorter.stderr)
    assert shorter.returncode==0
    variation=json.loads(shorter.stdout)
    assert len(variation['generations'][0]['generated_ids'])==1
    assert variation['generations'][0]['generation_status']=='token_limit'
    checks['same_prompt_one_token_variation']={'answer':variation['answer'],'status':'token_limit','generated_tokens':1}
    checks['identical_destination_reuse']={'source':fetch_public_export('birdhackor/tiny-perceptron-course-models',REV,destination,prefix='selftrained/v2/moe-joint',manifest_sha256=PIN),'accepted':True}
    (destination/'tokenizer.json').write_text('different content')
    rejects('different_destination_refuses_overwrite',lambda:fetch_public_export('birdhackor/tiny-perceptron-course-models',REV,destination,prefix='selftrained/v2/moe-joint',manifest_sha256=PIN),FileExistsError)

    # These files are deliberately synthetic boundary fixtures, never trained checkpoints.
    source=root/'synthetic-local-source';source.mkdir()
    for name in ['best.pt','latest.pt']:(source/name).write_bytes(b'synthetic boundary fixture, not loaded')
    (source/'train-receipt.json').write_text(json.dumps({'completed_requested_steps':True,'interrupted':False}))
    execution={'backend':'local-subprocess','manifest_sha256':'a'*64,'status':'completed','returncode':0,'run_id':'synthetic','revision':'b'*40}
    (source/'execution.json').write_text(json.dumps(execution))
    def receipt():
        files=[{'path':p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in source.iterdir() if p.name!='receipt.json']
        (source/'receipt.json').write_text(json.dumps({**execution,'files':files}))
    receipt()
    checks['local_source_modes']={
        'best':train_local_stage.local_source(source/'best.pt','a'*64,False)['mode'],
        'latest':train_local_stage.local_source(source/'latest.pt','a'*64,True)['mode']}
    rejects('latest_is_not_fresh_best',lambda:train_local_stage.local_source(source/'latest.pt','a'*64,False))
    rejects('best_is_not_exact_latest',lambda:train_local_stage.local_source(source/'best.pt','a'*64,True))
    (source/'train-receipt.json').write_text(json.dumps({'completed_requested_steps':False,'interrupted':True}))
    execution.update(status='failed');(source/'execution.json').write_text(json.dumps(execution));receipt()
    rejects('incomplete_source_not_promoted',lambda:train_local_stage.local_source(source/'best.pt','a'*64,False))
    checks['incomplete_latest_can_resume']=train_local_stage.local_source(source/'latest.pt','a'*64,True)['mode']
    rejects('new_attempt_directory_required',lambda:source.mkdir(exist_ok=False),FileExistsError)

train.seed_all(29)
saved=train.rng_state()
expected_random=[random.random(),float(np.random.rand()),torch.rand(3).tolist()]
train.seed_all(97)
train.restore_rng(saved)
observed_random=[random.random(),float(np.random.rand()),torch.rand(3).tolist()]
assert expected_random==observed_random
checks['rng_restoration']={'expected':expected_random,'observed':observed_random,'exact_equal':True}
rows=[{'id':str(i),'task':'text','supervision':{}} for i in range(5)]
sampler=train.BalancedSampler(rows,seed=29);sampler.batch(7);saved_sampler=sampler.state_dict()
expected_ids=[r['id'] for r in sampler.batch(9)]
restored=train.BalancedSampler(rows,seed=97);restored.load_state_dict(saved_sampler)
observed_ids=[r['id'] for r in restored.batch(9)]
assert expected_ids==observed_ids and restored.draws==16
fresh=train.BalancedSampler(rows,seed=29)
assert fresh.draws==0
checks['sampler_restoration']={'saved_draws':7,'expected_ids':expected_ids,'observed_ids':observed_ids,'restored_draws':restored.draws,'fresh_stage_draws':fresh.draws}
# Populate state directly: no backward, optimizer.step, or parameter update.
parameter=torch.nn.Parameter(torch.tensor([1.,2.]))
optimizer=torch.optim.AdamW([parameter],lr=.001)
optimizer.state[parameter]={'step':torch.tensor(7.),'exp_avg':torch.tensor([.1,.2]),'exp_avg_sq':torch.tensor([.3,.4])}
saved_optimizer=optimizer.state_dict()
target_parameter=torch.nn.Parameter(torch.tensor([1.,2.]))
restored_optimizer=torch.optim.AdamW([target_parameter],lr=.01)
restored_optimizer.load_state_dict(saved_optimizer)
fresh_optimizer=torch.optim.AdamW([torch.nn.Parameter(torch.tensor([1.,2.]))],lr=.001)
assert torch.equal(restored_optimizer.state[target_parameter]['exp_avg'],torch.tensor([.1,.2])) and not fresh_optimizer.state
checks['optimizer_restore_versus_fresh']={'restored_step':restored_optimizer.state[target_parameter]['step'].item(),'restored_lr':restored_optimizer.param_groups[0]['lr'],'fresh_state_entries':len(fresh_optimizer.state),'updates_executed':0}
(OUT/'cpu-check-result.json').write_text(json.dumps({'environment':environment,'checks':checks,'training_steps_executed':0,'heldout_questions_evaluated':0},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'environment':environment,'checks':{k:v for k,v in checks.items() if k!='complete_neural_state'},'tensor_count':len(state),'training_steps_executed':0},ensure_ascii=False,indent=2))
