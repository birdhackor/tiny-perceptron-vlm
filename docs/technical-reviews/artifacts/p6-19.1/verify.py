"""Bounded independent CPU checks of public demo and original data; no training/test evaluation."""
from pathlib import Path
from collections import Counter
import ast, hashlib, json, os, platform, shutil, sys

ROOT = Path('/workspace/selftrained-v2')
OUT = ROOT / 'docs/technical-reviews/artifacts/p6-19.1'
sys.path.insert(0, str(ROOT))
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
import torch
from tiny_perceptron.selftrained.inference import InferenceAssistant, verify_export
from tiny_perceptron.selftrained.model import LimitedAssistant, SelftrainedConfig
from tiny_perceptron.selftrained.tokenizer import CharacterTokenizer, OCR_CHARACTERS
from tiny_perceptron.selftrained.tools import execute_tool_call, parse_tool_call, ToolCallError

torch.set_num_threads(2)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name, obj):
    (OUT/'evidence'/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def tensor_digest(model):
    h=hashlib.sha256()
    for name, t in model.state_dict().items():
        h.update(name.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
report={'command_argv':sys.argv,'environment':{'python':sys.version,'executable':sys.executable,'torch':str(torch.__version__),'device':'cpu','cuda_build':str(torch.version.cuda),'platform':platform.platform()},'inspected_pointers':[]}
def read_pointer(relative, pointer):
    path=ROOT/relative; data=json.loads(path.read_text())
    value=data
    for key in pointer.split('/')[1:]: value=value[int(key)] if isinstance(value,list) else value[key]
    report['inspected_pointers'].append({'path':relative,'sha256':sha(path),'pointer':pointer})
    return value

raw_root='docs/selftrained/results/public-cpu-raw/'
for name in ['text-1-argv.json','text-1-result.json','text-1-stdout.txt','text-1-stderr.txt']:
    source=ROOT/raw_root/name;dest=OUT/'inputs'/name;shutil.copyfile(source,dest)
    assert sha(source)==sha(dest)
stdout_path=ROOT/raw_root/'text-1-stdout.txt'
stdout=json.loads(stdout_path.read_text())
report['raw_stdout_keys']={k:type(v).__name__ for k,v in stdout.items()}
report['inspected_pointers'] += [{'path':raw_root+'text-1-stdout.txt','sha256':sha(stdout_path),'pointer':p} for p in ['/answer','/model/selected_step','/model/files','/generations/0/generated_ids','/generations/0/prompt_messages','/generations/0/eos','/generations/0/modality_kinds']]
assert sha(stdout_path)==read_pointer(raw_root+'text-1-result.json','/stdout_sha256')
assert read_pointer(raw_root+'text-1-result.json','/returncode')==0
messages=json.loads((ROOT/'docs/selftrained/examples/v2/text.messages.json').read_text())
assert messages==read_pointer(raw_root+'text-1-argv.json','/public_messages_before_execution')
assert messages==stdout['generations'][0]['prompt_messages']
assert len(messages)==4 and [m['role'] for m in messages]==['system','user','assistant','user']
assert '先檢查網路' not in messages[0]['content']
expected='1. 先檢查網路並重新啟動App。\n2. 仍失敗再詢問官方客服。'
assert stdout['answer']==expected and stdout['generations'][0]['eos'] is True
assert stdout['generations'][0]['modality_kinds']==[]
model_dir=Path('/tmp/p5-native-public-cpu-smoke-actual/public-model')
pin='f27856892b0c46ca81ba43bfadf7051ed6bd31d848e4868a05147e5799df1e5e'
manifest, manifest_sha=verify_export(model_dir,manifest_sha256=pin)
for name in ['inference-manifest.json','model-config.json','tokenizer.json']:
    shutil.copyfile(model_dir/name,OUT/'inputs'/name)
assert manifest['selected_step']==1000 and manifest['files']==stdout['model']['files']
tokenizer=CharacterTokenizer.load(model_dir/'tokenizer.json')
ids=stdout['generations'][0]['generated_ids']
assert ids[-1]==tokenizer.eos_id==2 and tokenizer.decode(ids)==expected
report['archived_demo']={'answer':stdout['answer'],'generated_token_count':len(ids),'eos_id':ids[-1],'modality_count':0,'selected_step':manifest['selected_step'],'manifest_sha256':manifest_sha,'model_payload_sha256':sha(model_dir/'model.safetensors')}

# Original provided public demonstration only, plus two bounded input variations.
assistant=InferenceAssistant(model_dir, '/tmp/p6-19.1-no-assets', device='cpu',manifest_sha256=pin)
before=tensor_digest(assistant.model)
cases=[('original',messages),('one_sentence',[{**m,'content':m['content'].replace('兩點','一句')} if i in (1,2) else dict(m) for i,m in enumerate(messages)]),('address',[dict(m) if i!=3 else {**m,'content':'我需要更改地址。'} for i,m in enumerate(messages)])]
results=[]
for case,input_messages in cases:
    result=assistant.reply(input_messages,task='text',max_new_tokens=128)
    save('generation-'+case+'.json',result)
    results.append({'case':case,'messages':input_messages,'answer':result['answer'],'eos':result['generations'][0]['eos'],'generated_token_count':len(result['generations'][0]['generated_ids']),'modality_kinds':result['generations'][0]['modality_kinds']})
assert results[0]['answer']==expected and results[0]['eos']
after=tensor_digest(assistant.model)
assert before==after
report['bounded_generations']=results
report['state_unchanged']={'before':before,'after':after}

torch.manual_seed(911)
small=LimitedAssistant(SelftrainedConfig(vocab_size=32,width=32,layers=1,heads=4,kv_heads=2,ffn_hidden=64,max_length=128))
norms={name:{'shape':list(p.shape),'all_ones':bool(torch.equal(p,torch.ones_like(p))),'requires_grad':p.requires_grad} for name,p in small.named_parameters() if name.endswith('norm.weight') or name.endswith('norm1.weight') or name.endswith('norm2.weight')}
assert norms and all(n['all_ones'] and n['requires_grad'] for n in norms.values())
report['initialization_literal_counterexample']=norms
payloads=[{'kind':'image','values':torch.zeros(2,1,32,32),'coordinates':torch.tensor([[.25,.5],[.75,.5]])},{'kind':'ocr','values':torch.ones(1,32,128)},{'kind':'audio','values':torch.zeros(20,40)}]
all_ids=torch.tensor([[1]+[7]*2+[8]*32+[9]*16+[3,10,2,4]])
emb,perception=small._embeddings(all_ids,None,[payloads])
report['shared_core']={'lm_count':sum(1 for n in small.named_modules() if n[0]=='lm'),'embedding_shape':list(emb.shape),'perception_kinds':[p['kind'] for p in perception],'finite':bool(torch.isfinite(emb).all())}
assert report['shared_core']['lm_count']==1 and report['shared_core']['perception_kinds']==['image','ocr','audio']

manifest_data=json.loads((ROOT/'docs/selftrained/v2-manifest.json').read_text())
data_stats={}
for entry in manifest_data['records']:
    path=ROOT/'outputs/selftrained-v2/data'/entry['path']
    assert sha(path)==entry['sha256']
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    stats={'file_sha256':sha(path),'rows':len(rows),'tasks':dict(Counter(r['task'] for r in rows)),'splits':dict(Counter(r['split'] for r in rows))}
    if entry['path'].startswith('ocr-'):
        stats.update(fonts=dict(Counter(r.get('supervision',{}).get('font_family') for r in rows)),ocr_lengths=dict(Counter(len(r.get('supervision',{}).get('ocr_text','')) for r in rows)),ocr_characters=''.join(sorted(set(''.join(r.get('supervision',{}).get('ocr_text','') for r in rows)))))
    elif entry['path'].startswith('voice-'):
        stats.update(intents=dict(Counter(r.get('supervision',{}).get('intent') for r in rows)))
    elif entry['path'].startswith('vision-'):
        stats.update(labels=dict(Counter(v for r in rows for v in r.get('supervision',{}).get('vision_labels',[]))),axes=dict(Counter(r.get('image_layout',{}).get('axis') for r in rows)))
    data_stats[entry['path']]=stats
report['dataset_statistics']=data_stats
report['ocr_alphabet']={'characters':OCR_CHARACTERS,'unique_count':len(set(OCR_CHARACTERS))}

stages=[]
for path in sorted((ROOT/'docs/selftrained/results/training-raw').glob('*/raw/train-receipt.json')):
    relative=path.relative_to(ROOT).as_posix(); d=json.loads(path.read_text()); e=json.loads(path.with_name('execution.json').read_text()); outer=json.loads(path.parents[1].joinpath('receipt.json').read_text())
    keys={k:type(v).__name__ for k,v in d.items()}
    pointers=['/stage','/architecture','/steps','/seed','/origin/kind','/stage_history','/inference_exported']
    history=[{'stage':s['stage'],'step':s['step'],'checkpoint_sha256':s['checkpoint_sha256']} for s in d['stage_history']]
    best=next(v['sha256'] for v in outer['files'] if v['path']=='best.pt')
    stages.append({'path':relative,'sha256':sha(path),'keys':keys,'pointers':pointers,'stage':d['stage'],'architecture':d['architecture'],'steps':d['steps'],'seed':d['seed'],'origin_kind':d['origin']['kind'],'history':history,'selected_best_sha256':best,'inference_exported':d['inference_exported'],'execution_sha256':sha(path.with_name('execution.json')),'outer_sha256':sha(path.parents[1]/'receipt.json'),'returncode':e['returncode'],'command':e['command']})
assert len(stages)==15
by_hash={s['selected_best_sha256']:s for s in stages}
for stage in stages:
    assert stage['origin_kind']=='all-neural-weights-random' and stage['returncode']==0
    assert all(h['checkpoint_sha256'] in by_hash for h in stage['history'])
assert manifest['selected_checkpoint_sha256']==next(s['selected_best_sha256'] for s in stages if '/moe-native/' in s['path'])
report['training_provenance']=stages
report['tool_bounds']={'add':execute_tool_call('{"tool":"calculator","operation":"add","a":8,"b":7}'),'subtract':execute_tool_call('{"tool":"calculator","operation":"subtract","a":0,"b":99}'),'multiply':execute_tool_call('{"tool":"calculator","operation":"multiply","a":99,"b":99}')}
try:
    parse_tool_call('{"tool":"calculator","operation":"add","a":100,"b":1}')
except ToolCallError: report['tool_bounds']['operand100_rejected']=True
save('verification.json',report)
print(json.dumps({k:report[k] for k in ['environment','archived_demo','bounded_generations','initialization_literal_counterexample','shared_core','dataset_statistics','tool_bounds']},ensure_ascii=False,indent=2))
