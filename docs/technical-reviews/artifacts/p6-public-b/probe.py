from pathlib import Path
import ast, collections, hashlib, json, re, shlex, subprocess, sys, tarfile

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,d): (OUT/name).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
audit=[]
def read(p,keys):
    p=ROOT/p; d=json.loads(p.read_text());
    audit.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p),'topkeys':{k:type(v).__name__ for k,v in d.items()},'pointers':['/'+k for k in keys]})
    return {k:d[k] for k in keys if k in d}

# Raw source identities and only named machine provenance / observations.
stages={}
for p in sorted((ROOT/'docs/selftrained/results/training-raw').iterdir()):
    if not p.is_dir(): continue
    base=p.relative_to(ROOT)
    e=read(base/'raw/execution.json',['status','returncode','revision','command','job','started_at','finished_at'])
    t=read(base/'raw/train-receipt.json',['stage','architecture','steps','tokens','target_tokens','train_records','validation_records','test_used_for_selection','seed','config','selection','freeze_perception_backbones','sampling_mode','tool_loss_weight','numeric_run_loss_weight','native_voice_loss_weight','origin','stage_history','completed_requested_steps','interrupted'])
    r=read(base/'receipt.json',['files','status','returncode'])
    entries={f['path']:f for f in r['files']}
    assert sha(p/'raw/execution.json')==entries['execution.json']['sha256']
    assert sha(p/'raw/train-receipt.json')==entries['train-receipt.json']['sha256']
    assert e['status']=='completed' and e['returncode']==0 and t['completed_requested_steps'] and not t['interrupted'] and not t['test_used_for_selection']
    history=[{'stage':h['stage'],'step':h['step'],'checkpoint_sha256':h['checkpoint_sha256'],'selection':h['selection']} for h in t['stage_history']]
    stages[p.name]={'completed_steps':t['steps'],'job':e['job'],'history':history,'best_sha256':entries['best.pt']['sha256'],'latest_sha256':entries['latest.pt']['sha256'],'inference_manifest_sha256':entries['inference-manifest.json']['sha256'],'origin_kind':t['origin']['kind'],'config':t['config'],'train_records':t['train_records'],'validation_records':t['validation_records'],'selection':t['selection'],'start':e['started_at'],'end':e['finished_at']}
    if 'new_joint_initialization' in t['origin']:
        stages[p.name]['new_joint_initialization']=t['origin']['new_joint_initialization']
for arch, chain in [('moe',['pretrain','sft','vision','ocr','audio','joint','weighted','native']),('dense',['pretrain','sft','vision','ocr','audio','joint','weighted'])]:
    for prev,nxt in zip(chain,chain[1:]):
        parent=stages[arch+'-'+prev];child=stages[arch+'-'+nxt]
        assert child['history'][-1]['checkpoint_sha256']==parent['best_sha256']
        parent['selected_step']=child['history'][-1]['step']
write('training-raw-derived.json',stages)

final={}
for arch in ['moe','dense']:
    base=Path('docs/selftrained/results/public-raw')/arch
    keys=['split','count','evaluation_complete','expected_count','interrupted','teacher_forcing_used_for_generation','checkpoint_sha256','selected_checkpoint_sha256','inference_manifest_sha256','safe_weights_sha256','per_task_final_reply','stratified_final_reply','perception','controls','voice_topic_continuation','thresholds']
    m=read(base/'test/metrics.json',keys)
    f=read(base/'freeze/frozen.json',['created_at','architecture','checkpoint_sha256','safe_weights_sha256','inference_manifest_sha256','test_once','thresholds','validation_metrics_sha256'])
    v=read(base/'freeze/metrics.json',['split','count','expected_count','evaluation_complete','checkpoint_sha256','inference_manifest_sha256','per_task_final_reply'])
    r=read(base/'test/evaluation-receipt.json',['split','status','expected_count','completed_count','outputs_sha256','completed_ids_sha256','resumed_from'])
    assert m['count']==m['expected_count']==r['completed_count']==r['expected_count']==3734 and m['evaluation_complete']
    assert v['count']==2435 and f['test_once']
    assert sha(ROOT/base/'freeze/metrics.json')==f['validation_metrics_sha256']
    assert m['checkpoint_sha256']==v['checkpoint_sha256']==f['checkpoint_sha256']
    final[arch]={'metrics':m,'freeze':f,'validation':v,'evaluation_receipt':r}
write('final-raw-derived.json',final)

cpu={}
for p in sorted((ROOT/'docs/selftrained/results/public-cpu-raw').glob('*-result.json')):
    name=p.name.replace('-result.json','');base=p.relative_to(ROOT).parent
    r=read(p.relative_to(ROOT),['task','chat','argv','returncode','actual_answer','saved_GPU_validation_answer','stdout_sha256','stderr_sha256'])
    a=read(base/(name+'-argv.json'),['argv','public_messages_before_execution'])
    assert r['argv']==a['argv'] and r['returncode']==0
    assert sha(ROOT/base/(name+'-stdout.txt'))==r['stdout_sha256']
    assert sha(ROOT/base/(name+'-stderr.txt'))==r['stderr_sha256']
    row={'argv':r['argv'],'chat':r['chat'],'returncode':r['returncode']}
    if r['chat']:
        s=read(base/(name+'-stdout.txt'),['answer','messages','generations','tool_trace','model','public_source'])
        assert s['answer']==r['actual_answer']==r['saved_GPU_validation_answer']
        assert s['public_source']['authentication']=='disabled'
        row.update({'answer':s['answer'],'messages':s['messages'],'generation_count':len(s['generations']),'tool_trace':s['tool_trace'],'model':{k:s['model'][k] for k in ['selected_step','files','manifest_sha256','config']},'public_source':s['public_source']})
        row['generations']=[{k:g[k] for k in ['prompt_messages','generated_ids','raw_output','modality_kinds']} for g in s['generations']]
    cpu[name]=row
assert sum(v['chat'] for v in cpu.values())==8 and len(cpu)==9
assert cpu['voice_topic_continuation-3']['generations'][0]['prompt_messages'][:-1]==cpu['voice_topic_continuation-1']['messages']
fences=re.findall(r'```(?:sh|bash)\n(.*?)```', (ROOT/'docs/selftrained/v2-public-cpu-commands.md').read_text(), re.S)
chats=[shlex.split(f.replace('\\\n',' ')) for f in fences if 'scripts/selftrained/chat.py' in f]
assert len(chats)==8
def parse_chat(args):
    pos=args.index('scripts/selftrained/chat.py') if 'scripts/selftrained/chat.py' in args else next(i for i,x in enumerate(args) if x.endswith('/scripts/selftrained/chat.py'))
    args=args[pos+1:];d={};i=0
    while i<len(args):
        k=args[i];i+=1
        if i==len(args) or args[i].startswith('--'): d[k]=True
        else: d[k]=args[i];i+=1
    return d
bytask=collections.defaultdict(list)
for v in cpu.values():
    if v['chat']:bytask[parse_chat(v['argv'])['--task']].append(parse_chat(v['argv']))
comparisons=[]
for c in chats:
    d=parse_chat(c); candidates=bytask[d['--task']]
    raw=candidates.pop(0)
    # Only locations and executable path were normalized for publication.
    keys=['--repo','--revision','--prefix','--manifest-sha256','--task','--device','--max-new-tokens','--threads','--image','--audio','--roi','--modality-message-index','--tools']
    assert all(d.get(k)==raw.get(k) for k in keys),(d,raw)
    comparisons.append({'task':d['--task'],'checked_keys':keys,'match':True})
write('cpu-raw-derived.json',{'invocations':cpu,'public_fence_comparisons':comparisons})

manifest=read(Path('docs/selftrained/v2-manifest.json'),['package','records','assets','model_config','initialization'])
archive=ROOT/manifest['package']['path']
assert archive.stat().st_size==manifest['package']['bytes'] and sha(archive)==manifest['package']['sha256']
pointer=subprocess.check_output(['git','show',manifest['package']['revision']+':'+manifest['package']['path']],cwd=ROOT).decode()
assert pointer==f"version https://git-lfs.github.com/spec/v1\noid sha256:{sha(archive)}\nsize {archive.stat().st_size}\n"
decl={x['path']:x for x in manifest['records']+manifest['assets']}
data={'files':0,'record_files':len(manifest['records']),'assets':len(manifest['assets']),'bytes':0,'records_by_split':collections.Counter(),'tasks':collections.Counter(),'audio_assets_by_split':collections.defaultdict(set),'speaker_key_count':0,'font_variants':collections.defaultdict(set)}
raw_records=[]
with tarfile.open(archive,'r:gz') as t:
    for member in t:
        if not member.isfile():continue
        b=t.extractfile(member).read();data['files']+=1;data['bytes']+=len(b)
        if member.name in decl:
            assert len(b)==decl[member.name]['bytes'] and hashlib.sha256(b).hexdigest()==decl[member.name]['sha256'], member.name
        if member.name in {r['path'] for r in manifest['records']}:
            for line in b.decode().splitlines():
                r=json.loads(line);raw_records.append(r);data['records_by_split'][r['split']]+=1;data['tasks'][(r['split']+'|'+r['task'])]+=1
                if r.get('audio'):data['audio_assets_by_split'][r['split']].add(r['audio'])
                data['speaker_key_count']+=sum('speaker' in k.lower() for k in r)
                if r['task']=='ocr':data['font_variants'][r['split']].add(str(r.get('provenance',{}).get('font',r.get('font',''))))
data['audio_assets_by_split']={k:len(v) for k,v in data['audio_assets_by_split'].items()};data['font_variants']={k:sorted(v) for k,v in data['font_variants'].items()}
data['declared_files']=len(decl); data['archive_sha256']=sha(archive); data['archive_bytes']=archive.stat().st_size;data['pointer']=pointer
sys.path.insert(0,str(ROOT))
import torch
from tiny_perceptron.selftrained.dataset import train_tokenizer
from tiny_perceptron.selftrained.model import LimitedAssistant,SelftrainedConfig
tok=train_tokenizer(raw_records);data['vocab_size']=tok.vocab_size
descriptions={}
for a in ['moe','dense']:
    model=LimitedAssistant(SelftrainedConfig(**manifest['model_config'],architecture=a,vocab_size=tok.vocab_size))
    descriptions[a]=model.description()
data['model_descriptions']=descriptions
data['environment']={'python':sys.version,'torch':torch.__version__,'device':'cpu'}
write('archive-model-derived.json',data)
write('input-audit.json',audit)
print(json.dumps({'stages':len(stages),'cpu_invocations':len(cpu),'archive':data['files'],'records':data['records_by_split'],'audio_assets':data['audio_assets_by_split'],'vocab':tok.vocab_size,'parameters':{k:v['parameters'] for k,v in descriptions.items()}},ensure_ascii=False))
