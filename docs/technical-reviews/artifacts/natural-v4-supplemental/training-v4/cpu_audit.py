"""Bounded offline probes and re-audit of original records; no full model load."""
import sys,json,hashlib,struct,math,re,shlex,subprocess,importlib.util,tempfile
from pathlib import Path
from collections import Counter
from fractions import Fraction as F
from types import SimpleNamespace as NS
ROOT=Path(__file__).resolve().parents[6]
sys.path.insert(0,str(ROOT))
import torch
from torch import nn
from tiny_perceptron import natural_assistant as core
from tiny_perceptron.alignment import LoRALinear
OUT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
result={'environment':{'python':sys.version,'torch':torch.__version__,'device':'cpu','cuda_available':torch.cuda.is_available(),'bf16_supported':torch.cuda.is_bf16_supported()},'scope':'Offline CPU probes; no install/download/full VLM/ASR/GPU training/new semantic grading.'}
torch.set_num_threads(1)
doc=(ROOT/'docs/natural-assistant/v4/TRAINING.md').read_text()
assert sha(ROOT/'docs/natural-assistant/v4/TRAINING.md')=='b79f2464e826d944c0a8f9774f6b897e14fa77981dc383bd48eaf4f13e2c4f71'
spec=importlib.util.spec_from_file_location('natural_cli',ROOT/'scripts/natural_assistant.py'); cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
blocks=re.findall(r'```bash\n(.*?)\n```',doc,re.S)
parsed=[]
for block in blocks:
    subprocess.run(['bash','-n'],input=block,text=True,check=True,capture_output=True)
    flat=block.replace('\\\n',' ')
    for line in flat.splitlines():
        if line.startswith('.venv-natural/bin/python scripts/natural_assistant.py '):
            args=shlex.split(line)[2:]
            args=[('1039,2077' if x=='$PENDING_CHECKPOINTS' else '/fixture/selected-adapter' if x=='$SELECTED_ADAPTER' else x) for x in args]
            ns=cli.parser().parse_args(args);parsed.append({'stage':ns.stage,'model_revision':ns.model_revision,'asr_variant':ns.asr_variant,'max_tokens':ns.max_tokens,'max_new_tokens':ns.max_new_tokens,'checkpoint_steps':ns.checkpoint_steps,'selected_only':ns.selected_only})
assert len(parsed)==6
result['commands']={'all_bash_blocks_syntax_valid':len(blocks),'actual_cli_parser_cases':parsed,'heredoc_scope':'bash -n only; training not started'}
m=json.loads((ROOT/'docs/natural-assistant/v4/manifest.json').read_text()); rows=[x for x in m['rows'] if x['split']=='train'];val=[x for x in m['rows'] if x['split']=='validation']
families={s:{x['family'] for x in m['rows']+m['audio_rows'] if x['split']==s} for s in ('train','validation','test')}
assert not any(families[a]&families[b] for a,b in [('train','validation'),('train','test'),('validation','test')])
result['manifest']={'sha256':sha(ROOT/'docs/natural-assistant/v4/manifest.json'),'row_splits':dict(Counter(x['split'] for x in m['rows'])),'audio_splits':dict(Counter(x['split'] for x in m['audio_rows'])),'validation_tasks':dict(Counter(x['task'] for x in val)),'families_disjoint':True,'full_asset_verification':'Canonical data/natural-v4 absent; documented --verify correctly rejected absent target; no dataset downloaded.'}
r=ROOT/'outputs/natural-v4/modal-runs/train-37217452291/natural-natural-v4-train-37217452291-1/review';tr=json.loads((r/'training.json').read_text());h=tr['history'];ex=json.loads((r/'execution.json').read_text())
ids=[x for step in h for x in step['row_ids']]
assert len(h)==tr['completed_steps']==2077 and len(ids)==tr['trained_rows']==4154
assert Counter(ids)=={x['id']:2 for x in rows}
assert ids==[core.training_row_at(rows,i,42)['id'] for i in range(4154)]
assert sum(x['supervised_tokens'] for x in h)==78872
assert all(math.isfinite(x['answer_token_loss_before_update']) and math.isfinite(x['gradient_norm_before_clip']) for x in h)
initial,final=tr['initial_adapter_tensors'],tr['final_adapter_tensors']; assert len(initial)==len(final)==112
assert sum(a!=final[n] for n,a in initial.items())==112
assert tr['frozen_parameter_samples_initial']==tr['frozen_parameter_samples_final']
headers=[]
for step in (1039,2077):
    cp=r/'checkpoints'/f'step-{step:06d}';md=json.loads((cp/'checkpoint.json').read_text());p=cp/'adapter_model.safetensors'
    meta=next(x for x in md['files'] if x['path']==p.name)
    assert p.stat().st_size==meta['bytes']==6438952 and sha(p)==meta['sha256']
    with p.open('rb') as f:
        length=struct.unpack('<Q',f.read(8))[0]; header=json.loads(f.read(length));data_start=8+length
        tensors={k:v for k,v in header.items() if k!='__metadata__'}
        assert len(tensors)==112 and sum(math.prod(v['shape']) for v in tensors.values())==1605632
        if step==2077:
            for name,meta in tensors.items():
                n=name.replace('.lora_A.weight','.lora_A.default.weight').replace('.lora_B.weight','.lora_B.default.weight')
                f.seek(data_start+meta['data_offsets'][0]); data=f.read(meta['data_offsets'][1]-meta['data_offsets'][0])
                assert hashlib.sha256(data).hexdigest()==final[n]['sha256_values']
    headers.append({'step':step,'adapter_bytes':p.stat().st_size,'adapter_sha256':sha(p),'tensor_count':len(tensors),'parameters':sum(math.prod(v['shape']) for v in tensors.values()),'dtype_counts':dict(Counter(v['dtype'] for v in tensors.values()))})
wrapper=json.loads((r.parent/'result.json').read_text())
result['original_training_audit']={'run_id':ex['run_id'],'source_revision':ex['revision'],'training_sha256':sha(r/'training.json'),'execution_sha256':sha(r/'execution.json'),'wrapper_sha256':sha(r.parent/'result.json'),'completed_steps':len(h),'training_readings':len(ids),'unique_training_rows':len(Counter(ids)),'reads_per_row':2,'actual_row_order_matches_seeded_algorithm':True,'supervised_tokens_sum':sum(x['supervised_tokens'] for x in h),'all_logged_losses_and_gradient_norms_finite':True,'changed_adapter_tensors':112,'frozen_sampled_tensors':len(tr['frozen_parameter_samples_initial']),'frozen_sampled_values':sum(len(v['flat_indices']) for v in tr['frozen_parameter_samples_initial'].values()),'frozen_samples_equal':True,'gpu_name':tr['gpu_name'],'versions':tr['versions'],'elapsed_seconds':tr['elapsed_seconds'],'elapsed_minutes':tr['elapsed_seconds']/60,'peak_allocated_bytes':tr['peak_cuda_memory_allocated_bytes'],'peak_allocated_GiB':tr['peak_cuda_memory_allocated_bytes']/2**30,'resource_spec':ex['resource_spec'],'archives':headers,'total_parameters_record':tr['total_parameters'],'wrapper_top_keys':list(wrapper),'replication':'Recomputed original recorded counts/order/changed tensors and hashed existing adapter bytes; not personal GPU reproduction.'}
cfg=json.loads((ROOT/'outputs/natural-v4/factual-research/training-v4/qwen-pinned-config.json').read_text());t=cfg['text_config'];v=cfg['vision_config'];hidden=t['hidden_size'];q=t['num_attention_heads']*t['head_dim'];kv=t['num_key_value_heads']*t['head_dim'];n=t['num_hidden_layers'];rank=8
lora=n*rank*((hidden+q)+(hidden+kv))
text= t['vocab_size']*hidden+n*(hidden*(q+kv+kv+hidden)+2*t['head_dim']+3*hidden*t['intermediate_size']+2*hidden)+hidden
vh=v['hidden_size'];mid=v['intermediate_size'];merge=vh*v['spatial_merge_size']**2
patch=vh*v['in_channels']*v['temporal_patch_size']*v['patch_size']**2+vh
vision_block=4*vh+3*vh*vh+3*vh+vh*vh+vh+2*vh*mid+mid+vh
merger_linear=merge*merge+merge+merge*v['out_hidden_size']+v['out_hidden_size']
vision=patch+v['num_position_embeddings']*vh+v['depth']*vision_block+(merger_linear+2*vh)+len(v['deepstack_visual_indexes'])*(merger_linear+2*merge)
assert lora==1605632 and text+vision==2127532032 and text+vision+lora==2129137664
base_score=sum([F(1,2),F(1),F(4,5),F(1,3),F(1)])/5;chat=(F(1)+F(3,4)+F(1))/3;candidate=sum([F(3,4),F(1),F(4,5),F(1,3),chat])/5
assert base_score==F(109,150) and chat==F(11,12) and candidate==F(19,25)
result['independent_numeric']={'q_output':q,'v_output':kv,'lora_parameters':lora,'lora_tensor_count':n*2*2,'text_parameters_tied_embedding':text,'vision_parameters':vision,'base_parameters':text+vision,'adapted_parameters':text+vision+lora,'file_MB':6438952/1e6,'base_hypothetical_fraction':str(base_score),'base_percent':float(base_score*100),'chat_fraction':str(chat),'candidate_fraction':str(candidate),'candidate_percent':float(candidate*100),'candidate_chat_gate':'fails 3/4 < 4/4 despite higher primary','pending_checkpoints_1000':','.join(str(x) for x in (1039,2077) if x>1000),'pending_checkpoints_1100':','.join(str(x) for x in (1039,2077) if x>1100)}
class Processor:
    def apply_chat_template(self,messages,tokenize=False,add_generation_prompt=False):
        return ''.join('<'+m['role']+'>'+m['content'][0]['text']+(';' if m['role']=='assistant' else '') for m in messages)+('<assistant>' if add_generation_prompt else '')
    def __call__(self,text,return_tensors):
        ids=torch.tensor([[ord(c)%256 for c in text[0]]]);return {'input_ids':ids,'attention_mask':torch.ones_like(ids)}
row={'id':'mask','user':'Q','answer':'AB','history':[{'role':'user','content':'prior'},{'role':'assistant','content':'old'}]};processor=Processor();batch=core.encode_training_row(processor,row,ROOT)
active=batch['labels'][batch['labels']!=-100].tolist();assert active==[ord('A'),ord('B'),ord(';')]
failures=[]
for budget in (1,):
    try: core.encode_training_row(processor,row,ROOT,budget);raise AssertionError('unexpected success')
    except ValueError as exc: failures.append(str(exc))
class BadProcessor(Processor):
    def apply_chat_template(self,*args,**kwargs): return ('bad' if kwargs.get('add_generation_prompt') else '')+super().apply_chat_template(*args,**kwargs)
try: core.encode_training_row(BadProcessor(),row,ROOT);raise AssertionError('bad prefix accepted')
except ValueError as exc: failures.append(str(exc))
torch.manual_seed(42);layer=LoRALinear(nn.Linear(4,3),rank=2,alpha=2);base=layer.base.weight.detach().clone();b=layer.b.detach().clone();opt=torch.optim.SGD([p for p in layer.parameters() if p.requires_grad],lr=.1);loss=layer(torch.ones(1,4)).square().mean();loss.backward();opt.step()
assert torch.equal(base,layer.base.weight) and not torch.equal(b,layer.b)
result['mask_and_lora']={'fake_processor_scope':'Exercises actual prefix/masking/length guards, not Qwen tokenizer or multimodal model','active_targets':active,'ignored_prior_and_prompt':True,'guard_rejections':failures,'small_lora_trainable':sum(p.numel() for p in layer.parameters() if p.requires_grad),'base_unchanged':torch.equal(base,layer.base.weight),'B_changed':not torch.equal(b,layer.b),'normalized_fullwidth':core.normalized('Ａ'),'no_script_conversion':core.normalized('臺')!='台','ordered_keeps_newline':core.normalized('牛奶\n麵包')=='牛奶\n麵包','single_removes_newline':core.normalized('牛奶\n麵包',True)=='牛奶麵包'}
# Execute the exact guide base-selection Python with fixture paths outside product outputs.
code=re.search(r"\.venv-natural/bin/python - <<'PY'\n(.*?)\nPY",doc,re.S).group(1)
vroot=ROOT/'outputs/natural-v4/modal-runs/validation-37219466611/natural-natural-v4-validation-37219466611-1/review'
with tempfile.TemporaryDirectory(prefix='training-v4-cpu-') as fixture:
    f=Path(fixture);(f/'docs/natural-assistant/v4').mkdir(parents=True);(f/'outputs/natural-my-v4/validation').mkdir(parents=True)
    for name in ('manifest.json','validation-protocol-lower-lr.json'):(f/'docs/natural-assistant/v4'/name).write_bytes((ROOT/'docs/natural-assistant/v4'/name).read_bytes())
    (f/'outputs/natural-my-v4/validation/result.json').write_bytes((vroot/'result.json').read_bytes())
    first=subprocess.run([sys.executable,'-c',code],cwd=f,text=True,capture_output=True);assert first.returncode==0,first.stderr
    snapshot=f/'outputs/natural-my-v4/chosen-base-before-test';config=json.loads((snapshot/'configuration.json').read_text());digestlines=(snapshot/'files.sha256').read_text().splitlines()
    assert len(digestlines)==4 and config['selected_variant']=='base' and config['adapter'] is None and config['do_sample'] is False and config['max_new_tokens']==384
    for line in digestlines:
        digest,name=line.split('  ');assert sha(snapshot/name)==digest
    second=subprocess.run([sys.executable,'-c',code],cwd=f,text=True,capture_output=True);assert second.returncode!=0 and 'FileExistsError' in second.stderr
    result['exact_base_snapshot_python']={'first_exit':first.returncode,'first_stdout':first.stdout,'configuration':config,'files_sha256':digestlines,'rerun_exit':second.returncode,'rerun_error':'FileExistsError: existing decision retained','no_model_import_or_GPU':True}
result['source_sha256']={p:sha(ROOT/p) for p in ['scripts/natural_assistant.py','tiny_perceptron/natural_assistant.py','scripts/fetch_natural_data.py','scripts/score_natural_v4_validation.py','scripts/modal_natural.py','.github/workflows/natural-assistant.yml','requirements-natural.txt']}
print(json.dumps(result,ensure_ascii=False,indent=2))
