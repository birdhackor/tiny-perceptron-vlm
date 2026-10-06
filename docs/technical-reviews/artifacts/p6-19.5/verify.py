"""Independent bounded CPU verification of section 19.5; no training or evaluation run."""
from pathlib import Path
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import platform
import re
import sys

ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.selftrained.dataset import RecordEncoder, read_records, train_tokenizer
from tiny_perceptron.selftrained.tokenizer import CharacterTokenizer
from tiny_perceptron.selftrained.model import LimitedAssistant, SelftrainedConfig
from tiny_perceptron.model import loss_sum

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save_raw(rel):
    p = ROOT / rel
    dest = BASE / 'raw' / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(p.read_bytes())
    assert digest(p) == digest(dest)
    return {'original': rel, 'copy': str(dest.relative_to(ROOT)), 'sha256': digest(p)}

def emit(label, value):
    print(label, json.dumps(value, ensure_ascii=False, sort_keys=True))

torch.set_num_threads(2)
torch.manual_seed(195)
emit('environment', {'python': sys.version, 'torch': torch.__version__, 'platform': platform.platform(), 'device': 'cpu', 'threads': torch.get_num_threads()})
body = (BASE / 'section.md').read_text()
emit('section', {'sha256': digest(BASE/'section.md'), 'full_chapter_frozen_sha256': digest(BASE/'chapter-frozen-input.md')})
fences = re.findall(r'```python\n(.*?)\n```', body, re.S)
assert len(fences) == 1
(BASE/'original-demo.py').write_text(fences[0]+'\n')
out = io.StringIO()
with contextlib.redirect_stdout(out):
    exec(compile(fences[0], 'course/chapters/19.md#19.5 fence1', 'exec'), {})
assert out.getvalue() == '請說明是地址、App還是卡片的問題。<eos>\n'
emit('original_fence_stdout', out.getvalue())

messages = [{'role': 'user', 'content': '那個問題要怎麼處理？'}, {'role': 'assistant', 'content': '請說明是地址、App還是卡片的問題。'}]
tok = CharacterTokenizer.build([m['content'] for m in messages])
row = {'id':'bounded-demo','task':'text','messages':messages}
encoded = RecordEncoder(tok, '.').encode(row)
target_ids = [x for x in encoded['labels'] if x != -100]
assert tok.unk_id not in encoded['input_ids']
assert tok.decode(encoded['input_ids'], False) == '<bos><user>那個問題要怎麼處理？<eos><assistant>請說明是地址、App還是卡片的問題。<eos>'
assert len(target_ids) == len(messages[-1]['content']) + 1
emit('original_encoding', {'input':tok.decode(encoded['input_ids'],False),'targets':tok.decode(target_ids,False),'input_tokens':len(encoded['input_ids']),'target_tokens':len(target_ids),'first_target_at_input_token':tok.decode([encoded['input_ids'][next(i for i,x in enumerate(encoded['labels']) if x != -100)]],False),'aligned_targets':[(i,tok.decode([encoded['input_ids'][i]],False),tok.decode([x],False)) for i,x in enumerate(encoded['labels']) if x != -100]})

history = [{'role':'system','content':'只回答已知問題。'}, {'role':'user','content':'先用兩點回答。'}, {'role':'assistant','content':'好。'}, {'role':'tool','content':'工具結果。'}, *messages]
tok2 = CharacterTokenizer.build([m['content'] for m in history])
enc2 = RecordEncoder(tok2, '.')
histrow = {'id':'history-variation','task':'text','messages':history}
hist_encoded = enc2.encode(histrow)
hist_targets = [x for x in hist_encoded['labels'] if x != -100]
assert tok2.decode(hist_targets,False) == '好。<eos>請說明是地址、App還是卡片的問題。<eos>'
assert not (set(hist_targets) & {tok2.system_id,tok2.user_id,tok2.assistant_id,tok2.tool_id})
assert hist_targets.count(tok2.eos_id) == 2
infer = enc2.encode(histrow,generation=True,messages=history[:-1])
assert infer['input_ids'][-1] == tok2.assistant_id
assert tok2.decode(infer['input_ids'],False).endswith('<user>那個問題要怎麼處理？<eos><assistant>')
emit('history_role_variation', {'input':tok2.decode(hist_encoded['input_ids'],False),'targets':tok2.decode(hist_targets,False),'generation_input':tok2.decode(infer['input_ids'],False),'supervised_EOS_count':hist_targets.count(tok2.eos_id)})
try:
    RecordEncoder(tok2,'.',context=3).encode(histrow)
    raise AssertionError('Expected context rejection')
except ValueError as error:
    emit('context_limit', str(error))
unknown_tok = CharacterTokenizer.build(['已知'],required_chars='')
assert unknown_tok.encode('未') == [unknown_tok.unk_id]
emit('vocabulary_variation', {'unseen_character_decode':unknown_tok.decode(unknown_tok.encode('未'),False),'original_unknown_count':encoded['input_ids'].count(tok.unk_id)})

logits = torch.randn(1,len(encoded['labels']),tok.vocab_size,dtype=torch.float64,requires_grad=True)
labels = torch.tensor([encoded['labels']])
total,count = loss_sum(logits,labels)
direct = torch.nn.functional.cross_entropy(logits.reshape(-1,tok.vocab_size)[labels.flatten()!=-100],labels.flatten()[labels.flatten()!=-100],reduction='sum')
assert torch.allclose(total,direct,rtol=0,atol=1e-12)
total.backward()
assert torch.count_nonzero(logits.grad[labels==-100]) == 0
assert torch.count_nonzero(logits.grad[labels!=-100]) > 0
emit('ignore_index_and_denominator', {'target_count':int(count),'loss_sum':float(total.detach()),'direct_unmasked_sum':float(direct.detach()),'maximum_ignored_gradient':float(logits.grad[labels==-100].abs().max()),'tolerance':'1e-12 float64; ignored gradients exactly zero','model_updates':0})

manifest = json.loads((ROOT/'docs/selftrained/v2-manifest.json').read_text())
paths = [ROOT/'outputs/selftrained-v2/data'/r['path'] for r in manifest['records']]
for spec,p in zip(manifest['records'],paths):
    assert digest(p)==spec['sha256'] and p.stat().st_size == spec['bytes']
records = read_records(paths)
tt = [r for r in records if r['task'] in ('text','text_pretrain','tool_call','tool_reply','tool_unavailable','tool_missing','tool_concept','tool_unsupported')]
split_counts = {s:sum(r['split']==s for r in tt) for s in ('train','validation','test')}
groups = {s:{r['group_id'] for r in tt if r['split']==s} for s in split_counts}
assert not any(groups[a]&groups[b] for a,b in [('train','validation'),('train','test'),('validation','test')])
emit('fixed_data', {'manifest_sha256':digest(ROOT/'docs/selftrained/v2-manifest.json'),'all_12_record_hashes_match':True,'text_tools_split_records':split_counts,'text_tools_group_counts':{s:len(v) for s,v in groups.items()},'cross_split_group_intersections':0,'test_inputs_used_for_inference_or_scoring':0})

train_file = ROOT/'outputs/selftrained-v2/data/text-tools-train.jsonl'
raw_lines = [(i,line,json.loads(line)) for i,line in enumerate(train_file.read_text().splitlines(),1)]
requirements = [('App打不開，該怎麼辦？','先檢查網路並重新啟動App，仍失敗再詢問官方客服。'),('那個問題要怎麼處理？','請說明是地址、App還是卡片的問題。'),('算0加0是多少。','計算器目前不可用，請開啟後再計算。'),('請預測明天天氣。','這超出我的範圍，我能協助有限聊天、地址、App、卡片或加減乘問題。')]
selected = []
for question,answer in requirements:
    candidates = [(i,line,r) for i,line,r in raw_lines if r['messages'][-1]['content']==answer and r['messages'][-2]['content']==question]
    assert candidates
    i,line,r=candidates[0]
    if question.startswith('App'): assert any('用一句回答' in m['content'] for m in r['messages'][:-2])
    if question.startswith('算'): assert '計算器不可用' in r['messages'][0]['content']
    selected.append((i,line))
    emit('table_training_example',{'line':i,'matching_exact_question_answer_pairs':len(candidates),**{k:r[k] for k in ('id','group_id','split','task','messages')}})
for task in ['tool_call','tool_reply']:
    i,line,r = next((i,line,r) for i,line,r in raw_lines if r['task']==task)
    selected.append((i,line));emit('tool_training_example',{'line':i,**{k:r[k] for k in ('id','group_id','split','task','messages')}})
for phrase in ['你記得','改成','選擇']:
    match = next(((i,line,r) for i,line,r in raw_lines if any(phrase in m['content'] for m in r['messages'])),None)
    if match:
        i,line,r=match;selected.append((i,line));emit('history_or_format_training_example',{'phrase':phrase,'line':i,**{k:r[k] for k in ('id','group_id','split','task','messages')}})
selected_unique = dict(selected)
(BASE/'train-selected-raw-lines.jsonl').write_text(''.join(line+'\n' for _,line in sorted(selected_unique.items())))
(BASE/'train-selected-line-map.json').write_text(json.dumps({'source':'outputs/selftrained-v2/data/text-tools-train.jsonl','source_sha256':digest(train_file),'copy_line_to_original_line':{str(j):i for j,(i,_) in enumerate(sorted(selected_unique.items()),1)}},indent=2)+'\n')

trained_tok = train_tokenizer(records)
trained_sha = hashlib.sha256(json.dumps(trained_tok.to_dict(),ensure_ascii=False,sort_keys=True).encode()).hexdigest()
receipt = json.loads((ROOT/'docs/selftrained/results/training-raw/moe-sft/raw/train-receipt.json').read_text())
assert trained_sha == receipt['tokenizer_sha256']
assert split_counts['train']==receipt['train_records'] and split_counts['validation']==receipt['validation_records']
assert receipt['steps']==8000 and receipt['completed_requested_steps'] and not receipt['test_used_for_selection']
for spec in manifest['records']: assert receipt['data_sha256'][spec['path']] == spec['sha256']
(BASE/'reconstructed-tokenizer.json').write_text(json.dumps(trained_tok.to_dict(),ensure_ascii=False,indent=2)+'\n')
emit('sft_receipt_crosscheck', {k:receipt[k] for k in ('stage','architecture','steps','tokens','target_tokens','train_records','validation_records','test_used_for_selection','seed','tokenizer_sha256','trainable_parameters','completed_requested_steps')})
execution=json.loads((ROOT/'docs/selftrained/results/training-raw/moe-sft/raw/execution.json').read_text())
outer=json.loads((ROOT/'docs/selftrained/results/training-raw/moe-sft/receipt.json').read_text())
outer_files={f['path']:f for f in outer['files']}
for name in ['execution.json','train-receipt.json']:
    assert digest(ROOT/'docs/selftrained/results/training-raw/moe-sft/raw'/name)==outer_files[name]['sha256']
assert outer['returncode']==execution['returncode']==0
assert outer['status']==execution['status']=='completed'
assert execution['manifest_sha256']==digest(ROOT/'docs/selftrained/v2-manifest.json')
cmd=execution['command']
assert cmd[cmd.index('--stage')+1]=='sft' and cmd[cmd.index('--steps')+1]=='8000'
assert cmd[cmd.index('--architecture')+1]=='moe'
emit('sft_raw_execution_binding',{'outer_and_raw_hashes_match':True,'manifest_sha256':execution['manifest_sha256'],'revision':execution['revision'],'status':execution['status'],'returncode':execution['returncode'],'steps':cmd[cmd.index('--steps')+1],'init_checkpoint':cmd[cmd.index('--init-checkpoint')+1]})
emit('trained_vocabulary',{'vocab_size':trained_tok.vocab_size,'source':'train messages only plus fixed ASCII and OCR alphabet','trainer_tokenizer_sha256':trained_sha})

spec = importlib.util.spec_from_file_location('train_19_5',ROOT/'scripts/selftrained/train.py')
train_module = importlib.util.module_from_spec(spec);spec.loader.exec_module(train_module)
small = LimitedAssistant(SelftrainedConfig(vocab_size=trained_tok.vocab_size,width=16,layers=1,heads=2,kv_heads=1,ffn_hidden=16,top_k=1))
params=train_module.set_trainable(small,'sft')
trainable_names = [n for n,p in small.named_parameters() if p.requires_grad]
assert trainable_names and all(n.startswith('lm.') for n in trainable_names)
assert all(not p.requires_grad for n,p in small.named_parameters() if not n.startswith('lm.'))
emit('sft_parameter_scope',{'trainable_names':trainable_names,'all_trainable_parameters_have_lm_prefix':True,'all_perception_parameters_frozen':True,'steps_run':0})

score_tree=ast.parse((ROOT/'scripts/selftrained/evaluate.py').read_text())
score_functions=[n for n in score_tree.body if isinstance(n,ast.FunctionDef) and n.name in ('normalize','format_pass','score_reply')]
scope={'re':re}
exec(compile(ast.Module(body=score_functions,type_ignores=[]),'evaluate.py selected scoring functions','exec'),scope)
checked=[]
trace={'tool_call':None,'status':'no_call'}
for question,answer in requirements:
    r=next(r for _,_,r in raw_lines if r['messages'][-1]['content']==answer and r['messages'][-2]['content']==question)
    correct=scope['score_reply'](r,answer,trace)
    wrong=scope['score_reply'](r,'不知道。',trace)
    assert correct['semantic'] and correct['format'] is not False and not wrong['semantic']
    checked.append({'id':r['id'],'intent':r.get('supervision',{}).get('intent'),'correct_training_target_score':correct,'wrong_synthetic_reply_score':wrong})
assert scope['format_pass']('1. 第一點。\n2. 第二點。','two_points')
assert not scope['format_pass']('只有一句。','two_points')
emit('scoring_contract_synthetic_checks',{'checks':checked,'heldout_rows_scored':0,'model_outputs_scored':0,'complete_generation_recording':'Generator.generate lines 395-457 records token IDs and EOS stop_reason; this script does not claim a heldout EOS success rate'})

cpu_base = ROOT/'docs/selftrained/results/public-cpu-raw'
result = json.loads((cpu_base/'text-1-result.json').read_text())
argv = json.loads((cpu_base/'text-1-argv.json').read_text())
stdout = json.loads((cpu_base/'text-1-stdout.txt').read_text())
input_msgs = json.loads((ROOT/'docs/selftrained/examples/v2/text.messages.json').read_text())
assert argv['public_messages_before_execution'] == input_msgs
assert stdout['answer'] == result['actual_answer']
assert stdout['generations'] == result['actual_generations']
assert digest(cpu_base/'text-1-stdout.txt')==result['stdout_sha256']
assert digest(cpu_base/'text-1-stderr.txt')==result['stderr_sha256']
assert result['returncode']==0 and result['generation_count']==1
generation=result['actual_generations'][0]
assert generation['prompt_messages']==input_msgs and input_msgs[-1]['role']=='user'
inference_encoding=RecordEncoder(trained_tok,'.').encode({'id':'public-prompt','task':'text','messages':input_msgs},generation=True,messages=input_msgs)
assert inference_encoding['input_ids']==generation['prompt_ids']
assert trained_tok.decode(generation['generated_ids'])=='1. 先檢查網路並重新啟動App。\n2. 仍失敗再詢問官方客服。'
assert generation['raw_output']==result['actual_answer'] and generation['eos'] and generation['generated_ids'][-1]==trained_tok.eos_id
assert generation['model_sha256']==result['model']['files']['model.safetensors']
assert '--device' in argv['argv'] and argv['argv'][argv['argv'].index('--device')+1]=='cpu'
assert result['public_source']['revision']=='979cdfacc588ad0536f1c64fff96f264571cf054'
assert result['model']['config']['architecture']=='moe'
emit('public_cpu_existing_run_verification',{'actual_argv':argv['argv'],'messages':input_msgs,'generated_text':trained_tok.decode(generation['generated_ids']),'generated_ids':generation['generated_ids'],'generated_token_count_including_EOS':len(generation['generated_ids']),'prompt_tokens':len(generation['prompt_ids']),'EOS':generation['eos'],'returncode':result['returncode'],'model_sha256':generation['model_sha256'],'public_revision':result['public_source']['revision'],'architecture':result['model']['config']['architecture'],'generation_count':1,'new_inference_calls_by_this_review':0,'claims_scope':'one selected existing CPU generation; no success rate or causal attribution to history alone'})

raw_files=['docs/selftrained/v2-manifest.json','docs/selftrained/examples/v2/text.messages.json','docs/selftrained/results/public-cpu-raw/text-1-argv.json','docs/selftrained/results/public-cpu-raw/text-1-result.json','docs/selftrained/results/public-cpu-raw/text-1-stdout.txt','docs/selftrained/results/public-cpu-raw/text-1-stderr.txt','docs/selftrained/results/training-raw/moe-sft/raw/train-receipt.json','docs/selftrained/results/training-raw/moe-sft/raw/execution.json','docs/selftrained/results/training-raw/moe-sft/receipt.json']
emit('saved_raw_files',[save_raw(f) for f in raw_files])
pointer_log={'docs/selftrained/v2-manifest.json':['/records','/package','/model_config'],'docs/selftrained/results/public-cpu-raw/text-1-result.json':['/argv','/returncode','/stdout_sha256','/stderr_sha256','/actual_answer','/model/config/architecture','/model/files','/model/origin','/model/stage','/model/selected_step','/model/selected_checkpoint_sha256','/model/manifest_sha256','/public_source/repo','/public_source/revision','/public_source/prefix','/generation_count','/actual_generations','/CPU_only_argv_verified'],'docs/selftrained/results/public-cpu-raw/text-1-argv.json':['/argv','/public_messages_before_execution'],'docs/selftrained/results/training-raw/moe-sft/raw/train-receipt.json':['/stage','/architecture','/steps','/tokens','/target_tokens','/train_records','/validation_records','/test_used_for_selection','/seed','/data_sha256','/tokenizer_sha256','/trainable_parameters','/freeze_perception_backbones','/origin','/completed_requested_steps','/inference_exported'],'docs/selftrained/results/training-raw/moe-sft/raw/execution.json':['/stage','/revision','/manifest_sha256','/job_sha256','/command','/returncode','/status'],'docs/selftrained/results/training-raw/moe-sft/receipt.json':['/files','/returncode','/status'],'text-tools train/validation/test':['/id','/group_id','/split','/task','/messages','/image','/audio','/evaluation_only'],'selected train examples':['/supervision/intent','/supervision/format','/supervision/semantic_all','/supervision/semantic_any','/supervision/semantic_forbidden','/supervision/rubric','/supervision/expected_tool']}
(BASE/'inspected-json-pointers.json').write_text(json.dumps(pointer_log,ensure_ascii=False,indent=2)+'\n')
emit('result','all bounded checks passed; no training, GPU, heldout generation scoring, downloads of weights or uploads')
