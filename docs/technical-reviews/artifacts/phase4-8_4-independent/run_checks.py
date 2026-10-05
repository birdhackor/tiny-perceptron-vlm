"""Short CPU checks; inspect saved results, never train or load/download weights."""
import ast
import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.data import ByteTokenizer, render_chat
from tiny_perceptron.model import ModelConfig, TinyLM, generate
from tiny_perceptron.tokenization import generation_report
from scripts.course_experiments.common import split_records, records_sha256
from scripts.course_experiments.text import arithmetic_records

torch.set_num_threads(1)
def write(name, value):
    (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n",encoding="utf-8")
sha=lambda raw:hashlib.sha256(raw).hexdigest()
environment={"python":sys.version,"python_executable":sys.executable,"torch":torch.__version__,
    "torch_git_revision":torch.version.git_version,"device":"cpu","cwd":str(Path.cwd()),
    "network":"offline environment for checks; no downloads in this script",
    "commands":"Original fence plus bounded data, token, scoring and greedy-generation variants; no optimizer/backward/training/checkpoint."}
write("environment.json",environment)

fence=(OUT/"fence-1.py").read_text()
def run_fence(code):
    ns={};s=io.StringIO()
    with contextlib.redirect_stdout(s):exec(compile(code,"fence-1.py","exec"),ns)
    return ns["records"],s.getvalue()
records,stdout=run_fence(fence)
(OUT/"fence-original.stdout.txt").write_text(stdout,encoding="utf-8")
assert len(records)==4
assert [r["提問"] for r in records]==["風格=簡短；1+1=?","風格=生動；1+1=?","風格=簡短；2+2=?","風格=生動；2+2=?"]
assert [r["回答"] for r in records]==["2","2；把兩組物件合起來數就知道了。","4","4；把兩組物件合起來數就知道了。"]
single=fence.replace('["簡短", "生動"]','["簡短"]')
(OUT/"fence-single-style.py").write_text(single,encoding="utf-8")
single_records,single_stdout=run_fence(single)
(OUT/"fence-single-style.stdout.txt").write_text(single_stdout,encoding="utf-8")
assert len(single_records)==2 and [r["回答"] for r in single_records]==["2","4"]
reversed_fence=fence.replace('["簡短", "生動"]','["生動", "簡短"]')
(OUT/"fence-reversed-style.py").write_text(reversed_fence,encoding="utf-8")
reversed_records,reversed_stdout=run_fence(reversed_fence)
(OUT/"fence-reversed-style.stdout.txt").write_text(reversed_stdout,encoding="utf-8")
assert reversed_records==[records[1],records[0],records[3],records[2]]

# Inspect the true renderer contract: input includes the style, all user targets
# are ignored, assistant bytes and EOS are next-position supervised.
tok=ByteTokenizer();render=[]
for r in records:
    x,y=render_chat([{"role":"user","content":r["提問"]},{"role":"assistant","content":r["回答"]}])
    indices=(y!=-100).nonzero().flatten().tolist()
    assistant_index=x.tolist().index(tok.assistant_id)
    assert indices==list(range(assistant_index,len(y)))
    assert y[indices].tolist()==tok.encode(r["回答"])+[tok.eos_id]
    assert x[:assistant_index].tolist()==[tok.bos_id,tok.user_id]+tok.encode(r["提問"])+[tok.eos_id]
    render.append({"request":r["提問"],"reply":r["回答"],"input_tokens":len(x),
        "effective_target_tokens_including_eos":len(indices),"first_answer_label_at_assistant_input":assistant_index,
        "last_target_eos":int(y[-1]),"input_ids":x.tolist(),"labels":y.tolist()})
write("render-contract.json",render)

raw=json.loads((OUT/"inputs/docs/course-experiments/results/style.json").read_text());result=raw["results"]
original=(OUT/"original-run-code/scripts/course_experiments/behavior.py").read_text()
tree=ast.parse(original)
selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {"_conversation","_style_record","_style_metrics"}]
ns={"json":json}
exec(compile(ast.Module(body=selected,type_ignores=[]),"original-behavior-functions","exec"),ns)
style_record,style_metrics=ns["_style_record"],ns["_style_metrics"]
arithmetic=split_records(arithmetic_records(),seed=raw["seed"])
dates=[]
for day in range(1,25):
    date=f"2026-10-{day:02d}"
    dates.append(ns["_conversation"](f"task=date;date={date};confirm","已確認"+date+"。","date-"+date,style="clarification"))
    dates.append(ns["_conversation"](f"task=date;id={day};date=?;confirm","請提供日期。","date-"+date,style="clarification"))
date_parts=split_records(dates,seed=raw["seed"])
conditional={s:[style_record(row,style,True) for row in rows for style in ("concise","vivid","json")]+date_parts[s] for s,rows in arithmetic.items()}
datafacts={}
for split,rows in conditional.items():
    data=("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows)).encode()
    assert sha(data)==result["data"][split]["sha256"]
    (OUT/f"reconstructed-{split}.jsonl").write_bytes(data)
    datafacts[split]={"arithmetic_questions":len(arithmetic[split]),"style_records":3*len(arithmetic[split]),
        "date_records":len(date_parts[split]),"total_records":len(rows),"sha256":sha(data),
        "families":sorted({r["family"] for r in rows})}
assert records_sha256(conditional["train"])==result["training"]["records_sha256"]
assert len(conditional["train"])==185
assert all(not(set(datafacts[a]["families"])&set(datafacts[b]["families"])) for a,b in [('train','validation'),('train','test'),('validation','test')])
assert '2+2' in datafacts['test']['families'] and '2+2' not in datafacts['train']['families']
lengths=[int((render_chat(r["messages"])[1]!=-100).sum()) for r in conditional['train']]
sampler=random.Random(raw['seed'])
effective=sum(sum(sampler.choices(lengths,k=16)) for _ in range(1000))
assert effective==result['training']['effective_tokens']==324973
datafacts['training_reconstruction']={"steps":1000,"batch_records":16,"draws":16000,
    "effective_target_tokens":effective,"includes_eos":True,"method":"Replay RNG choices over target lengths only; no neural training."}
write('data-reconstruction.json',datafacts)

def inspect_sample(sample,style):
    q=sample['messages'][0]['content'];m=re.fullmatch(r'style=(concise|vivid|json); (\d+)\+(\d+)=\?',q)
    assert m and m[1]==style
    value=int(m[2])+int(m[3]);text=sample['generated'];ids=sample['generated_ids']
    report=generation_report(tok,ids)
    assert report['valid_answer_tokens'] and report['eos'] and ids[-1]==tok.eos_id
    assert report['answer']==text and tok.decode(ids[:-1])==text
    assert all(i>=8 for i in ids[:-1])
    if style=='json':
        parsed=json.loads(text);style_ok=isinstance(parsed,dict) and set(parsed)=={'answer'} and type(parsed['answer']) is int
        content_ok=style_ok and parsed['answer']==value
    elif style=='vivid':
        style_ok='像把兩組積木合在一起再數' in text;content_ok=text.split('，',1)[0].strip()==str(value)
    else:
        style_ok=text.strip().isdigit();content_ok=text.strip()==str(value)
    return {'prompt':q,'expected_integer':value,'generated':text,'style_correct':style_ok,'content_correct':content_ok,
        'eos':True,'generated_token_count_including_eos':len(ids),'no_hidden_controls':True}

recomputed={}
for style,report in result['prompt_only_comparison_same_weights'].items():
    samples=[inspect_sample(s,style) for s in report['samples']]
    assert len(samples)==7
    rows=[style_record(row,style,True) for row in arithmetic['test']]
    historical=style_metrics(copy.deepcopy(report),rows)['rubric'][style]
    counts={'records':len(samples),'style_correct':sum(s['style_correct'] for s in samples),
        'content_correct':sum(s['content_correct'] for s in samples),'eos':sum(s['eos'] for s in samples)}
    assert counts=={'records':7,'style_correct':7,'content_correct':0,'eos':7}
    assert historical['style_correct']==7 and historical['content_correct']==0
    recomputed[style]={'counts':counts,'historical_rubric':historical,'samples':samples}
for split in ('validation','test'):
    historical=style_metrics(copy.deepcopy(result['after'][split]),conditional[split])
    assert historical['rubric']==result['after'][split]['rubric']
    for sample in historical['samples']:
        assert generation_report(tok,sample['generated_ids'])['valid_answer_tokens']
    recomputed[split]={'records':len(historical['samples']),'recomputed_rubric':historical['rubric']}
write('recomputed-results.json',recomputed)

json_variants=[]
row=style_record({'a':2,'b':2,'family':'2+2'},'json',True)
for text,valid,content in [('{"answer":3}',True,False),('{"answer": 3}',True,False),
    ('{"answer":4}',True,True),('{"answer":true}',False,False),('{"answer":4.0}',False,False),
    ('{"answer":4,"extra":0}',False,False),('```json\n{"answer":4}\n```',False,False),('[4]',False,False)]:
    observed=style_metrics({'samples':[{'generated':text}]},[row])['rubric']['json']
    assert observed['style_correct']==int(valid) and observed['content_correct']==int(content)
    json_variants.append({'text':text,'style_valid':valid,'content_correct':content})
write('json-rubric-boundaries.json',json_variants)

# Verify real TinyLM generate argmax mechanics, determinism, no parameter update.
# This random miniature model is an API test, never evidence of style ability.
torch.manual_seed(42)
model=TinyLM(ModelConfig(width=8,layers=1,max_length=128))
state_before={n:v.detach().clone() for n,v in model.state_dict().items()}
greedy=[]
for style in ('concise','vivid'):
    prompt=torch.tensor([[tok.bos_id,tok.user_id]+tok.encode(f'style={style}; 2+2=?')+[tok.eos_id,tok.assistant_id]])
    model.eval();expected=model(prompt)['logits'][:,-1].argmax(-1).item();model.train()
    generated=generate(model,prompt,max_new_tokens=2)
    repeated=generate(model,prompt,max_new_tokens=2)
    assert generated[0,prompt.shape[1]].item()==expected and torch.equal(generated,repeated) and model.training
    greedy.append({'style':style,'expected_argmax_first_id':expected,'generated_new_ids':generated[0,prompt.shape[1]:].tolist(),
        'repeat_identical':True,'training_mode_restored':True})
assert all(torch.equal(state_before[n],v) for n,v in model.state_dict().items())
greedy.append({'weights_unchanged':True,'parameter_gradients_absent':all(p.grad is None for p in model.parameters()),
    'scope':'Tiny random CPU API test only; original trained ability supported solely by saved run and implementation.'})
write('greedy-contract.json',greedy)
write('check-summary.json',{'original_fence_records':4,'single_style_records':2,'reversed_order_records':4,
    'rendered_record_count':len(render),'same_weight_saved_samples':21,'style_per_condition':7,'correct_per_condition':0,
    'eos_per_condition':7,'raw_json_answer':'{"answer": 3}','main_table_literal_json_answer':'{"answer":3}',
    'literal_spacing_discrepancy':True,'all_assertions_passed':True})
print(json.dumps({'all_assertions_passed':True,'same_weight_samples_checked':21,'reconstructed_train_records':185,
    'effective_training_targets_recomputed':effective,'local_device':'cpu','training_runs':0},ensure_ascii=False))
