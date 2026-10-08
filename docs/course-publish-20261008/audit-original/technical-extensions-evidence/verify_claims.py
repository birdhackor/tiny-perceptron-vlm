"""Independent, bounded checks of cited raw records; no model training or sampling."""
import ast
import copy
import hashlib
import io
import json
import math
import re
import time
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FREEZE = ROOT.parent / "freeze"
RESULT = {}
INPUT_METADATA_MISMATCHES = []
retrieval_path = FREEZE / "implementation/tiny_perceptron/retrieval.py"
scope = {}
exec(compile(retrieval_path.read_text(), str(retrieval_path), "exec"), scope)
app = FREEZE / "implementation/scripts/course_experiments/applications.py"
tree = ast.parse(app.read_text())
functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {"_parse_json_action", "_verify_reasoning"}]
scope.update(math=math, json=json, re=re)
exec(compile(ast.Module(body=functions, type_ignores=[]), str(app), "exec"), scope)

def load(name):
    return json.loads((ROOT / f"{name}.json").read_text())

def generation(g, allow_mutated_message_snapshot=False):
    ids = [1]
    role = {"user":3,"assistant":4,"system":7}
    for m in g["messages"]:
        ids += [role[m["role"]]] + [b+8 for b in m["content"].encode()] + [2]
    ids += [4]
    if ids != g["input_ids"]:
        assert allow_mutated_message_snapshot
        matching_prefixes=[]
        for n in range(len(g['messages'])):
            actual=[1]
            for m in g['messages'][:n]:actual += [role[m['role']]]+[b+8 for b in m['content'].encode()]+[2]
            actual += [4]
            if actual==g['input_ids']:matching_prefixes.append(n)
        assert matching_prefixes==[2]
        INPUT_METADATA_MISMATCHES.append({'actual_input_tokens':g['input_tokens'],'serialized_stored_messages_tokens':len(ids),'actual_matches_first_messages':2})
    assert len(g['input_ids']) == g["input_tokens"]
    for s in g.get("samples", []):
        generated_sample(s)

def generated_sample(s):
    ids=s["generated_ids"]
    ended=bool(ids and ids[-1]==2)
    content=ids[:-1] if ended else ids
    decoded=bytes(i-8 for i in content if i>=8).decode(errors="replace")
    assert decoded==s["generated"]
    assert ended==s["eos"]
    assert len(ids)==s["generated_tokens"]
    assert [i for i in content if i<8]==s["invalid_special_tokens"]

# All RAG rows, not just the aggregate table.
x=load("rag");r=x["results"];rows=r["samples"]
summary={}
for mode in sorted({row["mode"] for row in rows}):
    subset=[row for row in rows if row["mode"]==mode]
    for row in subset:
        generation(row["generation"])
        s=row["generation"]["samples"][0];text=s["generated"]
        clean=not s["invalid_special_tokens"]
        parsed=re.fullmatch(r"([A-E][0-9])\[(D[0-9])\]",text)
        address=parsed[1] if parsed else None;citation=parsed[2] if parsed else None
        independent={"exact_match":clean and text==row["expected"],
                     "citation_valid":clean and citation in {d['id'] for d in row['documents']},
                     "supported_by_cited_source":clean and bool(parsed) and any(d['id']==citation and f'address={address}' in d['text'] for d in row['documents']),
                     "unknown":clean and text=="UNKNOWN"}
        fact=row["context_fact"]
        independent["fact_answer_correct"]=clean and text==f"{fact['address']}[{fact['source']}]"
        for field,value in independent.items():assert value==row[field]
    summary[mode]={field:sum(row[field] for row in subset) for field in independent}
    summary[mode].update(n=len(subset),input_range=[min(row['generation']['input_tokens'] for row in subset),max(row['generation']['input_tokens'] for row in subset)],eos=sum(row['generation']['samples'][0]['eos'] for row in subset))
    for field in independent:
        assert summary[mode][field]==r['metrics'][mode][field]['numerator']
        assert len(subset)==r['metrics'][mode][field]['denominator']
for row in r['icl_samples_same_model_weights']:generation(row['generation'])
baseline={row['family']:row for row in rows if row['mode']=='correct_context'}
paired={}
for mode in ['changed_address_context','changed_source_context']:
    subset=[row for row in rows if row['mode']==mode]
    n=sum(row['fact_answer_correct'] and baseline[row['family']]['fact_answer_correct'] for row in subset)
    assert n==r['paired_counterfactuals'][mode]['both_answers_correct']['numerator'];paired[mode]=n
RESULT['rag']={'rows_checked':len(rows),'mode_summary':summary,'paired_both_correct':paired,'icl':[(z['mode'],z['expected'],z['generation']['samples'][0]['generated']) for z in r['icl_samples_same_model_weights']]}

# All normal and step-capped tool episodes, preserving executions and failures.
r=load('tools')['results'];normal=r['samples'];capped=r['one_step_cap_samples']
events=[]
for e in normal+capped:
    for ev in e['trace']:
        if 'generation' in ev:
            generation(ev['generation'],allow_mutated_message_snapshot=True);events.append(ev)
        if ev.get('executed'):
            result=scope['call_tool'](ev['parsed']);assert result==ev['tool_result']
    assert sum(bool(ev.get('executed')) for ev in e['trace'])==e['actual_tool_calls']
assert sum(e['correct'] for e in normal)==19
assert sum(e['actual_tool_calls'] for e in normal)==18
assert all(e['status']=='step_limit' and e['actual_tool_calls']==1 for e in capped)
RESULT['tools']={'normal':len(normal),'correct':sum(e['correct'] for e in normal),'normal_calls':sum(e['actual_tool_calls'] for e in normal),'step_capped':len(capped),'generation_count':len(events),'eos_count':sum(ev['generation']['samples'][0]['eos'] for ev in events),'failures':[{'question':e['question'],'expected':e['expected'],'status':e['status'],'generated':[ev['generation']['samples'][0]['generated'] for ev in e['trace'] if 'generation' in ev]} for e in normal if not e['correct']]}

# Every tool-choice validation/test/diagnostic row; independent group counts.
r=load('tool_choice')['results'];sc={}
for split in ['validation','test','paraphrase_diagnostic']:
    rows=r[split]['samples'];counts=Counter()
    for z in rows:
        record=z['record'];generation(z['generation']);s=z['generation']['samples'][0]
        pred=s['generated'] if s['eos'] and not s['invalid_special_tokens'] and s['generated'] in ['DIRECT','TOOL','ASK'] else None
        assert pred==z['chosen_action']
        counts['correct']+=pred==record['expected_action'];counts['n']+=1
        if record['expected_action']=='TOOL':counts['needed']+=1;counts['needed_selected']+=pred=='TOOL'
        else:counts['not_needed']+=1;counts['extra']+=pred=='TOOL'
        if not record['calculator_available']:counts['unavailable']+=1;counts['unavailable_selected']+=pred=='TOOL'
    assert counts['correct']==r[split]['metrics']['accuracy']['numerator']
    sc[split]=dict(counts)
RESULT['tool_choice']=sc

# Every candidate, reparse equations and all eight budget aggregates.
r=load('reasoning')['results'];com={}
for mode,v in r['comparison'].items():
    counts=[]
    for budget in v['budgets']:
        sums=Counter()
        for z in budget['samples']:
            generation(z['generation']);answers=[]
            for c in z['candidates']:
                generated_sample(c);check=scope['_verify_reasoning'](c,z,mode)
                for field,value in check.items():assert value==c[field],(mode,field,c)
                if c['final_answer'] is not None:answers.append(c['final_answer'])
                sums['tokens']+=len(c['generated_ids']);sums['candidates']+=1
            coverage=any(c['final_correct'] for c in z['candidates'])
            majority=Counter(answers).most_common(1)[0][0] if answers else None
            verified=next((c['final_answer'] for c in z['candidates'] if c['fully_verified']),None)
            assert coverage==z['oracle_coverage'] and majority==z['majority_answer'] and verified==z['verified_answer']
            sums['coverage']+=coverage;sums['majority']+=majority==z['truth'];sums['verifier']+=verified==z['truth'];sums['n']+=1
        for field,metric in [('coverage','oracle_coverage'),('majority','majority_accuracy'),('verifier','verifier_accuracy')]:
            assert sums[field]==budget[metric]['numerator'] and sums['n']==budget[metric]['denominator']
        assert sums['tokens']==budget['generated_tokens']
        counts.append({'k':budget['candidate_count'],**dict(sums),'generation_seconds':budget['generation_seconds']})
    com[mode]={'base_hash':v['base_state_sha256'],'optimizer_steps':v['training']['steps'],'supervised_tokens':v['training']['effective_tokens'],'budgets':counts}
policies={}
for mode in ['strict','weak_proxy']:
    after=r['reinforce'][mode]['after'];sums=Counter()
    for z in after['samples']:
        sums['greedy_correct']+=z['greedy_action']==str(z['truth'])
        for c in z['samples']:
            strict=bool(re.fullmatch(r'-?[0-9]+',c['generated'])) and int(c['generated'])==z['truth']
            proxy=str(z['truth']) in c['generated'].split()
            assert strict==bool(c['strict_reward']) and proxy==bool(c['proxy_reward'])
            sums['samples']+=1;sums['strict_correct']+=strict;sums['proxy']+=proxy;sums['enumeration']+=c['action_id']==16
    policies[mode]=dict(sums)
RESULT['reasoning']={'comparison':com,'supervised_token_ratio':com['steps']['supervised_tokens']/com['direct']['supervised_tokens'],'policy_after':policies}

# Execute all standard-library short authored examples (exclude framework cells).
short={}
for page in ['A.1','A.2','A.3','A.4','A.5','A.6','A.7','B.1','B.2','B.3','B.4','B.5','B.7','B.8','C.2','C.3','C.4','C.5','C.6']:
    source=(FREEZE/'sources'/f'{page}.md').read_text()
    blocks=re.findall(r'```python\n(.*?)```',source,re.S)
    output=[]
    for code in blocks:
        code=re.sub(r'from tiny_perceptron.retrieval import .*\n','',code)
        ns={k:scope[k] for k in ['retrieve','lexical_terms','call_tool','tool_loop']}
        stream=io.StringIO()
        with redirect_stdout(stream):exec(compile(code,f'{page} authored stdlib cell','exec'),ns)
        output.append(stream.getvalue())
    short[page]=output
RESULT['authored_stdlib_examples']=short
RESULT['manual_byte_and_gradient_checks']={'C.1_utf8_lengths':[len(z.encode()) for z in ['5','從2開始數三次：3、4、5，所以是5。']], 'B.6_answer_target_ids':[b+8 for b in b'TOOL']+[2], 'C.7_analytic_gradient':[0.25,-0.25], 'C.7_updated_logits':[-0.025,0.025], 'C.7_updated_probabilities':[1/(1+math.exp(0.05)),1/(1+math.exp(-0.05))]}
RESULT['tool_generation_message_snapshot_defect']={'mismatched_generations':len(INPUT_METADATA_MISMATCHES),'total_tool_generations':len(events),'instances':INPUT_METADATA_MISMATCHES}
RESULT['limits']=['No model weights were loaded, trained, or regenerated.','Raw-record and static-source audit; CUDA/MPS/Colab and full CLI training not executed.','PyTorch unavailable in this execution environment; C.7 numbers independently checked analytically, not by executing PyTorch.']
(ROOT/'verification-results.json').write_text(json.dumps(RESULT,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(RESULT,ensure_ascii=False,indent=2))
