"""本次獨立離線核對：只執行短例及重算已保存原始紀錄，不訓練、不呼叫模型。"""
import contextlib
import copy
import datetime
import hashlib
import io
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(ROOT))
B = ROOT / 'docs/course-repair-20261008/reviews/freeze-01'
E = B / 'evidence/technical-extensions'
M = json.loads((B / 'manifest.json').read_text())
T = json.loads((B / 'checks/technical-inputs-manifest.json').read_text())
used = {}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def raw(n):
    p = B / 'freeze/technical-data' / n
    h = sha(p)
    assert h == T['files_sha256'][n]
    used[str(p.relative_to(ROOT))] = h
    return json.loads(p.read_text())


def live(n):
    p = ROOT / n
    used[n] = sha(p)
    return json.loads(p.read_text())


out = {'reviewer': '/root/repair_tech_extensions', 'at': datetime.datetime.now(datetime.timezone.utc).isoformat()}
out['live_implementation_differences'] = {
    n: {'expected': h, 'actual': sha(ROOT / n)}
    for n, h in M['implementation_sha256'].items()
    if sha(ROOT / n) != h
}
assert set(out['live_implementation_differences']) <= {'pyproject.toml'}
# 凍結來源短例完整執行；沒有取出 bash 長配方。
examples = {}
for n in ['A.2', 'B.3', 'B.4', 'B.6', 'C.7']:
    p = B / f'freeze/sources/{n}.md'
    code = re.findall(r'```python\n(.*?)\n```', p.read_text(), re.S)[0]
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        exec(compile(code, str(p), 'exec'), {})
    examples[n] = {'source_sha256': sha(p), 'stdout': stream.getvalue()}
out['short_examples'] = examples

import torch
from tiny_perceptron.retrieval import call_tool, tool_loop
from tiny_perceptron.data import ByteTokenizer, render_chat, split_documents, toy_documents
out['runtime'] = {'python': sys.version, 'torch': torch.__version__, 'device': 'cpu'}
out['small_changes'] = {'B.3_b46': call_tool({'name': 'multiply', 'arguments': {'a': 123, 'b': 46}}),
    'B.4_budget2': tool_loop([{'name': 'add', 'arguments': {'a': 2, 'b': 3}}, {'done': True, 'answer': 5}], 2),
    'B.4_wrong_done': tool_loop([{'name': 'add', 'arguments': {'a': 2, 'b': 3}}, {'done': True, 'answer': 6}], 2)}
logits = torch.tensor([0., 0.], requires_grad=True)
loss = -(-.5) * logits.log_softmax(0)[1]
loss.backward()
out['small_changes']['C.7_reward0'] = {'gradient': logits.grad.tolist(), 'after': (logits - .1 * logits.grad).detach().softmax(0).tolist()}
out['toy_document_counts'] = {k: len(v) for k, v in split_documents(toy_documents(), 42).items()}

rag = raw('docs/course-experiments/results/rag.json')['results']
rows = rag['samples']
base = {r['family']: r for r in rows if r['mode'] == 'correct_context'}
rag_result = {}
for mode in ['correct_context', 'changed_address_context', 'changed_source_context']:
    selected = [r for r in rows if r['mode'] == mode]
    answers = []
    for r in selected:
        f = r['context_fact']
        expected = f"{f['address']}[{f['source']}]"
        s = r['generation']['samples'][0]
        assert r['fact_answer_correct'] == (s['generated'] == expected and not s['invalid_special_tokens'])
        answers.append({'key': r['family'], 'expected': expected, 'generated': s['generated'], 'correct': r['fact_answer_correct']})
    rag_result[mode] = {'count': len(selected), 'correct': sum(r['fact_answer_correct'] for r in selected),
        'both_with_baseline_correct': sum(r['fact_answer_correct'] and base[r['family']]['fact_answer_correct'] for r in selected), 'answers': answers}
out['rag_raw_recount'] = rag_result

tools = raw('docs/course-experiments/results/tools.json')['results']
role_ids = {'system': 7, 'user': 3, 'assistant': 4}
recon = []
for key in ['samples', 'one_step_cap_samples']:
    for j, episode in enumerate(tools[key]):
        first = episode['trace'][0]['generation']
        current = copy.deepcopy(first['messages'][:2])
        assert current[1]['content'] == episode['question']
        for k, event in enumerate(episode['trace']):
            g = event['generation']
            ids = [1]
            for msg in current:
                ids += [role_ids[msg['role']]] + [v + 8 for v in msg['content'].encode()] + [2]
            ids += [4]
            assert ids == g['input_ids'] and len(ids) == g['input_tokens']
            recon.append({'pointer': f'/results/{key}/{j}/trace/{k}/generation', 'metadata_differs': current != g['messages'], 'ids_equal': True})
            if event.get('executed'):
                req = event['parsed']
                actual = call_tool(req)
                assert actual == event['tool_result']
                current += [{'role': 'assistant', 'content': g['samples'][0]['generated']}, {'role': 'user', 'content': f'TOOL_RESULT:{actual}'}]
generations = [event['generation']['samples'][0] for key in ['samples', 'one_step_cap_samples'] for ep in tools[key] for event in ep['trace']]
calls = [ev for ep in tools['samples'] for ev in ep['trace'] if ev.get('executed')]
normal = tools['samples']
for ep in normal:
    executed = [ev for ev in ep['trace'] if ev.get('executed')]
    correct_answer = ep['status'] == 'done' and type(ep['answer']) in (int,float) and ep['answer'] == ep['expected']
    behavior = not executed if ep['operation'] == 'copy' else bool(executed and executed[-1]['finite_result'] and any(e['request_matches_question'] for e in executed) and ep['answer'] == executed[-1]['tool_result'])
    assert ep['correct'] == (correct_answer and behavior)
out['tools_raw_recount'] = {'normal_episodes': len(normal), 'success': sum(ep['correct'] for ep in normal),
    'normal_calls': len(calls), 'request_matches_question': sum(e['request_matches_question'] for e in calls),
    'copy_no_call': sum(ep['operation'] == 'copy' and ep['actual_tool_calls'] == 0 for ep in normal),
    'capped_episodes': len(tools['one_step_cap_samples']), 'capped_step_limit': sum(ep['status'] == 'step_limit' for ep in tools['one_step_cap_samples']),
    'generations': len(generations), 'eos': sum(s['eos'] for s in generations), 'invalid_controls': sum(bool(s['invalid_special_tokens']) for s in generations),
    'reconstruction_records': len(recon), 'metadata_mismatches': sum(r['metadata_differs'] for r in recon), 'all_input_ids_verified': True,
    'failure': [{'question':ep['question'],'status':ep['status'],'trace': [ev['generation']['samples'][0]['generated'] for ev in ep['trace']]} for ep in normal if not ep['correct']],
    'successful_add9': [ep for ep in normal if ep['question']=='CALC:add(9,9)'][0]}
(E/'independent-tool-reconstruction.json').write_text(json.dumps(recon, ensure_ascii=False, indent=2)+'\n')

reason = raw('docs/course-experiments/results/reasoning.json')['results']['reinforce']
policies = {}
for label in ['strict', 'weak_proxy']:
    s = reason[label]['after']['samples']
    sampled = [x for r in s for x in r['samples']]
    for r in s:
        assert r['truth'] == r['a']+r['b']+r['c']
        assert r['greedy_action'] == ([str(i) for i in range(16)]+[' '.join(map(str,range(16)))])[max(range(17),key=lambda i:r['probabilities'][i])]
    policies[label] = {'new_questions':len(s),'samples':len(sampled),'greedy_correct':sum(r['greedy_action']==str(r['truth']) for r in s),
        'sample_strict_correct':sum(x['generated']==str(r['truth']) for r in s for x in r['samples']),
        'proxy_correct':sum(str(r['truth']) in x['generated'].split() for r in s for x in r['samples']),
        'enumeration':sum(x['action_id']==16 for x in sampled), 'first_question':s[0]['question'], 'first_greedy':s[0]['greedy_action']}
out['reasoning_raw_recount'] = policies
choice = raw('docs/course-experiments/results/tool_choice.json')['results']
out['tool_choice_recorded_counts'] = {k:choice[k]['metrics'] for k in ['test','paraphrase_diagnostic']}
out['tool_choice_sample_keys'] = list(choice['test']['samples'][0])

v2 = raw('docs/selftrained/v2-manifest.json')
final = raw('docs/selftrained/results/v2-final-public-results.json')
summary = {'archive_bytes':v2['package']['bytes'], 'manifest_files':len(v2['records'])+len(v2['assets']), 'architectures':{}}
for arch,stage in [('moe','moe-native'),('dense','dense-weighted')]:
    metrics=raw(f'docs/selftrained/results/public-raw/{arch}/test/metrics.json')
    receipt=raw(f'docs/selftrained/results/training-raw/{stage}/raw/train-receipt.json')
    freeze=raw(f'docs/selftrained/results/public-raw/{arch}/freeze/frozen.json')
    summary['architectures'][arch]={'parameters':final['architectures'][arch]['capacity']['parameters'], 'completed_steps':receipt['steps'],
        'selection':receipt['selection'],'train_records':receipt['train_records'],'validation_records':receipt['validation_records'],
        'test_count':metrics['count'],'test_complete':metrics['evaluation_complete'],'thresholds':metrics['thresholds'],
        'tool_roundtrip':metrics['per_task_final_reply']['tool_call']['tool_roundtrip'],'voice_qa':metrics['per_task_final_reply']['voice_qa']['semantic'],
        'freeze_selection':freeze['selection'],'freeze_test_once':freeze['test_once']}
out['v2_raw_summary']=summary
models=live('docs/course-experiments/public-models.json')
old=live('docs/course-experiments/capstone-public.json')
out['public_model_counts']={'groups':len(models['models']),'all_manifest_files':sum(len(x['files']) for x in models['models']), 'checkpoint_files':sum(f['path'].endswith('.pt') for x in models['models'] for f in x['files']), 'old_capstone_groups':len(old['models']),'old_capstone_keys':[list(x) for x in old['models']][:1]}
release=live('docs/natural-assistant/v4/public-release.json')
out['natural_release']={k:release[k] for k in ['selected_variant','base_model','asr_model','dependency_versions','adapter_parameters']}
ci=live('docs/validation-artifacts/release-compatibility-ci.json')
out['compatibility_ci']={k:ci[k] for k in ['scope','revision','status','jobs','verification_program_sha256']}
out['additional_local_inputs_sha256']=used
(E/'offline-checks.json').write_text(json.dumps(out, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ['tools_raw_recount','compatibility_ci','additional_local_inputs_sha256','tool_choice_recorded_counts','rag_raw_recount']},ensure_ascii=False,indent=2))
print('tool_counts',json.dumps({k:v for k,v in out['tools_raw_recount'].items() if k!='successful_add9'},ensure_ascii=False))
print('rag_counts',json.dumps({k:{x:y for x,y in v.items() if x!='answers'} for k,v in out['rag_raw_recount'].items()},ensure_ascii=False))
