"""Recompute course-guide numbers using only adjacent small reviewer artifacts."""

import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
data = json.loads((HERE / 'measurement-row-projections-original.json').read_text())
expected = json.loads((HERE / 'measurement-recomputation-original.json').read_text())
av = data['selected_AV_rows']
sound = data['sound_rows']
chain = data['sound_chain_rows']
choice = data['choice_rows']
types = {}
stored_answers_match_logits = True
for kind in ['choice', 'info', 'rubric']:
    hit = 0
    for row in choice:
        task = row['tasks'][kind]
        index = min(range(len(task['ids'])), key=lambda i: (-task['logits'][i], task['ids'][i]))
        answer = task['ids'][index]
        stored_answers_match_logits &= answer == task['answer']
        hit += answer == task['gold']
    types[kind] = hit
accepted = errors = 0
for row in choice:
    task, info = row['tasks']['choice'], row['choice_info']
    logits = task['logits']
    index = min(range(len(logits)), key=lambda i: (-logits[i], task['ids'][i]))
    peak = max(logits) / data['frozen_T']
    exps = [math.exp(v / data['frozen_T'] - peak) for v in logits]
    p_max = max(exps) / sum(exps)
    info_index = min(range(len(info['ids'])), key=lambda i: (-info['logits'][i], info['ids'][i]))
    answer = task['ids'][index]
    accept = data['frozen_tau'] is not None and p_max >= data['frozen_tau'] and answer != 'none' and info['ids'][info_index] == 'yes'
    accepted += accept
    errors += bool(accept and (answer != task['gold'] or not row['world_supported'] or not row['choice_information_sufficient']))
b = data['B_rows']
a = data['old_A_rows']
old_a = {task:{'n':sum(r['task']==task for r in a),
         'old_correct':sum(r['old_correct'] for r in a if r['task']==task),
         'wrapper_correct':sum(r['wrapper_correct'] for r in a if r['task']==task)} for task in sorted(set(r['task'] for r in a))}
observed = {
    'AV': {'n':len(av), 'paired':sum(r['answers']==r['gold'] for r in av),
           'waveform_swap_paired':sum(r['waveform_swap_answers']==r['waveform_swap_gold'] for r in av)},
    'sound':{'n':len(sound),'mean_snr_db':sum(r['snr'] for r in sound)/len(sound),
             'gold_correct':sum(r['program_success'] for r in sound),'legal':sum(r['legal'] for r in sound),
             'chain_n':len(chain),'chain_correct':sum(r['chain_correct'] for r in chain)},
    'C':{'n':len(choice),'correct':types,'accepted':accepted,'errors':errors,'risk':errors/accepted},
    'B':{'n':len(b),'old_correct':sum(r['old_answer']==r['gold'] for r in b),
         'wrapper_correct':sum(r['wrapper_answer']==r['gold'] for r in b)},
    'stored_answers_match_raw_logits':stored_answers_match_logits,
    'old_A':{'observations':len(a),'parents':len(set(r['family'] for r in a)),
             'tasks':old_a,'all_counts_unchanged':all(r['old_correct']==r['wrapper_correct'] for r in old_a.values())},
}
assert stored_answers_match_logits
assert observed['AV']['paired'] == expected['AV'][data['selected_AV_job']]['paired_correct']
assert observed['AV']['waveform_swap_paired'] == expected['AV'][data['selected_AV_job']]['waveform_swap_paired_correct']
assert abs(observed['sound']['mean_snr_db'] - expected['sound']['mean_active_reference_snr_db']) <= 1e-12
assert observed['sound']['gold_correct'] == expected['sound']['gold_program_correct']
assert observed['sound']['chain_correct'] == expected['sound']['chain_correct']
assert types == expected['C']['type_correct']
assert [accepted, errors] == [expected['C']['accepted'], expected['C']['errors']]
assert observed['B']['old_correct'] == observed['B']['wrapper_correct'] == expected['B_retention']['old_correct']
assert old_a == expected['old_A_retention']['tasks']
output = {'created_at':datetime.now(timezone.utc).isoformat(), 'status':'pass',
          'environment':{'python':platform.python_version(),'device':'CPU pure arithmetic; no torch import'},
          'observed':observed, 'scope':'Only adjacent Git-portable projections were read; no ignored outputs or external archive is required for this replay.'}
(HERE / 'portable-replay-original.json').write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(output, ensure_ascii=False))
