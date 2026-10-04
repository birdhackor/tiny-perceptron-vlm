"""Bounded independent audit: existing records and tiny CPU examples; no model loading."""
from pathlib import Path
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
import math
import platform
import random
import unicodedata
import torch
from torch import nn
from tiny_perceptron.alignment import LoRALinear

ROOT = Path(__file__).resolve().parents[5]
RAW = ROOT / 'docs/natural-assistant/evidence/v4-runtime/validation-37219466611/blind-review/raw'
BLIND = RAW.parent
TRAIN = ROOT / 'outputs/natural-v4/modal-runs/train-37217452291/natural-natural-v4-train-37217452291-1/review'

def read(path):
    return json.loads(path.read_text())

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

manifest_path = ROOT / 'docs/natural-assistant/v4/manifest.json'
protocol_path = ROOT / 'docs/natural-assistant/v4/validation-protocol-lower-lr.json'
manifest, protocol = read(manifest_path), read(protocol_path)
variants = protocol['candidate_order']
generation_paths = {v: RAW / f'generations-{v}.json' for v in variants}
generations = {v: read(p) for v, p in generation_paths.items()}
records = {v: {(g['id'], g['task']): g for g in gs} for v, gs in generations.items()}
assert all(len(gs) == len(records[v]) == 132 for v, gs in generations.items())
assert all(set(records[v]) == set(records['base']) for v in variants)
rows = [r for r in manifest['rows'] if r['split'] == 'validation']
audio = [r for r in manifest['audio_rows'] if r['split'] == 'validation']
cases = {}
for row in rows:
    group = ('photo_summary' if row['references'].get('qa_type') == 'scene' else 'photo_fact') if row['task'] == 'scene' else {'chat': 'text_chat', 'text_presence': 'text_presence', 'ocr': 'single_ocr', 'ocr_order': 'ordered_ocr'}[row['task']]
    cases[(row['id'], row['task'])] = (group, row)
for row in audio:
    if row['task'] == 'speech_chat':
        cases[(row['id'], 'typed_chat')] = ('voice_typed_reference_chat', row)
        cases[(row['id'], 'speech_chat')] = ('voice_actual_asr_chat', row)
denominators = Counter(group for group, row in cases.values())
assert len(cases) == 132 and len(audio) == 16
assert all(denominators[g] == protocol['validation_denominators'][g] for g in denominators)

def complete(g):
    ids = g['generated_token_ids']
    return (bool(ids) and len(ids) <= 384 and len(ids) == g['generated_tokens'] and ids[-1] in g['eos_token_ids'] and g['ended_with_eos'] and g['stop_reason'] == 'eos' and not g['truncated'] and not g['completion_unknown'])

def normalize(s, whitespace=False):
    s = unicodedata.normalize('NFKC', s).strip()
    return ''.join(s.split()) if whitespace else s

aliases = read(BLIND / 'private-map.json')['aliases']
grade_path = BLIND / 'combined/grades.json'
grades = read(grade_path)['grades']
manual = {(g['case_id'], aliases[g['candidate']]): g['passed'] for g in grades}
assert len(manual) == len(grades) == 303
weights = {'photo_summary': 270, 'photo_fact': 135, 'text_presence': 840, 'single_ocr': 1512, 'ordered_ocr': 5040, 'text_chat': 560, 'voice_typed_reference_chat': 1260, 'voice_actual_asr_chat': 1260}
assert sum(denominators[k] * weights[k] for k in weights) == 75600
computed = {}
for variant in variants:
    counts = dict.fromkeys(weights, 0)
    incomplete = []
    for key, (group, row) in cases.items():
        g = records[variant][key]
        assert g['split'] == 'validation'
        if not complete(g):
            incomplete.append({'case_id': key[1]+':'+key[0], 'tokens': len(g['generated_token_ids']), 'final_token': g['generated_token_ids'][-1], 'eos_ids': g['eos_token_ids'], 'stop_reason': g['stop_reason'], 'unicode_characters': len(g['prediction'])})
        if group in {'photo_summary', 'photo_fact', 'text_chat', 'voice_typed_reference_chat', 'voice_actual_asr_chat'}:
            passed = manual[(key[1]+':'+key[0], variant)]
        else:
            ref = row['references']
            strip = ref.get('strip_whitespace', group == 'single_ocr')
            expected = ref.get('text', row['answer'])
            passed = normalize(g['prediction'], strip) == normalize(expected, strip)
        counts[group] += int(complete(g) and passed)
    macro = ( (Fraction(counts['photo_summary'],28)+Fraction(counts['photo_fact'],56))/2 + Fraction(counts['text_presence'],18) + Fraction(counts['single_ocr'],10) + Fraction(counts['ordered_ocr'],3) + (Fraction(counts['text_chat'],9)+Fraction(counts['voice_typed_reference_chat'],4)+Fraction(counts['voice_actual_asr_chat'],4))/3 )/5
    integer = sum(counts[k] * weights[k] for k in weights)
    assert macro == Fraction(integer,75600)
    computed[variant] = {'correct_counts': counts, 'primary_numerator': integer, 'primary_denominator':75600, 'primary_fraction':str(macro), 'primary_percent':float(macro)*100, 'incomplete':incomplete}
scores_path = BLIND / 'scored/scores.json'
scores = read(scores_path)
assert all(computed[v]['correct_counts'] == scores['variants'][v]['correct_counts'] for v in variants)
assert all(computed[v]['primary_numerator'] == scores['variants'][v]['primary_numerator'] for v in variants)
for variant in variants[1:]:
    c = computed[variant]
    gates = {k: c['correct_counts'][k] >= max(0,computed['base']['correct_counts'][k] - int(protocol['adapter_nonregression_gates_vs_base'][k+'_correct_count_minimum'] == 'base minus 1')) for k in weights}
    c['gates'] = gates
    c['eligible'] = not c['incomplete'] and all(gates.values())
selected = 'base'
for variant in variants[1:]:
    if computed[variant]['eligible'] and computed[variant]['primary_numerator'] > computed[selected]['primary_numerator']:
        selected = variant
selection_path = ROOT / 'docs/natural-assistant/v4/selection.json'
selection = read(selection_path)
assert selection['selected_variant'] == selected == 'base'
assert selection['validation_result_sha256'] == digest(RAW/'result.json')
assert selection['dataset_manifest_sha256'] == digest(manifest_path)
assert selection['validation_protocol_sha256'] == digest(protocol_path)
assert selection['manual_grades_sha256'] == digest(grade_path)
assert selection['validation_scoring_sha256'] == digest(scores_path)
for file in scores['artifact_binding']['generation_files']:
    assert digest(RAW/file['name']) == file['sha256']

# Same actual ASR transcript and image preprocessing for every candidate.
transcripts_path = RAW / 'transcripts.json'
transcripts = {t['id']:t for t in read(transcripts_path)}
controls = []
for row in audio:
    if row['task'] != 'speech_chat': continue
    original = transcripts[row['id']]['transcript']
    for variant in variants:
        assert records[variant][(row['id'],'speech_chat')]['user'] == original
        assert records[variant][(row['id'],'typed_chat')]['user'] == row['user']
    controls.append({'id':row['id'],'reference':row['user'],'actual_asr':original})
for key in records['base']:
    for variant in variants[1:]:
        for field in ['user','image','image_grid_thw','input_tokens']:
            assert records[variant][key].get(field) == records['base'][key].get(field)
raw_result = read(RAW/'result.json')
assert raw_result['status'] == 'completed' and raw_result['split'] == 'validation'
assert raw_result['model_revision'] == '89644892e4d85e24eaac8bacfd4f463576704203'
assert raw_result['asr_model'] == 'openai/whisper-large-v3-turbo'
assert raw_result['asr_revision'] == '41f01f3fe87f28c78e2fbf8b568835947dd65ed9'

# Recompute genuine historical GPU records; this is not GPU rerun evidence.
training_path = TRAIN/'training.json'
training = read(training_path)
train_rows = [r for r in manifest['rows'] if r['split']=='train']
assert len(train_rows)==2077
expected_ids=[]
for epoch in range(2):
    indices=list(range(len(train_rows))); random.Random(42+epoch).shuffle(indices)
    expected_ids.extend(train_rows[i]['id'] for i in indices)
history=training['history']
assert [h['step'] for h in history] == list(range(1,2078))
assert [id for h in history for id in h['row_ids']] == expected_ids
assert Counter(expected_ids) == Counter({r['id']:2 for r in train_rows})
assert all(math.isfinite(h['answer_token_loss_before_update']) and math.isfinite(h['gradient_norm_before_clip']) and h['supervised_tokens']>0 for h in history)
assert training['status']=='completed' and training['trained_rows']==4154
assert training['learning_rate']==0.00003 and training['gradient_accumulation']==2 and training['seed']==42
assert set(training['optimizer_parameter_names'])==set(training['trainable_parameter_names'])
assert all('lora_' in n and '.visual.' not in n for n in training['optimizer_parameter_names'])
assert len(training['optimizer_parameter_names'])==112
assert sum(training['initial_adapter_tensors'][k] != training['final_adapter_tensors'][k] for k in training['initial_adapter_tensors'])==112
assert training['frozen_parameter_samples_initial']==training['frozen_parameter_samples_final']
checkpoint_files=[]
for step in [1039,2077]:
    folder=TRAIN/f'checkpoints/step-{step:06}'
    t=read(folder/'training.json'); c=read(folder/'checkpoint.json');config=read(folder/'adapter_config.json')
    assert t['completed_steps']==c['completed_steps']==step and len(t['history'])==step
    assert config['r']==8 and config['lora_alpha']==16
    assert t['history']==history[:step]
    archive=next(x for x in training['archived_checkpoints'] if x['completed_steps']==step)
    assert digest(folder/'checkpoint.json')==archive['checkpoint_metadata_sha256']
    assert digest(folder/'training.json')==next(x['sha256'] for x in archive['files'] if x['path']=='training.json')
    checkpoint_files.append({'step':step,'training_sha256':digest(folder/'training.json'),'checkpoint_sha256':digest(folder/'checkpoint.json')})
families=defaultdict(set)
for row in manifest['rows']+manifest['audio_rows']:families[row['split']].add(row['family'])
assert not families['train']&families['validation'] and not families['train']&families['test'] and not families['validation']&families['test']

# Tiny prerequisite mechanism exercise in the current CPU runtime.
toy=[]
for rank in [2,1]:
    torch.manual_seed(42)
    layer=LoRALinear(nn.Linear(4,3),rank=rank,alpha=rank)
    before=layer.base.weight.detach().clone();b_before=layer.b.detach().clone()
    opt=torch.optim.SGD([p for p in layer.parameters() if p.requires_grad],lr=0.1)
    loss=layer(torch.ones(1,4)).square().mean();loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in layer.parameters() if p.requires_grad)
    opt.step()
    result={'rank':rank,'trainable_parameters':sum(p.numel() for p in layer.parameters() if p.requires_grad),'base_unchanged':torch.equal(before,layer.base.weight),'B_changed':not torch.equal(b_before,layer.b),'finite_loss':float(loss.detach())}
    assert result['trainable_parameters']==7*rank and result['base_unchanged'] and result['B_changed']
    toy.append(result)
paths=[manifest_path,protocol_path,selection_path,scores_path,grade_path,transcripts_path,training_path,RAW/'result.json',*generation_paths.values()]
output={'environment':{'python':platform.python_version(),'torch':torch.__version__,'device':'cpu','cuda_available':torch.cuda.is_available()},'denominators':dict(denominators),'variants':computed,'selected':selected,'typed_actual_controls':controls,'historical_training':{'completed_updates':2077,'rows':2077,'row_reads':4154,'each_row_reads':2,'seed':42,'finite_loss_and_gradient_records':len(history),'supervised_token_total':sum(h['supervised_tokens'] for h in history),'updated_recorded_adapter_tensors':112,'base_check':'disclosed sampled positions only','checkpoint_records':checkpoint_files,'original_environment':training['versions'],'source_device':training['device']},'tiny_cpu_prerequisite':toy,'bindings':[{'path':str(p.relative_to(ROOT)),'sha256':digest(p)} for p in paths],'limit':'Existing genuine historical GPU record audit and tiny CPU mechanism only; no GPU training, inference rerun, fresh model download, or population estimate. Manual counts are recomputed from original recorded rubric decisions after personal full raw-answer and source-photo inspection, not treated as infallible semantic truth.'}
print(json.dumps(output,ensure_ascii=False,indent=2))
