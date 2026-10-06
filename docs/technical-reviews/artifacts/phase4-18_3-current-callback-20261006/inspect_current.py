"""Same-reviewer current callback: fingerprints, necessary source slices and raw leaves only.

No torch import, model execution, network, training, model/data download or image render.
"""
import ast
import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OLD = ROOT / 'docs/technical-reviews/artifacts/phase4-18_3-independent-fresh'
TASK = '/root/phase4_factual_coordinator/factual_18_3'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load(path):
    return json.loads(path.read_bytes())

inputs = load(HERE / 'current-inputs.json')
prior = load(HERE / 'prior-canonical.opaque.json')
assert prior['reviewer_task'] == TASK
assert prior['source_sha256'] == inputs['current_primary_sha256']
assert (HERE / 'current-primary.md').read_bytes() == (OLD / 'section.md').read_bytes()
print('CURRENT_PRIMARY_COMPLETE_READ', inputs['current_primary_source'], inputs['current_primary_sha256'])

prior_artifacts = []
for item in prior['artifacts']:
    path = ROOT / item['path']
    assert sha(path) == item['sha256']
    prior_artifacts.append({'id': item['id'], 'path': item['path'], 'sha256': item['sha256'], 'current_hash_match': True})
print('PRIOR_ARTIFACT_HASHES all_match', len(prior_artifacts))

for item in inputs['necessary_context']:
    path = ROOT / item['slice_path']
    assert sha(path) == item['necessary_slice_sha256']
    lesson = item['source'].split('#')[1]
    old = (OLD / 'inputs' / (lesson + '.md')).read_bytes()
    old_prefix = old[:old.find(b'```')]
    item['original_necessary_prefix_sha256'] = hashlib.sha256(old_prefix).hexdigest()
    item['necessary_prefix_equal_to_original'] = path.read_bytes() == old_prefix
    item['original_complete_context_section_sha256'] = hashlib.sha256(old).hexdigest()
    item['complete_context_hash_changed_without_expanded_inspection'] = (
        item['section_sha256_hashed_only'] != item['original_complete_context_section_sha256'])
    print('NECESSARY_CONTEXT', json.dumps(item, ensure_ascii=False))
    print(path.read_text())

current_code = []
specs = [
    ('tiny_perceptron/data.py', [('ByteTokenizer', None), ('render_chat', None)]),
    ('tiny_perceptron/model.py', [('ModelConfig', None), ('loss_sum', None), ('masked_loss', None), ('generate', None)]),
    ('tiny_perceptron/tokenization.py', [('generation_report', None)]),
    ('scripts/course_experiments/compression.py', [('_prompt', None), ('_example', None), ('_hard_targets', None), ('_fit_text', (350, 410)), ('_distill_case', (749, 755)), ('_distill_case', (770, 796))]),
]
for name, selections in specs:
    current = ROOT / name
    saved = OLD / 'code' / name
    assert current.read_bytes() == saved.read_bytes()
    text = current.read_text()
    tree = ast.parse(text)
    for method, ranges in selections:
        node = next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == method)
        if ranges:
            first, last = ranges
        else:
            first, last = node.lineno, node.end_lineno
        lines = text.splitlines()
        selected = '\n'.join(f'{i+1}: {lines[i]}' for i in range(first-1, last))
        print('CURRENT_METHOD', name, method, f'{first}-{last}')
        print(selected)
        current_code.append({'path': name, 'sha256': sha(current), 'preserved_original_path': saved.relative_to(ROOT).as_posix(), 'full_bytes_match': True, 'method': method, 'read_lines': [first, last]})

authority_slices = [
    ('pytorch', 'sources/pytorch-functional.py', [(3493, 3504), (3517, 3522)]),
    ('tokenizers', 'sources/tokenizers-bpe-model.rs', [(593, 598), (614, 619)]),
    ('json', 'sources/python-json.rst', [(248, 255), (497, 499)]),
]
authority = []
for source_id, name, ranges in authority_slices:
    path = OLD / name
    source = next(s for s in prior['sources'] if s['id'] == source_id)
    assert sha(path) == source['snapshot_sha256']
    lines = path.read_text().splitlines()
    for first, last in ranges:
        print('REUSED_ORIGINAL_AUTHORITY', source_id, source['url'], source['version'], f'{first}-{last}', sha(path))
        print('\n'.join(f'{i+1}: {lines[i]}' for i in range(first-1, last)))
    authority.append({'source_id': source_id, 'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path), 'url': source['url'], 'version': source['version'], 'read_lines': ranges, 'reuse': 'Original-source content and support boundary rechecked locally; no refetch.'})

pdf = OLD / 'sources/fact_finish_g_4_minillm_original.pdf'
text_path = pdf.with_suffix('.txt')
source = next(s for s in prior['sources'] if s['id'] == 'minillm')
assert sha(pdf) == source['snapshot_sha256']
text = text_path.read_text()
first = text.index('1   Introduction')
last = text.index('In this work,', first)
excerpt = text[first:last]
assert 'black-box KD, where only' in excerpt and 'teacher-generated texts are accessible' in excerpt
print('REUSED_MINILLM_ORIGINAL_INTRO', source['url'], source['version'], sha(pdf), 'prior_pdftotext_sha', sha(text_path))
print(excerpt)
authority.append({'source_id':'minillm','path':pdf.relative_to(ROOT).as_posix(),'sha256':sha(pdf),'text_path':text_path.relative_to(ROOT).as_posix(),'text_sha256':sha(text_path),'url':source['url'],'version':source['version'],'locator':'p2 Introduction first paragraph through before In this work','reuse':'Same original PDF/text hash; text-only prompt-response classification rechecked, no new paper results used.'})

gsm = OLD / 'sources/gsm8k-readme.md'
source = next(s for s in prior['sources'] if s['id'] == 'gsm8k')
assert sha(gsm) == source['snapshot_sha256']
text = gsm.read_text()
details = text[text.index('## Dataset Details'):text.index('### Calculation Annotations')]
socratic = text[text.index('We generated each Socratic subquestion'):text.index('These data files can be found in:')]
print('REUSED_GSM8K_ORIGINAL_PROVENANCE', source['url'], sha(gsm))
print(details)
print(socratic)
authority.append({'source_id':'gsm8k','path':gsm.relative_to(ROOT).as_posix(),'sha256':sha(gsm),'url':source['url'],'version':source['version'],'locator':'Dataset Details before Calculation Annotations; Socratic provenance paragraph','reuse':'Only original human problem/answer provenance support reused; no dataset download or model result claim.'})

raw_path = OLD / 'code/docs/course-experiments/results/distillation.json'
raw = load(raw_path)
print('RAW_JSON_TOP_KEYS_TYPES_ONLY', {k:type(v).__name__ for k,v in raw.items()})
assert sha(raw_path) == sha(ROOT / 'docs/course-experiments/results/distillation.json')
raw_checks = []
for task, count, correct_count in [('attributes',45,45), ('style_transfer',185,180)]:
    hard = raw['results']['tasks'][task]['hard_target_generation']
    audit = hard['audit']
    assert len(audit) == count
    hits = ends = targets = zero = invalid = 0
    errors = []
    for row in audit:
        ids = row['teacher_ids']
        assert all(type(value) is int and 0 <= value < 264 for value in ids)
        before = ids[:ids.index(2)] if 2 in ids else ids
        gold = [value+8 for value in row['gold_answer'].encode('utf-8')]
        hit = before == gold
        assert hit == row['teacher_correct']
        assert (2 in ids) == row['eos']
        assert ids and ids[-1] == 2 and ids.count(2) == 1
        assert row['invalid_special_tokens'] == []
        assert row['valid_target_tokens'] == len(ids)
        hits += hit; ends += 2 in ids; targets += len(ids); zero += not ids
        invalid += bool(row['invalid_special_tokens'])
        if not hit:
            a=date.fromisoformat(row['gold_answer'].removeprefix('已確認').removesuffix('。'))
            b=date.fromisoformat(row['teacher_answer'].removeprefix('已確認').removesuffix('。'))
            assert (b-a).days == 10
            errors.append({'gold_answer':row['gold_answer'],'teacher_answer':row['teacher_answer'],'difference_days':10})
    assert hits == correct_count and ends == count and zero == invalid == 0
    value={'task':task,'records':count,'correct':hits,'eos':ends,'zero':zero,'invalid_control':invalid,'target_tokens':targets,'wrong':errors}
    raw_checks.append(value)
    print('SAVED_RAW_LEAF_RECHECK', json.dumps(value,ensure_ascii=False))

matched = []
for width in (16,32):
    runs=raw['results']['tasks']['attributes']['runs']
    ce, hard = runs[f'w{width}_ce'], runs[f'w{width}_teacher_hard']
    fields=['initialization_sha256','final_sha256','batch_plan_sha256','steps','training_examples','effective_supervised_tokens']
    assert all(ce['training'][key] == hard['training'][key] for key in fields)
    for split in ('validation','test'):
        assert ce[split]['generated_samples'] == hard[split]['generated_samples']
        assert ce[split]['correct'] == hard[split]['correct']
        assert ce[split]['examples'] == hard[split]['examples']
    matched.append({'width':width,'matched_training_leaves':fields,'final_sha256':ce['training']['final_sha256'],'validation_examples':ce['validation']['examples'],'test_examples':ce['test']['examples']})
print('SAVED_MATCHED_TRAINING_AND_SAMPLES_LEAVES',json.dumps(matched))

raw_targets = OLD / 'sources/sft-hard-targets.json'
sft=load(raw_targets)
print('RAW_TARGET_JSON_TOP_KEYS_TYPES_ONLY',{k:type(v).__name__ for k,v in sft.items()})
assert sft['audit'] == raw['results']['tasks']['attributes']['hard_target_generation']['audit']
for record, row in zip(sft['records'],sft['audit'],strict=True):
    assert record['_hard_ids'] == row['teacher_ids'] and record['_hard_eos'] == row['eos']
print('SAVED_RAW_TARGET_RECORD_LEAVES', len(sft['records']),sha(raw_targets))

cpu_stdout=OLD/'stdout.txt'
cpu_env=OLD/'environment.json'
cpu_code=OLD/'verify.py'
cpu_command=OLD/'commands.json'
for path in [cpu_stdout,cpu_env,cpu_code,cpu_command]:
    item=next(a for a in prior['artifacts'] if a['path']==path.relative_to(ROOT).as_posix())
    assert sha(path)==item['sha256']
for line in cpu_stdout.read_text().splitlines():
    if line.startswith(('保留筆數','TOKEN_CONTRACT','CONTROLLED_GENERATION','MASK_AND_LOSS','ALL_ASSERTIONS_PASSED')):
        print('REUSED_PRIOR_ACTUAL_CPU_STDOUT',line)

environment={'python':platform.python_version(),'python_executable':sys.executable,
             'torch_installed_metadata':importlib.metadata.version('torch'),
             'current_execution':'Python file/AST/hash/raw-leaf inspection only; torch not imported',
             'device':'cpu metadata inspection; no model execution','network':'no network calls','cwd':str(ROOT)}
(HERE/'environment.json').write_text(json.dumps(environment,ensure_ascii=False,indent=2)+'\n')
receipt={
    'reviewer_task':TASK,'callback_kind':'same original reviewer narrow technical current callback',
    'inspected_on':'2026-10-06','current_primary':inputs,'current_code_inspection':current_code,
    'prior_fingerprint_checks':prior_artifacts,'original_authority_support_reinspection':authority,
    'raw_result_source':{'path':raw_path.relative_to(ROOT).as_posix(),'sha256':sha(raw_path),'pointers':[
        '/results/tasks/{attributes,style_transfer}/hard_target_generation/audit/*/{teacher_ids,gold_answer,teacher_answer,teacher_correct,eos,valid_target_tokens,invalid_special_tokens}',
        '/results/tasks/attributes/runs/w{16,32}_{ce,teacher_hard}/training/{initialization_sha256,final_sha256,batch_plan_sha256,steps,training_examples,effective_supervised_tokens}',
        '/results/tasks/attributes/runs/w{16,32}_{ce,teacher_hard}/{validation,test}/{generated_samples,correct,examples}'],
        'rechecked':raw_checks,'matched_training_leaves':matched,'policy':'Only named original measurements/targets/method leaves; no author extra results or scope/correction interpretations read.'},
    'raw_target_source':{'path':raw_targets.relative_to(ROOT).as_posix(),'sha256':sha(raw_targets),'pointers':['/records/*/_hard_ids','/records/*/_hard_eos','/audit']},
    'prior_cpu_reuse':{'code_path':cpu_code.relative_to(ROOT).as_posix(),'code_sha256':sha(cpu_code),'stdout_path':cpu_stdout.relative_to(ROOT).as_posix(),'stdout_sha256':sha(cpu_stdout),'command_path':cpu_command.relative_to(ROOT).as_posix(),'command_sha256':sha(cpu_command),'environment_path':cpu_env.relative_to(ROOT).as_posix(),'environment_sha256':sha(cpu_env),'actual_reuse_scope':'Original fence and one answer variation; controlled original hard-target transformation; answer-loss gradient mask. Exact fingerprints and output/support boundaries verified; not rerun in callback.'},
    'scientific_support_boundary':'All eight unchanged18.3 claim scopes retain their original first-hand authority/code/CPU/raw evidence. Current context prefixes still state same token mapping/SFT/mask contracts; changes elsewhere in context sections are outside this dependence. Original parameter SHA, not new weight load/training, supports CE/hard equality; style full target records remain not independently loaded, original embedded raw audit+manifest+writing method support retained.',
    'intro':None,'figures':{},'figure_applicability':'No image reference in current18.3; explicit question/answer/role/ID text suffices, no rendering or viewport inspection claimed.',
    'events':[],'new_substantive_issues':[],'new_model_cpu_execution':False,'network_fetches':0,
    'preservation_path':(HERE/'prior-preservation.json').relative_to(ROOT).as_posix(),
    'result':'Current primary unchanged, truly necessary context support compatible, code/authority/CPU/JSON fingerprints and support bounds checked; no unresolved substantive question.'}
(HERE/'current-inspection.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print('CURRENT_INSPECTION_RECEIPT', str(HERE/'current-inspection.json'), sha(HERE/'current-inspection.json'))
print('ALL_CURRENT_CALLBACK_CHECKS_PASSED; no source edits, model execution, network, downloads, training or rendering')
