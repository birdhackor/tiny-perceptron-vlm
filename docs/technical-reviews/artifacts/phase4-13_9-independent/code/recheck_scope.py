"""Personally verify the revised 13.9 scope and every retained evidence version."""
import ast
import hashlib
import json
import platform
import re
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[5]
BASE=Path(__file__).resolve().parents[1]
DEST=BASE/'recheck'
DEST.mkdir(exist_ok=True)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sections(raw):
    headings=list(re.finditer(rb'(?m)^## [^\r\n]+',raw))
    return {h.group().decode().split()[1]:raw[h.start():headings[i+1].start() if i+1<len(headings) else len(raw)]
            for i,h in enumerate(headings)}


def select(document,pointer):
    value=document
    for part in pointer.strip('/').split('/'):
        value=value[int(part)] if isinstance(value,list) else value[part]
    return value


old=json.loads((BASE/'runs/initial-report-revise.json').read_text())
assert old['reviewer_task']=='/root/phase4_factual_coordinator/factual_13_9'
assert old['verdict']=='revise' and len(old['claims'])==16
assert sha(BASE/'runs/initial-report-revise.json')=='53c9e7f4eb157a3cc43043b9b14aa739831c21c0b9c44decb9e21a8fc1d1b493'
current=(ROOT/'course/chapters/13.md').read_bytes()
previous=(BASE/'inputs/chapter-13-frozen.md').read_bytes()
old_parts,new_parts=sections(previous),sections(current)
old_phrase='本章實跑只更新加法正誤偏好與附加句偏好'
new_phrase='本章的加法主線只更新加法正誤偏好與附加句偏好'
assert new_parts['13.9']==old_parts['13.9'].replace(old_phrase.encode(),new_phrase.encode())
unchanged=[key for key in old_parts if old_parts[key]==new_parts[key]]
assert len(unchanged)==16 and '13.9' not in unchanged
assert previous[:previous.index(b'## ')]==current[:current.index(b'## ')]
(DEST/'section-current.md').write_bytes(new_parts['13.9'])
(DEST/'chapter-current-frozen.md').write_bytes(current)
for name in ['extraction.json','fence-1.py','bootstrap.py']:
    shutil.copyfile(Path('/tmp/phase4-13_9-independent-recheck')/name,DEST/name)
extract=json.loads((DEST/'extraction.json').read_text())
assert extract['source_sha256']==sha(DEST/'section-current.md')
assert sha(DEST/'fence-1.py')==sha(BASE/'inputs/fence-1.py')
assert sha(DEST/'bootstrap.py')==sha(BASE/'inputs/bootstrap.py')
assert extract['svg_references']==[]

# Validate every previously cited evidence file, preserving the existing CPU measurements.
retained=[]
for item in old['artifacts']:
    actual=sha(ROOT/item['path'])
    assert actual==item['sha256'],item['id']
    retained.append({'artifact_id':item['id'],'path':item['path'],'sha256':actual})
for source in old['sources']:
    if source['kind']=='repository_code':assert sha(ROOT/source['path'])==source['sha256']
current_inputs={
 'docs/course-experiments/results/dpo.json':'inputs/dpo-original-results.json',
 'docs/course-experiments/results/style.json':'inputs/style-original-results.json',
 'tiny_perceptron/alignment.py':'inputs/tiny_perceptron__alignment.py',
 'tiny_perceptron/data.py':'inputs/tiny_perceptron__data.py',
 'tiny_perceptron/model.py':'inputs/tiny_perceptron__model.py',
 'scripts/course_experiments/common.py':'inputs/scripts__course_experiments__common.py',
 'scripts/course_experiments/text.py':'inputs/text-current.py',
 'scripts/course_experiments/behavior.py':'inputs/scripts__course_experiments__behavior.py',
 'data/training/behavior-initial/ultrafeedback-dpo/train-first-100.jsonl':'inputs/ultrafeedback-original-first100.jsonl',
 'data/training/behavior-initial/ultrafeedback-dpo/source-api.json':'inputs/ultrafeedback-original-api.json',
 'data/training/behavior-initial/ultrafeedback-dpo/source-README.md':'inputs/ultrafeedback-original-README.md',
}
for live,frozen in current_inputs.items():assert sha(ROOT/live)==sha(BASE/frozen),live

implementation=BASE/'inputs/behavior-revision-8a757184.py'
raw=implementation.read_bytes()
tree=ast.parse(raw)
nodes={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
lines=raw.decode().splitlines()
# These are the personally reread original computational ranges, not the extra author result strings.
read_ranges=[(537,549),(655,691),(697,744)]
reads=[]
for start,end in read_ranges:
    reads.append(f'Original behavior8a757184 lines{start}-{end}\n'+'\n'.join(lines[start-1:end]))
(DEST/'method-read.txt').write_text('\n\n'.join(reads)+'\n')
run_dpo=nodes['run_dpo']
assert any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='_natural_dpo_pilot' for n in ast.walk(run_dpo))
preference=nodes['_preference_parts']
rejected_value=next(v for node in ast.walk(preference) if isinstance(node,ast.Dict)
                    for k,v in zip(node.keys,node.values)
                    if isinstance(k,ast.Constant) and k.value=='rejected')
assert ast.dump(rejected_value)==ast.dump(ast.parse("str(row['a'] + row['b'] + 1)",mode='eval').body)
assert any(isinstance(n,ast.Constant) and n.value=='; answer complete' for n in ast.walk(run_dpo))

results=json.loads((BASE/'inputs/dpo-original-results.json').read_text())
assert sha(implementation)==results['code_sha256']['scripts/course_experiments/behavior.py']
raw_expectations={
 '/results/data/train/records':49,
 '/results/data/test/records':7,
 '/results/runs/model/beta':0.1,
 '/results/runs/model/training/steps':250,
 '/results/runs/beta1/beta':1.0,
 '/results/runs/beta1/training/steps':250,
 '/results/format_only/data/train/records':49,
 '/results/format_only/training/steps':200,
 '/results/ultrafeedback_pilot/source_records':100,
 '/results/ultrafeedback_pilot/sft_training/steps':80,
 '/results/ultrafeedback_pilot/training/steps':80,
}
measurements={pointer:select(results,pointer) for pointer in raw_expectations}
assert measurements==raw_expectations
(DEST/'raw-pointers.json').write_text(json.dumps(measurements,ensure_ascii=False,indent=2)+'\n')

# Every claim was reread against the complete revised section and its own original evidence scope.
support_review={
 'C1':'Unchanged hypothetical risk paragraph. Original InstructGPT regression evidence, DPO shared policy and one-scalar feasibility example still support possibility only, not a measured refusal rate.',
 'C2':'Unchanged plan/measurement distinction and held-out comparison guidance. Original papers separate preference, safety and traditional task evaluations; no new measurement implied.',
 'C3':'Original fence bytes identical, CPU execution artifact hash intact. Dictionary100/sixNone outputs and None→0 variant remain the exact displayed behavior; no model training/evaluation added.',
 'C4':'Unchanged30+30+20+20 and exercise20 transfer; original exact integer100/40/0 calculation and all metricsNone are retained.',
 'C5':'Unchanged30pairs,100/10 positions per side; original mask/EOS contract and executed6000/600 reconstruction retained. Exposure count still not a gradient magnitude claim.',
 'C6':'Unchanged20% data proportion warning. DPO Eq7 conditional policy expectation still supplies the distinction; no probability assumption introduced.',
 'C7':'Unchanged unmeasured safety/anti-sycophancy warning and exercise risk. Existing source scope still requires task-specific evaluation.',
 'C8':'Changed scope sentence personally reread in full section: now explicitly「本章的加法主線」. Original _preference_parts builds correct vs+1 answers; run_dpo builds appended-sentence control. The natural-data model is a separate branch; its100rows/80SFT/80DPO pointers personally rechecked. The arithmetic-only wording now agrees with these original methods.',
 'C9':'Unchanged no full style/safety/multiturn/anti-sycophancy remeasurement clause; original contract covers pair margins/arithmetic generation and no extra battery. Scope remains this chapter record only.',
 'C10':'Unchanged baseline and both correctness-DPO runs0/7 clause. Original style/content dependency and matching7 held-out prompts/raw-token checks remain version-identical. No checkpoint was reloaded.',
 'C11':'Unchanged format7/7relative movement and0/7generation clause. Same seven raw margins and generated token IDs, original distinct criteria retained.',
 'C12':'Unchanged separate natural width32/layers1 model and80+80 recipe. Personally reread _natural_dpo_pilot method and80/80 raw pointers; old fresh37728-parameter architecture check retained, not training.',
 'C13':'Unchanged natural validation6/10improved and7/10ranked clause. All10 original score/mask/source pointer calculations remain hash-identical.',
 'C14':'Unchanged natural test8/10improved and4/10ranked clause. All10 original score/mask/source pointer calculations and local family split hashes remain hash-identical.',
 'C15':'Unchanged cropped-label and natural/safety capability caveat. Original official dataset card full-completion label contract, local120-byte crop and88/90answer truncation counts retained.',
 'C16':'Unchanged absence of natural free-generation quality evaluation and raw report fields. Personally reread natural method returning only candidate-pair evaluations; all original raw sample/length/generated_ids/split pointers remain version-identical.',
}
assert set(support_review)=={c['id'] for c in old['claims']}
coverage=[]
for c in old['claims']:
    coverage.append({'claim_id':c['id'],'new_source_sha256':extract['source_sha256'],
                     'reviewed_support':support_review[c['id']],
                     'source_ids':[e['source_id'] for e in c['evidence']],
                     'status':'verified'})
(DEST/'claim-support-review.json').write_text(json.dumps(coverage,ensure_ascii=False,indent=2)+'\n')
(DEST/'retained-evidence-hashes.json').write_text(json.dumps(retained,indent=2)+'\n')
receipt={
 'id':'scope-recheck-receipt','reviewer_task':old['reviewer_task'],'reviewed_on':'2026-10-05',
 'old_source_sha256':old['source_sha256'],'new_source_sha256':extract['source_sha256'],
 'original_issue_id':'I1','original_quote':old['issues'][0]['original_quote'],
 'new_quote':'本章的加法主線只更新加法正誤偏好與附加句偏好，沒有重新測整套風格、安全、多輪或反附和能力，因此那些欄位仍應留為未測。',
 'full_section_personally_reread':True,'new_section_snapshot':'recheck/section-current.md',
 'read_locator':{'source':'course/chapters/13.md#13.9','first_line':extract['section_first_line'],
                 'last_line':extract['section_first_line']+new_parts['13.9'].count(b'\n')-1,
                 'scope':'Complete revised section, including original fence, exercise and both details paragraphs'},
 'new_whole_chapter_frozen_input':{'path':'recheck/chapter-current-frozen.md','sha256':sha(DEST/'chapter-current-frozen.md'),
                                 'meaning':'Own snapshot at this recheck, not a lasting current-whole-chapter claim'},
 'own_byte_comparison':{'changed_sections':['13.9'],'unchanged_other_sections':unchanged,'introduction_unchanged':True,
                        'sole_change':old_phrase+' -> '+new_phrase,'fence_byte_identical':True,'svg_references':[]},
 'method_original_version':results['revision'],'method_original_sha256':sha(implementation),
 'personally_reread_method_ranges':read_ranges,'raw_measurement_pointers_file':'recheck/raw-pointers.json',
 'all_16_claims_support_review_file':'recheck/claim-support-review.json',
 'resolved':True,
 'resolution':'The explicitly qualified arithmetic-main-branch statement matches the controlled arithmetic and appended-sentence methods; the separately described natural-dataSFT/DPO model remains an additional branch. The missing full capability-evaluation caveat and every original result are retained.',
 'limitations':'No GPU, full recipe, model/data download or existing model remeasurement. Original unmodified CPU and primary-source evidence reused only after own hash and support checks. No figure or visual-layout claim needs a render; no page inspection is represented as performed.',
 'command':'.venv/bin/python docs/technical-reviews/artifacts/phase4-13_9-independent/code/recheck_scope.py',
 'environment':{'python':platform.python_version(),'device':'cpu','shell':'bash login:false'},
 'result':f'All assertions passed: exact scope-only byte change,16 other sections/introduction/fence unchanged,{len(retained)} retained artifact hashes valid, all16 claim scopes reviewed, original arithmetic/natural raw pointers and original method identity personally verified.',
}
(DEST/'scope-recheck-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(receipt,ensure_ascii=False,indent=2))
