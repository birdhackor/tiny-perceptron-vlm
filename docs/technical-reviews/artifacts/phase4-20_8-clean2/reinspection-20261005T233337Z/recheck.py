"""Same-owner bounded verification of the new 20.8 threshold clarifications."""
import ast
from collections import Counter
from datetime import UTC, datetime
import hashlib
import json
import platform
import re
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
PRIOR=HERE.parent
TASK='/root/phase4_factual_coordinator/factual_20_8_clean2'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
prior=read(HERE/'prior-report.opaque.json')
assert prior['reviewer_task']==TASK
assert sha(HERE/'prior-report.opaque.json')=='ce6ec0dd316556a76260e37a2a187eb14b1b223408697646eb2fe952137969b3'
# Fingerprint unchanged original sources and all of my retained proof; no summaries from other owners.
fingerprints=[]
for source in prior['sources']:
    if source['kind']=='repository_code':
        retained=ROOT/source['path']
        original=ROOT/str(retained).split('/inputs/',1)[1]
        assert sha(original)==sha(retained)==source['sha256']
        fingerprints.append({'source_id':source['id'],'original':str(original.relative_to(ROOT)),
            'retained':source['path'],'sha256':source['sha256'],'exact_original_and_retained_match':True})
for artifact in prior['artifacts']:
    assert sha(ROOT/artifact['path'])==artifact['sha256']
for source in prior['sources']:
    if source['kind'] in {'paper','official_docs','official_source'}:
        fingerprints.append({'source_id':source['id'],'version':source['version'],
            'url':source['url'],'reuse':'My prior inspected original authority, retained artifact hashes reverified; no refetch or new inference.'})
namespace={'Counter':Counter}
raw=(ROOT/'scripts/score_natural_v4_validation.py').read_text()
tree=ast.parse(raw)
locators=[]
for n in tree.body:
    if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {'WEIGHTS','DENOMINATOR','MANUAL'} for t in n.targets):
        exec(compile(ast.Module(body=[n],type_ignores=[]),'original-constants','exec'),namespace)
    if isinstance(n,ast.FunctionDef) and n.name in {'require','case_id','expected_cases','eos_complete'}:
        exec(compile(ast.Module(body=[n],type_ignores=[]),'original-scorer','exec'),namespace)
        locators.append({'function':n.name,'line_start':n.lineno,'line_end':n.end_lineno})
m=read(ROOT/'docs/natural-assistant/v4/manifest.json')
p=read(ROOT/'docs/natural-assistant/v4/validation-protocol-lower-lr.json')
cases,audio=namespace['expected_cases'](m,p)
groups=Counter(c['group'] for c in cases.values())
assert len(cases)==sum(groups.values())==p['validation_denominators']['lm_generations_per_candidate']==132
assert p['adapter_nonregression_gates_vs_base']['all_generations_complete'] is True
for group in ['voice_typed_reference_chat','voice_actual_asr_chat']:
    assert p['adapter_nonregression_gates_vs_base'][group+'_correct_count_minimum']=='base'
assert len(namespace['WEIGHTS'])==8
shown={'photo_summary','photo_fact','voice_typed_reference_chat','voice_actual_asr_chat'}
assert shown < set(namespace['WEIGHTS'])
assert set(namespace['WEIGHTS'])-shown=={'text_presence','single_ocr','ordered_ocr','text_chat'}
assert sum(groups[g]*w for g,w in namespace['WEIGHTS'].items())==75600
prior_verified=read(PRIOR/'verification.json')
val=ROOT/'outputs/natural-v4/modal-runs/validation-37219466611/natural-natural-v4-validation-37219466611-1/review'
grades=read(ROOT/'docs/natural-assistant/evidence/v4-runtime/validation-37219466611/blind-review/combined/grades.json')['grades']
aliases={'base':'B','adapter-step-001039':'A','adapter-step-002077':'C'}
grade_index={(g['case_id'],g['candidate']):g for g in grades}
raw_results={}
for variant,alias in aliases.items():
    rows=read(val/('generations-'+variant+'.json'))
    assert len(rows)==132
    complete=Counter()
    voice_counts=Counter()
    voice_pointers=[]
    for i,row in enumerate(rows):
        key=namespace['case_id'](row['id'],row['task'])
        assert key in cases
        ok=namespace['eos_complete'](row,384)
        complete['complete' if ok else 'incomplete']+=1
        group=cases[key]['group']
        if group in {'voice_typed_reference_chat','voice_actual_asr_chat'}:
            voice_counts[group]+=int(ok and grade_index[(key,alias)]['passed'])
            voice_pointers.append({'pointer':f'/{i}','id':row['id'],'task':row['task'],
                'raw_token_count':len(row['generated_token_ids']),'eos':ok,
                'grade_passed':grade_index[(key,alias)]['passed']})
    assert complete['incomplete']==len(prior_verified['variants'][variant]['incomplete_case_ids'])
    assert all(voice_counts[g]==prior_verified['variants'][variant]['correct_counts'][g] for g in voice_counts)
    raw_results[variant]={'generations':len(rows),'completion_counts':dict(complete),'voice_correct_counts':dict(voice_counts),
        'voice_raw_pointers':voice_pointers}
assert raw_results['base']['voice_correct_counts']=={'voice_actual_asr_chat':3,'voice_typed_reference_chat':4}
# Only verify the new navigation target heading; TRAINING body/result explanations remain unread.
training=(ROOT/'docs/natural-assistant/v4/TRAINING.md').read_bytes()
heading=next(line for line in training.splitlines() if line.startswith('## 6. '.encode()))
assert heading.decode()=='## 6. 先驗證用途，再選定自己的版本'
source=(HERE/'section-current.md').read_bytes()
assert hashlib.sha256(source).hexdigest()=='e09afa95aed945142627309f2f7b99349b285ee3fc1818ea0930782dc688e7e3'
assert b'```' not in source and b'![' not in source
assert '全部132份驗證回答以EOS結束' in source.decode()
assert '各自都不能比底座少答對' in source.decode()
assert '並非全部選版得分' in source.decode()
result={'actual_reviewer_task':TASK,'executed_utc':datetime.now(UTC).isoformat(),
    'environment':{'python':platform.python_version(),'device':'CPU; standard-library AST/JSON audit only'},
    'current_source_sha256':hashlib.sha256(source).hexdigest(),
    'prior_source_sha256':prior['source_sha256'],'prior_report_sha256':sha(HERE/'prior-report.opaque.json'),
    'changed_claims':['acceptance-rules','macro-score'],
    'precise_reinspection_scope':{'source':'Whole current 20.8 including table/details; necessary context only paragraph2. Not a first subsection: chapter introduction not required or read.',
        'protocol_pointers':['/validation_denominators','/adapter_nonregression_gates_vs_base','/score','/selection','/declared_utc'],
        'raw_generation_pointers':'All396 rows only identity/task/generated_token_ids/count/EOS/stop/truncated/unknown leaves; each variant /124-/131 voice ids plus original grade passed labels. Predictions and gold semantics unchanged, supported by my prior actual visual/semantic inspection.',
        'original_code_locators':locators+[{'function':'score_blind acceptance and macro branches','line_start':393,'line_end':441}],
        'training_reference':'Only exact heading L155 inspected; no body read or used as scientific evidence.'},
    'unchanged_source_fingerprints':fingerprints,
    'prior_proof_artifacts_verified':len(prior['artifacts']),
    'current_group_denominators':dict(groups),'raw_results':raw_results,
    'all_EOS_rule':{'adapter_only':True,'required_responses_per_candidate':132,'eos_contract':'Original eos_complete requires final raw token in EOS IDs, consistent count/flags, no truncation or unknown completion. EOS does not imply semantic truth.'},
    'voice_nonregression_minimums':{'voice_typed_reference_chat':4,'voice_actual_asr_chat':3},
    'displayed_groups':sorted(shown),'additional_scored_groups':sorted(set(namespace['WEIGHTS'])-shown),
    'macro_denominator':75600,'prior_exact_primary_numerators':{k:x['primary_numerator'] for k,x in prior_verified['variants'].items()},
    'intro':{'status':'not_applicable','sha256':None,'reason':'20.8 is not the first subsection; added gate clarification does not depend on an introduction.'},
    'figures':{},'fences':{'python':0,'bash':0,'reason':'No fences or source figure references in the full current subsection.'},
    'verdict':'pass','unresolved_issues':[],'reuse_limit':'Same original owner reuses independently inspected, hash-identical original evidence. No full pipeline, model generation, training, weights/model download or other reviewers content.'}
(HERE/'recheck-result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['actual_reviewer_task','environment','current_source_sha256','prior_report_sha256','changed_claims','current_group_denominators','all_EOS_rule','voice_nonregression_minimums','additional_scored_groups','macro_denominator','prior_proof_artifacts_verified','verdict','unresolved_issues']},ensure_ascii=False,indent=2))
