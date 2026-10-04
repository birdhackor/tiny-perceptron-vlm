from pathlib import Path
import collections,hashlib,json,sys
import torch
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[4];sys.path.insert(0,str(ROOT))
from tiny_perceptron.capstone import build_dataset,expected_final,parse_action,calculator_runtime,TOK
data,manifest=build_dataset(42)
print('dataset keys',list(data))
# Local saved original per-row generations. No model loading, inference, or training.
paths={
 'joint':'outputs/integration-runs/v2-deployment/capstone-review/capstone_deployment/gha-37169529991-1/test-joint.json',
 'ce':'outputs/integration-runs/v2-student/capstone-review/capstone_student/gha-37170200956-1/test-ce.json',
 'kd':'outputs/integration-runs/v2-student/capstone-review/capstone_student/gha-37170200956-1/test-kd.json',
 'kd-int4':'outputs/integration-runs/v2-student/capstone-review/capstone_student/gha-37170200956-1/test-kd-ptq4.json',
}
dataset_rows=data['test'] if 'test' in data else data['splits']['test']
gold={r['id']:r for r in dataset_rows}
result={'environment':{'python':sys.version,'torch':torch.__version__,'device':'cpu'},'limits':'Recomputed scoring from 360 existing saved GPU generation records; no GPU replication, model load, new inference, or optimizer update. Semantic grades in the natural validation record were not regraded.'}
result['capstone']={};sets=[]
for variant,path in paths.items():
 content=(ROOT/path).read_bytes();d=json.loads(content);rows=[];by=collections.defaultdict(lambda:{'count':0,'action_correct':0,'end_to_end_correct':0})
 report=json.loads((ROOT/('docs/course-experiments/results/capstone_deployment.json' if variant=='joint' else 'docs/course-experiments/results/capstone_student.json')).read_text())
 expected_hash=next(a['sha256'] for a in report['artifacts'] if a['path']==Path(path).name)
 assert hashlib.sha256(content).hexdigest()==expected_hash
 for record in d['records']:
  row=gold[record['id']];trace=record['action_trace']
  assert record['family']==row['family'] and record['expected_action']==row['answer'] and record['expected_final']==expected_final(row)
  for tr in [trace,record.get('final_trace')]:
   if tr:
    assert TOK.decode(tr['generated_ids'])==tr['raw']
    assert tr['eos']==bool(tr['generated_ids'] and tr['generated_ids'][-1]==TOK.eos_id)
  action=parse_action(trace);answer=None
  if action['status'] in ('direct','ask'):answer=action['content']
  elif action['status']=='tool':
   runtime=calculator_runtime(action,row['available']);assert runtime==record['runtime']
   final=record.get('final_trace')
   if final and parse_action(final)['status']=='direct':answer=parse_action(final)['content']
  ac=bool(trace['eos'] and trace['raw']==row['answer']);e2e=bool(ac and answer==expected_final(row))
  assert ac==record['action_correct'] and e2e==record['end_to_end_correct']
  b=by[row['task']];b['count']+=1;b['action_correct']+=ac;b['end_to_end_correct']+=e2e
  rows.append({'id':record['id'],'family':row['family'],'action_correct':ac,'end_to_end_correct':e2e})
 totals={'count':len(rows),'action_correct':sum(r['action_correct'] for r in rows),'end_to_end_correct':sum(r['end_to_end_correct'] for r in rows)}
 assert all(totals[k]==d[k] for k in totals) and dict(by)==d['by_task']
 assert set(r['id'] for r in rows)==set(gold)
 result['capstone'][variant]={'original_path':path,'sha256':expected_hash,'recomputed':totals,'by_task':dict(by),'own_row_recomputations':rows,'seed':report['seed'],'gpu':report['gpu'],'torch_version':report['torch_version'],'original_revision':report['revision'],'data_counts':report['results']['data_manifest']['counts']}
 sets.append([r['id'] for r in rows])
assert all(s==sets[0] for s in sets)
assert [result['capstone'][v]['recomputed']['end_to_end_correct'] for v in paths]==[78,62,61,61]
families={s:set(r['family'] for r in data[s]) for s in ['train','validation','test']}
assert all(not families[a]&families[b] for a,b in [('train','validation'),('train','test'),('validation','test')])
result['capstone_split']={s:{'rows':len(data[s]),'families':len(families[s])} for s in families}
# Independently recompute the natural validation selection arithmetic/gates from fixed score counts.
sp=ROOT/'docs/natural-assistant/evidence/v4-runtime/validation-37219466611/blind-review/scored/scores.json'
scores=json.loads(sp.read_text());protocol=json.loads((ROOT/'docs/natural-assistant/v4/validation-protocol.json').read_text())
assert sum(scores['denominators'][k]*w for k,w in scores['integer_weights'].items())==75600
variants={};chosen='base'
for name,v in scores['variants'].items():
 numerator=sum(v['correct_counts'][k]*w for k,w in scores['integer_weights'].items());assert numerator==v['primary_numerator']
 eligibility=True
 if name!='base':
  for group in scores['integer_weights']:
   rule=protocol['adapter_nonregression_gates_vs_base'][group+'_correct_count_minimum']
   minimum=max(0,scores['variants']['base']['correct_counts'][group]-int(rule=='base minus 1'))
   actual=v['correct_counts'][group];assert v['gates'][group]=={'minimum':minimum,'actual':actual,'passed':actual>=minimum}
   eligibility=eligibility and actual>=minimum
  eligibility=eligibility and not v['incomplete_case_ids'];assert eligibility==v['eligible']
  if eligibility and numerator>variants[chosen]['numerator']:chosen=name
 variants[name]={'numerator':numerator,'denominator':75600,'eligible':eligibility if name!='base' else 'fallback','incomplete_count':len(v['incomplete_case_ids'])}
assert chosen==scores['selected_variant']=='base'
selection=json.loads((ROOT/'docs/natural-assistant/v4/selection.json').read_text());release=json.loads((ROOT/'docs/natural-assistant/v4/public-release.json').read_text())
assert hashlib.sha256(sp.read_bytes()).hexdigest()==selection['validation_scoring_sha256']
assert selection['selected_variant']==release['selected_variant']==chosen and release['adapter_parameters']==0
result['natural_selection']={'source_path':str(sp.relative_to(ROOT)),'source_sha256':hashlib.sha256(sp.read_bytes()).hexdigest(),'variants_recomputed':variants,'chosen':chosen,'denominators':scores['denominators'],'semantic_limits':'Verified arithmetic and documented gates using fixed underlying semantic counts; no independent regrading of photographs/dialogue or rerun of generation.'}
ci=json.loads((ROOT/'docs/validation-artifacts/release-compatibility-ci.json').read_text())
result['ci_record']={'source_sha256':hashlib.sha256((ROOT/'docs/validation-artifacts/release-compatibility-ci.json').read_bytes()).hexdigest(),'run_id':ci['run_id'],'revision':ci['revision'],'scope':ci['scope'],'jobs':ci['jobs']}
remote=json.loads((ROOT/'outputs/lfs-remote-validation.json').read_text())
manifest=json.loads((ROOT/'assets/training/manifest.json').read_text());matched=0
for asset in manifest['assets']:
 for f in asset['files']:
  p=ROOT/'outputs/remote-training-assets-validation'/f['path']
  assert p.stat().st_size==f['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256'];matched+=1
result['original_anonymous_asset_record_audit']={'record':remote,'record_sha256':hashlib.sha256((ROOT/'outputs/lfs-remote-validation.json').read_bytes()).hexdigest(),'existing_remote_downloaded_file_entries_rehashed':matched,'scope':'Audited existing bytes and retained 2026-10-02 anonymous-download execution receipt. Did not independently repeat the network download.'}
(OUT/'empirical-record.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({v:r['recomputed'] for v,r in result['capstone'].items()}));print('natural selection',chosen,variants);print('existing remote asset entries',matched)
