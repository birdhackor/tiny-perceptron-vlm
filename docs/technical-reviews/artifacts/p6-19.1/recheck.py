"""Owner's bounded repaired-introduction verification; no model generation or training."""
from pathlib import Path
import hashlib,json,re,sys

root=Path('/workspace/selftrained-v2')
out=root/'docs/technical-reviews/artifacts/p6-19.1'
ev=out/'evidence/recheck'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
initial=json.loads((out/'initial-revise-report.json').read_text())
first=json.loads((ev/'recheck-results.json').read_text())
raw=(root/'course/chapters/19.md').read_bytes()
heads=list(re.finditer(rb'(?m)^## [^\r\n]+',raw))
intro=raw[:heads[0].start()];section=raw[heads[0].start():heads[1].start()]
assert intro==(ev/'chapter-intro-current.md').read_bytes()
assert section==(ev/'section-current.md').read_bytes()
assert hashlib.sha256(section).hexdigest()==initial['source_sha256']
assert (out/'inputs/chapter-intro.md').read_text().replace(first['only_intro_change']['before'],first['only_intro_change']['after']).encode()==intro
for s in initial['sources']:
    if s['kind']=='repository_code':assert sha(root/s['path'])==s['sha256']
for a in initial['artifacts']:assert sha(root/a['path'])==a['sha256']
for p,h in initial['figure_sha256'].items():assert sha(root/p)==h

chain=[]
for entry in first['directly_reread_original_provenance']:
    p=root/entry['path'];assert sha(p)==entry['sha256']
    data=json.loads(p.read_text());execution=json.loads(p.with_name('execution.json').read_text());outer=json.loads((p.parents[1]/'receipt.json').read_text())
    assert data['origin']['kind']=='all-neural-weights-random'
    assert execution['returncode']==0
    if data['stage']=='pretrain':assert '--init-checkpoint' not in execution['command'] and '--resume' not in execution['command']
    history=[{'stage':h['stage'],'step':h['step'],'checkpoint_sha256':h['checkpoint_sha256']} for h in data['stage_history']]
    assert history==entry['stage_history']
    best=next(x['sha256'] for x in outer['files'] if x['path']=='best.pt')
    assert best==entry['best_sha256'];chain.append({'path':entry['path'],'best':best,'history':history})
assert len(chain)==15
own_best={v['best'] for v in chain}
assert all(h['checkpoint_sha256'] in own_best for v in chain for h in v['history'])
manifest=json.loads((out/'inputs/inference-manifest.json').read_text())
assert manifest['selected_checkpoint_sha256']==next(v['best'] for v in chain if '/moe-native/' in v['path'])
assert manifest['origin']['kind']=='all-neural-weights-random'
model=Path('/tmp/p5-native-public-cpu-smoke-actual/public-model')
for p,h in manifest['files'].items():assert sha(model/p)==h
prior=json.loads((out/'evidence/verification.json').read_text())
assert all(v['all_ones'] and v['requires_grad'] for v in prior['initialization_literal_counterexample'].values())

result={'command_argv':sys.argv,'python':sys.version,'device':'CPU file/provenance verification only','scope':'Own repaired chapter intro plus unchanged 19.1 dependent sources; no reader/other technical report accessed','initial_report_sha256':sha(out/'initial-revise-report.json'),'initial_owner_evidence_sha256':sha(out/'evidence/verification.json'),'direct_owner_reread_record_sha256':sha(ev/'recheck-results.json'),'current_intro_sha256':hashlib.sha256(intro).hexdigest(),'current_section_sha256':hashlib.sha256(section).hexdigest(),'chain_stages_checked':len(chain),'initial_pretrain_external_checkpoint':False,'own_checkpoint_chain_closed':True,'public_selected_source_matches_own_chain':True,'original_norm_counterexample_retained':True,'source_and_artifact_hashes_unchanged':True,'generation_rerun':False,'result':'PASS: repaired no-pretrained-weights/from-scratch sentence is supported; fixed norm initial values are compatible.'}
(ev/'recheck-execution.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
