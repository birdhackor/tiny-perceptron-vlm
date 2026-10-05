from pathlib import Path
import json,hashlib,re,ast
R=Path.cwd();A=R/'docs/technical-reviews/artifacts/phase4-13_15-independent';meta=json.loads((R/(A/'context-reinspection/latest-start-path.txt').read_text().strip()).read_text());O=R/meta['reinspection_path'];prior=json.loads((R/meta['prior_report_history_path']).read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fences(raw):return re.findall(rb'(?ms)^```(?:python|python3|py)\s*\n(.*?)^```\s*$',raw)
observed={}
for sid in ['13.12','13.14','13.15']:
 now=(O/(sid+'-current.md')).read_bytes();old=(O/(sid+'-original-frozen.md')).read_bytes();nf,of=fences(now),fences(old);assert len(nf)==len(of)
 observed[sid]={'python_fence_count':len(nf),'same_executable_ast':all(ast.dump(ast.parse(n.decode()),include_attributes=False)==ast.dump(ast.parse(o.decode()),include_attributes=False) for n,o in zip(nf,of)),'current_fence_sha256':[hashlib.sha256(n).hexdigest() for n in nf],'prior_fence_sha256':[hashlib.sha256(o).hexdigest() for o in of]}
 assert observed[sid]['same_executable_ast']
assert (O/'13.15-current.md').read_bytes()==(O/'13.15-original-frozen.md').read_bytes()
assert (O/'13.12-current.md').read_bytes()==(O/'13.12-original-frozen.md').read_bytes().replace('收集這批迴答時'.encode(),'收集這批回答時'.encode())
old=(O/'13.14-original-frozen.md').read_bytes();now=(O/'13.14-current.md').read_bytes();oldline='print("估計員代價與梯度", round(value_loss.item(), 4), round(value.grad.item(), 4))\n\n```'.encode();newline=oldline.replace(b'\n\n```',b'\n```');assert now==old.replace(oldline,newline,1)
repo_sources=[]
for s in prior['sources']:
 if s['kind']=='repository_code':
  actual=sha(R/s['path']);assert actual==s['sha256'];repo_sources.append({'id':s['id'],'path':s['path'],'previously_verified_sha256':s['sha256'],'current_sha256':actual,'byte_equal':True})
source_snapshots=[]
for s in prior['sources']:
 if s.get('snapshot_path'):
  actual=sha(R/s['snapshot_path']);assert actual==s['snapshot_sha256'];source_snapshots.append({'id':s['id'],'path':s['snapshot_path'],'sha256':actual,'recorded_version':s['version'],'reuse':'Original immutable source snapshot and exact prior personally read support retained; unchanged context does not require refetch.'})
figures=[]
for n in ['rewrite-13-clip-cases.svg','rewrite-13-model-roles.svg']:
 current=R/'course/figures'/n;original=A/'sources'/n;assert current.read_bytes()==original.read_bytes();figures.append({'path':str(current.relative_to(R)),'current_sha256':sha(current),'prior_snapshot_sha256':sha(original),'byte_equal':True,'reuse':'Same actually viewed prior raster and original SVG; labels/data flow unchanged.'})
artifacts=[]
for a in prior['artifacts']:
 actual=sha(R/a['path']);assert actual==a['sha256'];artifacts.append({'id':a['id'],'path':a['path'],'sha256':actual,'unchanged':True})
result={'reviewer_task':prior['reviewer_task'],'actual_new_read_scope':['Current entire 13.15 at course/chapters/13.md line493 through section end','Current entire necessary 13.14 at line443 through section end','Current entire necessary 13.12 at line373 through section end','Own original frozen counterparts and their actual unified diffs; no other reviewer report or author work note read'],'hash_only_scope':['13.10','13.11','13.13'],'own_body_byte_equal':True,'semantic_changes':[{'section':'13.12','actual_change':'迴答 → 回答 in one old-policy explanatory sentence','effect':'Spelling only. Ratio definition, historical probability storage, old/reference distinction and executable example unchanged.'},{'section':'13.14','actual_change':'One blank line removed immediately before the closing marker of the first Python fence','effect':'No Python AST or runtime change. Critic/RM distinction, value MSE, old/reference roles and exact KL explanation unchanged.'}],'fence_checks':observed,'repository_sources':repo_sources,'immutable_source_snapshots':source_snapshots,'prerequisite_figures':figures,'prior_artifact_integrity':artifacts,'reused_claim_ids':[c['id'] for c in prior['claims']],'reuse_support':'All 14 own claims and their precise source support remain applicable: own body is byte-identical, changed prerequisite sentences retain the same semantics, every relevant implementation/source snapshot/SVG and prior evidence artifact has its recorded SHA. No new CPU run, primary-source download, training or checkpoint evaluation is needed for these two edits.','open_questions':[],'provisional_verdict':'pass'}
(O/'context-check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['actual_new_read_scope','hash_only_scope','own_body_byte_equal','semantic_changes','fence_checks','repository_sources','prerequisite_figures','reused_claim_ids','open_questions']},ensure_ascii=False,indent=2))
