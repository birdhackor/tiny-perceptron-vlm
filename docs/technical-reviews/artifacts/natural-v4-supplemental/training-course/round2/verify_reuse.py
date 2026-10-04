from pathlib import Path
import json,hashlib,sys,re
import torch
ROOT=Path(__file__).resolve().parents[6];D=Path(__file__).parent;P=D.parent;R=json.loads((P/'report.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
receipts={'scope':'Actual raw-byte verification before reuse, not rerun GPUtraining or591/2639 CPU calculations','environment':{'python':sys.version.split()[0],'torch':torch.__version__,'torch_git_version':torch.version.git_version,'device':'cpu','cuda_available':str(torch.cuda.is_available())},'sources':[],'artifacts':[],'external_originals':[],'prerequisite_sections':[],'figures':[]}
for s in R['sources']:
 if s['kind']=='repository_code':receipts['sources'].append({'id':s['id'],'path':s['path'],'first_sha256':s['sha256'],'current_sha256':sha(ROOT/s['path']),'byte_identical':sha(ROOT/s['path'])==s['sha256']})
 elif 'retrieval_path_ignored' in s:receipts['external_originals'].append({'id':s['id'],'path':s['retrieval_path_ignored'],'first_sha256':s['retrieval_sha256'],'current_sha256':sha(ROOT/s['retrieval_path_ignored']),'byte_identical':sha(ROOT/s['retrieval_path_ignored'])==s['retrieval_sha256']})
for a in R['artifacts']:receipts['artifacts'].append({'id':a['id'],'path':a['path'],'first_sha256':a['sha256'],'current_sha256':sha(ROOT/a['path']),'byte_identical':sha(ROOT/a['path'])==a['sha256']})
pr=json.loads((P/'prerequisite-receipts.json').read_text())
for s in pr['sections']:
 path=ROOT/s['path'];raw=path.read_text();heads=list(re.finditer(r'^## .+$',raw,re.M));match=next(i for i,h in enumerate(heads) if h[0].startswith('## '+s['section']+' '));h=heads[match];end=heads[match+1].start() if match+1<len(heads) else len(raw);section=raw[h.start():end].encode();current=hashlib.sha256(section).hexdigest()
 receipts['prerequisite_sections'].append({**s,'current_fullfile_sha256':sha(path),'current_section_sha256':current,'section_byte_identical':current==s['section_sha256'],'inspection':'Actual necessary section previously personally read; current sectionbytes identical. Original snapshot/sourcehash kept if unrelated fullfile changed.'})
for f,h in R['figure_sha256'].items():receipts['figures'].append({'path':f,'first_sha256':h,'current_sha256':sha(ROOT/f),'byte_identical':sha(ROOT/f)==h})
receipts['guide_and_contract']={p:sha(ROOT/p) for p in ['docs/technical-review-guide.md','outputs/natural-v4/review-plan/supplemental-factual-contract.md','scripts/check_technical_reviews.py']}
precision=json.loads((ROOT/'docs/course-experiments/results/precision.json').read_text())['results'];receipts['corrected_own_precision_mapping']={k:{'matches':v['heldout']['test']['matches'],'records':v['heldout']['test']['records'],'weights_dtype':v['weights_dtype'],'updates':v['training']['optimizer_updates'],'skips':v['training']['skipped_updates']} for k,v in precision['variants'].items()}
(D/'reuse-verification.json').write_text(json.dumps(receipts,ensure_ascii=False,indent=2)+'\n');fail=[x for key in ['sources','artifacts','external_originals','figures'] for x in receipts[key] if not x['byte_identical']];sectionfail=[x for x in receipts['prerequisite_sections'] if not x['section_byte_identical']]
print(json.dumps({'counts':{k:len(receipts[k]) for k in ['sources','artifacts','external_originals','prerequisite_sections','figures']},'failures':fail,'changed_necessary_sections':sectionfail,'environment':receipts['environment'],'precision_actual':receipts['corrected_own_precision_mapping']},indent=2))
