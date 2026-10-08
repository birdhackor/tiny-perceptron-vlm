from pathlib import Path
import json,hashlib,sys,importlib.util
from datetime import datetime,timezone
B=Path(__file__).resolve().parent
M=json.loads((B/'manifest.json').read_text())
spec=importlib.util.spec_from_file_location('audreader',str(B/'reader.py'))
# Use the original unit splitter without invoking reader CLI.
spec=importlib.util.spec_from_file_location('audit_units','/workspace/tiny-perceptron-vlm/docs/review-tools/incremental_reader.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
import re
def main_text(pid):
 raw=(B/'freeze/sources'/f'{pid}.md').read_text()
 def hide(m):
  label=re.search(r'<summary[^>]*>(.*?)</summary>',m[0],re.S)
  return '\n[原位置選讀折疊區：'+(re.sub('<[^>]+>','',label[1]).strip() if label else '選讀')+'；正文路線完成後補讀]\n'
 return re.sub(r'<details\b[^>]*>.*?</details>',hide,raw,flags=re.S)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
errors=[];coverage={}
for section,folder in [('original_files_sha256','original'),('figures_sha256','original'),('criteria_sha256','criteria')]:
 for name,digest in M[section].items():
  if sha(B/'freeze'/folder/name)!=digest:errors.append('frozen input drift: '+name)
for p in M['inventory']['pages']:
 if sha(Path(p['snapshot']))!=p['source_sha256']:errors.append('canonical snapshot drift: '+p['page_id'])
for group,g in M['groups'].items():
 trace=B/'traces'/f'reader-{group}.jsonl'
 records=[json.loads(s) for s in trace.read_text().splitlines()] if trace.exists() else []
 cps=[r for r in records if r['kind']=='checkpoint' and r['phase']=='main']
 ends=[r['page_id'] for r in records if r['kind']=='page_end' and r['phase']=='main']
 expected=[(pid,i,hashlib.sha256(t.encode()).hexdigest()) for pid in g['pages'] for i,t in enumerate(mod.units(main_text(pid)) or [''])]
 actual=[(r['page_id'],r['unit_index'],r['unit_sha256']) for r in cps]
 if actual!=expected[:len(actual)]:errors.append(group+' has out-of-order/mismatched checkpoints')
 if any(r['note']['reviewer']!=g['reviewer'] for r in cps):errors.append(group+' reviewer identity mismatch')
 if ends!=g['pages'][:len(ends)]:errors.append(group+' page_end route mismatch')
 seals=[r for r in records if r['kind']=='sealed_main_report']
 for r in seals:
  if sha(Path(r['path']))!=r['sha256']:errors.append(group+' sealed main report drift')
 report=B/'reports'/f'reader-{group}-main.json'
 coverage[group]={'pages_completed':len(ends),'pages_assigned':len(g['pages']),'checkpoints_completed':len(cps),'checkpoints_expected':len(expected),'main_report_present':report.exists(),'main_report_sealed':bool(seals),'main_sha256':sha(report) if report.exists() else None}
complete=all(v['pages_completed']==v['pages_assigned'] and v['checkpoints_completed']==v['checkpoints_expected'] and v['main_report_present'] for v in coverage.values())
result={'at':datetime.now(timezone.utc).isoformat(),'errors':errors,'complete_main_coverage':complete,'groups':coverage,'meaning':'record/order/hash validation; not evidence of real human student testing or correctness by itself'}
(B/'checks/evidence-integrity.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
if errors:sys.exit(1)
