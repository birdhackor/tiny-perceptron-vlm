import json,sys
from pathlib import Path
m=json.loads(Path('docs/natural-assistant/v4/manifest.json').read_text());root=Path('docs/natural-assistant/evidence/v4-runtime/validation-37219466611/blind-review/raw');gens={v:{r['id']:r for r in json.loads((root/f'generations-{v}.json').read_text())} for v in ['base','adapter-step-001039','adapter-step-002077']}
for id in sys.argv[1:]:
 print('\nPHOTO',id)
 for r in m['rows']:
  if r['split']=='validation' and r['id'].startswith('vision-v4:docci/'+id+'/'):
   print(r['id'].split('/')[-1],r['user'],'target',r['answer'])
   for v,g in gens.items():q=g[r['id']];print(v,'complete',q['stop_reason'],q['prediction'])
