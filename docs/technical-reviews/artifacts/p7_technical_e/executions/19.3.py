import json,hashlib,itertools
from pathlib import Path
from collections import Counter,defaultdict
m=json.loads(Path('docs/selftrained/v2-manifest.json').read_text());base=Path('outputs/selftrained-v2/data');by=defaultdict(list)
for e in m['records']:
 p=base/e['path'];raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==e['sha256'];rows=[json.loads(l) for l in raw.splitlines()];print(e['path'],'count',len(rows),'sha256_verified',e['sha256']);
 for r in rows:assert e['path'].endswith(r['split']+'.jsonl');by[r['split']].append(r)
print('totals',{s:len(rs) for s,rs in by.items()})
def check(name,fn):
 sets={s:set().union(*(fn(r) for r in rs)) for s,rs in by.items()};ovs={a+'/'+b:len(sets[a]&sets[b]) for a,b in itertools.combinations(sets,2)};print(name,'unique',{s:len(v) for s,v in sets.items()},'overlaps',ovs);assert all(v==0 for v in ovs.values())
check('group_ids',lambda r:{r['group_id']});check('template_families',lambda r:{r['supervision']['template_family']} if 'template_family' in r.get('supervision',{}) else set());check('operand_families',lambda r:{r['supervision']['operand_family']} if 'operand_family' in r.get('supervision',{}) else set());check('vision_source_ids',lambda r:set(r.get('supervision',{}).get('source_ids',[])));check('ocr_render_sources',lambda r:{r['supervision']['render_source_id']} if 'render_source_id' in r.get('supervision',{}) else set());check('ocr_composition_strings',lambda r:{r['supervision']['ocr_text']} if r['task']=='ocr' and 2<=len(r['supervision']['ocr_text'])<=4 else set());check('audio_paths',lambda r:{r['audio']} if r.get('audio') else set())
