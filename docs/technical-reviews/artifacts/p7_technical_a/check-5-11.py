import json,sys,tarfile,hashlib
from pathlib import Path
from tiny_perceptron.data import fingerprint
from scripts.course_experiments.text import _deduplicate_text
b=Path('docs/technical-reviews/artifacts/p7_technical_a')
raw=json.load(open('docs/course-experiments/results/real_text.json'))
checks={}
for identifier,file in [('tinystories','tinystories-train-512.jsonl'),('chinese-poetry','chinese-classical-train-365.jsonl')]:
 asset=next(a for a in raw['assets'] if a['identifier']==identifier) if any('identifier' in a for a in raw['assets']) else next(a for a in raw['assets'] if any(x['path'].endswith(file) for x in a['files']))
 archive=Path('outputs/phase7-source-cache/p7_technical_a')/(identifier+'-v1.tar.gz')
 assert hashlib.sha256(archive.read_bytes()).hexdigest()==asset['archive_sha256']
 with tarfile.open(archive) as tf:
  data=tf.extractfile(next(x for x in tf.getnames() if x.endswith(file))).read()
 assert hashlib.sha256(data).hexdigest()==next(x['sha256'] for x in asset['files'] if x['path'].endswith(file))
 rows=[json.loads(x) for x in data.splitlines()]
 dedup=_deduplicate_text(rows)
 assert len(dedup)==len(rows)
 assert [r['text'] for r in dedup]==[r['text'] for r in rows]
 assert all(r['family']==fingerprint(r['text']) for r in dedup)
 saved=raw['results']['runs'][identifier]
 assert len(rows)==saved['source_records']==saved['deduplicated_records']
 checks[identifier]={'records':len(rows),'distinct_normalized':len(set(' '.join(r['text'].split()) for r in rows)),'distinct_full_hashes':len(set(r['family'] for r in dedup)),'raw_sha256':hashlib.sha256(data).hexdigest(),'original_text_preserved':True,'family_full_content_hash':True}
docs=['貓  看狗','鳥 看狗','狗 看貓'];hs=list(map(fingerprint,docs))
assert len(set(hs))==3 and hs[0]!=hs[1]
try:assert hs[0]==hs[1]
except AssertionError:expected_error='AssertionError'
assert fingerprint(' \t貓\n 看狗  ')==fingerprint('貓 看狗')
seen={};collision=None
for i in range(1000):
 h=fingerprint(str(i))[:1]
 if h in seen:collision=[seen[h],str(i),h];break
 seen[h]=str(i)
assert collision and fingerprint(collision[0])!=fingerprint(collision[1])
out={'command':'PYTHONPATH=. .venv/bin/python docs/technical-reviews/artifacts/p7_technical_a/check-5-11.py','python':sys.version,'device':'CPU no model training','variation_distinct':len(set(hs)),'old_assert':expected_error,'corrected_assert':True,'leading_trailing_tab_newline_normalization':True,'one_hex_prefix_collision':collision,'full_hash_different':True,'datasets':checks,'not_checked':'Near-duplicate semantic clusters not audited; historical training not rerun','source_failure':'NIST FIPS180-4 PDF HTTP403; used actual original IETF RFC6234 section1 and official Python docs instead. Broad rg source output truncated, only targeted subsequent ranges count as read.'}
p=b/'5-11-original-checks.json';assert not p.exists();p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(out,ensure_ascii=False,indent=2))
