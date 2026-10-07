"""Own read-only audit of original eight LFS payloads, not author explanations."""
import hashlib,io,json,tarfile,wave
from pathlib import Path
import pyarrow.parquet as pq
manifest=json.loads(Path('assets/training/manifest.json').read_text())
out=[]
for a in manifest['assets']:
 h=a['archive_sha256'];p=Path('/workspace/tiny-perceptron-vlm/.git/lfs/objects')/h[:2]/h[2:4]/h
 b=p.read_bytes();assert len(b)==a['archive_bytes'] and hashlib.sha256(b).hexdigest()==h
 rows={};samples={};rates={};parquet={};original_excerpts={}
 expected={f['path']:f for f in a['files']};seen=set()
 with tarfile.open(fileobj=io.BytesIO(b),mode='r:gz') as t:
  for m in t:
   assert m.isfile() and m.name in expected and m.name not in seen
   v=t.extractfile(m).read();e=expected[m.name];assert len(v)==e['bytes'] and hashlib.sha256(v).hexdigest()==e['sha256'];seen.add(m.name)
   if m.name.endswith('.jsonl'):
    ds=[json.loads(s) for s in v.splitlines() if s.strip()];rows[m.name]=len(ds)
    if ds:samples[m.name]={'fields':list(ds[0]),'last_fields':list(ds[-1])}
   if m.name.endswith('.wav'):
    with wave.open(io.BytesIO(v),'rb') as w:rates[m.name]={'rate':w.getframerate(),'channels':w.getnchannels(),'frames':w.getnframes()}
   if m.name.endswith('.parquet'):parquet[m.name]=pq.ParquetFile(io.BytesIO(v)).metadata.num_rows
   # Only publisher originals, excluding local README/metadata/attribution explanations.
   publisher=(m.name.endswith('source-README.md') or m.name.endswith('tinystories-source-README.md') or m.name.endswith('upstream/README.md') or m.name.endswith('upstream-README.md') or m.name.endswith('hf-dataset-README.md') or m.name.endswith('fashion-mnist-hf-README.md') or m.name.endswith('chinese-poetry-LICENSE') or m.name.endswith('upstream-LICENSE') or m.name.endswith('MIT-LICENSE.txt'))
   if publisher:
    ss=v.decode('utf8');lines=ss.splitlines();terms=['license:','License','license','8kHz','8 kHz','8kHz','sampling','human','humans','crowd','60000','60,000']
    excerpt=[{'line':i+1,'text':s[:420]} for i,s in enumerate(lines) if any(k in s for k in terms)]
    if not excerpt and 'LICENSE' in m.name:excerpt=[{'line':i+1,'text':s[:420]} for i,s in enumerate(lines[:12])]
    original_excerpts[m.name]={'sha256':hashlib.sha256(v).hexdigest(),'necessary_excerpt':excerpt[:10]}
 assert seen==set(expected)
 out.append({'id':a['id'],'archive_bytes':len(b),'archive_sha256':h,'all_member_hashes_verified':len(seen),'jsonl_counts':rows,'record_fields':samples,'wav_count':len(rates),'wav_rates':sorted({v['rate'] for v in rates.values()}),'parquet_rows':parquet,'publisher_original_excerpts':original_excerpts})
print(json.dumps({'archives':out,'compressed_total_bytes':sum(a['archive_bytes'] for a in out),'MiB':sum(a['archive_bytes'] for a in out)/2**20,'declared_training_total':sum(a['training_records'] for a in manifest['assets'])},ensure_ascii=False))
