import hashlib,json,platform,torch
from pathlib import Path
from tiny_perceptron.data import ByteTokenizer
t=ByteTokenizer();p=Path('docs/course-experiments/results/sft.json');j=json.loads(p.read_text());r=j['results'];snapshot=Path('docs/technical-reviews/artifacts/p7_technical_b/sources/sft-result.json');assert not snapshot.exists();snapshot.write_bytes(p.read_bytes())
out={'python':platform.python_version(),'torch':torch.__version__,'device':'cpu','raw_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'historical_revision':j['revision'],'historical_device':j['device'],'steps':r['training']['steps'],'splits':{},'checked_code_hashes':{}}
assert out['steps']==900
for name in ['validation','test']:
 z=r['after'][name];rows=z['samples'];assert len(rows)==z['records']
 eos=sum(t.eos_id in a['generated_ids'] for a in rows);exact=0
 for a in rows:
  ids=a['generated_ids'];raw=ids[:ids.index(t.eos_id)] if t.eos_id in ids else ids
  assert t.decode(raw)==a['generated'] and (raw==t.encode(a['expected']))==a['exact'] and (t.eos_id in ids)==a['eos']
  exact+=a['exact']
 assert eos/len(rows)==z['eos_rate'] and exact==z['matches']
 out['splits'][name]={'records':len(rows),'EOS_count':eos,'EOS_rate':eos/len(rows),'content_exact_count':exact,'content_exact_rate':exact/len(rows),'first':rows[0]}
assert out['splits']['validation']['first']['generated_ids']==[107,113,122,109,2] and out['splits']['validation']['first']['generated']=='cire' and out['splits']['validation']['first']['expected']=='circle'
for name in ['scripts/course_experiments/text.py','scripts/course_experiments/common.py','tiny_perceptron/data.py','tiny_perceptron/model.py']:
 h=hashlib.sha256(Path(name).read_bytes()).hexdigest();assert h==j['code_sha256'][name];out['checked_code_hashes'][name]=h
out['ABCD_byte_ids']=t.encode('ABCD');out['ABCD_budget2_visible']=t.decode(t.encode('ABCD')[:2]);assert out['ABCD_budget2_visible']=='AB'
print(json.dumps(out,ensure_ascii=False,indent=2))
