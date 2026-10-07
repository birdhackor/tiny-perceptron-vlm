import hashlib,json,platform,tarfile,torch
from pathlib import Path
from scripts.course_experiments.text import _digest,_utf8_prefix
from scripts.course_experiments.common import split_records,records_sha256
from tiny_perceptron.data import render_chat,ByteTokenizer
B=Path('docs/technical-reviews/artifacts/p7_technical_b');arc=B/'sources/ultrachat-sft-original.tar.gz';assert hashlib.sha256(arc.read_bytes()).hexdigest()=='26ed20005d930234ebf905af6575a034ca032970041cc79f3afe40192b2fe319'
with tarfile.open(arc,'r:gz') as t:rawbytes=t.extractfile('behavior-initial/ultrachat-sft/train-first-100.jsonl').read()
src=B/'sources/ultrachat-original-first100.jsonl';assert not src.exists();src.write_bytes(rawbytes)
raw=[json.loads(s) for s in rawbytes.decode().splitlines()];prepared=[]
for i,r in enumerate(raw):
 user=next((m['content'] for m in r['messages'] if m['role']=='user'),None);assistant=next((m['content'] for m in r['messages'] if m['role']=='assistant'),None)
 if user and assistant:prepared.append({'family':r.get('prompt_id',_digest(user)),'source_row':i,'source_record_sha256':_digest(r),'scope':'first-turn UTF-8-safe excerpt','messages':[{'role':'user','content':_utf8_prefix(user,120)},{'role':'assistant','content':_utf8_prefix(assistant,120)}]})
parts=split_records(prepared,seed=42);u=json.loads((B/'sources/sft-result.json').read_text())['results']['ultrachat_pilot'];out={'python':platform.python_version(),'torch':torch.__version__,'device':'cpu','source_records':len(raw),'prepared_records':len(prepared),'steps':u['training']['steps'],'parts':{},'max_message_bytes':max(len(m['content'].encode()) for r in prepared for m in r['messages']),'max_X_length':max(len(render_chat(r['messages'])[0]) for r in prepared),'min_effective':min(int((render_chat(r['messages'])[1]!=-100).sum()) for r in prepared)}
assert len(raw)==len(prepared)==100 and out['max_message_bytes']<=120 and out['max_X_length']<=256 and out['min_effective']>0
families=[]
for name,rows in parts.items():
 payload=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows).encode();h=hashlib.sha256(payload).hexdigest();assert h==u['data'][name]['sha256'];families.append(set(r['family'] for r in rows));entry={'records':len(rows),'families':len(families[-1]),'sha256':h}
 if name!='train':
  z=u['evaluation'][name];assert len(rows)==z['records']==len(z['samples']);exact=0
  for r,s in zip(rows,z['samples'],strict=True):
   assert r['messages'][:-1]==s['messages'] and r['messages'][-1]['content']==s['expected'];ids=s['generated_ids'];rawids=ids[:ids.index(2)] if 2 in ids else ids;ok=rawids==ByteTokenizer().encode(s['expected']);assert ok==s['exact'];exact+=ok;assert ByteTokenizer().decode(rawids)==s['generated']
  assert exact==z['matches']==0;entry['exact_matches']=exact;entry['exact_denominator']=len(rows)
 out['parts'][name]=entry
assert all(not a.intersection(b) for i,a in enumerate(families) for b in families[i+1:]);assert records_sha256(parts['train'])==u['training']['records_sha256']
first=raw[0];out['first_original_user_prefix300']=next(m['content'] for m in first['messages'] if m['role']=='user')[:300];out['first_prepared_messages']=prepared[0]['messages']
print(json.dumps(out,ensure_ascii=False,indent=2))
