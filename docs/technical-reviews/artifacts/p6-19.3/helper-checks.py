from pathlib import Path
import ast, hashlib, json, sys
from collections import Counter
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from tiny_perceptron.selftrained.dataset import read_records
from scripts.selftrained.hf_transport import verify_file
ART=Path(__file__).resolve().parent
DATA=ROOT/'outputs/selftrained-v2/data'
manifest=json.loads((ROOT/'docs/selftrained/v2-manifest.json').read_bytes())
rows=read_records([DATA/e['path'] for e in manifest['records']])
tree=ast.parse((ROOT/'scripts/selftrained/train.py').read_text())
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='stage_records')
namespace={};exec(compile(ast.Module(body=[node],type_ignores=[]),'train.py:stage_records','exec'),namespace)
result={}
for stage in ['pretrain','sft','vision','ocr','audio','joint']:
 result[stage]={}
 for split in ['train','validation']:
  found=namespace['stage_records'](rows,stage,split)
  assert all(r['split']==split for r in found)
  result[stage][split]=len(found)
print('stage_records_original_function',json.dumps(result,sort_keys=True))
vision=[r for r in rows if r['task'].startswith('vision') and r['split']=='test']
count=sum(r['supervision']['vision_labels'][r['supervision']['query_slot']]==1 for r in vision)
print('constant_bag_category_guess',json.dumps({'correct':count,'denominator':len(vision),'ratio':count/len(vision),'model_evaluation':False,'scope':'category alone, not final-reply exact/position/format scoring'}))
damaged=ART/'manifest-negative.txt';damaged.write_bytes(b'original')
entry={'path':damaged.name,'bytes':8,'sha256':hashlib.sha256(b'original').hexdigest()}
verify_file(ART,entry);damaged.write_bytes(b'modified')
try:verify_file(ART,entry)
except ValueError as e:print('manifest_negative',str(e))
else:raise AssertionError('size-preserving change not rejected')
# Preserve the two original raw record lines and verify bytes against original files.
ids={'9a6a4e2b5cecc4f0b367ad06','c85d56336bfef394376d00f5'};raw=[];provenance=[]
for item in manifest['records']:
 if item['path'].startswith('ocr-'):
  path=DATA/item['path']
  for line_no,line in enumerate(path.read_bytes().splitlines(keepends=True),1):
   row=json.loads(line)
   if row['id'] in ids:
    raw.append(line);provenance.append({'source_record_file':item,'jsonl_line':line_no,'record_id':row['id'],'raw_line_sha256':hashlib.sha256(line).hexdigest()})
assert len(raw)==2
target=ART/'collision-original-records.jsonl';target.write_bytes(b''.join(raw))
assert target.read_bytes().splitlines(keepends=True)==raw
(ART/'collision-record-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
print('raw_collision_record_copy',json.dumps(provenance,sort_keys=True))
print('verification_complete',True)
