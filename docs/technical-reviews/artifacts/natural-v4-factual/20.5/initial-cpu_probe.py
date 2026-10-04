"""Fresh 20.5 bounded CPU checks; no training, model downloads, or source writes."""
from pathlib import Path
from collections import Counter, defaultdict
from contextlib import redirect_stdout
import hashlib, io, json, re, sys, tempfile, importlib.util
import torch
ROOT=Path.cwd()
A=ROOT/'docs/technical-reviews/artifacts/natural-v4-factual/20.5'
RAW=ROOT/'outputs/natural-v4/factual-research/20.5'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def emit(name,value): print(name,json.dumps(value,ensure_ascii=False,sort_keys=True))
emit('environment',{'python':sys.version.split()[0],'torch':torch.__version__,'cuda_available':torch.cuda.is_available(),'device':'CPU','training_executed':False})
assert torch.__version__=='2.14.1+cpu' and not torch.cuda.is_available()
text=(A/'section-original.md').read_text()
code=re.search(r'```python\n(.*?)\n```',text,re.S).group(1)
(A/'lesson-example.py').write_text(code+'\n')
for name,program,expected in [
 ('original',code,"訓練家族 ['貓照片A']\n測試家族 ['招牌照片B']\n跨份重疊 []\n"),
 ('changed-split',code.replace('{"family": "貓照片A", "split": "train", "question": "哪隻貓抬起前腳？"}', '{"family": "貓照片A", "split": "test", "question": "哪隻貓抬起前腳？"}'),"訓練家族 ['貓照片A']\n測試家族 ['招牌照片B', '貓照片A']\n跨份重疊 ['貓照片A']\n")]:
 out=io.StringIO()
 with redirect_stdout(out): exec(compile(program,'20.5-'+name,'exec'),{})
 assert out.getvalue()==expected
 emit(name,{'expected':expected,'observed':out.getvalue(),'cards':4,'families':2})
# Independent count and disjointness computation, before project validators.
manifest_path=ROOT/'docs/natural-assistant/v4/manifest.json'
m=json.loads(manifest_path.read_text());rows=m['rows']+m['audio_rows']
by_family=defaultdict(set);by_asset=defaultdict(set)
for row in rows:
 by_family[row['family']].add(row['split'])
 for key in ['image','audio']:
  if row.get(key):by_asset[row[key]].add(row['split'])
assert all(len(v)==1 for v in by_family.values())
assert all(len(v)==1 for v in by_asset.values())
emit('manifest',{'sha256':sha(manifest_path),'question_rows':len(m['rows']),'audio_rows':len(m['audio_rows']),'row_splits':dict(Counter(r['split'] for r in m['rows'])),'audio_splits':dict(Counter(r['split'] for r in m['audio_rows'])),'families':len(by_family),'physical_asset_paths':len(by_asset),'cross_split_family_count':0,'cross_split_asset_path_count':0})
originals={r['example_id']:r for r in map(json.loads,(RAW/'docci-web-schema').read_text().splitlines())}
photos=[]
for name in ['vision-sources.json','vision-activity-sources.json','vision-activity-heldout-sources.json']:
 p=ROOT/'docs/natural-assistant/v4/data'/name;d=json.loads(p.read_text());photos.extend(d['rows'])
assert len({r['example_id'] for r in photos})==len(photos)
for r in photos:
 o=originals[r['example_id']]
 assert str(o['cluster_id'])==str(r['cluster_id'])
 assert r['family']=='vision:docci:cluster:'+str(o['cluster_id'])
 assert r['full_original_caption']==o['description']
family_sets={split:{r['family'] for r in photos if r['split']==split} for split in ['train','validation','test']}
assert all(not family_sets[a]&family_sets[b] for a,b in [('train','validation'),('train','test'),('validation','test')])
emit('docci-original-comparison',{'original_rows':len(originals),'selected_photos':len(photos),'photos_by_split':dict(Counter(r['split'] for r in photos)),'clusters_by_split':{k:len(v) for k,v in family_sets.items()},'original_cluster_and_caption_mismatches':0,'official_web_metadata_sha256':sha(RAW/'docci-web-schema')})
# Source-file fingerprints independently checked from declarations (renaming does not change digest).
source_splits=defaultdict(set)
for s in m['sources']:
 d=s['metadata']
 for r in d.get('rows',[]):
  if r.get('sha256'): source_splits[r['sha256']].add(r['split'])
 for r in d.get('sources',[]):
  if r.get('image_sha256'):source_splits[r['image_sha256']].add(r['split'])
 for r in d.get('audio_rows',[]):
  if r.get('sha256') and r['split']!='train':source_splits[r['sha256']].add(r['split'])
assert all(len(v)==1 for v in source_splits.values())
emit('original-source-digests',{'unique_digests':len(source_splits),'cross_split_digests':0})
# Project code is executed on available local assets, without model loading.
sys.path.insert(0,str(ROOT))
from tiny_perceptron import natural_assistant as core
loaded,data_root=core.load_manifest(manifest_path,ROOT/'outputs/natural-v4/data')
emit('load_manifest',{'status':'passed','actual_asset_hashes_verified':len(loaded['asset_sha256'])})
spec=importlib.util.spec_from_file_location('builder',ROOT/'scripts/build_natural_v4_assets.py');builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
assets=builder.validate_assets(data_root,m['rows'],m['audio_rows'],m['sources'])
emit('validate_assets',{'status':'passed','selected_asset_files':len(assets)})
# Renamed identical bytes across splits must be rejected before disk access.
sources=[{'path':'vision-sources.json','metadata':{'rows':[
 {'image':'a.jpg','bytes':1,'sha256':'1'*64,'family':'a','split':'train'},
 {'image':'renamed.jpg','bytes':1,'sha256':'1'*64,'family':'renamed','split':'test'}]}}]
try:builder.validate_assets(RAW,[],[],sources)
except ValueError as e:
 assert str(e)=='Identical source bytes cross dataset splits'
 emit('renamed-byte-negative-probe',{'rejected':True,'message':str(e)})
else:raise AssertionError('Renamed same bytes were accepted')
# Check final source tree identity and parent edges against preserved original messages.
chat=[r for r in m['rows'] if r['task']=='chat' and r.get('source_tree_id')]
snapshot_path=ROOT/json.loads((ROOT/'docs/natural-assistant/v4/data/chat-sources.json').read_text())['source_message_snapshot']
messages={r['message_id']:r for r in map(json.loads,snapshot_path.read_text().splitlines())}
trees=defaultdict(set)
for r in chat:
 ids=r['source_message_ids'];trees[r['source_tree_id']].add(r['split'])
 for index,mid in enumerate(ids):
  original=messages[mid]
  assert original['source_tree_id']==r['source_tree_id']
  assert original['parent_id']==(None if index==0 else ids[index-1])
assert all(len(v)==1 for v in trees.values())
emit('oasst-tree-check',{'original_snapshot_sha256':sha(snapshot_path),'selected_source_rows':len(chat),'original_message_records':len(messages),'source_trees':len(trees),'source_rows_by_split':dict(Counter(r['split'] for r in chat)),'trees_by_split':{k:sum(v=={k} for v in trees.values()) for k in ['train','validation','test']},'cross_split_trees':0,'parent_edge_mismatches':0})
# Bind genuine GPU-run records to selection; this is CPU record validation only.
selection_path=ROOT/'docs/natural-assistant/v4/selection.json';sel=json.loads(selection_path.read_text())
b=ROOT/'outputs/natural-v4/modal-runs'
vpath=b/'validation-37219466611/natural-natural-v4-validation-37219466611-1/review/result.json'
tpath=b/'evaluate-37221188153/natural-natural-v4-evaluate-37221188153-1/review/result.json'
v=json.loads(vpath.read_text());t=json.loads(tpath.read_text());ex=t['execution']
assert sha(manifest_path)==sel['dataset_manifest_sha256']==v['manifest_sha256']==t['manifest_sha256']
assert sha(vpath)==sel['validation_result_sha256']
assert sha(selection_path)==ex['selection_sha256']
assert v['status']==t['status']=='completed' and v['split']=='validation' and t['split']=='test'
assert sel['selected_variant']=='base' and list(t['variants'])==['base'] and t['selected_only'] is True
assert all(r['completed'] for r in v['variants'].values())
assert ex['pretest_selection']['validation_result_sha256']==sha(vpath)
assert v['observed_at']<t['observed_at']
trainpath=b/'train-37217452291/natural-natural-v4-train-37217452291-1/review/result.json'
train=json.loads(trainpath.read_text())
emit('selection-record-check',{'validation_result_sha256':sha(vpath),'test_result_sha256':sha(tpath),'selection_sha256':sha(selection_path),'selected_variant':sel['selected_variant'],'validation_variants':list(v['variants']),'test_variants':list(t['variants']),'validation_generation_counts':{k:r['generation_count'] for k,r in v['variants'].items()},'validation_requested_rows':v['requested_visual_text_rows'],'validation_audio_rows':v['requested_audio_rows'],'validation_seed':v['seed'],'validation_observed_at':v['observed_at'],'test_observed_at':t['observed_at'],'training_record_sha256':sha(trainpath),'training_steps':train.get('completed_steps'),'training_seed':train.get('seed'),'training_rows':train.get('trained_rows'),'own_gpu_training':False})
emit('overall',{'status':'passed','limits':'Only hand-card execution, independent manifest/source recomputation, local file hashes, validator negative probe and immutable project record checks. No model inference, new GPU training or quality benchmark replication.'})
