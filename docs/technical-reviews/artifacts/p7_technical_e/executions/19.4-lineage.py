import json,hashlib
from pathlib import Path
m=json.loads(Path('docs/selftrained/v2-manifest.json').read_text());hashes={e['path']:e['sha256'] for e in m['records']};root=Path('docs/selftrained/results/training-raw')
for arch,stages in [('moe',['pretrain','sft','vision','ocr','audio','joint','weighted','native']),('dense',['pretrain','sft','vision','ocr','audio','joint','weighted'])]:
 prev=None;history=None
 for stage in stages:
  p=root/(arch+'-'+stage)/'raw';ex=json.loads((p/'execution.json').read_text());r=json.loads((p/'train-receipt.json').read_text());assert ex['returncode']==0 and ex['status']=='completed' and r['completed_requested_steps'];assert r['data_sha256']==hashes;source=ex['job'].get('init_checkpoint');h=r['stage_history']
  if prev is None:assert source is None and h==[]
  else:assert source['run_id']==prev and source['path']=='best.pt' and h[-1]['checkpoint_sha256']==source['sha256'] and h[-1]['selection']=='validation_loss' and h[:-1]==history
  print(arch,stage,'run',ex['run_id'],'source',source,'history_depth',len(h),'steps',r['steps'],'language_weights',[r.get(k,1) for k in ['tool_loss_weight','numeric_run_loss_weight','native_voice_loss_weight']]);prev=ex['run_id'];history=h
print('all_15_raw_stages_and_13_source_links_verified')
