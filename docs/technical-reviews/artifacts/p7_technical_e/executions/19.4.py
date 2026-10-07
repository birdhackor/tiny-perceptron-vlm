import json
from pathlib import Path
repo=next(p for p in (Path.cwd(),*Path.cwd().parents) if (p/'docs/selftrained/v2-manifest.json').is_file());index=json.loads((repo/'docs/selftrained/v2-training-stage-index.json').read_text())
for row in index['stages']:
 if row['stage_key'].startswith('moe-'):print(row['stage_key'],'完成',row['completed_steps'],'選定',row['selected_step'])
selection=json.loads((repo/'docs/selftrained/results/v2-final-training-source-selection.json').read_text());native=selection['moe_joint'];print('moe-native','完成',native['completed_requested_steps'],'選定',native['selected_step'])
root=repo/'docs/selftrained/results/training-raw';nexts={'pretrain':'sft','sft':'vision','vision':'ocr','ocr':'audio','audio':'joint','joint':'weighted'}
for row in index['stages']:
 if row['stage_key'].startswith('moe-'):
  stage=row['stage_key'][4:];r=json.loads((root/row['stage_key']/'raw/train-receipt.json').read_text());nr=json.loads((root/('moe-'+nexts[stage])/'raw/train-receipt.json').read_text());assert row['completed_steps']==r['steps'];assert row['selected_step']==nr['stage_history'][-1]['step']
r=json.loads((root/'moe-native/raw/train-receipt.json').read_text());public=json.loads((repo/'outputs/p7-technical-e-cache/moe-joint/inference-manifest.json').read_text());assert native['completed_requested_steps']==r['steps']==4000;assert native['selected_step']==public['selected_step']==1000;print('mechanical_index_matched_original_receipts_and_public_manifest')
