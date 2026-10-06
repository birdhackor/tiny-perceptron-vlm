"""Bind fetched public export to preserved training and final measurement metadata."""
from pathlib import Path
import hashlib,json,platform
ROOT=Path(__file__).resolve().parents[4];OUT=Path(__file__).resolve().parent
a=json.loads((OUT/'audit-result.json').read_bytes())
p=json.loads((OUT/'public-actual-result.json').read_bytes())
n=next(x for x in a['training'] if x['stage']=='moe-native')
files={x['path']:x for x in n['safe_exports']}
manifest=ROOT/'outputs/p6-T.11-public/moe-joint/inference-manifest.json'
assert hashlib.sha256(manifest.read_bytes()).hexdigest()==files['inference-manifest.json']['sha256']=='f27856892b0c46ca81ba43bfadf7051ed6bd31d848e4868a05147e5799df1e5e'
assert p['commands'][0]['files']['model.safetensors']==files['model.safetensors']['sha256']
assert a['archived_final_completion'][0]['inference_manifest_sha256']==files['inference-manifest.json']['sha256']
m=json.loads(manifest.read_bytes())
assert m['selected_step']==1000 and m['origin']['kind']=='all-neural-weights-random'
q=OUT/'public-inference-manifest.original.json';q.write_bytes(manifest.read_bytes())
result={'environment':{'python':platform.python_version(),'device':'cpu'},'public_revision':p['commands'][0]['public_source']['revision'],
 'manifest_sha256':files['inference-manifest.json']['sha256'],'model_sha256':files['model.safetensors']['sha256'],
 'training_stage':'moe-native','completed_steps':n['steps'],'selected_step':m['selected_step'],
 'selected_checkpoint_sha256':m['selected_checkpoint_sha256'],
 'native_parent_selected_step':n['history'][-1]['step'],
 'same_training_export_and_public_cpu':True,'same_manifest_and_frozen_final_test':True,
 'actual_manifest_pointers_read':['/selected_step','/origin/kind','/selected_checkpoint_sha256','/files'],
 'scope':'identity/completion only; no new heldout measurement or ability pass'}
(OUT/'link-audit-result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
