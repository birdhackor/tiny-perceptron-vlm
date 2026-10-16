"""Check retained terminal v2 training receipt, without training or evaluation."""
import hashlib
import json
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
base = ROOT / 'docs/selftrained/results/training-raw/moe-native/raw'
receipt = json.loads((base/'train-receipt.json').read_text())
execution = json.loads((base/'execution.json').read_text())
final = json.loads((ROOT/'docs/selftrained/results/v2-final-public-results.json').read_text())
assert execution['status'] == 'completed' and execution['returncode'] == 0
assert execution['run_id'] == final['architectures']['moe']['selected_source']['run_id']
assert receipt['steps'] == 4000 and receipt['completed_requested_steps']
assert receipt['interrupted'] is False and receipt['selected_checkpoint_available'] and receipt['inference_exported']
assert receipt['origin']['kind'] == 'all-neural-weights-random'
result = {'environment':{'python':platform.python_version(),'device':'cpu'},
 'scope':'Read and compare original historical terminal-stage execution and training receipts; no model execution.',
 'training_receipt_path':(base/'train-receipt.json').relative_to(ROOT).as_posix(),
 'training_receipt_sha256':hashlib.sha256((base/'train-receipt.json').read_bytes()).hexdigest(),
 'execution_path':(base/'execution.json').relative_to(ROOT).as_posix(),
 'execution_sha256':hashlib.sha256((base/'execution.json').read_bytes()).hexdigest(),
 'run_id':execution['run_id'],'status':execution['status'],'returncode':execution['returncode'],
 'steps':receipt['steps'],'completed_requested_steps':receipt['completed_requested_steps'],
 'interrupted':receipt['interrupted'],'selected_checkpoint_available':receipt['selected_checkpoint_available'],
 'inference_exported':receipt['inference_exported'],'origin_kind':receipt['origin']['kind'],
 'selected_source_same_run':True}
out=Path(__file__).with_name('T.11-v2-terminal-check-output.json')
out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
