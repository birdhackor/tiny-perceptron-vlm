import ast
import json
from pathlib import Path

root=Path(__file__).resolve().parents[5]
constants={}
for node in ast.parse((root/'tiny_perceptron/natural_assistant.py').read_text()).body:
    if isinstance(node,ast.Assign) and isinstance(node.value,ast.Constant):
        for target in node.targets:
            if isinstance(target,ast.Name): constants[target.id]=node.value.value
assert constants['MODEL_ID']=='Qwen/Qwen3-VL-2B-Instruct'
assert constants['MODEL_REVISION']=='89644892e4d85e24eaac8bacfd4f463576704203'
m=json.loads((root/'docs/natural-assistant/v4/manifest.json').read_text())
a=json.loads((root/'assets/training/manifest.json').read_text())
pilot_paths={x['archive'] for x in a['assets']}
natural_paths={x['path'] for x in m['archives']}
assert not pilot_paths & natural_paths
assert '.venv-natural' in (root/'docs/natural-assistant/v4/TRAINING.md').read_text()
assert m['dataset_version']=='natural-assistant-v4'
print(json.dumps({'model':constants['MODEL_ID'],'model_revision':constants['MODEL_REVISION'],
 'dataset_version':m['dataset_version'],'natural_archives':sorted(natural_paths),
 'audio_sources':sorted({r['source'] for r in m['audio_rows']}),
 'text_image_tasks':sorted({r['task'] for r in m['rows']}),
 'scope':'Static source/manifest check; no model download, inference or training.'},indent=2))
