import json,urllib.request,hashlib,subprocess
from pathlib import Path
b=Path('docs/technical-reviews/artifacts/p7_technical_e/sources');rev='3f9fbb2a04f08034c0b630e837e7afba1f584402'
for src,name in [('tiny_perceptron/selftrained/model.py','selftrained-model-3f9.py'),('scripts/selftrained/train.py','selftrained-train-3f9.py')]:
 p=b/name;assert not p.exists();p.write_bytes(subprocess.check_output(['git','show',rev+':'+src]));print(name,hashlib.sha256(p.read_bytes()).hexdigest())
for n in ['moe-pretrain','moe-sft','moe-vision','moe-ocr','moe-audio','moe-joint','moe-weighted','moe-native']:
 p=Path('docs/selftrained/results/training-raw/'+n+'/raw/train-receipt.json');d=json.loads(p.read_text());print(n,'origin',d.get('origin'),'completed',d.get('completed_requested_steps'),'exported',d.get('inference_exported'))
api='https://huggingface.co/api/models/birdhackor/tiny-perceptron-course-models';info=json.load(urllib.request.urlopen(api));sha=info['sha'];url=api+'/tree/'+sha+'/selftrained/v2/moe-joint';raw=urllib.request.urlopen(url).read();p=b/'hf-moe-joint-tree.json';assert not p.exists();p.write_bytes(raw);tree=json.loads(raw);w=next(x for x in tree if x['path'].endswith('model.safetensors'));print('public_revision',sha,'tree_url',url,'public_weight',w['size'],w['lfs']['oid'])
f=json.loads(Path('docs/selftrained/results/public-raw/moe/freeze/frozen.json').read_text());print('freeze_match',w['lfs']['oid']==f['checkpoint_sha256'])
