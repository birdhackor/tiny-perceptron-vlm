import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent
OUT = ROOT / 'public-metadata'
OUT.mkdir(exist_ok=True)
REV = '979cdfacc588ad0536f1c64fff96f264571cf054'
REPO = 'birdhackor/tiny-perceptron-course-models'
EXPECTED = {'model.safetensors', 'model-config.json', 'tokenizer.json', 'inference-manifest.json'}
raw = ROOT / 'official/hf-immutable-metadata.json'
metadata = json.loads(raw.read_text())
assert metadata['sha'] == REV and metadata['private'] is False and metadata['gated'] is False
siblings = {item['rfilename']: item for item in metadata['siblings']}
files = {k: v for k, v in siblings.items() if k.startswith('selftrained/v2/')}
assert len(files) == 16
rows = []
for stage in ['moe-pretrain', 'moe-sft', 'moe-joint', 'dense-joint']:
    prefix = 'selftrained/v2/' + stage + '/'
    names = {k[len(prefix):] for k in files if k.startswith(prefix)}
    assert names == EXPECTED
    directory = OUT / stage
    directory.mkdir(exist_ok=True)
    values = {}
    for name in EXPECTED - {'model.safetensors'}:
        url = f'https://huggingface.co/{REPO}/resolve/{REV}/{prefix}{name}'
        request = urllib.request.Request(url, headers={'User-Agent': 'factual-review-19.11/1.0'})
        with urllib.request.urlopen(request, timeout=30) as response:
            content = response.read()
        (directory / name).write_bytes(content)
        values[name] = json.loads(content)
    manifest = values['inference-manifest.json']
    for name in ['model-config.json', 'tokenizer.json']:
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == manifest['files'][name]
    assert files[prefix+'model.safetensors']['lfs']['sha256'] == manifest['files']['model.safetensors']
    assert set(manifest['files']) == EXPECTED - {'inference-manifest.json'}
    assert not {'optimizer', 'rng', 'sampler', 'training_options'} & set(manifest)
    assert values['tokenizer.json']['type'] == 'selftrained_char_v1'
    assert len(values['tokenizer.json']['specials']) + len(values['tokenizer.json']['characters']) == values['model-config.json']['vocab_size']
    rows.append({'directory': prefix, 'public_filenames': sorted(names),
                 'architecture': values['model-config.json']['architecture'],
                 'stage': manifest['stage'], 'selected_step': manifest['selected_step'],
                 'origin_kind': manifest['origin']['kind'], 'origin_seed': manifest['origin']['seed'],
                 'manifest_sha256': hashlib.sha256((directory/'inference-manifest.json').read_bytes()).hexdigest(),
                 'payload_sha256': manifest['files'], 'selection': manifest['selection'],
                 'tensor_bytes': files[prefix+'model.safetensors']['size']})
result = {'repository': REPO, 'revision': metadata['sha'], 'private': metadata['private'],
          'gated': metadata['gated'], 'file_count': len(files),
          'inspected_pointers': ['/sha', '/private', '/gated', '/siblings'],
          'authentication': 'anonymous urllib request without Authorization header', 'exports': rows}
(ROOT/'public-metadata-result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
