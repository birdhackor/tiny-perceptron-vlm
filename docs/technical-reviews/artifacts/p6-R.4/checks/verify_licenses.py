from pathlib import Path
import hashlib
import json
import tomllib

root=Path('docs/technical-reviews/artifacts/p6-R.4')
files={'project':'inputs/LICENSE','fashion':'originals/fashion-license.txt','noto':'originals/noto-license.txt',
       'minds':'originals/minds-pinned-source.txt','qwen':'originals/qwen-2b-official-card.md',
       'whisper':'originals/whisper-license.txt'}
out={}
for name,rel in files.items():
    p=root/rel;s=p.read_text()
    out[name]={'path':str(p),'sha256_rawbytes':hashlib.sha256(p.read_bytes()).hexdigest()}
    if name in ['project','fashion','whisper']:
        assert 'MIT' in s and 'Permission is hereby granted' in s
        out[name]['license']='MIT'
    elif name=='noto':
        assert 'SIL OPEN FONT LICENSE Version 1.1' in s
        out[name]['license']='SIL OFL 1.1'
    elif name=='minds':
        assert 'license:\n- cc-by-4.0' in s
        out[name]['license']='CC BY 4.0'
    elif name=='qwen':
        assert s.startswith('---\nlicense: apache-2.0')
        out[name]['license']='Apache 2.0'
assert tomllib.loads((root/'inputs/pyproject.toml').read_text())['project']['license']=='MIT'
(root/'checks/license-inspection.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
