from pathlib import Path
import hashlib
import json
import platform
import sys
import torch

root=Path(__file__).resolve().parents[4]
base=Path(__file__).resolve().parent
messages=json.loads((root/'docs/selftrained/examples/v2/text.messages.json').read_text())
out={}
for split in ['train','validation','test']:
    p=root/('outputs/selftrained-v2/data/text-tools-'+split+'.jsonl')
    matches=[]
    for line_no,line in enumerate(p.read_text().splitlines(),1):
        r=json.loads(line)
        if r['messages'][:-1]==messages:
            matches.append({'line':line_no,'id':r['id'],'group_id':r['group_id']})
            if split=='validation':
                (base/'validation-public-example-raw-line.jsonl').write_text(line+'\n')
                (base/'validation-public-example-provenance.json').write_text(json.dumps({'source':str(p.relative_to(root)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'original_line':line_no,'inspected_pointers':['/messages','/id','/group_id','/split']},indent=2)+'\n')
    out[split]=matches
assert not out['train'] and len(out['validation'])==1 and not out['test']
(base/'public-prompt-split-membership.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'python':sys.version,'torch':torch.__version__,'device':'cpu','platform':platform.platform(),'membership':out,'method':'Exact public message-list equality against raw record prefix; no prediction, scoring, or model call'},ensure_ascii=False,indent=2))
