"""Small source and navigation supplement; no kernels, training or final evaluation."""

import hashlib
import importlib
import inspect
import json
import platform
import re
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path

import torch

ROOT = Path('/tmp/tpb').resolve()
OUT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def walk(value):
    if isinstance(value, str):
        if re.fullmatch(r'[0-9ABC]+\.\d+\.md', value):
            yield value[:-3]
    elif isinstance(value, (list, dict)):
        for item in value.values() if isinstance(value, dict) else value:
            yield from walk(item)


nav_path = ROOT/'zensical.toml'
ids = list(walk(tomllib.loads(nav_path.read_text())['project']['nav']))
assert ids[ids.index('7.26')+1:ids.index('8.1')] == [f'C.{i}' for i in range(1,6)]
assert ids[ids.index('13.17')+1:ids.index('14.1')] == ['C.6','C.7']
assert [pid for pid in ids if pid.startswith('14.')] == [f'14.{i}' for i in range(1,20)]
assert [pid for pid in ids if pid.startswith('19.')] == [f'19.{i}' for i in range(1,26)]
from tiny_perceptron.data import shifted, render_chat, IGNORE

x,y=shifted([4,7,9,3])
assert x.tolist()==[4,7,9] and y.tolist()==[7,9,3]
inputs,labels=render_chat([{'role':'user','content':'a'},{'role':'assistant','content':'b'}])
assert len(inputs)==len(labels) and int((labels!=IGNORE).sum())==2

sgd = importlib.import_module('torch.optim.sgd')
pieces=[]
provenance=[]
for symbol,relative,limit in [
    (torch.Tensor.backward,'torch/_tensor.py',65),
    (torch.optim.Optimizer.zero_grad,'torch/optim/optimizer.py',65),
    (torch.optim.SGD.step,'torch/optim/sgd.py',85),
    (sgd._single_tensor_sgd,'torch/optim/sgd.py',115),
]:
    original=inspect.unwrap(symbol)
    path=Path(inspect.getsourcefile(original))
    lines,start=inspect.getsourcelines(original)
    url=f'https://github.com/pytorch/pytorch/blob/{torch.version.git_version}/{relative}'
    record={'symbol':symbol.__qualname__,'unwrapped_file':str(path),'full_file_sha256':sha(path),'source_start_line':start,'url':url}
    provenance.append(record)
    pieces.append(json.dumps(record,ensure_ascii=False)+'\n')
    pieces.extend(f'{start+i}: {line}' for i,line in enumerate(lines[:limit]))
    pieces.append('\n')
with (OUT/'pytorch-provenance-supplement.txt').open('x') as target:
    target.write('The prior excerpt source path for decorated Optimizer.zero_grad used inspect.getsourcefile before unwrapping. The code lines were already optimizer.py lines; this supplement records the unwrapped physical file explicitly.\n\n'+''.join(pieces))
result={
    'created_at':datetime.now(UTC).isoformat(),
    'environment':{'python':sys.version,'torch':torch.__version__,'torch_git_version':torch.version.git_version,'platform':platform.platform(),'device':'cpu'},
    'navigation_source':{'path':'zensical.toml','sha256':sha(nav_path)},
    'navigation_assertions':{'C.1_to_C.5_after_7.26_before_8.1':True,'C.6_and_C.7_after_13.17_before_14.1':True,'14.1_to_14.19_in_order':True,'19.1_to_19.25_in_order':True},
    'shifted_probe':{'ids':[4,7,9,3],'X':x.tolist(),'Y':y.tolist()},
    'render_chat_probe':{'messages':[{'role':'user','content':'a'},{'role':'assistant','content':'b'}],'X':inputs.tolist(),'Y':labels.tolist(),'effective_supervision_positions':int((labels!=IGNORE).sum()),'scope':'one assistant byte plus assistant EOS; user text remains visible input but is ignored in targets'},
    'official_source_provenance':provenance,
    'result':'all navigation and X/Y assertions passed; official source was read from the installed release, no network request',
}
with (OUT/'navigation-and-source-probe.json').open('x') as target:
    target.write(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
