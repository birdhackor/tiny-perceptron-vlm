"""Pure course interface/convention checks; no trained neural weights are loaded."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import sys
from types import SimpleNamespace

import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from tiny_perceptron.data import fingerprint
from tiny_perceptron.model import generate, masked_loss, loss_sum
from tiny_perceptron.simple import ContextMLP
from tiny_perceptron.selftrained.tools import parse_tool_call, execute_tool_call, ToolCallError


class ArtificialLogitSource(nn.Module):
    def __init__(self, next_token, max_length=8):
        super().__init__()
        self.config=SimpleNamespace(max_length=max_length)
        self.next_token=next_token
    def forward(self, ids, cache=None):
        scores=torch.full((*ids.shape,4),-10.)
        scores[...,self.next_token]=10.
        return {'logits':scores,'cache':None}


def main():
    assert fingerprint(' a  b\n') == fingerprint('a b')
    logits=torch.tensor([[[2.,1.,0.],[1.,0.,2.],[0.,2.,1.]]],dtype=torch.float64)
    labels=torch.tensor([[2,-100,0]])
    total,count=loss_sum(logits,labels)
    assert count.item()==2 and abs(masked_loss(logits,labels).item()-total.item()/2)<1e-12
    model=ContextMLP(8,context=3,width=4)
    assert model(torch.tensor([[1,2,3]])).shape==(1,8)
    call=parse_tool_call('{"tool":"calculator","operation":"add","a":2,"b":2}')
    assert execute_tool_call(call)['result']==4
    rejected=[]
    for text in ['not JSON','{"tool":"calculator","operation":"add","a":2}',
                 '{"tool":"calculator","operation":"add","a":true,"b":2}']:
        try: parse_tool_call(text)
        except ToolCallError:rejected.append(text)
    assert len(rejected)==3
    prefix=torch.tensor([[0,1]])
    capped=generate(ArtificialLogitSource(3),prefix,max_new_tokens=2)
    eos=generate(ArtificialLogitSource(2),prefix,max_new_tokens=6)
    context=generate(ArtificialLogitSource(3,max_length=3),prefix,max_new_tokens=6)
    assert capped.tolist()==[[0,1,3,3]] and eos.tolist()==[[0,1,2]] and context.tolist()==[[0,1,3]]
    source_quotes=[]
    for file,section,phrase in [('09','9.16','開錯頁的成本是5單位'),('17','17.2','原值與還原值的差叫捨入誤差'),
                                ('08','8.10','裁判核對'),('17','17.12','範圍校準')]:
        path=ROOT/f'course/chapters/{file}.md';text=path.read_text()
        start=re.search(r'^## '+re.escape(section)+r' .*$',text,re.M)
        tail=text[start.end():];end=re.search(r'^## ',tail,re.M)
        unit=text[start.start():start.end()+end.start() if end else len(text)]
        assert phrase in unit
        line=next(line for line in unit.splitlines() if phrase in line)
        source_quotes.append({'path':str(path.relative_to(ROOT)),'file_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                             'section':section,'section_sha256':hashlib.sha256(unit.encode()).hexdigest(),'small_source_quote':line[:420]})
    out={'created_at':datetime.now(timezone.utc).isoformat(),'reviewer_task':'/root/omni_p5_technical_glossary_whole',
         'command':'/workspace/tiny-perceptron-vlm/.venv/bin/python docs/technical-reviews/artifacts/omni-P5-glossary/interface-checks-v1.py',
         'environment':{'python':platform.python_version(),'torch':torch.__version__,'device':'CPU'},'status':'passed',
         'results':{'whitespace_normalization_equal':True,'loss_sum':total.item(),'effective_targets':count.item(),'mean_loss':masked_loss(logits,labels).item(),
                    'context_mlp_output_shape':[1,8],'tool_arithmetic_result':4,'invalid_tool_calls_rejected':3,
                    'forced_stub_generate':{'max_new_tokens':capped.tolist(),'eos':eos.tolist(),'context_limit':context.tolist()},
                    'source_quotes':source_quotes},
         'limits':'Only randomly initialized tiny shape demo, forced logits and fixed scalar arithmetic; no trained model, dataset benchmark, calibration selection or formal final inference.'}
    path=Path(__file__).with_suffix('.json');path.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':'passed','path':str(path.relative_to(ROOT)),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}))


if __name__=='__main__':main()
