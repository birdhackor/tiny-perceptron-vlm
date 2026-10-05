"""Bounded CLI record-limit and reporting probe after initial semantic checks."""
import json
import platform
import subprocess
import sys
from pathlib import Path
import torch
from scripts.evaluate import evaluate
from tiny_perceptron.model import TinyLM, ModelConfig

root=Path.cwd(); out=Path(__file__).resolve().parent
work=root/'outputs/natural-v4/factual-research/T.4/cpu'
rows=[json.loads(line) for line in (work/'generated/attributes-sft/validation.jsonl').read_text().splitlines()]*5
data=work/'twenty-five-eval-rows.jsonl';data.write_text(''.join(json.dumps(row)+'\n' for row in rows))
receipts=[]
for option,count in [('default',20),('all',25)]:
    target=work/('limit-'+option+'.json')
    command=[sys.executable,'scripts/evaluate.py',str(work/'start.pt'),'--data',str(data),'--mode','sft','--tokens','2','--device','cpu','--output',str(target)]
    if option=='all': command+=['--limit','all']
    result=subprocess.run(command,text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    report=json.loads(target.read_text());terminal=json.loads(result.stdout)
    assert report['records_read']==25 and report['records_selected']==count and report['generation_evaluated_records']==count
    assert report['unselected_records']==25-count and len(report['samples'])==count and 'samples' not in terminal
    receipts.append({'option':option,'command':command,'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'records_read':25,'records_selected':count,'saved_samples':len(report['samples']),'terminal_excludes_samples':'samples' not in terminal,'metric_denominators':report['metric_denominators']})
torch.manual_seed(42);model=TinyLM(ModelConfig(width=32))
normal={'messages':[{'role':'user','content':'shape?'},{'role':'assistant','content':'circle'}]}
overlong={'messages':[{'role':'user','content':'?'*200},{'role':'assistant','content':'circle'}]}
report=evaluate(model,[normal,overlong],mode='sft',max_new_tokens=2)
assert report['effective_tokens']==7 and report['generation_evaluated_records']==1 and report['loss_evaluated_records']==1 and report['loss_skipped_records']==report['generation_skipped_records']==1
payload={'environment':{'python':platform.python_version(),'torch':str(torch.__version__),'device':'cpu'},'cli':receipts,'mixed_length_evaluation':{k:v for k,v in report.items() if k!='samples'}}
(out/'evaluate-flags.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(payload,ensure_ascii=False,indent=2))
