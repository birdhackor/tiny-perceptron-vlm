"""Inspect raw historical measurements only; skip limits and review commentary."""
from pathlib import Path
import hashlib,json,platform
ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
rows=[]
for name in ['rag','tools','reasoning']:
    p=ROOT/f'docs/course-experiments/results/{name}.json'
    q=OUT/'historical-originals'/p.name;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(p.read_bytes())
    assert hashlib.sha256(p.read_bytes()).digest()==hashlib.sha256(q.read_bytes()).digest()
    obj=json.loads(p.read_bytes());r=obj['results']
    pointers=['/revision','/device','/seed','/torch_version','/results/split']
    row={'experiment':name,'original':str(p.relative_to(ROOT)),'copy':str(q.relative_to(ROOT)),
         'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'revision':obj['revision'],'device':obj['device'],
         'seed':obj['seed'],'torch':obj['torch_version'],'split':r['split']}
    if name=='reasoning':
        samples=r['comparison']['direct']['budgets'][0]['samples']
        pointers+=['/results/comparison/direct/test','/results/comparison/direct/budgets/0/candidate_count',
                   '/results/comparison/direct/budgets/0/samples']
        row['test_denominators']=r['comparison']['direct']['test']
        row['sample_count']=len(samples)
        row['sample_input_output']=[{k:x[k] for k in ['question','truth','generation','candidates']} for x in samples]
        assert len(samples)==r['split']['test']['records']==24
    else:
        samples=r['samples'];pointers+=['/results/test','/results/samples']
        row['test_denominators']=r['test'];row['sample_count']=len(samples)
        fields=['query','expected','generation','documents'] if name=='rag' else ['question','expected','answer','trace','actual_tool_calls']
        row['sample_input_output']=[{k:x[k] for k in fields} for x in samples]
        if name=='tools':
            assert len(samples)==r['split']['test']['records']==20
            row['executed_trace_count']=sum(s['executed'] for x in samples for s in x['trace'])
    assert row['test_denominators']['examples']==r['split']['test']['records']
    row['pointers']=pointers;rows.append(row)
(OUT/'history-audit-result.json').write_text(json.dumps({'rows':rows,'environment':{'python':platform.python_version(),'device':'cpu'},
 'scope':'archived inputs, generated raw outputs and denominator presence; no quality re-evaluation'},ensure_ascii=False,indent=2)+'\n')
print(json.dumps([{'experiment':x['experiment'],'sample_count':x['sample_count'],'test_records':x['split']['test']['records'],'effective_tokens':x['test_denominators']['effective_tokens']} for x in rows],indent=2))
