"""核對變動的技術句、原證據與現行導覽；僅保存新檔。"""
import contextlib
import datetime
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import torch

ROOT=Path.cwd()
ART=Path(__file__).resolve().parent
ORIGINAL=ROOT/'docs/course-review-20261019/reviews/rv1019_technical_first-steps/technical-first-steps-original-01.json'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def run(args):
    p=subprocess.run(args,cwd=ROOT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True,timeout=45)
    return {'command':args,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}

def section(path,identifier):
    raw=Path(path).read_text()
    m=re.search(r'^## '+re.escape(identifier)+r'\s.*?(?=^## |\Z)',raw,re.M|re.S)
    if not m:
        raise ValueError('找不到指定小節：'+str(path)+'#'+identifier)
    return m.group(0)

def execute(code):
    output=io.StringIO()
    ns={'__name__':'__main__'}
    with contextlib.redirect_stdout(output):
        exec(compile(code,'技術複查算例','exec'),ns)
    return {'code':code,'stdout':output.getvalue()},ns

original=json.loads(ORIGINAL.read_bytes())
assert sha(ORIGINAL)=='1ce5dde65636c0ff9b6a480e7a9eea09a5b4f38ee746439abcac0f50fae13a4d'
data={'command':'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python '+str(Path(__file__).relative_to(ROOT)),
      'environment':{'python':platform.python_version(),'torch':torch.__version__,'device':'cpu','os':platform.platform()},
      'accessed_on':datetime.date.today().isoformat(),'original_report_sha256':sha(ORIGINAL),
      'original_issues':original['issues']}
data['retained_artifacts']=[{'path':a['path'],'recorded_sha256':a['sha256'],'current_sha256':sha(ROOT/a['path']),
                            'unchanged':a['sha256']==sha(ROOT/a['path'])} for a in original['artifacts']]
assert all(a['unchanged'] for a in data['retained_artifacts'])
data['repository_inputs']=[{'id':s['id'],'path':s['path'],'recorded_sha256':s['sha256'],
                            'current_sha256':sha(ROOT/s['path']),'unchanged':s['sha256']==sha(ROOT/s['path'])}
                           for s in original['sources'] if s['kind']=='repository_code']
freeze=ROOT/'docs/course-review-20261019/freeze-06.json'
page=next(p for p in json.loads(freeze.read_bytes())['pages'] if p['page_id']=='first-steps')
data['freeze']={'path':str(freeze.relative_to(ROOT)),'sha256':sha(freeze),'page':page,
                'snapshot_sha256':sha(ROOT/page['snapshot']),'current_source_sha256':sha(ROOT/page['source'])}
assert data['freeze']['sha256']=='9f5489d6e94946c6aa8690a1672b7f931b02e8c61ebc59f84f179eba5e44a5c9'
assert data['freeze']['snapshot_sha256']==data['freeze']['current_source_sha256']==original['source_sha256']
data['figures']=[{'path':name,'sha256':sha(ROOT/name),'expected':digest} for name,digest in original['figure_sha256'].items()]
assert all(f['sha256']==f['expected'] for f in data['figures'])
data['diffs']=[]
for name,base in [('01','f47d47b91432'),('02','5ef56bc756f3'),('03','5ef56bc756f3')]:
    path='course/chapters/'+name+'.md'
    diff=run(['git','diff',base,'6a7aa1de5227','--',path])
    assert diff['returncode']==0
    old=run(['git','show',base+':'+path])
    target=run(['git','show','6a7aa1de5227:'+path])
    expected=next(s['sha256'] for s in original['sources'] if s.get('path')==path)
    assert hashlib.sha256(old['stdout'].encode()).hexdigest()==expected
    assert hashlib.sha256(target['stdout'].encode()).hexdigest()==sha(ROOT/path)
    data['diffs'].append({'path':path,'base':base,'target':'6a7aa1de5227','old_sha256':expected,
                          'current_sha256':sha(ROOT/path),'diff':diff})

# 原文沒有變動，重新執行的所有數字與文字輸出仍須符合原有結果。
previous=json.loads((ART/'verification.json').read_bytes())
current=json.loads((ART/'verification-current-01.json').read_bytes())
keys=['python_blocks','table','division_boundary','animals_exercise','function','tensor','scores_exercise',
      'log_values','log_identity','cross_entropy','bias_exercise','gradient_exercises','gradient_accumulation',
      'finite_differences','roundtrip_exercise','checkpoint_created']
data['replayed_outputs']=[{'field':key,'identical_to_original':previous[key]==current[key]} for key in keys]
assert all(x['identical_to_original'] for x in data['replayed_outputs'])
assert current['check_env']['returncode']==0 and current['train_dry_run']['returncode']==0

# 導覽只查實作位置，不把連結章檔的所有技術結論納入本頁審查。
old_repository=json.loads((ART/'repository-inspection.json').read_bytes())
data['implementation_table']=[]
for row in old_repository['implementation_table']:
    entries=[]
    for entry in row['entries']:
        raw=section(ROOT/entry['source'],entry['anchor'])
        mentions=[line for line in raw.splitlines() if row['package'] in line or '節錄自' in line or '精簡自' in line]
        assert mentions
        entries.append({'source':entry['source'],'anchor':entry['anchor'],'source_sha256':sha(ROOT/entry['source']),
                        'mentions':mentions,'old_mentions_retained':all(line in raw for line in entry['mentions'])})
    data['implementation_table'].append({'package':row['package'],'entries':entries})
data['curriculum_order']=[]
for name in ['01','02','03']:
    path=ROOT/'course/chapters'/(name+'.md')
    raw=path.read_text()
    mentions=[]
    for m in re.finditer(r'^## (.+)$',raw,re.M):
        part=re.split(r'^## ',raw[m.end():],maxsplit=1,flags=re.M)[0]
        hits=[line for line in part.splitlines() if 'from torch import nn' in line or '.copy_(' in line
              or 'nn.Linear' in line or ' @ ' in line or 'class CausalAttention' in line]
        if hits:
            mentions.append({'heading':m.group(1),'mentions':hits})
    data['curriculum_order'].append({'path':str(path.relative_to(ROOT)),'sha256':sha(path),'sections':mentions})

# 改過的平方代價練習：係數 0.5 改成 2。
torch.set_num_threads(1)
raw=section(ROOT/'course/chapters/01.md','1.11')
block=re.search(r'```python\n(.*?)\n```',raw,re.S).group(1)
record,ns=execute(block)
data['coefficient_original']=record
changed=block.replace('second_cost = 0.5 * w[1].square()','second_cost = 2 * w[1].square()').replace('[-4.0, 2.0]','[-4.0, 8.0]')
record,ns=execute(changed)
data['coefficient_changed']=record
assert ns['loss'].item()==12 and ns['w'].grad.tolist()==[-4.,8.]
data['coefficient_derivation']='係數比 2/0.5=4；第二項代價從 0.5*2^2=2 變成 2*2^2=8，偏導從 0.5*2*2=2 變成 2*2*2=8。第一項代價 4、偏導 -4 不變，所以總代價 12。'

# 改過的有限樣本敘述：保留原 seed、分母與 dtype。
raw=section(ROOT/'course/chapters/03.md','3.5')
block=re.search(r'```python\n(.*?)\n```',raw,re.S).group(1)
record,ns=execute(block)
data['finite_sample_execution']=record
sample=torch.tensor([1.]*6+[-1.]*3)
data['finite_sample_exact_example']={'values':sample.tolist(),'N':sample.numel(),'mean':sample.mean().item(),
    'correction':1,'sample_std':sample.std().item(),'theoretical_population_std':1.0,
    'derivation':'九次獨立 Rademacher 抽樣有六正三負的可能；sum=3、sum of squares=9，樣本變異數=(9-3^2/9)/(9-1)=1。有限樣本也可能恰等於理論標準差，不能說必定不等。'}
assert sample.std().item()==1.0

# 直接以 curl 讀原出版者的現代定義，只保留短摘錄。
spec=importlib.util.spec_from_file_location('source_reader',ART/'inspect_sources.py')
reader=importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)
data['new_primary_sources']=[reader.fetch(x) for x in [
 ('cornell_perceptron','https://cvw.cac.cornell.edu/SciML/mlp/perceptron',['It takes an input vector','activation function']),
 ('d2l_mlp','https://en.d2l.ai/chapter_multilayer-perceptrons/mlp.html',['commonly called a multilayer perceptron','nonlinear activation function']),
 ('torch_std_current','https://docs.pytorch.org/docs/2.14/generated/torch.std.html',['correction','Bessel'])]]
assert all(s['returncode']==0 for s in data['new_primary_sources'])
assert all(any(e['found'] for e in s['excerpts']) for s in data['new_primary_sources'])
with (ART/'recheck-current-01.json').open('x') as f:
    json.dump(data,f,ensure_ascii=False,indent=2)
    f.write('\n')
print(json.dumps({'retained_artifacts_unchanged':len(data['retained_artifacts']),
    'original_page_claims':len(original['claims']),'replayed_fields':len(keys),
    'implementation_rows':len(data['implementation_table']),
    'coefficient_changed':data['coefficient_changed']['stdout'],
    'finite_sample_execution':data['finite_sample_execution']['stdout'],
    'finite_sample_exact_std':data['finite_sample_exact_example']['sample_std'],
    'artifact':str((ART/'recheck-current-01.json').relative_to(ROOT))},ensure_ascii=False))
