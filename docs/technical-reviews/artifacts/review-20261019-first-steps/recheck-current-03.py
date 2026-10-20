"""核對暖身與其參照來源，只保存本次新證據。"""
import contextlib
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import torch

ROOT=Path.cwd()
ART=Path(__file__).resolve().parent
PREVIOUS=ROOT/'docs/course-review-20261019/reviews/rv1019_technical_first-steps/technical-first-steps-current-02.json'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def run(args):
    p=subprocess.run(args,cwd=ROOT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True,timeout=45)
    return {'command':args,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}

def section(raw,identifier):
    match=re.search(r'^## '+re.escape(identifier)+r'\s.*?(?=^## |\Z)',raw,re.M|re.S)
    if not match:
        raise ValueError('找不到小節：'+identifier)
    return match.group(0)

assert sha(PREVIOUS)=='c08a850aecb91e8d9f997dff9332a7be3b8c534cc869520d6ba0ee3b1192f44c'
previous=json.loads(PREVIOUS.read_bytes())
data={'command':'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python '+str(Path(__file__).relative_to(ROOT)),
      'environment':{'python':platform.python_version(),'torch':torch.__version__,'device':'cpu','os':platform.platform()},
      'accessed_on':datetime.date.today().isoformat(),'previous_report':{'path':str(PREVIOUS.relative_to(ROOT)),'sha256':sha(PREVIOUS)},
      'previous_issues':previous['issues'],'technical_review_guide_sha256':sha(ROOT/'docs/technical-review-guide.md')}
data['retained_artifacts']=[{'path':a['path'],'sha256':sha(ROOT/a['path']),'expected':a['sha256']} for a in previous['artifacts']]
assert all(a['sha256']==a['expected'] for a in data['retained_artifacts'])
data['repository_inputs']=[{'id':s['id'],'path':s['path'],'previous_sha256':s['sha256'],'current_sha256':sha(ROOT/s['path'])}
                           for s in previous['sources'] if s['kind']=='repository_code']
freeze=ROOT/'docs/course-review-20261019/freeze-07.json'
page=next(p for p in json.loads(freeze.read_bytes())['pages'] if p['page_id']=='first-steps')
data['freeze']={'path':str(freeze.relative_to(ROOT)),'sha256':sha(freeze),'page':page,
                'current_source_sha256':sha(ROOT/page['source']),'snapshot_sha256':sha(ROOT/page['snapshot'])}
assert data['freeze']['sha256']=='67358baeb4833b9b08445fb06bb5ccae0f31fbb7727163f93e1265bbcb420753'
assert data['freeze']['current_source_sha256']==data['freeze']['snapshot_sha256']==previous['source_sha256']
data['figures']={name:sha(ROOT/name) for name in page['figures_sha256']}
assert data['figures']==page['figures_sha256']==previous['figure_sha256']
data['pngs']={str((ROOT/'outputs/course-review-20261019/figure-renders-04'/(s+'.png')).relative_to(ROOT)):
              sha(ROOT/'outputs/course-review-20261019/figure-renders-04'/(s+'.png')) for s in data['figures'].values()}

base='ecf6dec71280';target='c6b9339d2595';path='course/chapters/03.md'
data['diff']=run(['git','diff',base,target,'--',path])
old=run(['git','show',base+':'+path]);new=run(['git','show',target+':'+path])
assert all(x['returncode']==0 for x in [data['diff'],old,new])
assert hashlib.sha256(old['stdout'].encode()).hexdigest()==next(s['sha256'] for s in previous['sources'] if s.get('path')==path)
assert hashlib.sha256(new['stdout'].encode()).hexdigest()==sha(ROOT/path)=='5313a025920d2916e1fcf14fbe51c8f6e784fa873cbf17497d4e768cfae160d5'
data['unchanged_referenced_sections']=[]
for identifier in ['3.5','3.6','3.7']:
    before=section(old['stdout'],identifier);after=section(new['stdout'],identifier)
    assert before==after
    data['unchanged_referenced_sections'].append({'heading':after.splitlines()[0],
         'previous_sha256':hashlib.sha256(before.encode()).hexdigest(),'current_sha256':hashlib.sha256(after.encode()).hexdigest()})

# 重用既有核對原碼，只在記憶體內選取本次指定 manifest；不改工具檔。
verifier=ART/'verify-current-01.py'
source=verifier.read_text()
old_manifest='docs/course-review-20261019/freeze-06.json'
new_manifest='docs/course-review-20261019/freeze-07.json'
assert source.count(old_manifest)==1
stream=io.StringIO()
with contextlib.redirect_stdout(stream):
    exec(compile(source.replace(old_manifest,new_manifest),str(verifier),'exec'),
         {'__name__':'__main__','__file__':str(Path(__file__).resolve())})
warmup=json.loads(stream.getvalue())
data['warmup_execution']={'base_program':str(verifier.relative_to(ROOT)),'base_program_sha256':sha(verifier),
    'in_memory_replacement':{'from':old_manifest,'to':new_manifest},'result':warmup}
last_warmup=json.loads((ART/'verification-current-01.json').read_bytes())
keys=['python_blocks','table','division_boundary','animals_exercise','function','tensor','scores_exercise',
      'log_values','log_identity','cross_entropy','bias_exercise','gradient_exercises','gradient_accumulation',
      'finite_differences','roundtrip_exercise','checkpoint_created']
data['warmup_comparison']=[{'field':k,'unchanged':warmup[k]==last_warmup[k]} for k in keys]
assert all(x['unchanged'] for x in data['warmup_comparison'])
assert warmup['check_env']['returncode']==warmup['train_dry_run']['returncode']==0

# 實跑現行查詢／匹配段落的兩個程式區塊，核對列、欄與相乘結果。
raw=(ROOT/path).read_text();part=section(raw,'3.3')
blocks=re.findall(r'```python\n(.*?)\n```',part,re.S)
namespace={'__name__':'__main__'}
data['changed_sentence_execution']=[]
for block in blocks:
    output=io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(block,'查詢與匹配算例','exec'),namespace)
    data['changed_sentence_execution'].append({'code':block,'stdout':output.getvalue()})
has=namespace['has'];need=namespace['need']
shared=has@has.T;separate=need@has.T
assert shared.tolist()==[[1.,0.],[0.,1.]]
assert separate.tolist()==[[0.,1.],[1.,0.]]
data['changed_sentence_verification']={'has':has.tolist(),'need':need.tolist(),'shared':shared.tolist(),
    'separate':separate.tolist(),'rows':'提問的 A、B','columns':'被讀的 A、B','shared_argmax':shared.argmax(-1).tolist(),
    'derivation':'A=[1,0]、B=[0,1]。A·A=1、B·B=1、A·B=B·A=0；共用表示的分數表為 I2，每列自位置的分數 1 大於另一位置的 0。分開表示後 query 為交換兩格，need@has.T 為 [[0,1],[1,0]]。實數點積的對稱性由逐項乘法交換律成立；自分數最高的本次敘述只限明示算例。',
    'scope':'核對原始匹配分數，不解讀為 softmax 後機率、實際讀回內容或訓練品質。'}

# 逐個確認現行導覽位置；不重審被連結章節的其他結論。
old_navigation=json.loads((ART/'recheck-current-01.json').read_bytes())
data['implementation_table']=[]
for row in old_navigation['implementation_table']:
    entries=[]
    for entry in row['entries']:
        current=section((ROOT/entry['source']).read_text(),entry['anchor'])
        mentions=[line for line in current.splitlines() if row['package'] in line or '節錄自' in line or '精簡自' in line]
        assert mentions
        entries.append({'source':entry['source'],'anchor':entry['anchor'],'source_sha256':sha(ROOT/entry['source']),
                        'mentions':mentions,'previous_mentions_retained':all(line in current for line in entry['mentions'])})
    data['implementation_table'].append({'package':row['package'],'entries':entries})
data['curriculum_order']=[]
for name in ['01','02','03']:
    p=ROOT/'course/chapters'/(name+'.md');text=p.read_text();matches=[]
    for m in re.finditer(r'^## (.+)$',text,re.M):
        part=re.split(r'^## ',text[m.end():],maxsplit=1,flags=re.M)[0]
        hits=[line for line in part.splitlines() if 'from torch import nn' in line or '.copy_(' in line
              or 'nn.Linear' in line or ' @ ' in line or 'class CausalAttention' in line]
        if hits:matches.append({'heading':m.group(1),'mentions':hits})
    data['curriculum_order'].append({'path':str(p.relative_to(ROOT)),'sha256':sha(p),'sections':matches})
data['page_claim_rechecks']=[{'claim_id':c['id'],'status':'verified',
     'basis':'現行導覽位置重查' if c['id'].startswith('map_') or c['id'] in ['python_class','warmup_curriculum_order']
             else '暖身輸出相同；原證據完整；參照來源的相關段落未變'} for c in previous['claims']]
with (ART/'recheck-current-03.json').open('x') as f:
    json.dump(data,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({'artifacts_retained':len(data['retained_artifacts']),'warmup_blocks':len(warmup['python_blocks']),
    'identical_output_fields':len(data['warmup_comparison']),'shared':shared.tolist(),'separate':separate.tolist(),
    'prior_issues':data['previous_issues'],'navigation_rows':len(data['implementation_table'])},ensure_ascii=False))
