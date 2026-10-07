import json,pathlib,hashlib,datetime,subprocess
ROOT=pathlib.Path('/workspace/selftrained-v2')
BASE=ROOT/'docs/technical-reviews/artifacts/p7_technical_d'
MAN='docs/course-revision-20261007-phase7/reviews/freeze-03/manifest.json'
TRACE='docs/course-revision-20261007-phase7/reviews/freeze-03/traces/technical/d/b166e7ad87554247a44d5cb37345cf2c.jsonl'
SESSION='b166e7ad87554247a44d5cb37345cf2c'
def sha(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
def write(p,obj):
 (ROOT/p).parent.mkdir(parents=True,exist_ok=True);(ROOT/p).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def cp(page,i,u,m,e,c,v,**extra):
 d=dict(page_id=page,unit_index=i,understanding=u,materials_and_labels=m,expected_change=e,confusion_and_quote=c,missing_visuals=v,**extra)
 p=f'outputs/reader-checkpoints/p7_technical_d/{page}-{i}.json';write(p,d)
 r=subprocess.run([str(ROOT/'.venv/bin/python'),'docs/review-tools/phase7_review.py','next',SESSION,'--checkpoint',p],cwd=ROOT,text=True,capture_output=True);print(r.stdout);print(r.stderr) if r.stderr else None
 if r.returncode:raise RuntimeError(r.returncode)
def paper(id,title,url,version,note):
 return dict(id=id,kind='paper',title=title,verified=True,url=url,version=version,authority_reason='作者原始論文，非研究者摘要',checked_original=True,inspection_note=note,accessed_on='2026-10-07')
def claim(id,kind,statement,location,scope,evidence,artifact_ids=None,verification=None,status='verified'):
 d=dict(id=id,kind=kind,statement=statement,location=location,scope=scope,status=status,evidence=[dict(source_id=s,locator=l,supports=t) for s,l,t in evidence],artifact_ids=artifact_ids or [])
 if verification is not None:d['verification']=verification
 return d
def checks(f,n,g,s,l):
 return {key:dict(status=status,details=detail,claim_ids=ids) for key,(status,detail,ids) in zip(('factual_accuracy','numeric_verification','figure_consistency','source_verification','limitations'),(f,n,g,s,l))}
def page(id,summary,variation,checks,sources,claims,artifacts=None,visual_checks=None,page_visual_check=None,issues=None,verdict='pass',missing_visuals='',applicability=None):
 m=json.load(open(ROOT/MAN));meta=next(p for p in m['pages'] if p['page_id']==id)
 d=dict(page_id=id,source_sha256=meta['source_sha256'],figures_sha256=meta['figures_sha256'],verdict=verdict,summary=summary,trace_file=TRACE,question_refs=[],issues=issues or [],missing_visuals=missing_visuals,visual_checks=visual_checks or [],page_visual_check=page_visual_check,variation=variation,artifacts=artifacts or [],sources=sources,claims=claims,checks=checks)
 if applicability:d['applicability']=applicability
 write('docs/technical-reviews/artifacts/p7_technical_d/pages/'+id+'.json',d)
def visual_receipt(name,source_sha,refs):
 for r in refs:r['sha256']=sha(r['path'])
 p='docs/technical-reviews/artifacts/p7_technical_d/'+name+'.json';write(p,dict(tool='tools.view_image',reviewer_task='/root/p7_technical_d',received_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256=source_sha,artifacts=refs));return dict(path=p,sha256=sha(p))
def execute(id,code):
 p='docs/technical-reviews/artifacts/p7_technical_d/'+id+'.py';write_code=ROOT/p;write_code.write_text(code)
 cmd=[str(ROOT/'.venv/bin/python'),p];r=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True)
 version=subprocess.run([cmd[0],'--version'],text=True,capture_output=True).stdout.strip()
 out={'command':'.venv/bin/python '+p,'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'environment':{'python':version,'device':'CPU'},'code':{'path':p,'sha256':sha(p)}}
 op='docs/technical-reviews/artifacts/p7_technical_d/'+id+'-execution.json';write(op,out);print(r.stdout)
 if r.stderr:print(r.stderr)
 if r.returncode:raise RuntimeError('Execution artifact saved with exit '+str(r.returncode))
 return dict(id=id,kind='execution',path=op,sha256=sha(op),description='本人執行當前教材code，保存命令、真stdout/stderr、code SHA及Python版本；沒有模型訓練。',command=out['command'],result='exit '+str(r.returncode)+'; '+r.stdout,environment=out['environment'])
