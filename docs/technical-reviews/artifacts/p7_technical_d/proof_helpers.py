from record import *
import json,hashlib

def refs(*files):return [json.load(open(BASE/f)) for f in files]
def execsources(arts):return [dict(id=a['id'],kind='execution',title=a['description'],verified=True,artifact_id=a['id']) for a in arts]
def rawsource(id,path,note):return dict(id=id,kind='repository_code',title=path,path=path,sha256=sha(path),verified=True,inspection_note=note)
def authority(id,title,url,ver,note,local):
 p=paper(id,title,url,ver,note);p['original_sha256']=hashlib.sha256(open(local,'rb').read()).hexdigest();return p

def finish(id,summary,variation,sources,claims,arts,details,visual=None,placement=None):
 check=checks(*[(status,detail,ids) for status,detail,ids in details])
 page(id,summary,variation,check,sources+execsources(arts),claims,arts,visual_checks=visual or [],page_visual_check=placement or dict(status='unverified',required=False,details='本頁無圖表，已讀code與行內解釋足夠，無必要額外placement。'))

def viewed_figure(id,figure,fs,name,capturedir,names,observation,placement_observation):
 refs=[dict(path=f'outputs/p7-technical-d/figures/{name}-{w}.png',width=w) for w in [640,360]]
 rec=visual_receipt(name+'-visual',fs,refs)
 visual=[dict(figure=figure,source_sha256=fs,status='verified',details='本人640/360完整工具返回後實看，後續獨立記receipt。',observation=observation,artifacts=refs,receipt=rec)]
 return visual,viewed_placement(id,capturedir,names,placement_observation)

def viewed_placement(id,capturedir,names,observation):
 meta=next(p for p in json.load(open(ROOT/MAN))['pages'] if p['page_id']==id)
 prs=[dict(path=capturedir+'/'+n+'.png',viewport=([1280,800] if n.startswith('1280') else [390,844])) for n in names]
 rec=visual_receipt(id+'-placement',meta['source_sha256'],prs)
 return dict(status='verified',required=True,source_sha256=meta['source_sha256'],details=f'全頁逐段讀完後{len(prs)}PNG兩viewport必要圖表/原句真view完，再另寫receipt；未看部分不算。',observation=observation,artifacts=prs,receipt=rec)
