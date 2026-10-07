"""Only store the owner's supplied observations after actual view tool results."""
import json
from datetime import datetime,timezone
from pathlib import Path
from schema_helpers import BASE,digest

def receipt(filename,source_sha,paths,observation,widths=None):
 r={'tool':'tools.view_image','reviewer_task':'/root/p7_technical_a','received_at':datetime.now(timezone.utc).isoformat(),'source_sha256':source_sha,'artifacts':[],'observation':observation}
 for i,p in enumerate(paths):
  a={'path':p,'sha256':digest(p)}
  if widths:a['width']=widths[i]
  else:a['viewport']=[1280,800] if Path(p).name.startswith('1280') else [390,844]
  r['artifacts'].append(a)
 p=BASE+filename
 if Path(p).exists():raise RuntimeError('no overwrite')
 Path(p).write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');return p,r

def visual(figure,path,r,details):
 return {'figure':figure,'source_sha256':r['source_sha256'],'status':'verified','required':True,'details':details,'observation':r['observation'],'receipt':{'path':path,'sha256':digest(path)},'artifacts':r['artifacts']}
