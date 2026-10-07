"""Mechanical report formatting; all judgments, evidence and verification text supplied by owner."""
import json
from schema_helpers import *
def attention():
 return dict(json.load(open(BASE+'pages/chapter-03.json'))['sources'][0])
def cpu(page,result):
 return execution('cpu',BASE+'cpu/'+page+'/execution.json','.venv/bin/python '+BASE+'run-read-page-cpu.py '+page,result,'Python3.13.5;torch2.14.1+cpu;CPU','全页原句全读之后实际逐块执行')
def var(path,command,result):
 return execution('variation',BASE+path,command,result,'Python3.13.5;torch2.14.1+cpu;CPU','本人必要CPU变体/核算，非长训练')
def ver(expected,observed,details,tolerance=None):
 d={'method':'executed','expected':expected,'observed':observed,'details':details}
 if tolerance is not None:d['tolerance']=tolerance
 return d
def emit(page,summary,claims,artifacts,sources,checks,variation,missing,visuals=None,placement_check=None,issues=None,verdict='pass'):
 save({'page_id':page,'verdict':verdict,'summary':summary,'issues':issues or [],'missing_visuals':missing,'visual_checks':visuals or [],'page_visual_check':placement_check or {'status':'unverified','required':False,'details':'本页没有需要核对的图表/位置，未做网页观看。'},'variation':variation,'artifacts':artifacts,'sources':sources,'claims':claims,'checks':checks})
