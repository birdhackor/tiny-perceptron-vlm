from pathlib import Path
import json,hashlib,shutil,re,sys,importlib.util
from datetime import datetime,timezone
repo=Path('/workspace/tiny-perceptron-vlm')
sys.path.insert(0,str(repo))
from scripts import export_course,reading_time
batch=Path('/workspace/work/tutorial-audit-20261008');freeze=batch/'freeze';freeze.mkdir()
idx=json.loads((repo/'course/lesson-index.json').read_text())
inv=reading_time.build_inventory(repo,idx,export_course.DOCUMENTS,export_course.home_introduction(idx,True),freeze/'sources')
source_names=sorted({p['source'] for p in inv['pages']}|{'course/lessons.md','course/lesson-index.json','zensical.toml'})
files={}
for name in source_names:
 target=freeze/'original'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((repo/name).read_bytes());files[name]=hashlib.sha256(target.read_bytes()).hexdigest()
figs={k:v for p in inv['pages'] for k,v in p['figures_sha256'].items()}
for name,digest in figs.items():
 target=freeze/'original'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((repo/name).read_bytes());assert hashlib.sha256(target.read_bytes()).hexdigest()==digest
criteria={}
for name in ['SKILL.md','references/review-protocol.md','references/calibration.md','references/project-context.md','README.md','UPSTREAM.json']:
 target=freeze/'criteria'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((repo/'.agents/skills/clear-tutorial'/name).read_bytes());criteria[name]=hashlib.sha256(target.read_bytes()).hexdigest()
def chapter(pid):
 if pid.startswith('chapter-'):return pid[8:]
 if '.' in pid:return pid.split('.')[0].zfill(2)
 return None
groups={
 'foundations':{'chapters':['01','02','03','04'],'extra':['index','course','first-steps','glossary']},
 'learning':{'chapters':['05','06','07','08','09'],'extra':[]},
 'modalities':{'chapters':['10','11','12','13'],'extra':[]},
 'architecture':{'chapters':['14','15','16','17','18'],'extra':[]},
 'integration':{'chapters':['19','20'],'extra':['natural-v4-student','natural-v4-data','natural-v4-training']},
 'extensions':{'chapters':['0A','0B','0C'],'extra':['training','readme','environment','asset-storage','curriculum','validation','training-assets','publishing']}}
for name,g in groups.items():
 pids=[]
 if name=='foundations':pids=['index','course','first-steps']
 for ch in g['chapters']:
  pids.append('chapter-'+ch)
  pids.extend(p['page_id'] for p in inv['pages'] if p['kind']=='lesson' and chapter(p['page_id'])==ch)
 pids.extend(p for p in g['extra'] if p not in pids)
 g['pages']=pids;g['reviewer']='/root/read_'+name
assert len([p for g in groups.values() for p in g['pages']])==325
assert set(p for g in groups.values() for p in g['pages'])=={p['page_id'] for p in inv['pages']}
manifest={'schema_version':1,'kind':'diagnostic whole-course review, not publication acceptance','created_at':datetime.now(timezone.utc).isoformat(),'repository':str(repo),'baseline_commit':'5224564f52b0bbe04c6958ea2693ad5afd188420','inventory':inv,'original_files_sha256':files,'figures_sha256':figs,'criteria_sha256':criteria,'groups':groups,'restrictions':['Do not edit curriculum, frozen files or historical reviews','Reader initial judgments precede cross-review and historical comparison','Record unseen visuals and unread prerequisites as unverified rather than inferred passes']}
(batch/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
for folder in ['reports','traces','sessions','notes','renders','checks','synthesis']:(batch/folder).mkdir(exist_ok=True)
print(json.dumps({'pages':325,'lesson_sections':287,'numbered_course_sections':313,'source_files':len(files),'figures':len(figs),'groups':{k:len(v['pages']) for k,v in groups.items()}},ensure_ascii=False))
