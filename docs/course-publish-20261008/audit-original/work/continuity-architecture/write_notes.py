from pathlib import Path
import json,datetime
B=Path('/workspace/work/tutorial-audit-20261008')
P={p['page_id']:p for p in json.loads((B/'manifest.json').read_text())['inventory']['pages']}
N=B/'work/continuity-architecture/actual-main-notes.json'
notes=json.loads(N.read_text()) if N.exists() else {'reviewer':'/root/continuity_architecture','role':'continuity','group':'architecture','mode':'full-source continuity review, not blind first read','pages':[], 'issues':[], 'actual_prerequisites':[], 'visual_observations':[]}
def add(pid,quote,judgment,mechanism,need,deps,next_use,unverified=None):
 raw=Path(P[pid]['snapshot']).read_text(); assert quote in raw,(pid,quote)
 assert pid not in [p['page_id'] for p in notes['pages']]
 notes['pages'].append({'page_id':pid,'source_sha256':P[pid]['source_sha256'],'source_judgment':judgment,'quoted_basis':[{'quote':quote,'source':'freeze/sources/'+pid+'.md'}],'checks':{'current_understanding':judgment,'mechanism':mechanism,'need':need,'actual_dependency':deps,'transition_next':next_use},'unverified':unverified or ['未執行程式；不作效果實證驗收','實際站頁runtime outputs與reading annotations未驗']})
def save(): N.write_text(json.dumps(notes,ensure_ascii=False,indent=2)+'\n')
