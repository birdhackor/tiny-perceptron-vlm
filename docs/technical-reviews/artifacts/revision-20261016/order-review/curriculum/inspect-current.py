"""Narrow informed callback: inspect current curriculum order and carried evidence."""
import difflib
import hashlib
import json
import platform
import sys
import tomllib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[6]
sys.path.insert(0,str(ROOT))
from scripts.reading_time import lesson_slices,chapter_introduction
from scripts.check_lesson_order import navigation_pages
ART=Path(__file__).parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads((ROOT/p).read_text())
oldpath='docs/course-revision-20261016/reviews/technical-curriculum-backfill-original-01.json'
old=read(oldpath)
freeze=read('docs/course-revision-20261016/freeze-18.json')
oldfreeze=read('docs/course-revision-20261016/freeze-16.json')
pages={p['page_id']:p for p in freeze['pages']}
oldpages={p['page_id']:p for p in oldfreeze['pages']}
page=pages['curriculum'];body=(ROOT/page['snapshot']).read_text()
previous=(ROOT/'docs/course-revision-20261016/freeze-11/sources/curriculum.md').read_text()
a=previous.splitlines(True);b=body.splitlines(True)
diff=list(difflib.unified_diff(a,b,fromfile='original owner freeze14',tofile='callback freeze18'))
changes=[x for x in difflib.SequenceMatcher(None,a,b).get_opcodes() if x[0]!='equal']
assert len(changes)==1 and changes[0][2]-changes[0][1]==changes[0][4]-changes[0][3]==1
assert body==(ROOT/'docs/curriculum.md').read_text()
assert sha(ROOT/page['snapshot'])==page['source_sha256']
chapters={p.name:lesson_slices(p.read_text()) for p in sorted((ROOT/'course/chapters').glob('*.md'))}
nav=navigation_pages(tomllib.loads((ROOT/'zensical.toml').read_text())['project']['nav'])
index=read('course/lesson-index.json')
indexed=[r['id'] for r in index]
orders={}
for name,n,prefix in [('14.md',18,'14'),('19.md',20,'19')]:
 expected=[f'{prefix}.{i}' for i in range(1,n+1)]
 actuals={'authored':list(chapters[name]),'indexed':[i for i in indexed if i.startswith(prefix+'.')],'navigation':[i for i in nav if i.startswith(prefix+'.')]}
 assert all(x==expected for x in actuals.values())
 orders[prefix]=actuals
assert nav.index('7.25')<nav.index('C.1')<nav.index('C.5')<nav.index('8.1')
assert nav.index('13.17')<nav.index('C.6')<nav.index('C.7')<nav.index('14.1')
routes=read('course/reading-time-routes.json')['routes']
capstone=next(r['page_ids'] for r in routes if r['id']=='capstone')
assert capstone==['chapter-19']+[f'19.{n}' for n in range(1,21)]
guide=(ROOT/pages['course']['snapshot']).read_text()
assert guide==(ROOT/'course/README.md').read_text()
r3=lesson_slices(guide)['R.3']
assert '14.7–14.10' in r3 and '再讀14.11–14.18' in r3
assert '19.1–19.12' in r3 and '再讀19.13–19.20' in r3
repo_checks={}
for s in old['sources']:
 if s['kind']=='repository_code':
  current=sha(ROOT/s['path']);repo_checks[s['id']]={'path':s['path'],'original_sha256':s['sha256'],'current_sha256':current,'same':current==s['sha256']}
assert all(x['same'] for x in repo_checks.values())
artifact_checks={}
for a in old['artifacts']:
 current=sha(ROOT/a['path']);artifact_checks[a['id']]={'path':a['path'],'original_sha256':a['sha256'],'current_sha256':current,'same':current==a['sha256']}
assert all(x['same'] for x in artifact_checks.values())
oldaudit=read('docs/technical-reviews/artifacts/revision-20261016/backfill/curriculum-audit-output.json')
oldstatic=read('docs/technical-reviews/artifacts/revision-20261016/backfill/curriculum-static-check-output.json')
inputchecks={}
for path,digest in {**oldaudit['source_sha256'],**oldstatic['source_sha256']}.items():
 current=sha(ROOT/path);inputchecks[path]={'original_sha256':digest,'current_sha256':current,'same':current==digest}
assert all(v['same'] for p,v in inputchecks.items() if p not in ['scripts/export_course.py','scripts/build_course.py'])
lessons=[p for p in pages.values() if p['kind']=='lesson']
lesson_deltas={p['page_id']:{'old':oldpages[p['page_id']]['source_sha256'],'current':p['source_sha256']} for p in lessons if p['source_sha256']!=oldpages[p['page_id']]['source_sha256']}
assert set(lesson_deltas)=={'19.12'}
nb_deltas={p['page_id']:{'old':oldpages[p['page_id']]['notebook_sha256'],'current':p['notebook_sha256']} for p in lessons if p['notebook_sha256']!=oldpages[p['page_id']]['notebook_sha256']}
assert set(nb_deltas)=={'19.12'}
# Save only small needed excerpts; not a technical verdict for these whole lessons.
dependencies={}
for id in ['14.6','14.7','14.10','14.11','19.12','19.13','19.14']:
 p=pages[id];snapshot=(ROOT/p['snapshot']).read_text()
 live=chapters[Path(p['source']).name][id]
 assert live==snapshot and sha(ROOT/p['snapshot'])==p['source_sha256']
 dependencies[id]={'snapshot':p['snapshot'],'sha256':p['source_sha256'],'inspection_scope':'Opening problem and route prerequisite; plus end transition for19.12; not full lesson re-review','opening_excerpt':snapshot[:450]}
 if id=='19.12':dependencies[id]['transition_excerpt']=snapshot[snapshot.rfind('\n這個有限成品'):].strip()
figure_checks={p:sha(ROOT/p)==h for p,h in old['figure_sha256'].items()}
assert all(figure_checks.values()) and old['figure_sha256']==page['figures_sha256']
front=sum(len(lesson_slices((ROOT/'course'/p).read_text())) for p in ['README.md','first-steps.md','training.md','glossary.md'])
current_source_paths=['docs/curriculum.md','course/README.md','course/lesson-index.json','course/reading-time-routes.json','zensical.toml','scripts/build_course.py','scripts/export_course.py','scripts/check_lesson_order.py','pyproject.toml']
out={'reviewer_task':'/root/backfill_tech_curriculum','current_context':'informed','environment':{'python':platform.python_version(),'device':'cpu','operation':'read-only hash/diff/route parsing, not model execution'},
 'scope':'Current entire curriculum reread; narrow route callback with original owner evidence carried only where unchanged. No reader/continuity conclusions read, no network fetch or model/test/Colab rerun.',
 'original_report':{'path':oldpath,'sha256':sha(ROOT/oldpath)},'freeze':{'path':'docs/course-revision-20261016/freeze-18.json','sha256':sha(ROOT/'docs/course-revision-20261016/freeze-18.json')},
 'current_page':page,'same_as_live':True,'single_paragraph_change':True,'diff':diff,'orders':orders,'C_navigation_order_verified':True,'capstone_route':capstone,
 'counts':{'chapter_files':len(chapters),'chapter_lessons':sum(len(v) for v in chapters.values()),'notebooks':len(list((ROOT/'notebooks').glob('*/*.ipynb'))),'numbered_including_front':sum(len(v) for v in chapters.values())+front},
 'guide_R3_snapshot':pages['course']['snapshot'],'guide_R3_sha256':hashlib.sha256(r3.encode()).hexdigest(),'guide_R3_route_excerpt':r3.splitlines()[2],
 'repository_sources_carried':repo_checks,'original_artifacts_carried':artifact_checks,'original_execution_input_pins':inputchecks,
 'freeze16_to18_lesson_source_changes':lesson_deltas,'freeze16_to18_notebook_changes':nb_deltas,'needed_dependencies':dependencies,'unchanged_figures':figure_checks,
 'current_source_sha256':{p:sha(ROOT/p) for p in current_source_paths},
 'actual_prior_tool_runs':[{'command':'.venv/bin/python scripts/check_lesson_order.py','exit_code':0,'result':{'status':'passed','chapters':23,'lessons':312,'failures':[]}}, {'command':'.venv/bin/python scripts/build_course.py --check','exit_code':0,'stdout':'312 個小節：同步檢查通過\n'}],
 'interpretation':{'c15':'Replaces original now-deleted25-new-sections/historical-order wording with verified current C/14/19 route contract. It does not relabel the earlier route check as a check of this new order.', 'c01':'Current source counts rechecked.', 'c32':'Current generator/exporter relevant sections and synchronization rechecked; original old exporter receipt is historical, not current-source proof.', 'other_claims':'Original39 non-route statements remain byte-identical on the assigned page. Sources/artifacts and measured implementation/data inputs are unchanged, so original evidence can be inherited without retraining or formal retesting.', 'dependency_effect':'14.6 remains prior to14.11, including intervening position lessons.19.12 now sends readers to19.13 purposes and19.14common split; current19.13/19.14 openings supply those stated roles. No model capability or numerical threshold is strengthened by ordering.'}}
(ART/'current-inspection.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'counts':out['counts'],'chapter14':orders['14']['authored'],'chapter19':orders['19']['authored'],'all_old_repo_source_pins_match':all(x['same'] for x in repo_checks.values()),'all_old_artifacts_match':all(x['same'] for x in artifact_checks.values()),'changed_original_execution_inputs':[p for p,v in inputchecks.items() if not v['same']],'freeze16_to18_lesson_changes':list(lesson_deltas),'status':'passed'},ensure_ascii=False,indent=2))
