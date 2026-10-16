"""原owner知情窄審：新guard對真舊Git inputs及獨立合理/共同錯序fixtures。"""
import copy
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
import tempfile
import tomllib
from datetime import UTC,datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
ART=Path(__file__).parent
spec=importlib.util.spec_from_file_location('independent_order_guard',ROOT/'scripts/check_lesson_order.py');guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def ref(name):return {'path':name,'sha256':sha(ROOT/name)}
def toml(value):
 if isinstance(value,str):return json.dumps(value,ensure_ascii=False)
 if isinstance(value,list):return '['+', '.join(toml(item) for item in value)+']'
 if isinstance(value,dict):return '{'+', '.join(json.dumps(key,ensure_ascii=False)+' = '+toml(item) for key,item in value.items())+'}'
 raise TypeError(type(value))


def main():
 start=datetime.now(UTC).isoformat();names=['scripts/check_lesson_order.py','tests/test_lesson_order.py','.github/workflows/pages.yml'];before=[ref(name) for name in names]
 results=[];current=guard.audit(ROOT);assert current['status']=='passed' and not current['failures']
 current_routes=json.loads((ROOT/'course/reading-time-routes.json').read_bytes())['routes'];capstone=next(row for row in current_routes if row['id']=='capstone')
 assert capstone['page_ids']==['chapter-19',*[f'19.{i}' for i in range(1,21)]]
 results.append({'id':'actual_current_positive','passed':True,'result':current,'capstone_keeps_all_20_sections':True})
 workflow=(ROOT/names[2]).read_text();assert workflow.count('scripts/check_lesson_order.py')==1
 assert workflow.index('scripts/check_lesson_order.py')<workflow.index('scripts/build_course.py')<workflow.index('scripts/export_course.py')<workflow.index('actions/upload-pages-artifact')
 assert 'continue-on-error' not in workflow and 'needs: build' in workflow
 old_commit=subprocess.check_output(['git','rev-parse','f8188f88'],cwd=ROOT,text=True).strip()
 with tempfile.TemporaryDirectory(dir=ROOT/'work',prefix='order-guard-audit-') as folder:
  work=Path(folder);oldroot=work/'f8188f88';oldroot.mkdir()
  paths=subprocess.check_output(['git','ls-tree','-r','--name-only',old_commit,'course/chapters'],cwd=ROOT,text=True).splitlines()
  paths=[name for name in paths if name.endswith('.md')]+['course/lesson-index.json','course/reading-time-routes.json','zensical.toml']
  original_files=[]
  for name in paths:
   raw=subprocess.check_output(['git','show',old_commit+':'+name],cwd=ROOT);path=oldroot/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
   original_files.append({'path':name,'sha256':hashlib.sha256(raw).hexdigest()})
  run=subprocess.run([sys.executable,str(ROOT/names[0]),'--root',str(oldroot)],cwd=ROOT,capture_output=True,text=True)
  old_result=json.loads(run.stdout);assert run.returncode==1 and old_result['status']=='failed'
  assert any('course/chapters/14.md' in item for item in old_result['failures']) and any('course/chapters/19.md' in item for item in old_result['failures'])
  results.append({'id':'actual_f8188f88_rejected_by_current_cli','passed':True,'command':'.venv/bin/python scripts/check_lesson_order.py --root <byte-exact f8188f88 scratch>','exit_code':run.returncode,'result':old_result,'input_commit':old_commit,'copied_input_files':original_files})
  baseline={'9':list(range(1,12)),'10':[1,2],'14':list(range(1,19)),'19':list(range(1,21)),'C':list(range(1,8))}
  def fixture(kind):
   destination=work/kind;chapters=copy.deepcopy(baseline)
   if kind=='joint_14_7_at_end':chapters['14'].remove(7);chapters['14'].append(7)
   elif kind=='joint_jump_back':chapters['14']=[1,3,4,2,*range(5,19)]
   elif kind=='joint_lexical_order':chapters['14']=sorted(chapters['14'],key=str)
   elif kind=='joint_duplicate':chapters['14'].append(7)
   elif kind=='joint_missing_internal':chapters['14'].remove(7)
   authored=[]
   for chapter,numbers in chapters.items():
    stem=chapter.zfill(2) if chapter.isdigit() else '0'+chapter
    path=destination/'course/chapters'/f'{stem}.md';path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text('# Explicit synthetic chapter\n\n'+''.join(f'## {chapter}.{n} Explicit fixture heading\n\ntext\n\n' for n in numbers))
    authored.extend(f'{chapter}.{n}' for n in numbers)
   indexed=authored.copy()
   if kind=='index_duplicate':indexed.append('14.7')
   elif kind=='index_missing':indexed.remove('14.7')
   # Lettered unit intentionally spans two navigation groups among numerical chapters.
   nav=[{'九':[f'9.{n}.md' for n in chapters['9']]},{'C前半':['C.1.md','C.2.md']},{'十':['10.1.md','10.2.md']},{'C後半':[f'C.{n}.md' for n in range(3,8)]},{'十四':[f'14.{n}.md' for n in chapters['14']]},{'十九':[f'19.{n}.md' for n in chapters['19']]}]
   if kind=='nav_duplicate':nav[-1]['十九'].append('19.1.md')
   elif kind=='nav_missing':nav[-1]['十九'].remove('19.20.md')
   elif kind=='nav_wrong_14':nav[-2]['十四'][6],nav[-2]['十四'][7]=nav[-2]['十四'][7],nav[-2]['十四'][6]
   capstone=['chapter-19',*[f'19.{n}' for n in chapters['19']]]
   if kind=='capstone_missing_last':capstone.remove('19.20')
   elif kind=='capstone_reordered':capstone[1],capstone[2]=capstone[2],capstone[1]
   elif kind=='capstone_no_overview':capstone.remove('chapter-19')
   routes=[] if kind=='capstone_missing_route' else [{'id':'capstone','page_ids':capstone}]
   (destination/'course/lesson-index.json').write_text(json.dumps([{'id':item} for item in indexed]))
   (destination/'course/reading-time-routes.json').write_text(json.dumps({'routes':routes}))
   (destination/'zensical.toml').write_text('[project]\nnav = '+toml(nav)+'\n')
   return destination
  base=guard.audit(fixture('reasonable_numeric_and_grouped_positive'));assert base['status']=='passed'
  results.append({'id':'reasonable_numeric_and_grouped_positive','passed':True,'result':base,'covers':['14.7 in continuous1..18','14.9 ->14.10','9.9 ->9.10','chapter9->10 navigation','C.1..7 across nested navigation groups','all19.1..20 capstone preserved']})
  for kind in ['joint_14_7_at_end','joint_jump_back','joint_lexical_order','joint_duplicate','joint_missing_internal','index_duplicate','index_missing','nav_duplicate','nav_missing','nav_wrong_14','capstone_missing_last','capstone_reordered','capstone_no_overview','capstone_missing_route']:
   observed=guard.audit(fixture(kind));assert observed['status']=='failed' and observed['failures'],kind
   results.append({'id':kind,'valid_base_passed':True,'passed':True,'result':observed,'shared_authored_index_nav_error':kind.startswith('joint_')})
 tools=json.loads((ROOT/'docs/course-revision-20261016/reviews/tool-audit-callback-04.json').read_bytes());toolmap={item['path']:item['sha256'] for item in tools['checked_files']}
 core=['docs/review-tools/incremental_revision.py','docs/review-tools/baseline_replay.py','docs/review-tools/phase7_review.py','docs/course-revision-20261016/tools/reader.py','docs/course-revision-20261016/tools/freeze_sources.py','tests/test_incremental_revision_review.py']
 for name in core:assert sha(ROOT/name)==toolmap[name]
 assert before==[ref(name) for name in names]
 report={'reviewer_task':'/root/tech_20261016_review_tools','initial_context':'informed','current_context':'informed','started_at':start,'completed_at':datetime.now(UTC).isoformat(),'environment':{'executable':sys.executable,'python':sys.version,'platform':platform.platform()},'checked_files':before,'results':results,'all_expected':True,'pages_workflow_order_guard_precedes_build_export_upload':True,'core6_unchanged_from_audit04':[ref(name) for name in core],'limits':['Guard scope is intra-chapter contiguous numeric section order and full relative inventory against authored sections, with capstone route matching chapter19; it does not fix historical cardinality or impose global chapter ordering across permitted interleaved lettered groups.','No baseline replay, fresh reader/technical course review, model/formaltest/paid/HF/training/cloud or Git write. Scratch old source inputs are local byte-exact Git reads and removed after this probe.']}
 path=ART/'order-guard-technical-probes-results-01.json'
 with path.open('x') as stream:json.dump(report,stream,ensure_ascii=False,indent=2);stream.write('\n')
 print(json.dumps({'cases':len(results),'all_expected':True,'old_commit':old_commit,'current':current,'receipt_sha256':sha(path)}))


if __name__=='__main__':main()
