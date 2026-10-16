import collections,datetime,hashlib,json,pathlib,platform,subprocess
R=pathlib.Path('/workspace/tiny-perceptron-vlm');P=R/'docs/technical-reviews/artifacts/revision-20261016/readme-en-backfill-primary-check.json';H=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
reportp=R/'docs/lesson-products/results/final-report-v1.json';report=json.loads(reportp.read_text());rawp=R/'outputs/lesson-products'/report['predictions']['file'];assert H(rawp)==report['predictions']['sha256'];assert rawp.stat().st_size==report['predictions']['bytes']
counts=collections.defaultdict(lambda:{'count':0,'correct':0,'reverse_same':0})
with rawp.open() as stream:
 for line in stream:
  row=json.loads(line)
  if row['product']=='b' and row['split']=='test':
   key='b/test/'+row['task']+'/loops5';counts[key]['count']+=1;counts[key]['correct']+=row['answers_by_depth']['5']==row['gold']
  elif row['product']=='c' and row['question']=='choice':
   key='c/'+row['split']+'/choice';selected=row.get('selected',row['candidates'][max(range(len(row['raw_scores'])),key=lambda i:row['raw_scores'][i])]);counts[key]['count']+=1;counts[key]['correct']+=selected==row['gold'];counts[key]['reverse_same']+=('reverse_selected' in row and selected==row['reverse_selected'])
expected={'b/test/boxes/loops5':(2956,2928),'b/test/sum/loops5':(42,2),'c/test/choice':(11797,11797)}
for key,(n,k) in expected.items():assert counts[key]['count']==n and counts[key]['correct']==k
assert report['b']['5']['boxes']['correct']==2928 and report['b']['5']['sum']['correct']==2
assert report['c']['choice']['after']['count']==11797==report['c']['choice']['after']['correct']
assert report['shifts']['c_wording']['choice']['after']['count']==11797 and report['shifts']['c_wording']['choice']['after']['correct']==11493
assert counts['c/wording_shift/choice']['correct']==11493
freeze_path=R/'docs/lesson-products/results/settings-freeze-v1.json';freeze=json.loads(freeze_path.read_text());pretest=json.loads((R/'docs/lesson-products/results/pretest-freeze-record-v1.json').read_text());started=json.loads((R/'docs/lesson-products/results/settings-freeze-v1.test-started.json').read_text());assert report['settings_freeze_sha256']==pretest['settings_freeze_sha256']==started['settings_freeze_sha256']==H(freeze_path);assert freeze['recorded_at']<pretest['recorded_at']<started['recorded_at']<report['recorded_at']
commit='821a2c717f05ad7b57e08e88cd965e92ff0c1200';committed=subprocess.check_output(['git','show',commit+':docs/lesson-products/results/settings-freeze-v1.json'],cwd=R);assert hashlib.sha256(committed).hexdigest()==H(freeze_path)
training={}
for product in ['b','c']:
 path=R/f'docs/lesson-products/results/initial-{product}-execution.json';j=json.loads(path.read_text());assert j['gpu_name']=='NVIDIA L4' and j['status']=='completed' and j['training']['random_initialization'] is True
 training[product]={'path':path.relative_to(R).as_posix(),'sha256':H(path),'run_id':j['run_id'],'code_revision':j['code_revision'],'gpu_name':j['gpu_name'],'status':j['status'],'random_initialization':j['training']['random_initialization'],'last_step':j['training']['last_step'],'selected_step':j['training']['selected_step'],'environment':j['training']['environment']}
releasep=R/'tiny_perceptron/lesson_products/release.json';release=json.loads(releasep.read_text());guide=(R/'docs/lesson-products/README.md').read_text();assert release['revision']=='2c6a1062f3e66a602cf83eda6c703456ec18bca0' and release['revision'] in guide
zh=(R/'README.md').read_text();en=(R/'README_en.md').read_text()
for text in [zh,en]:assert '2928/2956' in text and '2/42' in text and 'docs/lesson-products/README.md' in text and 'NVIDIA L4' in text
assert 'three optional branches' not in en and 'a required reasoning-compute unit C' in en
prioren=(R/'docs/course-revision-20261016/auxiliary-sources/6a4d03dde28af07096c4615f5269db4a1c892aa31b7021c3b0dbc9d5f1335c2a.md').read_text().replace('The course has 20 chapters, three optional branches and 312 lesson notebooks.','The course has 20 chapters, the A/B extensions, a required reasoning-compute unit C and 312 lesson notebooks.')
priorzh=(R/'docs/course-revision-20261016/freeze-07/sources/readme.md').read_text()
changes={}
for label,old,new in [('README.md',priorzh,zh),('README_en.md',prioren,en)]:
 oldparas=old.split('\n\n');newparas=new.split('\n\n');assert len(oldparas)==len(newparas);pairs=[{'paragraph_index':i,'before':a,'after':b} for i,(a,b) in enumerate(zip(oldparas,newparas)) if a!=b];assert len(pairs)==1;changes[label]=pairs
result={'command':'.venv/bin/python /workspace/work/curriculum-20261016/readme-review/readme_en_backfill_check.py','reviewer_task':'/root/tech_20261016_readme','current_context':'informed','created_at':datetime.datetime.now(datetime.UTC).isoformat(),'environment':{'python':platform.python_version(),'platform':platform.platform(),'device':'cpu; saved-score arithmetic only'},'result':'pass','original_results':{'report_path':reportp.relative_to(R).as_posix(),'report_sha256':H(reportp),'raw_path':rawp.relative_to(R).as_posix(),'raw_bytes':rawp.stat().st_size,'raw_sha256':H(rawp),'raw_recount':dict(counts),'algorithm':'Stream saved JSONL: product b + split test -> compare answers_by_depth[5] with gold per task; product c + question choice -> test uses saved selected/reverse_selected; wording_shift uses argmax(raw_scores) mapped through that row candidates, then compares with gold; split labels retained separately. No model execution.'},'training_primary_fields':training,'freeze_primary_chain':{'freeze_path':freeze_path.relative_to(R).as_posix(),'freeze_sha256':H(freeze_path),'recorded_at':freeze['recorded_at'],'pretest_recorded_at':pretest['recorded_at'],'test_started_recorded_at':started['recorded_at'],'final_report_recorded_at':report['recorded_at'],'settings_sha_matches_all_three_records':True,'committed_before_test_at':commit,'committed_file_sha_matches':True,'test_outcomes_observed_before_start':pretest['test_outcomes_observed']},'release_primary':{'path':releasep.relative_to(R).as_posix(),'sha256':H(releasep),'repo':release['repo'],'revision':release['revision'],'products':release['products'],'guide_mentions_same_fixed_revision':True},'whole_file_change_comparison':changes,'scope_limits':['Read existing primary receipts and raw scores only; no training or formal-test restart.','No HF download, publication, remote write or new endpoint calls.','This is bilingual synchronization reinspection, not the current canonical README technical review.']}
with P.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({'path':str(P),'sha256':H(P),'result':'pass','raw_recount':dict(counts),'changed_paragraphs':{k:len(v) for k,v in changes.items()},'zh_sha256':H(R/'README.md'),'en_sha256':H(R/'README_en.md')},ensure_ascii=False))
