"""Independent README evidence inspection. Reads preserved bytes only; no model inference/training."""
import collections, datetime, gzip, hashlib, json, math, re, sys, zipfile
from pathlib import Path
ROOT=Path('/workspace/work/omni-book-integrated-v1')
OUT=ROOT/'docs/technical-reviews/artifacts/omni-P5-readme'
RAW=Path('/workspace/work/omni-P5-readme-independent-raw')
FINAL=Path('/workspace/work/omni-final-completed-41004229262/verified')
CORE=Path('/workspace/work/omni-core-completed-40988311100/verified')
def digest(p):
 p=Path(p);h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return {'path':str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),'bytes':p.stat().st_size,'sha256':h.hexdigest()}
def read(p):return json.loads(Path(p).read_text())
proof={'reviewer_task':'/root/omni_p5_technical_readme','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Independent pure byte/data recomputation; zero model loads/forwards, training or retuning. External absolute paths are input provenance only; all conclusions/receipts persist here.'}
freeze=read(ROOT/'docs/course-revision-20261029-omni/reviews/P5-actual/source-freeze-v10.json')
p=next(p for p in freeze['pages'] if p['page_id']=='readme')
proof['frozen_page']=p
proof['frozen_page_checks']={x:digest(ROOT/x) for x in ['README.md',p['snapshot'],*p['figures_sha256']]}
assert all(v['sha256']==p['source_sha256'] for k,v in proof['frozen_page_checks'].items() if k.endswith('md'))
chapters=list((ROOT/'course/chapters').glob('*.md'))
sections={q.name:re.findall(r'^## ([A-Z0-9]+\.\d+) ',q.read_text(),re.M) for q in chapters}
guides={q:re.findall(r'^## ([A-Z0-9]+\.\d+) ',(ROOT/q).read_text(),re.M) for q in ['course/README.md','course/first-steps.md','course/training.md','course/glossary.md']}
proof['counts']={'chapter_files':len(chapters),'chapter_section_counts':{k:len(v) for k,v in sections.items()},'chapter_lessons':sum(map(len,sections.values())),'guide_counts':{k:len(v) for k,v in guides.items()},'guide_total':sum(map(len,guides.values())),'canonical_counts':freeze['counts']}
proof['counts']['total_numbered']=proof['counts']['chapter_lessons']+proof['counts']['guide_total']
assert proof['counts']['chapter_lessons']==331 and proof['counts']['guide_total']==26 and proof['counts']['total_numbered']==357
new=['7.26','9.17','9.18','10.12','12.17','12.18','14.19','16.15','16.16','17.16','17.17','19.21','19.22','19.23','19.24','20.14','20.15','20.16']
proof['new18_headings']={i:next(line for q in chapters for line in q.read_text().splitlines() if line.startswith('## '+i+' ')) for i in new}
text=(ROOT/'README.md').read_text();links=[]
for label,target in re.findall(r'\[([^\]]+)\]\(([^)]+)\)',text):
 if '://' in target:continue
 path,_,anchor=target.partition('#');q=ROOT/path
 row={'label':label,'target':target,'file_exists':q.is_file()}
 if anchor:row['numbered_anchor_exists']=bool(re.search(r'^## '+re.escape(anchor)+r' ',q.read_text(),re.M))
 links.append(row)
assert all(r['file_exists'] and r.get('numbered_anchor_exists',True) for r in links)
proof['local_links']=links
# Exact current code and cell-type/source identity, independent of stored output metadata.
receipt=ROOT/'outputs/omni-P5-current-kernel-bundle-v9-receipt.json';r=read(receipt);checks=[]
for row in r['rows']:
 src=ROOT/row['source']['path'];executed=ROOT/row['joined']['path'];a,b=read(src),read(executed)
 cells=lambda n:[(c['cell_type'],''.join(c['source'])) for c in n['cells']]
 code=[c for c in b['cells'] if c['cell_type']=='code'];errs=[o for c in code for o in c.get('outputs',[]) if o.get('output_type')=='error']
 checks.append({'id':row['id'],'same_cells':cells(a)==cells(b),'source_sha256_valid':digest(src)['sha256']==row['source']['sha256'],'executed_sha256_valid':digest(executed)['sha256']==row['joined']['sha256'],'all_code_executed':all(isinstance(c.get('execution_count'),int) for c in code),'error_count':len(errs)})
proof['notebook_outputs']={'receipt':digest(receipt),'total':len(checks),'all_cells_same':all(x['same_cells'] for x in checks),'all_hashes_same':all(x['source_sha256_valid'] and x['executed_sha256_valid'] for x in checks),'all_executed':all(x['all_code_executed'] for x in checks),'error_count':sum(x['error_count'] for x in checks),'fresh_current_kernels':r['current_fresh_revised_lessons'],'earlier_exact_source_executions':r['earlier_exact_source_executions'],'rows':checks}
assert len(checks)==331 and proof['notebook_outputs']['all_cells_same'] and proof['notebook_outputs']['all_hashes_same'] and proof['notebook_outputs']['error_count']==0
# Raw current final archive and source selection, read as data only.
archive=FINAL.parent/'final-results.zip';proof['final_archive']=digest(archive)
assert proof['final_archive']['sha256']=='50b11a952e9ceb57eafa00891b710f44157a17adb742b69abad61655365c8de3'
selected=read(FINAL/'frozen/model-selection.json');proof['selection']={k:selected[k] for k in ['selected','temperature','threshold','final_opened']}
for part in ['av','sound','choice','integration']:
 f=FINAL/f'measurements/{part}/results.json'; completion=read(f.with_name('part-completion.json'))
 assert digest(f)['sha256']==completion['results_sha256']
proof['final_input_receipts']=[digest(FINAL/f'measurements/{part}/{name}') for part in ['av','sound','choice','integration'] for name in ['results.json','part-origin.json','part-completion.json']]
av=read(FINAL/'measurements/av/results.json')['models'][selected['selected']['av']];rows=av['records'];pair_correct=sum(x['answers']==x['gold'] for x in rows);swap_correct=sum(x['waveform_swap_answers']==x['waveform_swap_gold'] for x in rows)
proof['av']={'selected':selected['selected']['av'],'families':len(rows),'unique_families':len({x['family'] for x in rows}),'observations':sum(len(x['answers']) for x in rows),'paired_correct_recomputed':pair_correct,'waveform_swap_paired_correct_recomputed':swap_correct,'pair_flags_agree':all(x['paired_correct']==(x['answers']==x['gold']) for x in rows),'source_summary_agrees':pair_correct==av['primary']['correct'] and swap_correct==av['same_metadata_waveform_swap']['paired']['correct'],'first_raw_row':rows[0]}
assert pair_correct==200 and swap_correct==148
s=read(FINAL/'measurements/sound/results.json')['codec_and_gold_program'];rows=s['rows'];ok=sum(x.get('generation',{}).get('program_success',False) for x in rows)
proof['sound']={'families':len(rows),'program_successes':ok,'sample_rate':8000,'program_success_conditions':'EOS and not capped and program_success waveform analysis; matches core_evaluation.generated_sound, not WAV existence','event_count':dict(collections.Counter(len(x['program']) for x in rows)),'frequency_values':sorted({e['frequency'] for x in rows for e in x['program']}),'every_saved_analysis_agrees':all(x['generation']['program_success']==(x['generation']['eos'] and not x['generation']['capped'] and x['generation']['analysis']['success']) for x in rows),'summary_agrees':ok==s['gold_program']['correct']}
assert ok==200
c=read(FINAL/'measurements/choice/results.json')['primary'];rows=c['rows'];T=selected['temperature'];tau=selected['threshold'];calc=[]
for r in rows:
 ch=r['tasks']['choice'];info=r['choice_info'];logits=ch['logits'];top=min(range(len(logits)),key=lambda j:(-logits[j],ch['ids'][j]));answer=ch['ids'][top];mx=max(logits);exps=[math.exp((x-mx)/T) for x in logits];pmax=max(exps)/sum(exps);ii=min(range(len(info['logits'])),key=lambda j:(-info['logits'][j],info['ids'][j]));accepted=pmax>=tau and answer!='none' and info['ids'][ii]=='yes';err=accepted and (answer!=ch['gold'] or not r['world_supported'] or not r['choice_information_sufficient']);calc.append({'family':r['family'],'answer':answer,'gold':ch['gold'],'accepted':accepted,'error':err,'p_max':pmax})
proof['choice']={'families':len(rows),'unique_families':len({r['family'] for r in rows}),'correct':sum(r['answer']==r['gold'] for r in calc),'accepted':sum(r['accepted'] for r in calc),'errors_among_accepted':sum(r['error'] for r in calc),'T':T,'tau':tau,'rule':'argmax raw logits (stable semantic ID tie); max softmax(logits/T)>=tau AND choice != none AND same-object info argmax yes. Gold only used for scoring after acceptance.','all_saved_rows_match':all(a['answer']==b['answer'] and a['accepted']==b['accepted'] and a['error']==b['error'] and abs(a['p_max']-b['p_max'])<1e-12 for a,b in zip(calc,c['policy']['rows'])),'pure_cpu_recomputed_rows':calc}
assert (proof['choice']['correct'],proof['choice']['accepted'],proof['choice']['errors_among_accepted'])==(39,53,19)
core=[]
for j in sorted((CORE/'core').iterdir()):
 result=j/'job-result.json';train=j/'training/training-result.json';log=j/'training/training-log.json';d=read(result);t=read(train)
 core.append({'job':j.name,'job_receipt':digest(result),'training_receipt':digest(train),'training_log':digest(log),'success':d['success'],'runtime':d['runtime'],'execution_host':d['execution_host'],'complete':t['complete'],'updates':t['updates'],'optimizer_updates_observed':t['optimizer_updates_observed'],'different_parameter_digest':t['initial_parameter_sha256']!=t['latest_parameter_sha256'],'best':t['best'],'final_opened':t['final_opened']})
proof['core_training']=core
assert len(core)==10 and all(x['success'] and x['complete'] and x['updates']>0 and x['optimizer_updates_observed'] and x['different_parameter_digest'] and x['runtime']['device']=='cpu' and not x['final_opened'] for x in core)
# V2 data counts and retained original producer receipts, no reviewer verdicts used.
m=read(ROOT/'docs/selftrained/v2-manifest.json');proof['v2_data']={'manifest':digest(ROOT/'docs/selftrained/v2-manifest.json'),'package':m['package'],'files':len(m['records'])+len(m['assets']),'data_records_files':m['records']}
assert proof['v2_data']['files']==8962
for arch in ['moe','dense']:
 root=ROOT/f'docs/selftrained/results/public-raw/{arch}/test';metrics=read(root/'metrics.json');receipt=read(root/'receipt.json');journal=read(root/'evaluation-receipt.json')
 proof.setdefault('v2_final',{})[arch]={'metrics':digest(root/'metrics.json'),'receipt':digest(root/'receipt.json'),'journal':digest(root/'evaluation-receipt.json'),'count':metrics['count'],'expected_count':metrics['expected_count'],'evaluation_complete':metrics['evaluation_complete'],'run_id':receipt['run_id'],'revision':receipt['revision'],'command':receipt['command'],'returncode':receipt['returncode'],'journal_completed':journal['completed_count'],'journal_output_digest':journal['outputs_sha256'],'thresholds':metrics['thresholds'],'tool_roundtrip':metrics['per_task_final_reply']['tool_call']['tool_roundtrip'],'voice_qa':metrics['per_task_final_reply']['voice_qa']['semantic']}
 assert metrics['count']==3734 and journal['completed_count']==3734 and metrics['evaluation_complete']
# Pure replay preserved old B/C rows (compressed public data, not neural forward).
gz=Path('/workspace/work/reference/omni-technical-19-17-v1/raw/final-report-v1.predictions.jsonl.gz');h=hashlib.sha256();counts=collections.Counter();sample_c=None;rows_n=0
with gzip.open(gz,'rb') as f:
 for raw in f:
  h.update(raw);r=json.loads(raw);rows_n+=1
  if r['product']=='b' and r['split']=='test':
   key='b5_'+r['task'];counts[key+'_n']+=1;counts[key+'_ok']+=r['answers_by_depth']['5']==r['gold']
  elif r['product']=='c':
   label='c_'+r['split']+'_'+r['question'];counts[label+'_n']+=1;scores=r['raw_scores'];pred=r['candidates'][max(range(len(scores)),key=lambda j:scores[j])];counts[label+'_ok']+=pred==r['gold']
   if sample_c is None:sample_c={k:v for k,v in r.items() if k not in ['logits','scores','probabilities']}
report=read(ROOT/'docs/lesson-products/results/final-report-v1.json');proof['old_bc']={'compressed_input':digest(gz),'decompressed_sha256':h.hexdigest(),'manifest_decompressed_sha256':report['predictions']['sha256'],'raw_rows':rows_n,'counts':dict(counts),'sample_c_metadata':str(sample_c)[:1600],'freeze_recorded_at':read(ROOT/'docs/lesson-products/results/settings-freeze-v1.json')['recorded_at'],'test_recorded_at':report['recorded_at'],'old_C_choice':report['c']['choice']['after'],'wording_shift_choice':report['shifts']['c_wording']['choice']['after']}
assert h.hexdigest()==report['predictions']['sha256'] and counts['b5_boxes_n']==2956 and counts['b5_boxes_ok']==2928 and counts['b5_sum_n']==42 and counts['b5_sum_ok']==2
# Public list counts and the one original command's preserved stdout; do not rerun its model.
models=read(ROOT/'docs/course-experiments/public-models.json')['models'];caps=read(ROOT/'docs/course-experiments/capstone-public.json')['models'];proof['public_counts']={'experiment_groups':len(models),'weight_files':sum(Path(x['output']).suffix in {'.pt','.pth','.ckpt','.safetensors'} for s in models for x in s['files']),'capstone_entries':len(caps)}
assert proof['public_counts']=={'experiment_groups':30,'weight_files':120,'capstone_entries':11}
oldp=ROOT/'outputs/omni-P5-readme-old-public-text-foundation-generated-record-v1.json';old=read(oldp)
proof['original_text_foundation_command']={'original_receipt':digest(oldp),'model':old['model'],'checked_at':old['checked_at'],'public_source':old['public_source'],'anonymous_download':old['anonymous_download'],'environment':old['environment'],'status':old['status'],'commands':[{k:v for k,v in r.items() if k in ['command','returncode','status']} for r in old['commands']],'last_output':old['commands'][-1]['output'],'current_source_hashes_agree':all(digest(ROOT/p)['sha256']==h for p,h in old['source_hashes'].items()),'current_public_manifest_digest_agrees':digest(ROOT/'docs/course-experiments/public-models.json')['sha256']==old['manifest_sha256']}
assert proof['original_text_foundation_command']['current_source_hashes_agree'] and proof['original_text_foundation_command']['last_output']['answer']=='side=right.'
# Independent provider API readbacks gathered by GitHub plugin and HF unsigned metadata GET.
proof['providers']={}
for name in ['core-run-provider.json','final-run-provider.json','core-release-provider.json','final-release-provider.json']:
 q=RAW/name;d=read(q);is_release='assets' in d
 proof['providers'][name]={'decoded_content_receipt':digest(q),'fields':{k:d[k] for k in (['id','tag_name','target_commitish','draft','prerelease','published_at'] if is_release else ['id','head_sha','status','conclusion','created_at','updated_at'])}}
 if is_release:proof['providers'][name]['assets']=[{k:a.get(k) for k in ['id','name','size','digest','browser_download_url']} for a in d['assets']]
proof['hf_tree']={}
for name in ['hf-v2-tree','hf-bc-tree']:
 q=RAW/(name+'.raw');d=read(q);files=[x for x in d if x['type']=='file'];proof['hf_tree'][name]={'receipt':digest(q),'file_count':len(files),'files':files}
assert proof['hf_tree']['hf-v2-tree']['file_count']==16 and proof['hf_tree']['hf-bc-tree']['file_count']==32
mp=RAW/'macos-job-111052346622-decoded.log';txt=mp.read_text();proof['mps_original_log']={'decoded_log':digest(mp),'url':'https://github.com/birdhackor/tiny-perceptron-vlm/actions/runs/37071683289/job/111052346622','inspected_excerpts':[line for line in txt.splitlines() if re.search(r'torch\s+2|device\s+Apple|compute\s+OK|tests/test_env.py|330 passed',line)],'current_tests_sha256':digest(ROOT/'tests/test_env.py')['sha256'],'recorded_tests_sha256':read(ROOT/'docs/validation-artifacts/release-compatibility-ci.json')['source_sha256']['tests/test_env.py'],'scope':'Producer log confirms MPS basic environment forward/backward and all seven env tests passed; test_accelerator_matches_cpu is 32x32 allclose rtol1e-3 atol1e-2; not whole-course MPS or model training.'}
used=['README.md','course/figures/rewrite-01-character-ids.svg','pyproject.toml','uv.lock','.python-version','scripts/check_env.py','scripts/train.py','scripts/prepare_data.py','tiny_perceptron/model.py','tiny_perceptron/training.py','tiny_perceptron/assets.py','scripts/fetch_course_models.py','scripts/check_course_models.py','scripts/fetch_training_assets.py','scripts/omni_revision/core_evaluation.py','scripts/omni_revision/core_final.py','tiny_perceptron/omni_products/reliability.py','tiny_perceptron/omni_products/signal_metrics.py','tiny_perceptron/omni_products/data.py','tiny_perceptron/omni_products/training_runtime.py','docs/omni-products/v1/model-selection.json','docs/omni-products/v1/metrics.json','docs/omni-products/v1/public-release.json','docs/selftrained/v2-manifest.json','docs/selftrained/results/v2-final-public-results.json','docs/lesson-products/results/final-report-v1.json','docs/lesson-products/results/settings-freeze-v1.json','docs/course-experiments/public-models.json','docs/course-experiments/capstone-public.json','zensical.toml','scripts/export_course.py','scripts/check_notebooks.py','scripts/build_course.py','scripts/format_course_code.py','scripts/build_visuals.py','.github/workflows/ci.yml','LICENSE','tests/test_env.py']
proof['repository_source_receipts']=[digest(ROOT/p) for p in used]
path=OUT/'data-recompute-v1.json'
with path.open('x') as f:json.dump(proof,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({'artifact':digest(path),'counts':proof['counts'],'av':{k:v for k,v in proof['av'].items() if k!='first_raw_row'},'sound':proof['sound'],'choice':{k:v for k,v in proof['choice'].items() if k!='pure_cpu_recomputed_rows'},'old_bc_counts':dict(counts),'old_C_row_sample':proof['old_bc']['sample_c_metadata'],'notebook_summary':{k:v for k,v in proof['notebook_outputs'].items() if k!='rows'}},ensure_ascii=False,indent=2))
