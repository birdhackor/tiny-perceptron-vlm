from pathlib import Path
import hashlib,json,sys,platform,datetime,importlib.util,subprocess,tempfile
R=Path.cwd();A=Path(__file__).resolve().parent;C=Path('/tmp/g4-factual-20261005');h=lambda b:hashlib.sha256(b).hexdigest()
initial=json.loads((A/'report-initial-revise.json').read_bytes());assert h((A/'report-initial-revise.json').read_bytes())=='fbd88dfb7fd8fc8ff543d9e9637df4def74fcffe74d9d9b97d5f1ab3f78bc060'
s=importlib.util.spec_from_file_location('sf',R/'docs/review-tools/section_facts.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
body,whole,line=m.original_section(R/'course/glossary.md','G.4');assert body==(A/'inputs/G.4-revised.md').read_bytes();assert h(body)=='be13f56ac03b3193904d9d876df973671cfe71c0666bd95270ef15f5a1699503'
old=(A/'inputs/G.4.md').read_bytes();before='保留後訓練起點的模型副本，用來比較回答傾向偏移多少';after='保留這次偏好或強化學習訓練起點的模型副本，用來比較回答傾向偏移多少';assert old.decode().replace(before,after).encode()==body
versions=[]
for claim in initial['claims']:
 sources=[]
 for reference in claim['evidence']:
  source=next(x for x in initial['sources'] if x['id']==reference['source_id']);snapshots=[]
  for aid in source.get('artifact_ids',[]):
   a=next(x for x in initial['artifacts'] if x['id']==aid);assert h((R/a['path']).read_bytes())==a['sha256'];snapshots.append({'id':aid,'path':a['path'],'sha256':a['sha256']})
  if source['kind']=='repository_code':assert h((R/source['path']).read_bytes())==source['sha256']
  sources.append({'source_id':source['id'],'version':source.get('version','execution'),'locator':reference['locator'],'snapshots':snapshots})
 for aid in claim['artifact_ids']:
  a=next(x for x in initial['artifacts'] if x['id']==aid);assert h((R/a['path']).read_bytes())==a['sha256']
 versions.append({'claim_id':claim['id'],'unchanged_original_evidence_verified':True,'sources':sources})
for a in initial['artifacts']:assert h((R/a['path']).read_bytes())==a['sha256']
# Reverify original authority bytes and the original source text personally re-read this turn.
for name,digest in [('dpo','92cb3a2b71362acda98a789b03d88688fd33cf5fcf13f81d2b1de30ee7d3b67a'),('instructgpt','c1984bb50a5b90fddb895fdc3a0f72e5bc977148c9f63ef6040cbe7a3e1f0d98')]:assert h((C/(name+'.pdf')).read_bytes())==digest
with tempfile.TemporaryDirectory(prefix='g4-reinspection-') as d:
 p=Path(d)/'rl.txt';subprocess.run(['pdftotext','-layout','-f','9','-l','9',str(C/'instructgpt.pdf'),str(p)],check=True);assert p.read_bytes()==(A/'sources/instructgpt-rl.txt').read_bytes()
method=(R/'scripts/course_experiments/posttraining.py').read_bytes();lines=method.splitlines(keepends=True);assert b''.join(lines[272:289])==(A/'code/reference-setup.py').read_bytes()
stdout=json.loads((A/'cpu-checks.stdout.json').read_bytes());assert stdout['status']=='passed';assert not (A/'cpu-checks.stderr.txt').read_bytes()
result={'status':'passed','checked_at':datetime.datetime.now(datetime.UTC).isoformat(),'reviewer_task':'/root/phase4_factual_coordinator/factual_g_4','source':'course/glossary.md#G.4','source_sha256':h(body),'initial_source_sha256':h(old),'initial_report_sha256':h((A/'report-initial-revise.json').read_bytes()),'scope':'Full revised G.4 personally reread in this turn; only one reference-row phrase changed. Original DPO v3 §3 (lines145–188), InstructGPT v1 §3.5 page9 (lines20–42), and original SFT→frozen-reference setup lines273–289 personally re-read. All20 claims original evidence versions verified unchanged. No CPU method rerun, training, GPU, new weights or quality measurement.','reference_before':before,'reference_after':after,'dpo_original_sha256':h((C/'dpo.pdf').read_bytes()),'instructgpt_original_sha256':h((C/'instructgpt.pdf').read_bytes()),'instructgpt_rl_snapshot_sha256':h((A/'sources/instructgpt-rl.txt').read_bytes()),'reference_setup_sha256':h((A/'code/reference-setup.py').read_bytes()),'prior_cpu_stdout_sha256':h((A/'cpu-checks.stdout.json').read_bytes()),'historical_frozen_glossary_sha256':h((A/'inputs/glossary-frozen.md').read_bytes()),'historical_glossary_policy':'Original frozen whole bytes/hash kept unchanged as historical input, not mechanically treated as current whole-file version.','environment':{'python':sys.version,'python_executable':sys.executable,'platform':platform.platform(),'device':'CPU only; metadata, hashes and pdftotext extraction'},'claims':versions}
print(json.dumps(result,ensure_ascii=False,indent=2))
