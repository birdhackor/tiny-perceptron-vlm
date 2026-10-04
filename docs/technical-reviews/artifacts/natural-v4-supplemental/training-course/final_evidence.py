from pathlib import Path
import json,hashlib,subprocess,urllib.request,sys,statistics,math
ROOT=Path(__file__).resolve().parents[5];D=Path(__file__).parent
sys.path.insert(0,str(ROOT));from tiny_perceptron.data import ByteTokenizer
import torch
T=ByteTokenizer();torch.set_num_threads(1)
sha=lambda b:hashlib.sha256(b).hexdigest()
source=(ROOT/'course/training.md').read_bytes();pinned=subprocess.check_output(['git','show','f8ca78bc3ffef82b0faa352c464909b76bfd8976:course/training.md'],cwd=ROOT)
with urllib.request.urlopen('http://127.0.0.1:8783/training.html') as r:html=r.read()
receipt={'command':' .venv/bin/python '+str(Path(__file__).relative_to(ROOT)),'current_source_sha256':sha(source),'bytes':len(source),'lines':len(source.splitlines()),'matches_read_snapshot':source==(D/'training.md.read-snapshot').read_bytes(),'preview_revision':'f8ca78bc3ffef82b0faa352c464909b76bfd8976','preview_source_sha256':sha(pinned),'preview_matches_current_source':source==pinned,'http_html_sha256':sha(html),'http_url':'http://127.0.0.1:8783/training.html','http_is_not_browser_receipt':True}
(D/'final-byte-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
# Existing individual CPU measurement records: hash and decode original stdout; ignore author summary/pass fields.
student={}
for name in ['text_foundation','media-and-raw-adapter-cli','fsdd-format-and-resampling','application-workflows']:
 p=ROOT/'docs/course-experiments/student-checks'/(name+'.json');x=json.loads(p.read_text());entry={'path':str(p.relative_to(ROOT)),'sha256':sha(p.read_bytes()),'environment':x['environment'],'scope':'audit original CPU operation record, no new accuracy score and no reviewer model run'}
 if name=='text_foundation':entry['original_command']=x['command'];entry['generation']=x['generation'];entry['decoded_original_ids']=T.decode(x['generation']['generated_ids']);entry['verified_file_count']=x['verified_files']
 elif name=='media-and-raw-adapter-cli':
  entry['commands']=[]
  for c in x['commands']:
   o=c['output'];entry['commands'].append({'command':c['command'],'returncode':c['returncode'],'stdout_sha256_recomputed':sha(c['stdout'].encode()),'stdout_hash_equal':sha(c['stdout'].encode())==c['stdout_sha256'],'answer':o['answer'],'decoded_ids':T.decode(o['generated_ids']),'raw_ids':o['generated_ids'],'eos':2 in o['generated_ids'],'no_illegal_control_ids':all(i>=8 or i==2 for i in o['generated_ids'])})
 elif name=='fsdd-format-and-resampling':entry['record_count']=len(x['records']);entry['amplitude_processing']=x['amplitude_processing'];entry['sample_rate_checks']=[{'source_rate':r.get('sample_rate',r.get('source_rate')),'source':r.get('source',r.get('filename'))} for r in x['records']];entry['source_resampling_function']=x['resampling_function'];entry['initial_record_failure_preserved']=x['initial_comparison_failure']
 else:entry['operations']={k:{'operation':v} for k,v in x['operations'].items()}
 student[name]=entry
(D/'student-record-audit.json').write_text(json.dumps(student,ensure_ascii=False,indent=2)+'\n')
# Small existing data, no new download or extraction.
datasets={}
for p in [ROOT/'data/training/text-initial/tinystories-train-512.jsonl',ROOT/'data/training/text-initial/chinese-classical-train-365.jsonl',ROOT/'data/training/reasoning-initial/gsm8k-train-first200.jsonl',ROOT/'data/training/vision-initial/train.jsonl',*[ROOT/'data/training/fsdd-initial'/(n+'.jsonl') for n in ['train','validation','test']]]:
 rows=[json.loads(s) for s in p.read_text().splitlines() if s.strip()];datasets[str(p.relative_to(ROOT))]={'rows':len(rows),'sha256':sha(p.read_bytes()),'normalized_text_unique':len({' '.join(r['text'].split()) for r in rows}) if 'text' in rows[0] else None,'speakers':sorted({r['speaker'] for r in rows}) if 'speaker' in rows[0] else None}
# Recompute each retained Flash measurement median and MiB directly from raw sample bytes.
f=json.loads((ROOT/'docs/course-experiments/results/flash_probe.json').read_text())['results'];flash=[]
for dtype,route in f['routes'].items():
 for mode,variants in route['measurements'].items():
  for name,m in variants.items():
   median=statistics.median(m['samples_seconds']);flash.append({'dtype':dtype,'mode':mode,'route':name,'samples':len(m['samples_seconds']),'warmup':m['warmup_calls'],'expected_median_seconds':m['median_seconds'],'observed_median_seconds':median,'equal':median==m['median_seconds'],'baseline_mib':m['allocated_before_bytes']/2**20,'peak_mib':m['peak_allocated_bytes']/2**20,'additional_mib':m['additional_peak_allocated_bytes']/2**20,'memory_difference_equal':m['peak_allocated_bytes']-m['allocated_before_bytes']==m['additional_peak_allocated_bytes'],'original_cuda_kernel_names':route['profiles'][mode]['cuda_kernel_names']})
# OCR CER: full edit distance, original answers, all 30 samples.
o=json.loads((ROOT/'docs/course-experiments/results/ocr.json').read_text())['results']['test'];edits=0;characters=0
for s in o['samples']:
 a,b=s['target'],s['generated'];prev=list(range(len(b)+1))
 for i,ac in enumerate(a,1):
  row=[i]
  for j,bc in enumerate(b,1):row.append(min(row[-1]+1,prev[j]+1,prev[j-1]+(ac!=bc)))
  prev=row
 edits+=prev[-1];characters+=len(a)
extra={'environment':{'python':sys.version.split()[0],'torch':torch.__version__,'device':'cpu'},'datasets':datasets,'flash_raw_measurement_audit':flash,'ocr_CER':{'edits':edits,'reference_characters':characters,'rate':edits/characters,'examples':len(o['samples']),'not_reviewer_GPU_replication':True}}
(D/'bounded-final-audit.json').write_text(json.dumps(extra,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'source':receipt,'data_counts':datasets,'ocr_CER':extra['ocr_CER'],'flash_measurements':len(flash),'all_flash_medians_equal':all(r['equal'] and r['memory_difference_equal'] for r in flash),'student_records':list(student)},ensure_ascii=False,indent=2))
