from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[4];OUT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
loc=json.loads((OUT/'gpu-original-journal-locator.json').read_text());path=ROOT/loc['source'];rows=[json.loads(x) for x in path.read_text().splitlines()]
cpu=json.loads((OUT/'cpu-raw-derived.json').read_text())['invocations']
def clean(messages):return [{k:v for k,v in m.items() if k in ['role','content']} for m in messages]
matches={};saved={}
for name,c in cpu.items():
 if not c['chat']:continue
 task=c['argv'][c['argv'].index('--task')+1];gs=c['generations'];found=[]
 for i,r in enumerate(rows):
  if r['record']['task']!=task:continue
  args=c['argv']
  if any(flag in args and r['record'].get(key)!=args[args.index(flag)+1] for flag,key in [('--image','image'),('--audio','audio')]):continue
  if '--roi' in args and r['record'].get('roi')!=[int(x) for x in args[args.index('--roi')+1].split(',')]:continue
  if task=='voice_topic_continuation' and '--audio' not in args:
   audios=[m['audio'] for m in c['generations'][0]['prompt_messages'] if 'audio' in m]
   if audios and r['record'].get('audio')!=audios[0]:continue
  gpu=r['model_generations']
  for j in range(len(gpu)-len(gs)+1):
   segment=gpu[j:j+len(gs)]
   if all(clean(cg['prompt_messages'])==clean(gg['prompt_messages']) and cg['generated_ids']==gg['generated_ids'] and cg['raw_output']==gg['raw_output'] for cg,gg in zip(gs,segment)):
    found.append({'line':i+1,'record_id':r['record']['id'],'generation_start':j,'generated_ids_and_raw_outputs_equal':True})
    saved[i+1]={k:r[k] for k in ['record','trace','score','model_generations','perception']}
 assert found,name
 matches[name]=found
snap={'original_source':loc['source'],'original_sha256':sha(path),'line_count':len(rows),'scope':'Only matched original raw validation record / trace / score / model_generations / perception fields; no author commentary.','records':[{'line':n,'raw':r} for n,r in sorted(saved.items())]}
(OUT/'gpu-validation-original-excerpts.json').write_text(json.dumps(snap,ensure_ascii=False,indent=2)+'\n')
(OUT/'gpu-cpu-comparison.json').write_text(json.dumps(matches,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:len(v) for k,v in matches.items()}))
