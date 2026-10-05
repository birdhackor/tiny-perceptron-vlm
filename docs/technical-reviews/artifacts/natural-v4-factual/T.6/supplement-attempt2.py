import hashlib,json,random,subprocess,sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from tiny_perceptron.data import ByteTokenizer,render_chat
from tiny_perceptron.modal_data import modal_example
from scripts.prepare_ocr import draw_digits
from scripts.prepare_data import generate_records
from scripts.course_experiments.common import split_records
torch.set_num_threads(1)
tok=ByteTokenizer();temp=Path('outputs/natural-v4/factual-research/T.6');out={}
def load(name):return json.loads(Path(f'docs/course-experiments/results/{name}.json').read_text())['results']
sft=load('sft');replay=split_records(generate_records('attributes-sft'),seed=42)['train']
replay_raw=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in replay).encode()
assert hashlib.sha256(replay_raw).hexdigest()==sft['data']['train']['sha256']
out['replay_input_reconstruction']={'records':len(replay),'sha256':hashlib.sha256(replay_raw).hexdigest(),'matches_original_sft_train_file_fingerprint':True}
counts={}
groups=[('projector',load('projector'),4,None),('ocr',load('ocr'),8,None)]
vqa=load('vqa')
groups += [(f'vqa/{k}',{**g,'data':vqa['data']},4,replay if k=='all_replay' else None) for k,g in vqa['variants'].items()]
groups += [(f'real/{k}',g,4,None) for k,g in load('real_modal').items()]
for name,g,batch,text in groups:
 rows=g['data']['splits']['train']['records'];history=g['training']['history'];total=0;modal_targets=text_targets=modal_examples=text_examples=0
 for step,h in enumerate(history):
  rng=random.Random(42+step);count=0
  for _ in range(batch):
   if text and rng.random()<0.5:
    r=rng.choice(text);_,labels=render_chat(r['messages']);n=int((labels!=-100).sum());text_targets+=n;text_examples+=1
   else:
    r=rng.choice(rows);n=len(tok.encode(r['answer']))+1;modal_targets+=n;modal_examples+=1
   count+=n
  assert count==h['effective_targets'],(name,step,count,h['effective_targets'])
  total+=count
 counts[name]={'targets_reconstructed_from_original_inputs':total,'batch_examples':batch,'updates':len(history),'modal_targets':modal_targets,'text_targets':text_targets,'modal_examples':modal_examples,'text_examples':text_examples}
out['independent_sampling_budget']=counts
fashion=load('real_modal')['fashion-mnist'];path=Path('data/training/vision-initial/images')/Path(fashion['data']['splits']['test']['records'][0]['image']).name
raw=Image.open(path);pixels=np.array(raw.convert('RGB').resize((16,16)),copy=True)
ids,labels,image,_=modal_example({'image':path.name,'question':'clothing?','answer':'Ankle boot'},path.parent,'vision')
out['image_conversion']={'input_mode':raw.mode,'input_size':list(raw.size),'output_shape':list(image.shape),'all_RGB_channels_equal':bool(torch.equal(image[0],image[1]) and torch.equal(image[1],image[2])),'matches_RGB_resize_divide_255':bool(torch.equal(image,torch.from_numpy(pixels).permute(2,0,1).float()/255)),'no_input_standardization_mean':float(image.mean()),'range':[float(image.min()),float(image.max())]}
ids,labels,_,wave=modal_example({'audio':'one-derived.wav','question':'digit?','answer':'0'},temp,'audio')
out['custom_audio_reader']={'derived_audio_samples':wave.numel(),'valid_answer_targets':int((labels!=-100).sum())}
try:modal_example({'audio':'0_jackson_5.wav','question':'digit?','answer':'0'},Path('data/training/fsdd-initial/recordings'),'audio')
except ValueError as e:out['custom_audio_8k_rejection']=str(e)
families=list(range(100));random.Random(42).shuffle(families);out['ocr_grouping']={side:{'families':len(values),'records':len(values)*3,'values':values} for side,values in [('train',families[:80]),('validation',families[80:90]),('test',families[90:])]}
records=[]
for i,v in enumerate(families[80:83]):
 name=f'ocr-{i}.png';draw_digits(str(v)).save(temp/name);records.append({'image':name,'question':'read digits','answer':str(v)})
data=temp/'three-ocr.jsonl';data.write_text(''.join(json.dumps(x)+'\n' for x in records))
ablation=[]
for mode in ['none','blank','shuffle']:
 cmd=[sys.executable,'scripts/evaluate_modal.py','checkpoints/course/ocr/model.pt','--data',str(data),'--ablation',mode,'--limit','3','--tokens','16','--seed','42','--device','cpu','--output',str(temp/(mode+'.json'))]
 p=subprocess.run(cmd,capture_output=True,text=True);assert p.returncode==0,p.stderr;r=json.loads((temp/(mode+'.json')).read_text())
 assert len(r['samples'])==3
 if mode=='shuffle':assert all(x['donor_row']!=x['row'] for x in r['samples'])
 ablation.append({'command':cmd,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'samples':r['samples']})
out['own_three_row_ablation_execution']=ablation
base=[sys.executable,'scripts/train.py','--task','vision','--checkpoint',str(temp/'resumed.pt'),'--resume','--freeze','partial','--train','--steps','3','--batch-size','2','--device','cpu']
negative=[]
for flags in [[],['--vision-encoder',str(temp/'absent.pt')],['--lr','0.002']]:
 p=subprocess.run(base+flags,capture_output=True,text=True);assert p.returncode!=0;negative.append({'command':base+flags,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
out['expected_resume_rejections']=negative
print(json.dumps(out,ensure_ascii=False,indent=2))
