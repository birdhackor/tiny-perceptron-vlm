"""Bounded CPU review: existing data read only; one derived WAV in ignored research."""
import hashlib,json,math,random,re,struct,subprocess,sys,urllib.request
from pathlib import Path
import numpy as np
import PIL
from PIL import Image
import soundfile as sf
import torch
from tiny_perceptron.data import ByteTokenizer,render_chat
from tiny_perceptron.model import TinyLM,ModelConfig,masked_loss
from tiny_perceptron.multimodal import MultiModalLM,AudioEncoder,log_mel,tone,scene
from tiny_perceptron.modal_data import modal_example
from tiny_perceptron.training import load_checkpoint,save_checkpoint
from scripts.audio_utils import load_mono_audio,resample_waveform
from scripts.course_experiments.modalities import _freeze,_resample_8_to_16
from scripts.evaluate import evaluate

torch.set_num_threads(1)
art=Path('docs/technical-reviews/artifacts/natural-v4-factual/T.6')
temp=Path('outputs/natural-v4/factual-research/T.6');temp.mkdir(parents=True,exist_ok=True)
tok=ByteTokenizer()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
out={'environment':{'python':sys.version.split()[0],'torch':torch.__version__,'numpy':np.__version__,'soundfile':sf.__version__,'libsndfile':sf.__libsndfile_version__,'pillow':PIL.__version__,'device':'cpu','threads':'1','cuda_available':str(torch.cuda.is_available())}}
out['source_hashes']={}
requests={'course/training.md':['T.6','T.3','T.4'],'course/chapters/04.md':['4.3'],'course/chapters/10.md':['10.5'],'course/chapters/11.md':['11.1','11.3','11.5','11.6','11.7','11.12','11.13'],'course/chapters/12.md':['12.7','12.8']}
for path,ids in requests.items():
 raw=Path(path).read_bytes()
 for ident in ids:
  match=re.search(rb'^## '+re.escape(ident.encode())+rb'\s',raw,re.M);end=raw.find(b'\n## ',match.start()+1);section=raw[match.start():] if end<0 else raw[match.start():end+1]
  out['source_hashes'][path+'#'+ident]=hashlib.sha256(section).hexdigest()
  if ident=='T.6':(art/'source-original.md').write_bytes(section)
out['code_hashes']={p:sha(p) for p in ['scripts/train.py','scripts/pretrain_encoders.py','scripts/audio_utils.py','scripts/prepare_ocr.py','scripts/evaluate.py','scripts/evaluate_modal.py','scripts/infer_modal.py','tiny_perceptron/training.py','tiny_perceptron/multimodal.py','tiny_perceptron/modal_data.py','tiny_perceptron/model.py','tiny_perceptron/data.py','scripts/course_experiments/modalities.py','scripts/course_experiments/common.py','scripts/course_experiments/run.py']}
with urllib.request.urlopen('http://127.0.0.1:8788/training.html') as response:html=response.read()
out['live_page']={'url':'http://127.0.0.1:8788/training.html#T.6','bytes':len(html),'sha256':hashlib.sha256(html).hexdigest(),'T6_anchor_present':bool(re.search(rb'id=["\']T\.6["\']',html))}

# Independent arithmetic and preprocessing checks.
x=torch.tensor([[[1.,2.,3.],[10.,20.,30.]]]);layer=torch.nn.LayerNorm(3);y=layer(x)
out['layernorm']={'output':y.tolist(),'per_position_mean':y.mean(-1).tolist(),'per_position_variance':y.var(-1,unbiased=False).tolist(),'epsilon':layer.eps,'formula_variance_first':(2/3)/(2/3+layer.eps)}
wave=tone()[None];features=AudioEncoder(bands=16,width=8)(wave)
out['audio_shapes']={'waveform':list(wave.shape),'log_mel':list(log_mel(wave).shape),'features':list(features.shape),'silent_log_mel_finite':bool(torch.isfinite(log_mel(torch.zeros(1600))).all())}
out['cli_encoder_holdout']={'vision':[(c,s,o,int(s=='circle')) for c in ('red','green','blue') for s in ('square','circle') for o in (-1,0,1) if c=='blue' and o==1],'audio':[(f,int(f>500)) for f in (180,220,260,780,880,1000) if f in (180,1000)],'one_of_two':1/2}
m=MultiModalLM(TinyLM(ModelConfig(width=8,layers=2)))
_freeze(m,'partial');fixed=[n for n,p in m.named_parameters() if p.requires_grad]
m.requires_grad_(False);m.image_projector.requires_grad_(True);m.audio_projector.requires_grad_(True);m.language.blocks[0].requires_grad_(True);m.language.blocks[-1].requires_grad_(True)
cli=[n for n,p in m.named_parameters() if p.requires_grad]
out['partial_scopes']={'fixed_block_indices':sorted(set(n.split('.')[2] for n in fixed if n.startswith('language.blocks.'))),'cli_block_indices':sorted(set(n.split('.')[2] for n in cli if n.startswith('language.blocks.'))),'cli_embedding_frozen':not m.language.embedding.weight.requires_grad,'cli_output_frozen':not m.language.output.weight.requires_grad}
m.requires_grad_(False);m.image_projector.requires_grad_(True);m.audio_projector.requires_grad_(True)
prefix=[tok.bos_id,tok.user_id,tok.image_id]+tok.encode('shape?')+[tok.eos_id,tok.assistant_id];tail=tok.encode('circle')+[tok.eos_id]
result=m(torch.tensor(prefix+tail),torch.tensor([-100]*len(prefix)+tail),image=scene('blue','circle'))
loss=masked_loss(result['logits'],result['labels']);loss.backward()
out['mask_and_gradient']={'unexpanded_prefix':len(prefix),'expanded_prediction_positions':list(result['labels'].shape),'effective_targets':int((result['labels']!=-100).sum()),'expected_targets':len(tail),'image_projector_nonzero_gradient':bool(m.image_projector.weight.grad.norm()>0),'audio_projector_gradient_none':m.audio_projector.weight.grad is None,'audio_encoder_all_gradients_none':all(p.grad is None for p in m.audio.parameters())}
assert out['mask_and_gradient']['effective_targets']==len(tail)

headers=[];split_counts={};maximum=0
for side,speaker in [('train','jackson'),('validation','nicolas'),('test','theo')]:
 rows=[json.loads(line) for line in Path(f'data/training/fsdd-initial/{side}.jsonl').read_text().splitlines()]
 assert len(rows)==20 and {r['speaker'] for r in rows}=={speaker} and {r['recording_index'] for r in rows}=={5,6}
 split_counts[side]={'records':len(rows),'speaker':speaker,'digits':sorted({r['digit_label'] for r in rows}),'indices':[5,6]}
 for row in rows:
  p=Path('data/training/fsdd-initial')/row['path'];info=sf.info(p);assert info.samplerate==8000 and info.channels==1 and info.subtype=='PCM_16';assert sha(p)==row['sha256'];maximum=max(maximum,info.frames*2)
  headers.append({'file':p.name,'sha256':sha(p),'rate':info.samplerate,'channels':info.channels,'frames':info.frames,'subtype':info.subtype})
out['fsdd_headers']={'count':len(headers),'all_original_hashes_match':True,'all_mono_8k_PCM16':True,'split_counts':split_counts,'max_resampled_samples':maximum,'required_context':math.ceil(maximum/160)+100}
p=Path('data/training/fsdd-initial/recordings/0_jackson_5.wav');v,rate=sf.read(p,dtype='float32');integer,_=sf.read(p,dtype='int16');derived=_resample_8_to_16(v)
dest=temp/'one-derived.wav';sf.write(dest,derived,16000,subtype='FLOAT');di=sf.info(dest);back,_=sf.read(dest,dtype='float32')
out['one_wav']={'source':str(p),'source_sha256':sha(p),'source_frames':len(v),'source_rate':rate,'derived_frames':di.frames,'derived_rate':di.samplerate,'derived_subtype':di.subtype,'source_duration':len(v)/rate,'derived_duration':di.frames/di.samplerate,'all_float_decoding_equals_int16_div_32768':bool(np.array_equal(v,integer.astype('float32')/32768)),'amplitude_range_before':[float(v.min()),float(v.max())],'amplitude_range_after':[float(derived.min()),float(derived.max())],'generic_vs_formal_resample_equal':bool(np.array_equal(derived,resample_waveform(v,8000))),'float_wav_roundtrip_equal':bool(np.array_equal(derived,back)),'derived_payload_sha256':hashlib.sha256(derived.tobytes()).hexdigest()}
try:load_mono_audio(p)
except ValueError as error:out['raw_8k_without_flag_rejected']=str(error)
for width,length in [(64,128),(64,176)]:
 model=MultiModalLM(TinyLM(ModelConfig(width=width,layers=2,max_length=length)))
 out.setdefault('parameter_counts',{})[str(length)]=sum(p.numel() for p in model.parameters())
out['position_table_delta']=(176-128)*64

# Audit original records from predictions, targets, histories, and actual split data.
def edit_distance(a,b):
 row=list(range(len(b)+1))
 for i,left in enumerate(a,1):
  new=[i]
  for j,right in enumerate(b,1):new.append(min(new[-1]+1,row[j]+1,row[j-1]+(left!=right)))
  row=new
 return row[-1]
audits={}
for name in ('encoders','projector','vqa','real_modal','ocr'):
 path=Path(f'docs/course-experiments/results/{name}.json');d=json.loads(path.read_text());r=d['results']
 audit={'record_sha256':sha(path),'run':{k:d[k] for k in ('revision','device','seed','torch_version','python_version','gpu','timing_scope')},'current_modalities_source_matches_original_run':sha('scripts/course_experiments/modalities.py')==d['code_sha256']['scripts/course_experiments/modalities.py'],'groups':{}}
 groups=r if name in ('encoders','real_modal') else r['variants'] if name=='vqa' else {name:r}
 for key,g in groups.items():
  t=g['training'];history=t['history'];count=sum(h['effective_targets'] for h in history)
  assert len(history)==t['steps'] and count==t['effective_targets']
  ga={'updates':len(history),'effective_targets':count,'parameters':t['parameters'],'trainable_parameters':t['trainable_parameters'],'probe_losses':[t['initial_loss'],t['final_loss']]}
  for side in ('validation','test'):
   e=g[side];samples=e['samples'];correct=0;eos=0
   for sample in samples:
    if 'predicted' in sample:ok=sample['predicted']==sample['target'];assert ok==sample['correct']
    else:
     ids=sample['generated_ids'];raw=ids[:ids.index(tok.eos_id)] if tok.eos_id in ids else ids;ok=raw==tok.encode(sample['target']);assert ok==sample['exact_match'];eos+=tok.eos_id in ids
    correct+=ok
   assert correct==e['correct']
   ga[side]={'correct':correct,'examples':len(samples),'eos':eos if name!='encoders' else None,'effective_targets':e.get('effective_tokens')}
  if 'data' in g:ga['splits']={s:{'records':len(v['records']),'families':len({x['family'] for x in v['records']})} for s,v in g['data']['splits'].items()}
  if 'text_after' in g:
   ga['text_before']={k:g['text_before'].get(k) for k in ('matches','records')};ga['text_after']={k:g['text_after'].get(k) for k in ('matches','records')}
  if name=='ocr':
   samples=g['test']['samples'];edits=sum(edit_distance(s['generated'],s['target']) for s in samples);characters=sum(len(s['target']) for s in samples);assert edits==40 and characters==57
   ga['CER']={'edits':edits,'reference_characters':characters,'ratio':edits/characters,'correct_rows':[s['row'] for s in samples if s['exact_match']]}
  audit['groups'][key]=ga
 if name=='vqa':audit['budget']={'alignment':r['two_stage_alignment_training']['effective_targets'],'two_stage':r['two_stage_alignment_training']['effective_targets']+r['variants']['all']['training']['effective_targets'],'direct':r['variants']['direct_vqa']['training']['effective_targets']}
 audits[name]=audit
out['original_record_audit']=audits

# Existing public checkpoints: execute exactly the three cited CPU media examples.
original=json.loads(Path('docs/course-experiments/student-checks/media-and-raw-adapter-cli.json').read_text());media=[]
for entry in original['commands'][:3]:
 proc=subprocess.run(entry['command'],capture_output=True,text=True);assert proc.returncode==0,proc.stderr;result=json.loads(proc.stdout);assert result['answer']==entry['output']['answer']
 media.append({'command':entry['command'],'returncode':proc.returncode,'stdout':result,'stderr':proc.stderr,'original_stdout_sha256':hashlib.sha256(entry['stdout'].encode()).hexdigest(),'original_stdout_hash_matches_record':hashlib.sha256(entry['stdout'].encode()).hexdigest()==entry['stdout_sha256']})
out['own_media_execution']=media

# Three-update native CLI resume proves schedule and state behavior, not 500-step quality.
base=[sys.executable,'scripts/train.py','--task','vision','--freeze','partial','--train','--steps','3','--width','8','--layers','2','--batch-size','2','--device','cpu','--seed','42']
receipts=[]
for tailargs in [['--output',str(temp/'uninterrupted.pt')],['--stop-after','1','--output',str(temp/'interrupted.pt')],['--checkpoint',str(temp/'interrupted.pt'),'--resume','--output',str(temp/'resumed.pt')]]:
 proc=subprocess.run(base+tailargs,capture_output=True,text=True);assert proc.returncode==0,proc.stderr;receipts.append({'command':base+tailargs,'returncode':proc.returncode,'stdout':proc.stdout,'stderr':proc.stderr})
full=torch.load(temp/'uninterrupted.pt',weights_only=True);resume=torch.load(temp/'resumed.pt',weights_only=True)
out['resume_execution']={'receipts':receipts,'all_model_tensors_equal':all(torch.equal(full['model'][k],resume['model'][k]) for k in full['model']),'step':resume['step'],'format_version':resume['format_version'],'saved_fields':list(resume),'remaining_updates':2}
assert out['resume_execution']['all_model_tensors_equal'] and resume['step']==3
model,_=load_checkpoint(temp/'resumed.pt','cpu');records=[{'messages':[{'role':'user','content':'shape?'},{'role':'assistant','content':'circle'}]}]
snapshot={n:p.detach().clone() for n,p in model.named_parameters()};report=evaluate(model.language,records,mode='sft',max_new_tokens=24)
out['language_evaluation']={'records':len(records),'max_new_tokens':24,'effective_tokens':report['effective_tokens'],'metric_denominators':report['metric_denominators'],'samples':report['samples'],'no_weights_changed':all(torch.equal(snapshot[n],p) for n,p in model.named_parameters())}
print(json.dumps(out,ensure_ascii=False,indent=2))
