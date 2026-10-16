"""Tiny local format checks; no optimizer updates or historical models."""
import copy
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from PIL import Image

from scripts.evaluate import answer_sample,evaluate
from tiny_perceptron.adapters import base_state_sha256,load_lora_adapter
from tiny_perceptron.modal_data import modal_example
from tiny_perceptron.model import ModelConfig,TinyLM
from tiny_perceptron.training import load_checkpoint,seed_everything
from tiny_perceptron.tokenization import load_tokenizer

OUT=Path(__file__).parent
WORK=Path('/workspace/work/textbook-repair-20261017/training-technical/cpu')
torch.set_num_threads(2);seed_everything(42)
model=TinyLM(ModelConfig())
state={k:v.clone() for k,v in model.state_dict().items()}
# The named projection may differ across implementation versions; use an actual Linear path.
name,layer=next((n,m) for n,m in model.named_modules() if isinstance(m,torch.nn.Linear))
rank=2
payload={'format_version':'lora-v1','config':asdict(model.config),'base_sha256':base_state_sha256(state),'scaling':'alpha/rank','adapter':{name:{'a':torch.zeros(rank,layer.in_features),'b':torch.zeros(layer.out_features,rank),'rank':rank,'alpha':2}}}
path=WORK/'toy-adapter.pt';torch.save(payload,path)
accepted=load_lora_adapter(model,path,base_state=state)
wrong=TinyLM(ModelConfig());rejection=None
try:load_lora_adapter(wrong,path)
except ValueError as e:rejection=str(e)
assert rejection and 'base_sha256' in rejection
image=Image.new('L',(23,11),color=128);image.save(WORK/'toy-image.png')
record={'image':'toy-image.png','question':'read digits','answer':'12'}
ids,labels,pixels,_=modal_example(record,WORK,'vision')
sf.write(WORK/'tone16.wav',np.zeros(160,dtype=np.float32),16000)
sf.write(WORK/'tone8.wav',np.zeros(80,dtype=np.float32),8000)
audio_record={'audio':'tone16.wav','question':'pitch?','answer':'low'}
_,_,_,wave=modal_example(audio_record,WORK,'audio')
audio_record['audio']='tone8.wav';audio_rejection=None
try:modal_example(audio_record,WORK,'audio')
except ValueError as e:audio_rejection=str(e)
assert audio_rejection is not None
from tiny_perceptron.data import ByteTokenizer
tok=ByteTokenizer()
matching={'terminated':answer_sample(tok,tok.encode('circle')+[2],'circle'),'no_eos':answer_sample(tok,tok.encode('circle'),'circle'),'whitespace':answer_sample(tok,tok.encode(' circle')+[2],'circle'),'control':answer_sample(tok,[4]+tok.encode('circle')+[2],'circle')}
assert matching['terminated']['completed_exact_match']
assert matching['no_eos']['exact_match'] and not matching['no_eos']['completed_exact_match']
assert not matching['whitespace']['exact_match'] and not matching['control']['exact_match']
ab=evaluate(TinyLM(ModelConfig()),[{'text':'AB'}],mode='text',max_new_tokens=2)
bigram_rejection=None
try:load_checkpoint(WORK/'bigram.pt')
except ValueError as e:bigram_rejection=str(e)
assert bigram_rejection
tokenizer_rejections=[]
try:load_tokenizer(None,512)
except ValueError as e:tokenizer_rejections.append(str(e))
tokpath=WORK/'byte-tokenizer.json';tokpath.write_text(json.dumps(tok.state()))
try:load_tokenizer(tokpath,264,{'tokenizer':{'sha256':'0'*64}})
except ValueError as e:tokenizer_rejections.append(str(e))
assert len(tokenizer_rejections)==2
Image.new('RGB',(12,23),color='white').save(WORK/'toy-image-2.png')
rows=[{'image':'toy-image.png','question':'read digits','answer':'12'},{'image':'toy-image-2.png','question':'read digits','answer':'34'}]
jsonl=WORK/'toy-modal-rows.jsonl';jsonl.write_text(''.join(json.dumps(r)+'\n' for r in rows))
modal_ablations=[]
for mode in ('none','blank','shuffle'):
 destination=WORK/f'modal-{mode}-evaluation.json'
 cmd=[sys.executable,'scripts/evaluate_modal.py',str(WORK/'vision-contract.pt'),'--data',str(jsonl),'--ablation',mode,'--limit','2','--tokens','1','--device','cpu','--output',str(destination)]
 p=subprocess.run(cmd,cwd=Path(__file__).resolve().parents[4],capture_output=True,text=True,timeout=30)
 assert p.returncode==0,p.stderr
 report=json.loads(destination.read_text())
 if mode=='shuffle':assert [x['donor_row'] for x in report['samples']]==[1,0] and [x['target'] for x in report['samples']]==['12','34']
 modal_ablations.append({'command':cmd,'returncode':p.returncode,'stdout':p.stdout,'report':report})
result={'environment':{'python':platform.python_version(),'torch':str(torch.__version__),'device':'cpu'},'scope':'Format/metric and input fixtures only. No historical checkpoint or optimizer update.','lora':{'same_base_accepted':accepted,'wrong_base_rejected':rejection},'modal_file_input':{'image_shape':list(pixels.shape),'image_dtype':str(pixels.dtype),'wave_shape':list(wave.shape),'wrong_sample_rate_rejected':audio_rejection},'sft_matching':matching,'AB_full_evaluation_effective_tokens':ab['effective_tokens'],'AB_note':'Actual full-file text evaluator counts A, B, EOS = 3; T.8 explicitly assumes only B and EOS = 2 for its artificial table, not claimed runtime output.','bigram_incompatible_with_transformer_loader':bigram_rejection,'tokenizer_rejections':tokenizer_rejections,'modal_ablations':modal_ablations,'manual_examples':{'storage_delta_bytes':60000-80000,'storage_ratio':60000/80000,'correct_delta':3-4,'eos_id':2,'A_id':ord('A')+8}}
(OUT/'edges-check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
