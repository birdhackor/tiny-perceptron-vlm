"""Save small selected original passages and exact read locators, not whole papers."""
from pathlib import Path
import hashlib,json,re
from html.parser import HTMLParser
ROOT=Path(__file__).resolve().parents[5];HERE=Path(__file__).resolve().parent;RAW=ROOT/'outputs/natural-v4/factual-research/curriculum'
class Visible(HTMLParser):
 def __init__(self):super().__init__();self.parts=[];self.skip=0
 def handle_starttag(self,t,a):
  if t in ['script','style']:self.skip+=1
  if t in ['p','div','h1','h2','h3','li','pre','tr','section']:self.parts.append('\n')
 def handle_endtag(self,t):
  if t in ['script','style']:self.skip-=1
 def handle_data(self,d):
  if not self.skip:self.parts.append(d)
chunks=[];index=[]
queries={
 'original-56.txt':['Applies a linear transformation'],
 'original-57.txt':['scaled_dot_product_attention(query','A boolean mask','FlashAttention'],
 'original-58.txt':['Short-time Fourier transform','window','hop_length'],
 'original-59.txt':['To avoid underflow','batchmean','target'],
 'torch.nn.Module.txt':['eval()','This is equivalent with self.train(False)'],
 'torch.nn.CrossEntropyLoss.txt':['ignore_index','The unreduced'],
 'torch.nn.LayerNorm.txt':['The mean and standard-deviation','normalized_shape'],
 'torch.optim.AdamW.txt':['θt','weight_decay'],
 'torch.optim.Adam.txt':['betas','Algorithm'],
 'torch.optim.SGD.txt':['momentum','weight_decay'],
 'autograd.txt':['accumulated in the .grad','Evaluation mode','no-grad mode'],
 'amp.txt':['autocast','GradScaler'],
 'tokenizers.txt':['BPE','byte_fallback'],
 'unicode.txt':['UTF-8','multi'],
}
for filename,patterns in queries.items():
 p=RAW/filename
 if not p.exists():continue
 raw=p.read_text(); parser=Visible();parser.feed(raw);visible=re.sub(r'\n\s*\n+','\n', ''.join(parser.parts))
 (RAW/(p.stem+'.visible.txt')).write_text(visible)
 for pattern in patterns:
  # Prefer API body after page navigation where a match occurs repeatedly.
  matches=list(re.finditer(re.escape(pattern),visible,re.I))
  if not matches:continue
  chosen=matches[-1] if len(matches)>1 and filename in ['torch.nn.Module.txt'] else matches[0]
  lo=max(0,chosen.start()-200);hi=min(len(visible),chosen.end()+1500)
  passage=visible[lo:hi];chunks.append(f'\nSOURCE {filename}; pattern {pattern}; visible chars {lo}:{hi}\n{passage}\n')
  index.append({'path':str(p.relative_to(ROOT)),'raw_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'pattern':pattern,'visible_char_range':[lo,hi]})
# Official lab originals: only selected original prose/control code.
for n in [38,39]:
 p=RAW/f'original-{n}.txt';d=json.loads(p.read_text())
 for i,c in enumerate(d['cells']):
  s=''.join(c['source'])
  if re.search(r'control|judge|bias|finetun|debias|under.represent|style',s,re.I):
   chunks.append(f'\nSOURCE original-{n}; notebook cell {i}\n{s[:1500]}\n')
   index.append({'path':str(p.relative_to(ROOT)),'cell':i})
   if sum(x.get('path')==str(p.relative_to(ROOT)) for x in index)>=5:break
(HERE/'original-excerpts.txt').write_text(''.join(chunks))
(HERE/'original-excerpts.index.json').write_text(json.dumps(index,ensure_ascii=False,indent=2)+'\n')
print('Saved',len(index),'selected original passages')
