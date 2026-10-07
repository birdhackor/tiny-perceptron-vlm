"""Own re-parsing of all original candidates, no saved grading flags."""
import json,re
from collections import Counter
from pathlib import Path
from tiny_perceptron.data import ByteTokenizer
v=json.loads(Path('docs/course-experiments/results/reasoning.json').read_text())['results']['comparison']; t=ByteTokenizer(); output={}; examples=[]
for mode,result in v.items():
 output[mode]=[]
 for budget in result['budgets']:
  rows=[];k=budget['candidate_count']
  for sample in budget['samples']:
   a,b,c=map(int,re.fullmatch(r'\((\d+)\+(\d+)\)\+(\d+)=\?',sample['question']).groups());truth=a+b+c;finals=[];good=[];texts=[]
   for raw in sample['candidates']:
    ids,text=raw['generated_ids'],raw['generated'];assert t.decode(ids)==text;text=text.strip();clean=all(x>=8 or x==t.eos_id for x in ids);texts.append(text)
    m=re.fullmatch(r'-?[0-9]+',text) if mode=='direct' else re.search(r';answer=(-?[0-9]+)$',text)
    final=int(m[0] if mode=='direct' else m[1]) if m and clean else None;finals.append(final)
    if mode=='direct': valid=final==truth
    else:
     q=re.fullmatch(r'(-?[0-9]+)\+(-?[0-9]+)=(-?[0-9]+);(-?[0-9]+)\+(-?[0-9]+)=(-?[0-9]+);answer=(-?[0-9]+)',text)
     valid=False
     if q and clean:
      aa,bb,s,p,cc,total,last=map(int,q.groups());valid=aa+bb==s and p+cc==total and p==s and total==last and (aa,bb,cc)==(a,b,c) and last==truth
    good.append(valid)
   counted=[x for x in finals if x is not None]; majority=Counter(counted).most_common(1)[0][0] if counted else None; verified=next((x for x,g in zip(finals,good) if g),None);coverage=truth in finals
   rows.append({'coverage':coverage,'majority_correct':majority==truth,'verifier_correct':verified==truth,'covered_with_fully_verified':coverage and any(good),'final_correct_but_not_verified':sum(x==truth and not g for x,g in zip(finals,good))})
   if (mode=='direct' and k==4 and (a,b,c)==(3,3,0)) or (mode=='steps' and k==2 and (a,b,c)==(5,0,3)):
    examples.append({'mode':mode,'k':k,'question':sample['question'],'truth':truth,'finals':finals,'texts':texts,'majority':majority,'verified':verified})
  output[mode].append({'k':k,'questions':len(rows),'candidates':k*len(rows),'coverage':sum(r['coverage'] for r in rows),'majority_correct':sum(r['majority_correct'] for r in rows),'verifier_correct':sum(r['verifier_correct'] for r in rows),'majority_correct_given_coverage':sum(r['coverage'] and r['majority_correct'] for r in rows),'covered_with_fully_verified':sum(r['covered_with_fully_verified'] for r in rows),'final_correct_but_not_verified':sum(r['final_correct_but_not_verified'] for r in rows)})
output['examples']=examples;output['scope']='Original 720 raw candidates independently final/equation/link/task checked, selector reconstructed; no old oracle/majority/verifier flags, no new generation.';print(json.dumps(output,ensure_ascii=False,indent=2))
