import json,random,math,torch,hashlib
from scripts.course_experiments.text import arithmetic_records
from scripts.course_experiments.common import split_records
from scripts.course_experiments.behavior import _preference_parts,_pair_examples
from tiny_perceptron.alignment import sequence_log_probability
j=json.load(open('docs/course-experiments/results/dpo.json'));pairs=_preference_parts(split_records(arithmetic_records(),seed=j['seed']))['train'];examples=_pair_examples(pairs,1000)
for name in ('model','beta1'):
 tr=j['results']['runs'][name]['training'];rng=random.Random(j['seed']);n=0
 for step in range(tr['steps']):
  for pair in rng.choices(examples,k=8):
   for x,y in pair:n+=int((y!=-100).sum())
 raw=j['results']['runs'][name]['training']['effective_answer_tokens_both_sides'];print(name,'steps',tr['steps'],'computed_exposure',n,'raw_exposure',raw);assert n==raw==9448 and tr['steps']==250
p=math.exp(2)/(math.exp(2)+2);print('p',p,'log2',2*math.log(p),'p2',p*p,'log3',3*math.log(p),'mean',math.log(p));print('assumed .8 product',.8*.8)
l=torch.tensor([[[0.,2.,0.],[2.,0.,0.],[0.,0.,2.]]]);y=torch.tensor([[1,0,2]]);print('three_valid',sequence_log_probability(l,y).item())
print('torch',torch.__version__)
