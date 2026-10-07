import json
r=json.load(open('docs/course-experiments/results/posttraining.json'))['results']['evaluations']['test']
expected={'number':0,'explain':1,'missing':3}
for name in ['sft','ppo','dpo']:
 for mode in expected:
  rows=[x for x in r['rows'] if x['mode']==mode]
  n=sum(x['policies'][name]['chosen_action']==expected[mode] for x in rows)
  print(name,mode,n,len(rows));assert n==r['policies'][name]['by_mode'][mode]['numerator']
 n=sum(x['policies'][name]['chosen_action']==expected[x['mode']] for x in r['rows']);print(name,'total',n,len(r['rows']))
pairs=[(x,w,l) for x in r['rows'] for w,l in x['preference_pairs']]
n=sum(x['reward_model_raw_scores'][w]>x['reward_model_raw_scores'][l] for x,w,l in pairs)
print('RM pairs',n,len(pairs),r['reward_model_pairwise_accuracy']);assert n==90 and len(pairs)==90
row=next(x for x in r['rows'] if x['mode']=='explain' and x['family']=='pair:1:2')
print('explain 1+2',row['reward_model_raw_scores'],row['policies']['ppo']['probabilities'],row['policies']['ppo']['chosen_response'])
answers=['3','我已經仔細思考過，答案是4。'];scores=[int(answer==str(1+2)) for answer in answers];chosen=max(range(len(answers)),key=lambda i:scores[i]);print('variation',scores,answers[chosen])
