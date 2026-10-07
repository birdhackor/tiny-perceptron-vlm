import json
r=json.load(open('docs/course-experiments/results/posttraining.json'))['results']['evaluations']['train']
for name in ['ppo','dpo']:
 rows=[x for x in r['rows'] if x['mode']=='explain'];n=sum(x['policies'][name]['chosen_action']==1 for x in rows);print(name,'train explain',n,len(rows));assert n==0 and len(rows)==44
