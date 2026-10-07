import json
r=json.load(open('docs/course-experiments/results/posttraining.json'))['results']
for k in ['ppo','dpo','reward']:print(k,r[k]['seconds'],round(r[k]['seconds'],2))
