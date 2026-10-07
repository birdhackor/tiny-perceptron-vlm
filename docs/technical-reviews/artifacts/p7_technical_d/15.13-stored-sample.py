import json
from tiny_perceptron.data import ByteTokenizer
r=json.load(open('docs/course-experiments/results/moe.json'))['results'];s=r['variants']['top2_aux0.01']['heldout']['test']['samples'][0];d=ByteTokenizer().decode(s['generated_ids']);print('teacher',r['teacher_variant'],'prompt',repr(s['prompt']),'stored',repr(s['generated']),'decoded',repr(d));assert d==s['generated']
