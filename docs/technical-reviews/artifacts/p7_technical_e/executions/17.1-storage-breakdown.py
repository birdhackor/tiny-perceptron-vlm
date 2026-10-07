import json
from pathlib import Path
d=json.loads(Path('docs/course-experiments/results/quantization.json').read_text())
for name in ['packed4','packed8']:
 s=d['results']['runs'][name]['storage'];b=s['buffers']
 values=sum(v for k,v in b.items() if k.endswith('.values'))
 scales=sum(v for k,v in b.items() if k.endswith('.scale'))
 bias=sum(v for k,v in b.items() if k.endswith('.bias'))
 retained=s['parameter_tensor_bytes']+bias
 bits=4 if name=='packed4' else 8
 assert values*8//bits==115200 and scales//4==1416 and retained//4==26368
 assert values+retained+scales==s['tensor_bytes']
 print(name,'weight coefficient bytes',values,'retained+bias bytes',retained,'scale bytes',scales,'coefficient count',values*8//bits,'fp32 count',retained//4,'scale count',scales//4,'sum',values+retained+scales)
