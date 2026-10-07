import json,torch
from pathlib import Path
from tiny_perceptron.selftrained.inference import InferenceAssistant
torch.set_num_threads(2);m=json.loads(Path('docs/selftrained/examples/v2/text.messages.json').read_text());m[1]['content']='接下來請用一句回答。';m[2]['content']='好，接下來用一句回答。'
a=InferenceAssistant('outputs/p7-technical-e-cache/moe-joint','outputs/selftrained-v2/data',device='cpu',manifest_sha256='f27856892b0c46ca81ba43bfadf7051ed6bd31d848e4868a05147e5799df1e5e');r=a.reply(m,task='text',max_new_tokens=128);p=Path('docs/technical-reviews/artifacts/p7_technical_e/executions/19.1-variation-result.json');assert not p.exists();p.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');print('changed_messages',m[1:3]);print('answer',r['answer']);print('eos',r['generations'][0]['eos'],'generated_ids',r['generations'][0]['generated_ids'])
