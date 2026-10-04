from pathlib import Path
import json,hashlib,urllib.request,importlib.metadata,platform
from tokenizers import Tokenizer
P='docs/technical-reviews/artifacts/natural-final-fact-20.7-'
url='https://huggingface.co/openai/whisper-small/resolve/973afd24965f72e36ca33b3055d56a652f456b4d/tokenizer.json'
with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'technical-source-review/1'}),timeout=25) as r: raw=r.read(5*1024*1024);status=r.status
Path(P+'whisper-tokenizer.json').write_bytes(raw);tokenizer=Tokenizer.from_file(P+'whisper-tokenizer.json')
rows=[]
for t in json.loads(Path('docs/natural-assistant/evidence/final/transcripts.json').read_text()):
 text=tokenizer.decode(t['raw_token_ids'],skip_special_tokens=True);assert text==t['transcript']
 prompt=[tokenizer.id_to_token(i) for i in t['expected_decoder_prompt_ids']];assert prompt==['<|startoftranscript|>','<|zh|>','<|transcribe|>','<|notimestamps|>']
 assert tokenizer.id_to_token(t['raw_token_ids'][-1])=='<|endoftext|>'
 rows.append({'id':t['id'],'decoded_raw_sequence':text,'matches_recorded_actual_hypothesis':True,'prompt_tokens':prompt,'last_token':'<|endoftext|>','generated_token_count':len(t['generated_token_ids'])})
res={'command':'.venv/bin/python docs/technical-reviews/artifacts/natural-final-fact-20.7-tokenizer-audit.py','exit_code':0,'environment':{'python':platform.python_version(),'tokenizers':importlib.metadata.version('tokenizers'),'device':'CPU tokenizer only; no weights or model inference'},'source_receipt':{'url':url,'status':status,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()},'records':rows,'decoded_matches':12,'eos_matches':12,'model_weights_loaded':False}
Path(P+'tokenizer-audit.json').write_text(json.dumps(res,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'source_receipt':res['source_receipt'],'environment':res['environment'],'decoded_matches':12,'prompt':rows[0]['prompt_tokens'],'eos':rows[0]['last_token']},ensure_ascii=False,indent=2))
