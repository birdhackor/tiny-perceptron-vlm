"""Bounded factual probes. Network denied for examples; UI runners explicitly injected.
No model, ASR inference, quality benchmark or GPU training replication occurs here.
"""
import ast, base64, contextlib, copy, hashlib, io, json, platform, re, socket, sys, tempfile
from pathlib import Path
from types import SimpleNamespace
import torch
from PIL import Image
from tiny_perceptron import natural_assistant as core, natural_ui as ui
OUT=Path('docs/technical-reviews/artifacts/natural-v4-factual/20.1')
ROOT=Path('.')
environment={'python':platform.python_version(),'torch':torch.__version__,'device':'cpu','cuda_available':str(torch.cuda.is_available())}
assert torch.__version__=='2.14.1+cpu'
print('ENVIRONMENT',json.dumps(environment))
source=Path('course/chapters/20.md').read_text(); blocks=list(re.finditer(r'```python\n(.*?)```',source,re.S))
original_connect=socket.socket.connect; original_connect_ex=socket.socket.connect_ex
attempts=[]
def denied(self,address):
 attempts.append(str(address)); raise RuntimeError('Network is prohibited in short examples')
socket.socket.connect=denied; socket.socket.connect_ex=denied
results=[]
try:
 for index,m in enumerate(blocks,1):
  buffer=io.StringIO(); namespace={}
  with contextlib.redirect_stdout(buffer):exec(compile(m.group(1),f'20.md:python-block-{index}','exec'),namespace)
  result={'index':index,'line':source[:m.start()].count('\n')+1,'stdout':buffer.getvalue()}; results.append(result)
  print('EXAMPLE',index,buffer.getvalue().strip())
  if index==1:
   assert namespace['typed_chat']==namespace['spoken_chat']
   assert len(namespace['history'])==1
   assert namespace['typed_chat'] is not namespace['history']
   namespace['recognized_question']='請介紹這張照片'
   changed=namespace['history']+[{'role':'user','content':namespace['recognized_question']}]
   assert namespace['typed_chat']!=changed
   print('EXERCISE changed recognized_question -> False; original history unchanged')
 assert len(results)==10 and not attempts
finally: socket.socket.connect=original_connect; socket.socket.connect_ex=original_connect_ex
print('ALL_SHORT_EXAMPLES',len(results),'network_attempts',len(attempts))
# Real session and upload/message/reset methods, with explicit synthetic runners.
# This tests wiring only and is never counted as actual ASR/model behavior.
seen=[]
def injected_generate(model,processor,row,data_root,options):
 seen.append(copy.deepcopy(row)); return {'prediction':'明確注入的測試回答'}
def injected_transcribe(model,processor,path): return {'transcript':'辨識器測試原稿，可能漏字'}
genuine_generate=core.generate; genuine_transcribe=core.transcribe
core.generate=injected_generate; core.transcribe=injected_transcribe
try:
 with tempfile.TemporaryDirectory(prefix='factual-20-1-') as tmp:
  server=ui.create_server(None,None,SimpleNamespace(),tmp,port=0,asr=(None,None))
  try:
   session=server.operation('/api/session',{})['session']
   image=io.BytesIO(); Image.new('RGB',(4,4),(255,255,255)).save(image,format='PNG')
   img=server.operation('/api/upload',{'session':session,'filename':'synthetic.png','kind':'image','base64':base64.b64encode(image.getvalue()).decode()})['asset']
   first=server.operation('/api/chat',{'session':session,'prompt':'接下來請用繁體中文回答。','image':img})
   state=server.session(session); photo_path=state['assets'][img]['path']
   # A one-byte internal stub stands in for uploaded audio; no audio decoding/model run is claimed.
   audio_path=Path(server.upload_directory.name)/'synthetic.wav';audio_path.write_bytes(b'not-real-audio')
   state['assets']['test-audio']={'path':str(audio_path.relative_to(tmp)),'kind':'audio','media_type':'audio/wav'}
   tr=server.operation('/api/transcribe',{'session':session,'audio':'test-audio'})
   assert len(state['history'])==2 and len(seen)==1
   response=server.operation('/api/chat',{'session':session,'prompt':'使用者已更正的招牌問題','speech':'test-audio'})
   assert seen[-1]['history']==state['history'][:2]
   assert seen[-1]['history'][0]['content'][0]['image']==photo_path
   assert seen[-1]['user']=='使用者已更正的招牌問題'
   assert response['asr']['transcript']==tr['asr']['transcript'] and response['asr']['corrected'] is True
   messages=core.messages_for(seen[-1],tmp)
   assert messages[-1]['role']=='user' and messages[-1]['content'][-1]['text']=='使用者已更正的招牌問題'
   assert messages[0]['content'][0]['type']=='image'
   reset=server.operation('/api/reset',{'session':session})
   assert not state['history'] and not state['assets'] and not state['transcriptions']
   assert not (Path(tmp)/photo_path).exists() and not audio_path.exists()
   ui_result={'scope':'Injected runners validate actual UI operation/messages_for wiring, not pretrained inference','chats':len(seen),'transcribe_calls':1,'transcribe_did_not_chat':True,'history_image_retained':True,'submitted_edit_received':True,'original_transcript_retained':True,'reset_deleted_uploads_and_state':True}
   print('UI_WIRING',json.dumps(ui_result,ensure_ascii=False))
  finally:server.server_close()
finally: core.generate=genuine_generate;core.transcribe=genuine_transcribe
(OUT/'cpu-results.json').write_text(json.dumps({'environment':environment,'short_examples':results,'network_attempts':attempts,'ui_wiring':ui_result},ensure_ascii=False,indent=2)+'\n')
