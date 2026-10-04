from pathlib import Path
import contextlib, io, json, hashlib, platform, importlib.metadata, copy, subprocess
import torch
from PIL import Image
from tiny_perceptron import natural_assistant as a, natural_ui
ROOT=Path(__file__).resolve().parents[3]
ART=ROOT/'docs/technical-reviews/artifacts'
raw=(ROOT/'course/chapters/20.md').read_text()
start=raw.index('## 20.1 '); end=raw.index('## 20.2 ')
body=raw[start:end]
code=body.split('```python\n',1)[1].split('```',1)[0]
results={}
for label,source in [('original',code),('exercise',code.replace('recognized_question = "請讀出照片裡的招牌。"','recognized_question = "請介紹這張照片。"'))]:
 scope={};capture=io.StringIO()
 with contextlib.redirect_stdout(capture):exec(compile(source,'20.1-'+label,'exec'),scope)
 text=capture.getvalue();print(label+'\n'+text)
 expected=label=='original'
 assert (scope['typed_chat']==scope['spoken_chat']) is expected
 assert scope['history']==[{'role':'user','content':'接下來請用繁體中文回答。'}]
 assert len(scope['typed_chat'])==len(scope['spoken_chat'])==2
 results[label]={'stdout':text,'equal':scope['typed_chat']==scope['spoken_chat'],'history_unchanged':True,'recognized_question':scope['recognized_question']}
class CaptureProcessor:
 def apply_chat_template(self,messages,**kwargs):
  self.messages=copy.deepcopy(messages); self.template_options=kwargs;return 'template fixture'
 def __call__(self,**kwargs):
  self.arguments=kwargs; return {'input_ids':torch.tensor([[1,2,3]])}
image=ROOT/'outputs/natural-extension/data/ocr/images/train-ocr-0000.png'
assert image.is_file()
row={'user':'下一問','history':[{'role':'user','content':[{'type':'image','image':str(image)},{'type':'text','text':'原來的問題'}]},{'role':'assistant','content':'原來的回答'}]}
processor=CaptureProcessor();msgs=a.messages_for(row,ROOT/'outputs/natural-extension/data');a.encode_messages(processor,msgs,ROOT/'outputs/natural-extension/data',generation_prompt=True)
assert len(msgs)==3 and msgs[-1]['content']==[{'type':'text','text':'下一問'}]
assert len(processor.arguments['images'])==1
actual=processor.arguments['images'][0]
with Image.open(image) as reference:assert actual.tobytes()==reference.convert('RGB').tobytes()
results['historical_image_input']={'messages':msgs,'pixel_sha256':hashlib.sha256(actual.tobytes()).hexdigest(),'dimensions':list(actual.size),'image_mode':actual.mode,'same_pixels_as_source':True,'template_generation_prompt':processor.template_options['add_generation_prompt'],'scope':'real PIL decoding and actual route functions; capture processor is a fixture, no model capability claim'}
assert '<button id="transcribe" type="button">辨識語音</button>' in natural_ui.PAGE
assert '<button id="send" class="send" type="button">送出問題</button>' in natural_ui.PAGE
try:
 a.encode_messages(CaptureProcessor(),[{'role':'user','content':[{'type':'audio','audio':'fixture.wav'}]}],ROOT,generation_prompt=True)
 raise AssertionError('audio content should be rejected')
except ValueError as error:
 assert 'speech is transcribed separately' in str(error)
 results['direct_audio_rejected']={'observed':str(error),'no_audio_chat_input':True}
results['ui_labels']={'transcribe':'辨識語音','send':'送出問題','matches_revised_prose': '「辨識語音」與「送出問題」' in body}
cmd=['.venv/bin/python','-m','pytest','-q','tests/test_natural_ui.py::test_upload_and_chat_preserve_actual_image_and_every_turn','tests/test_natural_ui.py::test_speech_is_two_explicit_stations_and_correction_reaches_same_core']
r=subprocess.run(cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
print(r.stdout);assert r.returncode==0
results['http_route_tests']={'command':' '.join(cmd),'exit_code':r.returncode,'stdout':r.stdout,'scope':'actual HTTP/upload/history and correction paths; ASR/generation fixtures are not model ability evidence'}
results['environment']={'python':platform.python_version(),'torch':torch.__version__,'device':'cpu','pillow':importlib.metadata.version('Pillow'),'pytest':importlib.metadata.version('pytest')}
results['command']='.venv/bin/python docs/technical-reviews/artifacts/natural-20.1-cpu-check.py'
results['section_sha256']=hashlib.sha256(body.encode()).hexdigest()
(ART/'natural-20.1-cpu-check.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
