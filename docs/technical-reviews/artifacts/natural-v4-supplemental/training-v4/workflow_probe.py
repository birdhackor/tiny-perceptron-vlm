"""Exercise actual train/resume/evaluation control flow with disclosed tiny doubles."""
from pathlib import Path
from types import SimpleNamespace as NS
import sys,tempfile,json,copy,hashlib,contextlib
ROOT=Path(__file__).resolve().parents[5];sys.path.insert(0,str(ROOT))
import torch
from torch import nn
from tiny_perceptron import natural_assistant as c
torch.set_num_threads(1)
class Processor:
    def apply_chat_template(self,messages,tokenize=False,add_generation_prompt=False):
        return ''.join('<'+m['role']+'>'+m['content'][0]['text']+(';' if m['role']=='assistant' else '') for m in messages)+('<assistant>' if add_generation_prompt else '')
    def __call__(self,text,return_tensors):
        ids=torch.tensor([[ord(x)%128 for x in text[0]]]);return {'input_ids':ids,'attention_mask':torch.ones_like(ids)}
class Tiny(nn.Module):
    def __init__(self):
        super().__init__();self.embedding=nn.Embedding(128,4);self.base=nn.Linear(4,128);self.embedding.requires_grad_(False);self.base.requires_grad_(False);self.lora_A=nn.Parameter(torch.randn(2,4)*.01);self.lora_B=nn.Parameter(torch.zeros(128,2));self.active_adapter='default';self.config=NS()
    def forward(self,input_ids,labels,**kwargs):
        x=self.embedding(input_ids);logits=self.base(x)+x@self.lora_A.T@self.lora_B.T
        return NS(loss=nn.functional.cross_entropy(logits[:,:-1,:].reshape(-1,128),labels[:,1:].reshape(-1),ignore_index=-100))
    def save_pretrained(self,path,**kwargs):
        Path(path).mkdir(parents=True,exist_ok=True);torch.save(self.state_dict(),Path(path)/'toy.pt')
    def disable_adapter(self):return contextlib.nullcontext()
    def set_adapter(self,name):self.active_adapter=name
    def load_adapter(self,path,adapter_name,is_trainable=False):pass
def load(options,adapter=None,train=False):
    model=Tiny()
    if adapter:model.load_state_dict(torch.load(Path(adapter)/'toy.pt',weights_only=True,map_location='cpu'))
    if not train:model.requires_grad_(False)
    return model,Processor()
trainrows=[{'id':f'tiny{i}','split':'train','user':'Q'+str(i),'answer':'A'* (i+1)} for i in range(3)]
manifest={'schema_version':1,'dataset_version':'offline-double','rows':trainrows,'audio_rows':[],'manifest_sha256':'0'*64,'asset_sha256':{}}
c.load_manifest=lambda *args:(manifest,ROOT)
c.load_core=load
def options(path,steps=3,adapter=None,cp=(1,3)):
    return NS(output=path,manifest='fixture',data_root=ROOT,steps=steps,checkpoint_steps=cp,seed=42,adapter=adapter,model='tiny-double',model_revision='offline',asr_model='none',asr_revision='none',device='cpu',dtype='float32',max_pixels=1,min_pixels=1,max_tokens=2048,cache_dir=None,local_files_only=True,lora_rank=2,gradient_accumulation=2,learning_rate=3e-5,max_seconds=100,checkpoint_every=1,max_new_tokens=384)
with tempfile.TemporaryDirectory(prefix='training-v4-workflow-') as tmp:
    root=Path(tmp);whole=c.run_train(options(root/'whole'))
    first=c.run_train(options(root/'first',steps=1,cp=(1,))); old=hashlib.sha256((root/'first/checkpoints/step-000001/toy.pt').read_bytes()).hexdigest()
    resumed=c.run_train(options(root/'resume',adapter=root/'first/adapter',cp=(3,)))
    w=torch.load(root/'whole/adapter/toy.pt',weights_only=True);r=torch.load(root/'resume/adapter/toy.pt',weights_only=True)
    maxdiff=max(float((w[k]-r[k]).abs().max()) for k in w)
    assert maxdiff==0 and whole['history'][-1]['row_ids']==resumed['history'][-1]['row_ids']
    assert old==hashlib.sha256((root/'first/checkpoints/step-000001/toy.pt').read_bytes()).hexdigest()
    try:c.archive_checkpoint(load(None)[0],torch.optim.SGD([nn.Parameter(torch.ones(1))],lr=.1),root/'first',first);raise AssertionError('overwrite allowed')
    except ValueError as exc:archive_error=str(exc)
    bad=options(root/'bad',adapter=root/'first/adapter',cp=(3,));bad.learning_rate=1e-4
    try:c.run_train(bad);raise AssertionError('changed LR accepted')
    except ValueError as exc:resume_error=str(exc)
    trained={'whole_completed':whole['completed_steps'],'whole_trained_rows':whole['trained_rows'],'resumed_completed_total':resumed['completed_steps'],'resumed_trained_total':resumed['trained_rows'],'prior_completed':resumed['resume_from']['completed_steps'],'whole_vs_resume_parameter_max_difference':maxdiff,'frozen_samples_equal':whole['frozen_parameter_samples_unchanged'] and resumed['frozen_parameter_samples_unchanged'],'original_archive_unchanged':True,'archived_checkpoint_rejection':archive_error,'changed_resume_lr_rejection':resume_error,'scope':'Actual run_train/save_checkpoint/archive_checkpoint/optimizer+RNG resume on CPU 3-row tiny double, not full VLM training or GPU evidence'}
    # Run actual evaluation variant/shared-transcript selection logic with generated-text doubles.
    manifest['rows']=[{'id':'visual0','split':'validation','task':'chat','user':'hello','answer':'hi'}]
    manifest['audio_rows']=[{'id':'voice0','split':'validation','task':'speech_chat','user':'source transcript','answer':'hi','audio':'unused'}]
    c.asset_path=lambda *args:Path('unused')
    calls=[];asr_calls=[]
    c.load_asr=lambda *args:('fake-asr','fake-processor')
    def transcribe(*args):
        asr_calls.append('once');return {'transcript':'actual hypothesis','truncated':False,'completion_unknown':False,'stop_reason':'eos'}
    c.transcribe=transcribe
    def generate(model,processor,row,data_root,opt):
        calls.append({'task':row['task'],'user':row['user'],'adapter':model.active_adapter});return {'id':row['id'],'task':row['task'],'user':row['user'],'prediction':'hi','score':None}
    c.generate=generate
    ev=options(root/'eval',adapter=root/'first/adapter',cp=());ev.split='validation';ev.selected_only=False;ev.comparison_adapters=[('adapter-step-000001',root/'first/adapter'),('adapter-step-000003',root/'resume/adapter')]
    allvariants=c.run_evaluate(ev)
    assert len(asr_calls)==1 and len(calls)==9 and len(allvariants['variants'])==3
    assert [x['user'] for x in calls if x['task']=='speech_chat']==['actual hypothesis']*3
    assert [x['user'] for x in calls if x['task']=='typed_chat']==['source transcript']*3
    single=options(root/'selected',adapter=root/'resume/adapter',cp=());single.split='validation';single.selected_only=True;single.comparison_adapters=[]
    selected=c.run_evaluate(single);assert list(selected['variants'])==['adapter']
    base=options(root/'base',adapter=None,cp=());base.split='validation';base.selected_only=True;base.comparison_adapters=[]
    base_only=c.run_evaluate(base);assert list(base_only['variants'])==['base']
    print(json.dumps({'environment':{'python':sys.version.split()[0],'torch':torch.__version__,'device':'cpu'},'training_resume':trained,'evaluation_control':{'three_variant_names':list(allvariants['variants']),'one_shared_asr_generation_per_three_variants':True,'first_three_variant_calls':calls[:9],'selected_adapter_names':list(selected['variants']),'selected_base_names':list(base_only['variants']),'scope':'Actual run_evaluate flow, mocked model/ASR text; no new generation quality or speech recognition claim'}},ensure_ascii=False,indent=2))
