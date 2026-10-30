"""Independent verbatim-AST replay of installed4.57.6 loader/cached_file and dtype guard; no neural imports."""
import ast,copy,datetime,hashlib,json,os,sys,types,warnings
from pathlib import Path
BASE=Path(__file__).resolve().parent;source=Path('/workspace/work/omni-qwen25-cpu-api-venv/lib/python3.12/site-packages/transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py');hub=source.parents[2]/'utils/hub.py';tree=ast.parse(source.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Qwen2_5OmniForConditionalGeneration');method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='from_pretrained');generate=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='generate')
root=BASE/'probes/current-v2-qwen-upstream';root.mkdir(parents=True,exist_ok=False);local=root/'fixed-local-snapshot';local.mkdir();speaker=local/'spk_dict.pt';speaker.write_bytes(b'explicit own fixture bytes; not a torch checkpoint or neural weights')
super_calls=[];speaker_calls=[];loads=[]
class DummyLoaded:
 def load_speakers(self,path):loads.append({'path':str(path),'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),'bytes':Path(path).stat().st_size})
class Base:
 @classmethod
 def from_pretrained(cls,identifier,*args,**kwargs):super_calls.append({'identifier':str(identifier),'revision':kwargs.get('revision'),'local_files_only':kwargs.get('local_files_only'),'token':kwargs.get('token')});return DummyLoaded()

def capture(identifier,filename,**kwargs):speaker_calls.append({'identifier':str(identifier),'filename':filename,**kwargs});return str(speaker)
namespace={'Base':Base,'cached_file':capture};derived=ast.ClassDef(name='ExactLoaderReplay',bases=[ast.Name(id='Base',ctx=ast.Load())],keywords=[],body=[copy.deepcopy(method)],decorator_list=[]);module=ast.Module(body=[derived],type_ignores=[]);ast.fix_missing_locations(module);exec(compile(module,'verbatim-upstream-from_pretrained-AST','exec'),namespace)
revision='f75b40e3da2003cdd6e1829b1f420ca70797c34e';namespace['ExactLoaderReplay'].from_pretrained('Qwen/Qwen2.5-Omni-3B',revision=revision,local_files_only=True,token=False,torch_dtype='argument-binding-only-bfloat16-label')
assert super_calls[0]['revision']==revision and super_calls[0]['local_files_only'] is True
assert speaker_calls[0]['revision'] is None and speaker_calls[0]['local_files_only'] is False
# Replay the real installed cached_file/cached_files definitions through their local-folder early return.
hubtree=ast.parse(hub.read_text());definitions=[copy.deepcopy(next(x for x in hubtree.body if isinstance(x,ast.FunctionDef) and x.name==n)) for n in ('cached_file','cached_files')];future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0);hubmodule=ast.Module(body=[future,*definitions],type_ignores=[]);ast.fix_missing_locations(hubmodule);hubns={'os':os,'Path':Path,'warnings':warnings,'is_offline_mode':lambda:False};exec(compile(hubmodule,'verbatim-upstream-cached_file-AST','exec'),hubns)
namespace['cached_file']=hubns['cached_file'];namespace['ExactLoaderReplay'].from_pretrained(str(local),revision=revision,local_files_only=True,token=False)
assert loads[-1]['path']==str(speaker) and loads[-1]['sha256']==hashlib.sha256(speaker.read_bytes()).hexdigest()
# Only the exact upstream conditional is executed on a non-neural fake component.
condition=next(n for n in ast.walk(generate) if isinstance(n,ast.If) and ast.unparse(n.test)=='self.token2wav.dtype != torch.float')
class Component:
 dtype='torch.bfloat16'
 def float(self):self.dtype='torch.float32';return self
component=Component();dtype_ns={'self':types.SimpleNamespace(token2wav=component),'torch':types.SimpleNamespace(float='torch.float32')};guard=ast.Module(body=[copy.deepcopy(condition)],type_ignores=[]);ast.fix_missing_locations(guard);exec(compile(guard,'verbatim-token2wav-float-guard','exec'),dtype_ns);assert component.dtype=='torch.float32'
record={'captured_at_utc':datetime.datetime.now(datetime.UTC).isoformat(),'scope':'Verbatim methods extracted by AST from installed4.57.6 sources and executed in independent fake superclass/cached_file namespace. The real-class import/weight loader is not used. No model or parameter is created, no checkpoint parsed, no network/provider/NN/final call.','sources':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in (source,hub)],'remote_identifier_case':{'superclass':super_calls[0],'speaker_cached_file':speaker_calls[0],'revision_lost':True,'explicit_local_only_also_lost':True},'local_snapshot_control':{'real_cached_file_source_local_branch_executed':True,'local_file_exact':True,'fixture_load':loads[-1]},'precision_guard':{'upstream_condition':ast.unparse(condition.test),'upstream_action':'self.token2wav.float()','fake_input_dtype':'torch.bfloat16','fake_output_dtype':component.dtype,'actual_model_component_dtype_measured':False},'real_neural_or_paid_or_final_calls':0}
p=BASE/'current-v2-qwen-upstream-result.json';p.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');print(json.dumps(record,ensure_ascii=False,indent=2))
