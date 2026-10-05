import ast,contextlib,io,json,sys,hashlib,itertools,subprocess
from pathlib import Path
import torch
root=Path(__file__).resolve().parent
repo=root.parents[3]
sys.path.insert(0,str(repo))
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.multimodal import generate_modal

def extracted_function(path,name,parent=None):
 tree=ast.parse(path.read_bytes())
 if parent:
  tree=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==parent)
 node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
 unit=ast.Module(body=[node],type_ignores=[])
 namespace={}
 exec(compile(unit,str(path), 'exec'),namespace)
 return namespace[name],{'path':str(path.relative_to(root)),'function':name,'lines':[node.lineno,node.end_lineno],'file_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
original_distance,repo_receipt=extracted_function(root/'inputs/scripts/course_experiments/modalities.py','edit_distance','run_ocr')
official_distance,official_receipt=extracted_function(root/'sources/torchmetrics-helper-v1.8.2.py','_edit_distance')
code=(root/'inputs/fence-1.py').read_bytes()
namespace={}; buf=io.StringIO()
with contextlib.redirect_stdout(buf): exec(compile(code,'original-fence-1','exec'),namespace)
assert namespace['matched']==3
variants=[]
for prediction,edits in [('1010',0),('10100',1),('010',1),('101',1)]:
 tree=ast.parse(code)
 next(n for n in tree.body if isinstance(n,ast.Assign) and n.targets[0].id=='predicted').value=ast.Constant(prediction)
 # The exercise explicitly changes the known edits for the correct-answer variation.
 if prediction=='1010':
  tree.body[-2].value.args[1].left=ast.Constant(0)
 local={}; buf2=io.StringIO()
 try:
  with contextlib.redirect_stdout(buf2):exec(compile(ast.fix_missing_locations(tree),'bounded-variant','exec'),local)
  assert prediction=='1010' and local['matched']==4
  status='executed_equal_length'
 except AssertionError:
  assert len(prediction)!=len('1010');status='assert_rejected_unequal_length'
 observed=original_distance(prediction,'1010')
 assert observed==edits==official_distance(list(prediction),list('1010'))
 variants.append({'truth':'1010','prediction':prediction,'fence_status':status,'stdout':buf2.getvalue(),'edits':observed,'reference_characters':4,'cer':observed/4})
# Equal-length strings can still be shifted; positional error and edit distance differ.
assert sum(a!=b for a,b in zip('0101','1010'))==4
assert original_distance('0101','1010')==2
assert official_distance(list('0101'),list('1010'))==2
weighted=[(bits,.9**sum(bits)*.1**(5-sum(bits))) for bits in itertools.product((0,1),repeat=5)]
p_all=sum(w for bits,w in weighted if all(bits))
marginals=[sum(w for bits,w in weighted if bits[i]) for i in range(5)]
assert abs(p_all-.59049)<1e-12 and all(abs(x-.9)<1e-12 for x in marginals)
probability={'independent_all_correct':p_all,'rounded':round(p_all,4),'independent_per_position':marginals,'fully_correlated_all_correct':.9,'fully_correlated_per_position':[.9]*5,'scope':'Correlated example is an exact two-outcome distribution: all five correct with mass 0.9, all five wrong with mass 0.1.'}

d=json.loads((root/'inputs/docs/course-experiments/results/ocr.json').read_bytes())
pointers=['/schema_version','/experiment_id','/revision','/device','/seed','/torch_version','/python_version','/step_scale','/elapsed_seconds','/code_sha256','/artifacts','/assets','/results/training/config','/results/training/modal_config','/results/training/steps','/results/data/seed','/results/data/split_policy','/results/data/splits/test/count','/results/data/splits/test/sha256','/results/data/splits/test/records']
metric_names=('examples','correct','exact_match','effective_tokens','eos_rate','generation_errors','invalid_special_tokens','character_edits','reference_characters','character_error_rate')
pointers += ['/results/test/'+k for k in metric_names]
sample_fields=('row','family','question','target','generated','generated_ids','exact_match','eos','generation_error','invalid_special_tokens','donor_row')
pointers += ['/results/test/samples/*/'+k for k in sample_fields]
samples=d['results']['test']['samples'];tok=ByteTokenizer(); details=[]
for s,row in zip(samples,d['results']['data']['splits']['test']['records'],strict=True):
 ids=s['generated_ids'];eos=tok.eos_id in ids
 raw=ids[:ids.index(tok.eos_id)] if eos else ids
 generated=tok.decode(raw);exact=raw==tok.encode(s['target'])
 assert s['target']==row['answer'] and s['family']==row['family'] and s['question']==row['question']
 assert generated==s['generated'] and eos==s['eos'] and exact==s['exact_match']
 assert generated.isascii() and s['target'].isascii()
 distance=original_distance(generated,s['target'])
 assert distance==official_distance(list(generated),list(s['target']))
 assert eos and ids[-1]==tok.eos_id and s['generation_error'] is None
 assert s['invalid_special_tokens']==sum(i<8 for i in raw)==0
 details.append({'row':s['row'],'family':s['family'],'target':s['target'],'generated':generated,'generated_ids':ids,'eos':eos,'exact_match':exact,'edit_distance':distance,'reference_characters':len(s['target'])})
observed={'examples':len(samples),'correct':sum(s['exact_match'] for s in details),'eos_count':sum(s['eos'] for s in details),'character_edits':sum(s['edit_distance'] for s in details),'reference_characters':sum(s['reference_characters'] for s in details),'effective_tokens':sum(len(tok.encode(s['target']))+1 for s in details)}
observed.update(character_error_rate=observed['character_edits']/observed['reference_characters'],exact_match=observed['correct']/observed['examples'],eos_rate=observed['eos_count']/observed['examples'])
for k in ('examples','correct','character_edits','reference_characters','effective_tokens','character_error_rate','exact_match','eos_rate'):assert observed[k]==d['results']['test'][k],(k,observed[k],d['results']['test'][k])
assert observed['examples']==d['results']['data']['splits']['test']['count']==30
assert observed['correct']==2 and observed['eos_count']==30 and observed['character_edits']==40 and observed['reference_characters']==57
assert round(100*observed['character_error_rate'],1)==70.2

# A bounded synthetic output sequence checks the original EOS stop branch; no real weights or media are used.
class FakeModel(torch.nn.Module):
 def __init__(self,outputs): super().__init__(); self.outputs=outputs; self.calls=0; self.language=type('Language',(),{'config':type('Config',(),{'max_length':128})()})()
 def forward(self,ids,**kwargs):
  logits=torch.full((1,len(ids),tok.vocab_size),-1000.0)
  logits[0,-1,self.outputs[self.calls]]=1000.0;self.calls+=1
  return {'logits':logits}
fake=FakeModel(tok.encode('10')+[tok.eos_id]+tok.encode('1'))
prefix=torch.tensor([tok.bos_id,tok.user_id,tok.assistant_id])
output=generate_modal(fake,prefix,max_new_tokens=5)
assert output[len(prefix):].tolist()==tok.encode('10')+[tok.eos_id] and fake.calls==3 and fake.training
result={'kind':'bounded_cpu_and_historical_measurement_verification','environment':{'python':sys.version,'executable':sys.executable,'torch':str(torch.__version__),'torch_cuda_build':str(torch.version.cuda),'cuda_available':str(torch.cuda.is_available()),'device':'cpu','threads':str(torch.get_num_threads())},'original_fence_stdout':buf.getvalue(),'variants':variants,'equal_length_shift_counterexample':{'reference':'1010','prediction':'0101','positional_errors':4,'minimum_edits':2},'independence':probability,'original_functions_executed':[repo_receipt,official_receipt],'json_original_sha256':hashlib.sha256((root/'inputs/docs/course-experiments/results/ocr.json').read_bytes()).hexdigest(),'inspected_json_pointers':pointers,'historical_provenance':{k:d[k] for k in ('experiment_id','revision','device','seed','torch_version','python_version','step_scale')},'historical_training_steps':d['results']['training']['steps'],'historical_config':d['results']['training']['config'],'historical_modal_config':d['results']['training']['modal_config'],'test_split_seed':d['results']['data']['seed'],'test_split_policy':d['results']['data']['split_policy'],'observed':observed,'recomputed_samples':details,'eos_synthetic_execution':{'generated_ids':output.tolist(),'answer_token_ids':output[len(prefix):].tolist(),'calls':fake.calls,'training_mode_restored':fake.training},'scope':'No existing-model evaluation, checkpoint load, model save, training, GPU operation, or data/model download. Historical JSON output is recomputed from stored raw samples.'}
(root/'bounded-results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps({'original_fence':buf.getvalue(),'variants':variants,'observed':observed,'probability':probability,'synthetic_eos_calls':fake.calls,'source_pointers':len(pointers)},ensure_ascii=False,indent=2))
