import hashlib
import json
import platform
import sys
from pathlib import Path
import torch
ROOT=Path('/workspace/tiny-perceptron-vlm')
WORK=Path('/workspace/work/curriculum-20261016/T.3')
sys.path.insert(0,str(ROOT))
from scripts.check_course_models import validate_folder
from scripts.course_release import inference_payload
from tiny_perceptron.simple import BigramLM,ContextMLP
from tiny_perceptron.data import split_documents,toy_documents

torch.set_num_threads(2)
torch.manual_seed(42)
contexts=torch.tensor([[0,2,3],[4,5,3]])
bigram=BigramLM(17)
mlp=ContextMLP(17,3,16)
manifest=json.loads((ROOT/'docs/course-experiments/public-models.json').read_text())
spec=next(x for x in manifest['models'] if x['id']=='text_foundation')
public_weights=validate_folder(spec,WORK/'course-models/text_foundation')
clean=inference_payload({'format_version':1,'config':{},'model':{},'optimizer':{'step':2},'torch_rng':torch.tensor([1]),'training_state':{'sampler_rng':1}}, {'revision':'review-synthetic-fixture'})
history_path=ROOT/'docs/course-experiments/results/simple_models.json'
history=json.loads(history_path.read_text())
current=json.loads((WORK/'course-experiments/simple_models/result.json').read_text())
fields=['parameters','context','steps']
history_check={'path':str(history_path.relative_to(ROOT)),'sha256':hashlib.sha256(history_path.read_bytes()).hexdigest(),'revision':history['revision'],'device':history['device'],'python_version':history['python_version'],'torch_version':history['torch_version'],'step_scale':history['step_scale'],'evidence_status':history['evidence_status'],'seed':history['seed'],'split_counts':{k:history['results']['data'][k]['records'] for k in ['train','validation','test']},'runs':{k:{f:v[f] for f in fields} for k,v in history['results']['runs'].items()},'all_current_configs_match':all(history['results']['runs'][k][f]==current['results']['runs'][k][f] for k in history['results']['runs'] for f in fields),'split_fingerprints_match':all(history['results']['data'][k]['sha256']==current['results']['data'][k]['sha256'] for k in ['train','validation','test'])}
record={'environment':{'python':platform.python_version(),'torch':str(torch.__version__),'platform':platform.platform(),'device':'cpu'},'bigram_same_last_character_equal_logits':torch.equal(bigram(contexts)[0],bigram(contexts)[1]),'mlp_context_shape':list(contexts.shape),'mlp_different_prefix_max_difference':float((mlp(contexts)[0]-mlp(contexts)[1]).abs().max().detach()),'dedup_counts':{k:len(v) for k,v in split_documents(toy_documents()+toy_documents(),42).items()},'text_foundation_public_weights_all_validated':public_weights,'synthetic_export_stripping':{'input_contains_optimizer_rng_training_state':True,'output_keys':list(clean),'stripped':all(k not in clean for k in ['optimizer','torch_rng','training_state'])},'history_check':history_check}
assert record['bigram_same_last_character_equal_logits']
assert record['mlp_different_prefix_max_difference']>0
assert history_check['all_current_configs_match'] and history_check['split_fingerprints_match']
p=ROOT/'docs/technical-reviews/artifacts/revision-20261016/T.3-supplement.json'
p.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print(p.read_text())

# Retain only short quotations from the already inspected public model card.
p=ROOT/'docs/technical-reviews/artifacts/revision-20261016/T.3-execution.json'
r=json.loads(p.read_text())
if isinstance(r.get('public_card'),str):
 card=r['public_card']
 r['public_card']={'sha256':hashlib.sha256((WORK/'course-models/text_foundation/README.md').read_bytes()).hexdigest(),'excerpts':[line for line in card.splitlines() if 'from-scratch' in line or 'optimizer' in line]}
 p.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
