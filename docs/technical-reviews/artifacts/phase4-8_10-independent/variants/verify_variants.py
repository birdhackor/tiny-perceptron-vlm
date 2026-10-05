"""Bounded CPU checks; every judge label is supplied by hand, no judge/model call."""
from pathlib import Path
import json,hashlib,platform,sys,torch
base=Path(__file__).resolve().parent
torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()
human=torch.tensor([1,0,1,0],device='cpu')
results=[]
for name,predicted in [('original',[1,1,1,0]),('second_changed_to_zero',[1,0,1,0]),('all_zero_judge_counterexample',[0,0,0,0])]:
 judge=torch.tensor(predicted,device='cpu');agree=human==judge
 disagreement=(~agree).nonzero().flatten().tolist()
 observed={'case':name,'human':human.tolist(),'given_recorded_judge':judge.tolist(),'comparison':agree.tolist(),'comparison_dtype':str(agree.dtype),'converted_dtype':str(agree.float().dtype),'agreement_numerator':int(agree.sum().item()),'denominator':human.numel(),'agreement_fraction':agree.float().mean().item(),'disagreement_indices':disagreement,'judge_positive_count':int(judge.sum().item()),'human_positive_count':int(human.sum().item())}
 expected={'original':(3,0.75,[1]),'second_changed_to_zero':(4,1.0,[]),'all_zero_judge_counterexample':(2,0.5,[0,2])}[name]
 assert (observed['agreement_numerator'],observed['agreement_fraction'],disagreement)==expected
 assert agree.dtype==torch.bool and agree.float().dtype==torch.float32
 results.append(observed)
 print(json.dumps(observed,ensure_ascii=False))
# Exercise materials are independently composed and manually labelled by the section's rule.
exercise=[
 {'answer':'4。桌上左邊放了兩顆蘋果，右邊也放了兩顆蘋果；左邊兩顆和右邊兩顆合併到同一盤後，總共有四顆。因此兩組各兩顆，合起來就是四顆。','manual_rule_label':1,'reason':'Correct count; two groups of two and a concrete combining operation.'},
 {'answer':'4。數字是星光，星光是數字。數字是星光，星光是數字。數字是星光，星光是數字。數字是星光，星光是數字。數字是星光，星光是數字。','manual_rule_label':0,'reason':'Correct numeral alone; repetitive imagery has no two groups of two or combining explanation.'},
]
for row in exercise: row['unicode_code_points']=len(row['answer'])
print(json.dumps({'exercise_manual_examples':exercise,'scope':'Handwritten exercise materials and manual labels only. No automatic adjudication, training, model inference or measured length bias.'},ensure_ascii=False))
(base/'variant-results.json').write_text(json.dumps({'cases':results,'exercise_manual_examples':exercise},ensure_ascii=False,indent=2)+'\n')
(base/'environment.json').write_text(json.dumps({'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'torch':torch.__version__,'torch_git_version':torch.version.git_version,'cuda_build':str(torch.version.cuda),'cuda_available':torch.cuda.is_available(),'default_device':str(torch.get_default_device()),'torch_threads':torch.get_num_threads(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scope':'Short deterministic CPU tensors and manually supplied labels; no model, downloads, training or optimizer.'},indent=2)+'\n')
