from pathlib import Path
import ast,hashlib,importlib.util,json,platform,sys
ROOT=Path(__file__).resolve().parents[4];OUT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
import torch
from tiny_perceptron.selftrained.model import LimitedAssistant,SelftrainedConfig
from tiny_perceptron.selftrained.dataset import train_tokenizer
from tiny_perceptron.selftrained.inference import fetch_public_export
torch.set_num_threads(2)
config=SelftrainedConfig(vocab_size=32,width=16,layers=1,heads=2,kv_heads=1,ffn_hidden=32,experts=2,top_k=1,max_length=512)
torch.manual_seed(17);a=LimitedAssistant(config).state_dict()
torch.manual_seed(17);b=LimitedAssistant(config).state_dict()
torch.manual_seed(18);c=LimitedAssistant(config).state_dict()
assert all(torch.equal(a[k],b[k]) for k in a)
changed=[k for k in a if not torch.equal(a[k],c[k])]
assert any(k.startswith('lm.') for k in changed)
assert all(any(k.startswith(prefix) for k in changed) for prefix in ['vision_encoder.','ocr_encoder.','audio_encoder.'])
tok=train_tokenizer([{'split':'train','messages':[{'content':'甲乙'}]}, {'split':'test','messages':[{'content':'🔒'}]}])
assert tok.unk_id not in tok.encode('甲乙') and tok.unk_id in tok.encode('🔒')
rejections=[]
for revision,prefix in [('main','selftrained/v2/moe-joint'),('979cdfacc588ad0536f1c64fff96f264571cf054','../bad')]:
    try:fetch_public_export('birdhackor/tiny-perceptron-course-models',revision,'/tmp/t11-unused',prefix=prefix)
    except ValueError as e:rejections.append(str(e))
    else:raise AssertionError('bad immutable identity unexpectedly accepted')
functions=[]
for name,names in [('tiny_perceptron/natural_assistant.py',['load_core','load_asr']),
                  ('scripts/selftrained/modal_runner.py',['native_joint_source_gate','native_target_source_options_gate'])]:
    p=ROOT/name;tree=ast.parse(p.read_text())
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name in names:
            calls=[ast.unparse(x.func) for x in ast.walk(node) if isinstance(x,ast.Call)]
            functions.append({'path':name,'function':node.name,'lines':[node.lineno,node.end_lineno],'calls':calls})
            if node.name in ['load_core','load_asr']:assert any('from_pretrained' in s for s in calls)
            else:assert not any(s.endswith(('.remote','.spawn','.get')) and s.startswith(('modal','function')) for s in calls)
    q=OUT/'code'/name;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(p.read_bytes())
for name in ['tiny_perceptron/model.py','tiny_perceptron/selftrained/dataset.py']:
    p=ROOT/name;q=OUT/'code'/name;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(p.read_bytes())
result={'environment':{'python':platform.python_version(),'torch':torch.__version__,'device':'cpu'},
        'random_init':{'same_seed_all_equal':True,'different_seed_changed_tensors':len(changed),'changed_families':['lm','vision_encoder','ocr_encoder','audio_encoder'],'config':config.__dict__},
        'tokenizer_train_only':{'train_ids':tok.encode('甲乙'),'heldout_char_ids':tok.encode('🔒'),'unk_id':tok.unk_id},
        'public_identity_rejections':rejections,'ast_contracts':functions,
        'scope':'construction/tokenizer and pre-download guards only; native training gate audited as local filesystem contract; no actual production restart or Qwen/Whisper load'}
(OUT/'contract-audit-result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['environment','random_init','tokenizer_train_only','public_identity_rejections']},ensure_ascii=False,indent=2))
