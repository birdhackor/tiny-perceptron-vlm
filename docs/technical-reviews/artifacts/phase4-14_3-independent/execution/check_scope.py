"""Read only explicit metadata/configuration/provenance pointers, never interpretation strings."""
from pathlib import Path
import ast
import hashlib
import json

root=Path('/workspace/tiny-perceptron-vlm')
base=root/'docs/technical-reviews/artifacts/phase4-14_3-independent'
p=base/'frozen/docs/course-experiments/results/modern.json'
data=json.loads(p.read_text())
checked={}
checked['/experiment_id']=data['experiment_id']
checked['/revision']=data['revision']
variants=data['results']['variants']
checked['/results/variants (keys only)']=list(variants)
assert set(variants)=={'baseline','rope','rmsnorm','relu2','swiglu','tied'}
for name,record in variants.items():
    checked[f'/results/variants/{name}/changes']=record['changes']
    checked[f'/results/variants/{name}/model/config']=record['model']['config']
    assert not any('qk' in k.lower() or 'q_norm' in k.lower() or 'k_norm' in k.lower() for k in record['changes'])
    assert not any('qk' in k.lower() or 'q_norm' in k.lower() or 'k_norm' in k.lower() for k in record['model']['config'])
for path in ('tiny_perceptron/model.py','tiny_perceptron/attention.py','scripts/course_experiments/architecture.py'):
    checked['/code_sha256/'+path]=data['code_sha256'][path]
    digest=hashlib.sha256((base/'frozen'/path).read_bytes()).hexdigest()
    checked['frozen_file_sha256:'+path]=digest
    if path != 'scripts/course_experiments/architecture.py':
        assert digest==data['code_sha256'][path]
    else:
        original=base/'sources/architecture-result-revision.py'
        recorded_digest=hashlib.sha256(original.read_bytes()).hexdigest()
        checked['git_result_revision_source_sha256:'+path]=recorded_digest
        assert recorded_digest==data['code_sha256'][path]
        oldtree=ast.parse(original.read_text())
        method=next(n for n in oldtree.body if isinstance(n,ast.FunctionDef) and n.name=='run_modern')
        variants_node=next(n for n in method.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='variants' for t in n.targets))
        method_variants=ast.literal_eval(variants_node.value)
        assert method_variants=={name:record['changes'] for name,record in variants.items()}
        checked['recorded_revision_run_modern_variants']=method_variants
model_tree=ast.parse((base/'frozen/tiny_perceptron/model.py').read_text())
config=next(n for n in model_tree.body if isinstance(n,ast.ClassDef) and n.name=='ModelConfig')
fields=[n.target.id for n in config.body if isinstance(n,ast.AnnAssign)]
assert not any('qk' in k.lower() or 'q_norm' in k.lower() or 'k_norm' in k.lower() for k in fields)
print(json.dumps({'checked_pointers':checked,'model_config_fields':fields,'raw_result_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'scope':'Only experiment identity, variant names, changes, configs and necessary source provenance. No measurements, training or heldout values and no author result interpretation were read. TinyLM forward nodes were separately inspected by the reviewer.','checks':'all assertions passed; no model instantiation, training or evaluation'},ensure_ascii=False,indent=2))
