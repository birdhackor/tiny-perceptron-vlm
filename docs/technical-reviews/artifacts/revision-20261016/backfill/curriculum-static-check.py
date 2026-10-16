"""Execute narrow source-contract and original-release-metadata inspections."""

import ast
import hashlib
import json
import platform
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
WORK = Path('/workspace/work/curriculum-20261016/curriculum-backfill')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


source = (ROOT / 'scripts/selftrained/train.py').read_text()
tree = ast.parse(source)
export = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'export_inference')
tensors = next(n.value for n in export.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'tensors' for t in n.targets))
save = next(n.value for n in export.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name) and n.value.func.id == 'save_file')
checkpoint = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'checkpoint')
full = next(n.value for n in checkpoint.body if isinstance(n, ast.Assign) and isinstance(n.value, ast.Dict))
keys = [k.value for k in full.keys if isinstance(k, ast.Constant)]
release = json.loads((ROOT / 'docs/selftrained/v2-jobs/release-final-four-exports.json').read_text())['release']
spec = {f'selftrained/v2/{g["name"]}/{f["path"]}': f for g in release['exports'] for f in g['files']}
remote = {r['path']:r for r in json.loads((WORK/'hfv2.txt').read_text()) if r['type']=='file'}
deps = tomllib.loads((ROOT/'pyproject.toml').read_text())['dependency-groups']['notebook']
publisher = (ROOT/'scripts/export_course.py').read_text()
products_readme = (ROOT/'docs/lesson-products/README.md').read_text()
pub = json.loads((ROOT/'docs/lesson-products/results/publication-v1.json').read_text())['execution']['public']
out = {
    'environment': {'python': platform.python_version(), 'device':'CPU, static AST/text and fetched original metadata only'},
    'scope':'This executes inspection of the actual source contracts; it does not execute training/export production or rerun notebooks/Colab.',
    'safe_export': {'tensor_input':ast.unparse(tensors.generators[0].iter), 'save_first_argument':ast.unparse(save.args[0]), 'checkpoint_state_keys_present':{k:k in keys for k in ('model','optimizer','rng','sampler')}},
    'v2_public': {'declared_files':len(spec),'observed_files':len(remote),'same_paths':set(spec)==set(remote),'all_declared_MIT':all(f['license']=='MIT' for f in spec.values()),'all_sizes_match':all(spec[p]['bytes']==r['size'] for p,r in remote.items()),'safe_weight_hashes_match':{p:spec[p]['sha256']==r['lfs']['oid'] for p,r in remote.items() if p.endswith('.safetensors')}},
    'notebook_dependencies':deps,
    'publisher_same_section_colab': "{item['notebook']}" in publisher and 'colab.research.google.com' in publisher,
    'publisher_notebook_download': '下載本節 .ipynb' in publisher and ' download>' in publisher,
    'new_product_readme_matches_fixed_public_revision':pub['revision'] in products_readme,
    'source_sha256':{p:sha(ROOT/p) for p in ('scripts/selftrained/train.py','docs/selftrained/v2-jobs/release-final-four-exports.json','pyproject.toml','scripts/export_course.py','docs/lesson-products/README.md','docs/lesson-products/results/publication-v1.json')},
    'original_HF_tree_sha256':sha(WORK/'hfv2.txt'),
}
assert out['safe_export']['tensor_input']=="checkpoint['model'].items()"
assert out['safe_export']['save_first_argument']=='tensors'
assert all(out['safe_export']['checkpoint_state_keys_present'].values())
assert all(out['v2_public'][k] for k in ('same_paths','all_declared_MIT','all_sizes_match'))
assert all(out['v2_public']['safe_weight_hashes_match'].values())
assert all(out[k] for k in ('publisher_same_section_colab','publisher_notebook_download','new_product_readme_matches_fixed_public_revision'))
Path(__file__).with_name('curriculum-static-check-output.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(out,ensure_ascii=False,indent=2))
