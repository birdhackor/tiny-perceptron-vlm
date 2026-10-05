from pathlib import Path
import ast
import collections
import datetime
import hashlib
import json
import platform
import re

ART = Path(__file__).resolve().parent
ROOT = ART.parents[4]
sha = lambda raw: hashlib.sha256(raw).hexdigest()
old_report_raw = (ROOT/'docs/technical-reviews/11.12.json').read_bytes()
assert sha(old_report_raw) == 'ad8a5d658e3ec48cf9bf467180012e4e7ee4fa3104e88f6a48563774a6bfb9d8'
assert old_report_raw == (ROOT/'docs/technical-reviews/history/phase4-11_12-own-family-unit-revise-ad8a5d658e3ec48cf9bf467180012e4e7ee4fa3104e88f6a48563774a6bfb9d8.json').read_bytes()
report = json.loads(old_report_raw)
verified_artifacts = []
for artifact in report['artifacts']:
    assert sha((ROOT/artifact['path']).read_bytes()) == artifact['sha256'], artifact['id']
    verified_artifacts.append({'id':artifact['id'],'path':artifact['path'],'sha256':artifact['sha256']})

section = (ART/'current-section.md').read_bytes()
assert sha(section) == '2c27af389ebc2c9ce31d2120351ccdb1fecc38142e4c4e01340af51ca41daf12'
all_fences = re.findall(rb'(?ms)^```python\r?\n(.*?)^```\s*$', section)
assert len(all_fences) == 1
fence = all_fences[0]
assert fence == (ART.parent/'original/fence-1.py').read_bytes()
figure = ROOT/'course/figures/rewrite-11-glyph-labels.svg'
assert sha(figure.read_bytes()) == report['figure_sha256'][figure.relative_to(ROOT).as_posix()]
original_hashes = {}
for source in report['sources']:
    if source.get('kind') == 'repository_code':
        path = ROOT/source['path']
        assert sha(path.read_bytes()) == source['sha256']
        original_hashes[source['path']] = source['sha256']

raw_result = (ROOT/'docs/course-experiments/results/ocr.json').read_bytes()
j = json.loads(raw_result)
pointers = []
coverage = {}
for split in ('train','validation','test'):
    pointer = '/results/data/splits/'+split+'/records'
    pointers.append(pointer)
    records = j['results']['data']['splits'][split]['records']
    families = collections.defaultdict(list)
    chars = set()
    for row in records:
        assert row['family'] == row['answer'] == row['digits']
        families[row['family']].append(row['offset'])
        chars.update(row['answer'])
    assert all(sorted(offsets)==[-1,0,1] for offsets in families.values())
    coverage[split] = {'question_count':len(records),'string_family_count':len(families),
                       'string_families':sorted(families),'character_classes':sorted(chars),'offsets_per_family':[-1,0,1]}
assert set(coverage['train']['character_classes']) == set('0123456789')
assert not set(coverage['test']['character_classes'])-set(coverage['train']['character_classes'])
assert not set(coverage['train']['string_families']) & set(coverage['test']['string_families'])

tree = ast.parse(fence)
glyphs = ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='glyphs' for t in n.targets)))
assert set(glyphs) == {'0','1'}
class_coverage_if_source_0_heldout = sorted(set(glyphs)-{'0'})
assert '0' not in class_coverage_if_source_0_heldout

method_locators = []
for path,names in [('scripts/course_experiments/modalities.py',{'_manifest','run_ocr'}),('scripts/prepare_ocr.py',{'draw_digits'})]:
    source_ast = ast.parse((ROOT/path).read_bytes())
    for node in source_ast.body:
        if isinstance(node,ast.FunctionDef) and node.name in names:
            method_locators.append({'path':path,'function':node.name,'start':node.lineno,'end':853 if node.name=='run_ocr' else node.end_lineno,'sha256':sha((ROOT/path).read_bytes())})
result = {
    'reviewer_task': '/root/phase4_factual_coordinator/factual_11_12',
    'checked_at': datetime.datetime.now(datetime.UTC).isoformat(),
    'environment': {'python':platform.python_version(),'device':'CPU stdlib-only source/version/raw-record verification; no torch/model import'},
    'current_source_sha256':sha(section),'source_read_scope':'Personally reread complete current11.12 including changed family/source and experiment scope paragraphs, fence, figure reference, exercise and details.',
    'prior_pass_sha256':'2d01eecb347398175f51397a1b5bf643b144c632a48557970c0d4f7bea64c5d3',
    'prior_revise_sha256':sha(old_report_raw),
    'prior_revise_history':'docs/technical-reviews/history/phase4-11_12-own-family-unit-revise-ad8a5d658e3ec48cf9bf467180012e4e7ee4fa3104e88f6a48563774a6bfb9d8.json',
    'unchanged_original_inputs':original_hashes,'verified_old_artifact_hashes':verified_artifacts,
    'fence_sha256':sha(fence),'figure_sha256':sha(figure.read_bytes()),
    'named_raw_pointers_reread':pointers,'raw_result_sha256':sha(raw_result),'coverage':coverage,
    'two_fixture_holdout':{'source_glyph_count_per_label':{k:1 for k in glyphs},'if_original_0_and_all_its_variants_held_out_train_character_classes':class_coverage_if_source_0_heldout,
                           'scope':'Exact consequence of the two teaching fixtures, not a trained-model prediction and not the0-99 actual split.'},
    'original_method_reread':method_locators,
    'official_grouping_source':{'path':'docs/technical-reviews/artifacts/phase4-11_12-independent/sources/sklearn-group-split-1.7.2.rst',
                                'sha256':sha((ART.parent/'sources/sklearn-group-split-1.7.2.rst').read_bytes()),'reread_locator':'Grouped-data/GroupKFold lines628-660, domain-specific dependent groups and no group overlap'},
    'honest_reuse':{
        'unchanged_original_fence':'Original fence execution onCPU reused only for identical fence bytes, data matrices, dtype and no optimizer behavior. No fence re-execution.',
        'unchanged_visual_and_rgb_checks':'All30 original SVG grid cells, one-pixel exercise andRGB/shape/gradient demonstration reuse unchanged fixture, SVG and implementation hashes. No forward/backward/model run repeated.',
        'original_historical_result':'Original2/30 raw-token scoring and loss/history checks reused because full original JSON, tokenizer and scoring code hashes remain exact. Character/string family coverage was additionally reread as above.',
        'old_html_parity':'Old desktop/mobile HTML snapshots matched the old section only; they are retained as history and are not current-text parity proof. Current raw source gets a separate bounded artifact render, explicitly not the production page.'},
    'scope':'No change to textbook,SVG,implementation or shared server/builder; no full export, model/data download, training, model re-evaluation,GPU work or.pt save.'
}
(ART/'current-verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='verified_old_artifact_hashes'},ensure_ascii=False,indent=2))
