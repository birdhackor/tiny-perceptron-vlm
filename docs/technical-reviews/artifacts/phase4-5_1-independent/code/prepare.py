"""Fresh 5.1 provenance capture; copies raw inputs, never trains a model."""
from pathlib import Path
import hashlib
import importlib.util
import json
import re
import subprocess

ROOT = Path('/workspace/tiny-perceptron-vlm')
OUT = ROOT / 'docs/technical-reviews/artifacts/phase4-5_1-independent'
def sha(raw): return hashlib.sha256(raw).hexdigest()
def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
spec = importlib.util.spec_from_file_location('section_facts', ROOT/'docs/review-tools/section_facts.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
section, whole, first_line = helper.original_section(ROOT/'course/chapters/05.md', '5.1')
first_header = re.search(rb'(?m)^## ', whole).start()
intro = whole[:first_header]
(OUT/'original/section.md').write_bytes(section)
(OUT/'original/chapter-intro.md').write_bytes(intro)
fence_metadata = []
for number, item in enumerate(helper.fences(section, first_line), 1):
    raw = item.pop('raw')
    item.pop('marker')
    name = f'original/fence-{number}.py'
    (OUT/name).write_bytes(raw)
    fence_metadata.append(dict(item, number=number, snapshot=name, sha256=sha(raw)))
current_paths = [
    'tiny_perceptron/model.py', 'tiny_perceptron/data.py', 'tiny_perceptron/attention.py',
    'tiny_perceptron/modern.py', 'scripts/build_course.py',
    'docs/review-tools/factual-reviewer-instructions.md', 'docs/review-tools/section_facts.py',
    'scripts/check_technical_reviews.py', '.agents/skills/clear-tutorial/references/review-protocol.md',
    'docs/course-experiments/results/text_foundation.json',
]
inputs = {}
for name in current_paths:
    raw = (ROOT/name).read_bytes()
    target = OUT/'original/current'/name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
    inputs[name] = dict(sha256=sha(raw), snapshot=target.relative_to(OUT).as_posix())
result = json.loads((ROOT/'docs/course-experiments/results/text_foundation.json').read_bytes())
revision = result['revision']
experiment_paths = [
    'scripts/course_experiments/common.py', 'scripts/course_experiments/text.py',
    'scripts/course_experiments/run.py', 'scripts/prepare_data.py',
    'tiny_perceptron/model.py', 'tiny_perceptron/data.py', 'tiny_perceptron/training.py',
]
original_experiment = {}
for name in experiment_paths:
    raw = subprocess.check_output(['git','show',f'{revision}:{name}'], cwd=ROOT)
    target = OUT/'original/experiment-version'/name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
    expected = result['code_sha256'].get(name)
    if expected is not None: assert sha(raw) == expected, (name, sha(raw), expected)
    original_experiment[name] = dict(sha256=sha(raw), recorded_code_sha256=expected,
                                     snapshot=target.relative_to(OUT).as_posix())
manifest = dict(
    reviewer_task='/root/phase4_factual_coordinator/factual_5_1',
    source='course/chapters/05.md#5.1', source_sha256=sha(section),
    source_file_sha256=sha(whole), section_first_line=first_line,
    source_bytes=len(section), crlf_count=section.count(b'\r\n'),
    intro_sha256=sha(intro), intro_bytes=len(intro),
    intro_summary='本章先讓固定接字题改善，再形成批次更新與優化器流程，最後用留出資料、有效答案數、參數數及秒數解讀配方；這些單位不能混用。',
    newline_policy='original UTF-8 bytes, without stripping or newline normalization',
    fences=fence_metadata, figures=[], figure_note='No image/SVG or HTML image references in chapter intro or 5.1; no figure to render.',
    current_inputs=inputs, empirical_revision=revision, empirical_original_inputs=original_experiment,
    scope='Only 5.1 and chapter intro reviewed; other lesson bodies and prior review judgments not used.',
)
dump(OUT/'input-manifest.json', manifest)
print(json.dumps(manifest, ensure_ascii=False, indent=2))
