"""Construct this review from this reviewer's own evidence; never read the old canonical."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
PREFIX = OUT.relative_to(ROOT).as_posix()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
inspection = json.loads((OUT / 'inspection-receipt.json').read_text())
observed = json.loads((OUT / 'verification.json').read_text())
originals = json.loads((OUT / 'original-sources-receipt.json').read_text())['sources']
env = {k: str(v) for k, v in observed['environment'].items()}
artifacts = []

def artifact(identifier, filename, kind, description, command=None, result=None, environment=None):
    item = {'id': identifier, 'path': PREFIX + '/' + filename, 'sha256': sha(OUT / filename),
            'kind': kind, 'description': description}
    if command:
        item.update(command=command, result=result, environment=environment or env)
    artifacts.append(item)

artifact('inspection-receipt', 'inspection-receipt.json', 'source_snapshot',
         'Own fresh identity, actual original section/prerequisite/code read ranges, whole-input snapshot meaning, safe raw pointers, code-provenance SHA correspondence and visual inspection.')
artifact('frozen-section', 'frozen-section.md', 'source_snapshot', 'Raw UTF-8 11.9 section actually reviewed; no newline normalization.')
artifact('frozen-chapter-input', 'frozen-chapter-11.md', 'source_snapshot',
         'Whole chapter raw bytes frozen on 2026-10-05 for this read; only 11.9 is substantively reviewed. This is a dated input snapshot, not a claim about a later entire chapter.')
artifact('prerequisite', 'prerequisite-10_1.md', 'source_snapshot', 'Necessary 10.1 original pixel/channel/axis convention personally read.')
artifact('raw-pointers', 'raw-selected-pointers.json', 'source_snapshot', 'Only named raw provenance, configuration, split records, sample IDs/answers and measured aggregates; original full SHA retained, author commentary excluded.')
artifact('original-sources', 'original-sources-receipt.json', 'source_snapshot', 'Own HTTPS URLs, release/doc versions, original full SHA, authority reasons, actual locators and support boundaries.')
artifact('commands', 'commands.json', 'source_snapshot', 'Own exact bounded CPU/browser argv, selected environment overrides, exit codes, stdout/stderr and preserved initial invocation failure.')
artifact('cpu-code', 'verify.py', 'code', 'Actual independent CPU code for input rules, raw sample scoring, split hashes, noninjective crop/resize examples and exact SVG pixel geometry.')
artifact('cpu-code-v1', 'verify.v1.py', 'code', 'Preserved exact earlier verification-code version corresponding to initial failure and first successful command; subsequent source hash check added in final code.')
artifact('cpu-verification', 'verification.json', 'execution', 'Final CPU observations and successful assertions; no trained model execution.',
         'PYTHONPATH=/workspace/tiny-perceptron-vlm CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /workspace/tiny-perceptron-vlm/.venv/bin/python ' + str(OUT / 'verify.py'),
         'Exit 0: offset7 45/8, offset0 81/64, offset8 36/0; 12/12 materials retained color; original split counts/hashes reproduced; both stored interventions 9/12, color6/6, shape3/6; exact SVG grids/box; loss-of-information constructions succeeded.')
artifact('cpu-stdout', 'cpu-stdout.txt', 'execution', 'Actual stdout from final CPU invocation.',
         '/workspace/tiny-perceptron-vlm/.venv/bin/python ' + str(OUT / 'verify.py'), 'Exit 0; full JSON observations printed.')
artifact('cpu-stderr', 'cpu-stderr.txt', 'source_snapshot', 'Final CPU stderr is empty; initial invocation failure remains separately preserved.')
artifact('original-fence', 'fence/fence-1.py', 'code', 'Exact original Python fence L325-331 extracted as original bytes (source lines are also in extraction.json).')
artifact('fence-extraction', 'fence/extraction.json', 'source_snapshot', 'Original byte extraction metadata, section/fence/figure/helper hashes and exact source line locator.')
artifact('fence-execution', 'fence/execution.json', 'execution', 'Exact helper subprocess receipt for executing the unchanged original fence.',
         '.venv/bin/python docs/review-tools/section_facts.py course/chapters/11.md#11.9 --output outputs/reviewer-tools/runs/phase4-11_9-clean-executed --execute --timeout 30',
         'execution_exit_code=0; original one Python fence executed on CPU/offline guard.')
artifact('fence-stdout', 'fence/stdout.txt', 'execution', 'Actual unchanged original fence stdout.',
         '.venv/bin/python docs/review-tools/section_facts.py course/chapters/11.md#11.9 --output outputs/reviewer-tools/runs/phase4-11_9-clean-executed --execute --timeout 30',
         '原圖形狀 (3, 16, 16) 裁切形狀 (3, 8, 8); 原圖紅色格 45; 裁切紅色格 8.')
artifact('fence-environment', 'fence/environment.json', 'source_snapshot', 'Actual Python/Torch CPU environment, attempted fences and imported repository-module SHA.')
artifact('fence-bootstrap', 'fence/bootstrap.py', 'code', 'Exact runtime bootstrap extracted from original build_course.py; not an edited substitute fence.')
artifact('figure-render', 'figure.png', 'figure_render', 'Original SVG actually rendered with Inkscape and personally viewed; fixed yellow crop box, 45 red input cells and 8 red crop cells.')
artifact('figure-render-command', 'inkscape-command.json', 'execution', 'Exact original SVG render argv, output/source SHA, exit status and actual Inkscape version.',
         '/usr/bin/inkscape course/figures/rewrite-11-crop-evidence.svg --export-filename=' + str(OUT / 'figure.png'),
         'Exit 0; PNG rendered. Pango/Gtk wrapper warnings retained in inkscape-stderr.txt; rendered pixels and text personally viewed.',
         {'inkscape': 'Inkscape 1.4 (e7c3feb100, 2024-10-09)', 'device': 'CPU SVG renderer'})
artifact('desktop-render', 'desktop.png', 'figure_render', 'Actual Chromium 1280x800 section screenshot personally viewed; figure labels and arrows visible.')
artifact('mobile-render', 'mobile.png', 'figure_render', 'Actual Chromium 390x844 section screenshot personally viewed; complete crop comparison and labels visible.')
artifact('page-code', 'render-page.py', 'code', 'Own page/body/fence/original SVG equivalence assertions and actual browser capture code; no full export.')
artifact('page-verification', 'page-verification.json', 'execution', 'Actual page equality receipt and browser sizes.',
         '/workspace/tiny-perceptron-vlm/.venv/bin/python ' + str(OUT / 'render-page.py'),
         'Exit 0; all five substantive paragraphs, original fence, section title and original SVG bytes match; desktop/mobile Chromium screenshots captured.',
         {'python': env['python'], 'browser': '/usr/bin/chromium --headless --no-sandbox', 'viewports': '1280x800 and 390x844', 'device': 'CPU browser renderer'})
artifact('page-original', 'page.html', 'source_snapshot', 'Actual local 11.9 HTML personally compared with the current raw source; not rebuilt.')
artifact('report-builder', 'write-report.py', 'code', 'Own complete report construction code; it never opens the prior canonical report.')
for i, item in enumerate(originals):
    artifact('authority-snapshot-' + str(i), item['snapshot'], 'source_snapshot',
             'Fetched original bytes for ' + item['title'] + '; personally inspected only the recorded original locators.')
for item in inspection['actual_read_scope']:
    if isinstance(item, dict):
        artifact('inspected-' + item['path'].replace('/', '-').replace('.', '-'), item['excerpt'], 'source_snapshot',
                 'True personally read original implementation lines with source full SHA: ' + item['path'])
for name in ['python-slicings.inspected.txt', 'python-sequence-slicing.inspected.txt',
             'pillow-12.3.0-Image.py.inspected.txt', 'pillow-12.3.0-concepts.rst.inspected.txt']:
    artifact('excerpt-' + name.replace('.', '-'), name, 'source_snapshot', 'Exact original authority paragraphs/line ranges personally read.')

sources = []
for identifier, item in zip(['python-slicings', 'python-sequences', 'pillow-image', 'pillow-filters'], originals):
    sources.append({'id': identifier, 'kind': 'official_docs' if identifier.startswith('python') or identifier == 'pillow-filters' else 'official_source',
                    'title': item['title'], 'verified': True, 'url': item['url'], 'version': item['version'],
                    'authority_reason': item['authority_reason'], 'checked_original': True, 'accessed_on': item['accessed_on'],
                    'inspection_note': item['locator'] + '. Personally inspected saved original bytes, not a search summary; ' + item['inspection_scope']})
for identifier, filename, title, lines, version in [
    ('scene', 'tiny_perceptron/multimodal.py', 'Original scene generator', 'L177-190, RGB tensor creation, ij axis convention, center/region rules', 'Exact current file SHA also matches stored experiment code_sha256'),
    ('experiment', 'scripts/course_experiments/modalities.py', 'Original finite VQA/crop data and exact-match scorer', 'L37-53 hashing/manifest; L171-192 original split and answer rules; L219-243 materials; L246-257 target tokens; L287-364 scorer; L377-384 trainable settings; L772-802 paired experiment and shifted crop construction. Return-result commentary at L822-830 was not read.', 'File SHA equals stored original experiment provenance; revision 22a0bb1b870e4df4af76630243ef55b7ca27840c'),
    ('tokenizer', 'tiny_perceptron/data.py', 'Original ByteTokenizer answer-ID contract', 'L14-25, special IDs and UTF-8 byte encode/decode; runtime used only this contract', 'Exact current file SHA matches raw experiment provenance'),
    ('result', 'docs/course-experiments/results/vision_ablation.json', 'Original stored NVIDIA L4 vision-ablation measurements', 'Only named safe pointers in raw-selected-pointers.json, plus four named /code_sha256 pointers in inspection-receipt.json. All original bytes retained unchanged; /results/crop_note and author interpretation values unread.', 'revision 22a0bb1b870e4df4af76630243ef55b7ca27840c; PyTorch2.14.1+cu126/Python3.13.3/NVIDIA L4; current CPU verification is not a model rerun')]:
    sources.append({'id': identifier, 'kind': 'repository_code', 'title': title, 'verified': True, 'path': filename,
                    'sha256': sha(ROOT / filename), 'version': version, 'inspection_note': lines})
sources.extend([
    {'id': 'cpu-run', 'kind': 'execution', 'title': 'Own bounded CPU inputs/raw measurements/figure checks', 'verified': True, 'artifact_id': 'cpu-verification'},
    {'id': 'fence-run', 'kind': 'execution', 'title': 'Own execution of the unchanged original Python fence', 'verified': True, 'artifact_id': 'fence-execution'},
    {'id': 'page-run', 'kind': 'execution', 'title': 'Own actual rendered-page and original-source equality check', 'verified': True, 'artifact_id': 'page-verification'},
    {'id': 'geometry', 'kind': 'derivation', 'title': 'Explicit crop pixel geometry and ambiguity constructions', 'verified': True,
     'details': 'scene size16 radius4: offset7 center x15; original x11..15 and y4..12 gives5*9=45, crop x4..11/y4..11 leavesx11,y4..11 =>1*8=8. Offset0 original x4..12/y4..12 gives9*9=81, crop retains8*8=64. Offset8 x12..15 leaves no crop pixel. Red/blue offset8 distinct originals give identical zero crops, hence deterministic pad/resize cannot recover their different color targets. Distinct2x2 patterns with2white/2black pixels each BOX-resize to the same1x1 value128.'}
])

def evidence(source_id, locator, supports):
    return {'source_id': source_id, 'locator': locator, 'supports': supports}

def verification(expected, observed_text, details, denominators=None, tolerance=None):
    item = {'method': 'executed', 'expected': expected, 'observed': observed_text, 'details': details}
    if denominators: item['denominators'] = denominators
    if tolerance: item['tolerance'] = tolerance
    return item

claims = [
 {'id': 'pixel-geometry', 'kind': 'numeric', 'statement': 'size16 scene("red","square",offset7) has CHW shape(3,16,16), center x15 and45 positive red cells; [4:12,4:12] keeps indices4..11, shape(3,8,8) and8 red cells. Offset0 exercise yields81 and64.',
  'location': '11.9 opening, original Python fence, numeric paragraph and offset0 exercise', 'status': 'verified',
  'scope': 'Counts of red-channel cells greater than zero in this deterministic synthetic input; not sums of brightness, model scores or percentage measures.',
  'evidence': [evidence('scene','L177-190','Exact center, radius, CHW tensor and mask rules.'),evidence('python-sequences','Common sequence operations note(4)','Half-open sequence slice convention; tensor behavior independently executed.'),evidence('geometry','45=5*9;8=1*8;81=9*9;64=8*8','Explicit axes, boundaries and count derivation.'),evidence('cpu-run','verification.json /counts','Own offset7/0/8 measurements.'),evidence('fence-run','fence/stdout.txt','Unchanged original fence gives stated outputs.')],
  'artifact_ids': ['cpu-verification','fence-execution','fence-stdout'],
  'verification': verification('Shapes(3,16,16)/(3,8,8), counts45/8; offset0 counts81/64','Exact match for all shapes/indices/counts','Independent original-fence run plus scene source-rule CPU variants; coordinates and denominators explicitly derived.',tolerance='Exact integer/shape/index equality; no rounded tolerance.')},
 {'id': 'original-fence-contract', 'kind': 'software', 'statement': 'The original import, scene call, CHW slicing, tuple(shape), boolean >0 sum and int conversion form a material-inspection demonstration; they perform no model update. Changing offset changes only generated input material.',
  'location': '11.9 entire original Python fence and final exercise sentence', 'status': 'verified',
  'scope': 'Consolidated coverage of every original Python operation and the offset exercise; no optimizer, gradient, trained model or evaluation was added.',
  'evidence': [evidence('scene','L177-190','Actual helper constructs input values only.'),evidence('python-slicings','6.3.3 Slicings','Comma-separated indexing constructs a tuple of indices/slices.'),evidence('fence-run','fence/execution.json, fence/environment.json, fence/stdout.txt','Original fence really executes with CPU Torch2.14.1; module SHA tracked.'),evidence('cpu-run','verify.py counts loop and verification.json','Offset variations run without a model.')],
  'artifact_ids': ['original-fence','fence-execution','fence-environment','cpu-verification'],
  'verification': verification('The complete original fence executes and inspects tensors; offset0 yields stated material changes','One original Python fence executed with exit0; all source-rule variations passed','No training recipe or existing checkpoint executed. Actual CPU Python3.13.5/Torch2.14.1+cpu environment recorded separately from historical GPU measurements.')},
 {'id': 'preprocessing-evidence', 'kind': 'concept', 'statement': 'The explicit crop box selects fixed coordinates. Padding or upscaling can restore input dimensions but cannot guarantee recovery of discarded pixels; downsampling can combine neighboring foreground/background pixels and lose detail.',
  'location': '11.9 opening and paragraph beginning 圖中的框是實際裁切範圍', 'status': 'verified',
  'scope': 'Deterministic material-level information loss and fixed crop coordinates. The small synthetic checks demonstrate a mechanism, not measured OCR degradation or a universal impossibility of guessing a shape from priors.',
  'evidence': [evidence('pillow-image','Image.crop/_crop L1369-1415; resize L2329-2375,L2394-2439','Crop accepts an explicit rectangle; resize accepts target dimensions and resampling filter.'),evidence('pillow-filters','Filters L160-221, especially BOX L174-182 and BICUBIC L202-208','Resampling maps input pixels to output pixels; BOX uses identical weights and BICUBIC interpolates contributors.'),evidence('geometry','Distinct red/blue originals with identical zero crop; two distinct2x2 patterns -> same128','Constructive noninjectivity proves dimension restoration cannot recover all discarded content.'),evidence('cpu-run','verification.json /information_loss','Those constructions actually ran using installed Pillow12.3.0 and CPU Torch.')],
  'artifact_ids': ['cpu-verification','cpu-code','original-sources']},
 {'id': 'stored-performance', 'kind': 'empirical', 'statement': 'Stored none and edge_cropped measurements each score9/12 exact matches. Each has6/6 color answers and3/6 shape answers, with every shape answer generated as circle; the equal aggregate cannot quantify shape degradation caused by cropping.',
  'location': '11.9 paragraph beginning 既有裁切介入後', 'status': 'verified',
  'scope': 'Recalculation of the original finite stored12-question result, not new inference/training or a general visual-recognition conclusion. Baseline offset2 and crop material offset7 differ; no estimate of isolated crop causality is asserted.',
  'evidence': [evidence('result','/results/interventions/{none,edge_cropped}/{examples,correct,exact_match,groups,samples/*}; /results/data/splits/{train,validation,test}','Original per-question targets, generated IDs, exact-match flags, grouping and held-out offset split.'),evidence('experiment','L171-192;L246-257;L287-364;L772-802','Targets come from generator attributes; exact match compares generated bytes before firstEOS to target bytes; crop treatment shifts tooffset7 and retains row targets.'),evidence('tokenizer','L14-25','Actual UTF-8 byte token IDs and EOS ID used to reconstruct score.'),evidence('cpu-run','verification.json /aggregates; raw-selected-pointers.json','All stored sample flags/text/targets/EOS reconstructed; group counts and72 target tokens independently summed.')],
  'artifact_ids': ['raw-pointers','cpu-verification','cpu-code','inspection-receipt'],
  'verification': verification('Each treatment12 samples,9 exact matches; color6/6; shape3/6, allcircle;72 answer-plus-EOS tokens','Exact match of all stored sample targets/generated IDs, grouping and aggregate values; original split counts/hashes matched','Exact-match denominator is questions, not token count or EOS success. Six scene families each have one shape and one color question; original test offset2, crop offset7/crop8x8/resize16x16.',denominators={'questions_per_treatment':12,'color_questions':6,'shape_questions':6,'test_scene_families':6,'answer_tokens_including_eos':72,'train_records':36,'validation_records':12,'test_records':12})},
 {'id': 'visibility-and-targets', 'kind': 'concept', 'statement': 'Debugging should retain original input, crop settings and processed input; when a requested target disappears, allowing a not-visible answer prevents an impossible demand to recover the original truth from an identical missing-target input.',
  'location': '11.9 paragraph beginning 排錯時保存原圖', 'status': 'verified',
  'scope': 'Conditional task-design recommendation. It does not claim that the historical exact-match benchmark already implements an invisible label. All current offset7 test crops retain1(circle) or8(square) positive color cells; the missing-target counterexample is offset8.',
  'evidence': [evidence('pillow-image','Image.crop/_crop L1369-1415','Only the selected rectangle is present in the processed image.'),evidence('geometry','Red and blue square offset8 both produce the same all-black8x8 crop','No input-only answer rule can give distinct original color truths for identical missing-target pixels.'),evidence('experiment','L171-192 answers;L287-364 exact-match;L793-802 crop input and retained targets','Separates existing original-attribute target rule from the recommended visibility-aware future rule.'),evidence('cpu-run','verification.json /materials and /information_loss','Every actual offset7 crop preserves active color, while offset8 is the bounded missing-target change.')],
  'artifact_ids': ['cpu-verification','cpu-code']},
 {'id': 'figure-consistency', 'kind': 'numeric', 'statement': 'The figure faithfully depicts all256 original16x16 cells and64 cropped8x8 cells, with45 and8 red cells, yellow [4:12,4:12] box and original-to-crop arrow; rendered desktop and mobile agree with current source.',
  'location': '11.9 referenced rewrite-11-crop-evidence.svg and its alt/caption', 'status': 'verified',
  'scope': 'Current exact SVG geometry, source/HTML equivalence and actually viewed renders at1280x800 and390x844; not a reconstruction from SVG strings alone.',
  'evidence': [evidence('cpu-run','verify.py SVG geometry assertions;verification.json /figure','Every original/crop cell fill equals actual scene/slice values; yellow box at(109,185),112x112 spans indices4..11.'),evidence('page-run','page-verification.json','Actual served SVG bytes, section title, all5 substantive paragraphs and original fence equal frozen source.')],
  'artifact_ids': ['cpu-verification','figure-render','desktop-render','mobile-render','page-verification'],
  'verification': verification('Original256/crop64 cells; masks and fixed8x8 box match; 45/8 labels and rendered current source agree','All pixel-mask and box-coordinate assertions passed; Inkscape and both Chromium screenshots personally viewed','Figure pixels tested against fresh CPU scene values, then visually checked labels/arrows/materials; current served SVG checked byte-for-byte.',tolerance='Exact integer coordinates/cell colors and source bytes; rendering readability checked visually.')}
]

report = {'schema_version': 1, 'review_stage': 'technical', 'lesson_id': '11.9',
          'source': 'course/chapters/11.md#11.9', 'source_sha256': sha(OUT / 'frozen-section.md'),
          'reviewer_task': '/root/phase4_factual_coordinator/factual_11_9_clean', 'reviewer_context': 'fresh',
          'reviewed_on': '2026-10-05', 'verdict': 'pass', 'author_tasks': [],
          'figure_sha256': {'course/figures/rewrite-11-crop-evidence.svg': sha(ROOT / 'course/figures/rewrite-11-crop-evidence.svg')},
          'frozen_input_snapshot': {**inspection['frozen_chapter_input'], 'path': PREFIX + '/frozen-chapter-11.md'},
          'actual_read_scope': inspection['actual_read_scope'], 'artifacts': artifacts, 'sources': sources, 'claims': claims,
          'issues': [],
          'checks': {
              'factual_accuracy': {'status': 'pass', 'details': 'Original source rules, finite historical answer/scoring scope and deterministic preprocessing information loss personally checked; no prior review conclusions used.', 'claim_ids': [c['id'] for c in claims]},
              'numeric_verification': {'status': 'pass', 'details': 'Original45/8, exercise81/64, axes/half-open slice,9/12 groups and72 target tokens, original split hashes and SVG pixel geometry all independently executed/recalculated.', 'claim_ids': ['pixel-geometry','stored-performance','figure-consistency']},
              'figure_consistency': {'status': 'pass', 'details': 'Original Inkscape render and real desktop/mobile Chromium screenshots personally viewed;256/64 actual pixel masks, crop box, arrow and45/8 labels agree; served source/fence/SVG equality verified without exporting full course.', 'claim_ids': ['figure-consistency']},
              'source_verification': {'status': 'pass', 'details': 'Personally fetched and read official Python3.13 slice contracts and original Pillow12.3.0 crop/resize/filter source; saved URLs/version/authority/full SHA and true locators. Original finite experimental JSON inspected only via named raw pointers, with matched provenance code SHA.', 'claim_ids': ['pixel-geometry','original-fence-contract','preprocessing-evidence','stored-performance','visibility-and-targets']},
              'limitations': {'status': 'pass', 'details': 'History was recalculated, not model inference rerun. Original GPU and current CPU versions are distinguished. Crop experiment changes offset2 to7 and input size then resizes; equal aggregate and allcircle shape outputs cannot establish preserved shape information or isolated crop causality. Current12 crops retain color. Missing-target/not-visible is conditional advice, not a claimed implemented label. No OCR score, GPU work, model/data download, full training or .pt save.', 'claim_ids': ['preprocessing-evidence','stored-performance','visibility-and-targets']}}}

# Replace canonical with this own complete report without ever opening the previous file.
(ROOT / 'docs/technical-reviews/11.9.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'report': 'docs/technical-reviews/11.9.json', 'source_sha256': report['source_sha256'],
                  'verdict': report['verdict'], 'claims': len(claims), 'artifacts': len(artifacts)}, ensure_ascii=False))
