from pathlib import Path
import datetime
import hashlib
import json
import re

ART = Path(__file__).resolve().parent
ROOT = ART.parents[3]
sha = lambda raw: hashlib.sha256(raw).hexdigest()
read = lambda p: json.loads((ART / p).read_bytes())
ident = lambda p: re.sub(r'[^a-zA-Z0-9_]', '_', p)
verification = read('verification.json')
inspection = read('source-inspection.json')
snapshot = read('input-snapshot-receipt.json')
render = read('render-receipt.json')
assert read('verify-execution.json')['exit_code'] == 0
assert read('render-execution.json')['exit_code'] == 0
assert read('original/execution.json')['exit_code'] == 0
body = (ART / 'original/section.md').read_bytes()
raw = (ROOT / 'course/chapters/11.md').read_bytes()
headings = list(re.finditer(rb'(?m)^## [^\r\n]+', raw))
i = next(i for i, h in enumerate(headings) if h[0].startswith(b'## 11.12 '))
end = headings[i+1].start() if i+1 < len(headings) else len(raw)
assert raw[headings[i].start():end] == body
for f in ['scripts/course_experiments/modalities.py', 'scripts/prepare_ocr.py', 'tiny_perceptron/multimodal.py', 'tiny_perceptron/data.py']:
    p = ART / 'original/repository-code' / f
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes((ROOT / f).read_bytes())

executions = {
    'original/execution.json': {
        'command': '.venv/bin/python docs/review-tools/section_facts.py course/chapters/11.md#11.12 --output outputs/phase4-11_12-independent-original --execute --timeout 60',
        'result': 'Exit 0; attempted original Python fence 1; two exact float32 matrices of shape (5,3), labels 0/1; no guard events. Actual worker command and hashes are in this receipt; stdout/stderr/environment/fence/bootstrap copied unchanged into permanent original/ artifacts.',
        'environment': {'python':'3.13.5','torch':'2.14.1+cpu','device':'cpu','cuda_build':'None','cuda_available':'False'}},
    'verify-execution.json': {
        'command': '.venv/bin/python docs/technical-reviews/artifacts/phase4-11_12-independent/verify.py',
        'result': 'Exit 0 in 1.7758 seconds. All thirty SVG pixels match; exercise adds only upper-left pixel; RGB conversion gives (3,16,16), original reshape/encoder input rejected; provided-label backward gives nonzero gradient without update; splits/counts/hashes and all saved raw-token scores recalculate to 2/30; 500 historical history rows sum to 11584 effective targets.',
        'environment': verification['environment']},
    'render-execution.json': {
        'command': '.venv/bin/python docs/technical-reviews/artifacts/phase4-11_12-independent/render.py',
        'result': 'Exit 0; real Chromium SVG plus desktop/mobile full-page screenshots viewed. At 1280x800 and 390x844 all six source body paragraphs and the exact original fence match, evidence link is present, and served SVG bytes equal source SVG. Runtime HTML snapshots and PNG screenshots retained.',
        'environment': {'python':render['python'],'chromium':render['chromium_version'],'device':'CPU; --disable-gpu','renderer':'/usr/bin/chromium via Playwright; --no-sandbox'}}
}
artifacts = []
for p in sorted(ART.rglob('*')):
    if not p.is_file() or p.name in {'evidence-manifest.json','checker.stdout','checker.stderr','checker-execution.json'}:
        continue
    relative = p.relative_to(ART).as_posix()
    kind = 'code' if p.suffix == '.py' else ('figure_render' if p.suffix == '.png' else 'source_snapshot')
    description = {
        'source-inspection.json':'Personal primary-source version/locator/support boundaries and truthful implementation/JSON/visual read ranges.',
        'input-snapshot-receipt.json':'Dated frozen full Markdown input with explicit snapshot-only wholefile scope, section raw-byte hash and prerequisite snapshots.',
        'verification.json':'Executed independent raw-result pointer receipt, pixel/axis/count checks, bounded gradient and per-sample raw-token decisions.',
        'verify.py':'Actual bounded CPU verification code, with no training, checkpoint evaluation, downloads or neural weight saves.',
        'render.py':'Actual target-only Chromium render, original-paragraph/fence fidelity and exact served-SVG verification code.',
        'sources/retrieval.json':'Actual HTTPS 200 retrieval receipts, exact URLs, access date and downloaded original-source hashes.',
        'original/raw-ocr.json':'Complete unchanged original result bytes retained opaquely; only listed raw pointers were inspected.',
        'original/frozen-chapter-11.md':'Frozen whole chapter input saved during this review; does not assert whole chapter currentness or full sequential review.',
        'original/section.md':'Raw UTF-8 original 11.12 subsection, including heading and original newline bytes.',
        'original/fence-1.py':'Unmodified original 11.12 Python fence executed by section_facts helper.',
        'glyphs-render.png':'Viewed real 640x530 rendering of original glyph SVG.',
        'desktop-page.png':'Viewed real desktop page screenshot, 1280x800 viewport, full-page capture.',
        'mobile-page.png':'Viewed real mobile page screenshot, 390x844 viewport, full-page capture.',
    }.get(relative, 'Preserved actual evidence file: ' + relative)
    item = {'id':ident(relative),'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p.read_bytes()),'kind':kind,'description':description}
    if relative in executions:
        item.update(kind='execution', **executions[relative])
    artifacts.append(item)
manifest = {'reviewer_task': inspection['reviewer_task'], 'written_at': datetime.datetime.now(datetime.UTC).isoformat(),
            'scope':'Hashes of permanent evidence present before report writing. Later checker receipt is separately dated and includes final report hash; manifest does not hash itself.',
            'files':[{'path':x['path'],'sha256':x['sha256'],'kind':x['kind']} for x in artifacts]}
(ART / 'evidence-manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
artifacts.append({'id':'evidence_manifest','path':(ART/'evidence-manifest.json').relative_to(ROOT).as_posix(),'sha256':sha((ART/'evidence-manifest.json').read_bytes()),
                  'kind':'source_snapshot','description':'Full permanent evidence hash manifest, excluding itself and later checker receipt to avoid circular claims.'})
sources=[]
for item in inspection['sources']:
    sources.append({'id':item['id'],'kind':'paper' if item['id']=='trocr' else 'official_source',
                    'title':{'trocr':'TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models',
                             'torch_tensor':'PyTorch official torch.tensor source documentation','torch_cross_entropy':'PyTorch official cross_entropy target contract',
                             'torch_autograd':'PyTorch official Autograd mechanics','python_basics':'CPython official standard types: dictionary items and integer conversion',
                             'group_split':'scikit-learn official grouped cross-validation documentation'}[item['id']],
                    'verified':True,'url':item['url'],'version':item['version'],'accessed_on':inspection['accessed_on'],
                    'authority_reason':item['authority_reason'],'checked_original':True,'inspection_note':item['inspected']+' Support boundary: '+item['supports'],
                    'snapshot_artifact_id':ident(item['snapshot'])})
for identifier, path, title, note in [
    ('ocr_code','scripts/course_experiments/modalities.py','Original OCR experiment generation, training and scoring code',
     'Read _manifest 41-53, _fit 73-168, _media 219-243, _sequence 246-257, _loss_fn 260-283, _evaluate 287-364, _modal 367-374, _freeze 377-384 and run_ocr 833-881. Original training uses updates; no training was rerun. See source-inspection.json for all actually printed ranges.'),
    ('draw_digits','scripts/prepare_ocr.py','Original fixed 5x7 glyph lookup and RGB drawing code',
     'Read complete file. GLYPHS 11-22 and draw_digits 25-36 verified font, white strokes, black RGB 16x16 canvas, centering and offset. git show at historical report revision equals current complete bytes.'),
    ('multimodal','tiny_perceptron/multimodal.py','Original VisionEncoder shape and multimodal generation contracts',
     'Read lines 1-210 including VisionEncoder 30-41, image entry and iterative generate_modal 139-174. Actual bounded forward used only a fresh encoder to check dimensions; no historical model or weight files loaded.'),
    ('tokenizer','tiny_perceptron/data.py','Original ByteTokenizer byte/EOS contract',
     'Read lines 1-145, focused on SPECIALS line12 and ByteTokenizer 14-28. All thirty test generated_ids were sliced before EOS and compared to tokenizer.encode(target), including special-token rejection.'),
    ('ocr_result','docs/course-experiments/results/ocr.json','Original immutable historical OCR result measurements',
     'Inspected named raw training/config/provenance, split records/count/sha and all test samples; full original SHA retained. Never read /results/scope or author correction values. Exact final recalculation pointers in verification.json; initial broader data access honestly recorded in source-inspection.json.')]:
    sources.append({'id':identifier,'kind':'repository_code','title':title,'verified':True,'path':path,'sha256':sha((ROOT/path).read_bytes()),
                    'version':'Historical OCR run 22a0bb1b870e4df4af76630243ef55b7ca27840c; inspected raw input on 2026-10-05', 'inspection_note':note})
sources += [
    {'id':'original_run','kind':'execution','title':'Unmodified section fence on CPU','verified':True,'artifact_id':ident('original/execution.json')},
    {'id':'bounded_run','kind':'execution','title':'Independent bounded CPU and original raw-result recalculation','verified':True,'artifact_id':ident('verify-execution.json')},
    {'id':'render_run','kind':'execution','title':'Target page and SVG real Chromium rendering','verified':True,'artifact_id':ident('render-execution.json')}
]
ev = lambda s,l,t:{'source_id':s,'locator':l,'supports':t}
ver = lambda expected,observed,details,**extra:dict(method='executed',expected=expected,observed=observed,details=details,**extra)
claims = [
    {'id':'pixel_label_example','kind':'software','status':'verified','statement':'Dictionary keys are string labels 0/1; each character in five three-character rows is converted to numeric brightness, producing float32 (5,3) tensors without updating a recognizer.',
     'location':'course/chapters/11.md:419-435; original Python fence lines 424-430',
     'scope':'Grouped coverage of dict.items, nested comprehension, int, torch.tensor(dtype=float32), image.shape, printing and data-only behavior. The text string 0 is the intended answer, while numeric pixel zero is dark background. No learned OCR accuracy follows.',
     'evidence':[ev('python_basics','stdtypes.rst 4795-4799 and 5770-5783','Dictionary item pairs preserve string keys; base-10 conversion maps character 0/1 to integers.'),
                 ev('torch_tensor','torch/_torch_docs.py 9582-9643, torch.tensor','Nested input creates a data tensor, default requires_grad=False; dtype keyword specifies float32.'),
                 ev('original_run','original/execution.json; original/stdout.txt and environment.json','Actual unmodified fence prints both exact matrices as (5,3), with no model, backward, optimizer or parameter update.')],
     'artifact_ids':[ident('original/execution.json'),ident('original/fence-1.py'),ident('original/stdout.txt'),ident('original/environment.json'),ident('verify-execution.json'),ident('glyphs-render.png')],
     'verification':ver('Both string labels, two 5x3 float32 matrices with bits mapped exactly 0/1; no model update.','Unmodified fence exit 0; original rows match; both tensors do not require grad; no optimizer steps.','Verified all original matrix entries and software contracts, rather than one claim for each basic syntax token.')},
    {'id':'ocr_classification_generation','kind':'concept','status':'verified','statement':'OCR converts visible text to machine text; a finite 0/1 classification answer and an iterative generated full string have different correctness contracts, so generated 00 does not equal target 0.',
     'location':'course/chapters/11.md:435',
     'scope':'Method existence and task-specific scoring, not equivalence of classifier accuracy and generative OCR accuracy. The original fence prepares data only. The historical generative evaluator compares raw answer bytes before EOS to full target bytes.',
     'evidence':[ev('trocr','Introduction p.1; Model Architecture pp.2-3; Task Pipeline p.3','OCR definition and image-conditioned iterative wordpiece generation supported by original paper; no repository score inferred from it.'),
                 ev('torch_cross_entropy','torch/nn/functional.py 3478-3532, cross_entropy target/shape','Finite classes use class-index targets and C logits per example.'),
                 ev('ocr_code','_evaluate 287-340, especially raw slicing and exact_match','Full raw-token equality rejects an extra answer character; EOS is handled separately.'),
                 ev('bounded_run','verification.json#/classification_vs_generation and #/sample_receipt','Full equality 00 vs 0 is false; original thirty saved raw-byte scoring decisions independently checked.')],
     'artifact_ids':[ident('verify-execution.json'),ident('verification.json')]},
    {'id':'gradient_label_limit','kind':'concept','status':'verified','statement':'A gradient pass does not certify that a drawing and the manually assigned intended text label are semantically paired correctly.',
     'location':'course/chapters/11.md:437, first sentence',
     'scope':'Autograd computes derivatives of the supplied graph/loss and supplied target. The claim is about absence of a semantic-label validator in this supervised calculation; it does not assert that dataset errors can never be detected by separate statistical methods.',
     'evidence':[ev('torch_autograd','Autograd mechanics, How autograd encodes the history, lines 12-35','Differentiation traces executed operations by the chain rule, not author intent.'),
                 ev('torch_cross_entropy','cross_entropy lines 3493-3504 and 3520-3522','Ground-truth class target is provided by caller.'),
                 ev('bounded_run','verification.json#/bounded_gradient','Given target 1 with logits choosing 0 produces a nonzero gradient [0.952574,-0.952574] and no update; presence of a gradient cannot validate the human target.')],
     'artifact_ids':[ident('verify-execution.json'),ident('verification.json')]},
    {'id':'family_holdout','kind':'concept','status':'verified','statement':'Identity-preserving appearance variants retain the intended label and should stay together in a source family when testing unseen families.',
     'location':'course/chapters/11.md:437, family split sentences; 441 exercise',
     'scope':'Label invariance is explicitly conditional on the character not changing. In the 0-99 experiment the family is the entire digit string, with offsets -1/0/1 together; digits/font components are reused across groups. This tests unseen strings, not new fonts or handwriting.',
     'evidence':[ev('group_split','cross_validation.rst 628-677, grouped-data/GroupKFold','Dependent samples in the same group should not be shared between paired training and test/validation sets when assessing unseen groups.'),
                 ev('ocr_code','_manifest 41-53; run_ocr 833-851','All digit-string variants grouped by family, with group-overlap rejection.'),
                 ev('draw_digits','GLYPHS 11-22; draw_digits 25-36','Same font/glyph lookup and digit text, with a spatial offset only.'),
                 ev('bounded_run','verification.json#/historical_experiment/split_counts; #/exercise','80/10/10 distinct string families, 240/30/30 samples, no family overlap; upper-left one-pixel variant retains declared label 1.')],
     'artifact_ids':[ident('verify-execution.json'),ident('verification.json')]},
    {'id':'rgb_entry_contract','kind':'software','status':'verified','statement':'A two-dimensional 5x3 glyph cannot directly stand in for a three-channel 16x16 encoder input; use a fixed canvas and explicit channel replication.',
     'location':'course/chapters/11.md:437, RGB entry sentence',
     'scope':'Repository VisionEncoder configured for RGB 16x16; axes C,H,W and batch are preserved. Padding and copying produce a compatible input, without proving OCR recognition or recovering information.',
     'evidence':[ev('multimodal','VisionEncoder 30-41; patchify 12-17','Configured projection expects three channels and forward enforces (3,image_size,image_size).'),
                 ev('draw_digits','draw_digits 28-36','Historical original generator already uses a black RGB 16x16 canvas with white strokes.'),
                 ev('bounded_run','verification.json#/rgb_entry','15 input numbers cannot reshape to 768; direct entry rejected; canvas placement plus repeat gives (3,16,16) and one-batch features (1,16,8).')],
     'artifact_ids':[ident('verify-execution.json'),ident('verification.json')],
     'verification':ver('Original 5x3 input rejected; explicit canvas/channel copying yields a valid CPU encoder entry.','Both invalid reshape and encoder entry reject; replicated RGB has shape (3,16,16); fresh encoder forward returns (1,16,8).','One short no-grad forward on a fresh encoder verifies only shape/axis contract. Row and column coordinates are retained in every copied RGB channel.')},
    {'id':'glyph_numbers_and_exercise','kind':'numeric','status':'verified','statement':'The original glyphs have five rows and three columns; changing digit-1 first row 010 to 110 adds only the upper-left white pixel, with the declared label still 1.',
     'location':'course/chapters/11.md:419-433 and 441; course/figures/rewrite-11-glyph-labels.svg',
     'scope':'Exactly two teaching fixtures with 15 monochrome positions each, not trained-classifier predictions. The label is a manual identity-preserving design choice. White-count checks are inspection aids: 0 has 12, 1 has 8; the variant has 9.',
     'evidence':[ev('bounded_run','verification.json#/glyph_matrices, #/svg_grid_matches, #/exercise','All 30 SVG cells match original numeric pixels; exercise delta has only [row0,column0] = +1.'),
                 ev('render_run','render-receipt.json#/views; glyphs-render.png, desktop-page.png, mobile-page.png','Actual viewed glyphs, string labels and white/black legend match source and executed matrices.')],
     'artifact_ids':[ident('verify-execution.json'),ident('verification.json'),ident('render-execution.json'),ident('glyphs-render.png'),ident('desktop-page.png'),ident('mobile-page.png')],
     'verification':ver('5x3 = 15 entries per glyph; exactly one changed entry at row0,column0; white count 8 to 9 for variant.','All exact; SVG grid comparison 30/30. Two labels are string 0/1, independent of pixel values.','Rows are height, columns width, indexes start at zero, brightness is a unitless 0/1 convention, and count denominators are 15 per glyph / 30 total grid cells. Manual label 1 does not claim machine recognition.',tolerance='Exact integer, shape and index equality; no rounding tolerance used.')},
    {'id':'historical_fixed_font_result','kind':'empirical','status':'verified','statement':'In the existing fixed-font 0-99 digit-string experiment the training probe loss decreased, but only 2/30 final held-out image questions had a fully correct generated digit string; no different font or handwriting was tested.',
     'location':'course/chapters/11.md:439',
     'scope':'Historical run at revision 22a0bb1, CUDA PyTorch 2.14.1+cu126 / Python 3.13.3, audited from saved measurements and matching original code. New checks use CPU 2.14.1+cpu / Python 3.13.5 and do not repeat training or model evaluation. The 30 questions are ten unseen digit strings with three offsets, not thirty independent string families. Lower fixed training-probe loss alone does not show unseen-string exact accuracy or explain the cause of failures.',
     'evidence':[ev('ocr_result','/results/training/initial_loss, /final_loss, /history, /steps, /effective_tokens; /results/data/splits/*/{records,count,sha256}; /results/test/{samples,examples,correct,exact_match}; provenance and code_sha256 pointers','Original measurements/configuration retained with complete source SHA; pointer receipt lists every final inspected field.'),
                 ev('ocr_code','_fit 73-168; _loss_fn 260-283; _evaluate 287-340; _modal/_freeze 367-384; run_ocr 833-881','Fixed eight-example training probe, 500 optimizer updates, whole-string raw-token exact score, digit-string family split and existing dependencies; no font change in run.'),
                 ev('draw_digits','GLYPHS 11-22 and draw_digits 25-36 at historical revision','Single original 5x7 font lookup produces all one/two-digit strings on RGB 16x16; no alternate-font or handwriting generator.'),
                 ev('tokenizer','SPECIALS line12 and ByteTokenizer 14-28','Byte token IDs and EOS boundary for direct scoring recomputation.'),
                 ev('bounded_run','verification.json#/historical_experiment and #/sample_receipt','Recomputed every saved generated_ids decision and all split hashes/counts; correct total 2, denominator30; probe loss 9.281709671020508 to 0.8092095255851746.')],
     'artifact_ids':[ident('verify-execution.json'),ident('verification.json'),ident('original/raw-ocr.json'),ident('source-inspection.json')],
     'verification':ver('Lower fixed training-probe loss and exactly 2 raw-token complete answers out of 30 test questions on disjoint string families.','Probe loss 9.281709671020508 → 0.8092095255851746; recomputed correct=2, examples=30, exact_match=0.06666666666666667; test families10; offsets3 each; 500 history rows and 11584 effective supervised targets.','Every sample row/family/target and generated IDs checked against split records; EOS stripped consistently before equality, invalid specials separately checked, and stored flags/aggregate exact fraction agree. Generator bytes at historical revision match inspected source; font boundary derives from actual code, not result scope prose.',
                         denominators={'test_image_questions':'30','unseen_test_digit_string_families':'10','offset_variants_per_string':'3 (-1,0,1)','train_image_questions':'240','train_digit_string_families':'80','validation_image_questions':'30','validation_digit_string_families':'10','training_optimizer_steps':'500','effective_supervised_targets_including_EOS':'11584','fixed_training_probe_examples':'8 sampled training rows, same deterministic batch before/after'})}
]
report = {
    'schema_version':1,'review_stage':'technical','lesson_id':'11.12','source':'course/chapters/11.md#11.12','source_sha256':sha(body),
    'reviewer_task':inspection['reviewer_task'],'reviewer_context':'fresh','verdict':'pass','reviewed_at':datetime.datetime.now(datetime.UTC).isoformat(),
    'frozen_input':{'path':snapshot['frozen_file_path'],'sha256':snapshot['frozen_file_sha256'],'captured_at':snapshot['frozen_at'],'scope':snapshot['frozen_file_scope'],'receipt_artifact_id':ident('input-snapshot-receipt.json')},
    'figure_sha256':{'course/figures/rewrite-11-glyph-labels.svg':sha((ROOT/'course/figures/rewrite-11-glyph-labels.svg').read_bytes())},
    'read_scope':inspection['implementation_read_scope'],
    'independence_note':inspection['independence'],'sources':sources,'artifacts':artifacts,'claims':claims,'issues':[],
    'checks':{
        'factual_accuracy':{'status':'pass','details':'Complete current 11.12 reviewed. Original pixels/string roles, OCR method existence, supplied-label differentiation, source-family split, RGB contract and bounded historical result are consistent with original code and personally inspected primary/official sources. Seven substantive grouped claims retain their individual support boundaries.','claim_ids':[x['id'] for x in claims]},
        'numeric_verification':{'status':'pass','details':'Ran exact original fence and independent CPU verification. Shape5x3, all30 SVG cells, row0/column0 delta, 15 vs768 input counts, complete 2/30 score, 80/10/10 family splits and historical500-step/11584-target aggregation verified; all denominators named.','claim_ids':['glyph_numbers_and_exercise','historical_fixed_font_result','rgb_entry_contract']},
        'figure_consistency':{'status':'pass','details':'Original SVG hash recorded; real Chromium standalone, desktop1280x800 and mobile390x844 screenshots viewed. White/black cells, string labels0/1 and legend agree with source and exact tensors. All six source body paragraphs, original fence, evidence link and exact served-SVG bytes verified in target page only.','claim_ids':['pixel_label_example','glyph_numbers_and_exercise']},
        'source_verification':{'status':'pass','details':'Personally verified TrOCR arXivv4 first-page identity and original Introduction/architecture/task pipeline; official PyTorch sources pinned to installed git commit; CPythonv3.13.5 and scikit-learn1.7.2 original documents. Fresh HTTPS200 receipts, original snapshots and locator/support notes retained. Source library used only for locating originals.','claim_ids':[x['id'] for x in claims]},
        'limitations':{'status':'pass','details':'Original fence only constructs data. Added CPU check includes one supplied-label gradient and one fresh no-grad encoder shape forward, zero optimizer updates. Historical raw result audited without loading/re-evaluating/training/saving a model. Only fixed5x7 font, ten unseen test strings×three offsets support the2/30 claim; no alternate-font/handwriting or causal failure diagnosis follows. Fullfile SHA explicitly dated as frozen input only.','claim_ids':[x['id'] for x in claims]}
    },
    'verification_boundary':'No material unresolved correctness issue found within these scoped claims. No textbook, figure or implementation edit; no commit. Original implementation experiment metadata was visible as recorded; no prior verdict or author correction summary was read.'
}
target = ROOT / 'docs/technical-reviews/11.12.json'
# Authorized complete replacement. Do not read, merge, or inspect legacy report contents.
target.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({'report':target.relative_to(ROOT).as_posix(),'verdict':report['verdict'],'source_sha256':report['source_sha256'],
                  'report_sha256':sha(target.read_bytes()),'claims':len(claims),'artifacts':len(artifacts)},ensure_ascii=False,indent=2))
