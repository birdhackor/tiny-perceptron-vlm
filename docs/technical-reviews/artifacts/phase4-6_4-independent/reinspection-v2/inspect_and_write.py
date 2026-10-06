"""Same original owner: actual current whole-read, claim support checks and immutable-proof reuse."""
import difflib
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path('/workspace/tiny-perceptron-vlm')
BASE = Path('docs/technical-reviews/artifacts/phase4-6_4-independent')
CURRENT = BASE / 'reinspection-v2'
OUT = ROOT / CURRENT
EXPECTED_SOURCE = 'f5f3c1976333b0561be17a101a519076d4e8c316b9b0707557d687394c5d7234'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


prepared = json.loads((OUT / 'prepare-receipt.json').read_text())
prior_raw = (OUT / 'prior-report.json').read_bytes()
assert sha(prior_raw) == prepared['prior_report_sha256']
assert (ROOT / prepared['history_path']).read_bytes() == prior_raw
report = json.loads(prior_raw)  # Only this same reviewer's own prior report is read.
body = (OUT / 'section.md').read_bytes()
assert sha(body) == EXPECTED_SOURCE
spec = importlib.util.spec_from_file_location('section_facts_current_6_4', ROOT / 'docs/review-tools/section_facts.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
active_body, active_whole, active_first_line = helper.original_section(ROOT / 'course/chapters/06.md', '6.4')
assert active_body == body
assert (OUT / 'chapter-06.md').read_bytes() == active_whole
old_body = (ROOT / BASE / 'recheck/section.md').read_bytes()
assert sha(old_body) == report['source_sha256']
assert old_body.replace('接着'.encode(), '接著'.encode()) == body
diff = ''.join(difflib.unified_diff(old_body.decode().splitlines(keepends=True), body.decode().splitlines(keepends=True), fromfile='same-owner-prior-section', tofile='same-owner-frozen-current-section'))
(OUT / 'section.diff').write_text(diff)
old_fence = (ROOT / BASE / 'original/fence-1.py').read_bytes()
assert (OUT / 'fence-1.py').read_bytes() == old_fence
assert not re.findall(r'!\[[^\]]*\]\([^)]+\.(?:svg|png|jpg|webp)\)', body.decode())
old_context, _, _ = helper.original_section(ROOT / BASE / 'recheck/chapter-06.md', '6.3')
new_context, _, _ = helper.original_section(ROOT / 'course/chapters/06.md', '6.3')
(OUT / 'context-6.3.md').write_bytes(new_context)
context_equal = old_context == new_context

reused = []
for artifact in report['artifacts']:
    actual = sha((ROOT / artifact['path']).read_bytes())
    assert actual == artifact['sha256'], artifact['id']
    reused.append({'id': artifact['id'], 'permanent_path': artifact['path'], 'expected_sha256': artifact['sha256'], 'actual_sha256': actual, 'reuse': 'unchanged original proof; no numerical rerun'})
provenance = json.loads((ROOT / BASE / 'input-provenance.json').read_text())
historical = []
for entry in provenance['historical_code']:
    command = ['git', 'show', provenance['historical_revision'] + ':' + entry['original_path']]
    raw = subprocess.run(command, cwd=ROOT, capture_output=True, check=True, timeout=15).stdout
    actual = sha(raw)
    assert actual == entry['expected_sha256']
    historical.append({'original_path': entry['original_path'], 'command_argv': command, 'revision': provenance['historical_revision'], 'expected_sha256': entry['expected_sha256'], 'actual_sha256': actual, 'permanent_copy': entry['path']})
inputs = []
for entry in provenance['inputs']:
    raw = (ROOT / entry['original_path']).read_bytes()
    actual = sha(raw)
    assert actual == entry['expected_sha256']
    assert actual == sha((ROOT / entry['path']).read_bytes())
    inputs.append({'original_locator': entry['original_path'], 'expected_sha256': entry['expected_sha256'], 'actual_sha256': actual, 'required_permanent_copy': entry['path'], 'original_cache_required_for_formal_review': False})
original_record = (ROOT / 'docs/course-experiments/results/tokenizer.json').read_bytes()
saved_record = (ROOT / BASE / 'inputs/tokenizer-result.json').read_bytes()
assert original_record == saved_record  # Hash only, no author note/correction fields opened.

import torch
import tokenizers
assert torch.version.cuda is None and not torch.cuda.is_available()
environment = {'python': sys.version, 'executable': sys.executable, 'torch': str(torch.__version__), 'torch_git_version': str(torch.version.git_version), 'tokenizers': tokenizers.__version__, 'device': 'cpu', 'cuda_build': str(torch.version.cuda)}
(OUT / 'environment.json').write_text(json.dumps(environment, ensure_ascii=False, indent=2) + '\n')
proof = json.loads((ROOT / BASE / 'bounded-results.json').read_text())
assert [proof['historical_saved_tokenizer_recomputation'][name]['ordinary_validation_tokens'] for name in ['byte256', 'bpe512']] == [3894, 2112]
assert [proof['historical_saved_tokenizer_recomputation'][name]['parameters'] for name in ['byte256', 'bpe512']] == [41824, 57696]
assert [run['V'] for run in proof['fixed_v_piece_length_counterexample']['runs']] == [260, 260]
claim_checks = {
    'piece-size-versus-V': {'actual_judgment': '首句仍明示同一width及V擴大，原固定V260/T3→1而參數8320不變反例已被正確排除於增加參數的前提；原resolved I-6.4-1保持成立。', 'support_reinspected': 'Sennrich§3.1–3.2 p1717、Embedding/Linear形狀；own fixed-v counterexample永久SHA未變。', 'substantive_changed': False},
    'table-cost-and-exercises': {'actual_judgment': 'V×32、4bytes/FP32、兩個untied weight表2V×32與V300/1000/3000數值，width64及V600意義/數值全未變；只將「接着」正體寫成「接著」，不改推論。', 'support_reinspected': 'FP32 dtype原表、Embedding/Linear API及原真stdout/width-V variation proof指紋；weight payload不是整機budget。', 'substantive_changed': False},
    'matching-merges-and-attention': {'actual_judgment': 'T減少仍限實際合併覆蓋文本；T×T限單篇dense attention，不把19篇總tokens當一個矩陣，也未新增速度或全模型更省保證。', 'support_reinspected': 'Sennrich§3.2及Vaswani§3.4/Table1，Eq1原支持及own per-record token長度永久proof未變。', 'substantive_changed': False},
    'coverage-heldout-and-roundtrip': {'actual_judgment': '英文新增merge可能幫不上中文的限制與同文heldout/roundtrip/品質核對仍完整；必要6.3把train153/val19/test20、完整byte alphabet/normalizer/prefix條件說明一致。', 'support_reinspected': 'current6.3全文、official BpeTrainer compute_alphabet及ByteLevel pre_tokenize/decode_chain；英文微例V264/300各22tokens、原192篇exact roundtrips proof的SHA未變。', 'substantive_changed': False},
    'rare-pieces-and-no-universal-V': {'actual_judgment': '原rare-token句仍按少直接input/正例理解，不泛稱dense output無梯度；小片段復用/序列可能長與任務成本取捨未新增通用最佳V。', 'support_reinspected': 'Sennrich§3/§3.1 task-specific原文；own rare-row gradient原實跑SHA未變。', 'substantive_changed': False},
    'historical-19-record-tradeoff': {'actual_judgment': '原19共同篇ordinary tokens3894/2112、不含BOS/EOS、means205/111、V264/512、模型41824/57696及samewidth32/layer1全部不變；分母/版本與原proof仍相符。', 'support_reinspected': '原result完整byte SHA一致（只hash，未讀作者附註），23historical code gitshow SHA及8原input SHA又真核相符；own original CPU split/length/params proof永久SHA一致。原CUDA成績身分保留，不稱本轮新training。', 'substantive_changed': False},
}
receipt = {
    'schema_version': 1,
    'reviewer_task': '/root/phase4_factual_coordinator/factual_6_4',
    'reviewer_role': 'same original independent technical owner; third actual full-section reinspection',
    'date': '2026-10-05',
    'source': 'course/chapters/06.md#6.4',
    'source_sha256': sha(body),
    'frozen_whole_file': {'path': (CURRENT / 'chapter-06.md').as_posix(), 'sha256': sha(active_whole), 'meaning': 'entire Markdown captured at this actual read; not a permanent assertion about future whole-chapter version'},
    'prior_report': {'history_path': prepared['history_path'], 'sha256': prepared['history_sha256'], 'opaque_snapshot_artifact_path': (CURRENT / 'prior-report.json').as_posix()},
    'prior_source_sha256': sha(old_body),
    'whole_read': {'performed': True, 'range': '6.4完整原文31行含fence與展開補充；current6.3完整必要上下文也親讀。', 'first_line': active_first_line, 'line_count': len(body.splitlines()), 'conclusion': '全節原意/主張/範圍逐項核對；和我已核實前版僅「接着」→「接著」，沒有changed substantive claim或新疑問。'},
    'actually_changed': [{'section_line': 21, 'before': '接着把V300改600', 'after': '接著把V300改600', 'classification': 'orthography only; same instruction and meaning', 'claim_id': 'table-cost-and-exercises'}],
    'changed_claims': [],
    'code_unchanged': {'raw_equal_to_original_true_run': True, 'sha256': sha(old_fence), 'new_execution': False},
    'necessary_context': {'path': (CURRENT / 'context-6.3.md').as_posix(), 'sha256': sha(new_context), 'equal_to_same_owner_prior_context': context_equal, 'support': 'formal tokenizer preparation/splits/roundtrip definitions; current wording personally inspected'},
    'figure_applicability': {'status': 'not_applicable', 'details': '6.4亲读原文没有图/SVG引用/图片素材/坐标箭头，也没有新增视觉claim；无需臆造render/view或页面验收。未使用preview作视觉证据。'},
    'chapter_intro': {'status': 'not_applicable', 'reason': '6.4不是章首小节'},
    'claim_support_checks': claim_checks,
    'own_permanent_proofs_hash_verified': reused,
    'historical_code_hash_verified_again': historical,
    'original_inputs_hash_verified_again': inputs,
    'raw_result_hash': {'path': 'docs/course-experiments/results/tokenizer.json', 'sha256': sha(original_record), 'equal_to_permanent_snapshot': True, 'inspection_this_time': 'raw bytes/hash only; no author correction/review notes read'},
    'reuse_scope': 'All previous numerical CPU proof, saved original JSON, original official-source snapshots and exact fence are unchanged and support exactly the same claims. After this reviewer checked bytes/hash and support, reuse is explicit. This command performs new provenance/hash checks and report persistence, not another model/tokenizer training or repeated original numerical experiment.',
    'no_author_repair_answers_or_other_reviews_read': True,
    'verdict': 'pass',
    'unresolved_issues': [],
    'environment': environment,
    'command': '.venv/bin/python docs/technical-reviews/artifacts/phase4-6_4-independent/reinspection-v2/inspect_and_write.py > docs/technical-reviews/artifacts/phase4-6_4-independent/reinspection-v2/inspect.stdout.txt 2> docs/technical-reviews/artifacts/phase4-6_4-independent/reinspection-v2/inspect.stderr.txt',
}
(OUT / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')

report['source_sha256'] = sha(body)
report['verdict'] = 'pass'
report['latest_reinspection'] = {'canonical_artifact_id': 'current-reinspection-6_4', 'receipt_path': (CURRENT / 'receipt.json').as_posix(), 'receipt_sha256': sha((OUT / 'receipt.json').read_bytes()), 'same_reviewer_task': receipt['reviewer_task'], 'prior_history_path': prepared['history_path'], 'prior_history_sha256': prepared['history_sha256'], 'actual_changed_scope': receipt['actually_changed'], 'new_substantive_claims': [], 'unresolved_issues': []}
report['review_scope']['actually_read'].append('第三次本人真複查：current6.4完整原文及current6.3必要上下文，frozen source f5f3c197…；親比本人前次新版只「接着」→「接著」，逐claim支持與全部永久proof/hash核對記於current-reinspection-6_4。')
report['review_scope']['current_reinspection'] = receipt['reuse_scope']
report['review_scope']['frozen_current_markdown'] = receipt['frozen_whole_file']
for claim in report['claims']:
    check = claim_checks[claim['id']]
    claim['latest_reinspection'] = check
    claim['artifact_ids'].append('current-reinspection-6_4')
    claim['evidence'].append({'source_id': 'current-reinspection-source', 'locator': 'reinspection-v2/receipt.json#/claim_support_checks/' + claim['id'], 'supports': check['actual_judgment'] + ' ' + check['support_reinspected']})
report['sources'].append({'id': 'current-reinspection-source', 'kind': 'execution', 'title': 'Same-owner current whole-read, immutable-proof/version reuse verification', 'verified': True, 'artifact_id': 'current-reinspection-6_4'})
for check_name in ['factual_accuracy', 'numeric_verification', 'source_verification', 'limitations']:
    report['checks'][check_name]['details'] += ' 本人第三次完整讀現節，唯一orthographic改字不改此支持；已真核舊永久證據hash與支持，明示沿用而非重跑，詳current-reinspection-6_4。'
report['checks']['figure_consistency']['details'] = receipt['figure_applicability']['details']


def add_artifact(identifier, rel, kind, description, **extra):
    path = CURRENT / rel
    report['artifacts'].append({'id': identifier, 'path': path.as_posix(), 'sha256': sha((ROOT / path).read_bytes()), 'kind': kind, 'description': description, **extra})


add_artifact('current-prior-report-6_4', 'prior-report.json', 'source_snapshot', '本人上一輪canonical report先opaque保存原bytes；formal副本與history SHA一致。')
add_artifact('current-section-6_4', 'section.md', 'source_snapshot', '本人本輪親讀的current完整6.4原UTF8 bytes。')
add_artifact('current-frozen-chapter-6_4', 'chapter-06.md', 'source_snapshot', '本輪frozen整章輸入；不是對未來wholechapter版本的宣稱。')
add_artifact('current-context-6_4', 'context-6.3.md', 'source_snapshot', '本輪親讀的必要6.3 tokenizer/splits/roundtrip定義。')
add_artifact('current-diff-6_4', 'section.diff', 'source_snapshot', '本人前次section/current section實際diff，只改正體字。')
add_artifact('current-fence-6_4', 'fence-1.py', 'code', '現原fence原bytes，親核與已真正執行原fence相等，因此明示reuse。')
add_artifact('current-prepare-code-6_4', 'prepare.py', 'code', '先保存本人prior/history及frozen current原稿的實際程式。')
add_artifact('current-inspection-code-6_4', 'inspect_and_write.py', 'code', '本人真執行的current proof/hash檢查與逐claim復查保存程式。')
add_artifact('current-environment-6_4', 'environment.json', 'source_snapshot', '本次實際CPU驗證環境；不把原CUDA訓練結果換成本輪CPU成績。')
add_artifact('current-reinspection-6_4', 'receipt.json', 'execution', 'Same original owner整節親讀、差異分類、逐claim支持、50件own proof/23historical code/8原輸入SHA真核對與明确reuse。', command=receipt['command'], result='exit0；current source f5f3c197…；只有「接着」→「接著」；50 own artifacts、23historical code、8原input全部SHA相符；6claims支持範圍成立、無未解問題，判pass；未再跑數值proof或訓練。', environment=environment)
(ROOT / 'docs/technical-reviews/6.4.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
(OUT / 'canonical-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'verdict': report['verdict'], 'report_sha256': sha((ROOT / 'docs/technical-reviews/6.4.json').read_bytes()), 'source_sha256': report['source_sha256'], 'prior_history_path': prepared['history_path'], 'prior_history_sha256': prepared['history_sha256'], 'canonical_artifact_id': 'current-reinspection-6_4', 'receipt_path': (CURRENT / 'receipt.json').as_posix(), 'receipt_sha256': sha((OUT / 'receipt.json').read_bytes()), 'actual_changed_scope': '接着→接著; no changed substantive claim', 'unresolved_issues': []}, ensure_ascii=False, indent=2))
