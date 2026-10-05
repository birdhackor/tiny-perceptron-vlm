"""Record this same reviewer's actual whole-section reread and evidence reuse after local revision."""
import difflib
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path('/workspace/tiny-perceptron-vlm')
BASE = Path('docs/technical-reviews/artifacts/phase4-6_4-independent')
OUT = ROOT / BASE
RECHECK = OUT / 'recheck'
RECHECK.mkdir(exist_ok=True)
HISTORY = Path('docs/technical-reviews/history/phase4-6_4-own-initial-revise-f9204d6388545982314914fbdae174f153c164f6871c0320402068a09ef48b53.json')
EXPECTED_INITIAL = 'f9204d6388545982314914fbdae174f153c164f6871c0320402068a09ef48b53'
EXPECTED_REVISED = 'd5477cd6fd4797b45c968665f0e788757cb9767b389003a56304463105d4c8a5'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


helper_path = ROOT / 'docs/review-tools/section_facts.py'
spec = importlib.util.spec_from_file_location('section_facts_recheck', helper_path)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
body, whole, first_line = helper.original_section(ROOT / 'course/chapters/06.md', '6.4')
original_body = (OUT / 'original/section.md').read_bytes()
assert sha(body) == EXPECTED_REVISED
history_raw = (ROOT / HISTORY).read_bytes()
assert sha(history_raw) == EXPECTED_INITIAL
assert history_raw == (OUT / 'initial-review.json').read_bytes()
initial = json.loads(history_raw)
assert sha(original_body) == initial['source_sha256']
old_lines = original_body.splitlines(keepends=True)
new_lines = body.splitlines(keepends=True)
assert len(old_lines) == len(new_lines)
changed_lines = [index + 1 for index, (old, new) in enumerate(zip(old_lines, new_lines, strict=True)) if old != new]
assert changed_lines == [3], changed_lines
assert new_lines[2].decode().startswith('同一特徵寬度下，擴大詞表、加入更多合併項，可能讓句子用較少格，也需保存更多輸入特徵與輸出候選。')
(RECHECK / 'section.md').write_bytes(body)
(RECHECK / 'chapter-06.md').write_bytes(whole)
diff = ''.join(difflib.unified_diff(original_body.decode().splitlines(keepends=True), body.decode().splitlines(keepends=True), fromfile='original/section.md', tofile='recheck/section.md'))
(RECHECK / 'section.diff').write_text(diff)

# The original fence and all earlier permanent evidence are immutable; do not claim another run.
fences = helper.fences(body, first_line)
assert len(fences) == 1 and fences[0]['language'] == 'python'
assert fences[0]['raw'] == (OUT / 'original/fence-1.py').read_bytes()
(RECHECK / 'fence-1.py').write_bytes(fences[0]['raw'])
reuse = []
for artifact in initial['artifacts']:
    path = ROOT / artifact['path']
    actual = sha(path.read_bytes())
    assert actual == artifact['sha256'], artifact['id']
    reuse.append({'artifact_id': artifact['id'], 'path': artifact['path'], 'expected_sha256': artifact['sha256'], 'actual_sha256': actual, 'matches': True, 'rerun': False})

import torch
import tokenizers
assert torch.version.cuda is None and not torch.cuda.is_available()
environment = {'python': sys.version, 'executable': sys.executable, 'torch': str(torch.__version__), 'torch_git_version': str(torch.version.git_version), 'tokenizers': tokenizers.__version__, 'device': 'cpu', 'cuda_build': str(torch.version.cuda)}
(RECHECK / 'environment.json').write_text(json.dumps(environment, ensure_ascii=False, indent=2) + '\n')
original_environment = json.loads((OUT / 'bounded-environment.json').read_text())
for field in ['python', 'torch', 'torch_git_version', 'tokenizers', 'device', 'cuda_build']:
    assert environment[field] == original_environment[field]
historical_model = (OUT / 'historical/tiny_perceptron/model.py').read_text()
assert 'tied: bool = False' in historical_model
assert 'if c.tied:\n            self.output.weight = self.embedding.weight' in historical_model
fixed_v = json.loads((OUT / 'bounded-results.json').read_text())['fixed_v_piece_length_counterexample']
assert [run['V'] for run in fixed_v['runs']] == [260, 260]
assert [run['sequence_length'] for run in fixed_v['runs']] == [3, 1]
assert [run['input_parameters_width32'] for run in fixed_v['runs']] == [8320, 8320]
recheck = {
    'reviewer_task': '/root/phase4_factual_coordinator/factual_6_4',
    'date': '2026-10-05',
    'actual_inspection': '本審閱者收到followup後，親讀新版6.4從標題至</details>完整31行，包括原fence、公式/數字解釋、T與中文覆蓋段、rare token段、width/V練習與原19篇補充；不是只按協調者意見或更新hash。再親讀原Sennrich§3.2、官方Embedding/Linear shape、saved固定V反例，以及historical ModelConfig與TinyLM sharing分支。',
    'source_sha256_before': sha(original_body),
    'source_sha256_after': sha(body),
    'source_file_sha256_after': sha(whole),
    'section_first_line': first_line,
    'line_count': len(new_lines),
    'changed_section_lines': changed_lines,
    'new_opening_exact': new_lines[2].decode().rstrip('\n'),
    'original_fence_sha256': sha(fences[0]['raw']),
    'original_fence_bytes_equal': True,
    'execution_reuse': '原fence、width64/V600、有界token/參數/split/roundtrip/反例結果全部親核永久bytes與原report hash一致；本輪不再跑相同CPU算例、不重訓。此command真執行原source/fence/artifact hash與metadata檢查，不把reuse寫成rerun。',
    'evidence_reused': reuse,
    'issue_resolution_judgment': 'I-6.4-1已解決：新增同一特徵寬度且V擴大的前提，固定V而token變長的反例不再符合新版前提。由官方V×d矩陣shape可直接推出固定d時V增則保存更多weight數與output種類；可能減T仍只在合併覆蓋原文時成立，新版與後段一致。未見新增實質主張或未解問題。',
    'source_reinspection_locators': ['Sennrich ACL2016 P16-1162 §3.2 p1717: initial alphabet+merge operations決定V、頻繁片段可合併', 'PyTorch v2.11.0 Embedding Attributes: weight(num_embeddings,embedding_dim)', 'PyTorch v2.11.0 Linear Attributes: weight(out_features,in_features)', 'historical model.py L24–25 tied=False, L64–66 output.bias=False / if c.tied sharing', 'bounded-results.json fixed_v_piece_length_counterexample 原真run：V260兩種詞表，T3/1、參數8320相同'],
    'report_field_correction': {'old': 'historical-model.inspection_note: untied默認False', 'correct': '原字段tied=False（預設不共享輸入與輸出weight），不是untied=False。', 'cause': '本審閱者原報告字段措辭寫反，親查原ModelConfig字段後修正；原count程式以默認tied=False實跑，數字41824/57696及差15872不變。'},
    'figures': '無引用圖；完整親讀後仍無圖可render/view，非漏驗。',
    'chapter_intro': '6.4非章首；not_applicable。',
    'no_training_or_new_score': True,
    'environment': environment,
}
(RECHECK / 'whole-read-inspection.json').write_text(json.dumps(recheck, ensure_ascii=False, indent=2) + '\n')

report = initial
report['source_sha256'] = sha(body)
report['verdict'] = 'pass'
report['author_tasks'] = ['/root/phase4_factual_coordinator']
report['review_history'] = {'initial_verdict': 'revise', 'initial_source_sha256': sha(original_body), 'initial_report_path': HISTORY.as_posix(), 'initial_report_sha256': EXPECTED_INITIAL, 'initial_artifact_report_path': (BASE / 'initial-review.json').as_posix(), 'same_reviewer_actual_recheck': (BASE / 'recheck/whole-read-inspection.json').as_posix(), 'scope': '保留本人原始needs_revision claim、反例及open問題於immutable initial report；現報告明列original statement/status與實際resolution，而非刪去問題換pass。'}
claim = next(claim for claim in report['claims'] if claim['id'] == 'piece-size-versus-V')
claim['initial_statement'] = claim['statement']
claim['initial_status'] = claim['status']
claim['statement'] = '新版開場：同一特徵寬度下，擴大詞表、加入更多合併項，可能讓句子用較少格，也需保存更多輸入特徵與輸出候選。'
claim['status'] = 'verified'
claim['scope'] = '明示固定width且V擴大才需更多V×width weight數與輸出候選；T減少仍是可能並需合併覆蓋文本。原固定V反例保留作為條件邊界，不再反駁新版前提。'
claim['recheck'] = {'original_issue_id': 'I-6.4-1', 'judgment': recheck['issue_resolution_judgment'], 'original_counterexample_retained': 'bounded-results.json fixed_v_piece_length_counterexample', 'not_a_claim_deletion': 'initial_statement與initial_status=needs_revision保留，完整原report及其SHA保留。'}
claim['artifact_ids'].append('recheck-whole-read')
for source in report['sources']:
    if source['id'] == 'historical-model':
        source['inspection_note'] = source['inspection_note'].replace('untied默認False', '字段tied=False，預設不共享輸入與輸出weight')
issue = report['issues'][0]
issue['initial_status'] = issue['status']
issue['status'] = 'resolved'
issue['revised_quote'] = recheck['new_opening_exact'].split('詞表大小的取捨')[0]
issue['resolution'] = recheck['issue_resolution_judgment'] + ' 本輪重新親讀全節、核diff僅第3行、核原fence與全部永久證據hash完全相同，明記reuse而非rerun。'
issue['actual_recheck'] = {'reviewer_task': recheck['reviewer_task'], 'source_sha256': sha(body), 'inspection_artifact': 'recheck-whole-read', 'code_artifact': 'recheck-code'}
report['issues'].append({'id': 'R-6.4-model-field-wording', 'status': 'resolved', 'severity': 'review_record_wording', 'original_quote': 'untied默認False', 'problem': '我原inspection_note錯寫字段名稱/布林意義，實際原碼字段為tied=False。', 'resolution': recheck['report_field_correction']['correct'] + ' ' + recheck['report_field_correction']['cause'], 'location': 'report sources[historical-model].inspection_note', 'claim_ids': ['historical-19-record-tradeoff']})
report['checks']['factual_accuracy'] = {'status': 'pass', 'details': '完整親讀新版全節；首句現在明示固定width及V增加，I-6.4-1以原paper/官方weight shape與固定V反例真複查解決；其他主張與原bytes相同，逐項原永久證據hash親核後reuse。', 'claim_ids': [claim['id'] for claim in report['claims']]}
report['review_scope']['actually_read'].append('followup：新版course/chapters/06.md#6.4完整31行，再親讀Sennrich§3.2、Embedding/Linear shape、historical tied字段及fixed-V真結果；完整全節結论與hash/diff/evidence reuse記於recheck。')
report['review_scope']['recheck_execution'] = recheck['execution_reuse']


def add_artifact(identifier, rel, kind, description, **extra):
    path = BASE / rel
    report['artifacts'].append({'id': identifier, 'path': path.as_posix(), 'sha256': sha((ROOT / path).read_bytes()), 'kind': kind, 'description': description, **extra})


add_artifact('initial-own-review', 'initial-review.json', 'source_snapshot', '本人完整初輪revise報告原bytes，SHA f9204d63…；保留needs_revision claim/反例/未解問題。')
add_artifact('recheck-section', 'recheck/section.md', 'source_snapshot', '本輪親讀的新版完整原小節UTF8 bytes。')
add_artifact('recheck-chapter', 'recheck/chapter-06.md', 'source_snapshot', '實際新版whole chapter原bytes，供source版本追查；不代替本節完整親讀。')
add_artifact('recheck-diff', 'recheck/section.diff', 'source_snapshot', '本人原小節與新版逐byte文本diff，只有開場第3行。')
add_artifact('recheck-fence', 'recheck/fence-1.py', 'code', '新版fence raw bytes與已真執行原fence完全相同；本輪核hash後reuse，不宣稱重跑。')
add_artifact('recheck-environment', 'recheck/environment.json', 'source_snapshot', '本轮实际CPU版本/装置；與原有界驗證環境一致。')
add_artifact('recheck-code', 'recheck_review.py', 'code', '本審閱者實際執行的復查保存/原bytes比對/evidence hash核對程式。')
add_artifact('recheck-whole-read', 'recheck/whole-read-inspection.json', 'execution', '新版完整親讀記錄、逐byte/fence/每件proof hash真核對、原問題resolution、tied字段更正與明确reuse範圍。', command='.venv/bin/python docs/technical-reviews/artifacts/phase4-6_4-independent/recheck_review.py > docs/technical-reviews/artifacts/phase4-6_4-independent/recheck.stdout.txt 2> docs/technical-reviews/artifacts/phase4-6_4-independent/recheck.stderr.txt', result='实际exit0；新source d5477c…，仅原節第3行改、原fence/所有proof hash完全相等；I-6.4-1解決，历史字段tied=False更正。', environment=environment)
(ROOT / 'docs/technical-reviews/6.4.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
(RECHECK / 'final-review.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'verdict': report['verdict'], 'source_sha256': report['source_sha256'], 'report_sha256': sha((ROOT / 'docs/technical-reviews/6.4.json').read_bytes()), 'initial_report_sha256': EXPECTED_INITIAL, 'recheck_sha256': sha((RECHECK / 'whole-read-inspection.json').read_bytes()), 'changed_section_lines': changed_lines, 'prior_evidence_hashes_rechecked': len(reuse), 'rerun_original_fence': False, 'new_unresolved_issues': 0, 'report_field_corrected': 'tied=False'}, ensure_ascii=False, indent=2))
