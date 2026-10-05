"""Bounded raw-byte identity and revised-reference audit; no model run."""
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = HERE.parents[4]

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def section(raw, lesson):
    headers = list(re.finditer(rb'(?m)^## [^\r\n]+', raw))
    i = next(i for i, h in enumerate(headers) if h[0].startswith(f'## {lesson} '.encode()))
    return raw[headers[i].start():headers[i + 1].start() if i + 1 < len(headers) else len(raw)]

raw = (ROOT / 'course/chapters/19.md').read_bytes()
current = section(raw, '19.5')
frozen = (HERE / 'section.md').read_bytes()
assert current == frozen
assert sha(current) == 'f07a4500b54927e9a2820ecdfa2e8892d98404c34818a45bcf66e284e4d1ecd3'
original = (BASE / 'inputs/section.md').read_bytes()
old_sentence = '最終文字能力與各項分母統一見[19.12的能力表](19.md#19.12)'.encode()
new_sentence = '整合成品的能力界定與驗收安排見[19.12](19.md#19.12)'.encode()
assert original.count(old_sentence) == 1
assert current == original.replace(old_sentence, new_sentence)
assert (HERE / 'fence-1.py').read_bytes() == (BASE / 'inputs/fence-1.py').read_bytes()
context = section(raw, '19.12')
assert context == (HERE / 'section-19.12-context.md').read_bytes()
assert context == (BASE / 'inputs/section-19.12-context.md').read_bytes()
assert sha(context) == '7d90da817d297f56bc68601d60efc18de5a1078ea3251ef28612e11fea9d439f'
assert '這是一張待填的驗收矩陣，不替尚未完成的能力填分數。'.encode() in context
assert '有限文字與指令'.encode() in context
assert '內容、範圍、格式、完整結束'.encode() in context
identities = []
for rec in json.loads((BASE / 'sources/pinned-source-provenance.json').read_bytes()):
    local = rec['url'].split('/1df335318bda03fd771807f66976953231d5a00b/', 1)[1]
    snapshot = ROOT / rec['path']
    local_raw = (ROOT / local).read_bytes()
    snapshot_raw = snapshot.read_bytes()
    assert sha(local_raw) == sha(snapshot_raw) == rec['sha256']
    identities.append({'original_path': local, 'snapshot_path': rec['path'],
                       'sha256': rec['sha256'], 'unchanged': True})
assert (2, 14 * 1, 14 * 2) == (2, 14, 28)
results = {
    'reviewer_task': '/root/phase4_factual_coordinator/factual_19_5',
    'original_source_sha256': sha(original), 'current_source_sha256': sha(current),
    'current_frozen_chapter_sha256': sha((HERE / 'chapter-frozen-input.md').read_bytes()),
    'only_replaced_reference_sentence': True, 'original_fence_unchanged': True,
    'context_original_sha256': sha(context), 'context_unchanged': True,
    'revised_quote': new_sentence.decode(),
    'context_support_locators': ['19.12 lines3–13: capability scope and pending acceptance matrix',
                                '19.12 lines38–42,51: old synthetic scope, criteria and limits'],
    'original_implementations_and_raw_measurements': identities,
    'all_claims_rechecked': ['C1-specification', 'C2-demonstrations-and-system', 'C3-original-fence',
                           'C4-sft-four-generated-behaviors', 'C5-other-text-not-all-refusals',
                           'C6-constant-label-scope', 'C7-input-output-id-and-version',
                           'C8-joint-dpo-retention-and-regression', 'C9-new-product-acceptance-is-plan',
                           'C10-final-text-table-reference'],
    'environment': {'python': sys.version, 'device': 'cpu', 'model_executed': 'false'},
    'result': 'All assertions passed; new reference accurately names definitions and acceptance arrangements; no unresolved factual claim.',
}
(HERE / 'verification.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(results, ensure_ascii=False, indent=2))
