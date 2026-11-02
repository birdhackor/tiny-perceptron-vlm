"""Check only the explicit 15.1 navigation claim about the table in 15.13."""
from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parents[4]
freeze = json.loads((root / 'docs/course-revision-20261029-omni/reviews/P5-actual/source-freeze-v7.json').read_text())
page = next(p for p in freeze['pages'] if p['page_id'] == '15.13')
raw = (root / page['snapshot']).read_bytes()
assert hashlib.sha256(raw).hexdigest() == page['source_sha256']
text = raw.decode()
variants = json.loads((root / 'docs/course-experiments/results/moe.json').read_text())['results']['variants']
assert len(variants) == 7
full_table = text.split('| 模型 | 總參數 | 每token參數代理 | 驗證代價 | 測試代價 |')[1]
rows = [r for r in full_table.splitlines() if r.startswith('| Dense') or r.startswith('| MoE')]
assert len(rows) == 7
assert '39,256、41,914個有效目標' in full_table
for v in variants.values():
    assert v['heldout']['validation']['effective_tokens'] == 39256
    assert v['heldout']['test']['effective_tokens'] == 41914
print(json.dumps({'scope': 'Only navigation/table-presence statement in 15.1; not a separate technical review of 15.13.',
                  'target': page['snapshot'], 'target_sha256': page['source_sha256'],
                  'variant_names': list(variants), 'full_table_rows': len(rows),
                  'validation_effective_tokens_all_seven': 39256,
                  'test_effective_tokens_all_seven': 41914,
                  'all_assertions_passed': True}, ensure_ascii=False, indent=2))
