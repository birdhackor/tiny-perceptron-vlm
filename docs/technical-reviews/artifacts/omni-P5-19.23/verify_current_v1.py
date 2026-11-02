"""Reviewer callback: full current lesson, saved-logit recomputation, synthetic CPU only.

The immutable original portable probe is reused with explicit changes to the
lesson/output paths and the now-fixed counterexample expectation. No original
proof is overwritten, and no trained model is loaded.
"""

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
original_probe = OUT / 'recompute_and_inline.py'
assert hashlib.sha256(original_probe.read_bytes()).hexdigest() == 'cd5b2ce70ddcf994807208703d5100778ecb2e26234f33eab5f878ac0012b9c2'
source = original_probe.read_text()
changes = [
    ('source-freeze-v3-final/sources/19.23.md', 'source-freeze-v6/sources/19.23.md'),
    ("and tie_selected == 'a'", "and tie_selected == 'b'"),
    ("'recompute-output.json'", "'recompute-output-current-v1.json'"),
    ('Only short TinyDecision selected(); full ChoiceModel correctly selects raw logits.',
     'Current short TinyDecision selected() and full ChoiceModel both select raw logits; original false tie is fixed.'),
]
for old, new in changes:
    assert source.count(old) == 1, old
    source = source.replace(old, new)
namespace = {'__name__': 'current_19_23_portable_review', '__file__': str(__file__)}
exec(compile(source, 'reused-original-review-probe-with-explicit-current-expectations', 'exec'), namespace)

torch = namespace['torch']
module = namespace['module']
from tiny_perceptron.omni_products.inference import CJudge
from tiny_perceptron.omni_products.reliability import calibrated


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


permutations = []
for ids, scores, expected in [
    (['a', 'b'], [0.0, 1e-8], 'b'),
    (['b', 'a'], [1e-8, 0.0], 'b'),
    (['a', 'b'], [0.0, 0.0], 'a'),
    (['b', 'a'], [0.0, 0.0], 'a'),
    (['card', 'address'], [0.0, 0.0], 'address'),
    (['address', 'card'], [0.0, 0.0], 'address'),
]:
    tensor = torch.tensor(scores, dtype=torch.float32)
    p = tensor.softmax(0)
    observed = module.selected({'ids': ids, 'scores': tensor, 'p': p})
    assert observed == expected
    permutations.append({'ids': ids, 'scores': tensor.tolist(), 'p': p.tolist(),
                         'expected': expected, 'observed': observed})

finite_guards = []
for name, temperature in [('NaN', math.nan), ('positive_infinity', math.inf), ('zero', 0), ('negative', -1)]:
    for api in ['CJudge', 'calibrated']:
        try:
            if api == 'CJudge':
                CJudge(module.formal_c, temperature=temperature)
            else:
                calibrated([0.0, 1e-8], temperature)
        except ValueError as error:
            finite_guards.append({'api': api, 'temperature': name, 'rejected': True, 'error': str(error)})
        else:
            raise AssertionError((api, name, 'should reject'))

for temperature in [1.0, namespace['T']]:
    CJudge(module.formal_c, temperature=temperature)
    assert all(math.isfinite(x) for x in calibrated([0.0, 1e-8], temperature))

lesson = ROOT / 'docs/course-revision-20261029-omni/reviews/P5-actual/source-freeze-v6/sources/19.23.md'
text = lesson.read_text()
precondition_quote = '直接呼叫這裡的`ChoiceModel.predict`時，呼叫端須先確保`temperature`為正有限值；下例沿用預設1.0。這個method只檢查`temperature <= 0`，會漏過NaN與正無窮；正式服務的`CJudge`與校準入口另用`math.isfinite`拒絕非有限值，本輪固定T也通過該檢查。'
assert precondition_quote in text
assert '正式C-v2的文字輸入驗證、編碼、共享材料與候選計算' in text
freezes = []
for filename in ['source-freeze-v6.json', 'source-freeze-v7.json']:
    p = ROOT / 'docs/course-revision-20261029-omni/reviews/P5-actual' / filename
    f = json.loads(p.read_text())
    page = next(row for row in f['pages'] if row['page_id'] == '19.23')
    assert page['source_sha256'] == digest(lesson)
    assert page['figures_sha256'] == {}
    freezes.append({'path': str(p.relative_to(ROOT)), 'sha256': digest(p),
                    'source_sha256': page['source_sha256'], 'same_assigned_source_bytes': True})
result = {
    'created_at': datetime.now(timezone.utc).isoformat(),
    'reviewer_task': '/root/omni_p5_technical_19_23',
    'full_current_622_line_source_personally_read': True,
    'source_sha256': digest(lesson), 'freezes': freezes,
    'explicit_reuse_changes_to_own_original_probe': changes,
    'tiny_raw_score_and_true_tie_cases': permutations,
    'formal_service_and_calibration_invalid_temperature_cases': finite_guards,
    'valid_default_and_recorded_T_accepted': [1.0, namespace['T']],
    'standalone_precondition_quote_actually_present': precondition_quote,
    'standalone_method_not_claimed_to_have_acquired_a_finite_guard': True,
    'formal_inline_AST_still_equals_original_repository_implementation': True,
    'source_code_model_recipe_rows_weights_and_selection_not_changed_by_review': True,
    'no_trained_NN_rerun_training_or_T_tau_retuning': True,
    'environment': namespace['result']['environment'],
    'unverified_scope': namespace['result']['limits'],
}
(OUT / 'issue-closure-output-current-v1.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
print(json.dumps({'issue_0_raw_score_counterexample_and_ties': 'all six pass',
                  'issue_1_documented_precondition': 'present and exact',
                  'service_and_calibration_invalid_T': 'all eight rejected',
                  'formal_NN_or_training_rerun': False}, ensure_ascii=False))
