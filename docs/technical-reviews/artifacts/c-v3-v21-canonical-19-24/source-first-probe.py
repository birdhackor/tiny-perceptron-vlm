"""19.24 的新 source-first 小型核對；不載入模型權重、不訓練、不取新 final。"""

import contextlib
import hashlib
import io
import itertools
import json
import platform
import re
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import torch

from scripts.course_product import dispatch_request
from tiny_perceptron.decision_product.service import DeviceSession, Prediction
from tiny_perceptron.lesson_products.inference import parse_state, rule_depth
from tiny_perceptron.omni_products.choice import Candidate, Question
from tiny_perceptron.omni_products.inference import CJudge, OmniAssistant
from tiny_perceptron.omni_products.routing import Request, auto_panel, b_families, b_sum_adapter, routing_question


ROOT = Path('/tmp/tpb').resolve()
ART = ROOT / 'docs/technical-reviews/artifacts/c-v3-v21-canonical-19-24'
ARCHIVE = Path('/workspace/work/c-product-v3-20261101/outputs/c3-development/delivery/tiny-perceptron-c-v3-verified.zip')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def rejects(call):
    try:
        call()
    except (ValueError, TypeError, KeyError) as error:
        return type(error).__name__
    raise AssertionError('expected input rejection')


started = datetime.now(UTC).isoformat()
text = (ROOT / 'course/chapters/19.md').read_text()
body = re.search(r'^## 19\.24\s.*?(?=^## |\Z)', text, re.M | re.S).group(0)
snapshot = ROOT / 'docs/course-revision-20261101-c-v3/source-freeze-v19/sources/19.24.md'
assert body.encode() == snapshot.read_bytes()
assert digest(body.encode()) == '3118c0054adf229741788ab742f8a55e1dd67a6149570e3c489f9e21605d0bbf'
blocks = re.findall(r'```python\n(.*?)```', body, re.S)
assert len(blocks) == 2
namespace = {}
demo_stdout = []
for block in blocks:
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(block, 'course/chapters/19.md#19.24', 'exec'), namespace)
    demo_stdout.append(output.getvalue())
assert (namespace['gold_ok'], namespace['route_ok'], namespace['whole_ok']) == (3, 2, 1)
legal = namespace['select_and_dispatch']({'answer': 'B'}, {'numbers': [1, 2, 3]}, namespace['demo_dispatch'])
assert legal == {'route': 'B', 'result': {'backend_input': '三數加法;x=1;y=2;z=3', 'loops': 1}, 'contract_error': False}
edge_inputs = [[], [1, 2], [1, 2, 7], [-1, 2, 3], [True, 2, 3], [1.0, 2, 3], (1, 2, 3)]
edge_rejections = [{'input_repr': repr(values), 'exception': rejects(lambda values=values: namespace['adapt_sum'](values))} for values in edge_inputs]
auto_none = namespace['select_and_dispatch']({'answer': 'none'}, {'numbers': [1, 2, 3]}, namespace['demo_dispatch'])
manual_b = namespace['select_and_dispatch']({'answer': 'none'}, {'numbers': [1, 2, 3]}, namespace['demo_dispatch'], manual='B')
assert auto_none['route'] == 'none' and auto_none['result'] == {'status': 'unsupported'}
assert manual_b == legal
invalid = namespace['select_and_dispatch']({'answer': 'none'}, {'numbers': [1, 2, 7]}, namespace['demo_dispatch'], manual='B')
assert invalid['route'] == 'B' and invalid['contract_error'] and invalid['result'] is None

for values in ([0, 1, 1], [6, 6, 6], [1, 2, 3]):
    assert parse_state(b_sum_adapter(values)) == {'task': 'sum'}
assert rejects(lambda: b_sum_adapter([False, 2, 3])) == 'ValueError'
assert rejects(lambda: parse_state('三數加法;x=1;y=2;z=7')) == 'ValueError'
families = list(itertools.combinations_with_replacement(range(7), 3))
ordered = list(itertools.product(range(7), repeat=3))
assert len(families) == 84 and len(ordered) == 343
assert {tuple(sorted(v)) for v in ordered} == set(families)
panel = auto_panel()
panel_counts = {route: sum(row['gold_route'] == route for row in panel) for route in ('A', 'B', 'C', 'none')}
assert panel_counts == {'A': 20, 'B': 30, 'C': 30, 'none': 20}
assert len({row['family'] for row in panel}) == 100
assert all(tuple(v) == tuple(sorted(v)) for v in b_families()['retention'])


class OldBFixture:
    # 這是公開契約的控制流程 fixture，絕非已釘 manifest 或神經模型測量。
    auto_policy = {'version': 1, 'sum_depth': 1}

    def solve(self, state, loops):
        parse_state(state)
        return {'fixture_no_model_inference': True, 'state': state, 'loops': loops}


assistant = OmniAssistant.__new__(OmniAssistant)
assistant.old_b = OldBFixture()
bound_b = assistant.dispatch('B', {'numbers': [1, 2, 3]})
assert bound_b['loops'] == rule_depth(assistant.old_b.auto_policy, 'sum') == 1
assert rejects(lambda: assistant.dispatch('B', {'numbers': [1, 2, 7]})) == 'ValueError'


class ChoiceFixture(torch.nn.Module):
    # 只用來驗證入口和採用規則；答案與 logits 均由人工固定。
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.golds_seen = []
        self.info_answer = 'yes'

    def predict(self, question, temperature):
        self.golds_seen.append(question.gold)
        ids = [c.id for c in question.candidates] + ([] if question.kind == 'info' else ['none'])
        desired = self.info_answer if question.kind == 'info' else 'B' if 'A' in ids and 'B' in ids else 'record'
        logits = [2.0 if key == desired else 0.0 for key in ids]
        return {'answer': desired, 'ids': ids, 'logits': logits}


fixture = ChoiceFixture().eval()
judge = CJudge(fixture, temperature=1.0, threshold=0.99)
request = Request('三數加總', video_present=True, audio_present=True)
route = judge.route(request)
assert route['answer'] == 'B' and route['p_max'] < 0.99
judge.predict(routing_question(request, gold='A'))
state = 'A: task=record; permission=yes; ready=yes.'
choice = Question(state, 'choice', "Choose A's action. Require permission and ready known; no permission: stop; not ready: wait; else requested task.", (Candidate('record', 'Device A: record'), Candidate('stop', 'Device A: stop')), None).validate()
info = Question(state, 'info', "Are permission and ready both known for A's action?", (Candidate('yes', 'Required information is sufficient'), Candidate('no', 'Required information is missing')), None).validate()
judge.threshold = 0.6
accepted = judge.judge(choice, info)
assert accepted['accepted'] is True
fixture.info_answer = 'no'
not_enough = judge.judge(choice, info)
assert not_enough['accepted'] is False
assert judge.judge(choice)['decision'] == 'handoff'
assert all(value is None for value in fixture.golds_seen)


class TypedFixture:
    def __init__(self):
        self.examples = []

    def session(self, devices, language):
        return DeviceSession(self, devices, language)

    def _predict(self, example, use_cache=True):
        self.examples.append({'state': example.state, 'question': example.question, 'kind': example.kind})
        return Prediction(example.kind, 'fixture', None, 'fixture_no_model_inference', 0.0, '人工控制流程核對')


typed = TypedFixture()
typed_request = json.loads((ROOT / 'examples/decision_product/record-ready.json').read_text())
typed_result = dispatch_request(typed, typed_request)
assert typed_result['product'] == 'C-v3' and typed_result['dispatch'] == 'typed-devices'
assert set(typed_result) == {'product', 'dispatch', 'target', 'results'}
typed_rejections = [rejects(lambda incoming=incoming: dispatch_request(typed, incoming)) for incoming in ({'text': 'add three numbers'}, [], None, {**typed_request, 'gold': 'record'})]
assert typed_rejections == ['ValueError'] * 4

descriptor = json.loads((ROOT / 'docs/decision-product/v3/public-release.json').read_text())
archive_digest = file_digest(ARCHIVE)
assert ARCHIVE.stat().st_size == descriptor['bytes'] == 6168350
assert archive_digest == descriptor['sha256'] == '552d64b788291ce19157680f51dfc6bbc5394fb6bed7a22fcaa11f2ad3615d88'
implementation = {path: file_digest(ROOT / path) for path in descriptor['implementation']}
assert implementation == descriptor['implementation'] and len(implementation) == 17
acquisition = json.loads((ROOT / 'docs/decision-product/v3/metadata/acquisition.json').read_text())
member_checks = []
with zipfile.ZipFile(ARCHIVE) as source:
    for item in acquisition['files']:
        member = source.read(item['archive_member'])
        local = (ROOT / item['path']).read_bytes()
        assert member == local and digest(member) == item['sha256'] and len(member) == item['bytes']
        member_checks.append({'archive_member': item['archive_member'], 'path': item['path'], 'bytes': len(member), 'sha256': digest(member)})
    for name, expected in descriptor['bundle'].items():
        assert digest(source.read('tiny-perceptron-c-v3/bundle/' + name)) == expected
    manifest = json.loads(source.read('tiny-perceptron-c-v3/bundle/bundle.json'))
assert manifest['config']['indexed'] is True

# 作者摘要只做算術一致性核對；不把摘要或本 probe 當成重新執行的歷史量測。
summary = json.loads((ROOT / 'docs/omni-products/v1/results-summary.json').read_text())
auto = summary['integration']['auto']
assert [sum(row[key] for row in auto['by_gold_route'].values()) for key in ('families', 'route_correct', 'gold_route_end_to_end_correct', 'real_route_end_to_end_correct')] == [100, 89, 66, 55]
a_tasks = summary['integration']['old_A_retention']['tasks']
assert sum(row['families'] for row in a_tasks.values()) == 258
assert all(row['old_correct'] == row['wrapper_correct'] and row['difference_percentage_points'] == 0 for row in a_tasks.values())

result = {
    'kind': 'source-first bounded software/numeric probe; artificial fixtures explicitly separated from neural evidence',
    'started_at': started,
    'completed_at': datetime.now(UTC).isoformat(),
    'environment': {'python': sys.version, 'torch': str(torch.__version__), 'device': 'cpu; no trained weights loaded', 'platform': platform.platform(), 'root': str(ROOT)},
    'source': {'path': 'course/chapters/19.md#19.24', 'sha256': digest(body.encode()), 'snapshot_identical': True, 'figures': {}},
    'textbook_code': {'blocks': len(blocks), 'stdout': demo_stdout, 'legal_B_plan': legal, 'edge_rejections': edge_rejections, 'auto_none': auto_none, 'manual_B': manual_b, 'manual_invalid_B': invalid},
    'B_contract': {'family_formula': 'C(7+3-1,3)=C(9,3)=84', 'families': len(families), 'ordered': len(ordered), 'distinct_sorted_parents': len({tuple(sorted(v)) for v in ordered}), 'wrapper_fixture_result': bound_b, 'scope': 'grammar and wrapper control flow only; no historical depth manifest or B prediction reproduced'},
    'C_v2_control_fixture': {'route_below_service_threshold_still_B': route, 'choice_accept': accepted['accepted'], 'choice_info_no_handoff': not_enough['decision'], 'all_gold_inputs_stripped': all(value is None for value in fixture.golds_seen), 'Auto100_generated_counts': panel_counts, 'scope': 'artificial model outputs; no C-v2 accuracy or risk measurement'},
    'C_v3_typed_fixture': {'result': typed_result, 'examples_forwarded': typed.examples, 'rejections': typed_rejections, 'scope': 'typed format dispatch only; fixture does not establish neural quality, current model scores, or anonymous download'},
    'retained_product_identity': {'archive_path': str(ARCHIVE), 'bytes': ARCHIVE.stat().st_size, 'sha256': archive_digest, 'implementation_files': implementation, 'metadata_member_checks': member_checks, 'bundle_config': manifest['config'], 'public_transport_attempted': False, 'weights_loaded': False, 'scope': 'in-place hash/ZIP checks; no archive or weight copy; identity only, qualification evidence retention awaits coordinator'},
    'author_summary_arithmetic_only': {'totals': [100, 89, 66, 55], 'A_observations_from_row_denominators': 258, 'all_task_drop_pp': 0, 'scope': 'not a read of the immutable original prediction rows and not a new measurement; historical empirical claims remain unresolved pending retained evidence'},
    'assertions_passed': True,
}
with (ART / 'source-first-execution.json').open('x', encoding='utf-8') as handle:
    handle.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'assertions_passed': True, 'output': str(ART / 'source-first-execution.json'), 'sha256': file_digest(ART / 'source-first-execution.json'), 'scope': result['kind']}, ensure_ascii=False))
