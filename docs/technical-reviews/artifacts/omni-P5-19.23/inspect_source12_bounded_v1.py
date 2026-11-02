"""Same-owner bounded source12 callback; AST/text/provenance only, no torch/kernel."""

import ast
import difflib
import hashlib
import json
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
BASE = ROOT / 'docs/course-revision-20261029-omni/reviews/P5-actual'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


before = BASE / 'source-freeze-v6/sources/19.23.md'
after = BASE / 'source-freeze-v12/sources/19.23.md'
prior_report = BASE / 'technical-19.23/technical-current-v1.json'
freeze = BASE / 'source-freeze-v12.json'
repairs = BASE / 'coherence-three-author-repairs-source12-v1.json'
assert sha(prior_report) == '4531b5490f41be42c46fc70363b830aa59a466b037dd0c02ff3bf2f09dc47d32'
assert sha(before) == '18231f8b2f5882e09fd08bde6e58ddb0d8b914b0e6b8d3eff6c3be948626de60'
assert sha(after) == '7a85128160f6ee16e59a9fa900005f718efb4ae6e1ffd16f98dc7ad13ae3032f'
assert sha(freeze) == '1d95d71c09a20a07334cd30707d61bcfe7a6334f8bc12241b04ee2dc66aed4b1'
assert sha(repairs) == '76e29aef2e528bf06ae20eacfdf48f28327fa9432e6477d50996be44301a3ea7'
old, new = before.read_text(), after.read_text()
old_lines, new_lines = old.splitlines(keepends=True), new.splitlines(keepends=True)
changes = [(i + 1, a, b) for i, (a, b) in enumerate(zip(old_lines, new_lines)) if a != b]
assert len(old_lines) == len(new_lines) == 622
assert len(changes) == 1 and changes[0][0] == 7
assert '問題與各級描述合起來須給本題可核對的分級規準' in changes[0][2]
assert '每級描述仍須能辨認對應的等級' in changes[0][2]
assert '不能指「上一項」' in changes[0][2]
pattern = r'```python\n(.*?)\n```'
old_blocks = re.findall(pattern, old, re.S)
new_blocks = re.findall(pattern, new, re.S)
assert len(new_blocks) == 3 and old_blocks == new_blocks
assert [ast.dump(ast.parse(x), include_attributes=False) for x in old_blocks] == [
    ast.dump(ast.parse(x), include_attributes=False) for x in new_blocks
]
tree = ast.parse(new_blocks[0])
rubric = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Name) and target.id == 'rubric' for target in node.targets))
assert rubric == ('rubric', '評任務與申請編號兩欄的完整度。',
                  [('0', '兩欄都缺'), ('1', '只有一欄'), ('2', '兩欄都有')])
assert 'u = self.pool(question_ids + [SEP] + utf8(description, 128))' in new_blocks[0]
assert 'branches.append([CLS, *q, SEP, *text_bytes(description, 128, "description")])' in new_blocks[2]
formal_definitions = new_blocks[2].split('# 完整正式架構的隨機初始化核對')[0]
formal_tree = ast.parse((ROOT / 'tiny_perceptron/omni_products/choice.py').read_text())
formal_tree.body = formal_tree.body[1:]  # unchanged module docstring only
assert ast.dump(formal_tree, include_attributes=False) == ast.dump(ast.parse(formal_definitions), include_attributes=False)
inputs = json.loads((OUT / 'fixed-inputs.json').read_text())
formal_rubric = inputs['first_final_input']['questions']['rubric']
assert formal_rubric['text'] == 'Grade C: record=count yes fields; play=2 if permission else ready(0/1); save=2 if ready else permission(0/1).'
assert [(c['id'], c['description']) for c in formal_rubric['candidates']] == [
    (str(i), f'Device C has grade {i}') for i in range(3)
]
assert formal_rubric['text'] in new and formal_rubric['gold'] == '0'
prior = json.loads(prior_report.read_text())
repository_receipts = []
for source in prior['sources']:
    if source['kind'] == 'repository_code':
        path = ROOT / source['path']
        assert sha(path) == source['sha256']
        repository_receipts.append({'path': source['path'], 'sha256': source['sha256'], 'unchanged': True})
for artifact in prior['artifacts']:
    assert sha(ROOT / artifact['path']) == artifact['sha256']
page = next(row for row in json.loads(freeze.read_text())['pages'] if row['page_id'] == '19.23')
assert page['source_sha256'] == sha(after) and page['figures_sha256'] == {}
chapter = (ROOT / 'course/chapters/19.md').read_text()
start = re.search(r'^## 19\.23 ', chapter, re.M).start()
end = re.search(r'^## ', chapter[start + 1:], re.M)
current = chapter[start:start + 1 + end.start()] if end else chapter[start:]
assert current == new
proof = {
    'created_at': datetime.now(timezone.utc).isoformat(),
    'reviewer_task': '/root/omni_p5_technical_19_23',
    'mode': 'same-owner bounded source12 text/data-flow verification; prior whole review reused explicitly',
    'source_freeze': {'path': str(freeze.relative_to(ROOT)), 'sha256': sha(freeze)},
    'prior_current_report': {'path': str(prior_report.relative_to(ROOT)), 'sha256': sha(prior_report)},
    'author_diff_location_only': {'path': str(repairs.relative_to(ROOT)), 'sha256': sha(repairs),
                                  'author_interpretation_not_used_as_acceptance_evidence': True},
    'source_sha256': sha(after), 'only_changed_source_line': 7,
    'reviewer_computed_exact_diff': ''.join(difflib.unified_diff(old_lines, new_lines, fromfile='reviewed-source6', tofile='source12')),
    'all_three_python_blocks_exact_bytes_and_AST_unchanged': True,
    'python_block_sha256': [hashlib.sha256(x.encode()).hexdigest() for x in new_blocks],
    'formal_inline_AST_equals_unchanged_repository_code': True,
    'course_current_19_23_equals_source12_snapshot_bytes': True,
    'short_rubric_fixture': rubric,
    'short_branch_data_flow': 'pool(question_ids + SEP + description) before reading state M',
    'formal_branch_data_flow': '[CLS, question bytes, SEP, own description bytes] -> branch CLS -> two reads of M',
    'formal_first_family_rubric': formal_rubric,
    'unchanged_repository_inputs_from_prior_current': repository_receipts,
    'prior_current_all_artifact_hashes_unchanged': True,
    'manual_label_inspection': {
        'short': 'Task and application number are both supplied; the stated two-field completeness rule identifies grade 2. This is a label interpretation, not a model prediction.',
        'formal': 'C requests record with permission=no and ready=no; its question counts yes fields, giving 0. Each candidate description identifies its own grade.',
        'self_containment': 'Each branch receives the shared question and its own description; neither visible rubric needs a reference to another candidate.'
    },
    'necessary_dependency_check': {
        '9.17': 'Only the supplied raw-score-selection diff was checked for compatibility; it agrees with 19.23 raw-score selection and does not change the rubric rule/data flow.',
        '12.17': 'No required 19.23 dependency in the changed energy-scale explanation; no technical acceptance of 12.17 is claimed.'
    },
    'no_new_NN_kernel_training_calibration_or_paid_execution': True,
    'environment': {'python': sys.version.split()[0], 'platform': platform.platform(), 'execution': 'stdlib AST/text/hash/JSON only; torch not imported'},
}
(OUT / 'source12-bounded-output-v1.json').write_text(json.dumps(proof, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
print(json.dumps({'only_changed_line': 7, 'python_blocks_exactly_unchanged': 3,
                  'both_rubric_examples_match_joint_question_description_contract': True,
                  'prior_repository_and_artifact_hashes_unchanged': True,
                  'new_NN_or_kernel': False}, ensure_ascii=False))
