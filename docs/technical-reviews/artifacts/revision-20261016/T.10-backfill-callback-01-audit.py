"""Informed unchanged-section callback: identity/metadata only, no model execution."""

import ast
import copy
import hashlib
import inspect
import json
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.check_technical_reviews import _validate
from scripts.reading_time import lesson_slices


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_asts(body):
    return [ast.dump(ast.parse(code)) for code in re.findall(r'```python\n(.*?)```', body, flags=re.S)]


def main():
    base = ROOT / 'docs/technical-reviews/artifacts/revision-20261016'
    report_dir = ROOT / 'docs/course-revision-20261016/reviews'
    output = report_dir / 'technical-T.10-backfill-callback-01.json'
    audit_path = base / 'T.10-backfill-callback-01-audit.json'
    assert not output.exists() and not audit_path.exists(), 'Preserve existing callback records.'
    initial_path = report_dir / 'technical-T.10-original.json'
    prior_path = report_dir / 'technical-T.10-revise.json'
    initial = json.loads(initial_path.read_text())
    prior = json.loads(prior_path.read_text())
    initial_sha, prior_sha = sha(initial_path), sha(prior_path)
    assert prior['report_correction']['original_sha256'] == initial_sha
    old_validation = json.loads((base / 'T.10-validation.json').read_text())
    assert old_validation['report_sha256'] == prior_sha and old_validation['errors'] == []
    assert all(c['status'] == 'verified' for c in prior['claims'])
    r = copy.deepcopy(prior)
    freeze08 = json.loads((ROOT / 'docs/course-revision-20261016/freeze-08.json').read_text())
    freeze15_path = ROOT / 'docs/course-revision-20261016/freeze-15.json'
    freeze15 = json.loads(freeze15_path.read_text())
    training_source = ROOT / 'course/training.md'
    current_sections = lesson_slices(training_source.read_bytes().decode('utf-8'))
    p15 = next(p for p in freeze15['pages'] if p['page_id'] == 'training')
    frozen15_sections = lesson_slices((ROOT / p15['snapshot']).read_bytes().decode('utf-8'))
    p08 = next(p for p in freeze08['pages'] if p['page_id'] == 'training')
    frozen08_sections = lesson_slices((ROOT / p08['snapshot']).read_bytes().decode('utf-8'))
    body = current_sections['T.10']
    digest = hashlib.sha256(body.encode('utf-8')).hexdigest()
    assert body == frozen15_sections['T.10'] == frozen08_sections['T.10']
    assert digest == prior['source_sha256'] == initial['source_sha256']
    assert sha(training_source) == p15['source_sha256']
    assert not re.search(r'!\[|<img\b', body)
    source_checks, artifact_checks = [], []
    for s in prior['sources']:
        if s['kind'] == 'repository_code':
            actual = sha(ROOT / s['path'])
            source_checks.append({'id': s['id'], 'path': s['path'], 'prior_sha256': s['sha256'],
                                  'current_sha256': actual, 'unchanged': actual == s['sha256']})
            assert actual == s['sha256']
    for a in prior['artifacts']:
        actual = sha(ROOT / a['path'])
        artifact_checks.append({'id': a['id'], 'path': a['path'], 'prior_sha256': a['sha256'],
                                'current_sha256': actual, 'unchanged': actual == a['sha256']})
        assert actual == a['sha256']
    receipt = json.loads((base / 'T.10-source-receipts.json').read_text())
    external_identity = []
    for s in receipt['sources']:
        if 'original_file' in s:
            p = Path('/workspace/work/curriculum-20261016/fulltexts') / s['original_file']
            assert sha(p) == s['original_sha256']
            external_identity.append({'id': s['id'], 'path': str(p), 'sha256': sha(p), 'unchanged': True,
                                      'reading_scope': 'Original reading retained; full text not reread in callback.'})
    import torch
    from torch.nn import KLDivLoss
    from torch.nn import functional as F
    torch_receipt = next(s for s in receipt['sources'] if s['id'] == 'torch-kl')
    for obj, key in [(KLDivLoss, 'source_sha256'), (F.kl_div, 'functional_source_sha256')]:
        p = Path(inspect.getsourcefile(obj))
        assert sha(p) == torch_receipt[key]
        external_identity.append({'id': 'torch-' + obj.__name__, 'sha256': sha(p), 'unchanged': True,
                                  'reading_scope': 'Installed official source identity only; no tensor/model run.'})
    contexts = []
    for lesson in ['T.4', 'T.5', 'T.6', 'T.8']:
        a, b = frozen08_sections[lesson], current_sections[lesson]
        assert a == b
        contexts.append({'lesson_id': lesson, 'source': 'course/training.md',
                         'prior_section_sha256': hashlib.sha256(a.encode()).hexdigest(),
                         'current_section_sha256': hashlib.sha256(b.encode()).hexdigest(), 'unchanged': True,
                         'decision': 'Fixed recipe commands and anchors remain identical; no extra review scope.'})
    for lesson in ['14.13', '14.14', '14.15', '18.8', '18.9', '18.13', '12.12']:
        old = next(p for p in freeze08['pages'] if p['page_id'] == lesson)
        a = (ROOT / old['snapshot']).read_bytes().decode('utf-8')
        b = lesson_slices((ROOT / old['source']).read_bytes().decode('utf-8'))[lesson]
        row = {'lesson_id': lesson, 'source': old['source'],
               'prior_section_sha256': hashlib.sha256(a.encode()).hexdigest(),
               'current_section_sha256': hashlib.sha256(b.encode()).hexdigest(), 'unchanged': a == b}
        if lesson in ['14.13', '14.14']:
            assert code_asts(a) == code_asts(b)
            row['decision'] = 'Diff actually read: numeric-literal/spacing formatting only; Python AST unchanged; objectives unchanged.'
        elif lesson == '14.15':
            assert code_asts(a) == code_asts(b)
            row['decision'] = ('Diff actually read: adds two-box mechanism measurements/table/figure and separates deployment B; '
                               'T.10 claims only recipe distinction and work accounting. Current primary recipe selectors inspected below; '
                               'no new metric/figure/capability claim imported into T.10 review.')
        else:
            assert a == b
            row['decision'] = 'Necessary prior mechanism/context bytes unchanged.'
        contexts.append(row)
    result_path = ROOT / 'docs/lesson-products/results/final-report-v1.json'
    settings_path = ROOT / 'docs/lesson-products/results/settings-freeze-v1.json'
    final = json.loads(result_path.read_text())
    settings = json.loads(settings_path.read_text())
    assert final['settings_freeze_sha256'] == sha(settings_path)
    assert final['mechanisms']['fixed3']['terminal_supervision'] == [3]
    assert final['mechanisms']['odd']['terminal_supervision'] == [1, 3, 5]
    fixed, odd = settings['mechanisms']['fixed3'], settings['mechanisms']['odd']
    assert fixed['initial_weights_sha256'] == odd['initial_weights_sha256']
    assert fixed['last_step'] == odd['last_step'] == 1500
    assert settings['products']['b']['model.pt'] not in [fixed['last.pt'], odd['last.pt']]
    recipe_identity = {'result': {'path': str(result_path.relative_to(ROOT)), 'sha256': sha(result_path)},
                       'settings': {'path': str(settings_path.relative_to(ROOT)), 'sha256': sha(settings_path)},
                       'fixed3_terminal_supervision': [3], 'odd_terminal_supervision': [1, 3, 5],
                       'same_initialization': fixed['initial_weights_sha256'], 'actual_last_step': 1500,
                       'product_b_checkpoint_distinct': True,
                       'protocol': settings['protocol']['mechanism'],
                       'scope': 'Recipe identity fields only, no rerun and no verification of added empirical table or figure.'}
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    audit = {'reviewer_task': prior['reviewer_task'], 'current_context': 'informed',
             'recorded_at': datetime.now(UTC).isoformat(), 'current_revision': revision,
             'scope': 'Owner callback for T.10 only. File identities, prior report/receipt readback, changed necessary-context diff and selected primary recipe fields.',
             'prior_reports': [{'path': str(initial_path.relative_to(ROOT)), 'sha256': initial_sha},
                               {'path': str(prior_path.relative_to(ROOT)), 'sha256': prior_sha}],
             'freeze': {'path': str(freeze15_path.relative_to(ROOT)), 'sha256': sha(freeze15_path),
                        'training_snapshot': p15['snapshot'], 'exact_section_match': True},
             'section_sha256': digest, 'figures': {}, 'prior_source_checks': source_checks,
             'prior_artifact_checks': artifact_checks, 'primary_source_identity': external_identity,
             'necessary_context_checks': contexts, 'changed_context_primary_recipe_check': recipe_identity,
             'new_training_or_formal_test': False, 'old_contract_probe_rerun': False,
             'new_model_or_tensor_execution': False, 'full_page_or_other_section_review': False,
             'all_identity_and_recipe_assertions': 'passed'}
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    r['current_context'] = 'informed'
    r['reviewer_context_note'] = 'reviewer_context=fresh describes the initial independent task only. This owner callback is informed and is not a new blind/fresh review.'
    r['source_sha256'] = digest
    r['figure_sha256'] = {}
    r['callback'] = {'kind': 'backfill_unchanged_section', 'sequence': '01', 'prior_report': str(prior_path.relative_to(ROOT)),
                     'prior_report_sha256': prior_sha, 'current_freeze': 'docs/course-revision-20261016/freeze-15.json',
                     'current_context': 'informed', 'prior_reports_preserved': True,
                     'performed': audit['scope'],
                     'not_performed': 'No old fixture/KL/infer probe rerun, formal test, new model/tensor run, training, Modal/GPU/HF write, paid call or curriculum edit.'}
    r['review_scope'] = {'included': 'Exact T.10 UTF-8 bytes and its necessary dependencies only; bytes equal last owner report and freeze-15 training snapshot.',
                         'necessary_context': 'T.4/T.5/T.6/T.8; 18.8/18.9/18.13;12.12 unchanged. 14.13/14.14 formatting diffs inspected. 14.15 added backfill inspected only for recipe identity affecting T.10.',
                         'not_performed': r['callback']['not_performed']}
    for a in r['artifacts']:
        a['description'] = 'Preserved initial-review evidence; not rerun or rewritten in this informed callback. ' + a['description']
    env = {'python': platform.python_version(), 'torch': str(torch.__version__), 'device': 'metadata-only CPU process; no model execution'}
    r['artifacts'].extend([
        {'id': 'a4', 'kind': 'execution', 'path': str(audit_path.relative_to(ROOT)), 'sha256': sha(audit_path),
         'description': 'Current owner identity/diff/selected-primary-recipe audit only; no previous experiment rerun.',
         'command': '.venv/bin/python docs/technical-reviews/artifacts/revision-20261016/T.10-backfill-callback-01-audit.py',
         'result': 'All prior source/artifact identities and exact section checks passed; context differences assessed; mechanism terminal selectors match T.10.', 'environment': env},
        {'id': 'a5', 'kind': 'code', 'path': str(Path(__file__).resolve().relative_to(ROOT)), 'sha256': sha(Path(__file__).resolve()),
         'description': 'Reproducible metadata-only callback audit and full report construction; does not invoke training or formal tests.'}])
    for s in r['sources']:
        s['callback_confirmation'] = 'Owner checked retained source/evidence identity in a4; initial primary-source reading is retained, not claimed as newly reread.'
    for sid, p, title, note in [
        ('r-backfill-result', result_path, 'Current original looped-mechanism result recipe identity', 'Inspected only settings-freeze SHA and mechanisms.fixed3/odd terminal_supervision plus actual last-step metadata. No new score/figure review or formal test.'),
        ('r-backfill-settings', settings_path, 'Current pretest frozen mechanism and product checkpoint identity', 'Inspected mechanisms fixed3/odd initial_weights, last_step and checkpoint hashes, product b checkpoint hash and protocol.mechanism. Distinct product checkpoint; same mechanism init; no author selection plan read.')]:
        r['sources'].append({'id': sid, 'kind': 'repository_code', 'title': title, 'path': str(p.relative_to(ROOT)),
                             'sha256': sha(p), 'version': 'Current original primary JSON at HEAD ' + revision,
                             'verified': True, 'inspection_note': note})
    for c in r['claims']:
        c['artifact_ids'].append('a4')
        c['callback_confirmation'] = 'Current T.10 unchanged; original dependencies/artifacts byte-identical. Prior owner verification retained; no old execution repeated.'
        if 'verification' in c:
            c['verification']['execution_context'] = 'Executed during initial owner review; not rerun in this informed callback. Current identity audit is a4.'
            c['verification']['details'] = 'Initial-review operation retained (not newly executed here): ' + c['verification']['details']
        if c['id'] == 'c6':
            c['scope'] = c['scope'].replace('本輪只以未訓練 fixture 跑 CLI', '初審只以未訓練 fixture 跑 CLI，本次 callback 未重跑')
        if c['id'] == 'c12':
            c['scope'] = ('T.10 only distinguishes fixed3/odd135 mechanism objectives and separately stored deployment supervision. '
                          'Linked 14.15 now contains measured backfill; current original result/settings selectors confirm [3] versus [1,3,5] '
                          'and a separate B checkpoint. This callback imports no new metric, interpolation guarantee or whole-14.15 figure verdict into T.10.')
            c['callback_confirmation'] = 'Actually read the changed 14.15 diff and selected original result/settings fields; T.10 recipe/work distinction remains supported.'
            c['evidence'].extend([
                {'source_id': 'r-backfill-result', 'locator': 'settings_freeze_sha256; mechanisms.fixed3/odd.terminal_supervision', 'supports': 'New linked backfill preserves the two mechanism terminal-supervision sets described by T.10.'},
                {'source_id': 'r-backfill-settings', 'locator': 'mechanisms.*.initial_weights_sha256/last.pt; products.b.model.pt; protocol.mechanism', 'supports': 'Current original metadata keeps matched-initialization mechanism runs separate from the deployment checkpoint.'}])
    r['checks']['factual_accuracy']['details'] = 'Informed owner callback: exact T.10 bytes unchanged; all 12 initial claims remain supported by identical implementation/evidence. Changed necessary 14.15 backfill checked specifically for recipe distinction; not a whole-guide re-review.'
    r['checks']['numeric_verification']['details'] = 'Retain initial width32/KL/gradient and original-report denominator audit; a1 unchanged. These operations were not rerun. Current callback only checked evidence identities/selected recipe metadata, not added backfill quality metrics.'
    r['checks']['figure_consistency']['details'] = 'T.10 still contains no image/SVG reference. Linked 14.15 now has a measured plot, but T.10 makes no claim about its measured values; its visual/metric correctness is outside this callback and is not asserted.'
    r['checks']['source_verification']['details'] = 'All 15 original repository source hashes, three preserved artifacts and external original/installed-torch identities checked unchanged. Original primary readings retained. Actual context diff and current original recipe selectors inspected; no author plan or other review conclusions read.'
    r['checks']['limitations']['details'] = 'Initial evidence/platform/seed/denominator limits retained. Current callback is informed, metadata-only and T.10-specific. Linked 14.15 measured backfill is acknowledged without converting T.10 into new score, capability or figure claims; no old probe/formal test/training repeated.'
    errors = _validate(ROOT, training_source, 'T.10', body, r, {prior['reviewer_task']: ['T.10']}, set())
    assert errors == [], errors
    output.write_text(json.dumps(r, ensure_ascii=False, indent=2) + '\n')
    assert sha(initial_path) == initial_sha and sha(prior_path) == prior_sha
    validation_path = base / 'T.10-backfill-callback-01-validation.json'
    assert not validation_path.exists()
    validation_path.write_text(json.dumps({'report': str(output.relative_to(ROOT)), 'report_sha256': sha(output),
                                           'command': '_validate(ROOT, training_source, T.10, exact lesson_slices body, callback report, owner map, set())',
                                           'errors': errors, 'current_context': 'informed', 'prior_reports_preserved': True,
                                           'environment': env}, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'report': str(output.relative_to(ROOT)), 'verdict': r['verdict'], 'current_context': 'informed',
                      'source_sha256': digest, 'repository_sources_unchanged': len(source_checks),
                      'old_artifacts_unchanged': len(artifact_checks), 'validate_errors': errors}, ensure_ascii=False))


if __name__ == '__main__':
    main()
