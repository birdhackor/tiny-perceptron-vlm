"""Bounded identity/native/source-retention verification after genuine intake."""

import hashlib
import importlib.util
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path('/tmp/tpb').resolve()
ART = 'docs/technical-reviews/artifacts/c-v3-v21-canonical-glossary/'
ROUND = 'docs/course-revision-20261101-c-v3/'
BUNDLE = {'path': 'docs/technical-reviews/artifacts/c-v3-v21-numbered/numbered-section-reviews-source22-final-v1.json',
          'sha256': 'd103c2981757718d0eb2869142ce415984ab754aa7edc59697373a6b8ca5b1fb'}
RAW = {'path': 'docs/technical-reviews/artifacts/omni-p6-wave04-glossary/inspection-v1.json',
       'sha256': '506e428f125f98b184bacfa76223e2dabc478aa86c75fe67fd2b13e87de96407'}
ORIGINAL = {'path': ROUND + 'technical-v21/canonical_glossary/original-v1.json',
            'sha256': '60ff0ebdea32be3a94018117ccba586ac67dbdc17ff6e07b658f55c5827f27b9'}

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(ref):
    path = ROOT / ref['path']
    actual = digest(path)
    if actual != ref['sha256']:
        raise ValueError('hash mismatch: ' + ref['path'])
    return json.loads(path.read_text())

def module(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

setup = module('glossary_retention_setup', 'scripts/omni_revision/review_setup.py')
validator = module('glossary_retention_validator', 'scripts/check_technical_reviews.py')
api = module('glossary_retention_api', 'docs/review-tools/incremental_revision.py')
reading = module('glossary_retention_reading', 'scripts/reading_time.py')
setup.ROOT = ROOT
raw = read(RAW)
original = read(ORIGINAL)
bundle = read(BUNDLE)
canonical = read(bundle['canonical_source_freeze'])
section_freeze = read(bundle['section_freeze'])
assert bundle['canonical_source_freeze']['sha256'] == '0f4e9d63bfcb2243234c17c62242ff79cd4ab640a21d1f313f94c99946a93a59'
assert bundle['section_freeze']['sha256'] == '3f2671acc39c31215473924da29987c80ac444565184634dda9c8e033fd291c6'
page = next(x for x in canonical['pages'] if x['page_id'] == 'glossary')
body = setup.frozen_body(page)
assert hashlib.sha256(body.encode()).hexdigest() == original['source_sha256'] == raw['source']['sha256']
assert digest(ROOT / 'course/glossary.md') == original['source_sha256']
assert len(body.splitlines()) == 269 and not page['figures_sha256']

result = {'created_at': datetime.now(UTC).isoformat(), 'reviewer_task': original['reviewer_task'],
          'scope': 'Identity checks and native validation of exact26, plus retention of immutable raw primary/CPU evidence. No fresh mature-method replication, training, final, model forward, performance or public-download claim.',
          'original_report': ORIGINAL, 'source_sha256': original['source_sha256'],
          'raw_primary_receipt': RAW, 'retained_actor': raw['reviewer_task'], 'retained_created_at': raw['created_at'],
          'retained_execution_command': raw['execution']['command'],
          'retained_execution_environment': raw['execution']['output']['environment'],
          'source_freeze': bundle['canonical_source_freeze'], 'section_freeze': bundle['section_freeze'],
          'bundle': BUNDLE,
          'environment': {'python': platform.python_version(), 'device': 'CPU',
                          'operation': 'local JSON/hash/AST-free native review validation only'}}
dependencies = []
for rec in raw['original_primary_source_receipts']:
    p = Path(rec['path'])
    if not p.is_absolute():
        p = ROOT / p
    dependencies.append({'source_id': rec['source_id'], 'path': str(p),
                         'recorded_sha256': rec['sha256'], 'actual_sha256': digest(p),
                         'identical': digest(p) == rec['sha256'], 'locator': rec['inspected'],
                         'retention': 'Primary read retained from /root/omni_p6_technical_glossary on 2026-10-30; current actor checked immutable bytes and stated scope, not a new full paper read.'})
    if rec.get('original_file'):
        f = rec['original_file'];p = Path(f['path'])
        dependencies.append({'source_id': rec['source_id'] + '/original_file', 'path': str(p),
                             'recorded_sha256': f['sha256'], 'actual_sha256': digest(p),
                             'identical': digest(p) == f['sha256']})
for rec in raw['upstream_current_installation_receipts']:
    p = Path(rec['path'])
    dependencies.append({'source_id': 'installed/' + rec['symbol'], 'path': str(p),
                         'recorded_sha256': rec['file_sha256'], 'actual_sha256': digest(p),
                         'identical': digest(p) == rec['file_sha256'],
                         'locator': 'line ' + str(rec['start_line'])})
for rec in raw['public_primary_metadata']:
    p = Path(rec['path'])
    dependencies.append({'source_id': 'public/' + rec['name'], 'path': str(p),
                         'recorded_sha256': rec['sha256'], 'actual_sha256': digest(p),
                         'identical': digest(p) == rec['sha256'], 'selected': rec['selected'],
                         'retention': '2026-10-30 fixed published version metadata only; no new anonymous download or model-quality run.'})
result['retained_dependencies'] = dependencies
assert all(x['identical'] for x in dependencies)

rows = bundle['sections']
assert len(rows) == 26 and len({x['lesson_id'] for x in rows}) == 26
reports = {x['lesson_id']: read(x['report']) for x in rows}
owners = {r['reviewer_task']: [key] for key, r in reports.items()}
assert len(owners) == 26
validations = []
for row in rows:
    lesson = row['lesson_id']
    current = reports[lesson]
    orig = read(row['original_report'])
    origfreeze = read(row['original_section_freeze'])
    oldpage = next(x for x in origfreeze['pages'] if x['page_id'] == lesson)
    newpage = next(x for x in section_freeze['pages'] if x['page_id'] == lesson)
    parent = oldpage['canonical_page_id']
    setup.check_original(orig, row['original_section_freeze'], lesson, parent,
                         orig['reviewer_task'], validator, api,
                         original_source_snapshots=row.get('original_source_snapshots', []))
    assert orig['reviewer_task'] == current['reviewer_task']
    errors = validator._validate(ROOT, ROOT / newpage['source'], lesson,
                                 setup.frozen_body(newpage), current, owners, set())
    assert not errors, (lesson, errors)
    assert current['source_sha256'] == newpage['source_sha256']
    assert current['figure_sha256'] == newpage['figures_sha256']
    if row['report'] != row['original_report']:
        resolutions = current.get('original_issue_resolutions', [])
        assert all(any(x.get('json_pointer') == '/issues/' + str(i) and x.get('resolution')
                       for x in resolutions) for i in range(len(orig.get('issues', [])))), lesson
    dep = []
    for src in current['sources']:
        if src['kind'] == 'repository_code':
            p = ROOT / src['path'];dep.append({'kind': 'repository_code', 'path': src['path'],
                 'recorded_sha256': src['sha256'], 'actual_sha256': digest(p)})
    for artifact in current['artifacts']:
        p = ROOT / artifact['path'];dep.append({'kind': 'artifact', 'path': artifact['path'],
              'recorded_sha256': artifact['sha256'], 'actual_sha256': digest(p)})
    assert all(x['recorded_sha256'] == x['actual_sha256'] for x in dep)
    receipt_refs = []
    for key in ['original_intake_receipt', 'root_identity_receipt']:
        if isinstance(current.get(key), dict) and 'path' in current[key]:
            value = read(current[key]);receipt_refs.append({'field': key, 'ref': current[key],
                                                           'record_created_at': value.get('created_at')})
    validations.append({'lesson_id': lesson, 'parent_id': parent,
                         'report': row['report'], 'original_report': row['original_report'],
                         'owner': current['reviewer_task'], 'same_owner': True,
                         'source_identity': True, 'original_native': 'passed', 'current_native_errors': errors,
                         'original_created_at': orig['created_at'], 'current_created_at': current['created_at'],
                         'original_collection': row['original_collection'], 'current_collection': row['current_collection'],
                         'current_dispatch_window': row.get('current_dispatch_window'),
                         'retained_receipt_refs': receipt_refs,
                         'scope': 'Retained prior completed review with original receipt convention; not restamped as fresh T7.' if lesson not in {'R.2','R.3','R.4','T.11','W.7'} else 'Same-owner actual T7 callback; prior originals/currents preserved.',
                         'dependencies': dep})

result['section_validation'] = validations
result['counts'] = {'sections': 26, 'retained_unchanged': 21, 'w7_source22_callback': 1,
                    'fresh_parts': 4, 'original_native_pass': 26, 'current_native_pass': 26,
                    'dependencies_identical': sum(len(x['dependencies']) for x in validations),
                    'retained_raw_dependency_count': len(dependencies)}
retention_meta = read(bundle['retention_metadata'])
assert retention_meta['retained_count'] == 21
for row in retention_meta['sections']:
    if row['lesson_id'] != 'W.7':
        selected = next(x for x in rows if x['lesson_id'] == row['lesson_id'])
        assert selected['report'] == row['entry']['report']
        assert selected['original_report'] == row['entry']['original_report']
result['retention_metadata'] = bundle['retention_metadata']
result['glossary_comparison'] = []
slices = reading.lesson_slices(body)
for lesson in ['G.1','G.2','G.3','G.4']:
    row = next(x for x in rows if x['lesson_id'] == lesson)
    page = next(x for x in section_freeze['pages'] if x['page_id'] == lesson)
    assert slices[lesson] == setup.frozen_body(page)
    result['glossary_comparison'].append({'lesson_id': lesson, 'report': row['report'],
         'source_sha256': page['source_sha256'], 'same_exact_slice': True,
         'scope_comparison': {'G.1': 'ID/vector, Unicode/byte, role/BOS/EOS, tensor/Linear/activation及JSON契約與本人g013–g028和p009一致；單句EOS不外推batch。',
                              'G.2': 'loss有效目標分母、梯度/Adam、split/heldout、生成/SFT/DPO/LoRA及文字MTP與本人g040–g088、n036/n044及p059/p061一致。',
                              'G.3': 'QKV/causal與loss mask、norm/FFN/encoder、影音支持區間與訊號重建界線與本人g098–g129及p092/p094/p113/p131一致。',
                              'G.4': '儲存/計算、loop/cache、位置/量化/VQ、RL/reward、oracle/分母、機率校準與成熟外部模型角色及索引，與本人g139–g219、p170/p189/n204一致；不以索引核整章品質。'}[lesson]})

w7 = reports['W.7']
assert w7['format_token_clarification']['AST_equivalent_format_only'] is True
assert w7['format_token_clarification']['strict_whitespace_only'] is False
result['w7_format_scope'] = w7['format_token_clarification']
result['bundle_has_no_collector_backreference'] = 'technical-v21/canonical_' not in (ROOT / BUNDLE['path']).read_text()
assert result['bundle_has_no_collector_backreference']
result['not_global_seal'] = True
result['result'] = 'All named identities/native checks passed; old evidence retained with actor/date and scope. Exact source22 final26 bound without a collector hash cycle.'
destination = ROOT / ART / 'retention-proof-output.json'
with destination.open('x') as f:
    f.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'path': str(destination.relative_to(ROOT)), 'sha256': digest(destination),
                  'counts': result['counts'], 'result': result['result']}, ensure_ascii=False))
