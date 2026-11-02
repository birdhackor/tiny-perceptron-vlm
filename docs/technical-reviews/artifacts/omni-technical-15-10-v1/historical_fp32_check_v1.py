"""只核15.10的既有FP32命名修正；不代替新的reader或發布閘門。"""

import contextlib
import datetime
import hashlib
import importlib.util
import io
import json
import platform
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    api = module('fp32_history_api', ROOT / 'docs/review-tools/incremental_revision.py')
    legacy = module('fp32_history_legacy', ROOT / 'docs/review-tools/historical_fix_rechecks.py')
    request_path = ROOT / 'docs/course-revision-20261029-omni/reviews/prior-fix-appendix-requests-v2.json'
    request = json.loads(request_path.read_bytes())
    group = request['groups'][3]
    assert (group['page_id'], group['old_owner'], group['old_role']) == (
        '15.10', '/root/rv1019_reader_ch15', 'reader')
    assert len(group['items']) == 1
    row = group['items'][0]
    original_path = ROOT / 'docs/course-revision-20261029-omni/reviews/technical/technical_15_10/technical-original-v1.json'
    assert sha(original_path) == 'e36cc3e84f278ab78fb3bb212af0b2e217c32ace0b317ba11bb7b22b44348472'
    original = json.loads(original_path.read_bytes())
    freeze_path = ROOT / 'docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix11.json'
    freeze = json.loads(freeze_path.read_bytes())
    page = next(p for p in freeze['pages'] if p['page_id'] == '15.10')
    body = api.artifact(ROOT, {'path': page['snapshot'], 'sha256': page['source_sha256']}).decode()
    current_body = re.search(r'^## 15\.10 .*?(?=^## |\Z)', (ROOT / page['source']).read_text(), re.M | re.S)[0]
    assert body == current_body
    assert page['source_sha256'] == original['source_sha256']
    assert page['figures_sha256'] == original['figure_sha256'] == {}

    # Real legacy API authenticates the actual prior generation through the live
    # active-pointer history, prior completed gate, freeze, dispositions and tasks.
    history = legacy.Rechecks.__new__(legacy.Rechecks)
    history.root = ROOT
    history.api = api
    history.prior(group['prior_active_pointer'])
    assert history.old_disposition_ref == {k: row['disposition_ref'][k] for k in ('path', 'sha256')}
    decision = api.node(history.old_dispositions, row['disposition_ref']['json_pointer'])
    assert decision['action'] == 'fixed'
    assert decision['final_necessity'] == row['necessity'] == 'optional'
    assert decision['issue_id'] == row['issue_id'] == 'READER-CH15-15.10-1'
    assert decision['report'] == row['original_report']
    assert decision['json_pointer'] == row['json_pointer']
    assert row['historical_fix_verification'] in decision['verification']
    assert decision['affected_pages'] == row['affected_pages'] == ['15.10']
    old_original = api.document(ROOT, row['original_report'])
    old_issue = api.node(old_original, row['json_pointer'])
    old_recheck = api.document(ROOT, row['historical_fix_verification']['report'])
    previous = api.node(old_recheck, row['historical_fix_verification']['json_pointer'])
    assert api.owner(old_original) == api.owner(old_recheck) == group['old_owner']
    assert history.old_tasks[group['old_owner']]['role'] == 'reader'
    old_page = history.old_pages['15.10']
    assert previous['source_sha256'] == old_page['source_sha256']
    assert previous['figures_sha256'] == old_page['figures_sha256'] == {}
    old_body = api.artifact(ROOT, {'path': old_page['snapshot'], 'sha256': old_page['source_sha256']}).decode()

    paragraph = next(p for p in body.split('\n\n') if p.startswith('程式只計數'))
    old_paragraph = next(p for p in old_body.split('\n\n') if p.startswith('程式只計數'))
    assert paragraph == old_paragraph
    quote = 'FP32（[5.9](05.md#5.9)的32位元浮點格式）每個權重用四bytes'
    assert quote in paragraph

    # Verify the actual recorded inability; do not repeat a failed followup or
    # invent a current manifest/author list or sealed collection for this test.
    unavailable_doc = api.document(ROOT, group['unavailability']['report'])
    unavailable = api.node(unavailable_doc, group['unavailability']['json_pointer'])
    assert unavailable['reviewer_task'] == group['old_owner']
    assert unavailable['unavailable'] is True
    assert unavailable_doc['author_task'] == request['author_task']
    assert history.prior_activated <= api.at(unavailable['recorded_at']) <= datetime.datetime.now(datetime.UTC)
    followups = [e for e in unavailable['events'] if e['tool'] == 'collaboration.followup_task'
                 and e['request']['target'] == group['old_owner']]
    assert len(followups) == 1
    assert group['old_owner'] in followups[0]['response'] and 'not found' in followups[0]['response']
    assert any(e['tool'] == 'collaboration.list_agents'
               and group['old_owner'].startswith(e['request']['path_prefix'])
               and e['response']['agents'] == [] for e in unavailable['events'])

    # Official source bytes were freshly read for this appendix. Reuse only the
    # existing immutable small acquisition receipt; keep third-party full text out.
    inspection_ref = next(a for a in original['artifacts'] if a['id'] == 'a_sources')
    inspection = api.document(ROOT, inspection_ref)
    selected = [s for s in inspection['original_sources'] if s['source_id'] in ('s_torch_dtype', 's_torch_tensor')]
    for source in selected:
        raw = Path(source['local_cache'])
        assert sha(raw) == source['raw_sha256']
    dtype = next(s for s in selected if s['source_id'] == 's_torch_dtype')
    dtype_raw = Path(dtype['local_cache']).read_text()
    assert '| `torch.float32` or `torch.float` | 32-bit floating point' in dtype_raw
    prereq_ref = next(a for a in original['artifacts'] if a['id'] == 'a_prereqs')
    prereqs = api.document(ROOT, prereq_ref)
    p59 = next(p for p in prereqs['slices'] if p['source'] == 'course/chapters/05.md#5.9')
    source59 = (ROOT / 'course/chapters/05.md').read_text()
    assert p59['exact_text'] in source59
    assert '一個 byte 有 8 個 bit' in p59['exact_text']
    assert 'FP32 是用 32 個 bit 表示一個浮點數的格式' in p59['exact_text']

    code = re.findall(r'```python\n(.*?)```', body, re.S)
    assert len(code) == 1
    outputs = []
    for k, expected in ((2, ['expert數 4 總權重 272 每token使用估計 144 FP32權重bytes 1088',
                            'expert數 8 總權重 544 每token使用估計 160 FP32權重bytes 2176']),
                        (1, ['expert數 4 總權重 272 每token使用估計 80 FP32權重bytes 1088',
                            'expert數 8 總權重 544 每token使用估計 96 FP32權重bytes 2176'])):
        source = code[0] if k == 2 else code[0].replace('width, hidden, k = 4, 8, 2', 'width, hidden, k = 4, 8, 1', 1)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(compile(source, '15.10-current-inline-historical-fp32', 'exec'), {})
        assert output.getvalue().splitlines() == expected
        outputs.append({'k': k, 'code_sha256': hashlib.sha256(source.encode()).hexdigest(),
                        'stdout': output.getvalue(), 'exact_expected_lines': expected, 'match': True})
    bytes_per_parameter = torch.empty(1, dtype=torch.float32).element_size()
    assert 32 // 8 == bytes_per_parameter == 4
    assert torch.empty(272, dtype=torch.float32).nbytes == 1088
    assert torch.empty(544, dtype=torch.float32).nbytes == 2176

    result = {
        'created_at': datetime.datetime.now(datetime.UTC).isoformat(),
        'reviewer_task': original['reviewer_task'],
        'command': 'uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-15-10-v1/historical_fp32_check_v1.py',
        'environment': {'python': platform.python_version(), 'torch': torch.__version__, 'device': 'cpu'},
        'request': {'path': str(request_path.relative_to(ROOT)), 'sha256': sha(request_path), 'json_pointer': '/groups/3'},
        'prior_generation': group['prior_generation'], 'prior_active_pointer': group['prior_active_pointer'],
        'legacy_prior_api': 'passed; actual pointer-history/prior-gate/freeze/disposition/task authentication',
        'original_and_current_report': {'path': str(original_path.relative_to(ROOT)), 'sha256': sha(original_path)},
        'current_freeze': {'path': str(freeze_path.relative_to(ROOT)), 'sha256': sha(freeze_path)},
        'current_source_sha256': page['source_sha256'], 'current_figure_sha256': {},
        'old_original_issue': {'report': row['original_report'], 'json_pointer': row['json_pointer'],
                               'quote': old_issue['quote'], 'problem': old_issue['problem'], 'necessary': old_issue['necessary']},
        'old_disposition_and_verification_identity': row,
        'prior_page': old_page,
        'current_source_quote': quote, 'prior_fixed_paragraph_matches_current_exactly': True,
        'unavailability': group['unavailability'],
        'unavailability_events_and_chronology': 'verified original-task not-found response and empty original-prefix listing; author matches request recorder',
        'official_source_receipt': {'path': inspection_ref['path'], 'sha256': inspection_ref['sha256']},
        'official_sources_reread': [{'source_id': s['source_id'], 'url': s['url'], 'raw_sha256': s['raw_sha256']} for s in selected],
        'prerequisite_receipt': {'path': prereq_ref['path'], 'sha256': prereq_ref['sha256']},
        'current_prerequisite_5_9_exact_text_matches': True,
        'inline_executions': outputs,
        'float32_bits': 32, 'bits_per_byte': 8, 'element_size_bytes': bytes_per_parameter,
        'float32_tensor_nbytes_for_272_and_544_elements': [1088, 2176],
        'result': 'still_fixed: FP32 explicitly named 32位元浮點格式, official dtype agrees, 32/8=4 and exact current inline outputs agree.',
        'limits': 'technical factual succession only; not the old reader, a new blind reader, or continuity. No new model training, GPU, speed, peak-memory or quality result. Final manifest same-wave fresh-reader evidence, appendix collection and seals are pending coordinator integration; no full historical gate passed.',
    }
    path = Path(__file__).with_name('historical-fp32-execution-v1.json')
    with path.open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({'path': str(path.relative_to(ROOT)), 'sha256': sha(path), 'result': result['result'],
                      'full_historical_gate': 'pending; not run'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
