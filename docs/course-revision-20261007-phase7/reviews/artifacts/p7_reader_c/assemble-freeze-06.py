import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/workspace/selftrained-v2')
PREFIX = 'docs/course-revision-20261007-phase7/reviews/'
old_path = PREFIX + 'reports/reader-c-freeze-04-callback.json'
manifest_path = PREFIX + 'freeze-06/manifest.json'
old_manifest_path = PREFIX + 'freeze-04/manifest.json'
trace_path = PREFIX + 'freeze-06/traces/reader/c/7634834a05eb413d922cbdf87842c9c5.jsonl'
new_path = PREFIX + 'reports/reader-c-freeze-06-callback.json'
audit_path = PREFIX + 'artifacts/p7_reader_c/freeze-06-same-bytes-audit.json'

def read(path):
    return json.loads((ROOT / path).read_text())

def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()

def create(path, value):
    with (ROOT / path).open('x') as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

old = read(old_path)
current_manifest = read(manifest_path)
old_manifest = read(old_manifest_path)
assert sha(manifest_path) == '0bbf4113fbd8288786bca540330fa9e2109e4fe848304a43b0321efa7bd23709'
assert sha(old_path) == 'da7742c1d67720689abfbc3fecc00ba4a368ef072df7402949db9018556cbc28'
current = {p['page_id']: p for p in current_manifest['pages']}
previous = {p['page_id']: p for p in old_manifest['pages']}
ids = next(g['primary_page_ids'] for g in current_manifest['groups'] if g['group'] == 'c')
assert len(ids) == 48
old_rows = {p['page_id']: p for p in old['pages']}
new_authored = read(PREFIX + 'artifacts/p7_reader_c/freeze-06-pages.json')
new_rows = {p['page_id']: p for p in new_authored['pages']}
changed = ['11.1', '11.13', '12.11']
assert set(new_rows) == set(changed)
traces = {}
for item in old['trace_files']:
    assert sha(item['path']) == item['sha256']
    traces[item['path']] = [json.loads(line) for line in (ROOT / item['path']).read_text().splitlines()]
traces[trace_path] = [json.loads(line) for line in (ROOT / trace_path).read_text().splitlines()]
assert traces[trace_path][-1]['event'] == 'complete'
assert traces[trace_path][-1]['event_index'] == 29
assert sum(e['event'] == 'checkpoint' for e in traces[trace_path]) == 28

checks = []
rows = []
for pid in ids:
    meta = current[pid]
    prior = previous[pid]
    same = all(meta.get(k) == prior.get(k) for k in ('source_sha256', 'figures_sha256', 'notebook_sha256'))
    assert same == (pid not in changed), pid
    if pid in changed:
        row = copy.deepcopy(new_rows[pid])
        row['previous_review'] = copy.deepcopy(old_rows[pid])
        row['page_visual_check']['required'] = True
        if meta.get('notebook_sha256'):
            row['notebook_sha256'] = meta['notebook_sha256']
    else:
        row = copy.deepcopy(old_rows[pid])
    assert row['source_sha256'] == meta['source_sha256']
    assert row['figures_sha256'] == meta['figures_sha256']
    assert row.get('notebook_sha256') == meta.get('notebook_sha256')
    evs = traces[row['trace_file']]
    page_evs = [e for e in evs if e.get('event') == 'checkpoint' and e.get('page_id') == pid]
    assert page_evs and all(e['source_sha256'] == meta['source_sha256'] for e in page_evs), pid
    for index in row['question_refs']:
        e = next(e for e in evs if e['event_index'] == index)
        assert e.get('event') == 'checkpoint' and e.get('page_id') == pid and e.get('four_questions'), (pid, index)
    checks.append({'page_id': pid, 'same_bytes_as_freeze_04': same,
                   'source_sha256': meta['source_sha256'], 'figures_sha256': meta['figures_sha256'],
                   'notebook_sha256': meta.get('notebook_sha256'), 'selected_trace': row['trace_file'],
                   'checkpoint_events': [e['event_index'] for e in page_evs],
                   'four_question_events': row['question_refs']})
    rows.append(row)
assert sum(c['same_bytes_as_freeze_04'] for c in checks) == 45

report = copy.deepcopy(old)
report['manifest_sha256'] = sha(manifest_path)
report['completed_at'] = datetime.now(timezone.utc).isoformat()
report['verdict'] = 'pass'
report['summary'] = '原owner freeze-06 callback，非新盲讀：本人逐段完整回讀11.1、11.13、12.11與工具指定9.8–9.10前文，28個checkpoint及頁末四題實記。11.1清零操作不會倒改舊difference；11.13的0.9^5標為獨立錯誤假設，CER補正式全名；12.11補ASR全名並明示手寫條件非識音執行。本次三頁皆pass，其餘45頁正文、圖與notebook逐項同bytes，沿用本人真讀。原3個burden及真正resolved回讀、所有舊trace/初判/限制與10.8 eval更正的真正callback原樣保留。'
report['pages'] = rows
report['trace_files'].append({'path': trace_path, 'sha256': sha(trace_path)})
report['reading_scope'].update({'callback_trace': trace_path, 'callback_primary_pages': 3, 'reuse_same_bytes_pages': 45})
report['reading_scope']['callback_traces'].append(trace_path)
report['reviewer_context_note'] = 'reviewer_context保留初輪真fresh起點的provenance；freeze-02、freeze-04與freeze-06均同owner callback，不是新盲讀者。'
callback = {'same_owner': True, 'fresh_reader': False, 'manifest': manifest_path,
            'manifest_sha256': sha(manifest_path),
            'recheck_of': PREFIX + 'freeze-01/traces/reader/c/1cc68419932142fdbcf2cdb0feb1937f.jsonl',
            'trace_file': trace_path, 'changed_primary_pages': changed, 'unchanged_primary_pages': 45,
            'new_context_pages': ['9.8', '9.9', '9.10'],
            'previous_report': {'path': old_path, 'sha256': sha(old_path), 'verdict': old['verdict']},
            'same_bytes_audit': {'path': audit_path}}
report['callback_history'].append(callback)
report['callback'] = copy.deepcopy(callback)
report['callback_context_visuals'].extend(new_authored['context_visuals'])
report['verification_scope']['required_figures_and_page_captures'] += '；freeze-06三個新改頁無必要SVG，新桌面/手機共14PNG本人真view後各存新receipt，11.1含2張末段定點補拍；必要前文9.10兩SVG640/360再本人真view，原所有證據保留。'
report['verification_scope']['unverified'].append('freeze-06只讀授權既存CPU：11.1清W後cell沒有stdout，未新執行插入清零後重算False；11.13空字串小變化只作定義推理，未新跑；12.11手寫low/high非ASR執行或音高測量，沒有親聽、重跑ASR或驗歷史聲音結果。')
report['verification_scope']['callback_output_observations'] += '；freeze-06三頁四個解鎖cell只讀授權既存CPU stdout/artifact，沒有新執行。'
audit = {'reviewer_task': '/root/p7_reader_c', 'manifest': manifest_path,
         'manifest_sha256': sha(manifest_path), 'previous_manifest': old_manifest_path,
         'previous_manifest_sha256': sha(old_manifest_path), 'previous_report': old_path,
         'previous_report_sha256': sha(old_path), 'new_trace_complete_event': 29,
         'primary_pages': 48, 'new_primary_pages': changed, 'same_bytes_pages': 45, 'checks': checks,
         'recorded_at': report['completed_at']}
create(audit_path, audit)
report['callback']['same_bytes_audit']['sha256'] = sha(audit_path)
report['callback_history'][-1]['same_bytes_audit']['sha256'] = sha(audit_path)
create(new_path, report)
print(json.dumps({'report': new_path, 'report_sha256': sha(new_path), 'audit': audit_path,
                  'audit_sha256': sha(audit_path), 'pages': len(rows), 'new_pages': changed,
                  'same_bytes_pages': 45, 'trace_files': len(report['trace_files']),
                  'new_trace_sha256': sha(trace_path), 'complete_event': 29,
                  'verdict': report['verdict']}, ensure_ascii=False))
