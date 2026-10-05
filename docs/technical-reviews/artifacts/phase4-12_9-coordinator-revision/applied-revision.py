import datetime
import hashlib
import json
import re
import shutil
from pathlib import Path

root = Path('/workspace/tiny-perceptron-vlm')
sha = lambda data: hashlib.sha256(data).hexdigest()
report = root / 'docs/technical-reviews/12.9.json'
expected_report = '61c337daa4b6362ed7757481d7decac74fe625860af443c5931674650a77fa38'
assert sha(report.read_bytes()) == expected_report
history = root / f'docs/technical-reviews/history/phase4-12_9-own-initial-revise-{expected_report}.json'
assert not history.exists()
shutil.copyfile(report, history)
source = root / 'course/chapters/12.md'
before = source.read_bytes()
old = '既有單音訓練另測 14 題，生成答案對 11/14，錯誤集中在邊界與時長變化；完整配方放補充，不與這段梯度示範混稱成果。'.encode()
new = '既有單音訓練另測 14 題，生成答案對 11/14；三個錯例的頻率都是 290 或 300 Hz。資料把振幅與時長一起改動，無法分辨兩者各自的影響。完整配方放補充，不與這段梯度示範混稱成果。'.encode()
assert before.count(old) == 1
after = before.replace(old, new)

def sections(raw):
    headings = list(re.finditer(rb'(?m)^## ([A-Z\d]+\.\d+) [^\r\n]+', raw))
    return {h[1].decode(): raw[h.start():headings[i + 1].start() if i + 1 < len(headings) else len(raw)]
            for i, h in enumerate(headings)}

a, b = sections(before), sections(after)
assert sha(a['12.9']) == '409619baa10be2b64f80d549d541280646be65ae3ffa8ff7f97fba539d48d378'
changed = [key for key in a if a[key] != b[key]]
assert changed == ['12.9']
source.write_bytes(after)
now = datetime.datetime.now(datetime.UTC).isoformat()
receipt = {
    'recorded_at': now,
    'scope': 'Coordinator manuscript edit after original reviewer formal duration-scope revise; this is change provenance, not independent scientific verification.',
    'source': 'course/chapters/12.md#12.9',
    'before_source_sha256': sha(a['12.9']),
    'after_source_sha256': sha(b['12.9']),
    'before_complete_chapter_sha256': sha(before),
    'after_complete_chapter_sha256': sha(after),
    'changed_sections': changed,
    'unchanged_sections': [key for key in a if key not in changed],
    'history_path': history.relative_to(root).as_posix(),
    'history_sha256': sha(history.read_bytes()),
    'replacement_old': old.decode(),
    'replacement_new': new.decode(),
    'figures_code_data_unchanged': True,
}
output = Path(__file__).with_name('revision.stdout.json')
output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
progress_path = root / 'docs/course-revision-20261005/factual-review-progress.json'
progress = json.loads(progress_path.read_bytes())
record = progress['records']['12.9']
record['events'].append({
    'at': now, 'event': 'single_writer_source_revision_after_own_initial_revise',
    'source_before_sha256': sha(a['12.9']), 'source_after_sha256': sha(b['12.9']),
    'history_path': receipt['history_path'], 'history_sha256': receipt['history_sha256'],
    'change_artifact_path': Path(__file__).relative_to(root).as_posix(),
    'change_artifact_sha256': sha(Path(__file__).read_bytes()),
    'change_stdout_path': output.relative_to(root).as_posix(),
    'change_stdout_sha256': sha(output.read_bytes()),
    'note': 'Only the unsupported separate duration attribution is rewritten; raw observed count/frequency cases remain. Original owner and original reader callbacks are necessary and not yet dispatched.',
})
record['report_version_current'] = False
record['status'] = 'recheck_queued'
for key in ('12.8', '12.10'):
    affected = progress['records'][key]
    affected['status'] = 'recheck_queued'
    affected['events'].append({'at': now, 'event': 'context_recheck_queued_not_dispatched',
        'changed_context': 'course/chapters/12.md#12.9',
        'old_context_sha256': sha(a['12.9']), 'new_context_sha256': sha(b['12.9']),
        'note': 'Original owner personally read this context; ask original owner to assess its own support against revised context, preserving own prior report. No callback dispatch has occurred yet.'})
progress_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(receipt, ensure_ascii=False))
