import datetime
import hashlib
import json
import re
import shutil
from pathlib import Path

root = Path('/workspace/tiny-perceptron-vlm')
sha = lambda data: hashlib.sha256(data).hexdigest()
report = root / 'docs/technical-reviews/13.9.json'
expected = '53c9e7f4eb157a3cc43043b9b14aa739831c21c0b9c44decb9e21a8fc1d1b493'
assert sha(report.read_bytes()) == expected
history = root / f'docs/technical-reviews/history/phase4-13_9-own-initial-revise-{expected}.json'
assert not history.exists()
shutil.copyfile(report, history)
source = root / 'course/chapters/13.md'
before = source.read_bytes()
old = '本章實跑只更新加法正誤偏好與附加句偏好'.encode()
new = '本章的加法主線只更新加法正誤偏好與附加句偏好'.encode()
assert before.count(old) == 1
after = before.replace(old, new)

def slices(raw):
    headings = list(re.finditer(rb'(?m)^## ([A-Z\d]+\.\d+) [^\r\n]+', raw))
    return {h[1].decode(): raw[h.start():headings[i + 1].start() if i + 1 < len(headings) else len(raw)]
            for i, h in enumerate(headings)}

a, b = slices(before), slices(after)
assert sha(a['13.9']) == 'dbd937be75aec59989f1251a319120e6c68b64d01d510725c7eedc1d1a25ade5'
changed = [key for key in a if a[key] != b[key]]
assert changed == ['13.9']
source.write_bytes(after)
now = datetime.datetime.now(datetime.UTC).isoformat()
receipt = {
    'recorded_at': now,
    'scope': 'Coordinator localized prose edit after original reviewer formal I1/C8 revise; change provenance only, not independent scientific verification.',
    'source': 'course/chapters/13.md#13.9',
    'before_source_sha256': sha(a['13.9']), 'after_source_sha256': sha(b['13.9']),
    'before_complete_chapter_sha256': sha(before), 'after_complete_chapter_sha256': sha(after),
    'changed_sections': changed, 'unchanged_sections': [key for key in a if key not in changed],
    'history_path': history.relative_to(root).as_posix(), 'history_sha256': sha(history.read_bytes()),
    'replacement_old': old.decode(), 'replacement_new': new.decode(),
    'figures_fences_code_data_intro_unchanged': True,
}
output = Path(__file__).with_name('revision.stdout.json')
output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
progress_path = root / 'docs/course-revision-20261005/factual-review-progress.json'
progress = json.loads(progress_path.read_bytes())
record = progress['records']['13.9']
record['events'].append({
    'at': now, 'event': 'single_writer_source_revision_after_own_initial_revise',
    'source_before_sha256': sha(a['13.9']), 'source_after_sha256': sha(b['13.9']),
    'history_path': receipt['history_path'], 'history_sha256': receipt['history_sha256'],
    'change_artifact_path': Path(__file__).relative_to(root).as_posix(),
    'change_artifact_sha256': sha(Path(__file__).read_bytes()),
    'change_stdout_path': output.relative_to(root).as_posix(), 'change_stdout_sha256': sha(output.read_bytes()),
    'note': 'Restrict the arithmetic experiment statement to its actual main branch, retaining separately described natural-data branch and all original numbers/limitations. Necessary original-owner and original-reader callbacks have not yet been dispatched.',
})
record['status'] = 'recheck_queued'
record['report_version_current'] = False
progress_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2) + '\n')
reader_path = root / 'docs/course-revision-20261005/review-progress.json'
readers = json.loads(reader_path.read_bytes())
readers['records']['13.9']['status'] = 'recheck_queued'
reader_path.write_text(json.dumps(readers, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(receipt, ensure_ascii=False))
