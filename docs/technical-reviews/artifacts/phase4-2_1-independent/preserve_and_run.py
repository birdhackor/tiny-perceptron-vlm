from datetime import datetime, UTC
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

root = Path(__file__).resolve().parents[4]
dest = Path(__file__).resolve().parent
original = root / 'outputs/reviewer-tools/phase4-2_1-independent'
permanent = dest / 'original'
shutil.copytree(original, permanent, ignore=shutil.ignore_patterns('workspace'), dirs_exist_ok=True)
for path in ['docs/review-tools/section_facts.py', 'scripts/build_course.py']:
    target = permanent / Path(path).name
    target.write_bytes((root / path).read_bytes())
raw = (root / 'course/chapters/02.md').read_bytes()
intro = raw[:raw.index(b'## 2.1 ')]
(dest / 'chapter-intro.md').write_bytes(intro)
(dest / 'chapter-original.md').write_bytes(raw)
metadata = {
    'source_file': 'course/chapters/02.md', 'source_file_sha256': hashlib.sha256(raw).hexdigest(),
    'intro_sha256': hashlib.sha256(intro).hexdigest(), 'intro_bytes': len(intro),
    'intro_line_range': [1, intro.count(b'\n')], 'newline_policy': 'original UTF-8 bytes without normalization',
}
(dest / 'intro-receipt.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')

def run(name, argv, timeout=30):
    started = datetime.now(UTC).isoformat()
    completed = subprocess.run(argv, cwd=root, capture_output=True, timeout=timeout, check=False)
    (dest / (name + '-stdout.txt')).write_bytes(completed.stdout)
    (dest / (name + '-stderr.txt')).write_bytes(completed.stderr)
    receipt = {'command_argv': argv, 'cwd': str(root), 'started_utc': started,
               'exit_code': completed.returncode, 'timeout_seconds': timeout,
               'stdout_sha256': hashlib.sha256(completed.stdout).hexdigest(),
               'stderr_sha256': hashlib.sha256(completed.stderr).hexdigest()}
    (dest / (name + '-execution.json')).write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(receipt, ensure_ascii=False))
    if completed.returncode:
        print(completed.stderr.decode('utf-8', errors='replace'))
        raise SystemExit(completed.returncode)

run('probe', [str(root / '.venv/bin/python'), str(dest / 'probe.py')])
run('figure-render', ['inkscape', str(root / 'course/figures/rewrite-02-embedding-purpose.svg'),
                      '--export-type=png', '--export-filename=' + str(dest / 'embedding-purpose.png')])
for name in ['bengio-2003-jmlr', 'vaswani-2017-v7']:
    run(name + '-pdftotext', ['pdftotext', '-layout', str(dest / 'sources' / (name + '.pdf')),
                              str(dest / 'sources' / (name + '.txt'))])
