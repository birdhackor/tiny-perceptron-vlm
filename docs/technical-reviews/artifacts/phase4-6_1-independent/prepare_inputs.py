"""Freeze only independently inspected source inputs, never other reviews."""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path('/workspace/tiny-perceptron-vlm')
OUT = Path(__file__).parent
sha = lambda b: hashlib.sha256(b).hexdigest()
for name in ['inputs', 'original', 'sources', 'historical-code']:
    (OUT / name).mkdir(exist_ok=True)
for p in Path('/tmp/phase4-6_1-independent-original').iterdir():
    if p.is_file():
        shutil.copy2(p, OUT / 'original' / p.name)
for name in ['course/chapters/06.md', 'docs/review-tools/factual-reviewer-instructions.md',
             'docs/review-tools/section_facts.py', 'scripts/check_technical_reviews.py',
             '.agents/skills/clear-tutorial/references/review-protocol.md',
             'docs/course-experiments/results/real_text.json',
             'tiny_perceptron/data.py', 'tiny_perceptron/model.py',
             'tiny_perceptron/attention.py', 'scripts/course_experiments/common.py',
             'scripts/course_experiments/text.py']:
    p = OUT / 'inputs' / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes((ROOT / name).read_bytes())
chapter = (ROOT / 'course/chapters/06.md').read_bytes()
intro = chapter[:chapter.index(b'## 6.1 ')]
(OUT / 'inputs' / 'chapter-intro.raw.md').write_bytes(intro)
result = json.loads((ROOT / 'docs/course-experiments/results/real_text.json').read_text())
historical = []
for name in ['tiny_perceptron/data.py', 'tiny_perceptron/model.py',
             'scripts/course_experiments/common.py', 'scripts/course_experiments/text.py']:
    raw = subprocess.check_output(['git','show',result['revision']+':'+name],cwd=ROOT)
    assert sha(raw) == result['code_sha256'][name], name
    path = OUT / 'historical-code' / name
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(raw)
    historical.append({'path':name, 'git_revision':result['revision'], 'sha256':sha(raw),
                       'matches_original_result_hash':True})
manifest = {
    'intro_sha256':sha(intro),
    'intro_summary':'本章從小小貓的可見文字、UTF-8 byte 與模型 token 的差異出發，逐步建立可還原未見文字的 BPE；以詞表與序列長度的共同成本、結構標記及半字解碼，追問每個模型位置代表什麼。',
    'raw_utf8_policy':'Read original bytes; no newline normalization or stripping.',
    'read_scope':['chapter 6 title/introduction','6.1 entirety','6.2–6.5 incidental context from initial chapter excerpt','T.4 fixed real-text recipe and original evaluation implementation'],
    'historical_code':historical,
    'source_section_sha256':json.loads((OUT/'original/extraction.json').read_text())['source_sha256'],
    'frozen_inputs':[{'path':str(p.relative_to(OUT)), 'sha256':sha(p.read_bytes())}
                     for p in sorted((OUT/'inputs').rglob('*')) if p.is_file()]
}
(OUT/'input-provenance.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(manifest,ensure_ascii=False,indent=2))
