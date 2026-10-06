"""Opaque preservation of this reviewer's prior report before a third actual read."""
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path('/workspace/tiny-perceptron-vlm')
OUT = ROOT / 'docs/technical-reviews/artifacts/phase4-6_4-independent/reinspection-v2'
prior = (ROOT / 'docs/technical-reviews/6.4.json').read_bytes()
prior_sha = hashlib.sha256(prior).hexdigest()
history = ROOT / ('docs/technical-reviews/history/phase4-6_4-own-prior-before-current-' + prior_sha + '.json')
if history.exists():
    assert history.read_bytes() == prior
else:
    history.write_bytes(prior)
(OUT / 'prior-report.json').write_bytes(prior)
helper_path = ROOT / 'docs/review-tools/section_facts.py'
spec = importlib.util.spec_from_file_location('section_facts_reinspection_v2', helper_path)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
body, whole, first_line = helper.original_section(ROOT / 'course/chapters/06.md', '6.4')
(OUT / 'section.md').write_bytes(body)
(OUT / 'chapter-06.md').write_bytes(whole)
fences = helper.fences(body, first_line)
fence_metadata = []
for index, item in enumerate(fences, 1):
    raw = item['raw']
    filename = 'fence-' + str(index) + ('.py' if item['language'] == 'python' else '.txt')
    (OUT / filename).write_bytes(raw)
    fence_metadata.append({'language': item['language'], 'closed': item['closed'], 'file': filename, 'sha256': hashlib.sha256(raw).hexdigest(), 'code_line': item['code_line']})
for rel in ['docs/review-tools/factual-reviewer-instructions.md', 'docs/review-tools/section_facts.py', 'scripts/check_technical_reviews.py']:
    target = OUT / 'inputs' / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes((ROOT / rel).read_bytes())
receipt = {'scope': 'Prior report saved as opaque original bytes before inspecting the new source; no author repair answer or other review read.', 'prior_report_sha256': prior_sha, 'history_path': history.relative_to(ROOT).as_posix(), 'history_sha256': hashlib.sha256(history.read_bytes()).hexdigest(), 'new_source_sha256': hashlib.sha256(body).hexdigest(), 'source_file_sha256': hashlib.sha256(whole).hexdigest(), 'first_line': first_line, 'fences': fence_metadata, 'helper_sha256': hashlib.sha256(helper_path.read_bytes()).hexdigest()}
(OUT / 'prepare-receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(receipt, ensure_ascii=False, indent=2))
