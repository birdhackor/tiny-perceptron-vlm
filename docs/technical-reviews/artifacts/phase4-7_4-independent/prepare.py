"""Freeze only current section inputs and execute its unchanged Python fence on CPU."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys

ROOT = Path('/workspace/tiny-perceptron-vlm')
OUT = ROOT / 'docs/technical-reviews/artifacts/phase4-7_4-independent'
SCRATCH = Path('/tmp/phase4-7_4-independent-original')

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

inputs = [
    'docs/review-tools/factual-reviewer-instructions.md',
    'docs/review-tools/section_facts.py',
    'scripts/check_technical_reviews.py',
    '.agents/skills/clear-tutorial/references/review-protocol.md',
    'scripts/build_course.py', 'pyproject.toml',
    'tiny_perceptron/data.py', 'tiny_perceptron/model.py',
    'tiny_perceptron/attention.py', 'tiny_perceptron/modern.py',
]
for relative in inputs:
    target = OUT / 'inputs' / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / relative, target)

cmd = [str(ROOT / '.venv/bin/python'), 'docs/review-tools/section_facts.py',
       'course/chapters/07.md#7.4', '--output', str(SCRATCH), '--execute', '--timeout', '45']
run = subprocess.run(cmd, cwd=ROOT, capture_output=True, check=False)
(OUT / 'prepare.stdout.txt').write_bytes(run.stdout)
(OUT / 'prepare.stderr.txt').write_bytes(run.stderr)
metadata = json.loads((SCRATCH / 'extraction.json').read_text())
for name in ['section.md', 'fence-1.py', 'bootstrap.py', 'extraction.json',
             'execution.json', 'environment.json', 'stdout.txt', 'stderr.txt']:
    target = OUT / 'original' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SCRATCH / name, target)
for item in metadata['svg_references']:
    if item.get('snapshot'):
        target = OUT / 'original' / item['snapshot']
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SCRATCH / item['snapshot'], target)

paper = ROOT / 'docs/technical-reviews/artifacts/fact_finish_r_3-transformer.original.pdf'
target = OUT / 'sources/vaswani-1706.03762v7.pdf'
target.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(paper, target)
pdfcmd = ['pdftotext', '-layout', str(target), str(target.with_suffix('.txt'))]
pdf = subprocess.run(pdfcmd, capture_output=True, check=False)
(OUT / 'paper-extract.stdout.txt').write_bytes(pdf.stdout)
(OUT / 'paper-extract.stderr.txt').write_bytes(pdf.stderr)
rendercmd = ['inkscape', str(ROOT / 'course/figures/rewrite-07-04-answer-alignment.svg'),
             '--export-type=png', '--export-width=1280',
             '--export-filename=' + str(OUT / 'figure.png')]
render = subprocess.run(rendercmd, capture_output=True, check=False, timeout=30)
(OUT / 'render.stdout.txt').write_bytes(render.stdout)
(OUT / 'render.stderr.txt').write_bytes(render.stderr)
record = {'commands': [{'argv': cmd, 'cwd': str(ROOT), 'exit_code': run.returncode},
                       {'argv': pdfcmd, 'cwd': str(ROOT), 'exit_code': pdf.returncode},
                       {'argv': rendercmd, 'cwd': str(ROOT), 'exit_code': render.returncode}],
          'external_pdf_provenance': {'path_locator_only': str(paper.relative_to(ROOT)),
             'sha256': digest(target), 'official_url': 'https://arxiv.org/pdf/1706.03762v7',
             'independent_inspection': 'PDF identity and sections independently read after pdftotext; no prior review conclusions read.'},
          'section_sha256': metadata['source_sha256'], 'figure_sha256': metadata['figure_sha256']}
(OUT / 'prepare-receipt.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(record, ensure_ascii=False, indent=2))
sys.exit(run.returncode or pdf.returncode or render.returncode)
