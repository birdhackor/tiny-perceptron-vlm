"""Persist small primary-source/version receipts; full originals stay outside Git."""
import hashlib
import inspect
import json
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch
from scripts.course_experiments import capstone
from scripts.course_experiments.common import Context
from scripts.course_experiments.run import experiment_spec

CACHE = Path('/workspace/work/reference/omni-technical-t-10-v1')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
receipt = {'created_at': datetime.now(timezone.utc).isoformat(),
    'environment': {'python': platform.python_version(), 'torch': str(torch.__version__), 'device': 'cpu'},
    'original_sources': []}
for filename, url, version, locator, note, quote in [
    ('hinton-1503.02531v1.pdf', 'https://arxiv.org/pdf/1503.02531v1', 'arXiv:1503.02531v1, 2015-03-09', 'pp.2-3, sections 1-2; p.3 equation (1); p.4 section 3',
     'Read the original PDF text through section 3. Distillation uses the same high temperature on teacher/student, combines soft-target and hard-label cross entropy, and multiplies the soft gradients by T squared. Section 3 compares small architecture with ordinary training; none of this promises universal improvement.',
     'The first objective function is the cross entropy with the soft targets'),
    ('gsm8k-pinned-readme.md', 'https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/README.md', 'OpenAI grade-school-math commit 3101c7d5072418e28b9008a6636bde82a006892c', 'README Dataset Details and Solution Extracting, lines 7-32',
     'Read official upstream README, including dataset purpose, grade-school arithmetic word problems, question/answer fields and #### final answer extraction. Inspected downloaded English records from the pinned subset; no GSM8K benchmark-quality inference.',
     'GSM8K consists of 8.5K high quality grade school math problems created by human problem writers.'),
    ('torch-loss-5c488690.py', 'https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/torch/nn/modules/loss.py', 'PyTorch installed 2.14.1+cpu, git 5c4886908584029761b579af026dcfb627c84070', 'KLDivLoss lines 464-515; CrossEntropyLoss lines 1200-1260',
     'Read official exact-commit source; verified byte equality with installed torch.nn.modules.loss. KL input is log-probability, target is probability, candidate sum precedes valid-token mean. CrossEntropyLoss takes unnormalized logits and class-index targets with ignore_index.',
     'loss_pointwise = target * (target.log() - input)'),
    ('torch-functional-5c488690.py', 'https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/torch/nn/functional.py', 'PyTorch installed 2.14.1+cpu, same git commit', 'kl_div lines 3401-3476; cross_entropy lines 3478 onwards',
     'Read official exact-commit Python wrapper and reduction handling. Byte-equal with installed functional.py. reduction=none preserves pointwise terms, so project code explicitly sums candidates and averages only valid positions.',
     'input: Tensor of arbitrary shape in log-probabilities.'),
]:
    p = CACHE / filename
    receipt['original_sources'].append({'cache_path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p),
        'url': url, 'version': version, 'locator': locator, 'accessed_on': '2026-10-29',
        'checked_original': True, 'inspection_note': note, 'short_quote': quote})
receipt['official_docs_fallback'] = 'docs.pytorch.org stable KLDivLoss and CrossEntropyLoss returned HTTP 403; read official GitHub exact-commit source instead, byte-matched to installed package.'
receipt['historical_compression'] = {'revision': '5af615e5d7c9642afee800390fa072257f895d0c',
    'cache_path': str(CACHE / '5af615e5-compression.py'), 'sha256': sha(CACHE / '5af615e5-compression.py'),
    'current_sha256': sha(ROOT / 'scripts/course_experiments/compression.py'),
    'diff_cache': str(CACHE / 'compression-version-diff.txt'), 'diff_sha256': sha(CACHE / 'compression-version-diff.txt'),
    'finding': 'Only current multimodal teacher task/data consistency guard and copied task metadata differ; core text and multimodal CE/KL training, widths, filenames, masks and schedules are unchanged.'}
receipt['necessary_prerequisite_sections'] = []
for path, ids in [('course/training.md', ['T.4','T.5','T.6','T.7','T.8','T.9']),
                  ('course/chapters/01.md',['1.8']), ('course/chapters/18.md',['18.8','18.9','18.13']),
                  ('course/chapters/12.md',['12.12'])]:
    text = (ROOT / path).read_text()
    for identifier in ids:
        section = re.search(r'^## ' + re.escape(identifier) + r' .*?(?=^## |\Z)', text, re.M|re.S).group(0)
        receipt['necessary_prerequisite_sections'].append({'source': path + '#' + identifier,
            'section_sha256': hashlib.sha256(section.encode()).hexdigest(),
            'scope': 'Read for linked recipe/method context. Figures outside T.10 were not visually audited or credited as T.10 figure evidence.'})
receipt['freeze'] = {p: sha(ROOT / p) for p in [
    'docs/course-revision-20261029-omni/reviews/numbered-freeze-v5.json',
    'docs/course-revision-20261029-omni/reviews/numbered-freeze-v5/sections/T.10.md']}
ctx = Context('cpu', CACHE / 'capstone-route-only', CACHE / 'capstone-parent', ROOT / 'assets/training')
with patch.object(capstone, 'train_stage', return_value={'intercepted': True}) as call:
    capstone.run_joint(ctx)
    kwargs = call.call_args.kwargs
receipt['capstone_joint_route_dry_run'] = {'config': {k:v for k,v in experiment_spec('capstone_joint').items() if k in ('id','module','function','dependencies')},
    'input_checkpoint': str(kwargs['input_checkpoint']), 'steps': kwargs['steps'],
    'scope': 'train_stage intercepted; confirms capstone_sft predecessor, no training.'}
receipt['capstone_code'] = {'path': 'scripts/course_experiments/capstone.py',
    'sha256': sha(ROOT / 'scripts/course_experiments/capstone.py'), 'locator': '_run_context lines 280-294 and run_joint lines 304-305'}
OUT = Path(__file__).with_name('source-receipts.json')
assert not OUT.exists()
OUT.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'output': str(OUT.relative_to(ROOT)), 'sha256': sha(OUT)}))
