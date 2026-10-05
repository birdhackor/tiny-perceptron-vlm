"""Freeze only this review's original inputs; official sources are fetched separately."""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path('/workspace/tiny-perceptron-vlm')
OUT = ROOT / 'docs/technical-reviews/artifacts/phase4-5_12-independent'
revision = '5581462ef01959636425eb7795ae3153142dbfeb'
files = [
 'docs/review-tools/factual-reviewer-instructions.md', 'docs/review-tools/section_facts.py',
 'scripts/check_technical_reviews.py', '.agents/skills/clear-tutorial/references/review-protocol.md',
 'scripts/course_experiments/text.py', 'scripts/course_experiments/common.py', 'tiny_perceptron/data.py',
 'docs/course-experiments/results/real_text.json',
 'data/training/text-initial/tinystories-train-512.jsonl',
 'data/training/text-initial/tinystories-train-prefix-complete.txt',
 'data/training/text-initial/tinystories-acquisition.json',
 'data/training/text-initial/chinese-classical-train-365.jsonl',
 'data/training/text-initial/tang300-source.json',
 'data/training/text-initial/chinese-poetry-dedup-log.json',
 'assets/training/sources/tinystories.json', 'assets/training/sources/chinese-poetry.json',
]
records = []
for name in files:
 p = ROOT/name
 dst = OUT/'inputs'/name
 dst.parent.mkdir(parents=True, exist_ok=True)
 shutil.copyfile(p, dst)
 records.append({'original_path':name,'snapshot':str(dst.relative_to(ROOT)),
 'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
 'type':'existing original repository input; no new dataset/model download'})
for name in ['scripts/course_experiments/text.py','scripts/course_experiments/common.py','tiny_perceptron/data.py']:
 raw = subprocess.check_output(['git','show',revision+':'+name],cwd=ROOT)
 dst = OUT/'inputs/historical'/name
 dst.parent.mkdir(parents=True,exist_ok=True)
 dst.write_bytes(raw)
 records.append({'original_path':name,'git_revision':revision,'snapshot':str(dst.relative_to(ROOT)),
 'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'type':'original historical implementation'})
for run, destination in [('outputs/phase4-5_12-independent-extraction','extraction'),
 ('outputs/phase4-5_12-independent-original-cpu','original-cpu')]:
 for p in (ROOT/run).iterdir():
  if p.is_file():
   dst=OUT/destination/p.name
   dst.parent.mkdir(parents=True,exist_ok=True)
   shutil.copyfile(p,dst)
   records.append({'original_path':str(p.relative_to(ROOT)),'snapshot':str(dst.relative_to(ROOT)),
   'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'type':'fresh current review extraction/execution'})
from importlib.util import spec_from_file_location,module_from_spec
spec=spec_from_file_location('facts',ROOT/'docs/review-tools/section_facts.py')
mod=module_from_spec(spec);spec.loader.exec_module(mod)
for lesson,path,label in [('5.11','course/chapters/05.md','5.11-context'),('T.4','course/training.md','T.4-context')]:
 raw,whole,line=mod.original_section(ROOT/path,lesson)
 dst=OUT/'inputs'/(label+'.md');dst.write_bytes(raw)
 records.append({'original_path':path+'#'+lesson,'snapshot':str(dst.relative_to(ROOT)),
 'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'first_line':line,'type':'necessary predecessor or explicitly linked recipe; read only'})
(OUT/'input-provenance.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'frozen_inputs':len(records),'historical_revision':revision,'downloaded_datasets':False,'copied_weights':False}))
