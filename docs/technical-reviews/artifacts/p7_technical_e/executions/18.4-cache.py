import json
from pathlib import Path
r=json.loads(Path('docs/technical-reviews/artifacts/p7_technical_e/sources/distillation.json').read_text())['results']['tasks']['attributes'];c=r['teacher_cache'];print(c['file'],c['file_bytes']);assert c['file_bytes']==345101
