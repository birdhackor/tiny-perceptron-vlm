import json,shutil
from pathlib import Path
s=Path('docs/course-experiments/results/moe.json');d=Path('docs/technical-reviews/artifacts/p7_technical_e/sources/moe.json');assert not d.exists();shutil.copyfile(s,d)
a=json.loads(d.read_text())['results'];b=json.loads(Path('docs/technical-reviews/artifacts/p7_technical_e/sources/distillation.json').read_text())['results']['tasks']['moe_to_dense']['teacher_provenance'];print('teacher_train_records',a['dataset']['train']['records'],'chosen_variant',a['teacher_variant'],'dataset_matches_teacher',a['dataset']['train']['sha256']==b['metadata']['records_sha256']);assert a['dataset']['train']['records']==409 and a['dataset']['train']['sha256']==b['metadata']['records_sha256']
