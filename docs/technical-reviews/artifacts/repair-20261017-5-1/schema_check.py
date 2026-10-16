"""Call the requested original validator on this independent report only."""
import hashlib
import importlib.util
import json
import platform
from pathlib import Path

R = Path(__file__).resolve().parents[4]
source = R / 'course/chapters/05.md'
report_path = R / 'docs/course-repair-20261017/reviews/technical-5-1-original-01.json'
spec = importlib.util.spec_from_file_location('original_technical_validator', R / 'scripts/check_technical_reviews.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
body = next(body for identifier, body in module.sections(source) if identifier == '5.1')
report = json.loads(report_path.read_text())
errors = module._validate(R, source, '5.1', body, report, reviewer_owners={'/root/repair_tech_5_1': ['5.1']}, reader_tasks=['/root/astra_repair_01_readability'])
result = {'reviewer_task': '/root/repair_tech_5_1', 'validator': 'scripts/check_technical_reviews.py::_validate', 'report_path': str(report_path.relative_to(R)), 'report_sha256': hashlib.sha256(report_path.read_bytes()).hexdigest(), 'source_sha256': hashlib.sha256(body.encode()).hexdigest(), 'reviewer_owners': {'/root/repair_tech_5_1': ['5.1']}, 'reader_tasks': ['/root/astra_repair_01_readability'], 'environment': {'python': platform.python_version(), 'device': 'cpu'}, 'errors': errors, 'meaning': 'Only format, identity declarations, local evidence hashes and current-section consistency; does not automatically establish substantive correctness.'}
print(json.dumps(result, ensure_ascii=False, indent=2))
assert not errors, errors
