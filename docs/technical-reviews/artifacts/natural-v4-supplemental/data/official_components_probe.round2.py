"""Run unchanged official supplemental-compatible components on this owner's report."""
from pathlib import Path
import hashlib,importlib.util,json,sys

root=Path.cwd()
report_path=Path(sys.argv[1]);prefix=Path(sys.argv[2])
checker=root/'scripts/check_technical_reviews.py'
before=hashlib.sha256(checker.read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location('official_technical_components',checker)
official=importlib.util.module_from_spec(spec)
spec.loader.exec_module(official)
report=json.loads(report_path.read_bytes())
errors=[]
artifacts=official._artifacts(root,report['artifacts'],errors)
sources=official._sources(root,report['sources'],artifacts,errors)
claims=official._claims(report['claims'],sources,artifacts,errors)
after=hashlib.sha256(checker.read_bytes()).hexdigest()
assert before==after
result={'report_path':str(report_path),'report_sha256':hashlib.sha256(report_path.read_bytes()).hexdigest(),
        'checker_path':'scripts/check_technical_reviews.py','checker_sha256_before':before,'checker_sha256_after':after,
        'unchanged_checker':True,'components':['official._artifacts','official._sources','official._claims'],
        'artifact_count':len(artifacts),'source_count':len(sources),'claim_count':len(claims),
        'errors':errors,'error_count':len(errors),'status':'pass' if not errors else 'revise',
        'scope':'Official component schema/support validation only; does not call numbered-section identity validator, invent lesson_id or adjudicate factual judgment.'}
prefix.with_suffix('.result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
raise SystemExit(1 if errors else 0)
