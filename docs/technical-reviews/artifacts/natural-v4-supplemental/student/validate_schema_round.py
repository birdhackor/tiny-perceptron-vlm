"""Use unchanged official component rules without inventing a numbered lesson."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[5]
OUT=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
report=json.loads((OUT/'report.json').read_text())
official=ROOT/'scripts/check_technical_reviews.py'
spec=importlib.util.spec_from_file_location('unchanged_official_technical_components',official)
checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)
errors=[]
artifacts=checker._artifacts(ROOT,report['artifacts'],errors)
sources=checker._sources(ROOT,report['sources'],artifacts,errors)
claims=checker._claims(report['claims'],sources,artifacts,errors)
assert report['reviewer_task']=='/root/v4_review_coordinator/factual_guide_student'
assert report['reviewer_context']=='fresh' and 'lesson_id' not in report
for path,digest in report['current_document_sha256'].items():assert sha(ROOT/path)==digest
for path,digest in report['figure_sha256'].items():assert sha(ROOT/path)==digest
assert (ROOT/report['source']).read_bytes()==(OUT/'STUDENT.initial-fullfile.md').read_bytes()
assert sha(OUT/'report.first-genuine-FINAL.json')=='966e2712c5dc949accfb9d62dbed993368b523847efcadb7cd49eecefd8a8039'
for source in sources.values():
    if source['kind'] in {'official_source','official_docs','paper'}:
        assert sha(ROOT/source['ignored_original_path'])==source['retrieval_sha256']
assert set(report['checks'])==set(checker.CHECKS)
assert report['verdict']=='pass' and not report['issues']
result={'schema_component_status':'passed' if not errors else 'failed','actual_errors':errors,'scope':'Unmodified official _artifacts/_sources/_claims plus supplemental fullfile/identity/hash preservation; no fake lesson_id or per-lesson validation; factual verdict remains personal judgment','command':'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python '+str(Path(__file__).relative_to(ROOT)),'environment':{'python':sys.version,'device':'cpu'},'official_component_source_sha256':sha(official),'report_sha256':sha(OUT/'report.json'),'current_document_sha256':report['current_document_sha256'],'figure_sha256':report['figure_sha256'],'claims':len(claims),'sources':len(sources),'artifacts':len(artifacts),'first_FINAL_preserved':True}
print(json.dumps(result,ensure_ascii=False,indent=2))
if errors:sys.exit(1)
