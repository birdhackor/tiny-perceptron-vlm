from pathlib import Path
import hashlib,importlib.util,json,sys
ROOT=Path(__file__).resolve().parents[5];HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('official_component',ROOT/'scripts/check_technical_reviews.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
report=json.loads((HERE/'report.json').read_text());errors=[]
artifacts=m._artifacts(ROOT,report['artifacts'],errors)
sources=m._sources(ROOT,report['sources'],artifacts,errors)
claims=m._claims(report['claims'],sources,artifacts,errors)
assert set(report['checks'])=={'factual_accuracy','numeric_verification','figure_consistency','source_verification','limitations'}
for path,expected in report['current_document_sha256'].items():
 if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=expected:errors.append('Complete source SHA mismatch: '+path)
for path,expected in report['figure_sha256'].items():
 if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=expected:errors.append('Figure SHA mismatch: '+path)
literal=ROOT/'outputs/natural-v4/factual-research/curriculum/literal-preview-source.txt'
if literal.read_bytes()!=(ROOT/'docs/curriculum.md').read_bytes():errors.append('Literal preview source bytes do not match assigned source')
snapshot=HERE/'curriculum.snapshot.md'
if snapshot.read_bytes()!=(ROOT/'docs/curriculum.md').read_bytes():errors.append('Full raw snapshot mismatch')
out={'official_checker_sha256':hashlib.sha256((ROOT/'scripts/check_technical_reviews.py').read_bytes()).hexdigest(),'report_sha256':hashlib.sha256((HERE/'report.json').read_bytes()).hexdigest(),'components_actually_called':['_artifacts','_sources','_claims'],'claims':len(claims),'sources':len(sources),'artifacts':len(artifacts),'errors':errors,'full_source_and_snapshot_equal':True,'literal_preview_source_equal':True,'scope':'Read-only official evidence components only; no invented lesson_id and no fake numbered reviewer validation. Format/hash pass is not automatic factual proof.'}
(HERE/'component-check.result.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2));sys.exit(bool(errors))
