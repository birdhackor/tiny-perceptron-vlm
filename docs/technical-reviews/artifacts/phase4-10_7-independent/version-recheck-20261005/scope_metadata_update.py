"""Own scope clarification after neighboring section edits; preserve all original proof bytes."""
from pathlib import Path
import json, hashlib
from datetime import datetime, UTC

R=Path(__file__).resolve().parents[5]
A=Path(__file__).resolve().parents[1]
B=Path(__file__).resolve().parent
report_path=R/'docs/technical-reviews/10.7.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
report=json.loads(report_path.read_bytes())
prior_sha=sha(report_path)
comparisons=json.loads((B/'sha-comparison.json').read_bytes())
assert all(c['unchanged'] for c in comparisons)
assert sha(B/'current-10.7.md')==report['source_sha256']
initial_hash=report.pop('source_file_sha256')
initial_metadata=json.loads((A/'original-execution/extraction.json').read_bytes())
assert initial_hash==initial_metadata['source_file_sha256']
receipt={
    'schema_version':1,'reviewer_task':report['reviewer_task'],
    'checked_at':datetime.now(UTC).isoformat(),
    'reason':'A localized neighboring-section edit changes whole chapter bytes without changing this section. Clarify the initial extraction file-hash scope and personally recheck only this lesson and its actual necessary inputs.',
    'prior_own_report_sha256':prior_sha,
    'current_source':'course/chapters/10.md#10.7',
    'current_section_sha256':report['source_sha256'],
    'actually_reread':[
        'Current10.7 complete section, including opening, IDs, original fence, output/mapping explanation, diagram discussion, mask paragraph, exercise and details.',
        'Current10.6 complete section and current7.4 complete section.',
        'Current7.5 main prose/code/explanation/exercise through the details opening, matching the initial reading range.',
        'Current original rewrite-10-image-targets.svg complete source; existing desktop/mobile rendered PNGs personally viewed again after confirming identical current SVG hash.',
        'Current expand_modalities83-100, TinyLM.forward68-86, loss_sum/masked_loss92-105, attention_mask10-18 and CausalAttention.forward49-70.'
    ],
    'comparisons_path':(B/'sha-comparison.json').relative_to(R).as_posix(),
    'comparisons_sha256':sha(B/'sha-comparison.json'),
    'unchanged_comparisons':len(comparisons),
    'proof_reuse':{
        'original_section_bytes_unchanged':True,
        'necessary_context_snapshots_unchanged':True,
        'current_figure_bytes_unchanged':True,
        'repository_code_sources_unchanged':True,
        'registered_original_artifacts_integrity_checked':True,
        'reason':'The exact section, helper/model/attention/data code, required context and original figure are unchanged. Original standalone execution, bounded row/axis/gradient/causal checks and rendered pictures therefore remain evidence for the same six claims with their original limited support scope.',
        'new_cpu_execution':False,'new_training_or_model_measurement':False,
        'new_render_execution':False,
        'scope_expanded':False
    },
    'initial_whole_chapter_measurement':{
        'path':'course/chapters/10.md',
        'sha256':initial_hash,'measured_on':'2026-10-05',
        'scope':'Historical SHA computed from entire chapter bytes by the original extraction. It is not a current file fingerprint or a claim that the whole chapter was reviewed.',
        'retained_original_extraction_metadata':(A/'original-execution/extraction.json').relative_to(R).as_posix(),
        'retained_original_extraction_metadata_sha256':sha(A/'original-execution/extraction.json'),
        'retained_original_section_snapshot':(A/'original-execution/section.md').relative_to(R).as_posix(),
        'retained_original_section_snapshot_sha256':sha(A/'original-execution/section.md'),
        'full_historical_chapter_snapshot_retained':False,
        'honest_snapshot_boundary':'Original extraction saved exact section/fence/bootstrap/figure snapshots plus whole-file SHA metadata; it did not save a complete historical10.md copy. No full-chapter snapshot is invented or reconstructed.'
    },
    'current_whole_chapter_measurement_only':{
        'sha256':sha(R/'course/chapters/10.md'),
        'measured_at':datetime.now(UTC).isoformat(),
        'whole_chapter_reread':False,
        'use':'Diagnostics explaining file-level change only. Current review remains10.7 plus the named necessary context; not a current whole-chapter correctness certificate.'
    },
    'claim_and_verdict_changes':False,
    'independence':'No previous foreign report, author summary,10.5 content or opaque historical own-pass backup was opened. Only the current own canonical report was loaded to clarify its metadata.'
}
(B/'proof-reuse-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
def add_artifact(id,path,description):
    p=R/path
    assert id not in {a['id'] for a in report['artifacts']}
    report['artifacts'].append({'id':id,'path':path,'sha256':sha(p),'kind':'source_snapshot','description':description})
add_artifact('frozen-original-section',(A/'original-execution/section.md').relative_to(R).as_posix(),'Actual exact10.7 raw-byte snapshot retained from original extraction on2026-10-05; unchanged current section, not a full historical chapter copy.')
add_artifact('frozen-original-extraction',(A/'original-execution/extraction.json').relative_to(R).as_posix(),'Unmodified historical extraction metadata, including section/whole-file hashes. Whole-file hash is measured at initial extraction and not asserted current.')
add_artifact('version-proof-reuse-receipt',(B/'proof-reuse-receipt.json').relative_to(R).as_posix(),'Own current-version rereading, historical/current whole-file scope clarification, unchanged input/proof comparisons and limited proof-reuse reason.')
add_artifact('version-sha-comparison',(B/'sha-comparison.json').relative_to(R).as_posix(),'Actual36 SHA/byte comparisons of prior registered evidence, current repository sources, necessary context and figure; all unchanged.')
for lesson in ('10.7','10.6','7.4','7.5'):
    add_artifact('version-current-'+lesson.replace('.','_'),(B/f'current-{lesson}.md').relative_to(R).as_posix(),'Current source section snapshot used for this reviewer actual named-range rereading. Full snapshot retention does not imply reading beyond the range declared in proof-reuse receipt.')
report['initial_extraction_source_file_measurement']={
    'path':'course/chapters/10.md','sha256':initial_hash,'measured_on':'2026-10-05',
    'scope':'Historical whole-file SHA at initial extraction only; not current, not whole-chapter review coverage. Exact frozen10.7 section and unmodified extraction metadata are retained; a complete historical chapter snapshot was not retained.',
    'extraction_metadata_artifact_id':'frozen-original-extraction',
    'exact_section_snapshot_artifact_id':'frozen-original-section',
    'full_historical_chapter_snapshot_retained':False
}
report['source_sha256_scope']='Current10.7 original UTF-8 section bytes, personally reread during the version recheck and identical to the original frozen section. No current whole-file SHA is asserted as review coverage.'
report['version_recheck_receipt_artifact_id']='version-proof-reuse-receipt'
report['checks']['limitations']['details']+=' Whole-file SHA is now explicitly initial-extraction metadata; no full historical chapter snapshot was retained. The current-version proof-reuse receipt lists actual reread ranges, unchanged evidence and unchanged claim scope.'
report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'new_report_sha256':sha(report_path),'receipt_artifact_id':'version-proof-reuse-receipt','receipt_sha256':sha(B/'proof-reuse-receipt.json'),'current_section_sha256':report['source_sha256'],'source_file_measurement_scope':'Historical initial extraction only'},ensure_ascii=False,indent=2))
