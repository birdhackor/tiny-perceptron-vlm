from pathlib import Path
import json,runpy
ROOT=Path(__file__).resolve().parents[4]
ns=runpy.run_path(str(ROOT/'scripts/check_technical_reviews.py'))
result=ns['check'](root=ROOT,lessons=['19.3'])
expected=[
 '19.3: 審閱要求修訂／尚未核實',
 '19.3: claim c8: 未核實／矛盾主張不能通過，應維持revise',
 '19.3: 尚有未解決問題，不能記為pass',
 '19.3: factual_accuracy: 未通過或不適用的NA',
 '19.3: limitations: 未通過或不適用的NA',
]
assert result['checked_sections']==1 and result['failures']==expected,result['failures']
report=json.loads((ROOT/'docs/technical-reviews/19.3.json').read_bytes())
assert report['verdict']=='revise' and len(report['claims'])==21
assert sum(c['status']=='verified' for c in report['claims'])==20
assert [c['id'] for c in report['claims'] if c['status']!='verified']==['c8']
print(json.dumps({'lesson':'19.3','schema_evidence_and_current_source_hashes_valid':True,'pass_blocked_by_expected_revise':True,'claim_count':21,'verified_claims':20,'issue_claim':'c8','checker_failure_messages':result['failures']},ensure_ascii=False,indent=2))
