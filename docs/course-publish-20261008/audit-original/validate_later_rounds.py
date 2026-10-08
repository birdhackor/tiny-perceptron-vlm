from pathlib import Path
from datetime import datetime, timezone
import json, hashlib, re

B = Path(__file__).resolve().parent
M = json.loads((B/'manifest.json').read_text())
pages = {p['page_id']:p for p in M['inventory']['pages']}
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def norm(s): return re.sub(r'\s+', '', s).replace('`','').replace('**','')
def quotes(v):
    if isinstance(v,str): return [v]
    if isinstance(v,list): return sum((quotes(x) for x in v), [])
    if isinstance(v,dict):
        return quotes(v.get('quote',v.get('text',v.get('source_quote',[]))))
    return []

errors, groups = [], []
seals = []
for p in B.rglob('*seal*.json'):
    try:
        s = json.loads(p.read_text())
        if isinstance(s,dict) and (s.get('role') in ['technical','continuity'] or re.match(r'(technical|continuity)-.+-initial[.-]seal',p.name)):
            seals.append((p,s))
    except (ValueError,UnicodeError): pass
for role in ['technical','continuity']:
    for g, meta in M['groups'].items():
        p = B/'reports'/f'{role}-{g}-initial.json'
        if not p.exists():
            groups.append({'role':role,'group':g,'present':False}); continue
        d = json.loads(p.read_text())
        ps = d['pages']; ps = list(ps.values()) if isinstance(ps,dict) else ps
        ids = [x['page_id'] for x in ps]
        if ids != meta['pages']: errors.append(f'{role}-{g}: page route mismatch')
        criteria = d.get('criteria_sha256',{})
        if isinstance(criteria,str):
            criteria = {'SKILL.md':criteria}
            criteria.update({('references/'+k if not k.startswith('references/') else k):v for k,v in d.get('criteria_references',{}).items()})
            criteria.update({k.removeprefix('freeze/criteria/'):v for k,v in d.get('criteria_files',{}).items()})
        for k in ['SKILL.md','references/review-protocol.md','references/calibration.md']:
            if criteria.get(k) != M['criteria_sha256'][k]:
                errors.append(f'{role}-{g}: criteria mismatch {k}')
        quote_matches, no_match = 0, []
        for x in ps:
            pid = x['page_id']
            if x.get('source_sha256') != pages[pid]['source_sha256']:
                errors.append(f'{role}-{g}: source hash mismatch {pid}')
            body = norm((B/'freeze/sources'/f'{pid}.md').read_text())
            qs = quotes(x.get('quoted_basis',[]))
            if any(norm(q) in body for q in qs if q): quote_matches += 1
            else: no_match.append(pid)
        matching = [(s,raw) for s,raw in seals if (raw.get('role')==role and raw.get('group')==g) or s.name in [f'{role}-{g}-initial.seal.json',f'{role}-{g}-initial-seal.json']]
        sealed_hash = next((raw.get('report_sha256',raw.get('initial_report_sha256',raw.get('initial_sha256',raw.get('sha256')))) for s,raw in matching),None)
        if sealed_hash and sealed_hash != sha(p): errors.append(f'{role}-{g}: sealed report drift')
        groups.append({'role':role,'group':g,'present':True,'pages':len(ps),'expected_pages':len(meta['pages']),'sha256':sha(p),'seal_path':str(matching[0][0].relative_to(B)) if matching else None,'sealed_hash_matches':sealed_hash==sha(p),'pages_with_at_least_one_verbatim_quote_match':quote_matches,'quote_match_manual_followup':no_match})
impl = json.loads((B/'checks/implementation-source-manifest.json').read_text())
for name,digest in impl['files_sha256'].items():
    if sha(B/'freeze/implementation'/name) != digest: errors.append('implementation frozen drift: '+name)
result = {'at':datetime.now(timezone.utc).isoformat(),'errors':errors,'groups':groups,'quote_match_normalization':'whitespace and inline-code/emphasis markup only','meaning':'hash/route/quote mechanics only; actual source reading, judgment quality and execution scopes remain separately recorded'}
(B/'checks/later-round-integrity.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
raise SystemExit(bool(errors))
