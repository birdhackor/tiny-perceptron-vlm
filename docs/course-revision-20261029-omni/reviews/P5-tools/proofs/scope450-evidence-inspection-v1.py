"""Whole450 strict parse and portable-reference inspection; not synthetic judgments."""
import ast
import hashlib
import json
import re
import tomllib
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parents[5]
SNAPSHOT=Path('/workspace/work/omni-P5-full-tools-review-source-v6')
BASE=Path('docs/course-revision-20261029-omni')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pairs(rows):
    result={}
    for key,value in rows:
        if key in result:
            raise ValueError('duplicate JSON key '+key)
        result[key]=value
    return result


def nonfinite(value):
    raise ValueError('nonfinite JSON constant '+value)


def parse(raw):
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=nonfinite)


def walk(value,pointer=''):
    if isinstance(value,dict):
        yield pointer,value
        for key,child in value.items():
            yield from walk(child,pointer+'/'+str(key))
    elif isinstance(value,list):
        for index,child in enumerate(value):
            yield from walk(child,pointer+'/'+str(index))


inventory=parse((ROOT/BASE/'reviews/P5-tools/scope-inventory-v6.json').read_bytes())['checked_file_scope']
prior={row['path']:row for row in parse((ROOT/BASE/'reviews/P5-tools/scope-inventory-v4.json').read_bytes())['checked_file_scope']}
documents={}
rows=[]
for binding in inventory:
    name=binding['path']
    path=SNAPSHOT/name
    raw=path.read_bytes()
    assert len(raw)==binding['bytes'] and hashlib.sha256(raw).hexdigest()==binding['sha256']
    row={**binding,'live_bytes_match_at_inspection':sha(ROOT/name)==binding['sha256'],'previous336_exact_identity':prior.get(name)==binding}
    if name.endswith('.json'):
        value=parse(raw)
        documents[name]=value
        row.update(inspection='entire strict JSON parse, structure/provenance role; semantic and primary-byte checks recorded separately',top_keys=list(value)if isinstance(value,dict)else[],type='JSON')
        if isinstance(value,dict):
            row['status_and_identity']={key:value[key]for key in ('schema','schema_version','created_at','created_at_utc','status','verdict','reviewer_task','kind','publication_status','run_id','source_revision')if key in value and not isinstance(value[key],(dict,list))}
            findings=[]
            for key in ('issues','necessary_findings','optional_findings','resolved_findings','finding_closure','runtime_finding_closure'):
                field=value.get(key)
                items=field if isinstance(field,list)else[field]if isinstance(field,dict)else[]
                for index,item in enumerate(items):
                    if isinstance(item,dict):
                        findings.append({'source_key':key,'index':index,**{k:item[k]for k in ('id','issue_id','severity','necessity','status','summary','problem','statement','resolution')if k in item}})
            if findings:
                row['preserved_findings_and_closures']=findings
    elif name.endswith('.py'):
        tree=ast.parse(raw)
        row.update(type='Python',inspection='whole AST and actual source identity; semantic read level recorded in coverage appendix',definitions=[node.name for node in ast.walk(tree)if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))])
    elif name.endswith('.toml') or name=='uv.lock':
        tomllib.loads(raw.decode())
        row.update(type='TOML',inspection='whole strict TOML parse and byte identity')
    else:
        raw.decode('utf-8')
        row.update(type='UTF8',inspection='whole UTF-8/byte identity; semantic read level recorded in coverage appendix')
    rows.append(row)

# Build a digest->portable-copy map from actual explicitly supplied location maps.
aliases=defaultdict(set)
alias_rows=[]
for name,value in documents.items():
    for pointer,node in walk(value):
        canonical=node.get('canonical_path')
        digest=node.get('canonical_sha256',node.get('sha256'))
        if isinstance(canonical,str)and isinstance(digest,str)and re.fullmatch('[0-9a-f]{64}',digest):
            path=Path(canonical)
            path=path if path.is_absolute()else ROOT/path
            assert path.is_file()and sha(path)==digest,(name,pointer,canonical)
            if 'bytes'in node:
                assert path.stat().st_size==node['bytes']
            aliases[digest].add(str(path))
            alias_rows.append({'map':name,'json_pointer':pointer,'canonical_path':canonical,'sha256':digest})

references=Counter()
unresolved=[]
bad=[]
for name,value in documents.items():
    for pointer,node in walk(value):
        location=node.get('path')
        digest=node.get('sha256')
        if not isinstance(location,str)or not isinstance(digest,str)or not re.fullmatch('[0-9a-f]{64}',digest):
            continue
        path=Path(location)
        path=path if path.is_absolute()else ROOT/path
        if path.is_file()and sha(path)==digest:
            if 'bytes'in node and path.stat().st_size!=node['bytes']:
                bad.append({'document':name,'pointer':pointer,'reason':'size differs despite matching digest','path':location})
            references['actual_original_or_local_path_exact']+=1
        elif aliases.get(digest):
            references['preserved_digest_resolves_to_canonical_copy']+=1
        else:
            unresolved.append({'document':name,'pointer':pointer,'path':location,'sha256':digest,'exists_with_other_current_bytes':path.is_file()})
assert not bad,bad
out={'reviewer_task':'/root/omni_p5_full_tools_review','created_at':datetime.now(UTC).isoformat(),'scope_files':len(rows),'scope_bytes':sum(row['bytes']for row in rows),'strict_parse_counts':dict(Counter(row['type']for row in rows)),'files':rows,'canonical_portable_entries_verified':len(alias_rows),'canonical_maps':alias_rows,'reference_categories':dict(references),'unresolved_original_references_for_semantic_triage':unresolved,'not_an_automatic_pass':'Historical original findings and current closures are recorded verbatim in their original roles. Missing historical path or stale live path is not automatically a software defect or silently converted to pass; specific digest/location/provenance must be assessed.','limits':['Strict JSON/AST/TOML parsing is not a substitute for actual software semantic reads or original package inspection.','No textbook/coherence/reader acceptance, model remeasurement, new permit or provider write.']}
path=Path(__file__).with_name('scope450-evidence-inspection-result-v1.json')
with path.open('x',encoding='utf-8')as stream:
    stream.write(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'scope_files':len(rows),'parse_counts':out['strict_parse_counts'],'canonical_portable_entries_verified':len(alias_rows),'reference_categories':dict(references),'unresolved_original_refs':len(unresolved),'unresolved_examples':unresolved[:12]},ensure_ascii=False,indent=2))
