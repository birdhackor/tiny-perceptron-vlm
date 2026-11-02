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
historical_aliases=[]
for name,value in documents.items():
    for pointer,node in walk(value):
        canonical=node.get('canonical_path')
        digest=node.get('canonical_sha256',node.get('sha256'))
        if isinstance(canonical,str)and isinstance(digest,str)and re.fullmatch('[0-9a-f]{64}',digest):
            path=Path(canonical)
            path=path if path.is_absolute()else ROOT/path
            if not path.is_file()or sha(path)!=digest:
                historical_aliases.append({'map':name,'json_pointer':pointer,'historical_canonical_path':canonical,'sha256':digest})
                continue
            if 'bytes'in node:
                assert path.stat().st_size==node['bytes']
            aliases[digest].add(str(path))
            alias_rows.append({'map':name,'json_pointer':pointer,'canonical_path':canonical,'sha256':digest})

for item in historical_aliases:
    item['current_digest_copies']=sorted(aliases.get(item['sha256'],[]))


import subprocess
MODELS=Path('/workspace/work/omni-models-v1')
UPSTREAM=Path('/workspace/work/omni-P5-skill-upstream-v6')
original_root=Path('/workspace/tiny-perceptron-vlm')
new_documents={row['path']for row in inventory if row['path']not in prior}
references=Counter()
remaining=[]
resolved_new=[]

def exact(path,digest):
    return path.is_file()and sha(path)==digest

def commit_values(value):
    result=set()
    if isinstance(value,dict):
        for key,child in value.items():
            if isinstance(child,str)and re.fullmatch('[0-9a-f]{40}',child):
                result.add(child)
            elif isinstance(child,(dict,list)):
                result.update(commit_values(child))
    elif isinstance(value,list):
        for child in value:
            result.update(commit_values(child))
    return result

for name,value in documents.items():
    for pointer,node in walk(value):
        location=node.get('path');digest=node.get('sha256')
        if not isinstance(location,str)or not isinstance(digest,str)or not re.fullmatch('[0-9a-f]{64}',digest):
            continue
        path=Path(location);absolute=path if path.is_absolute()else ROOT/path
        kind=None
        if exact(absolute,digest):
            kind='actual_original_or_local_path_exact'
        elif aliases.get(digest):
            kind='preserved_digest_resolves_to_canonical_copy'
        elif name.endswith('.agents/skills/clear-tutorial/UPSTREAM.json'):
            if pointer.startswith('/source_files/')and exact(UPSTREAM/location,digest):
                kind='fresh_pinned_upstream_original_exact'
            elif pointer.startswith('/local_history/files/'):
                history=value['local_history']
                raw=subprocess.run(['git','show',history['source_commit']+':'+history['source_directory']+'/'+location],cwd=MODELS,capture_output=True)
                if raw.returncode==0 and hashlib.sha256(raw.stdout).hexdigest()==digest:
                    kind='explicit_original_skill_Git_object_exact'
        if kind is None and name not in new_documents:
            # Relative archive/dataset/member paths do not purport to be repo
            # document references. Their unchanged manifests/roles and previous
            # actual receipt inspections are carried explicitly, not remeasured.
            kind='unchanged_prior_archive_dataset_or_historical_artifact_scope_not_reexecuted'
        if kind is None:
            candidates=[]
            if not path.is_absolute():
                candidates.extend([SNAPSHOT/path,MODELS/path,original_root/path,Path('/workspace/work/omni-qwen25-original-attempt-41015365689/artifact')/path])
                relative=path
            else:
                relative=None
                for root in (ROOT,MODELS,original_root):
                    if path.is_relative_to(root):
                        relative=path.relative_to(root);break
            if relative is not None:
                candidates.extend(Path('/workspace/work')/('omni-P5-full-tools-review-source-v'+str(version))/relative for version in (1,2,3,4,6))
            matches=[candidate for candidate in candidates if exact(candidate,digest)]
            if matches:
                kind='new_reference_exact_primary_context_or_preserved_snapshot'
            elif relative is not None and relative.parts and relative.parts[0]in {'scripts','tests','.github','docs','tiny_perceptron'}:
                for commit in commit_values(value):
                    raw=subprocess.run(['git','show',commit+':'+relative.as_posix()],cwd=MODELS,capture_output=True)
                    if raw.returncode==0 and hashlib.sha256(raw.stdout).hexdigest()==digest:
                        kind='new_historical_source_exact_declared_Git_version';break
        if kind:
            references[kind]+=1
            if name in new_documents and kind not in ('actual_original_or_local_path_exact','preserved_digest_resolves_to_canonical_copy'):
                resolved_new.append({'document':name,'pointer':pointer,'original_path':location,'sha256':digest,'resolution':kind})
        else:
            remaining.append({'document':name,'pointer':pointer,'path':location,'sha256':digest,'existing_path_has_other_current_bytes':absolute.is_file()})

out={'reviewer_task':'/root/omni_p5_full_tools_review','created_at':datetime.now(UTC).isoformat(),'scope_files':len(rows),'scope_bytes':sum(row['bytes']for row in rows),'strict_parse_counts':dict(Counter(row['type']for row in rows)),'files':rows,'canonical_portable_entries_verified':len(alias_rows),'canonical_maps':alias_rows,'preserved_old_maps_with_superseded_live_aliases':historical_aliases,'reference_categories':dict(references),'new_context_references_resolved':resolved_new,'remaining_new_references_requiring_manual_triage':remaining,'previous_generic_scan':{'raw_verbose_sha256':'ad2ffbeab9e2c0b0dc18435f04a31a5b9d8898ca9d227588e01bcba9d47dbf2c','raw_verbose_bytes':5841956,'location_receipt':'scope450-evidence-inspection-verbose-location-v1.json','reason':'Raw17190 generic unresolved-path nodes include dataset/archive members. They were never classified as missing documentation or silently marked pass. Full trace preserved outside Git; this smaller semantic classification avoids repeating file manifests.'},'reviewer_first_harness_failure':'v1 required immutable old v1 map live aliases to match current files. New v2 map preserves exact old test0362946. v1 script/log and unmodified old maps retained.','not_an_automatic_pass':'Original necessary/optional findings stay in original roles. Old source/evidence exact bytes do not inherit model capability or current stage approval. All software semantic reads/own controls/raw primary inspections are separately recorded.','limits':['280JSON files fully parsed; no new historical experiment/model reexecution is claimed for unchanged prior manifests.','Primary locations may be outside Git for inspection here; current published report will reference only small portable own proof files and exact scope identities.','No textbook/coherence or newNN/permit/provider-write approval.']}
path=Path(__file__).with_name('scope450-evidence-inspection-result-v3.json')
with path.open('x',encoding='utf-8')as stream:
    stream.write(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'scope':len(rows),'parse_counts':out['strict_parse_counts'],'portable_entries':len(alias_rows),'reference_categories':dict(references),'remaining_new_refs':remaining,'proof_bytes':path.stat().st_size},ensure_ascii=False,indent=2))
