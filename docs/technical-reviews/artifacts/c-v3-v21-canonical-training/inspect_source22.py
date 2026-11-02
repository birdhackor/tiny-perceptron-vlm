"""只核 source22 真 delta、最終26節、實收與當節kernel身份，不執行模型。"""

import ast
import difflib
import hashlib
import io
import json
import platform
import re
import tokenize
from datetime import UTC, datetime
from pathlib import Path

from scripts.omni_revision import review_setup

ROOT = Path('/tmp/tpb').resolve()
HERE = Path(__file__).resolve().parent
ROUND = 'docs/course-revision-20261101-c-v3/'
BUNDLE = {'path':'docs/technical-reviews/artifacts/c-v3-v21-numbered/numbered-section-reviews-source22-final-v1.json',
          'sha256':'d103c2981757718d0eb2869142ce415984ab754aa7edc59697373a6b8ca5b1fb'}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load(path):
    return json.loads((ROOT/path).read_bytes())


def verify(ref):
    raw=(ROOT/ref['path']).read_bytes()
    assert sha(raw)==ref['sha256'], ref
    return json.loads(raw)


def chapter(freeze, source):
    parts=[row for row in freeze['pages'] if row['source']==source]
    intro=next(row for row in parts if row['selector']=='before-first-section')
    sections=sorted([row for row in parts if row['selector'].startswith('section:')], key=lambda row:tuple(int(x) for x in row['page_id'].split('.')))
    result=b''
    for row in [intro,*sections]:
        raw=(ROOT/row['snapshot']).read_bytes();assert sha(raw)==row['source_sha256'];result+=raw
    return result, {row['page_id']:(ROOT/row['snapshot']).read_text() for row in sections}


result={'created_at':datetime.now(UTC).isoformat(),'actor':'/root/c_v3_technical_coord/canonical_training',
        'environment':{'python':platform.python_version(),'device':'CPU static JSON/hash/AST only','root':str(ROOT)},
        'scope':'Source21 evidence retained by identity. New source22 exact raw/AST/token/prose and final26 source/receipt checks only. No NN/training/final/research/network/installation/publication.'}
freeze21=load(ROUND+'source-freeze-v21.json');freeze22=load(ROUND+'source-freeze-v22.json')
oldraw,oldsections=chapter(freeze21,'course/chapters/19.md');newraw,newsections=chapter(freeze22,'course/chapters/19.md')
assert sha(oldraw)=='6fb67227641d879216eaf5506763552a6a3173a106140c8f0ff94b700edfd61a'
assert sha(newraw)=='d72c66984bac350585d79a6d21369096b372926d77069b90f1b8e81e15ebe4e5'==sha((ROOT/'course/chapters/19.md').read_bytes())
assert [key for key in oldsections if oldsections[key]!=newsections[key]]==['19.25']
pattern=r'```python\n(.*?)```'
oldblocks=re.findall(pattern,oldsections['19.25'],flags=re.S);newblocks=re.findall(pattern,newsections['19.25'],flags=re.S)
assert len(oldblocks)==len(newblocks)==2
assert re.sub(pattern,'<PYTHON>',oldsections['19.25'],flags=re.S)==re.sub(pattern,'<PYTHON>',newsections['19.25'],flags=re.S)
tokens=[]
for i,(old,new) in enumerate(zip(oldblocks,newblocks,strict=True),1):
    assert ast.dump(ast.parse(old),include_attributes=False)==ast.dump(ast.parse(new),include_attributes=False)
    ignored={tokenize.ENCODING,tokenize.NL,tokenize.NEWLINE,tokenize.INDENT,tokenize.DEDENT,tokenize.ENDMARKER}
    before=[(x.type,x.string) for x in tokenize.generate_tokens(io.StringIO(old).readline) if x.type not in ignored]
    after=[(x.type,x.string) for x in tokenize.generate_tokens(io.StringIO(new).readline) if x.type not in ignored]
    changes=[{'operation':op,'before':before[a:b],'after':after[c:d]} for op,a,b,c,d in difflib.SequenceMatcher(a=before,b=after).get_opcodes() if op!='equal']
    tokens.append({'block':i,'AST_identical':True,'nonlayout_token_changes':changes})
assert tokens[0]['nonlayout_token_changes']==[]
assert [x['before'][0][1] for x in tokens[1]['nonlayout_token_changes']]==['(',')']
result['format_delta']={'old_chapter19_sha256':sha(oldraw),'new_chapter19_sha256':sha(newraw),'changed_sections':['19.25'],
                        'python_blocks':tokens,'prose_and_nonpython_fences_identical':True,
                        'interpretation':'Format-only with removal of exactly one redundant outer parenthesis pair around accepted; not strictly whitespace-only. No source or figure claim changed.'}
bundle=verify(BUNDLE);numbered=verify(bundle['section_freeze']);canonical=verify(bundle['canonical_source_freeze']);pages={x['page_id']:x for x in numbered['pages']}
assert bundle['canonical_source_freeze']['sha256']=='0f4e9d63bfcb2243234c17c62242ff79cd4ab640a21d1f313f94c99946a93a59'
assert bundle['section_freeze']['sha256']=='3f2671acc39c31215473924da29987c80ac444565184634dda9c8e033fd291c6'
assert len(bundle['sections'])==26
validator=review_setup.module('canonical_training_source22_validator','scripts/check_technical_reviews.py')
owners={};reports={}
for row in bundle['sections']:
    report=verify(row['report']);reports[row['lesson_id']]=report;owners.setdefault(report['reviewer_task'],[]).append(row['lesson_id'])
result['final26']={'reference':BUNDLE,'canonical_freeze':bundle['canonical_source_freeze'],'section_freeze':bundle['section_freeze'],
                   'scope':'26 genuine owner reports;21 retained+W7 bounded callback+4 fresh section owners; not globalT/C seal and no collector hash cycle.', 'rows':[]}
for row in bundle['sections']:
    identifier=row['lesson_id'];report=reports[identifier];original=verify(row['original_report']);page=pages[identifier]
    assert report['source_sha256']==page['source_sha256']==sha((ROOT/page['snapshot']).read_bytes())
    assert report['reviewer_task']==original['reviewer_task']
    assert original['created_at']<=row['original_collection']['received_at'] and report['created_at']<=row['current_collection']['received_at']
    dependencies=[]
    for item in report['sources']+report['artifacts']:
        if item.get('path') and item.get('sha256'):
            actual=sha((ROOT/item['path']).read_bytes());assert actual==item['sha256'],(identifier,item['path'])
            dependencies.append({'path':item['path'],'sha256':actual})
    errors=validator._validate(ROOT,ROOT/page['source'],identifier,(ROOT/page['snapshot']).read_text(),report,owners,set())
    assert not errors,(identifier,errors)
    item={'lesson_id':identifier,'original':row['original_report'],'current':row['report'],'owner':report['reviewer_task'],
          'original_created_at':original['created_at'],'current_created_at':report['created_at'],
          'original_collection':row['original_collection'],'current_collection':row['current_collection'],
          'native_current_errors':errors,'dependencies':dependencies}
    if identifier in ('R.2','R.3','R.4','T.11','W.7'):
        label={'R.2':'numbered-r2','R.3':'numbered-r3','R.4':'numbered-r4','T.11':'numbered-t11','W.7':'w7'}[identifier]
        version='v3' if identifier=='W.7' else 'v2' if identifier in ('R.4','T.11') else 'v1'
        path=ROUND+'technical-coordination-v21/receipts/'+label+'-current-'+version+'.json';receipt=load(path)
        assert receipt['report']==row['report'] and receipt['received_at']==row['current_collection']['received_at']
        assert receipt['reviewer_task']==report['reviewer_task']
        item['actual_current_receipt']={'path':path,'sha256':sha((ROOT/path).read_bytes()),'received_at':receipt['received_at']}
    result['final26']['rows'].append(item)
index=verify(bundle['issue_inventory']);assert len(index['reports'])==41 and len(index['issues'])==5
for ref in index['reports']:verify(ref)
for row in index['issues']:
    report=verify(row['report']);value=report
    for part in row['json_pointer'].strip('/').split('/'):value=value[int(part)] if isinstance(value,list) else value[part]
    assert isinstance(value,dict)
result['issue_inventory']={**bundle['issue_inventory'],'actual_report_refs':41,'actual_issue_pointers':5,'all_parse_hash_pointer_checks_passed':True}
w7=reports['W.7'];w7page=pages['W.7'];w7text=(ROOT/w7page['snapshot']).read_text()
visible=re.sub(r'\[([^\]]+)\]\([^)]+\)',r'\1',w7text)
target_set=set(re.findall(r'19\.\d+',visible))
for start,end in re.findall(r'19\.(\d+)–19\.(\d+)',visible):
    target_set.update('19.'+str(number) for number in range(int(start),int(end)+1))
targets=sorted(target_set,key=lambda x:tuple(map(int,x.split('.'))))
assert len(targets)==13 and all(oldsections[x]==newsections[x] for x in targets)
result['w7_named_targets']={'ids':targets,'count':13,'raw_identical':True,'owner':w7['reviewer_task'],'current':next(x['report'] for x in bundle['sections'] if x['lesson_id']=='W.7')}
kernel_ref={'path':ROUND+'evidence/kernel-refresh-source22-v1.json','sha256':'c490237cce379d5e2d47bb4d4fc4103ddd78ce48779f08e1fff51784e620d3ed'}
kernel=verify(kernel_ref);assert kernel['retained_current_count']==331 and len(kernel['actual_executions'])==1 and kernel['bundle_count']==332
page19=next(x for x in canonical['pages'] if x['page_id']=='19.25');assert sha((ROOT/page19['notebook']).read_bytes())==kernel['actual_executions'][0]['source_sha256']
executed=list((ROOT/'outputs/omni-c-v3-kernel-bundle-v22').rglob('19.25.ipynb'))
assert len(executed)==1 and sha(executed[0].read_bytes())==kernel['actual_executions'][0]['executed_sha256']
result['kernel_scope']={'receipt':kernel_ref,'retained_identity_only':331,'actually_refreshed':['19.25'],
                       'source_notebook':page19['notebook'],'source_sha256':kernel['actual_executions'][0]['source_sha256'],
                       'executed_sha256':kernel['actual_executions'][0]['executed_sha256'],
                       'actual_new_output_path':str(executed[0].relative_to(ROOT)),
                       'retained_bundle_field_scope':'receipt.retained_bundle names the old v21 input for331 retained notebooks; the new matching19.25 output is in v22, not the unchanged v21 output.',
                       'limits':'Only19.25 execution identity checked here. No332-kernel replay or product qualification inferred.'}
result['training_source22_identity']=next(x for x in canonical['pages'] if x['page_id']=='training')
assert result['training_source22_identity']['source_sha256']==sha((ROOT/'course/training.md').read_bytes())=='4e304bb370cf257e185122b8eddd9a1422608f27cb784d438faafcd8b5e6a990'
result['completed_at']=datetime.now(UTC).isoformat();out=HERE/'source22-delta-final26-v1.json'
with out.open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n')
print(json.dumps({'path':str(out.relative_to(ROOT)),'sha256':sha(out.read_bytes()),'final26_native_errors':0,'training_source_unchanged':True,'format_only_not_whitespace_only':True,'kernel_retention':'331+1 only'},ensure_ascii=False))
