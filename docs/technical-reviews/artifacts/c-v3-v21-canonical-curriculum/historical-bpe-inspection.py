"""One optional historical BPE navigation issue, after actual original/current intake."""

import hashlib
import json
import platform
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT=Path('/tmp/tpb').resolve()
OUT=Path(__file__).resolve().parent
BASE='docs/course-revision-20261101-c-v3/'
PACKET=BASE+'technical-coordination-v21/history-prompts/child-01-group-00-historical-identities.json'


def sha(data): return hashlib.sha256(data).hexdigest()
def load_ref(ref):
    data=(ROOT/ref['path']).read_bytes();assert sha(data)==ref['sha256'];return json.loads(data)
def node(value,pointer):
    if not pointer:return value
    for key in pointer.lstrip('/').split('/'):
        key=key.replace('~1','/').replace('~0','~');value=value[int(key)] if isinstance(value,list) else value[key]
    return value


packet_ref={'path':PACKET,'sha256':'1d59618155816c88e14ef5e1d3f7de8cb8017c933edc2a4f6902764ad1f0bf61'}
packet=load_ref(packet_ref);identity=packet['historical_identities_only'][0]
assert len(packet['historical_identities_only'])==1 and identity['issue_id']=='READER-REF-curriculum-1'
original=load_ref(identity['original_report']);issue=node(original,identity['json_pointer'])
disposition=node(load_ref(identity['disposition_ref']),identity['disposition_ref']['json_pointer'])
fix=node(load_ref(identity['historical_fix_verification']['report']),identity['historical_fix_verification']['json_pointer'])
assert disposition['final_necessity']==identity['original_necessity']=='optional' and issue['necessary'] is False
assert disposition['issue_id']==identity['issue_id'] and disposition['action']=='fixed'
pointer=load_ref(packet['prior_active_pointer'])
manifest=load_ref({'path':pointer['manifest'],'sha256':pointer['manifest_sha256']})
freeze=load_ref(manifest['source_freeze']);prior_page=next(p for p in freeze['pages'] if p['page_id']=='curriculum')
prior_source=(ROOT/prior_page['snapshot']).read_bytes();assert sha(prior_source)==prior_page['source_sha256']==fix['source_sha256']
fixed_clause='BPE（位元組對編碼，[6.2](../course/chapters/06.md#6.2)）可回讀'
assert fixed_clause in prior_source.decode()
current=(ROOT/'docs/curriculum.md').read_bytes();lines=current.decode().splitlines()
hits=[(n,line) for n,line in enumerate(lines,1) if fixed_clause in line]
assert len(hits)==1 and hits[0][0]==17
target=ROOT/'course/chapters/06.md';text=target.read_text();heads=list(re.finditer(r'^## ([A-Z\d]+\.\d+) (.+)$',text,re.M));position=next(i for i,h in enumerate(heads) if h[1]=='6.2')
section=text[heads[position].start():heads[position+1].start()]
assert '## 6.2 常見片段怎麼合併？' in section
assert '位元組對編碼' in section and 'Byte Pair Encoding，BPE' in section
assert '數相鄰對、合併最常見的一對' in section
unavailability=node(load_ref(packet['unavailability']['report']),packet['unavailability']['json_pointer'])
assert unavailability['reviewer_task']=='/root/rv1019_reader_ref' and unavailability['unavailable'] is True
result={
    'created_at':datetime.now(UTC).isoformat(),'reviewer_task':'/root/c_v3_technical_coord/canonical_curriculum','packet':packet_ref,
    'environment':{'python':sys.version,'platform':platform.platform(),'device':'CPU; JSON/hash/link inspection only'},
    'command':'UV_PROJECT_ENVIRONMENT=/workspace/tiny-perceptron-vlm/.venv PYTHONPATH=/tmp/tpb UV_CACHE_DIR=/tmp/uv-c3 UV_OFFLINE=1 uv run --extra cpu --no-sync python docs/technical-reviews/artifacts/c-v3-v21-canonical-curriculum/historical-bpe-inspection.py',
    'issue_identity':{k:identity[k] for k in ['issue_id','disposition_ref','original_report','json_pointer','historical_fix_verification']},
    'original_issue':issue,'accepted_final_necessity':disposition['final_necessity'],'original_disposition_reason':disposition['reason'],
    'original_fix_evidence':{'report':identity['historical_fix_verification']['report'],'json_pointer':identity['historical_fix_verification']['json_pointer'],'source_sha256':fix['source_sha256'],'frozen_source':{'path':prior_page['snapshot'],'sha256':prior_page['source_sha256']},'relevant_original_fix_check':next(c for c in fix['checks'] if c['concept']=='共同基礎與BPE回查入口'),'fixed_clause_in_original_frozen_source':fixed_clause},
    'current_source_sha256':sha(current),'current_figure_sha256':{'course/figures/curriculum_learning_flow.svg':sha((ROOT/'course/figures/curriculum_learning_flow.svg').read_bytes())},
    'current_inspection':{'line_number':17,'line_text':hits[0][1],'resolved_target':'course/chapters/06.md#6.2','target_file_sha256':sha(target.read_bytes()),'target_section_sha256':sha(section.encode()),'target_heading':heads[position][0],'target_definition_excerpt':section.splitlines()[4],'finding':'Current name expansion and direct6.2link preserve the original fix. The target starts with a concrete common-pair merge and explicitly namesByte Pair Encoding; no algorithm retraining or full6.2technical review was needed for this optional navigation concern.'},
    'prior_active_pointer':packet['prior_active_pointer'],'unavailability':packet['unavailability'],'historical_unavailability_recorded_at':unavailability['recorded_at'],
    'necessity':'optional','status':'still_fixed','verdict':'pass',
    'reason':'原問題是首次BPE縮寫無名稱／無回讀節號，屬optional路徑改善。本人核原issue與固定原修正文，再讀當前L17及真6.2目標：中文名稱和直鏈仍在，目標以常見相鄰片段合併具體介紹BPE。這個有界修正仍成立，不以舊reader pass或先前still_fixed作現在證據。',
    'scope':'OnlyREADER-REF-curriculum-1; preserve prior20261019pointer, originaloptionalgrade and historicalunavailabilityevent. No new unavailability, reader study, research rerun, training, publication or source edit.',
}
with (OUT/'historical-child-01-group-00-inspection.json').open('x') as target:target.write(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'created_at':result['created_at'],'issue_id':identity['issue_id'],'necessity':'optional','status':'still_fixed','verdict':'pass','current_line':17,'target_heading':heads[position][0]},ensure_ascii=False,indent=2))
