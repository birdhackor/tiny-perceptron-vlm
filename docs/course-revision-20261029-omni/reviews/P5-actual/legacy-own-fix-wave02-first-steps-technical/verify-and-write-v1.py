"""Same-owner bounded navigation-fix callback; no model or peer-review input."""
import hashlib
import json
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/workspace/work/omni-book-integrated-v1').resolve()
HERE = Path(__file__).resolve().parent
assert Path.cwd().resolve() == ROOT
BASE = Path('docs/course-revision-20261029-omni/reviews')
OWNER = '/root/omni_wave02_technical_first_steps'
SOURCE_HASH = '42824e38f493b977852c2b9d49d0482ef157d2627116a7ddd89d45642e5773d8'
EXPECTED_FIGURES = {
    'course/figures/rewrite-W-3-axes.svg':'74c93bfd89b0b164573f8dea1506179982c79250b77bfc862102895b02387138',
    'course/figures/rewrite-W-6-gradient-step.svg':'e81b9f6de2af7b7d9d8bc9cbe32528794d207d1817014c46624d1619820ca942',
}
EXPECTED_ORIGINALS = [
    ('technical-original-v1.json','75b8eb0ad2005de671e5e1cd5c09e648db0901669d79683c940ff0c635c6b043'),
    ('technical-current-v1.json','14c211835ecdd90c8b9064469d66cc9c06781210a5686a3ff4b257a7477f5170'),
]
started_at = datetime.now(timezone.utc).isoformat()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def reference(path):
    path = (ROOT/path).resolve() if not path.is_absolute() else path.resolve()
    assert path.is_relative_to(ROOT)
    return {'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path.read_bytes())}


def checked_json(path, expected=None):
    raw = (ROOT/path).read_bytes()
    if expected:
        assert sha(raw)==expected, str(path)
    return json.loads(raw), reference(path)


original_reports=[]
for filename,digest in EXPECTED_ORIGINALS:
    path=BASE/'wave-02/omni_wave02_technical_first_steps'/filename
    report,ref=checked_json(path,digest)
    assert report['reviewer_task']==OWNER and report['verdict']=='revise'
    issue_position,issue=next((i,x) for i,x in enumerate(report['issues']) if x.get('id')=='FS-T-OPT-001')
    claim_position,claim=next((i,x) for i,x in enumerate(report['claims']) if x.get('id')=='FS-C054')
    assert issue['claim_id']=='FS-C054' and issue['necessity']=='optional'
    assert issue['status']=='unresolved' and claim['status']=='contradicted'
    original_reports.append({'report':report,'ref':ref,'issue':issue,'claim':claim,
        'issue_pointer':'/issues/'+str(issue_position),'claim_pointer':'/claims/'+str(claim_position)})
assert original_reports[0]['issue']==original_reports[1]['issue']
assert original_reports[0]['claim']==original_reports[1]['claim']
old_author,author_ref=checked_json(BASE/'wave-02/author-first-steps-index-optional-disposition-v1.json',
    '97dd306519f4c79fc6b7a7d79a70e5c8d0c1b2c72944a58a4bae3c13bf2a7130')
decision=next(d for d in old_author['items'] if d.get('issue_id')=='FS-T-OPT-001')
assert decision['action']=='defer' and decision['final_necessity']=='optional' and decision['implemented'] is False
assert decision['report']==original_reports[0]['ref']

freeze,freeze_ref=checked_json(BASE/'P5-actual/source-freeze-v11.json',
    'd46a11ab7e27275f7e6908b8df8e40ed451a21ecc4fd6fedac02c4e01d0df7df')
page=next(p for p in freeze['pages'] if p['page_id']=='first-steps')
assert page['selector']=='whole-file' and page['source_sha256']==SOURCE_HASH
assert page['figures_sha256']==EXPECTED_FIGURES
snapshot_raw=(ROOT/page['snapshot']).read_bytes()
source_raw=(ROOT/page['source']).read_bytes()
assert sha(snapshot_raw)==sha(source_raw)==SOURCE_HASH
body=snapshot_raw.decode('utf-8')
lines=body.splitlines()
criterion=original_reports[0]['issue']['original_quote']
criterion_line=next(i for i,line in enumerate(lines,1) if line==criterion)
row_line,row=next((i,line) for i,line in enumerate(lines,1)
    if line.startswith('| `tiny_perceptron/lesson_products/` |'))
row_matches=list(re.finditer(r'\[([\d]+\.[\d]+)\]\(([^)#]+)#([\d]+\.[\d]+)\)(?:–([\d]+\.[\d]+))?',row))
expanded=[]
links=[]
for match in row_matches:
    assert match[1]==match[3]
    ids=[match[1]] if not match[4] else [
        match[1].rsplit('.',1)[0]+'.'+str(i)
        for i in range(int(match[1].rsplit('.',1)[1]),int(match[4].rsplit('.',1)[1])+1)]
    target=(ROOT/page['source']).parent/match[2]
    target_text=target.read_text()
    for identifier in ids:
        heading=re.search(r'^## '+re.escape(identifier)+r' .+$',target_text,re.M)
        assert heading, identifier
        links.append({'lesson_id':identifier,'target_source':target.relative_to(ROOT).as_posix(),
                      'heading':heading[0]})
    expanded.extend(ids)
assert expanded==['14.15','16.15','19.14','19.15','19.16','19.17']
assert any(m[1]=='16.15' and m[2]=='chapters/16.md' and m[3]=='16.15' for m in row_matches)

support=next(p for p in freeze['pages'] if p['page_id']=='16.15')
support_raw=(ROOT/support['snapshot']).read_bytes()
support_body=support_raw.decode('utf-8')
live_chapter=(ROOT/support['source']).read_text()
live_section=re.search(r'^## 16\.15 .*?(?=^## |\Z)',live_chapter,re.M|re.S)
assert live_section and live_section[0].encode('utf-8')==support_raw
assert sha(support_raw)==support['source_sha256']
support_lines=support_body.splitlines()
old_c_line,old_c_quote=next((i,line) for i,line in enumerate(support_lines,1) if '舊版C把材料' in line)
source_line,source_quote=next((i,line) for i,line in enumerate(support_lines,1)
    if 'CandidateScorer' in line and 'Block' in line and 'tiny_perceptron/lesson_products/model.py' in line)
assert '雙向注意力' in old_c_quote
figure_bindings=[]
for name,digest in EXPECTED_FIGURES.items():
    frozen=freeze['figure_snapshots'][name]
    assert frozen['sha256']==digest
    assert sha((ROOT/name).read_bytes())==sha((ROOT/frozen['path']).read_bytes())==digest
    figure_bindings.append({'path':name,'sha256':digest,'snapshot':frozen,'hash_match':True,
                            'visually_reinspected_in_this_callback':False})

instances=[]
first_instance={'report':original_reports[0]['ref'],'json_pointer':original_reports[0]['issue_pointer']}
for position,item in enumerate(original_reports):
    instances.append({
        'report':item['ref'],'json_pointer':item['issue_pointer'],'claim_json_pointer':item['claim_pointer'],
        'reviewer_task':OWNER,'issue_id':'FS-T-OPT-001','claim_id':'FS-C054',
        'old_created_at':item['report']['created_at'],'old_report_verdict_preserved':'revise',
        'old_issue_status_preserved':'unresolved','old_claim_status_preserved':'contradicted',
        'reported_necessity':'optional','final_necessity':'optional','severity_preserved':'選讀改善',
        'duplicate_group_id':'FS-T-OPT-001','is_duplicate_instance':position==1,
        'duplicate_of':first_instance if position==1 else None,
        'new_independent_finding':False,
        'status':'resolved_for_the_specific_navigation_omission_in_source_freeze_v11',
        'verification_page_id':'first-steps',
        'resolution':'W.7的lesson_products列已在14.15後加入可定位的16.15連結，並保留19.14–19.17。16.15仍說明舊C的雙向CandidateScorer/Block並連至lesson_products/model.py，符合原表前的納入準則。此結論只收束同一已知漏鏈；原報告及其revise/contradicted/unresolved bytes不改。',
    })

scope='只核FS-T-OPT-001（FS-C054）所指W.7導航補鏈，不代表102claims或新整頁blindpass。'
document={
    'schema_version':1,
    'kind':'same_original_owner_bounded_old_fix_callback_not_fresh_whole_page_review',
    'review_stage':'technical','reviewer_task':OWNER,
    'reviewer_context':'same_owner_bounded_post_original_callback',
    'started_at':started_at,'created_at':datetime.now(timezone.utc).isoformat(),
    'workspace_root':str(ROOT),'source_freeze':freeze_ref,'source_freeze_version':11,
    'verdict':'pass','verdict_scope':scope,
    'old_author_disposition':{'report':author_ref,'json_pointer':'/items/0','issue_id':'FS-T-OPT-001',
        'old_action_preserved':'defer','old_implemented_value_preserved':False,
        'interpretation':'舊處置是在source15尚未修改時排入P5；本次由原technical owner親讀新source11，獨立確認該最小修正已存在。沒有回寫舊作者紀錄。'},
    'verification_pages':[{
        'page_id':'first-steps','source':'course/first-steps.md','selector':'whole-file',
        'source_sha256':SOURCE_HASH,'figures_sha256':EXPECTED_FIGURES,'verdict':'pass',
        'verdict_scope':scope,'issue_ids':['FS-T-OPT-001'],'claim_ids':['FS-C054'],
        'snapshot':{'path':page['snapshot'],'sha256':SOURCE_HASH},
        'read_scope':'W.7表前納入準則、lesson_products一列，以及16.15必要的舊C說明/來源；其他列只核此次保留的連結目標標題。',
        'criterion':{'line':criterion_line,'quote':criterion,'unchanged_from_own_original_issue':True},
        'fixed_row':{'line':row_line,'quote':row,'expanded_lesson_ids':expanded,
                     'new_16_15_link':'chapters/16.md#16.15','link_targets_checked':links},
        'reason':'補鏈位置、目標與原minimum_fix一致；讀者能由同一列找到舊C與新共享記憶架構差異的既有說明，無新增概念或程式負擔。',
        'not_reasserted':'未重新判定所有符合準則小節的全局完整性，也未把原FS-C054的廣義敘述重簽為整頁verified。',
    }],
    'supporting_source_inspections':[{
        'page_id':'16.15','source':support['source'],'selector':support['selector'],
        'source_sha256':support['source_sha256'],
        'snapshot':{'path':support['snapshot'],'sha256':support['source_sha256']},
        'live_section_matches_snapshot':True,
        'inspected_passages':[{'snapshot_line':old_c_line,'quote':old_c_quote},
                              {'snapshot_line':source_line,'quote':source_quote}],
        'support_scope':'僅確認新導航目標確有符合原納入準則的零件說明及來源；不是16.15方法/實驗或P5模型實作驗收。',
    }],
    'legacy_issue_instances':instances,
    'duplicate_accounting':{'original_instances':2,'unique_old_findings':1,
        'duplicate_group_id':'FS-T-OPT-001','necessity_preserved_for_both':'optional',
        'note':'current-v1保留的是原判同一問題的重複實例，不當第二項獨立發現；兩個來源SHA及issue pointer都個別保留。'},
    'checks':{
        'original_identity_and_bytes':{'status':'pass','details':'兩份自己報告與舊作者處置SHA均符合派工；owner、optional、revise、unresolved及contradicted舊值一致。'},
        'current_source_binding':{'status':'pass','details':'source-freeze-v11、first-steps snapshot與工作樹UTF-8 bytes一致，source_sha256為42824e38...。'},
        'specific_fix':{'status':'pass','details':'criterion未改；該列依序涵蓋14.15、16.15、19.14–19.17，16.15連結到正確章檔/錨點。'},
        'target_suitability':{'status':'pass','details':'16.15仍有舊C雙向混合的說明，來源點名CandidateScorer/Block並指向lesson_products/model.py。'},
        'necessity_and_duplication':{'status':'pass','details':'逐兩個原instances保留optional與同一問題的duplicate關係；不升級成必要或兩項新發現。'},
        'figure_binding_only':{'status':'pass','details':'兩圖SHA與本頁freeze map及圖快照精確一致；這次沒有做新視覺審閱，導航修正不依賴圖變更。'},
    },
    'figure_bindings':figure_bindings,
    'actual_methods':['親讀自己的兩個原issue/FS-C054及舊作者defer處置','親讀source11的criterion、修後一列與必要16.15原位說明/來源','以標準函式庫核UTF-8 SHA、精確freeze snapshot切片、展開range及Markdown錨點標題'],
    'command':'uv run --offline --no-sync --extra cpu python '+str(Path(__file__).resolve()),
    'verification_script':reference(Path(__file__).resolve()),
    'environment':{'python':platform.python_version(),'executable':sys.executable,'platform':platform.platform()},
    'boundaries':{
        'fresh_blind_reading':False,'whole_page_resigning':False,'claims_102_revalidated':False,
        'new_p5_peer_passes_or_reader_coherence_answers_read':False,
        'new_visual_inspection':False,'neural_network_execution':False,
        'model_training_or_inference':False,'paid_operations':False,'workflow_triggered':False,
        'curriculum_checker_or_reading_time_modified':False,'old_reports_modified':False,
        'note':scope+' 無NN/付費/workflow；教材、checker、估時與原件均不修改。',
    },
}
for filename,digest in EXPECTED_ORIGINALS:
    assert sha((ROOT/BASE/'wave-02/omni_wave02_technical_first_steps'/filename).read_bytes())==digest
assert sha((ROOT/page['source']).read_bytes())==SOURCE_HASH
target=HERE/'native-verification-v1.json'
with target.open('x',encoding='utf-8') as stream:
    stream.write(json.dumps(document,ensure_ascii=False,indent=2)+'\n')
saved=json.loads(target.read_bytes())
assert saved['reviewer_task']==OWNER and saved['source_freeze']==freeze_ref
assert len(saved['legacy_issue_instances'])==2
node=saved['verification_pages'][0]
assert node['page_id']=='first-steps' and node['source_sha256']==SOURCE_HASH
assert node['figures_sha256']==EXPECTED_FIGURES and node['verdict']=='pass'
assert all(i['reported_necessity']==i['final_necessity']=='optional' for i in saved['legacy_issue_instances'])
print(json.dumps({'path':str(target),'relative_path':target.relative_to(ROOT).as_posix(),
    'bytes':target.stat().st_size,'sha256':sha(target.read_bytes()),'created_at':saved['created_at'],
    'verdict':saved['verdict'],'scope':saved['verdict_scope'],'original_instances':2,
    'unique_old_findings':1,'old_report_hashes_unchanged':True},ensure_ascii=False))
