"""Package three completed, genuinely sequential same-reader source22 rereads."""
import hashlib
import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'docs/course-revision-20261101-c-v3/readability-entries-v1'
TASK = '/root/c_v3_reader_entries'
WAVE = 'c-v3-integration-07'


def read(path):
    p = Path(path)
    return json.loads((p if p.is_absolute() else ROOT / p).read_text())


def ref(path):
    p = Path(path)
    p = p if p.is_absolute() else ROOT / p
    return {'path': p.relative_to(ROOT).as_posix(),
            'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


def now():
    return datetime.now(UTC).isoformat()


def write_new(name, value):
    p = OUT / name
    with p.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return ref(p)


freeze_ref = ref('docs/course-revision-20261101-c-v3/source-freeze-v22.json')
assert freeze_ref['sha256'] == '0f4e9d63bfcb2243234c17c62242ff79cd4ab640a21d1f313f94c99946a93a59'
freeze = read(freeze_ref['path'])
frozen = {p['page_id']: p for p in freeze['pages']}
plan_ref = ref('docs/course-revision-20261101-c-v3/reader-source22-trace-reading-plan-v1.json')
assert plan_ref['sha256'] == '606e16befe755768978f98442ee331cea59636c429781ba2ca003a7a7fef45fe'
plan = read(plan_ref['path'])
assert plan['reviewer_task'] == TASK and plan['author_task'] == '/root'
assert plan['main_pages'] == ['readme','training','curriculum']
assert plan['source_freeze'] == freeze_ref

prior_names = {
    'original': ('reader-original-v1.json', '27115271a2313bfc55b4ce146e961c660ef222841517f26043e990e5c6a3a239'),
    'basis': ('reader-basis-source21-v1.json', 'f91d334245bd0b257bbd63b23a58f18f00368ad4be177c3a54e5b7db38a045c3'),
    'current': ('reader-current-source21-v1.json', '1f1411f66094162ea4fa2171e7d8ef2f62e2a5ebefe1b82bcd1533458d21062c'),
    'closure': ('reader-checkpoint-closure-source21-v1.json', '46f659c8a9526d7e4511cb7f2ec6734b886c3f5d84e3cd3601554a7c69eb0c43'),
    'reading_times': ('reading-times-v2.json', 'a14cd2d9058c51440a652caa0b68442a5ace371ec20f708abbededefbfcfbdb4'),
    'folds': ('folded-supplement-report-v1.json', 'ef5b5badd3c7dfff68ee585a6e874aa77fb80202e6dad8b70ecb0986612f97d2'),
    'visual19': ('visual-addendum-v1.json', '2045a5f1739e790c7bde79d584c35762038b337361a065da1b2a6ec98bd12eb2'),
    'visual21': ('visual-addendum-source21-v1.json', '47ea662bcf53dd9f6c539b78dba298db7b5ff0fdf1012c4d7e1b654dcc7c505f'),
}
prior = {}
for key, (name, digest) in prior_names.items():
    prior[key] = ref(OUT / name)
    assert prior[key]['sha256'] == digest, (key, 'prior artifact changed')
original = read(prior['original']['path'])
old_current = read(prior['current']['path'])
old_pages = {p['page_id']: p for p in old_current['pages']}
original_pages = {p['page_id']: (i,p) for i,p in enumerate(original['pages'])}
estimates = {p['page_id']: p for p in read(prior['reading_times']['path'])['pages']}

spec = importlib.util.spec_from_file_location('source22_reader_native_check', ROOT / 'docs/review-tools/incremental_revision.py')
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)
session_ids = {
    'readme': '456f9e7b86c14964a6a15fe87872d4d0',
    'training': '94fb59ac4ee84453b7907f19646b31cd',
    'curriculum': 'c3a9ee856f034293b11180446e51317a',
}
counts = {'readme':25, 'training':73, 'curriculum':18}
summary = {
    'readme': '重讀完整25單位後，接字問題到課程/成品範圍、CPU環境/最小通路、推論與完整續訓、維護模式可按原文分辨。332/358、torch/kernel、Auto/final與B角色在當前來源仍清楚。未新增必要問題。',
    'training': '重讀完整73單位後，材料/家族、前反與真更新、父權重與resume、手動與固定配方、各題分母/時點、生成與排序、量化保存/品質、教師配套與C-v3預設例均有具體notes。沒有把列指令當真執行或把同總分冒同題保留。未新增必要問題。',
    'curriculum': '重讀完整18單位後，大綱設計/來源索引而非第二正式目錄的定位可達，三目標/資源分工、X/Y及兩遮罩、局部證據邊界可說明。順讀終點再真讀為19.1–19.25，與推薦C-v3相接。未新增必要問題。',
}
estimate_reason = {
    'readme': '本次25單位重讀確認範圍未變：須理解導覽/分母、安裝與兩種模式、loss/反向不更新、下載/推論/續訓界線；同SHA字ID圖已由本人舊視覺實看。保留完整頁18–28分鐘估計，非本次已知本文的快速耗時。',
    'training': '本次73單位重讀確認手動/固定配方、資料/父權重、各報告時點與分母、生成/停止、量化和教師配套仍需停讀；完整inventory另含本人先前真補讀的9折疊，source/fig與21同，保留85–125分鐘。此trace未把折疊重新開讀，亦不將已知答案的重讀速度替代首次完整估時。',
    'curriculum': '本次18單位重讀確認設計取捨表、方法角色/理由、資料流圖X/Y與兩遮罩、來源限制仍是完整頁閱讀工作；同SHA流程圖原已實看。保留30–45分鐘，不將本次已知本文速度、外部原碼或補讀算法算入。',
}

sessions = {}
pages = []
audits = []
latest = None
total_notes = 0
for pid in plan['main_pages']:
    fp = frozen[pid]
    folder = ROOT / 'docs/course-revision-20261016/reader-sessions' / session_ids[pid]
    opening_ref = ref(folder / 'opening.json')
    state_ref = ref(folder / 'state.json')
    opening = read(opening_ref['path'])
    state = read(state_ref['path'])
    assert state['next_index'] == counts[pid]
    assert opening['reading_plan'] == state['reading_plan'] == plan_ref
    assert opening['source_freeze'] == state['source_freeze'] == freeze_ref
    assert opening['source_sha256'] == fp['source_sha256'] == old_pages[pid]['source_sha256']
    assert opening['figures_sha256'] == fp['figures_sha256'] == old_pages[pid]['figures_sha256']
    source_ref = {'path': fp['snapshot'], 'sha256': fp['source_sha256']}
    assert ref(fp['snapshot']) == source_ref
    figures = {name: freeze['figure_snapshots'][name] for name in fp['figures_sha256']}
    for r in figures.values():
        assert ref(r['path']) == r
    note_refs = [ref(folder/f'note-{i:03d}.json') for i in range(counts[pid])]
    for r in note_refs:
        n = read(r['path'])
        assert n['reading_kind'] == 'same_reader_current_source_reread_not_blind'
        assert n['collection_role'] == 'technical' and n['wave'] == WAVE
        assert not n['issues']
    session = {
        'wave': WAVE, 'collection_role': 'technical',
        'session_id': session_ids[pid], 'opening': opening_ref,
        'source': source_ref, 'figures': figures, 'notes': note_refs,
        'state': state_ref, 'full_state': state_ref,
    }
    observed_pid, completed = native.reader_session(ROOT, session, TASK)
    assert observed_pid == pid
    assert datetime.fromisoformat(plan['created_at']) <= datetime.fromisoformat(opening['started_at']) <= completed
    assert completed <= datetime.now(UTC)
    latest = completed if latest is None else max(latest, completed)
    total_notes += len(note_refs)
    sessions[pid] = session
    audits.append({'page_id':pid,'validator':'incremental_revision.reader_session',
                   'result':'pass','unit_count':len(note_refs),'completed_at':completed.isoformat(),
                   'complete_state':state_ref})
    estimate = dict(estimates[pid])
    assert estimate['source_sha256'] == fp['source_sha256']
    assert estimate['figures_sha256'] == fp['figures_sha256']
    estimate['reason'] = estimate_reason[pid] + ' 排除安裝、下載、執行/訓練等待與範圍外前置補讀；依理解工作估計，未用字數公式。'
    old_index, old_original = original_pages[pid]
    pages.append({
        'page_id':pid,'source':fp['source'],'selector':fp['selector'],
        'reviewer_task':TASK,'source_sha256':fp['source_sha256'],
        'figures_sha256':fp['figures_sha256'],'verdict':'pass',
        'source_freeze':freeze_ref,'reading_plan':plan_ref,
        'reading_kind':'same_reader_current_source_reread_not_blind',
        'completed_at':completed.isoformat(),'reader_summary':summary[pid],
        'native_reader_trace':session,'sessions':[session],
        'notes_basis':note_refs,'unit_count':len(note_refs),
        'original_first_read_report':prior['original'],
        'original_json_pointer':f'/pages/{old_index}',
        'original_sessions_preserved':old_original['sessions'],
        'prior_source21_judgment':prior['current'],
        'ai_estimate':estimate,'estimate_range_preserved_from':prior['reading_times'],
        'folded_reading_scope': '本次只按native units重讀主文/書目；全部原folds是本人既有補讀，source22本頁與21同SHA，沿用原證據，不聲稱本次重開。',
        'existing_fold_evidence':prior['folds'],
        'visual_scope':'本次未新view/拍圖，未更動圖；原實看視覺證據依同SHA沿用。未驗設備、Colab、全部頁面像素。',
        'existing_visual_evidence':[prior['visual19'],prior['visual21']],
        'issues':[],
        'limits':'只判當前頁承諾的導覽/操作/設計/證據理解；未跑課程/模型/NN/原驗收、未重新研究/設計頁面，AI同讀者重讀非真人測試。',
    })
assert total_notes == 116
created = now()
assert datetime.fromisoformat(created) >= latest
basis = {
    'schema_version':1,'kind':'completed_current_source_reader_trace_basis',
    'role':'reader','reviewer_task':TASK,'created_at':created,
    'wave':WAVE,'collection_role':'technical',
    'source_freeze':freeze_ref,'reading_plan':plan_ref,'pages':pages,
    'counts':{'pages':3,'new_native_main_units':116},
    'reading_identity':'同一原讀者已有source19/21與後補折疊/視覺知識，這是真逐unit current-source重讀，非新的blind首次閱讀或新R phase。',
    'purpose':'補completed current-source native trace；各unit先當場保存觀察/四問才由既有reader.py開下一unit。沒有從舊notes換unit_hash補造。',
    'native_trace_validation':audits,
    'prior_artifacts_unchanged':prior,
    'issues':[],
    'collection_boundary':'只交basis待root真解析收T07；未建立正式current/closure、未代填received_at/未來seal，不回填已closed的R07。collection_role technical是實際收件窗，非冒充技術正確性角色。',
    'reading_time_policy':'三頁保留既有當前來源相同range，用本次理解重新說明理由；沒有更改reading-times檔案或全書估時。',
}
basis_ref = write_new('reader-source22-trace-basis-v1.json', basis)
handoff = {
    'schema_version':1,'kind':'source22_same_reader_trace_handoff',
    'reviewer_task':TASK,'created_at':now(),'wave':WAVE,
    'actual_collection_role':'technical','basis_report':basis_ref,
    'source_freeze':freeze_ref,'reading_plan':plan_ref,
    'new_sessions':[sessions[pid] for pid in plan['main_pages']],
    'pages':[{'page_id':pid,'basis_report':basis_ref,'basis_json_pointer':f'/pages/{i}',
              'new_current_source_session':sessions[pid],
              'original_sessions_preserved':original_pages[pid][1]['sessions']}
             for i,pid in enumerate(plan['main_pages'])],
    'all_original_seven_page_reports_unchanged':{
        k:prior[k] for k in ('original','basis','current','closure')},
    'helper':ref(Path(__file__).with_name('record_checkpoint.py')),
    'packaging_helper':ref(__file__),
    'native_validation':audits,
    'unit_count':total_notes,'issues':[],
    'dispatch_state':'basis及3真trace已完成，等待root真收。正式current/closure留待後續dispatch，未製造收件/phase seal。',
}
handoff_ref = write_new('source22-trace-handoff-v1.json', handoff)
for k,(name,digest) in prior_names.items():
    assert ref(OUT/name)['sha256'] == digest
for r in (basis_ref,handoff_ref):
    print(json.dumps(r,ensure_ascii=False))
print(json.dumps({'native_trace_validation':audits,'total_notes':total_notes},ensure_ascii=False))
