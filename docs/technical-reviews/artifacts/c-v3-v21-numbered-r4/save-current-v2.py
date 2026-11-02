"""Same-owner public delta judgment and native validation; no network/NN here."""
import copy
import hashlib
import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path

ROOT=Path('/tmp/tpb').resolve()
BASE=Path('docs/technical-reviews/artifacts/c-v3-v21-numbered-r4')
REPORT_BASE=Path('docs/course-revision-20261101-c-v3/technical-v21/numbered_r4')
ORIGINAL=REPORT_BASE/'original-v1.json'
CURRENT1=REPORT_BASE/'current-v1.json'
CURRENT2=REPORT_BASE/'current-v2.json'
TRANSPORT=BASE/'public-transport-current-v2.json'
NATIVE=BASE/'native-validate-current-v2.json'
ORIGINAL_SHA='3ee1c43932862165c1b2f58a48d10e22eeaa5631628eab379f53e130aa638092'
CURRENT1_SHA='5e634f883ea30e90f5698bde14c44a4fa11d47b8224806dccb131c6fa7d43237'
TRANSPORT_SHA='c7ed6f7e43b06fa328f78d19cd2bc028cd7d01cb43b137fa2407b45daf0e3d32'
COMMAND='UV_PROJECT_ENVIRONMENT=/workspace/tiny-perceptron-vlm/.venv PYTHONPATH=/tmp/tpb UV_CACHE_DIR=/tmp/uv-c3 UV_OFFLINE=1 uv run --extra cpu --no-sync python docs/technical-reviews/artifacts/c-v3-v21-numbered-r4/save-current-v2.py'


def sha(path):
    return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()


assert not (ROOT/CURRENT2).exists(), 'Never overwrite a saved judgment'
assert sha(ORIGINAL)==ORIGINAL_SHA and sha(CURRENT1)==CURRENT1_SHA and sha(TRANSPORT)==TRANSPORT_SHA
original=json.loads((ROOT/ORIGINAL).read_text())
prior=json.loads((ROOT/CURRENT1).read_text())
transport=json.loads((ROOT/TRANSPORT).read_text())
assert transport['verified'] and transport['transport_verified'] and transport['archive_members_verified']
assert transport['http_status']==200 and transport['observed_bytes']==6168350
assert transport['observed_sha256']=='552d64b788291ce19157680f51dfc6bbc5394fb6bed7a22fcaa11f2ad3615d88'
assert transport['request']['logical_get_attempts']==1 and not transport['request']['authorization_header_sent']
assert transport['manifest_member_count']==83 and transport['implementation_files_byte_identical']==17
assert len(prior['claims'])==18 and original['verdict']==prior['verdict']=='revise'
assert len(original['issues'])==1 and original['issues'][0]['status']=='unresolved'

report=copy.deepcopy(prior)
report['created_at']=datetime.now(UTC).isoformat()
report['verdict']='pass'
report['review_mode']='same-owner public-transport delta callback; not a new fresh original review'
report['prior_current_report']={'path':str(CURRENT1),'sha256':CURRENT1_SHA}
report['public_delta']={
    'fixed_public_url':transport['request']['url'],
    'personally_verified_at':transport['finished_at'],
    'receipt':{'path':str(TRANSPORT),'sha256':TRANSPORT_SHA},
    'http_status':200,'bytes':6168350,'sha256':transport['observed_sha256'],
    'authorization_header_sent':False,
    'archive_manifest_members_verified':83,
    'scope':'只補正式公開之後本人一次匿名GET及ZIP成員驗證；既有本機資格、三份CPU原例與來源按身分保留。'
}
report['current_scope']='正式同owner公開delta。source21/live R.4與33份repository來源、9項既有artifact及兩份真收件receipt身分保持；本人新增一次無Authorization匿名GET，HTTP200/6168350 bytes/固定SHA及83原始ZIP成員全吻合。除傳輸與hash外，未重跑NN、訓練、final、舊研究、Modal或整套檢查。'
report['independence_statement']='原判由本fresh reviewer完成，original與current-v1兩次真收件及當時必要HTTP404/revise均原樣保留。此為同一owner的正式公開delta callback，不是新fresh原判；root公開事件僅作啟動線索，本人另以無Authorization GET實核固定URL。其餘內容只核既有immutable證據身分，未以作者pass替代本人新事件。'
report['review_scope']='只更新R.4下載導覽的公開取得前置；完整18項主張與原始來源/證據仍保留。無圖。已核tmpfs遷移身分；未改教材、Notebook、17產品py、舊artifact、DP、active或release。未訓練、重跑NN/final或使用Modal。此pass只屬R.4，未宣告fullT/C。'
report['retention']['public_transport_chronology']='17:07:19原匿名GET為必要HTTP404；original及current-v1保持revise不動。16:27 acquisition的public_transport_verified=false是當時待公開的真收據，其bytes未改。正式公開之後，本人在22:16:55–22:16:58的一次新匿名GET核得200及相同固定ZIP；只由此新事件閉合取得前置。'
for source in report['sources']:
    if source['id']=='s_C3_acquisition':
        source['inspection_note']+=' current-v2補充：created_at=2026-11-01T16:27:03.731565+00:00的public_transport_verified=false為歷史狀態，不改舊檔；現在以本人22:16:58的新公開GET收據核現在的可取得性。'
report['sources'].append({'id':'s_transport_v2','kind':'execution','title':'同owner正式公開後本人一次匿名GET與全部原始ZIP成員核驗','verified':True,'artifact_id':'a_transport_v2'})
report['artifacts'].append({'id':'a_transport_v2','kind':'execution','path':str(TRANSPORT),'sha256':TRANSPORT_SHA,'description':'本人同owner公開delta：一次無Authorization匿名GET，實得200/固定bytes/SHA並於記憶體核83個MANIFEST成員、四推論檔、八metadata與17產品py；不保存ZIP/權重，不記signed redirect query。','command':transport['command'],'result':transport['result'],'environment':transport['environment']})
report['artifacts'].append({'id':'a_transport_v2_code','kind':'code','path':str(BASE/'verify-public-current-v2.py'),'sha256':sha(BASE/'verify-public-current-v2.py'),'description':'一次匿名GET與記憶體ZIP核驗、舊來源/證據身分核對程式；沒有NN/訓練/final或發版操作。'})
report['artifacts'].append({'id':'a_current_v2_code','kind':'code','path':str(BASE/'save-current-v2.py'),'sha256':sha(BASE/'save-current-v2.py'),'description':'同owner依真公開事件形成完整current-v2並呼叫原生_validate；不重跑網路或模型。'})
claim=next(item for item in report['claims'] if item['id']=='c16')
assert claim['status']=='unresolved'
claim['status']='verified'
claim['evidence']=[
    {'source_id':'s_C3_guide','locator':'下載固定配套：download命令與固定Release連結','supports':'本節實際導向的固定公開取得入口'},
    {'source_id':'s_C3_release','locator':'url/bytes/sha256/bundle/implementation','supports':'固定封包及各成員來源身分'},
    {'source_id':'s_C3_acquisition','locator':'2026-11-01T16:27:03.731565+00:00 archive/files/scope','supports':'舊待公開取得紀錄的固定ZIP身分；false保留為歷史狀態，未拿它宣稱現在已公開'},
    {'source_id':'s_transport_v2','locator':'started_at/finished_at/request/http_status/observed_bytes/observed_sha256/manifest_members','supports':'本人正式公開後一次無Authorization匿名GET，HTTP200、固定bytes/SHA及83原成員吻合，真正閉合原取得前置'},
]
claim['artifact_ids']=['a_transport_v2']
claim['verification']={
    'method':'executed',
    'expected':'固定公開URL匿名GET得HTTP200、6168350 bytes、SHA-256 552d64b788291ce19157680f51dfc6bbc5394fb6bed7a22fcaa11f2ad3615d88，原始MANIFEST成員/推論檔/metadata/產品來源吻合。',
    'observed':'2026-11-01T22:16:55.596211至22:16:58.737604+00:00，本人一次無Authorization匿名GET得HTTP200、6168350 bytes及完全相同SHA；83個MANIFEST成員、四推論檔、八份metadata與17產品py全核對一致。',
    'details':'使用新build_opener及明列User-Agent/Accept，不提供Authorization或credential參數；redirect也逐次查無Authorization。ZIP只存記憶體核成員，未落盤重複ZIP/權重、不保存signed redirect query。原17:07 HTTP404及兩份revise原判不改；新事件只處理公開transport。',
}
claim['scope']='只建立同一固定版本在本環境當次可匿名取得與原ZIP內容身分；既有本機資格及三CPU原例由immutable證據保留，這次沒有重跑模型或final。'
resolution='真正閉合原必要公開取得阻礙：本人於2026-11-01T22:16:55.596211至22:16:58.737604+00:00，對同一固定公開URL作一次無Authorization匿名GET，得HTTP200、6168350 bytes及SHA552d64b788291ce19157680f51dfc6bbc5394fb6bed7a22fcaa11f2ad3615d88；83原始ZIP成員與四推論檔/八metadata/17產品來源全吻合。原17:07必要HTTP404、original/current-v1的revise與原acquisition=false歷史bytes保持不動。修復scope只為正式公開後的取得前置，沒有重訓、重跑NN/final或以本機包替代匿名事件。'
issue=report['issues'][0]
issue['original_required_resolution']=issue.pop('required_resolution')
issue['details']='原17:07匿名GET的HTTP404當時是必要阻礙，先前原判與current-v1如實revise並已真收件。現在由本人正式公開後的新匿名GET及固定原ZIP成員身分實證閉合。'
issue['status']='resolved'
issue['resolution']=resolution
issue['resolution_artifact_ids']=['a_transport_v2']
report['original_issue_resolutions']=[{'json_pointer':'/issues/0','status':'resolved','resolution':resolution,'artifact_ids':['a_transport_v2']}]
all_ids=[item['id'] for item in report['claims']]
report['checks']['factual_accuracy']={'status':'pass','details':'原有局部機制、系譜、Auto、十工作、原始影音/有限聲音/C2失敗及C3固定資格證據身分保留；唯一必要公開下載前置由本人新匿名200/固定bytes/SHA與原成員核對實證閉合。僅R.4判定。','claim_ids':all_ids}
report['checks']['source_verification']={'status':'pass','details':'source21/live切片、33份repository來源、9項既有artifact及兩原收件receipt身分一致；新匿名公開GET核同一固定ZIP與83成員。原取得metadata中的false保留為16:27歷史狀態，以22:16新事件說明時序；未改舊證據或依作者pass閉合。','claim_ids':all_ids}
report['checks']['limitations']['details']='原局部機制/訓練品質、從零系譜/自家續訓、C2共享機制/品質失敗、有限音調/真人語音、C3typed-device/原文字Auto的界線與各分母保留。current-v2只新增公開transport與member hash，既有資格與三CPU例按身分保留，未重新測量能力；不把本節pass宣告fullT/C。'

assert len(report['claims'])==18 and all(row['status']=='verified' for row in report['claims'])
assert {row['json_pointer'] for row in report['original_issue_resolutions']}=={'/issues/0'}
assert all(row['status']=='resolved' for row in report['issues'])
assert datetime.fromisoformat(report['created_at'])>datetime.fromisoformat(transport['finished_at'])
assert sha(ORIGINAL)==ORIGINAL_SHA and sha(CURRENT1)==CURRENT1_SHA
validator_path=ROOT/'scripts/check_technical_reviews.py'
spec=importlib.util.spec_from_file_location('numbered_r4_native_current_v2',validator_path)
native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
body=(ROOT/'docs/course-revision-20261101-c-v3/numbered-source-freeze-v21/sections/R.4.md').read_text()
errors=native._validate(ROOT,ROOT/'course/README.md','R.4',body,report,{report['reviewer_task']:['R.4']},set())
validation_started_at=datetime.now(UTC).isoformat()
assert errors==[], repr(errors)
(ROOT/CURRENT2).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
result={
    'schema_version':1,'kind':'same-owner native_validate result','reviewer_task':report['reviewer_task'],
    'command':COMMAND,'environment':transport['environment'],
    'report':{'path':str(CURRENT2),'sha256':sha(CURRENT2),'created_at':report['created_at'],'verdict':report['verdict']},
    'validator':{'path':'scripts/check_technical_reviews.py','sha256':sha(Path('scripts/check_technical_reviews.py')),'call':'_validate(ROOT, ROOT/course/README.md, R.4, exact source21 body, current-v2, {same owner:[R.4]}, set())'},
    'errors':errors,'native_validate':'pass',
    'original_closure_pointers':['/issues/0'],
    'original_and_current_v1_unchanged':{'original_sha256':sha(ORIGINAL),'current_v1_sha256':sha(CURRENT1)},
    'validation_started_at':validation_started_at,'validated_at':datetime.now(UTC).isoformat(),
    'scope':'Only this complete R.4 current-v2 native structure/hashes; not full technical phase, C phase, publishing, or automatic semantic proof.'
}
(ROOT/NATIVE).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'report':result['report'],'native_validate':'pass','native_errors':errors,'native_receipt':str(NATIVE),'native_receipt_sha256':sha(NATIVE),'public_transport_receipt_sha256':TRANSPORT_SHA,'scope':'R.4 same-owner public-transport delta only; no fullT/C declaration'},ensure_ascii=False))
