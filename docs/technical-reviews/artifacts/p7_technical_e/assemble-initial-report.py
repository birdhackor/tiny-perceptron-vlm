"""Assemble this owner's already recorded judgments; validate metadata only."""
import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = Path('docs/technical-reviews/artifacts/p7_technical_e')
MANIFEST = Path('docs/course-revision-20261007-phase7/reviews/freeze-03/manifest.json')
TRACE = Path('docs/course-revision-20261007-phase7/reviews/freeze-03/traces/technical/e/7d5f9adc4c8746ffb2443beb78ea74e3.jsonl')
REPORT = Path('docs/technical-reviews/phase7-freeze03-e-p7_technical_e-initial.json')
sys.path.insert(0, str(ROOT / 'scripts'))
import check_technical_reviews as checker


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def read(path):
    return json.loads((ROOT / path).read_text())


def normalize_visual(v):
    if v.get('status') != 'verified':
        return
    if 'tool_receipt' in v:
        v['receipt'] = v.pop('tool_receipt')
    receipt = read(v['receipt']['path'])
    v.setdefault('observation', receipt['observation'])
    for artifact in v['artifacts']:
        if isinstance(artifact.get('viewport'), str):
            artifact['viewport'] = [int(n) for n in artifact['viewport'].split('x')]


def add_original_measurement(page, claim_id, locator):
    source = {
        'id': 'original_measurement', 'kind': 'repository_code',
        'title': '本人已核對的歷史原測量 JSON', 'verified': True,
        'path': str(BASE / 'sources/distillation.json'),
        'sha256': sha(BASE / 'sources/distillation.json'),
        'version': '原執行 JSON，原 revision 5af615e5d7c9642afee800390fa072257f895d0c',
        'inspection_note': '原欄位已由本頁本人實際 CPU probe 核算；新增完整報告的直接原檔綁定，不新增閱讀或改舊判斷。',
    }
    page['sources'].append(source)
    next(c for c in page['claims'] if c['id'] == claim_id)['evidence'].append({
        'source_id': source['id'], 'locator': locator,
        'supports': '該頁已實核的原測量欄位直接原檔/SHA，並非作者解說或新訓練。',
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    manifest = read(MANIFEST)
    group = next(g for g in manifest['groups'] if g['group'] == 'e')
    excerpts = read(BASE / 'sources/p7-paper-excerpt-index.json')
    pages = []
    metadata_errors = []
    for page_id in group['primary_page_ids']:
        page = copy.deepcopy(read(BASE / 'pages' / (page_id + '.json')))
        for v in page['visual_checks'] + [page['page_visual_check']]:
            normalize_visual(v)
        for c in page['claims']:
            if c['kind'] == 'numeric' and not isinstance(c.get('verification', {}).get('tolerance'), str):
                c['verification']['tolerance'] = 'absolute tolerance ' + str(c['verification']['tolerance'])
        for source in page['sources']:
            if source['kind'] != 'paper' or source['url'] not in excerpts:
                continue
            excerpt = excerpts[source['url']]
            source['original_sha256'] = excerpt['original_sha256']
            source['excerpt'] = {'path': excerpt['path'], 'sha256': excerpt['sha256']}
            source['inspection_note'] += ' 完整原 PDF/text 留本機；正式證據僅必要少量摘錄及原 PDF SHA。'
            artifact_id = None
            for artifact in page['artifacts']:
                if artifact['path'].endswith('.pdf') and (
                    ('jacob' in artifact['path'] and source['id'] == 'jacob')
                    or ('gptq' in artifact['path'] and source['id'] == 'gptq')
                ):
                    artifact['path'] = excerpt['path']
                    artifact['sha256'] = excerpt['sha256']
                    artifact['description'] = '本人真核原論文必要少量摘錄、原 URL/version/locator/PDF SHA；完整原檔留本機'
                    artifact_id = artifact['id']
            if artifact_id is None:
                artifact_id = 'original_excerpt_' + source['id']
                page['artifacts'].append({
                    'id': artifact_id, 'kind': 'source_snapshot',
                    'path': excerpt['path'], 'sha256': excerpt['sha256'],
                    'description': '本人已實讀原論文支持範圍的必要少量摘錄與原 PDF SHA',
                })
            for c in page['claims']:
                if any(e['source_id'] == source['id'] for e in c['evidence']) and artifact_id not in c['artifact_ids']:
                    c['artifact_ids'].append(artifact_id)
        if page_id in {'18.4', '18.7', '18.11', '18.12'}:
            locators = {
                '18.4': ('c4', 'results.tasks.attributes.teacher_cache.file_bytes'),
                '18.7': ('c3', 'results.tasks.*.teacher_frozen_and_unchanged/teacher_cache.seconds'),
                '18.11': ('c3', 'results.tasks.style_transfer.teacher_test/runs.*.test.generated_samples: style=json; 2+2=?'),
                '18.12': ('c3', 'results.tasks.moe_to_dense.data/runs.w32_ce/runs.w32_ce_kl.training/test'),
            }
            add_original_measurement(page, *locators[page_id])
        if page_id == '18.7':
            c = next(c for c in page['claims'] if c['id'] == 'c3')
            c['statement'] = '原三任務 teacher_frozen_and_unchanged true，由完整 state_dict（參數及 buffers）名稱/原 dtype CPU bytes hash 前後檢查後記錄。'
            c['verification']['denominators']['hash_scope'] = 'complete state_dict including parameters and buffers; original dtype bytes'
            c['verification']['details'] += ' 本人 checkpoint 101 已澄清早先 all-parameters/float 用語；此處按真正原函式47–56，舊 trace/頁記錄不改。'
            for e in c['evidence']:
                if e['source_id'] == 'code':
                    e['supports'] = '完整 state_dict（參數及 buffers）的名稱與原 dtype CPU bytes hash；不等即失敗。'
        if page_id == '18.1':
            page['visual_checks'][0]['observation'] += ' 本報告澄清原收據連寫標點：第三路的4、3、100對應70%、20%、10%。'
        errors = []
        artifacts = checker._artifacts(ROOT, page['artifacts'], errors)
        sources = checker._sources(ROOT, page['sources'], artifacts, errors)
        checker._claims(page['claims'], sources, artifacts, errors)
        metadata_errors.extend(page_id + ': ' + e for e in errors)
        pages.append(page)
    if metadata_errors:
        print(json.dumps({'metadata_errors': metadata_errors}, ensure_ascii=False, indent=2))
        return 1
    print('44 personally recorded pages: artifacts/sources/claims metadata valid; this is not a truth check.')
    if not args.write:
        return 0
    inventory_path = BASE / 'execution-inventory.json'
    assert not (ROOT / inventory_path).exists()
    inventory = []
    for path in sorted((ROOT / BASE / 'executions').glob('*.execution.json')):
        meta = json.loads(path.read_text())
        name = path.name.removesuffix('.execution.json')
        code = BASE / 'executions' / (name + '.py')
        stdout = BASE / 'executions' / (name + '.stdout.txt')
        assert sha(code) == meta['code_sha256'] and sha(stdout) == meta['stdout_sha256']
        inventory.append({
            'name': name, **meta,
            'metadata': {'path': str(path.relative_to(ROOT)), 'sha256': sha(path.relative_to(ROOT))},
            'code': {'path': str(code), 'sha256': sha(code)},
            'stdout': {'path': str(stdout), 'sha256': sha(stdout)},
        })
    (ROOT / inventory_path).write_text(json.dumps({
        'reviewer_task': '/root/p7_technical_e', 'executions': inventory,
        'scope': 'Actual saved commands/environments/output and exit statuses, including failed probes. No retrospective execution or success claim.',
    }, ensure_ascii=False, indent=2) + '\n')
    corrections = []
    for num, explanation in [
        ('048', '本人誤記3次暖機；實讀原 timing 後更正1次暖機/3次量測，教材原句準確。'),
        ('078', '本人預填隨機teacher/student參數錯數；真CPU回傳後16960/6104，非教材問題。'),
        ('083', '本人 raw probe 兩次錯鍵失敗保留；第三次才實際逐筆成功。'),
        ('101', '原教師 hash 遍 state_dict 含 buffers/原 dtype，不是僅 named parameters/轉float。'),
        ('125', '本人意譯誤包引號與失敗文字定位已用本人 past-unit-only 原句更正；失敗截圖不算觀看。'),
        ('128', '18.14 三次 raw schema 猜鍵 probe 失敗保留；最後完整10對成功才作證據。'),
        ('157', '19.6 probe 錯 answer key 與同 result 路徑第二次覆寫保存失誤已明記；原首次失敗 stdout 保留。'),
        ('159', '19.6 head module 誤名失敗保留，修正版另路徑成功；真head分類錯如教材。'),
        ('173', 'RMSNorm原定位精確Eq4，先前Eq4–5用語更正；原句與限制窄讀完成。'),
        ('183', '歷史evaluate source與現行不同，從兩原test revisions取回相同原SHA匹配frozen；不使用現行code冒原。'),
    ]:
        path = BASE / 'checkpoints' / (num + '.json')
        corrections.append({'checkpoint': {'path': str(path), 'sha256': sha(path)}, 'note': explanation})
    report = {
        'schema_version': 1, 'review_policy': 'phase7_grouped', 'batch_id': 'phase7',
        'stage': 'technical', 'group': 'e', 'reviewer_task': '/root/p7_technical_e',
        'reviewer_context': 'fresh', 'manifest_sha256': sha(MANIFEST), 'verdict': 'pass',
        'trace_files': [{'path': str(TRACE), 'sha256': sha(TRACE)}], 'pages': pages,
        'scope': '44 primary pages chapter-17/17.1–17.15/chapter-18/18.1–18.14/chapter-19/19.1–19.12；原fresh owner自己逐段五欄；必要context16.11–16.13。',
        'issues': [], 'reviewer_corrections': corrections,
        'execution_inventory': {'path': str(inventory_path), 'sha256': sha(inventory_path)},
        'review_limits': [
            '判定是本人限定範圍的技術審閱，不是 full-stage gate 或自動真理核實；各頁實讀/原來源定位/來源支持範圍與真正執行分開保存。',
            '僅必要相稱CPU機制/算式/公開MoE例與歷史原JSON核算；沒有重跑長訓練、GPU或每版全考卷，不把原論文benchmark當此教材模型實測。',
            '原私有teacher/train checkpoint不在本次取回重hash範圍；原source與保存JSON/receipt核歷史紀錄一致性，未獨立重造私人權重/速度/RAM量測。',
            '公開MoE固定revision四檔實下載/SHA/CPU推論；另三份公開輸出核實固定tree與manifest，未聲稱全部weight下載/load/生成。未在新clone完整uvsync安裝或長resume驗收。',
            '19.12核原彙總/原receipt/protocol/validation/public身份與3734 IDs，未讀全歷史逐題outputs或本次重生成3734題。兩版能力均未全部通過；這不是教材本身revise。',
            '每個必要SVG本人真view640/360；需要的頁表/圖文本人看desktop1280x800/mobile390x844，部分表需實際左右捲。不需要布局頁明列unverified requiredfalse，未冒稱全44頁全畫面實看。',
            '完整外部PDF和全文抽取文字只作本機cache，正式報告只綁本人真查的必要短摘錄、原URL/version/locator/PDF SHA。舊provisional頁/checkpoint/trace/receipt原樣保留。',
            '同一owner多次真current-only恢復，已存checkpoint/past-unit-only恢復原位置；不重造閱讀、不用raw state、作者或他人review補答案。',
        ],
        'metadata_adjustments': [
            '只在這份新完整報告將已真看的舊 tool_receipt 欄位轉receipt、viewport字串轉陣列，observation沿原收據；不生成新觀看或改旧收據時間。',
            '原數字tolerance兩個新頁由數值改明確字串以符schema，數值結果與判斷不變。',
            '已實核但舊頁未直接列出的18.4/18.7/18.11/18.12原distillation JSON補原檔SHA/evidence；18.7內容按已保存本人checkpoint101更正。',
            '必要短論文摘錄組裝曾因搜索詞memory movements未命中而StopIteration；窄讀原句reduced memory movement後完成，組裝note保留，未以失敗作來源已核證據。',
        ],
    }
    assert len(pages) == 44 and all(p['verdict'] == 'pass' for p in pages)
    assert not (ROOT / REPORT).exists()
    (ROOT / REPORT).write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'report': str(REPORT), 'sha256': sha(REPORT), 'verdict': report['verdict'], 'pages': len(pages), 'executions': len(inventory)}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
