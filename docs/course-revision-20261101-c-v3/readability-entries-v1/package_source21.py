"""Package this reader's already saved observations; never change original evidence."""
from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path('/workspace/work/omni-book-integrated-v1')
OUT = ROOT / 'docs/course-revision-20261101-c-v3/readability-entries-v1'
TASK = '/root/c_v3_reader_entries'


def now():
    return datetime.now(UTC).isoformat()


def read(path):
    p = Path(path)
    return json.loads((p if p.is_absolute() else ROOT / p).read_text())


def ref(path):
    p = Path(path)
    p = p if p.is_absolute() else ROOT / p
    try:
        label = str(p.relative_to(ROOT))
    except ValueError:
        label = str(p)
    return {'path': label, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


def save(name, data):
    p = OUT / name
    with p.open('x') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write('\n')
    return ref(p)


original_ref = ref(OUT / 'reader-original-v1.json')
assert original_ref['sha256'] == '27115271a2313bfc55b4ce146e961c660ef222841517f26043e990e5c6a3a239'
original = read(original_ref['path'])
freeze_ref = ref('docs/course-revision-20261101-c-v3/source-freeze-v21.json')
assert freeze_ref['sha256'] == 'd9100e058cae5bbd6427c8f449d7540c8d5a7e9682ed69b4dc0eda87ee4c9cea'
plan_ref = ref(OUT / 'recheck-source21-v1/plan.json')
plan = read(plan_ref['path'])
fold_ref = ref(OUT / 'folded-supplement-report-v1.json')
fold = read(fold_ref['path'])
old_visual_ref = ref(OUT / 'visual-addendum-v1.json')
estimate_ref = ref(OUT / 'reading-times-v2.json')
estimates = {p['page_id']: p for p in read(estimate_ref['path'])['pages']}
source_map = {p['page_id']: p for p in plan['sources']}
bounded = []
for i in range(19):
    r = ref(OUT / f'recheck-source21-v1/note-{i:03d}.json')
    bounded.append((r, read(r['path'])))
assert read(OUT / 'recheck-source21-v1/state.json')['next_index'] == 19
assert read(OUT / 'supplements/folded-v1/state.json')['next_index'] == 11

visual_dir = Path('/tmp/c-v3-reader-entries-source21-visual-20261101')
visual_meta_ref = ref(visual_dir / 'capture-metadata.json')
visual_meta = read(visual_meta_ref['path'])
assert visual_meta['build_info']['revision'] == 'worktree-c-v3-preview-21'
assert visual_meta['build_info']['publication_ready'] is False
assert len(visual_meta['records']) == 14
for item in visual_meta['records']:
    assert ref(item['path'])['sha256'] == item['sha256']
visual_ref = save('visual-addendum-source21-v1.json', {
    'schema_version': 1, 'kind': 'bounded_post_original_visual_addendum',
    'reviewer_task': TASK, 'created_at': now(), 'source_freeze': freeze_ref,
    'original_report': original_ref, 'previous_visual_report': old_visual_ref,
    'preview_build': visual_meta['build_info'],
    'preview_url': 'http://127.0.0.1:8765/',
    'capture_metadata': visual_meta_ref,
    'viewed_images': visual_meta['records'],
    'viewing_method': '14張均在有界文字重讀完成後，實際由view_image以桌機/手機成對開看；不是只由capture存在推定已看。',
    'checks': [
        {'page_id': 'readme', 'positions': ['hardware', 'torch-kernel', 'modes'],
         'finding': 'CPU/GPU簡義、torch套件與kernel角色、python新命名空間快檢及kernel獨立核心保存結果皆可讀；手機換行未藏住句意。'},
        {'page_id': 'training', 'positions': ['new-names', 'default-score'],
         'finding': 'codec/Talker/T/tau/final短義和record的permission/ready同true、數true得2分直接可讀；手機程式仍需橫向取全文，文字理由無被截掉。'},
        {'page_id': 'curriculum', 'positions': ['tts-attention', 'route19'],
         'finding': '文字轉人聲/注意力短義、19.1–19.25終點在兩種寬度皆可定位，段落可正常上下捲讀。'}
    ],
    'unchanged_reuse': {
        'evidence': old_visual_ref,
        'proof': plan_ref,
        'scope': 'course/asset-storage/validation/index source與figure同19；readme/curriculum兩SVG同19，沿用先前已實看完整圖及正常viewport證據。training首表/前文未改，保留原可選橫移負擔。',
        'figures_sha256': {
            'course/figures/rewrite-01-character-ids.svg': '61d45a7fdaef4ce990a51a9cf924ebc661cf41117f7964bd12d3b1c6c4fb02ad',
            'course/figures/curriculum_learning_flow.svg': '55c76a1ce7d7482025064283219b7449ab13ebd7c4c518979f5a55fa16d41da4'
        }
    },
    'retained_optional_issue': 'V-training-001-family-table-pan（見原visual-addendum-v1）；原判後發現，不回填首次觀察。',
    'new_issues': [],
    'limitations': [
        '這是未正式commit且未合併估時的內部preview，publication_ready=false；不是已公開發布版。',
        '這次只補變動定位的14個viewport；沒有宣稱重看每張長頁全部像素、全部連結/模式、Colab或所有學生裝置。',
        'README_en及兩操作附頁本次只做文字有界核對，沒有新增其瀏覽器整頁視覺驗證。',
        '不修改首次圖/版面not_verified紀錄；現在可用的是原判收件後的新證據。'
    ]
})

# Every original issue is retained verbatim, including severity and discovery time.
# The disposition is this reader's source-based judgment, not an author receipt.
issue_decisions = {
 'E-readme-000-CPU': ('resolved', 'source21頁首直接給電腦處理器與圖形運算處理器，足夠本頁辨裝置；兩寬度實看。', [0]),
 'E-readme-001-methodnames': ('nonblocking', '保留可選補名建議。入口已把它們作後續方法/架構預告，任務範圍及有限成果可獨立讀懂；此處無需按縮寫選參數或作算法推論。course明說導航名詞到主線再解，未宣稱本角色已懂內部。', []),
 'E-readme-002-Auto-final': ('resolved', 'source21加自動選路與最後測試短義，原本只猜標籤的負擔已減；沒有將最後測量說成可回調資料。', [1]),
 'E-readme-003-oldB': ('resolved', 'source21先正面交代B的自動循環預算與原本Auto effort用途，再區分C-v3；不用從不替換的否定句猜B角色。', [2]),
 'E-readme-007-torch': ('resolved', '版本表後明說torch是PyTorch套件，用於數字計算/訓練；已在實際桌機/手機讀到。', [5]),
 'E-readme-008-kernel': ('resolved', 'source21把kernel直接連負責執行程式格的核心；顯示名選擇與登記可接起。', [5]),
 'E-readme-009-runtime-names': ('nonblocking', '保留MPS/CI/Modal短註的可選建議。CPU主線有具體可走命令與成功判準，其他名稱所在句已說基本矩陣與未驗全訓練；目前無須懂這些平台內部才能閱讀或選CPU。未驗其他硬體。', []),
 'E-readme-015-stage-names': ('nonblocking', '保留階段英文短註。此段工作是按推論/完整續訓/公開清單選入口，這些角色已分清；各訓練階段詳情留精確training配方，沒有要求新讀者在此直接設SFT或joint。', []),
 'E-readme-017-token': ('nonblocking', '保留token/前向中文短註的可選建議。固定起始材料及示範提示由工具提供，前後句已說下載/檔案核對/CPU推論與一筆非總體率，不需在入口自行編token。', []),
 'E-readme-022-maintenance-modes': ('resolved', 'source21明說python替每節新命名空間、適合快檢，kernel另啟獨立Jupyter且保存結果；首次維護者可判選用。', [6]),
 'E-training-005-sft': ('nonblocking', '中文已給助手示範回答角色，命令值sft可依樣選，不依全英文名推理；保留展全名的可選改善，非通路必要缺口。', []),
 'E-training-008-MLP': ('nonblocking', '本節已交代讀前三字與固定視窗、相同材料切分和前置2.5，足夠執行本節比較；全名仍是可選辨名改善，未冒稱已推導網路。', []),
 'E-training-018-byte-EOS': ('nonblocking', '原文已教按UTF-8單位編號、助手目標與結束目標，足夠保正確單位/停止角色；位元組及EOS全稱仍可短補，完整編碼留6章。', []),
 'E-training-039-DPO': ('nonblocking', '同題chosen/rejected及固定參考角色已在本節教，前後排序不等生成亦另驗；英文全名仍是可選補名，不是當前比較所缺機制。', []),
 'E-training-068-newnames': ('resolved', 'source21直接給codec聲音編解碼器、Talker聲音碼生成器、T校準溫度、tau接受門檻、final最後測試；版本紀錄所需角色已足。', [7]),
 'E-training-072-default-score': ('resolved', 'source21給主錄音室要record、permission與ready皆true，按record數true得2；在本頁即可手核原三個期望，不需先開完整JSON。', [8]),
 'E-asset-001-LoRA': ('nonblocking', '保留展名可選建議。本頁只需知道修正權重必須配正確底座，原句已給；training折疊又實讀adapter/merged配法，未把全稱或算法推導當本頁下載前置。', []),
 'E-curriculum-000-OCR': ('resolved', 'source21把OCR對圖片讀字，現在可定位此任務；不需要在大綱提前教識字算法。', [9]),
 'E-curriculum-002-TTS-attention': ('resolved', 'source21給文字轉人聲/語音合成Text-to-Speech與Attention注意力，短角色可直接讀；內部仍合理待章節。', [10]),
 'E-curriculum-011-route19': ('resolved', 'source21實讀為第19章依19.1–19.25順讀，與同頁新增現行C-v3主線相容；兩尺寸實看終點，原必要導航缺口已解。', [11]),
 'E-index-000-count': ('resolved', 'index仍332；source21 readme/curriculum與English均332逐課加26延伸共358，分母語義一致，不需猜漏一節。未以網站頁數冒逐課數。', [0,9,12]),
 'E-readme-en-001-count': ('resolved', '英文source21列332 lesson sections加26 numbered extensions共358，與中文/index同口徑；保留此項原本是重複發現。', [12]),
 'E-readme-en-012-extra-reason': ('resolved', '英文source21補same PyTorch build理由，現在可理解每次同extra為保同一配套，非只是重複照抄。', [14]),
 'E-readme-en-026-maintenance': ('resolved', '英文source21對照python fresh namespace quick與kernel fresh kernel saves executed results，與中文一致；保留原duplicate_of。', [15]),
 'E-decision-006-rubric-scale': ('resolved', 'source21在本地列record數true、play優先permission、save優先ready的0/1/2尺度，且rubric任一欄缺就拒；已親自以play permission=false ready=true變例得1分。', [17])
}
assert set(issue_decisions) == {x['id'] for x in original['issues']}


def issue_followups(pid):
    result = []
    for issue in original['issues']:
        if issue['page_id'] != pid:
            continue
        status, reason, ids = issue_decisions[issue['id']]
        result.append({'original_issue': issue, 'final_status': status,
                       'reader_reason': reason,
                       'evidence': [bounded[i][0] for i in ids] or [issue['evidence']],
                       'source_sha256': source_map[pid]['source_sha256'],
                       'not_a_retroactive_first_read': True})
    return result


page_judgments = {
 'readme': {
  'identity': '可直接當書讀，想改數字才重跑；公開CPU試用、從零訓練、推論權重與維護入口分開。',
  'mechanism': '正文留程式/關鍵輸出使閱讀不依賴安裝；最小通路以已知答案算loss/反向訊號，mode明示不更新；下載檔核對與單筆生成各只支持局部工作。',
  'need': '要先理解運作且能選自己需要的入口，避免租GPU/下載全包才開始；要分清有數值訊號、已更新、答新題三個要求。',
  'usage': '首次點1.1；本機依CPU步驟檢compute OK後選命名核心；訓練去training；維護兩模式可按新短義選。',
  'example_scope': '示範一筆right不代總體能力，有限任務數據保失敗；C-v3與舊B自動循環/舊C-v2路由分工可辨。',
  'evidence_quotes': ['可以當書讀，不必先安裝', '這裡只算訊號，不更新參數，也不保存權重檔', '每次 uv run 都帶相同 extra，避免更換 PyTorch'],
  'small_change': {'question': 'grad_norm非零能當學會答新題嗎？', 'answer': '不能。', 'reason': '本例不更新，數值只核反向路，能力需另外驗收。'},
  'visual': '字與ID圖有必要，已實看同圖SHA的兩寬度；這次補hardware/torch/kernel/modes變動位置。',
  'burden': '名詞密度仍偏高，但必需裝置/核心/測試角色已補；方法縮寫的可選補名仍保留，導航並不要求先會算法。'
 },
 'course': {
  'identity': '正式閱讀路線與需求索引，含主線、C暖身插點、A/B延伸及成品入口。',
  'mechanism': '上一字表對相同尾字給同分布，故以更長前文引出後續；小手算揭機制、訓練與未見題能力分別驗，非用擬人話代證據。',
  'need': '學生需要從問題找位置且沿依賴讀，不必先懂索引表全部名詞；避免把手算例或一次成功當一般能力。',
  'usage': '1–7→C.1–C.5→8–13→C.6–C.7→14–19，按需看附錄；Notebook由同源短程式重跑。',
  'example_scope': '參數/權重及從零定義可懂；整合成品重用與新驗收分開，授權折疊完整補讀。',
  'evidence_quotes': ['表中的名詞...要到主線讀到那裡才解釋，現在看不懂沒關係', '理解眼前例子需要的說明仍留在正文裡', '不以舊成功題代替新能力'],
  'small_change': {'question': '同樣上一字、上文不同，上一字表會分開機率嗎？', 'answer': '不會。', 'reason': '索引只用上一字，這正是需要讀前文的限制。'},
  'visual': '無獨立必要圖；首屏與手機路線/表已在原判後實看，source21同19沿用。',
  'burden': '兩張索引表需要停看，但各名詞明示到主線教；沒有要求先補所有ML課。'
 },
 'training': {
  'identity': '依需要選T節的操作/驗收入口，手動小配方、固定實驗與公開成品各有不同依賴和證據範圍。',
  'mechanism': 'train開真正更新，前後用同資料/選項核；text只接字、sft只計助手目標；checkpoint新階段和resume同排程不同；教師比較固定起點/字表/切分；生成與候選排序另測。',
  'need': '要能重做目前機制而不誤載不配對底座/資料、用錯分母或拿短通路冒能力；固定配方需讓依賴/裝置/歷史時間可辨。',
  'usage': '根目錄已備環境後選對T節；先做各節指定父權重/資料，JSON欄和分母查本次產物；完整九折疊現已實讀。',
  'example_scope': '所有CPU/GPU歷史與未量時間分開。source21本地可核record/info yes/2；未把本角色閱讀當模型重訓或GPU可用性驗證。',
  'evidence_quotes': ['梯度大小不是答對率', '沒有更新參數', '許可與就緒都為true', '按record數true欄位的固定規準得2分'],
  'small_change': {'question': '同題DPO候選較偏chosen，能保證自由生成就是chosen嗎？', 'answer': '不能。', 'reason': '排序只比兩個給定答案，生成另從全字表逐步選，故要並排實答及舊題保留。'},
  'visual': '本頁無必要新圖。原首表/程式桌手機實看，T.11新名與default局部source21另實看；手機寬表橫移仍為可選呈現改善。',
  'burden': '完整inventory很長，折疊亦含實際依賴，故完整讀頁估85–125分鐘；按一條配方查用可較短。內部算法留精確章節，未冒稱全學完。'
 },
 'asset-storage': {
  'identity': '程式、資料、模型數字和完整訓練狀態的下載/保存分工。',
  'mechanism': 'LFS pointer不是包本體；核大小/SHA再解包。推論只需權重與設定等，resume另需optimizer/RNG/進度，adapter還須配正確底座。',
  'need': '要依讀書/試用/訓練/續訓選最少必要材料，避免只拿推論四檔就想精確續接或以同名當同版本。',
  'usage': '讀書可跳大包；訓練用manifest的固定資料，推論走對應操作，發布後從新cache回讀核對。',
  'example_scope': '大小/指紋只能證同檔，不能證內容能力；best由val選、latest供續接兩角色分開。',
  'evidence_quotes': ['才能知道自己現在需要下載什麼', '下載核對成功只證明取得相同檔案', '從新的快取重新下載核對'],
  'small_change': {'question': '只有公開推論四檔能精確resume嗎？', 'answer': '不能。', 'reason': '缺optimizer/RNG/步數等完整狀態，檔案角色不同。'},
  'visual': '無必要圖；原判後桌手機首屏與手機表實看，source21同19沿用。',
  'burden': '食譜比喻接回實際三材料清楚，指紋/大小列較密但用于選包；LoRA全稱可選補，底座配套角色已教。'
 },
 'curriculum': {
  'identity': '設計取捨/來源與新增主題定位頁，正式順讀仍依course。',
  'mechanism': '先小例再需求；X是輸入、Y是已知答案，可讀遮罩與哪些答案計分分開；不累積時先清舊梯度再算新/更新；研究品質要同條件和匹配預算另驗。',
  'need': '要讓讀者明白為何如此排序及證據只支持何種結論，避免用小分數推一般效益或把資源方法都叫省成本。',
  'usage': '帶問題查表定位，順讀去course；source21第19章範圍1–25與推薦C-v3相接，沒有漏新增必修。',
  'example_scope': '參考索引不等外部實作已重現，成熟模型/自訓模型/有限音型與人聲TTS各有界線；OCR/TTS/Attention短義已補。',
  'evidence_quotes': ['可讀哪些前文、哪些答案要計分，是兩個不同的問題', '它們的資源效果各自量測', '第19章依19.1–19.25順讀'],
  'small_change': {'question': '提問不計loss，是否代表模型不能看提問？', 'answer': '不是。', 'reason': '可讀遮罩和計分位置做兩件事，輸入線索可以可讀但不列答案代價。'},
  'visual': '教學資料流圖有必要，原判後真看兩尺寸完整圖及正常上下viewport；SVG同21可沿用。新術語/終點兩尺寸另核。',
  'burden': '來源表與多方法密度高，但本頁定位是設計/索引，算法未在此作新推理；唯一必要導航缺口已親自讀修後範圍確認。'
 },
 'validation': {
  'identity': '教材/通路/更新/能力和發版版本各種驗收的範圍說明。',
  'mechanism': 'train/val/final分工及先凍結再final；teacher forcing給正確前文而自由生成接己答，錯可累積；異CPU重播容差1e-11只核統計，SHA仍要精確同檔。',
  'need': '要知道哪個結論真的由哪種證據支持，避免把下載載入/CPU短通路/完成訓練當所有能力達標或把重算統計當新推論。',
  'usage': '每命令看正常exit及JSON/pytest摘要；能力看原判準分母/失敗；折疊完整補讀，並未自己重跑課程驗收。',
  'example_scope': '固定有限任務、家族新/文字新、既見型/新組合分開；AI閱讀不是真人，CPU不代Colab或全部設備。',
  'evidence_quotes': ['程序完整不表示全部能力達標', '沒有把重新整理統計當成新一輪模型測試', '每一步先給正確前文'],
  'small_change': {'question': '統計末位有2.22e-16差，可放寬檔案SHA核對嗎？', 'answer': '不能。', 'reason': '統計浮點容差與精確同檔檢查不同；折疊保科學門檻/整數不變，沒改檔SHA規則。'},
  'visual': '無必要圖；原判後桌手機首屏與手機表實看，source21同19沿用。',
  'burden': '分母/版本很多，需慢讀才能辨各證據；但界線句具體，折疊重播說清不同機器數值差而非新增能力。'
 },
 'index': {
  'identity': '網站第一入口，用接字限制串起模型、任務與成本，提供第一節/路線/重跑導航。',
  'mechanism': '上一字相同卻前文不同不能靠一列表區分，故擴到讀前文；多模態/服務和資源約束引出後續。',
  'need': '首次讀者需要先選從哪裡讀，且知道閱讀不必先安裝，重跑才開NB。',
  'usage': '第一次點第一節；需要選路點路線，按概念搜尋，手機可開選單；source21其他入口332/358已同口徑。',
  'example_scope': '入口不承諾已通用可靠，不以332數量當能力；本頁source/fig同19。',
  'evidence_quotes': ['可以先當書讀', '章節目錄收錄全部332個小節'],
  'small_change': {'question': '只想理解第一個例子，要先裝Notebook嗎？', 'answer': '不用。', 'reason': '正文包含理解必要程式/輸出，入口直接可點第一節。'},
  'visual': '原判後兩寬度首屏與手機選單正常完成動畫後實看；同19沿用，未稱所有UI已驗。',
  'burden': '入口短，先選一條路即可；原數量口徑可選疑問已由其他入口改為332/358解。'
 }
}

aux_pages = []
for old in original['supplementary_pages']:
    pid = old['page_id']
    src = source_map[pid]
    notes = [r for r, n in bounded if n['page_id'] == pid]
    aux_pages.append({
        'page_id': pid, 'source': src['source'], 'canonical': False,
        'native_370_coverage': False, 'reviewer_task': TASK,
        'source_sha256': src['source_sha256'], 'figures_sha256': src['figures_sha256'],
        'source_snapshot': ref(src['new_snapshot']), 'verdict': 'pass',
        'original_report': original_ref, 'original_json_pointer': f"/supplementary_pages/{len(aux_pages)}",
        'original_first_read_units': old['unit_count'],
        'bounded_notes': notes,
        'unchanged_source_reuse': src['old_sha256'] == src['source_sha256'],
        'reader_judgment': {
            'readme-en': '332/358、same PyTorch build、python/kernel模式與B正面角色現與中文範圍相接；英文只是對應入口，不把已有中文前文當重新首遇。',
            'omni-products': 'source同19，原20單位已實讀；影音實際材料/時間問法、下載到推論、固定與失敗邊界保持，不聲稱新重跑。',
            'decision-product': '新尺度在本地可手核0/1/2與缺欄拒答；action與info需求不同，不把permission=false當未知；新play變例已真答1分。'
        }[pid],
        'issue_followups': issue_followups(pid), 'new_issues': [],
        'visual_status': 'not_verified_for_auxiliary_full_render',
        'scope': '原55單位後的有界文字修後核；不混入canonical370頁或其估時。'
    })
aux_ref = save('auxiliary-bounded-source21-v1.json', {
    'schema_version': 1, 'kind': 'auxiliary_bounded_reader_report', 'reviewer_task': TASK,
    'created_at': now(), 'source_freeze': freeze_ref, 'original_report': original_ref,
    'bounded_plan': plan_ref, 'canonical': False, 'native_370_coverage': False,
    'pages': aux_pages, 'new_issues': [],
    'method': '原收件後依真snapshot只開19個有意義變更單位中的7個附頁單位，先保存觀察再下一單位；未變omni依SHA保留原首讀。',
    'limitations': 'AI閱讀，未執行下載/模型推論；三附頁未作整頁渲染驗證。原觀察/分級/發現時點均保留在original_issue。'
})

basis_pages = []
for i, old in enumerate(original['pages']):
    pid = old['page_id']
    src = source_map[pid]
    judgment = page_judgments[pid]
    page_folds = [r for r in fold['notes'] if read(r['path'])['page_id'] == pid]
    page_bounded = [r for r, n in bounded if n['page_id'] == pid]
    basis_pages.append({
        'page_id': pid, 'source': src['source'], 'selector': old['selector'],
        'reviewer_task': TASK, 'source_sha256': src['source_sha256'],
        'figures_sha256': src['figures_sha256'], 'verdict': 'pass',
        'source_snapshot': ref(src['new_snapshot']),
        'original_report': original_ref, 'original_json_pointer': f'/pages/{i}',
        'original_first_read_session': old['native_reader_trace'],
        'bounded_recheck_notes': page_bounded, 'folded_supplement_notes': page_folds,
        'source_transition': {
            'original_sha256': src['old_sha256'], 'current_sha256': src['source_sha256'],
            'unchanged_main_ranges': [x for x in src['opcodes'] if x[0] == 'equal'],
            'read_changed_new_unit_indices': src['selected_new_unit_indices'],
            'all_original_details_byte_unchanged': src['folded_blocks_unchanged'],
            'proof': plan_ref
        },
        'reader_questions': {
            'q1_identity_materials_labels': judgment['identity'],
            'q2_mechanism': {'answer': judgment['mechanism'], 'evidence': judgment['evidence_quotes']},
            'q2_need': {'answer': judgment['need'], 'evidence': judgment['evidence_quotes']},
            'q3_usage_and_small_change': {'usage': judgment['usage'], 'small_change': judgment['small_change']},
            'q4_example_and_evidence_scope': judgment['example_scope']
        },
        'necessary_visual_judgment': judgment['visual'],
        'entry_reading_burden': judgment['burden'],
        'visual_evidence': [old_visual_ref, visual_ref],
        'issue_followups': issue_followups(pid),
        'new_issues': [], 'necessary_open_issues': [],
        'ai_estimate': estimates[pid],
        'limits': '此pass限本頁導覽/操作/設計/證據範圍的可理解性；算法推導、課程命令實跑、硬體與模型能力未由本角色驗證。'
    })

basis_ref = save('reader-basis-source21-v1.json', {
    'schema_version': 1, 'kind': 'reader_bounded_basis', 'role': 'reader',
    'reviewer_task': TASK, 'created_at': now(), 'source_freeze': freeze_ref,
    'original_report': original_ref, 'bounded_plan': plan_ref,
    'folded_supplement': fold_ref, 'visual_addenda': [old_visual_ref, visual_ref],
    'reading_times': estimate_ref, 'auxiliary_report': aux_ref,
    'reader_profile': original['reader_profile'], 'pages': basis_pages,
    'coverage': {'canonical_pages': 7, 'original_native_main_units': 138,
                 'post_original_original_details_blocks': 11,
                 'bounded_changed_canonical_units': 12,
                 'bounded_changed_auxiliary_units_separate': 7,
                 'all_details_of_covered_inventory_pages_read': True},
    'method': '原判先被真正收件，後補11折疊，再逐單位讀source21變更並先保存觀察；其餘正文/圖以相同SHA及逐段diff沿用真首讀。此basis不是新的獨立首次閱讀。',
    'limitations': original['protocol_limitations'] + [
        '首讀後才補calibration、全部折疊與視覺，時點見各addendum；不倒寫成首讀已完成。',
        '原必要路線問題已在source21真讀修後解；原可選補名中未採的仍保存為nonblocking。',
        '未讀project-context、作者計畫、舊審閱或他人答案；沒有執行課程模型命令。',
        'source21視覺是內部preview，不等公開版。'
    ]
})

current_pages = []
for i, old in enumerate(original['pages']):
    pid = old['page_id']
    src = source_map[pid]
    current_pages.append({
        'page_id': pid, 'source': src['source'], 'selector': old['selector'],
        'reviewer_task': TASK, 'source_sha256': src['source_sha256'],
        'figures_sha256': src['figures_sha256'], 'verdict': 'pass',
        'basis_report': basis_ref, 'basis_json_pointer': f'/pages/{i}',
        'ai_estimate': estimates[pid],
        'sessions': old['sessions'],
        'necessary_open_issues': [],
        'reader_summary': page_judgments[pid]['identity'] + page_judgments[pid]['burden'],
        'evidence_scope': '完整inventory正文+全部details的理解估時；source21改動有界重讀、未改source/fig誠實沿用；視覺只限兩addenda列出的實看範圍。'
    })
current_ref = save('reader-current-source21-v1.json', {
    'schema_version': 1, 'kind': 'reader-current', 'role': 'reader',
    'reviewer_task': TASK, 'created_at': now(), 'source_freeze': freeze_ref,
    'basis_report': basis_ref, 'original_report': original_ref,
    'pages': current_pages, 'reading_times': estimate_ref,
    'auxiliary_report_separate': aux_ref,
    'limitations': 'AI獨立讀者，不是真人測試；pass限頁面所承諾理解範圍，未把實際執行/能力/Colab/其他硬體或所有像素視覺核對列為已驗證。'
})

# Every key below is one actual original checkpoint whose literal verdict was
# not exactly 已足夠. Qualified adequate notes still receive explicit closure.
reasons = {
 'readme': {
  '0.0': '入口文字原已足；後實看兩寬度/字編號圖，source21再核硬體短義。圖版面未知只在實看範圍解，不冒稱Colab或整頁每像素。',
  '1.0': '本段只導航從前文限制到任務/成本，已有獨立需求及精確後續位置；不需在此推導所有架構，保留方法細節待主線。',
  '1.1': 'MoE/Dense角色作成果預告可讀，沒有以未教結構作本段推理。原選讀補名仍保留為nonblocking，未聲稱算法已理解。',
  '2.0': 'source21已補Auto自動選路/final最後測試。影音配對、音型整段、接受者內錯率分母原已清，短名負擔已減。',
  '3.0': 'source21先正面交代舊B自動循環預算；C-v3明確欄位/有限裝置邊界仍保留。完整JSON於後實讀decision附頁可查，不假稱原入口已列全部。',
  '4.0': '同前source21把B角色說清，Auto先判支援/資訊再選輪數的文字機制未變；兩任務巨大分差仍保留，沒有用新C-v3覆蓋舊限制。',
  '4.1': '原後續unit5實開安裝命令，unit6–9讀成功準則/CPU實輸出/平台與一致性；準備到動作已接，不是永遠等待下段。',
  '5.0': '原unit6–9補正常退出/compute OK、Windows啟用與核心定位；source21又給kernel及torch角色。可理解操作順序，frozen/sys-prefix內部未擴成必要模型先備。',
  '6.0': '原unit7已真讀一次CPU節錄，與前述compute OK/版本可變規則相接；這不是本角色實際安裝成功證明。',
  '8.0': 'CPU指令/Notebook路線可讀，source21 kernel短義已補；可選硬體相容性/實機驗證維持未驗，頁面明限CPU檢查即可選主線。',
  '9.0': '同extra避免換PyTorch、基本MPS與正式訓練分證據的操作理由原已足；未實作環境是角色邊界，不是文字必要缺口。',
  '10.0': '同批正確答案機率低則loss高的方向與通路用途已給；原unit13數值例後讀到，不把低loss說成新題能力。',
  '10.1': '本頁只核反向可產有限非零訊號且不更新，原unit13/14有實值與限制；完整反向算法仍主線待教，不把它當入口所需。',
  '11.0': '先產toy-text再讀同train.jsonl的相依已可辨；原unit13/14教報告欄及--train差別，未執行仍不宣稱本機成功。',
  '12.0': '原unit13已看到dry-run/loss/grad_norm節錄，unit14解釋範圍；引介所預告內容確已開讀，未推未列的其餘欄。',
  '13.0': 'device/mode與有限非零grad_norm支持一次通路，無比較基準不判能力。這是符合本頁深度的已足夠限定，不需追加新算法。',
  '14.0': '不更新/--train更新、先list再指定包的分界原已教清；保留GPU smoke只驗管道而非能力的限制。',
  '15.0': '推論vs完整狀態與選用步數vs完成步數原已分清；階段英文仍列可選補名。後training完整配方補讀不回算為此處首遇已知。',
  '16.0': '原unit17明說第一命令只列、第二匿名下載/核檔/CPU推論/固定提示生成，精確工作及結果欄已在後讀完成。',
  '17.0': '本段工具提供固定開始材料且記錄輸入輸出，一筆非總率；token/前向的可選短註保留，沒要求讀者先手編碼。',
  '20.0': '正文→產生Notebook的單一來源表已具體對路徑；本段是目錄定位，內部build不需在入口重教。',
  '21.0': '原unit22說pytest/ruff/格式及三審，source21補python與kernel不同覆蓋；清單與結果判法已接，未實跑不報通過。',
  '22.0': 'source21直接交代python新命名空間快檢與kernel獨立核心保存輸出，首次維護者可判取捨；原可選模式問題已解。',
  '23.0': '整頁入口文字原可達，現hardware/核心/模式/Auto/final/B角色補讀且已核兩圖及指定版面；仍保留其餘可選補名和視覺/執行範圍限制。'
 },
 'course': {
  '0.0': '編號可還原與上一字表同尾限制在本段足以說明路線；完整tokenizer/Transformer依主線教，導覽不依其內部作推理。',
  '1.0': '按需表只定位章節，原文明說現在看不懂名詞沒關係；本頁不要求先會所有方法再使用索引。',
  '2.0': '主線及C插點明列，眼前必要說明仍正文；source21 curriculum終點也對到19.25，無新增路線矛盾。',
  '3.0': '手算看機制、訓練/能力另驗的分工本段已清，無需由小例冒成人類理解；後折疊亦補數字/梯度/更新角色。',
  '3.1': '可調數字叫參數/整組權重、從零未借預訓數字的定義已足本頁；訓練細節合理主線後教。',
  '4.0': 'source/fig同19，整合重用與新驗收分開；後折疊授權及PyTorch角色已實讀，桌手機首屏/手機表亦實看，未推所有設備執行。'
 },
 'training': {
  '0.0': '入口明指已備環境/根目錄，device工具支持與作者實測時間不互代；實讀可選命令路線，未實際計時/跑硬體仍保留。',
  '2.0': '原unit3隨即提供list與asset指定故事包命令，unit4說LFS實體核對/家族切分；前導預告已由後文兌現。',
  '3.0': '命令按名稱下載，原已讀README腳本核對解包角色；本節只選素材，不需重教壓縮或傳輸算法。',
  '4.0': '原材料/衍生同側規則回應避免假新資料，toy通路與正式包用途分開；無需先執行下載才理解這條資料規則。',
  '5.0': 'loss/梯度沿真已讀README，無--train不更新/不存的角色清楚；SFT中文示範角色足以選任務，全名仍可選。',
  '8.0': '同文同切分、context3/width16角色及參數混雜限制已有；MLP全稱仍可補但不擋比較命令，前置2.5未冒稱算法已驗。',
  '11.0': 'train/steps/output與JSON重導角色清楚，原後unit12/13前後數值已讀；未把命令列當已訓成功。',
  '14.0': '視窗變大同時改參數的限制與seed角色已分；T4兩種材料在後格式/計分及命令實讀，沒有只留預告。',
  '17.0': 'text/messages/role/family欄已可依產物定位，unit18再教助手有效目標與EOS；未用外部常識先補計分。',
  '18.0': 'byte編號、只算助手及EOS停止目標已教當前操作所需，完整編碼留6章；byte/EOS全名仍保留可選名稱改善。',
  '19.0': '原unit20接著解共同seed、尺寸/上下文/字表設定角色，後續同起點比較成立；只解讀腳本，未實跑。',
  '21.0': '原unit26明解tokens新增上限/limit評題數、parent非resume，text前後相同選項可核；未將token loss當完整生成。',
  '22.0': '原unit23具體三行SFT命令緊接開讀，unit26補共同選項；對話分支不是未給動作的孤立預告。',
  '23.0': '原unit26補tokens/limit/父checkpoint角色；前後只換權重、同資料/選項的比較可理解，未做能力背書。',
  '25.0': '接text權重再sft屬新階段，原unit26明非resume及其狀態要求；同評估條件另存避免覆寫。',
  '26.0': '父權重新optimizer和同次resume完整狀態原已分；後完整resume折疊補同排程/資料/可訓名單，不重載encoder。',
  '28.0': '兩獨立模型未載text起點可由命令核，原unit20已給context上限，T5/後續驗收讀max-length時需配該上限；未聲稱所有長度任選或已執行。',
  '29.0': '風格/安全/算術保留分測與局部encoder角色可理解；安全與LoRA折疊後補同題偏好vs安全標籤，無一般安全保證。',
  '30.0': '原unit31明說train/holdout小任務、模式/examples/accuracy及類別映射，精確材料/報告已在後讀補齊。',
  '31.0': '小留出兩題分母/映射及判報告欄原已給，僅支持小任務；未推真影音或一般泛化。',
  '31.1': 'projector/partial/none更新範圍及輸入梯度條件原已分清；後固定配方更明指來源，沒用凍結旗標冒實際能力。',
  '32.0': '原unit33具體三模態命令在下一段實讀，checkpoint與兩encoder先備相接；不是永待下一段。',
  '33.0': 'checkpoint/encoder/freeze各入參角色原已教、模型檔先達門檻後載；後resume折疊補舊狀態不重載encoder，未實跑。',
  '35.0': '原unit36親讀shape圓、pitch高、joint方低的精確期望與停止條件，先前未知答案已由後文而非外部常識補。',
  '36.0': '抽查不能代固定VQA/文字保留分測，固定依賴在unit37及後折疊完整實讀；不以純文字evaluate測圖/聲。',
  '37.0': 'sft→encoders→projector→vqa依賴原已列，後折疊再區分固定sft與手動attributes、各教師資料配套；尚未執行。',
  '38.0': '各指標同切分/各分母、partial位置及CPU降範圍已可讀，折疊補固定配方，不以歷史摘要當本次重現。',
  '39.0': 'DPO同題好壞回答/固定參考作用本地足夠，後折疊補style依賴/偏好與生成分測；全名可選保留，不假稱已推公式。',
  '45.0': '原unit46隨即同prompt/tokens/greedy只換preferred權重的後回答命令實讀，unit47核生成與排序兩工作。',
  '47.0': '本段清楚候選排序不等生成同字串、需舊題保留，已能作小變化推理；後固定DPO/PPO折疊再核任務差別。',
  '47.1': '保資料/尺寸/步數只換零件的比較條件已足；內部RMS/專家算法仍循14.2/15.1/15.4必要前文，未讀不冒學完。後折疊補固定modern/moe來源。',
  '48.0': '三版命令分norm/experts/top-k其餘條件相同可直接比，參數/時間另列；不需要在此重推各層公式才能核設定差。',
  '51.0': 'NLL低可同greedy輸出、EOS與時間另看已具體；後折疊另核Flash只probe集合非整排程，保留比較範圍。',
  '52.0': '共同浮點原版各轉兩位數保比較起點，真保存格式和未承諾加速已清；量化折疊後讀另120更新/QAT適應而非同一單步convert。',
  '57.0': '逐張量numel乘單格byte加總與整檔大小分兩標籤，當前腳本可手核角色；不把參數bytes說成整體RAM或延遲。',
  '62.0': '原理深度只需本節教師機率新增訊號和同起點對照；後固定配方/雙教師全部折疊已實讀，仍未讀18章推導或實際執行，故保操作範圍。',
  '63.0': '模型+dataset六檔與原切分配對原已明列，後折疊真核VQA/joint兩教師不同依賴及共同字表；缺依賴不自補的界線保留。',
  '64.0': '原unit65說GSM8K只是域外診斷、不是教師原切分內成功；額外包用途由後文解，未把域外低分說主任務失敗。',
  '65.0': '同w32比較CE或多教師KL訊號、同資料安排和KL差異角色可懂；公式仍精確18.8–9，不假稱本頁已推導。',
  '66.0': '原unit67已給joint固定方/高精確期望，與相同prompt/裝置/預算可相接；未以一例代同題全檔。',
  '67.0': '先前未讀必要折疊現已全部實讀：兩教師各原切分、兩組model/dataset、共字表/位置配對；未實做仍明記，閱讀理解缺口解而執行未證。',
  '68.0': 'source21已補codec/Talker/T/tau/final角色且兩寬度實看；開發選後final、推論大小非RAM的原限制保持。',
  '69.0': '原後unit70提供新版操作連結，之後真首讀omni20單位看到AV實際材料/時間問法/輸出與失敗範圍；不是先猜av命令等正式評測。',
  '70.0': 'C-v2研究/未過接受門檻與推薦C-v3明確JSON分線足夠；後decision附頁真讀完整輸入/固定尺度，未執行仍不稱demo本機成功。',
  '71.0': '原unit72接預設三期望；source21更補record/permission/ready與2分本地理由，命令輸入輸出現可手核。',
  '72.0': '完整九折疊已逐塊後補實讀、default理由source21真讀且兩寬度核。算法推導/實跑仍未驗，無需冒完備；手機寬表和補名可選改善仍保留。'
 },
 'asset-storage': {
  '0.0': '資產三類及可重建資料存生成器/seed/設定已明確，能依讀書或訓練選路；下載細節可留精確工具不必本頁重實作。',
  '2.0': 'LFS跳大包沿真已讀README角色、Windows分支精確W.1可查；本路線先拿程式可理解，未宣稱所有系統實機都過。',
  '6.0': 'source同19，best/latest、推論/續訓、同檔/能力分工原已足；後桌手機/表實看但執行未驗維持，非必要閱讀缺口。'
 },
 'curriculum': {
  '0.0': 'source21補332+26=358與OCR圖片讀字，設計頁/網站頁數清單不同原已明說；有限成果不當全部能力。',
  '2.0': 'source21補TTS語音合成與Attention注意力名義；新增影片/音型工作、舊核心不能冒新能力的邊界足以作概覽。算法仍待對應主線，未讀不冒驗證。',
  '3.0': 'MoE選專家示範與Dense另一完整配方的教學理由已直接給，且明說非同參數同計算受控比較；大綱不需先推MoE。',
  '4.0': '此段索引固定版本與取捨來源，明示未執行全訓練；不讀作者研究/舊審閱是本角色獨立界線，不是當前設計文字缺漏。',
  '5.0': '左觀察右教法一一對應，影片未看/鏡像非官網等來源邊界都明記；技術來源真偽另角色驗，不以本AI閱讀冒外部重現。',
  '5.1': '不同切詞token單位不同，改同原文byte分母比較的動機已教；完整BPB公式精確6.5，概覽未要求本地算值。',
  '6.0': '量化位數/蒸餾學生/MoE部分啟動/LoRA少更新參數角色分開，資源效果各量本段已有，不由一個省字推各種成本。',
  '7.0': '正文當下必要、模組共用、Notebook同源新kernel及本地CPU不等Colab的分工清楚；這是載體規約，不宣稱所有頁實跑已過。',
  '8.0': '單變項機制與匹配預算品質兩目的原已分，允許無差異/退步及列調整條件；算法細項本段不使用推理，仍待章節而非必要缺口。',
  '9.0': '原文字X/Y兩遮罩角色已足；後真看完整兩尺寸圖及正常上下viewport，同SVG21，箭頭與標籤可對上可讀/計分不同工作。',
  '9.1': '原文字說不累積清零時機，後真看圖③清舊→反向→更新可辨；執行未驗仍保留，概覽不需要先實訓。',
  '10.0': '單一正式目錄/路線分工可理解，source21終點19.25修後一致；表列方法只是定位，不冒已懂算法。',
  '11.0': '同句跨家族/錄音非speaker隔離、規則預算非收益學習器界線原足；不以新產品覆蓋限制。',
  '11.1': '原必要缺口同一導航關係已真讀source21改第19章19.1–19.25，與本頁新增主線相合；桌手機終點亦實看，照字面不再漏現行C-v3。',
  '12.0': '規約要求分角色/真讀證據及小例不代一般效益可理解；本角色只作AI閱讀/自己證據，不主張別人的流程或模型因此已驗。',
  '13.0': 'nanochat此處只按檔案定位BPB/CPU流程來源，非初教方法；索引深度已足，未開外部不稱已核原碼。',
  '14.0': 'MiniMind-V只列具體對齊/計分/凍結來源位置，非在此教佔位符算法；方法內部不屬這段必要理解。',
  '15.0': 'MiniMind四固定檔按用途查，DPO/蒸餾已有精確位置，列檔不冒實測；索引限定合理。',
  '16.0': '指令/偏好與有用/安全成本分來源可選查，此處無新RLHF機制要求；外部整訓練仍不稱已重現。',
  '17.0': '原整頁必要導航同11.1修後解；兩圖標籤/箭頭後真看，source21新名/終點兩尺寸再核。其餘設計文字/書目定位原已可懂，未升格外部技術驗證。'
 },
 'validation': {
  '1.0': '通路/載入/少數成功與全量能力分開，3734等固定材料範圍本段足；未開原報告不能冒技術重算，閱讀判準不因此缺。',
  '2.0': '同句跨家族、反序/換說法另測的具體限制原可懂，有限語法不等開放中文；沒有靠術語代掉需要。',
  '2.1': '校準/政策/final凍結鏈文字足以理解先後及檔案匹配目的，真hash/分數核是技術角色；後統計折疊補重播非新推論，仍不聲稱自己核原數據。',
  '3.0': '十工作程序完成與能力門檻未必達、C-v3另批新測三型共族/舊60回歸分清；本段範圍限定足。',
  '4.0': '波形內發聲控制、gold碼vsA預測、已見音型vs未見影音情境分開，成功和148失敗皆留；不是未教的單一泛化主張。',
  '5.0': 'teacher forcing每步給正確前文，自由生成接己答可能積錯的機制/需求分開給，選版只用val，final不可回調；閱讀深度足。',
  '8.0': '原unit9緊接解正常exit、JSON/pytest摘要及失敗定位/重審範圍，清單的各範圍已從後文實讀補齊；命令未跑不報過。',
  '9.0': 'source同19，原後段已給結果判讀；後完整統計折疊讀完、桌手機首屏/表實看。實做/全設備仍未驗明記，文字指南不需假執行才能pass。'
 },
 'index': {
  '0.0': '原332本身未改，source21其他入口332+26=358使數量疑問解；原判後兩寬度首屏及手機選單完成動畫後真看，入口可達。沒稱所有UI/Colab已驗。'
 }
}

# Exact references to subsequent observations, never summaries from the author.
later_native = {
 ('readme',4):[5,6,7,8,9], ('readme',5):[6,7,8,9], ('readme',6):[7],
 ('readme',10):[13,14], ('readme',11):[13,14], ('readme',12):[13,14],
 ('readme',16):[17], ('readme',21):[22],
 ('training',2):[3,4], ('training',14):[17,18], ('training',17):[18],
 ('training',19):[20], ('training',21):[26], ('training',22):[23,26],
 ('training',23):[26], ('training',25):[26], ('training',28):[20,29],
 ('training',30):[31], ('training',32):[33], ('training',35):[36],
 ('training',36):[37,38], ('training',45):[46,47],
 ('training',64):[65], ('training',66):[67], ('training',69):[70],
 ('training',71):[72], ('validation',8):[9]
}
bounded_links = {
 ('readme',0):[0], ('readme',2):[1], ('readme',3):[2], ('readme',4):[2,3,4],
 ('readme',5):[5], ('readme',8):[5], ('readme',21):[6], ('readme',22):[6],
 ('readme',23):[0,1,2,5,6], ('training',68):[7], ('training',71):[8],
 ('training',72):[7,8], ('curriculum',0):[9], ('curriculum',2):[10],
 ('curriculum',10):[11], ('curriculum',11):[11], ('curriculum',17):[9,10,11],
 ('index',0):[0,9,12]
}
fold_links = {
 ('course',3):[0], ('course',4):[0],
 ('training',26):[4], ('training',29):[3], ('training',33):[4],
 ('training',36):[2,8], ('training',37):[2,8], ('training',38):[8],
 ('training',39):[5], ('training',47):[5,6], ('training',48):[6],
 ('training',51):[6], ('training',52):[7], ('training',62):[2,6,8],
 ('training',63):[8], ('training',67):[8], ('training',72):list(range(1,10)),
 ('validation',2):[10], ('validation',9):[10]
}
resolved_keys = {
 ('readme','0.0'),('readme','2.0'),('readme','3.0'),('readme','4.0'),
 ('readme','4.1'),('readme','5.0'),('readme','6.0'),('readme','12.0'),
 ('readme','16.0'),('readme','21.0'),('readme','22.0'),('readme','23.0'),
 ('course','4.0'),('training','2.0'),('training','19.0'),('training','21.0'),
 ('training','22.0'),('training','23.0'),('training','30.0'),('training','32.0'),
 ('training','35.0'),('training','45.0'),('training','64.0'),('training','66.0'),
 ('training','68.0'),('training','69.0'),('training','71.0'),('training','72.0'),
 ('curriculum','0.0'),('curriculum','2.0'),('curriculum','9.0'),
 ('curriculum','9.1'),('curriculum','11.1'),('curriculum','17.0'),
 ('validation','8.0'),('validation','9.0'),('index','0.0')
}
closure_pages = []
total = 0
for page_index, old in enumerate(original['pages']):
    pid = old['page_id']
    resolutions = []
    expected_keys = set()
    for unit_index, note_ref in enumerate(old['native_reader_trace']['notes']):
        note = read(note_ref['path'])
        for concept_index, concept in enumerate(note['concept_checks']):
            if concept['verdict'] == '已足夠':
                continue
            key = f'{unit_index}.{concept_index}'
            expected_keys.add(key)
            reason = reasons[pid][key]
            evidence = [
                {'reference': note_ref, 'json_pointer': f'/concept_checks/{concept_index}',
                 'quote': concept['source_quote'], 'scope': '原判原句/分級，未改寫'},
                {'reference': basis_ref, 'json_pointer': f'/pages/{page_index}',
                 'scope': '自己的source21有界basis，含版本/折疊/四問/可選問題'}
            ]
            for later in later_native.get((pid, unit_index), []):
                r = old['native_reader_trace']['notes'][later]
                evidence.append({'reference': r, 'scope': '本角色原路線後讀釐清，保留早期判斷'})
            for idx in bounded_links.get((pid, unit_index), []):
                evidence.append({'reference': bounded[idx][0], 'scope': '原判收件後source21真有界觀察'})
            for idx in fold_links.get((pid, unit_index), []):
                evidence.append({'reference': fold['notes'][idx], 'scope': '原判收件後折疊補讀，非首次正文'})
            if ('圖' in concept['verdict'] or '視覺' in concept['verdict']
                    or '版面' in concept['verdict']):
                evidence += [{'reference': old_visual_ref, 'scope': '原判後真實看圖/頁面之指定範圍'},
                             {'reference': visual_ref, 'scope': 'source21變動位置及未改圖SHA沿用邊界'}]
            if pid == 'training' and unit_index in (69,70,72):
                evidence.append({'reference': aux_ref, 'scope': '三附頁真首讀後另有界報告；不算native370'})
            resolutions.append({
                'note': note_ref, 'concept_index': concept_index,
                'unit_index': unit_index, 'concept': concept['concept'],
                'original_verdict': concept['verdict'],
                'original_observation': concept,
                'first_observed_at': note['saved_at'],
                'discovery': 'independent_ai_first_read',
                'final_status': 'resolved' if (pid,key) in resolved_keys else 'nonblocking',
                'reason': reason, 'evidence': evidence,
                'not_claimed': '未執行課程命令/模型訓練、未讀未列先備或外部原碼；後補不改首次時點。'
            })
    assert expected_keys == set(reasons[pid]), (pid, expected_keys ^ set(reasons[pid]))
    total += len(resolutions)
    src = source_map[pid]
    closure_pages.append({
        'page_id': pid, 'reviewer_task': TASK, 'source_sha256': src['source_sha256'],
        'figures_sha256': src['figures_sha256'], 'verdict': 'pass',
        'checkpoint_resolutions': resolutions
    })
assert total == 108, total
closure_ref = save('reader-checkpoint-closure-source21-v1.json', {
    'schema_version': 1, 'kind': 'reader_checkpoint_closure', 'reviewer_task': TASK,
    'created_at': now(), 'source_freeze': freeze_ref,
    'basis_report': current_ref, 'original_report': original_ref,
    'pages': closure_pages,
    'counts': {'original_native_notes': 138, 'literal_non_adequate_concepts': total,
               'canonical_pages': 7},
    'method': '逐一處理原notes所有verdict不精確等於已足夠的concept，包括本來已足夠但有限定的判斷；original_verdict與原觀察/保存時點照存。resolved是修後或後讀釐清，nonblocking保留本頁定位以外或可選負擔，不宣稱未知技術已驗。',
    'collection_policy': 'created_at是真建立時刻；不虛填received_at或stage seal，交由上游真解析收件。'
})

handoff = {
    'schema_version': 1, 'kind': 'reader_source21_handoff', 'reviewer_task': TASK,
    'created_at': now(), 'original_report_unchanged': original_ref,
    'source_freeze': freeze_ref, 'basis': basis_ref, 'current': current_ref,
    'checkpoint_closure': closure_ref, 'auxiliary_separate': aux_ref,
    'folded_supplement': fold_ref, 'reading_times_complete_inventory': estimate_ref,
    'source21_visual_addendum': visual_ref,
    'pages': []
}
for i,p in enumerate(original['pages']):
    handoff['pages'].append({
        'page_id': p['page_id'], 'report': current_ref, 'json_pointer': f'/pages/{i}',
        'original_report': original_ref, 'original_json_pointer': f'/pages/{i}',
        'basis': {'report': basis_ref, 'json_pointer': f'/pages/{i}'},
        'checkpoint_closure': {'report': closure_ref, 'json_pointer': f'/pages/{i}'},
        'sessions': p['sessions']
    })
handoff_ref = save('handoff-source21-v1.json', handoff)
for name, value in [('basis', basis_ref),('current',current_ref),('closure',closure_ref),
                    ('auxiliary',aux_ref),('visual',visual_ref),('reading_times',estimate_ref),
                    ('handoff',handoff_ref)]:
    print(json.dumps({'kind':name,**value},ensure_ascii=False))
