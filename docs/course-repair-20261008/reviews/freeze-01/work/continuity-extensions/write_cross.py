import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone

B = Path('docs/course-repair-20261008/reviews/freeze-01')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
M = json.loads((B / 'manifest.json').read_text())
P = {p['page_id']: p for p in M['inventory']['pages']}
gate_path = B / 'checks/continuity-source-all-sealed.json'
G = json.loads(gate_path.read_text())
assert len(G['reports']) == 6 and G['manifest_sha256'] == sha(B / 'manifest.json')

names = ['reader-extensions-main', 'reader-extensions', 'technical-extensions-initial', 'continuity-extensions-initial']
documents, identities, immutable = {}, {}, {}
for name in names:
    path = B / 'reports' / (name + '.json')
    seal = B / 'reports' / (name + '.seal.json')
    doc = json.loads(path.read_text())
    receipt = json.loads(seal.read_text())
    assert sha(path) == receipt['sha256']
    assert doc['reviewer'] == receipt['reviewer']
    documents[name] = doc
    identities[name] = {'path': str(path.relative_to(B)), 'sha256': sha(path),
                        'reviewer': doc['reviewer'], 'seal_path': str(seal.relative_to(B)),
                        'seal_sha256': sha(seal), 'sealed_at': receipt['at']}
    immutable[path] = sha(path)
    immutable[seal] = sha(seal)

own = documents['continuity-extensions-initial']
reader = documents['reader-extensions']
main = documents['reader-extensions-main']
tech = documents['technical-extensions-initial']
main_notes = B / 'reports/continuity-extensions-main-notes.json'
main_seal = B / 'work/continuity-extensions/main-notes-seal.json'
own_main_receipt = json.loads(main_seal.read_text())
assert sha(main_notes) == own_main_receipt['sha256']
immutable[main_notes] = sha(main_notes)
immutable[main_seal] = sha(main_seal)
own_gate_entry = next(r for r in G['reports'] if r['group'] == 'extensions')
assert own_gate_entry['sha256'] == identities['continuity-extensions-initial']['sha256']
assert own_gate_entry['receipt_sha256'] == identities['continuity-extensions-initial']['seal_sha256']

report = {
    'schema_version': 1,
    'group': 'extensions',
    'reviewer': '/root/repair_continuity_extensions',
    'role': '原獨立銜接／累積負擔審閱者的交叉覆核；AI來源審閱，非真人學習實測',
    'at': datetime.now(timezone.utc).isoformat(),
    'gate': {'path': str(gate_path.relative_to(B)), 'sha256': sha(gate_path), 'at': G['at'],
             'meaning': '先確認六組初判封存；只讀本組報告，不讀其他組判斷。'},
    'manifest_sha256': sha(B / 'manifest.json'),
    'criteria_read_and_applied': {k: M['criteria_sha256'][k] for k in
        ['SKILL.md', 'references/review-protocol.md', 'references/calibration.md']},
    'cross_instructions': {'path': '/workspace/work/tutorial-repair-20261008/cross-role-instructions.md',
        'sha256': sha(Path('/workspace/work/tutorial-repair-20261008/cross-role-instructions.md'))},
    'read_initial_and_seal_identity': identities,
    'own_main_identity': {'path': str(main_notes.relative_to(B)), 'sha256': sha(main_notes),
                         'seal_path': str(main_seal.relative_to(B)), 'seal_sha256': sha(main_seal)},
    'actual_read_scope': {
        'assigned_pages': M['groups']['extensions']['pages'],
        'peer_report_sections_read': [
            'reader-main／reader-final：逐頁原句承諾、深度、四題及最短推論、未知、問題與限制；原main四題與final保存本比對一致。',
            'reader-final：逐頁complete_page四題、所有extras的Q2機制／需要、Q4、後來釐清、未知及問題；非零details頁另讀新增理解、引文與限制，training九區摘要及mandatory_route_validation。',
            'reader全組五項issues、全域限制、必要前文與追加路線記錄。',
            'technical：逐頁Q2機制／用途、已教／仍缺關係、findings與unknowns_limits；44項checks的check_id、核查結果、執行程度及未驗範圍；全域方法及限制。',
            '本人main notes與initial的既有實讀、選讀、圖、原位截圖及未知，仍以各自封存時判斷為原初判。'
        ],
        'not_claimed_read': '未逐字讀technical的151條引文登錄或131個原始證據檔，未讀reader逐單位原始notes全文；不是技術證據再執行或另一輪獨立首讀。',
        'source_rechecks': [
            {'page_id': 'training', 'scope': '本人已實讀delivery的T.6–T.7原句與命令，以及T.1、T.10既有讀取', 'prompt': 'reader findings之後重核；非獨立再次發現'},
            {'page_id': 'readme', 'scope': '環境連結原句；既有environment、W.1及README_en來源定位', 'prompt': 'reader findings之後重核'},
            {'page_id': '1.15', 'scope': '本階段新讀必要前文main全文，未開details／未看其圖或實作',
             'source_sha256': P['1.15']['source_sha256'], 'prompt': 'reader temperature-zero finding提示；continuity初判未曾讀此頁'}
        ],
        'output_handling': '過長頁按指定頁與欄位分段讀；先前三頁合併輸出截斷後分頁重讀。廣泛rg重核輸出亦有截斷，未把其未顯示匹配列當新讀取；相關原delivery已於來源階段分段讀完。',
        'prohibited_sources_not_opened': '作者答案／紀錄、root裁定／diagnosis、舊audit／問題清單、其他group報告均未開；本階段不讀實作或重跑模型。'
    },
    'pages': [],
    'finding_dispositions': [],
    'own_added_materials': [],
    'new_findings': [],
    'grading_differences': [],
    'unknowns_and_limits': [],
}

def page(pid, quotes, mechanism, need, operation_scope, figure_review, limits, finding_ids=()):
    p = P[pid]
    assert sha(Path(p['snapshot'])) == p['source_sha256']
    i = M['groups']['extensions']['pages'].index(pid)
    r = next(x for x in reader['pages'] if x['page_id'] == pid)
    rm = next(x for x in main['pages'] if x['page_id'] == pid)
    t = next(x for x in tech['pages'] if x['page_id'] == pid)
    assert r['main_four_question_review'] == rm['four_question_review']
    report['pages'].append({
        'page_id': pid, 'source_sha256': p['source_sha256'], 'figures_sha256': p['figures_sha256'],
        'initial_references': {'own': '/pages/' + str(i), 'reader_main_preserved_exactly': True,
                               'technical_check_ids': [c['check_id'] for c in t['technical_checks']]},
        'source_quotes': quotes,
        'Q2_mechanism_property_review': mechanism,
        'Q2_need_use_review': need,
        'identity_operation_example_support': operation_scope,
        'figure_current_objects_review': figure_review,
        'unknowns_and_deferred_use_review': limits,
        'finding_ids': list(finding_ids),
    })

page('A.2', ['檢索負責找資料，生成負責讀資料回答；來源編號用來回查。',
            '日期只是支持版本的另一欄，不應拿來代替地址。'],
     '已教：當前問題＋取回原公告可供生成讀取；A.1已區分本次輸入與更新權重。欄位展開不是模型答案。同行技術短例僅支持提示構造，與本人的語義判斷相稱。',
     '已教：訓練時未知的搬遷資料、文件多時須先取相關段；原句「有資料可用」未保證一定答對。A.3承接實際字詞排序，當下可延後檢索演算法，不是需讀實作才能補上理由。',
     '手寫公告的青街8號〔公告1〕是期待；正式隨機模型12店測例分開。基線5/12、改地址7/12但雙對5/12、改來源双對2/12，個別新公告成功不能當整體可靠。原提示有／無、改地址與編號可核兩種角色。',
     '原SVG及640／360已於初判視驗：公告1→原文加問題→期待地址＋來源→回查，日期不代地址；孤立圖可讀不等DOM通過。',
     'reader未驗原結果；technical有原記錄成對重算但未新訓練／新12題推論。本人不重核原檔；不把來源讀到的數字與同行查證合成兩次獨立驗證。')

page('B.3', ['這裏要區分兩種表示。', '工具成功還不是任務完成。'],
     '已教：人工request先解析／驗證，再真工具5535；外層assistant/tool trace不能直接當render_chat模板角色，改成user文字TOOL_RESULT後才讓模型再生成。B.1／B.2與7章的已讀前文足夠。',
     '已教：回填真資訊讓最後回答有工具結果可讀；留各站來源才能分辨請求、計算與回讀失敗。multiply(9,9)返回81但答8／JSON壞的反例回答最低需要，不須另加代理全面比較。',
     '短程式只建紀錄，不產最後答案；9+9是另一個舊正式模型完整往返。technical實跑短例與b=46及重建舊ID，支援這些對象邊界，不證123×45的模型能力。',
     '原SVG和兩尺寸的五站已追：人工／模型示例請求→parse+validate→真工具→user TOOL_RESULT→done+answer。圖末是工作站，不把短例當已完成生成。',
     '逐字重做正式system／角色／前綴可讀details；此為完整配方的條件式前置，基本回填主文已足夠。歷史messages污染需勘誤，真ID與新模型能力分開；本人未開原JSON／CLI。')

page('B.4', ['done也算一個。', '56/56停止率仍不能替代20道正常任務的19/20成功率。'],
     '已教：工具之後還需一次動作交done；EOS結束單次生成、done結束任務、預算由外程式控制。此頁trace每運算一筆與B.3兩訊息並非相同計量物。',
     '已教：算完卻没回答不能默認成功，四種終點保留不同失敗原因；max_steps從1到2增加回答餘額，不增加工具數。由已教流程可正常推得，未有理由缺口。',
     '人工run清單測流程而非模型預先計劃；done answer=6只完成訊號、不證答對。1/1/1/0工具數、56=36+2+18生成與20題分母分開。technical單測／重算支援，未新端到端任務。',
     '零圖頁：四組清單、status和計數足夠追踪當前停止對象，不必靠想像才能判完成。',
     'details新增正式協議合法done／真工具／COPY不工具的計分，沒有改主文四終點。reader未核歷史；本人保留只依來源理解，不稱56次生成已親自驗。')

page('B.6', ['TOOL…加EOS，共五個有效目標。', '沒有把新選卡器與舊JSON生成模型串接成已驗證的助手。'],
     '已教：-100排除位置的直接代價仍保留可讀user前文；assistant四ASCII字母＋EOS才是五目標。自然題／狀態變化避免只靠CALC/COPY標頭。技術132輸入／5目標核查未替代已讀7章關係。',
     '已教：B.5依任務、資訊與工具狀態選DIRECT／TOOL／ASK；把人填卡改為模型輸出需含多類對照。這是教材策略，不宣稱所有小加法本質上必用工具。',
     '短程式沒有訓練；人工完整策略system與正式僅狀態system分開。新隨機選卡權重及舊JSON權重不同，96/96原樣與42/48改寫、更小需工具0/6子集不能拼成助手成功率。',
     '零圖頁：messages／labels與既有X、有效Y、EOS前文足夠；動作卡與工具執行是不同對象。',
     'B.7可承接正式配方／驗收；來源初判只讀B.7 main，reader的extras後續較多，技術則重算原records；本人未補讀B.7 details。Toolformer屬方法聯繫，当前策略不靠論文內算法才能懂。')

page('C.7', ['advantage = reward - baseline', '希望提高這個動作的機率，但不是一次變成必選。',
            '這是24道不同題目的重複抽樣，不是384道新題。'],
     '已教：人工選答4，A=+0.5；減少−A log p使該動作更常，±0.25梯度、±0.025新分數、0.487503／0.512497概率按答3／答4順序。1.12與13.11主文提供更新及相對參考規則，反向不等更新。',
     '已教：讓較參考好的選擇更常、較差的更少；baseline是真值以外的相對參考。適當baseline可減波動只屬概念限定，這兩分數例未量減波動。嚴格reward驗正誤是C.6／正式小策略的目標，不能把它回填為唯一主例承諾。',
     '主例固定人工action、reward／baseline常量，不取驗證器梯度、未抽樣或訓語言模型。details18輸入、17動作、REINFORCE＋AdamW與TinyLM另模型；24×16=384樣本、代理滿分嚴格0揭示漏洞，不證內部推理。tech正／負reward短例及原記錄重算與此相稱。',
     '原SVG、640／360及原位1280／390捕捉圖已看：分數與概率兩行、前後答3／答4順序固定；圖是可追的人工兩數更新，非抽樣／語言模型訓練圖。',
     '不把適當baseline的泛化效用當本例實測；C.7公式可支持自加相對參考小變化，另列材料。未驗新長訓練、真實語言RL或頁面互動。')

page('training', ['各節是不同主題的操作入口…不必一次跑完所有實驗。',
     '先選要開放哪些參數更新。', '這個命令列工具的partial另允許首末文字區塊更新。',
     '此路線必須先打開並完成…sft、style、moe三組。'],
     '有條件保留：T.1家族切分；T.2 backward／--train分開；T.3窗口及存檔格式；T.4 X可讀／Y計分與分母；T.5各自目標；T.6各凍結範圍／有輸入梯度才學；T.7固定參考非真值；T.8同寬MoE非同成本；T.9 codes／scale／浮點bytes非執行記憶體；T.10架構縮小與教師訊號分離；T.11指紋非正確性，均有原句或已讀前文可推。T.7零溫度接口約定另缺，不能概括為全部操作已足夠。',
     '有條件保留：材料、通路、參數更新、相同條件及留出檢查各有用途；量化保存和品質分問，蒸餾大小與教師效益分問。T.6要求選開放範圍卻未交代joint選partial而其他选projector的目的／示範性質；機制通過不能抹去此最低選用缺口。',
     'T.1–T.9是各自小路線；T.10換固定教師和配對切分，主文不許用mini或inference匯出代替。讀完T.4／T.5／T.8固定區後可定位三model.pt＋三dataset.json，原main的「待補讀」只在後續階段解除。示意AB表、4/5→3/5非實跑；technical三dry-run不證長配方／一般能力。',
     '零SVG頁：文字表／欄位／父檔路径足以辨對象；已看桌面三固定區與手機捕捉圖，手機命令右端仍未橫捲／複製，DOM、錨點自動展開與實際完整命令入口未驗。',
     '技術T-4核CLI與固定vqa的partial各自範圍，不回答本機joint選用原因；T-5核DPO而未核零溫度。其餘短測／記錄／metadata有限支持不当长GPU配方、全題新推論、裝置性能實測。名称角色已足夠，跳讀縮寫改善及T.10摘要标示均可選。',
     ['extensions-training-names', 'extensions-training-joint-freeze-purpose', 'extensions-training-temperature-zero', 'CE-OPT-1'])

page('readme', ['各版本的顯卡與驅動限制、Colab和鏡像設定見環境說明。',
     '也不必一路訓練同一個模型。', '不是本版成品的成績。'],
     '入口深度已教：字元ID／接字表→各局部機制；網站讀、Notebook改、正式訓練另入口；同extra保持PyTorch配套。固定v2與20章上游不同來源，推論檔無optimizer／RNG不能精確續訓。',
     '課程／試用／重做目的與有限範圍清楚；環境リンク承諾鏡像設定，實讀environment未給做法／具體定位，這項使用目的尚不能沿链接完成。Colab則environment導向小節入口／W.1，而已讀W.1有登入、CPU、全部執行、先準備工具；不用要求環境頁再寫完整流程。',
     'clone／cd／sync／check_env／kernel／Jupyter及默認只forward/backward原操作可理解，不證各平台新實跑。19章有限v2、舊30組120權重、舊11產物與20章成熟成品另列；技术库存/固定metadata核查支援版本限定，未載weights。',
     '原圖上貓看狗，下狗看貓；ID 3,2,1,4／1,2,3,0與雙向查回已看，大小非字義。沒以孤立圖正確推原README頁面點擊已驗。',
     'Colab實runtime未驗、MPS僅矩陣而非整課；外部操作鏈、CI及歷史驗證本人未開。鏡像承諾需收窄或給真位置，非要求增設未曾承諾的整套部署教材。',
     ['extensions-readme-environment-mirror-route'])

page('environment', ['先選自己要做的事情，再準備對應工具。', 'uv run會先同步環境。',
     '沒有一般逐字聽寫或語音輸出。'],
     '已教：CPU／CUDA配套與驅動相容、互斥extra、兩venv及uv同步行為；自訓聲音直接特徵與成熟Whisper轉文字分開；LoRA仍依賴底座。',
     '已教：按閱讀／短例／v2試用／正式训练／成熟成品選工具，避免全下載或版本遷就；cu126／cu130非慢快鈕。鏡像設定不在這項已教內容中，README承諾須另審，不把環境頁本身擴張成鏡像教程。',
     '任務表與根目录sync/run、官方相容位置可追；8chat＋1append是既有通路示例，不是未知題率。12字OCR、連續1–4字ROI、訓練已見字型、3聲音意圖的范围有對應，不能推一般OCR／逐字聽寫。',
     '零圖：工具箱、裝置、格式與接口的明確文字足以選眼前路線。',
     'hardware最低需求、所有裝置／外鏈／安裝未由本人验；technical用原CPU／CI記錄、官方相容資料及固定artifact查證，不是新MPS／CUDA／Colab驗收；镜像未教保持不通過。',
     ['extensions-readme-environment-mirror-route'])

page('curriculum', ['本稿區分三類目標。', '這次不累積舊梯度。',
     '採MoE是為了實際展示選專家與整合，不假定它在小型硬體上必然比Dense省時或省記憶體。'],
     '已教：可訓Dense基準→對話→不同模態入口及局部比較，19章才追同底座；圖X／可見mask與Y／計分loss分開，zero_grad清舊梯度→backward填grad→step更新。',
     '已教：單一公開路線避免雙維護；不同方法分各自目標，架構比較分成本，量化保存與蒸餾學生訊號分開。MoE有展示專家／整合的最低理由及實際成本限定，不能類推T.6的joint選partial也已有理由。',
     '上游固定版本与既有課綱研究是記錄的設計參照，未執行其完整訓練；舊90題聚合與v2不同，final目标／選定歷史不同不能當受控Dense比較。原代碼配方入口與研究對照不是全面事實複驗。',
     'SVG／640／360的raw sample→roles IDs→X/Y→一次更新已追，旁文說明非梯度累積；原位捕捉圖浮動工具條遮首節點／局部X/Y是初判靜態觀察，沒實際滾動，不列確定持續缺陷也不計頁面通過。',
     'technical核資料及圖的計量／原固定來源並非真DOM檢查；Harvard預定內容、台大鏡像證據、影片未看與當時代理限制的歷史限定仍保留；未重新查證外部來源現況。')

page('readme-en', ['Use python scripts/fetch_training_assets.py --list to choose and unpack them.',
     'Execution in a real Colab runtime has not been verified.',
     'It checks loading and execution; task accuracy comes from the separate held-out evaluation.'],
     '入口／版本／工具與權重身份的有限說明與中文一致；default無更新／存檔，--train才訓；下載檢查與任務準確率分別。--list的清單用途不能直接推成解包動作，須分操作对应。',
     '已教閱讀、CPU動手、试成品與重做各路线理由；想取得固定資料的讀者照唯一--list只知清單，取包步未由此句交代。训练T.1已经给实际下载命令，可用最短回链而不增加独立下载教程。',
     '287＋26=313、四匯出16files與中文版本一致；成功例不等全criterion，旧成绩非v2。安装／维护／外链入口文本可读但本人未执行；英语閱讀在中文之后，不是独立英文零背景测试。',
     '零圖；不為入口各方法名額外要求架構圖。',
     'technical同入口库/metadata/历史核查未验--list动作用语；未下載、解包或运行CI。问题是原句承诺與training已教动作不符，不據此說目前脚本坏了。',
     ['extensions-readme-en-list-unpack'])

page('tool-trace-errata', ['generation.input_ids是實際送入模型的編號。',
     '這份補件是從舊紀錄重建的文字對照，並非重新執行模型的觀察。'],
     '已教：保存同列表別名後續append會污染早messages；input_ids與按過去事件重建文字另存，逐ID／總數匹配检查对照。deepcopy用于将来保存当时快照，已读Python／13.4独立副本可支持最低推论。',
     '已教：要判模型當時讀了什麼就不能把未来訊息回填当过去输入；保留原答案、工具、ID與19/20，修记录不冒充新能力实验。',
     '来源说明56生成、36文字差异及全部ID匹配；technical独立重建并跑离线CLI，支持这一输入核验且不新生成。本人只读同行结果／教学正文，不称亲自核56数据，也不由修复推成功率上升。',
     '零圖；原欄位、時序與副本关系已由文字足以理解，未读实现不妨碍理解主承诺。',
     '當查看舊B.3/B.4记录时为条件式必要勘误，原页已公开限定；实际JSON／补件真实性由technical范围支持，未新生成验证未来deepcopy trace。')

reader_issues = {x['id']: x for x in reader['issues']}
def disposition(fid, recommended, understanding, benefit, change, burden, limits, prior_difference):
    if fid in reader_issues:
        issue = reader_issues[fid]
        origin = {'role': '首次閱讀', 'reviewer': issue['discoverer'],
                  'main_report': identities['reader-extensions-main']['sha256'],
                  'complete_source_report': identities['reader-extensions']['sha256'],
                  'original_severity': issue['severity'], 'original_type': issue['type'],
                  'original_location': issue['location'], 'original_quote': issue['quote'],
                  'extras_result': issue.get('after_extras'), 'old_issue_prompted': False}
    else:
        issue = next(x for x in own['optional_findings'] if x['id'] == fid)
        origin = {'role': '銜接／累積負擔', 'reviewer': report['reviewer'],
                  'main_report': sha(main_notes), 'complete_source_report': identities['continuity-extensions-initial']['sha256'],
                  'original_severity': issue['severity'], 'original_type': issue['type'],
                  'original_location': issue['position'], 'original_quote': issue['quote'],
                  'old_issue_prompted': False}
    report['finding_dispositions'].append({'id': fid, 'initial_origin': origin,
        'recommended_disposition': recommended, 'cross_understanding': understanding,
        'concrete_benefit': benefit, 'minimum_change': change, 'added_reading_burden': burden,
        'unverified_scope': limits, 'independent_initial_difference': prior_difference,
        'independence': '本階段見同行之後的理解不算另一份獨立重現；只有CE-OPT-1本來是本人獨立初判。'})

disposition('extensions-training-names', '保留選讀改善；不列必修、不要求每頁重講',
    'training已給窗口可讀幾字、助手目标及有效位置平均代价。7.1已读SFT身份，不能把未在本页展开名字升级成主线机制缺漏。reader没有回读MLP原章，故其跳读名字联系改善是真正可选用途。',
    '让直接跳training的读者把短名与章节/外部名称对应，现有操作/分母理解并不靠补名。',
    '只在本页首次必要出现加短括注或精确已有章节定位；SFT／MLP／NLL不重复内算法、无须改各章。',
    '一到三个短括注；若前文/定位已足够则可保留原文。',
    '本阶段未扩读MLP原章名称历史，未评估真人跳读；不得据此声称这些名字全书未教。',
    '本人原初判无此可选项；交叉认可它对跳读的有限收益，技术无finding不反驳这项名字收益。')

disposition('extensions-training-joint-freeze-purpose', '建议必要修正，保留增加理解負擔分级',
    '更新范围已经教会，缺的是为何此joint配方开放首末块、单模态仅接头，或者明确这只是选择的示范设置。原句先选加三个命令要求读者判断选用，却没有选用目的；输入两个模态本身推不出必需partial。',
    '使读者能解释示例在选择什么，并避免把partial当joint唯一合法/必需设置。',
    '紧接三命令补一句真实的选用目的与条件；若只为示范，直接说明这是本练习选定设置、不是joint必须如此。不给未经验证的projector无法完成断言。',
    '一句，保留三方案机制及现有练习；无须追加全参数实验。',
    '未重训/projector对照，不断言哪方案必优；technical只验trainable mapping、不提供此目的。',
    'reader独立报必要；本人原T.6已足够判断没有单列选用理由，是初判漏项。此后接受精确缺口，不回填initial。')

disposition('extensions-training-temperature-zero', '建议必要修正，保留增加理解負擔分级',
    '前后固定生成规则的需要已经教过。新读1.15明说用正数T除分数，且argmax与sampling分开；不能由正T规则或低温直觉推出脚本0的特殊约定。缺的是0实际使用规则，不是温度整套数学。',
    '读者能说清这条固定条件选择怎样的输出规则，避免按已教除法误读0。',
    '先核infer.py零值分支，再于两命令旁补一句实际选择规则和正温度区别；若实际为argmax，则明确0走greedy、没有执行除0。此处不把未验候选句当脚本事实。',
    '一句加原1.15回链即可；不新增抽样算法长课。',
    '本人和本组technical都未核此CLI零值分支，实际零行为仍未验；不声称命令会错误/一定greedy。',
    'reader独立报必要；本人原T.7概括通过漏这项接口约定，本阶段在提示后核前文支持缺口，不算独立重现。')

disposition('extensions-readme-environment-mirror-route', '建议必要修正镜像链接承诺；收窄Colab部分',
    'README明确承诺镜像设置，完整environment仅给任务/工具箱/相容规则，无镜像操作或具体位置。Colab有environment→小节入口/W.1以及W.1登入CPU全部执行路线，不能因environment未单独重复流程就再报Colab必修。',
    '需要镜像的读者不会沿错误承诺跳完后仍找不到目标。保留现有可追Colab路线，减少重复内容。',
    '最小为README链接说明改成environment实际提供的设备、驱动与环境选择；删除该链接的镜像设置承诺。只有真有当前必要指引时才改连真实位置。',
    '改一个导航句，无需新增镜像教程或重复W.1。',
    '未运行Colab/验证外链现况；不声称课程任何别页绝无镜像指导，只确认此承诺指向页未提供。',
    'reader独立报必要且含Colab观察；本人原readme/environment未报。交叉支持镜像关系但不支持扩大为缺完整Colab独立教程。')

disposition('extensions-readme-en-list-unpack', '建议必要修正，保留增加理解負擔分级',
    '英文唯一命令称choose and unpack；training原句--list列名称与大小，第二行--asset tinystories才取包核对放入data/training。不能从已有清单操作推出解包动作。技术的入口数值/格式核查没有涵盖这句操作对应。',
    '让英文读者从看清单接到实际取起步包，不把目录输出当解包完成。',
    '分为--list查看名称大小；以原T.1已给--asset tinystories示例说明取得起步包，并链接Training T.1操作。无需自行承诺未读asset指南的完整步骤。',
    '一到两短句或一个明确回链；不写另一套下载说明。',
    '未运行下载/解包，判断仅为教学文本动作合同；不推脚本本身故障。中文类似“查看并按需解包”可同步澄清，但本次不另升级为新必要项。',
    'reader独立报必要；本人原英文入口通过漏单条命令动作，交叉源句核对后接受，不覆写原判。')

disposition('CE-OPT-1', '保留低优先级选读改善；原语义路线可保持',
    'T.10明说必须开T.4/T.5/T.8且不许mini替代；三固定区实读后能定位同root下三教师与三split。reader后读与technical T-8只支持这条文字/格式路线，没有把原main待读改为初读已知。',
    '少一次“选读”摘要与选定T.10必做关系的确认，对未选T.10者无收益。',
    '可仅在三个summary标“选T.10时需完成对应组”；不搬完整配方到主文，不改变全课选读性质。也可保留原句，因为T.10已明确授权回读。',
    '三个极短路由标签；文字完整无新增概念。',
    '未点锚/自动展开、手机横捲/复制或真跑CUDA配方；标签不能代替这些操作验证。',
    '本人独立初判已有可选，后续extras保留。reader／technical未报为必要，与此低优先级判断一致。')

report['own_added_materials'] = [
    {'id': 'CROSS-EXT-V1', 'page_id': 'C.7', 'origin': '本人在交叉阶段提出；同行A=0变体启发后，另选一组材料，不是原来源已有例或独立发现',
     'material': '仍固定答4、原分数[0,0]，改reward=0.5、baseline=1.0，学习率0.1。',
     'prediction': 'A=-0.5；沿原规则loss的动作梯度反向，updated为[+0.025,-0.025]，答4概率下降而答3上升。',
     'shortest_reason': '正reward不等高于参考；reward−baseline<0乘−logp后的梯度方向与原A=+0.5相反。这验证相对参考身份及图两列顺序，不把它当能力或baseline减方差证据。',
     'verification': '来源公式纸上代入；未新执行、抽样或训练。'},
    {'id': 'CROSS-EXT-V2', 'page_id': 'training', 'origin': '本人在reader选用理由提示后的关系查漏，不算独立重现',
     'material': '保持T.6 joint素材、问题和目标square,low，将示例--freeze partial改为projector。',
     'prediction': '依已教CLI定义可预测可更新集合缩到接头；仍有同样输入/目标。但无法从来源判断这样是否符合本示范目的，或partial为何被选。',
     'shortest_reason': '范围定义回答更新谁，不回答为何开放这些。两模态不逻辑蕴含必须更新首末语言块；不能凭想像补目标／结果。',
     'verification': '仅文字契约变化，未运行/重训，不预测成功率或scheme优劣。'},
]
report['grading_differences'] = [
    {'ids': ['extensions-training-joint-freeze-purpose', 'extensions-training-temperature-zero',
             'extensions-readme-environment-mirror-route', 'extensions-readme-en-list-unpack'],
     'difference': 'reader独立报增加理解负担；本人原初判未报、technical原未报。交叉支持四项精确缺口，保留已教机制与技术检查有限通过；不改三份初判bytes。'},
    {'id': 'extensions-readme-environment-mirror-route',
     'difference': '支持镜像错承诺；不将environment缺单独Colab流程另算必要缺口，已有W.1的可追基本操作路线。'},
    {'ids': ['extensions-training-names', 'CE-OPT-1'],
     'difference': '两项只保留可选收益；不把缩写或原已显式要求的折叠路线升级为必修。'},
]
report['unknowns_and_limits'] = [
    '全部初判、main notes及seal保留原bytes。本报告接受同行提示后的源句判断，不当额外独立首读；技术事实支持明确来自technical原报告，本人未读取其原证据再证。',
    'T.7零温度实际接口仍未验；T.6 freeze替代方案质量/作者实际选用目的未知；不能据未知编造方案必优或程序故障。',
    'T.10完整GPU路线/模型产物及外链、下载、平台、全题推论未由本人执行。technical的短例、dry-run、record重算、metadata各自有限，不升级为完整运行通过。',
    'curriculum静态捕捉图遮挡、手机固定配方命令右端、锚点/展开/横捲/复制未验证。原SVG和PNG可读不等完整页面可用，不把未知当缺陷也不当通过。',
    '入口数字、历史研究、原实验及不同模型/目标/分母的限定保留；不以24题384抽样、56生成20任务、96/96选卡或旧成品替代当前一般能力。',
    '此为有限AI来源交叉覆核，没有测真人学习负担/时间；英文已在中文之后读，不是独立英语首读。',
]
report['overall_recommendation'] = '四项必要修正、两项低成本可选改善；主要机制、对象转换、计量与例子边界的通过仍限于逐页所列。新增确定问题清单为空；四项必要均为reader原发现的提示后支持，未冒称本人独立重现。教材未改，等待root裁定。'
report['finding_counts'] = {'reader_initial': 5, 'technical_initial': 0, 'continuity_initial': 1,
                          'dispositions': 6, 'necessary_recommended': 4, 'optional_retained': 2,
                          'new_findings': 0, 'own_added_variations': 2}
assert [p['page_id'] for p in report['pages']] == M['groups']['extensions']['pages']
assert {f['id'] for f in report['finding_dispositions']} == set(reader_issues) | {'CE-OPT-1'}
assert sum(len(p['technical_checks']) for p in tech['pages']) == 44
for path, digest in immutable.items():
    assert sha(path) == digest, 'Original immutable report changed: ' + str(path)

import sys
sys.path.insert(0, str(B / 'work/continuity-extensions/pydeps'))
from opencc import OpenCC
cc = OpenCC('s2t')
path = B / 'reports/cross-extensions.json'
seal_path = B / 'reports/cross-extensions.seal.json'
assert not path.exists() and not seal_path.exists(), 'Refuse to overwrite sealed cross review'
path.write_text(cc.convert(json.dumps(report, ensure_ascii=False, indent=2)) + '\n')
seal = {'reviewer': report['reviewer'], 'at': datetime.now(timezone.utc).isoformat(),
        'path': str(path.relative_to(B)), 'sha256': sha(path), 'group': 'extensions',
        'kind': 'source cross review after all three roles initial sealing',
        'page_count': len(report['pages']), 'disposition_count': len(report['finding_dispositions']),
        'source_initial_preserved': identities,
        'main_notes_preserved_sha256': sha(main_notes)}
seal_path.write_text(json.dumps(seal, ensure_ascii=False, indent=2) + '\n')
for original, digest in immutable.items():
    assert sha(original) == digest
print(json.dumps({'report': str(path), 'sha256': sha(path), 'seal': str(seal_path),
                  'seal_sha256': sha(seal_path), 'pages': len(report['pages']),
                  'finding_counts': report['finding_counts'],
                  'all_read_original_report_and_seal_bytes_preserved': True}, ensure_ascii=False, indent=2))
