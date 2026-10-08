import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter

B = Path('/workspace/work/tutorial-audit-20261008')
S = B / 'freeze/sources'
R = B / 'reports'
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(name):
    return json.loads((R / name).read_text())
reader_main = read('reader-extensions-main.json')
reader = read('reader-extensions.json')
technical = read('technical-extensions-initial.json')
own = read('continuity-extensions-initial.json')
ownseal = read('continuity-extensions-seal.json')
mainseal = json.loads((B / 'work/continuity-extensions/main-notes-seal.json').read_text())
assert sha(R / 'continuity-extensions-initial.json') == ownseal['sha256']
assert sha(R / 'continuity-extensions-main-notes.json') == mainseal['sha256'] == ownseal['main_notes_sha256']

origin = {}
for filename, report in [('reader-extensions.json', reader), ('technical-extensions-initial.json', technical), ('continuity-extensions-initial.json', own)]:
    for issue in report['issues']:
        origin[issue['id']] = {'report': filename, 'original_severity': issue['severity'], 'original_quote': issue.get('quote', issue.get('original_quotes')), 'original_discovery': issue.get('discovery_phase', issue.get('discovered_at', issue.get('discovery_timing')))}

def ev(page, fragment):
    p = S / (page + '.md')
    lines = p.read_text().splitlines()
    hits = [(i + 1, line) for i, line in enumerate(lines) if fragment in line]
    assert hits, (page, fragment)
    n, line = hits[0]
    return {'page_id': page, 'source': str(p), 'source_sha256': sha(p), 'line': n, 'quote': line}

entries = []
def add(id, ids, pages, evidence, promise, prerequisites, mechanism, need, gap, necessity, scope, disposition, benefit, minimum, burden, timing='互見後來源覆核；不是新增獨立初判'):
    entries.append({'id': id, 'origin_ids': ids, 'origins': {x: origin[x] for x in ids}, 'affected_pages': pages, 'strongest_source_evidence': evidence, 'original_promise': promise, 'actual_prerequisite_basis': prerequisites, 'mechanism_judgment': mechanism, 'need_judgment': need, 'remaining_minimum_gap_or_counterevidence': gap, 'necessity': necessity, 'necessity_scope': scope, 'disposition': disposition, 'concrete_benefit': benefit, 'minimum_change': minimum, 'new_burden': burden, 'repair_scale': 'paragraph' if disposition == '採用' else 'none', 'candidate_repair_scale': 'paragraph', 'page_rewrite_needed': False, 'chapter_rewrite_needed': False, 'order_change_needed': False, 'timing_and_visible_information': timing})

add('CXE-01', ['EXT-A1-NAME'], ['A.1'], [ev('A.1','SFT會更新權重'),ev('7.11','監督式微調SFT'),ev('8.3','監督式微調（SFT）')],
    '區分本次提示中的示例與會更新權重的示範訓練。', '初判已全文讀7.11與8.3正文；中文名稱、目標回答和更新角色已教。英文全名未在这些实际来源中核得，不冒稱已教英文。',
    '充分：本頁不是SFT演算法課，移除提示不保證留下規則的限定成立。', '充分：比較的是示例留在前文和調參數的不同用途，無須英文全名才能懂。',
    '僅英文名稱對照；沒有當前依賴缺口。', 'optional', '外部名稱查找便利', '保留原文', '保留短句可讓焦點停在是否更新權重。', '0；不在A.1再補英文全名。', '補英名會重複已教角色並延長對比句，收益低。')

add('CXE-02', ['EXT-A2-NAME'], ['A.2'], [ev('A.2','叫RAG（檢索增強生成）')],
    '先按問題找資料，再把資料放入前文回答；引用可回查，非永久寫權重。', 'A.2本段直接教中文名稱、檢索/生成分工；初判主文路線不依赖英文全名。',
    '充分：找/讀/引用/權重邊界均已明說。', '充分：文件多而模型需用相關資料的原需要已教。', '英文縮寫展開僅支援同一方法的查找。', 'optional', '新方法名称对外定位', '採用',
    '英文检索時能對上已讀的中文方法。', '首次括號寫「檢索增強生成，Retrieval-Augmented Generation，RAG」，其餘原文保留。', '增加短括號；不新增檢索算法或全文模型訓練。')

add('CXE-03', ['EXT-B1-NAME'], ['B.1'], [ev('B.1','JSON是資料格式')],
    '把有name/arguments的字典轉成可傳文字，外層才驗證與執行。', '本頁字典、json.dumps和格式不自執行的角色足夠；不要求JavaScript背景。',
    '充分：能讀請求欄位並辨別文字與真執行。', '充分：結構化請求便於外層檢查；全名不解釋這個理由。', 'JavaScript Object Notation未展開不是操作缺口。', 'optional', '格式名称对照', '保留原文',
    '保持「資料格式」這個實際必要角色。', '0；保留JSON名稱與現有格式說明。', '加JavaScript全名可能讓新手多想一個語言依賴；這頁Python不需要它。')

add('CXE-04', ['EXT-B6-BYTE'], ['B.6'], [ev('B.6','ByteTokenizer按UTF-8 byte編號'),ev('8.3','UTF-8位元組作文字單位'),ev('6.1','小小貓')],
    '只檢查對話資料中TOOL與EOS的五個有效目標，沒有更新。', '初判8.3已讀UTF-8位元組與回答結束；交叉再全文讀6.1的字、UTF-8 byte數和encode例。',
    '充分：TOOL四個字母加EOS五位直接可核；byte是先前已教的編碼單位。', '充分：數目標為了確認有效監督，不需要重教UTF-8演算法。', '本頁英文字byte可換中文以助跳讀，不能把中文字多byte關係當全書未教。', 'optional', '局部跳读词汇提示', '採用',
    '避免把byte目標數誤讀成中文字數，並讓需要者回到原例。', '將「UTF-8 byte」寫成「UTF-8位元組（byte）」並加6.1定位；不另插編碼段。', '一句內數字不變，增加小括號/連結；沒有新增算法。')

add('CXE-05', ['EXT-C5-KV'], ['C.5'], [ev('C.5','KV快取省重算'),ev('16.3','cache保存每層注意力裡的Key與Value')],
    '估計多候選作答的完整等待，不把token數當秒數。', '初判全文讀16.3與3.7；16.3已說K/V保存、因果可重用與新位置仍做注意力。',
    '充分：本句只需省重算但每步仍讀前文，沒有聲稱所有工作消失。', '充分：完整等待包括整理/生成/解碼/選擇；KV全名不是這個計量邊界的缺因。', '無必要缺口，當前簡述與已教實作一致。', 'optional', '重複名称/缓存内容提示', '保留原文',
    '把篇幅留給本頁要核的等待範圍。', '0；不在C.5重述Key/Value實作。', '另加機制容易把作答計算頁拖回注意力細節；需要者可沿16.3。')

add('CXE-06', ['EXT-C7-RLVR','TEX-O1','CE-optional-01'], ['C.7'], [ev('C.7','可驗真值提供回饋的訓練常稱RLVR')],
    '用可驗回饋更新動作機率，區分固定權重的作答計算。', '本頁已教policy/reward/baseline/advantage、一步真更新及正式策略與語言模型不同；初判來源未找到RLVR英名。',
    '充分：英名不是梯度計算前置。', '充分：更新權重與作答計算的不同目的由本段明說。', '三份origin是同一名稱對照，合併為一項，不以票數升為必要。', 'optional', '名稱查找', '採用',
    '將首次RLVR與可驗回饋的已有中文角色對上外部名稱。', '在首次RLVR補「Reinforcement Learning with Verifiable Rewards」；保留原角色與限制。', '短括號；不要求逐token RL、演算法變體或完整LLM訓練。')

add('CXE-07', ['EXT-C7-BASELINE'], ['C.7'], [ev('C.7','baseline是降低抽樣波動的參考'),ev('13.11','實際分數減更新前的基準')],
    '以固定reward-baseline的正負控制這次動作方向；本例明說沒有抽樣。', '交叉全文讀13.11，已教預期分、固定優勢、contextual bandit範圍；其正文也沒有推導降低抽樣波動。',
    '充分：.5參考、±.5優勢與±.25梯度不依賴抽樣方差原理。', '當前比較參考的用途已教；「降低抽樣波動」是另個泛稱性質，不能由這個人為固定action例推出。', '不是一步操作負擔；可把無展示的泛稱收窄到本頁真正展示的參考用途。', 'optional', '說明限定', '採用',
    '讀者可用本例解釋所寫用途，避免誤以為兩分數程式已展示方差降低。', '改為「baseline是比較本次回饋的參考，不是正確答案」；抽樣方差機制若將來另教再準確定位。', '改寫一句，不新增方差證明或抽樣實驗。')

add('CXE-08', ['EXT-T3-MLP'], ['training'], [ev('training','固定窗口 MLP 可以同時讀幾個字'),ev('training','視窗取幾個位置的差別見'),ev('2.5','Multilayer Perceptron')],
    '用同資料比較接字表與固定窗口模型並讀各自前後代價。', '交叉全文讀2.5，中文/英文全名、窗口三字與成本已教；T3補充也準確連2.5。',
    '充分：context/width和格式不可混用已教。', '充分：是否借多個前字改變猜法由同資料前後對照回答。', '名稱已在實際前文展開，当前角色足夠。', 'optional', '重複全名', '保留原文',
    '保留T3操作節奏而沿用已教模型名稱。', '0；不在T3重補MLP全名。', '長括號重複2.5，不能改善比較設計。')

add('CXE-09', ['EXT-T7-DPO'], ['training'], [ev('training','DPO直接用這種同題偏好'),ev('7.18','Direct Preference Optimization'),ev('13.4','Direct Preference Optimization')],
    '用同題偏好相對固定起點更新，不混兩套起點與資料。', '交叉全文讀7.18和13.4；兩者有直接偏好最佳化全名，13.4固定reference與optimizer角色也明確。',
    '充分：偏好資料、固定參考和訓練入口在T7均可辨。', '充分：讓同題較合適回答相对参考更常被選的目的已明說。', '無名稱依賴缺口；不把EXT-T7-LINK錯定位混成名稱缺失。', 'optional', '重複全名', '保留原文',
    '讓T7保留實際材料與階段關係。', '0；DPO全名沿用已教前文。', '長名稱不修錯連結；另以CXE-10改定位。')

add('CXE-10', ['EXT-T7-LINK'], ['training'], [ev('training','角色見[13.1]'),ev('13.1','偏好'),ev('13.4','保留原起點、不跟著更新的副本叫')],
    '讀者可在指定前文查到固定reference的角色。', '交叉全文讀13.1：同題偏好和記錄；13.4：固定初始副本、eval/停梯度/不交optimizer。',
    'T7自身已解釋固定副本，沒有機制阻塞。', '本頁偏好更新的用途已成立；讀者為查reference角色點13.1時沒有找到承諾的來源。', '錯定位真實，但因緊接正文自足只列可選定位改善。', 'optional', '閱讀定位', '採用',
    '一次跳轉就找到已教的固定參考用途。', '將「角色見13.1」的連結改到13.4；文字保留。', '僅改anchor；不增篇幅、不要求重讀整章。')

add('CXE-11', ['EXT-T10-KL'], ['training'], [ev('training','KL在此衡量教師、學生'),ev('18.8','Kullback')],
    '比較同起點、同位置、同字表學生的真值與教師機率訊號。', '交叉全文讀18.8，KL全名、teacher||student、CE-Hteacher、位置與mask已教；T10明確連18.8/18.9。',
    '充分：本頁KL角色與對齊要求已清楚；全名不改兩支學生的資料契約。', '充分：分開小架構效果與教師訊號效果的原需要已教。', '全名已有準確來源，不需再展開。', 'optional', '重複名称', '保留原文',
    '保持教師訊號對照的主線並沿用精確原理定位。', '0；保留KL及18.8連結。', '補英名重複明列前文，沒有操作收益。')

add('CXE-12', ['EXT-T10-PREQ','CE-optional-02'], ['training'], [ev('training','先取得T.4完整'),ev('training','不能拿本節'),ev('training','固定 `style/` 另保存'),ev('training','可選`--experiment modern`或`--experiment moe`')],
    '選T10固定蒸餾操作時，先有固定sft/style/moe教師及其資料切分，再執行兩支學生比較。',
    '36頁正文/折疊初判都讀過。T4折疊有固定sft入口與model.pt/dataset，T5有固定style入口，T8折疊可選固定moe；各段有本機pt不同的提醒。交叉再讀compression.py 55–66、110–120、720–729、847–852及run.py 56–89：三支都讀固定identifier/model.pt與dataset.json，教師須width64 TinyLM，入口不自動跑教師依賴。',
    '蒸餾對照機制與用途完整；正文CPU小練習產物不是固定蒸餾入口的依賴。',
    '需要三個已完成且配對資料的教師，不是為了術語或任意增加實驗；T10當前「再執行」操作承諾依赖它們。',
    '先備配方存在但明稱選讀；只選T10的讀者未被明確告知要打開哪三個固定配方並完成它們。需把在此路線上必要的補充標成必讀，無需把所有長配方搬正文。',
    'burden', '只對實選T10完整固定蒸餾路線；不影響只讀機制的主文路線', '採用',
    '讀者可辨自己只有attributes/style/moe小練習pt還是已完成三個width64教師，避免跑固定入口才碰到缺檔/不合規模型。',
    '在T10前置句加「此路線必須先打開並完成T.4『故事、切詞與固定對話實驗』、T.5固定配方與T.8『現代零件與效率的固定配方』中的sft/style/moe；本機小練習checkpoint不替代。核對outputs/course-experiments/course-v1/{sft,style,moe}/model.pt及各自dataset.json後再執行。」可用穩定anchor直接定位三折疊，不重貼長訓練。',
    '增加前置短段/三定位；三段長訓練本來就是此固定路線所需，不能宣稱兩條CPU命令包办教师。未要求读者读完整旧报告或跑无关分支。',
    '互見reader main/final與技術靜態來源後改判：自己CE-optional-02原把「補充已存在」視為足夠；交叉以只選T10的實際操作路線區分可獲來源和必要入口可見性。舊初判不回改。')

add('CXE-13', ['EXT-README-SAFE'], ['readme'], [ev('readme','公開safe權重只供推論'),ev('asset-storage','model.safetensors')],
    '讀者知道公开v2包可推論，但缺精确續訓狀態。',
    '初判asset-storage已全文讀公開16配對檔；交叉直接核固定v2-final-public-results.json /hf_revision及/architectures/{moe,dense}/safe_weights/path，兩者model.safetensors。這是manifest/result來源讀，不是舊review。',
    '推論/精確續訓的已有區別成立；不能推成safetensors格式本身永遠不能存optimizer等張量。', '需要知道公开的是推論產物；safe簡稱與安全能力形容有歧義。',
    '條件性名稱疑問已被公開bundle檔名解決，剩名稱精確化收益。', 'optional', '公開入口名称准确', '採用',
    '避免把safe理解成此模型被保證安全。', '將「公開safe權重」改為「公開safetensors推論權重」；保留「不能精確續訓」。不要加「此格式只能保存模型張量」。', '只換名稱，無新安裝/格式教學。')

add('CXE-14', ['EXT-ENV-ASR-OCR'], ['environment'], [ev('environment','沒有執行Whisper或其他ASR'),ev('11.12','Optical Character Recognition'),ev('12.11','Automatic Speech Recognition')],
    '選環境並辨有限語音/讀字任務，無開放逐字聽寫保證。', '交叉全文讀11.12與12.11，OCR和ASR中英文全名、輸出任務與限制已教。environment本段亦自行描述銀行主題和12字有限區域。',
    '充分：環境與模態任務角色足夠，不缺英文名称机制。', '充分：語音回答不等一般轉寫、有限OCR不等任意圖像文字；可助直接跳讀環境的新手辨縮寫。',
    '只需中文角色提示，不必在環境頁放兩段長英名。', 'optional', '環境入口跳读提示', '採用',
    '直接選硬體/環境時也能對上ASR與OCR所指工作。', '首次寫「ASR（語音辨識）」及「OCR（讀字）」；完整名留11.12/12.11。', '兩個短中文括號，沒有新增模態架構或能力承諾。')

add('CXE-15', ['EXT-ASSET-LFS'], ['asset-storage'], [ev('asset-storage','LFS保存的是檔案'),ev('asset-storage','普通Git保存資料包的一張小卡片')],
    '區分程式/資料/權重和LFS pointer/實體下載，不把LFS當壓縮。', '本頁自行給Git LFS工具名、pointer、指紋大小、按需取得與計費角色，且已有官方Git LFS計費與資料入口。',
    '充分：實體未收到與格式不變的機制不依賴英名展開。', '充分：資料包另存/版本可核的用途已教。', '名稱展開收益小；實質的安裝入口問題另以CXE-21修C7，不混成LFS名稱缺口。', 'optional', '英文名稱對照', '保留原文',
    '保留已具體的pointer/下載解釋。', '0；維持Git LFS名稱與官方入口。', '長全名不修C7錯鏈，也不增加實際下載判讀能力。')

add('CXE-16', ['EXT-CURR-BPB'], ['curriculum'], [ev('curriculum','換 tokenizer 時加入 BPB'),ev('6.5','bits per byte'),ev('6.5','UTF-8')],
    '表格給課程採用某評估指標的設計理由：避免不同切詞的平均token loss直接比較。',
    '交叉全文讀6.5：同原文兩切分總NLL、naturallog轉bits、除同原始UTF-8 byte分母、共同計分邊界，且不能只從平均loss推出BPB。已教充分；不是全書缺這個方法。',
    '6.5機制充分；curriculum當前表只給指標名，沒有單位/共同分母或6.5定位。',
    '當前表自己承諾BPB能避免比較單位差異；單獨讀這份設計參考時，讀者只有「換指標」而無它如何提供共同單位的最短理由。',
    '最小缺口是同原文共同byte分母＋準確6.5定位，不需要搬公式/重教tokenizer。循序已讀6.5者可正常推論，不對那條路線列必要。',
    'burden', '單獨使用curriculum作設計參考的路線', '採用',
    '表格中的採用理由有共同單位的實際因果來源，避免新名稱代替解釋。',
    '表內寫「換tokenizer時加入BPB（bits per byte，每個原始UTF-8位元組的bits代價）：同一原文使用同一byte分母；比較與計分邊界見6.5。」保留避免直接比平均token loss的原句。',
    '一短說明＋6.5連結；完整換底/總NLL例留前文，不新增實驗或指標體系。',
    '互見reader主文必要項後，實讀6.5來源確認；不是新獨立發現。自己初判未報此項，交叉按當前表所承諾採用理由和實選跳讀用途裁。')

add('CXE-17', ['EXT-TEX-PPO'], ['training'], [ev('training','想逐步理解獎勵、價值估計與 PPO'),ev('13.11','Proximal Policy Optimization'),ev('7.18','Proximal Policy Optimization')],
    '折疊中的独立CPU预写卡路線讓讀者逐步理解評分/價值/更新，明說不逐token生成。', '交叉全文讀13.11和7.18，PPO全名與策略更新角色已教；折疊準確指向13.10–13.17。',
    '充分：CPU卡例与真正语言生成、DPO兩路分界已教。', '充分：想逐步理解回饋更新才選此支；本節不需全名才能選/執行。', '無必要缺口，英名已有明列前文。', 'optional', '折疊重複名称', '保留原文',
    '保持可選支線的短入口與範圍。', '0；不在training折疊再補PPO全名。', '長英名不增當前步驟來源，已連章內完整教學。')

add('CXE-18', ['EXT-TEX-SDPA'], ['training'], [ev('training','PyTorch的SDPA是呼叫'),ev('16.8','scaled dot-product attention')],
    '核驗接口實際選到的後端和可比數值，不能由SDPA名推出Flash。', '交叉全文讀16.8正文，完整英/中名稱、Q/K/V、遮罩/dropout語意和後端非保證；fold精確連16.8。',
    '充分：接口/底層算法/實際probe證據分開。', '充分：要確認真的使用Flash及範圍，不是只查API名。', '已教名稱且有直接來源，無補名必要。', 'optional', '折疊重複名稱', '保留原文',
    '讓選讀集中核後端而不重複注意力名。', '0；保留SDPA及16.8連結。', '長名不能替代backend核驗；已有正確定位。')

add('CXE-19', ['EXT-TEX-QAT'], ['training'], [ev('training','介紹QAT：讓訓練先適應模擬誤差'),ev('17.14','Quantization-Aware Training')],
    '選固定量化支線時區分訓練適應誤差與實際低位元格式/核心。', '交叉全文讀17.14正文，量化感知訓練全名、fake quant/STE、浮點主權重及部署另驗皆已教；此句直接連17.14。',
    '充分：QAT角色与真正部署成本不同已经明说。', '充分：是否有助品質需同起点预算验证，未以名称保證改善。', '已教名称且直接引用，不需再展開。', 'optional', '折疊重複名稱', '保留原文',
    '保持固定比較短入口，詳細學習留17.14。', '0；不在此折疊再補QAT全名。', '增加長名沒有改善部署或訓練邊界。')

add('CXE-20', ['TEX-N1'], ['B.3','B.4','training'], [ev('B.3','這份按順序留下的紀錄叫trace'),ev('B.4','完整紀錄見'),ev('training','RAG、工具與推理報告保存各自的原始輸入')],
    'trace原始輸入能回查每步當時看見什麼，工具結果在執行後才回填。',
    '初判B3/B4及圖/上下文已讀。交叉在看到technical初判與root source check後，自核原tools.json與applications.py 33–49、94–95、528–535、570–573；ByteTokenizer data.py 11–21。無模型呼叫、重取樣或重訓。',
    '正文流程/動作/工具/完成邊界仍充分；56步的原input_ids都對得上當時訊息前綴。共36次generation.messages多存了後续assistant請求與TOOL_RESULT，是共享列表被原地擴充。',
    '實際輸入不泄漏答案，原19/20不因這個metadata錯位而失效；但「原始輸入」可回查承諾被messages字段破壞。',
    'CALC:add(9,9)首步原97 IDs逐ID只對system/user兩筆；四筆stored messages編成155。全部20normal+18capped共56generation有36首步metadata錯位，全部都有唯一真實前綴。最小是正確快照與舊證據勘誤，非改概念或分數。',
    'burden', '歷史原報告字段回查/重放路線', '採用',
    '讀者不會誤把本步尚未取得的結果當模型輸入，也能用原IDs重建/重放真正時序。',
    '未來_sample返回messages時deepcopy當時列表。保留舊tools.json與SHA，發布逐generation的messages_at_generation重建補件（所有input_ids逐ID核對），B3/B4原報告入口加一句字段勘誤與補件定位，training實驗記錄可沿同補件。重建不是新模型執行證據。',
    '一行記錄程式修正、append-only歷史補件及短入口；無需改舊JSON/得分、重訓或把全部trace搬正文。',
    '封存後互見technical TEX-N1與root-tool-trace-source-check後做原ID/code核對；獨立初判發現歸technical，不宣稱第二份獨立發現。')

add('CXE-21', ['CE-burden-01'], ['C.7'], [ev('C.7','安裝Git LFS'),ev('training-assets','需安裝 [Git LFS]'),ev('training','每包的固定來源、授權和內容指紋')],
    '選讀重做的讀者先裝Git LFS，再初始化、取得gsm8k包、跑reasoning。', '初判W.1指定章與T.1正文已讀；T.1只選資料/fetch並連資料說明，training-assets「取得資料」才給官方Git LFS安裝入口。',
    '三行init/pull/run流程成立，但git lfs install --local不取得Git LFS程式。', '必須先有這個程式，才有能力跑本段命令；不是從重做目標自編新前置。',
    '按T.1「安裝Git LFS」沒有承諾的安裝說明。已有Git LFS者无阻礙，未安裝且選重做者需再自行找入口。', 'burden', 'C7完整重做且尚未安裝Git LFS的選讀路線', '採用',
    '一次連結便到真安裝入口，三行命令才能開始。', '將T.1安裝鏈改成assets/training/README.md「取得資料」或https://git-lfs.com/；保留三行初始化/取得/執行。', '僅換連結，不在C7重述各OS安裝或重教LFS。')

all_origin_ids = set(origin)
covered = [x for e in entries for x in e['origin_ids']]
assert set(covered) == all_origin_ids
assert len(covered) == len(set(covered)) == 24
assert len(entries) == 21
assert {x['id'] for x in reader_main['issues']} <= set(covered)

pass_annotations = {
 'A.8': ('仍通過', 'T在1/4/8而問題不動、U1–U7相對順序不變。這是固定材料的受控位置介入，非只列位置术语；短版用來診斷是否小題本來就會答，沒有普遍U形或內部機制承諾。', '無需將所有長文任務歸成同一能力，亦不需為紙上材料提供新的模型成績。'),
 'A.9': ('仍通過', '同八筆資料把查M17、列全部教室與T→M17→U3→二樓拆成不同判準，組合需要從明示同碼關係來，非標題推先備。', '找到單筆代碼不等多筆收集或關係組合；不要求補尚未承諾的摘要/版本衝突完整實驗。'),
 'B.4': ('概念仍通過；原報告字段另修CXE-20', 'add+done是兩動作、一工具；EOS僅生成停止，done要有完成answer；max_steps=1的step_limit不等工具沒算。technical原數據核對與自己紙上判準吻合。', 'messages metadata缺陷不能倒推成原始input_ids已含未來結果，也不推翻動作/完成定義或19/20。'),
 'B.8': ('仍通過', '假設直答.9的錯失1、全鏈工具.99加.02得.12；直答.999時.01可改選直答。完整往返品質與额外成本的原需要明示，非新方法產物自編目的。', '工具真算81而模型答8證明函数率不夠；不得要求這個人工尺度例已訓成逐題信心或量到秒/美元。'),
 'C.4': ('仍通過', '同[3,3,4]多數選3而精算驗4；集合命中、選中和交付有各自分母，全部不過回None。已讀圖也由同集合分流。', '精算2+2提供真值是明確強條件，不需要此頁擴成未知事實通用驗證器。'),
 'C.7': ('一步更新仍通過；名稱/限定和選讀入口分項局修', '固定action1答4，adv=.5→grad[.25,-.25]→updated[-.025,.025]→p=.512497；reward0反向。真算updated而非只backward，固定數字不求驗證器梯度。', '原例没有抽样/训练LM，也不把真正更新或高弱reward当新题能力；baseline泛稱收窄不改此完整机制。'),
 'training': ('T9仍通過；T10路線單獨修CXE-12', '同浮點基模各自int4/int8，碼/scale/保留張量總和與檔案包裝/optimizer分開；同row評內容和停止，validation選bits、test最後，反量化浮點不冒稱runtime加速。', '無需要求所有完整GPU配方搬到正文；未實跑的品質/記憶體/速度保持未驗，不拿技術peer静态核对当本轮执行。')
}
passes=[]
for original in own['strong_passes']:
    status, reason, counter = pass_annotations[original['page_id']]
    e = ev(original['page_id'], original['quote'])
    passes.append({'page_id':original['page_id'], 'unit': original.get('unit'), 'own_initial_claim':original['claim'], 'exact_source_evidence':e, 'mechanism':original['mechanism'], 'need':original['need'], 'cross_disposition':status, 'source_grounded_reason':reason, 'strong_counterexample_or_boundary':counter, 'verification_scope':'own初判全文/紙上及已看圖；封存後讀同組reader/technical該頁判據，沒有新增模型執行或新圖驗收。'})

report_files=['reader-extensions-main.json','reader-extensions.json','technical-extensions-initial.json','continuity-extensions-initial.json','continuity-extensions-seal.json']
prereqs=['2.5','6.1','6.5','7.18','13.1','13.4','11.12','12.11','18.8','13.11','17.14','16.8']
counts=Counter((e['necessity'],e['disposition']) for e in entries)
now=datetime.now(timezone.utc).isoformat()
report={
 'schema_version':'continuity-cross-review-v1', 'kind':'封存後同組交叉覆核，非獨立來源初判', 'reviewer':'/root/continuity_extensions', 'group':'extensions', 'created_at':now,
 'instructions':{'path':str(B/'cross-review-instructions.md'),'sha256':sha(B/'cross-review-instructions.md')},
 'initial_seal_verification':{'initial_sha256':ownseal['sha256'],'main_notes_sha256':mainseal['sha256'],'verified_before_peer_reads':True,'verified_at_write':True,'original_reports_modified':False},
 'source_reports_read':[{'path':str(R/n),'sha256':sha(R/n),'scope':'reader-main/final全文判據與補充影響；technical完整issues/coverage及來源核驗和相關逐頁checks；own已封存36頁初判/strong passes'} for n in report_files],
 'root_source_check_read':{'path':str(B/'synthesis/root-tool-trace-source-check.json'),'sha256':sha(B/'synthesis/root-tool-trace-source-check.json'),'scope':'封存後source/code核對，全部755bytes讀完；不是新獨立發現'},
 'origin_coverage':{'reader_main_ids': [x['id'] for x in reader_main['issues']], 'reader_final_ids':[x['id'] for x in reader['issues']], 'technical_ids':[x['id'] for x in technical['issues']], 'continuity_ids':[x['id'] for x in own['issues']], 'unique_origin_count':24,'deduplicated_relationship_count':21,'all_origins_exactly_once':True,'reader_final_optional_origins':17,'continuity_optional_origins':2,'technical_optional_origin':1,'deduplication':'RLVR三origin合一；T10 reader必要與own可選合一。reader main/final相同ID不重計；不同缺口不按頁或票數合併。'},
 'actual_source_read_scope':{
    'initial':'自己的main notes/initial已封存36頁全正文、31折疊與所記真正先備；不在交叉再讀全36頁。',
    'cross_additional_prerequisites':[{'page_id':p,'source_sha256':sha(S/(p+'.md')),'scope':'helper main完整正文；未補讀這些先備的折疊、未看新增先備圖'} for p in prereqs],
    'assigned_targeted_rechecks':'T10/T4/T5/T8主文與所選完整配方、C7重做、T1、training-assets安裝、curriculum BPB、asset-storage公開bundle、名稱候選當前段落和已教段落；精確quotes附逐項line/hash。',
    'implementation_text_only':[{'path':str(B/'freeze/implementation/scripts/course_experiments/applications.py'),'sha256':sha(B/'freeze/implementation/scripts/course_experiments/applications.py'),'read_ranges':['33–49','80–125','528–535','560–615']},{'path':str(B/'freeze/implementation/scripts/course_experiments/compression.py'),'sha256':sha(B/'freeze/implementation/scripts/course_experiments/compression.py'),'read_ranges':['55–120','720–733','847–897']},{'path':str(B/'freeze/implementation/scripts/course_experiments/run.py'),'sha256':sha(B/'freeze/implementation/scripts/course_experiments/run.py'),'read_ranges':['56–116']},{'path':str(B/'freeze/implementation/tiny_perceptron/data.py'),'sha256':sha(B/'freeze/implementation/tiny_perceptron/data.py'),'read_ranges':['11–51']},{'path':str(B/'freeze/implementation/tiny_perceptron/tokenization.py'),'sha256':sha(B/'freeze/implementation/tiny_perceptron/tokenization.py'),'read_ranges':['1–103']}],
    'raw_postseal_reads':[{'path':str(B/'technical-extensions-evidence/tools.json'),'sha256':sha(B/'technical-extensions-evidence/tools.json'),'scope':'全部normal/capped 56generation输入ID与message前缀，结构化程式核；非新的模型執行'},{'path':str(B/'technical-extensions-evidence/docs__selftrained__results__v2-final-public-results.json'),'sha256':sha(B/'technical-extensions-evidence/docs__selftrained__results__v2-final-public-results.json'),'scope':'只核顶层键/hf_revision及moe/dense safe_weights/path= model.safetensors；未全面重验最终成绩'},{'path':str(B/'technical-extensions-evidence/docs__selftrained__v2-manifest.json'),'sha256':sha(B/'technical-extensions-evidence/docs__selftrained__v2-manifest.json'),'scope':'只读顶层键，不算完整manifest验收'},{'path':str(B/'technical-extensions-evidence/docs__course-experiments__public-models.json'),'sha256':sha(B/'technical-extensions-evidence/docs__course-experiments__public-models.json'),'scope':'只结构检索safetensors相关字段；不是全文模型清单再审'}],
    'own_cross_trace_evidence':{'path':str(B/'work/continuity-extensions/cross-tool-input-source-check.json'),'sha256':sha(B/'work/continuity-extensions/cross-tool-input-source-check.json'),'steps':56,'messages_metadata_mismatches':36,'all_input_ids_unique_actual_prefix':True}
 },
 'issues_dispositions':entries,
 'strong_passes_reviewed':passes,
 'summary':{'necessary_adopted':counts[('burden','採用')],'optional_adopted':counts[('optional','採用')],'optional_retained':counts[('optional','保留原文')],'deferred':0,'pending_issue_checks':0,'necessary_relationship_ids':[e['id'] for e in entries if e['necessity']=='burden'],'necessary_origin_ids':[x for e in entries if e['necessity']=='burden' for x in e['origin_ids']],'required_repair_scales':{'paragraph':True,'page':False,'chapter':False,'order':False},'small_code_and_evidence_repair':'CXE-20另有記錄code一行、append-only重建补件；不改原JSON/原得分/重訓。','scope_limit':'四項必要均是特定完整操作/資料回查/獨立設計參考路線。沒有證據需要重寫ABC主文概念鏈、整頁、章或順序。'},
 'unknown_and_unverified':[
    '本輪沒有執行T10/C7固定模型配方、安裝Git LFS或下載權重；前置路線只做來源/code核，不能宣稱本機命令完整通過。',
    '未新增驗新題品質、runtime延遲/記憶體、HF遠端16檔實際下載或任何模型推論/重訓。',
    'initial只核9分配圖各640/360與B3/C7实际mobile/desktop上下文，另34頁原位頁面/互動/runtime outputs/reading annotations未驗；交叉沒有新視覺爭議或新增圖驗收。',
    '新增先備2.5/6.1/6.5/7.18/13.1/13.4/11.12/12.11/18.8/13.11/17.14/16.8只讀main，先備折疊/圖未驗，不能藉本輪英名核對聲稱它們所有數值/圖片已驗。',
    'baseline降方差一般性質沒有在實讀13.11及C7最小例展示；選擇收窄當前句，不判此一般方法錯誤或新增完整證明。',
    '保留名稱不代表全書所有首見都已核；以實讀前文與當前角色是否足夠裁。'
 ],
 'contamination_and_provenance':{'mutual_reports_visible_only_after_all_initial_seals':True,'no_historical_review_content_read':True,'inherent_source_historical_review_narrative':'初判已記curriculum/validation/readme/publishing原文泛稱審閱歷史；不可避免原文曝光未當作新證據。交叉只读授权同组封存报告/root源核。','filename_only_exposure':'初判與本輪一次rg --files找applications.py時顯示歷史review/artifact路徑名；未開檔或讀其結論。','postseal_new_checks':'工具ID/code重核由TEX-N1/root提示，來源發現歸technical；BPB/T10改判由互見後選定路線來源核，非再一份獨立發現。','tutorial_or_original_results_mutated':False,'agents_spawned':False}
}
out=R/'cross-extensions.json'
out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
seal={'kind':'cross review byte SHA seal','reviewer':report['reviewer'],'group':'extensions','at':now,'report_path':str(out),'sha256':sha(out),'initial_sha256_unchanged':ownseal['sha256'],'main_notes_sha256_unchanged':mainseal['sha256'],'mutual_visibility_after_initial_seal':True,'independent_initial_discovery':False}
(R/'cross-extensions.seal.json').write_text(json.dumps(seal,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'report':str(out),'sha256':seal['sha256'],'summary':report['summary'],'origin_count':len(covered)},ensure_ascii=False,indent=2))
