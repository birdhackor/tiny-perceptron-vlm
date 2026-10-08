import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

B = Path('/workspace/work/tutorial-audit-20261008')
S = B / 'freeze/sources'
now = datetime.now(timezone.utc).isoformat()

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def ev(page, quote, role='main'):
    p = S / (page + '.md')
    source = p.read_text()
    assert quote in source, (page, quote)
    return {'page_id': page, 'source_path': str(p), 'source_sha256': sha(p),
            'quote': quote, 'line': source[:source.index(quote)].count('\n') + 1,
            'role': role}

files = ['reports/reader-modalities-main.json', 'reports/reader-modalities.json',
         'reports/technical-modalities-initial.json', 'reports/continuity-modalities-initial.json',
         'reports/continuity-modalities-main-notes.json']
report_sources = [{'path': str(B / f), 'sha256': sha(B / f)} for f in files]
own = json.loads((B / files[3]).read_text())
own_seal = json.loads((B / 'reports/continuity-modalities-initial-seal.json').read_text())
main_seal = json.loads((B / 'work/continuity-modalities/main-notes-seal.json').read_text())
assert sha(B / files[3]) == own_seal['report_sha256']
assert sha(B / files[4]) == own_seal['main_notes_sha256'] == main_seal['sha256']
assert sha(B / 'work/continuity-modalities/source-delivery.jsonl') == own_seal['source_delivery_sha256']
technical_seal = json.loads((B / 'reports/technical-modalities-initial-seal.json').read_text())
assert sha(B / files[2]) == technical_seal['sha256']
reader_main_sha = (B / 'reports/reader-modalities-main.sha256').read_text().split()[0]
assert sha(B / files[0]) == reader_main_sha

issues = []
issues.append({
 'id': 'C-M-resample-action', 'origin_ids': ['M-12.2-resample-control'], 'pages': ['12.2'],
 'original_grades': {'reader_main': 'burden', 'reader_final': 'burden', 'continuity_initial_page': 'pass'},
 'cross_severity': 'burden', 'necessary': True, 'disposition': '採用',
 'strongest_source': [
   ev('12.2', '正確改取樣率需要重取樣：先控制超過新界線的成分，再在新的時間點計算數值，讓時長與原聲音的音高保持相應關係。'),
   ev('12.2', '理想條件下，高於界線的變化可能混成較低頻率，不能由這串樣本唯一重建。'),
   ev('12.2', '例如將 4591 點的 8000 Hz 聲音重取樣成 9182 點的 16000 Hz 聲音，兩者時長仍為 0.573875 秒。')],
 'actual_prerequisites': [ev('12.1', '440 Hz 表示每秒 440 個週期，不能把第 440 個樣本當作 440 Hz 的意思。', 'prior main')],
 'original_promise': '分清 rate、時長及音高，並介紹正確改 rate 的必要重取樣動作；不承諾完整濾波器算法。',
 'mechanism_check': {'judgment': '局部需補', 'taught': '樣本數/rate、rate/440、錯標 rate 會變音高、新時點重新算值均完整。',
   'missing': '「控制」沒有說對超界頻率做什麼。降低、增強、移動都不是由這個詞唯一決定；缺的是減弱/濾除基本動作，非頻率界線的完整推導。'},
 'need_check': {'judgment': '已足夠', 'source': '同頁超界變化可能混低頻、直接改標籤會使聲音慢/低；12.1 已分頻率及振幅。',
   'short_inference': '降 rate 使可分清界線變低，超出新界線成分可能被誤呈為低頻；先削弱它們再求新時間點，才有聲音保真的當前用途。這個目的由原文給出，沒有另加新任務。'},
 'strongest_counterevidence': '本頁一開始已教半 rate 界線與混低頻，所以不能報「不知道為何重取樣」；主要數量換算也不受此缺口影響。',
 'minimal_gap': '為一般重取樣補出最低限度的超界頻率处理動作，並限定它主要是降取樣時避免混疊；別讓隨後 8000→16000 例子看似要求新刪原可表示頻率。',
 'reader_impact': '讀者會算新點數，卻不能從原文說出第一步對聲音做了什麼；需自補未教的濾波語義。',
 'understanding_benefit': '把奈奎斯特限制、減弱超界成分、新時點計算連成可解釋的流程。',
 'minimal_patch': '在原重取樣句把「控制」改成「降取樣前先減弱/濾除新界線以上的成分（低通處理），避免混成假低頻，再在新時點計算數值」；另用短子句限定升取樣不新刪舊 rate 可表示的頻率。',
 'new_burden': '一個低通名稱及一句具體動作/限定；不引入濾波器設計、卷積或執行實驗。',
 'repair_scale': 'paragraph', 'page_chapter_order_rewrite': False,
 'verification': '正文、折疊、12.1 已讀；未執行重取樣，4591 錄音原始證據本輪未查。',
 'change_from_own_initial': '原初判把「先控高頻」當作已教動作，本輪讀 reader origin 後重看原文，改判只此動作關係不足；時間/音高/用途通過保留。',
 'discovery_phase': '封存後受 reader origin 提示的交叉修正，不計第二份獨立發現'
})
issues.append({
 'id': 'C-M-mel-adoption', 'origin_ids': ['M-12.6-mel-purpose'], 'pages': ['12.6'],
 'original_grades': {'reader_main': 'burden', 'reader_final': 'burden', 'continuity_initial_page': 'pass'},
 'cross_severity': 'burden', 'necessary': True, 'disposition': '採用',
 'strongest_source': [
   ev('12.6', '頻譜每框有 201 個頻率格，想用 16 個較寬區段表示能量。'),
   ev('12.6', '這種依 mel 尺度配置的頻帶，低頻安排得較密、高頻較疏。'),
   ev('12.6', '頻帶變少會合併差異，因此不能由 16 帶唯一還原所有 201 格。mel 配方是固定的聲音前處理；後面自行訓練的入口才學如何利用它答題。')],
 'actual_prerequisites': [
   ev('12.5', '希望每個短時間框都能指出「哪些頻率較強」，並核對最強的格子是否對應 440 Hz。', 'prior main'),
   ev('12.5', '400 點一框、每次移 160 點，輸出功率 `(201,21)`：201 個非負頻率格、21 個時間框。', 'prior main')],
 'later_scope_evidence': [
   ev('12.7', '把頻率功率合成 mel 頻帶後再取 log，就得到 log-mel 特徵。', 'later route; not retroactive first-use explanation'),
   ev('12.8', '一段 0.1 秒單音經 log-mel 處理，得到 16 個頻帶、11 個時間框。', 'later route; not retroactive first-use explanation'),
   ev('12.15', '先沿用前面學過的 log-mel 特徵：把錄音切成短時間段，記錄各頻帶的強弱，再做固定轉換，得到按時間排列的一列數字。', 'later adopted route')],
 'original_promise': '相鄰頻率合成少量頻帶的操作教學，以及本課接著實際採用的固定 mel 前處理。原文沒有承諾聽覺優越性、降噪、最佳16帶或比所有其他表示更好。',
 'mechanism_check': {'judgment': '已足夠', 'taught': '非負三角權重、相鄰重疊、16×201 乘 201×3、時間不被平均、手算5與2、Hz到mel非等間隔及不可逆皆清楚。'},
 'need_check': {'judgment': '局部採用理由需補',
   'taught': '作者明說要少量區段並承認合差，足以看懂當前操作目標；不是完全沒有目的。',
   'missing': '「想用16」只宣布輸出規模，低密/高疏句及公式只說新配方性質。已讀聲音需要是表示各框哪些頻率較強，尚未說為什麼本课選擇合併差異、尤其非均勻分配，來服務這個表示/入口用途。',
   'short_inference_limit': '可以從矩陣看出頻带少，但若把「因此成本低/符合人耳/較易辨語音」當原理由，就是自補未教需要或聲學事實；不能以此放行。'},
 'strongest_counterevidence': '本頁標題確實是「怎麼合成」且16/32可自由示範，原初判的操作/性質通過仍有效；後文沒有效果優越主張，所以不要求性能試驗或完整人耳理論。但正文明說此配方是後續答題路線的固定前處理，不能只靠操作標題把所有採用理由判為不適用。',
 'minimal_gap': '補一個實際選用這種固定表示的用途/選擇約定，將較少頻帶及低頻較細的分配與該用途連上；保留合差代價與16只是示例選擇。',
 'reader_impact': '讀者能算加權和與mel換算，但要自己猜作者為何在送入口前採用這種合差與頻率分配；這是局部理由負擔，非整條音訊鏈阻斷。',
 'understanding_benefit': '讓「現在如何算」與「本課为何選它作入口材料」分别有來源，避免把固定工具當作已有識別能力。',
 'minimal_patch': '用一到兩句交代本課的真實表示/教學選擇：為何以少量頻带呈現每框能量，為何採用低頻較細的mel配置，並明示16為小例的配置、會犧牲細頻率差異。若目的只是示範常用固定工具，可明說這是演示採用約定及其分配意義，別補成「全面更好」；若訴諸感知理由，須用最短新背景說明而非只寫「模仿人耳」。',
 'new_burden': '一到兩句用途/配置說明；不要求新實驗、完整聽覺學或最優頻带證明。需避免引入作者未原先承諾的效能目標。',
 'repair_scale': 'paragraph', 'page_chapter_order_rewrite': False,
 'verification': '本頁正文與回顧折疊、12.5、12.7、12.8、12.15 已讀；圖的重疊算例在原獨立輪已看640，本輪沒有視覺爭議，不新增圖判斷。',
 'change_from_own_initial': '原初判以操作/性質及無感知優勢承諾收窄 need。本輪受 reader origin 提示，保留原運算通過，但改判實際採用的頻帶分配用途需要局部銜接；不以同行票數改判。',
 'discovery_phase': '封存後受 reader origin 提示的交叉修正，不計第二份獨立發現'
})
issues.append({
 'id': 'C-M-contrastive-scope', 'origin_ids': ['M-10.11-empirical-reference'], 'pages': ['10.11'],
 'original_grades': {'reader_main': 'burden', 'reader_final': 'optional'},
 'cross_severity': 'optional', 'necessary': False, 'disposition': '採用',
 'strongest_source': [
   ev('10.11', '兩項不必相等：圖找文比較 2.0 對 0.1、0.4 對 1.8；文找圖比較 2.0 對 0.4、0.1 對 1.8，差距不同。'),
   ev('10.11', '雙向平均表達兩種查詢都重要，但不保證每次效果勝過單向。原六類小實驗中，兩種版本的最後題、兩個方向都為 6/6，沒有觀察到答對數優勢；驗證也非全部正確。'),
   ev('10.11', '原實驗的資料切分、更新設定與逐題結果見[既有實報](../../docs/course-experiments/results/contrastive.json)', 'folded lookup')],
 'actual_prerequisites': [ev('10.10', '兩張圖的正確描述依序放在文字欄 0、1。希望每張圖的正確欄分數比另一欄高。', 'prior main')],
 'original_promise': '解釋兩向候選競爭及雙向平均用途，另以有限實測提醒效果邊界；非教完整復現。',
 'mechanism_check': {'judgment': '已足夠', 'basis': '非對稱表、轉置、各向差距、兩交叉熵和梯度已呈現不同工作。'},
 'need_check': {'judgment': '已足夠', 'basis': '正文直接給兩種查詢都重要；不用6/6也成立。未觀察答對數優勢只是此有限實測的結論。'},
 'counterevidence_and_necessity': '折疊已直接定位contrastive；候選有限且检索/生成界線已說明。缺全部題目/設定不使雙向需要失效，不升為必要。10.5折疊的六類是相關分類實驗，不把它冒充本頁contrastive設定來源。',
 'understanding_benefit': '給「六類」一個簡短身份，並說最後題是新位置，可降低實測段回找資料的成本。',
 'minimal_patch': '將「原六類小實驗」微擴為「原紅綠藍×圓形/方形六類候選、留出圖片位置的小實驗」，附近給既有实報入口；不搬入250步歷史/逐題表。',
 'new_burden': '一個短子句、沿用已教顏色/形狀/留出概念；不增加理論先備。',
 'repair_scale': 'paragraph', 'page_chapter_order_rewrite': False,
 'actual_raw_check': '本輪讀contrastive快照data/config/scope与两variant最终汇总及候选记录；train offset -2,-1,0，validation1，test2；两版test两向1.0，validation圖找文5/6。未載入權重、未訓練，未把所有history看完。',
 'discovery_phase': '原reader獨立候選之提示後處置；可選微改收益已由實際來源核實'
})
issues.append({
 'id': 'C-M-projector-scope', 'origin_ids': ['M-11.3-empirical-reference'], 'pages': ['11.3'],
 'original_grades': {'reader_main': 'burden', 'reader_final': 'optional'},
 'cross_severity': 'optional', 'necessary': False, 'disposition': '保留原文',
 'strongest_source': [
   ev('11.3', '沒有優化器 step，所以這個短例是梯度通路，不是生成成功率。'),
   ev('11.3', '既有接頭小實驗真正更新後，完整描述仍為 0/6，雖然標準答案上下文裡的預測代價降低。紅方塊只生成 `red`，缺了空白與 square；綠圓還會重複 e。'),
   ev('11.3', '這次弱底座只有部分文字題做對，不能拿失敗推出所有接頭方案都無效'),
   ev('11.3', '原實驗的資料切分、更新設定與逐題結果見[既有實報](../../docs/course-experiments/results/projector.json)', 'folded lookup')],
 'actual_prerequisites': [
   ev('11.1', '原文字題先失敗，就不能把後面所有失敗歸給圖片接頭；文字題成功而圖片題失敗，才進一步查圖片特徵與接頭。', 'prior main'),
   ev('11.2', '凍結文字權重不等於禁止它參與計算。只要計算仍被記錄，答案代價可以穿過固定文字運算，傳回接頭', 'prior main')],
 'original_promise': '固定核心仍傳梯度到接頭，並區分標準前文代價下降、完整自由描述及弱底座失敗的證據。',
 'mechanism_check': {'judgment': '已足夠', 'basis': '梯度、無step、teacher forcing/自接上一字、red缺square及重複e均說明當前關係。'},
 'need_check': {'judgment': '已足夠', 'basis': '圖片及文字两端先有相關能力時，只調接頭連視覺線索到文字；先驗證通路，再分能力驗收。'},
 'counterevidence_and_necessity': '正文已用「既有」「真正更新後」清楚轉到另一實驗，原候選「沒说明換设置」不成立；有限六題的完整清單、寬64/兩層及底座5/10屬更多實測解讀，並非本頁關係所需。折疊有明確來源。',
 'understanding_benefit': '完整实验卡可能幫重做者定位六種描述及底座范围，但本页通路/失败差別不会因此获得新必要解释。',
 'minimal_patch_if_later_wanted': '最多在結果句點明「六種合成顏色/形狀完整描述」，沿用既有projector連結；不重講底座訓練。',
 'minimal_patch': '本輪不建議修改；保留現有結果句、兩個失敗樣例、弱底座限定及折疊來源。',
 'new_burden': '加完整底座實驗卡會引出模型寬度/層數、語言材料与另一baseline表，比本頁可得益更大；不值得為此擴段。',
 'repair_scale': 'paragraph candidate only; no adopted change', 'page_chapter_order_rewrite': False,
 'actual_raw_check': '本輪讀projector快照data、training匯總、before/validation/test總結与language_weights_unchanged；六種RGB×shape的留出offset材料、14408目标、final loss1.2093、test0/6可核；未看完整history或重新生成。',
 'discovery_phase': '原reader獨立候選之提示後處置；可選不自動採用'
})
issues.append({
 'id': 'C-M-vqa-shared-scope', 'origin_ids': ['M-11.5-experiment-scope', 'M-11.6-experiment-scope'], 'pages': ['11.5','11.6'],
 'original_grades': {'M-11.5-experiment-scope': {'reader_main': 'burden', 'reader_final': 'optional'},
                     'M-11.6-experiment-scope': {'reader_main': 'burden', 'reader_final': 'optional'}},
 'cross_severity': 'optional', 'necessary': False, 'disposition': '採用',
 'deduplication_relation': '兩項同為同一vqa實報的任務/模型/素材身份定位；共用短摘要，只在11.6追加阶段开放范围和沿用关系，保留两个origin。',
 'strongest_source': [
   ev('11.5', '既有較寬模型的小對照，每版都使用3,830個有效回答目標；驗證十二題與最後檢查十二題是分開的兩組題'),
   ev('11.5', '這批只有十二題，不能選成所有模型的最佳範圍。比較的是這份資料與配方，而不是單看參數排名。'),
   ev('11.6', '實際比較還要同一初始模型、明確的更新範圍、同一批留出圖文題，以及基模原本會做的文字題。第一階段接頭需要帶到第二階段，不能花完預算又從隨機接頭重新開始。'),
   ev('11.6', '既有小對照將兩階段合計 18,238 個有效目標，直接方案到最後一批為 18,256，差 18 個。最後圖片問答分別 9/12、10/12；一題差距不支持普遍優劣'),
   ev('11.6', '直接版本未另列原文字留出分數，所以這個比較不支持文字保留結論。')],
 'actual_prerequisites': [
   ev('11.4', '同一張紅圓圖，問顏色希望答「紅」，問形狀希望答「圓形」。圖片沒變，問題決定要取哪項資訊', 'prior main'),
   ev('11.4', '同一原圖的所有問答都要放在同一組；否則圖片可能已在訓練中出現，卻把換問法的題目當成新圖檢查。', 'prior main'),
   ev('7.3', 'Y中的-100表示忽略直接答案代價，不是輸入token。', 'prior main')],
 'original_promise': '11.5成本/效果比較与11.6兩阶段對直接方案的公平预算方法，有限实测不作普遍範圍/路線優劣結論。',
 'mechanism_check': {'judgment': '已足夠', 'basis': '11.5每輪重設凍結、参数统计、partial不同入口约定；11.6有效目标预算、两阶段carryover和最后批18差均解释。'},
 'need_check': {'judgment': '已足夠', 'basis': '相同題目/起点/有效目标避免额外資料解释优势、原文字题另验；原问题并非最优训练配置搜索。'},
 'origin_dispositions': [
   {'origin_id': 'M-11.5-experiment-scope', 'grade': 'optional', 'disposition': '採用',
    'reason': '較寬模型與十二題表已有限定，非必要缺口；将width8示例與width64/兩層歷史實驗身份相接，並说明十二题是六种颜色形状各两问/新位置，收益直接。',
    'minimum': '11.5表前共用两句任务/模型/留出定位，结果字段与全参数资料留选读。'},
   {'origin_id': 'M-11.6-experiment-scope', 'grade': 'optional', 'disposition': '採用（併入共用摘要）',
    'reason': '已有阶段、预算、失败及未测原任务界限；复用11.5材料身份，再点明描述阶段只更新接头/QA与直接更新全模型，能接回「材料与开放参数不同」而不重做卡。',
    'minimum': '沿用共用實報定位，加一短句阶段开放范围；勿重复模型/数据表。'}],
 'counterevidence_and_necessity': '11.4已教同图换问题/同问换图及原图家族留出，11.5区分验证/最后十二题，11.6给18差與direct缺文字成績。两页折疊都定位vqa實報；不需要完整CLI或逐題素材才能理解比較原理。',
 'understanding_benefit': '明白表格對的是已知RGB×shape在新位置的短答，而非一般照片QA；也能分清width8参数练习与另一个较宽真训练。两阶段和direct材料/更新身份由一处链接接上。',
 'minimal_patch': '11.5表前一到两句：实测文字核心宽64/两层、视觉入口宽16，任务是RGB×圆/方各问颜色和形状；train offset -2/-1/0，验证1、最后2，同原图所有问答同组，连既有vqa报告。11.6只回指同一任务/留出摘要，并短点描述阶段只调接头、问答阶段/直接方案调全部模型。无需列所有设备/步数/逐题表。',
 'new_burden': '一处2句摘要加一处短回指；宽度/深度/范围/留出均已教。说明的是这份历史设置，不制造推荐配置。',
 'repair_scale': 'paragraph across two pages; shared optional summary', 'page_chapter_order_rewrite': False,
 'actual_raw_check': '读取vqa快照data/split记录条件、training config与freeze_scope、各variant评测汇总、text_data、two_stage_alignment_training与direct budget；width64/layers2，视觉宽16，train36/val12/test12，offset留出，3830目标，14408+3830=18238/direct18256。未看全部history/完整checkpoint。',
 'discovery_phase': '两个reader独立origin之提示后合并处置；不计新独立发现'
})
issues.append({
 'id': 'C-M-padding-wording', 'origin_ids': ['T-M-01'], 'pages': ['13.3'],
 'original_grades': {'technical_initial': 'optional', 'reader_main_page': 'pass', 'continuity_initial_page': 'pass'},
 'cross_severity': 'optional', 'necessary': False, 'disposition': '採用',
 'strongest_source': [
   ev('13.3', '下面只把答案位置的log機率相加，前文和補長用的空格不計分。'),
   ev('13.3', '標籤`[-100,0,2]`忽略第一位置，第二取候選0，第三取候選2。工具先轉成log機率，再按標籤取值、加總。')],
 'actual_prerequisites': [
   ev('7.6', 'PAD是輸入專用ID，-100則是答案忽略值。', 'newly reread actual prior'),
   ev('7.6', '不能只把輸入值填0，就期待loss自動知道是補齊。', 'newly reread actual prior')],
 'original_promise': '给定每个目标前缀求回答/EOS总log，并区别sum/mean；不忽略答案中的正常空白字元。',
 'mechanism_check': {'judgment': '已足夠', 'basis': 'labels忽略、目标项求和、chain conditional、sum/mean和EOS约定均清楚。'},
 'need_check': {'judgment': '已足夠', 'basis': '长小数连乘难处理，log转加法；只计完整答案而非prompt/padding由7.3/7.6已教。'},
 'counterevidence_and_necessity': '「補長用」已限定填充用途，7.6明确PAD与-100，本页无需新算法解释，正常读者可推得不是答案空白。技术报告的byte tokenizer说明作为同行技术查证来源，未冒充本轮另跑编码检查。',
 'understanding_benefit': '防止把填充位置误读成自然语言空白，且使词汇与已教7.6一致；成本只有一词。',
 'minimal_patch': '「前文和補長用的空格不計分」改成「前文和補齊用的PAD位置不計分」。',
 'new_burden': '无新先备或段落；复用PAD名称。', 'repair_scale': 'paragraph (one phrase)', 'page_chapter_order_rewrite': False,
 'verification': '13.3和7.6来源实读；不执行ByteTokenizer或sequence_log_probability。',
 'discovery_phase': '原technical独立origin之提示后措辞裁定'
})

# Recheck all 13 substantive passes from the sealed initial; copy the original
# basis without rewriting it, then save the new, explicitly prompted judgment.
pass_reasons = [
 ('保留', '像素与patch轴/特征宽不同；变宽只换可表示形式，已学语义仍要训练。', '要依序处理图片与匹配文字宽由10.2/10.6直接给出；不从扩大宽度反推出“更多知识”。', '10.3的用途由目标建立、10.6尺寸相容不等于语义相容，排除常见过度结论。'),
 ('保留', '同位labels随模态展开，再只做一次下一项对齐。', '新增特征移后文和首答案预测位置，需同步移动监督；7.4的已对齐接口与10.7未对齐接口有清楚责任。', '不是把答案给助手抄；mask/位置按实际六格建，不能沿用原五格。'),
 ('保留', 'backward/grad、optimizer名单、step、无图一致、替图敏感、完整自由回答是不同证据。', '验证模态能接通、两端能力存在和真实任务完成需要不同检查，11.1/11.2直接建立其必要性。', '11.3真正训练0/6和teacher forcing说明loss下降不等于生成成功；无需补完整底座卡才懂此关系。'),
 ('保留', '归一化消倍率，行列竞争不同，非对称分数表和转置已展示。', '两个查詢都重要是直接给出的需要；平均权重表达重视程度。', '10.11未声称双向必胜，检索/生成明确分开；实测定位仅optional。'),
 ('保留', '同图换问选范围，同问换图查图片作用；替图需新真值，遮图查原答变化。', '原任务的颜色/形状与多数类失敗直接给这些控制的用途。', '“红色圆形”对只问颜色会多答，且总高分不保证稀少需求成功；不是另加审阅目标。'),
 ('保留', '可见笔画才有OCR监督，整串/CER/EOS、读序、已给区域与自动定位分开。', '编号须整串、两行信息有先后、指定范围须取所问信息均来自原问题。', '固定框路线明确只集中少量字形字序；不要求此路线自学自动检测。'),
 ('保留', '平均可使不同空间/时间排列成相同表示；位置序列可留线索但需任务训练。', '问左右/先后才产生位置需要，由原同物件不同排列任务直接给出。', '保序并不保证学会关系，更大文字核心也不能可靠补丢失像素。'),
 ('縮窄', '波形/时间、框重叠、窗接点、二相频率样板、功率、频带加权、logfloor和转轴操作仍通過。12.2超界“控制”基本动作须补。', '取样时间、局部频率、窗跳变、功率跨度、时间序列读取用途有已教来源；12.6采用mel非均匀频带理由须补，不能由操作通过代答。', '把原第8条宽泛的“分别解释”缩为上述已证关系；C-M-resample-action与C-M-mel-adoption保留局部必要修补，并未推翻整条音讯链。'),
 ('保留', 'ASR输出字串、直接音讯保原线索、历史由系统存并输入模型；手写希望回答不同于真实成绩。', '原问题需要音高、有限意图回复、跨输入保主题/格式，各有具体当前例。', '字串相同不能答哪段音高高；三意图不等于任意中文ASR。'),
 ('保留', '同题条件偏好，每回答条件链求sum含EOS，减固定参考、再比两答案；beta局部梯度与训练结果分开。', '偏好依题目限制而非泛化风格，log用于小数连乘，固定参考相对变化而非真值，由原问题及例子给出。', '相对改善可仍同分、候选胜出可仍自由生成失败；PAD词候选只一词optional。'),
 ('保留', 'reward看已选卡、critic看选前情境、old存本轮、ref存全期起点；fixed advantage、ratio/minclip、KL、step各有角色。', '相对预期确定好坏方向、重用旧批需知道已改多少、不同时间基准不可混，原文均有具体情境。', 'clipping只是停此项有利鼓励；有限单步卡不代表逐token大型RLHF。'),
 ('保留（限定用途）', '13.13明确ratio不是硬锁，13.12 old本轮/reference起点；13.14枚举KL度量完整分布偏移并加代价。', '来源已给共享更新可能影响别项、重用旧样本控制推力仍要量实际偏移；固定全期起点提供另一个基准。最短推论是对整体累计变化另量/付代价，不需自补新用途。', '不能把KL控制偏移写成保住原任务的保证，奖励也可能错；13.16明确裁切/ref不替错奖励改标准。'),
 ('保留', '评员正确、policy正确、完整要求不同；PPO/DPO同起点分叉但额外工作不同。', '任务规则独立于reward，且两种偏好训练安排成本有差，是原文的判断需要。', '90/90评员非策略90/90，12/18含句条件0/6，不称hacking已发生、方法普遍优越或已完成成品RLHF。')
]
strong = []
for i,(old,values) in enumerate(zip(own['strong_passes'],pass_reasons),1):
    status, mech, need, counter = values
    checked = [ev(q['page_id'], q['quote'], 'sealed independent basis rechecked') for q in old['quoted_basis']]
    strong.append({'id': 'SP-M-%02d'%i, 'initial_ref': i, 'pages': old['pages'],
                   'initial_judgment_preserved': old['judgment'], 'source_basis': checked,
                   'cross_disposition': status, 'mechanism_recheck': mech,
                   'need_recheck': need, 'strongest_counterexample_or_limit': counter})
strong[11]['additional_strongest_source'] = [
    ev('13.13','其他樣本與共享權重仍可能改機率，所以ratio不是被保證留在區間；還要量實際偏移。'),
    ev('13.12','old對照本輪收集那一刻，下輪重新收集時換新紀錄；reference對照PPO開始時的策略'),
    ev('13.16','固定reference或PPO裁切，都不能替錯誤獎勵改判準。')]

raw_names = ['result-contrastive.json','result-projector.json','result-vqa.json']
raw_sources = [{'path': str(B/'technical-modalities-work/originals'/f),
                'sha256': sha(B/'technical-modalities-work/originals'/f),
                'scope': 'selected metadata, task/splits/config/update scope and final summaries; not all training history, no checkpoint execution'} for f in raw_names]
full_reread = ['7.6','12.2','12.6','13.3','10.11','11.3','11.4','11.5','11.6','12.1','12.3','12.5','12.7','12.8','12.15','13.9','13.12','13.13','13.14','10.5','10.10','13.15','13.16','13.17']
origin_ids = [p['origin_ids'] for p in issues]
flat = [z for x in origin_ids for z in x]
expected = {p['id'] for p in json.loads((B/'reports/reader-modalities.json').read_text())['issues']}
expected |= {p['id'] for p in json.loads((B/'reports/technical-modalities-initial.json').read_text())['issues']}
assert len(flat) == len(set(flat)) == 7 and set(flat) == expected

out = {
 'schema_version': 1, 'reviewer': '/root/continuity_modalities', 'role': 'continuity cross reviewer',
 'group': 'modalities', 'phase': 'post-seal prompted cross review', 'created_at': now,
 'instructions_path': str(B/'cross-review-instructions.md'), 'instructions_sha256': sha(B/'cross-review-instructions.md'),
 'source_reports': report_sources,
 'seal_recheck': {'own_initial': True, 'own_main_notes': True, 'own_source_delivery': True,
                  'technical_initial': True, 'reader_main': True,
                  'own_initial_seal_path': str(B/'reports/continuity-modalities-initial-seal.json'),
                  'own_initial_seal_sha256': sha(B/'reports/continuity-modalities-initial-seal.json'),
                  'own_main_seal_path': str(B/'work/continuity-modalities/main-notes-seal.json'),
                  'own_main_seal_sha256': sha(B/'work/continuity-modalities/main-notes-seal.json'),
                  'same_group_supplements_found': [], 'original_reports_changed': False},
 'read_scope': {
   'reader_main': 'all page assessment/need/basis summaries and all 6 origins; final13.10–13.17 four-question records separately read after failed float page selection corrected',
   'reader_final': 'all 6 issue/followup records, all extras followup summaries, key page records and final grade/scope summary',
   'technical_initial': 'all 66 page mechanism/need judgments, one issue, empirical/visual/execution scope and relevant contrastive/projector/vqa evidence',
   'own_initial': 'all 13 strong passes, both disputed per-page records and sealed scope; initial judgments remain untouched',
   'cross_full_source_rereads': [{'page_id':p,'path':str(S/(p+'.md')),'sha256':sha(S/(p+'.md'))} for p in full_reread],
   'cross_selected_prior_passages_reread': ['7.3','11.1','11.2','10.7'],
   'initial_actual_prerequisite_range': own['actual_prerequisites'],
   'new_actual_prior_read': ['7.6'],
   'raw_results': raw_sources,
   'main_notes_or_source_delivery_modified': False,
   'new_training_or_execution': False,
   'images': 'no image/layout origin dispute; no new visual claims. Inherit independent 28×640, 6×360 and 8 local page capture scope; do not claim all current site runtime/mobile/interactive acceptance.'},
 'independence': {'cross_is_second_independent_discovery': False,
   'peer_visibility': 'same-group reports explicitly allowed after all three rounds sealed; original continuity and main notes preserved',
   'new_observation_timing': 'all new judgments recorded here after reader and technical report visibility; two own-pass revisions are prompted cross adjudications',
   'old_historical_reviews_read': False, 'tutorial_or_training_modified': False, 'subagents_spawned': False},
 'issues_dispositions': issues,
 'strong_pass_rechecks': strong,
 'unknowns': [
   {'id':'U-M-raw-results','status':'partially reduced', 'scope':'only contrastive/projector/vqa selected snapshot data/conditions and summaries read this round. Other OCR/audio/DPO/PPO raw histories/checkpoints and replay execution remain outside this continuity audit; no new training.'},
   {'id':'U-M-execution','status':'unverified scope','scope':'no snippets/training/inference rerun in this cross review; technical execution results remain separately attributed to technical report.'},
   {'id':'U-M-live-layout','status':'unverified scope','scope':'no added visual disagreement. Initial actual view scope preserved; site runtime outputs/reading annotations/remaining mobile figures/interactions not newly verified.'},
   {'id':'U-M-external-theory','status':'unverified scope','scope':'no external auditory mel theory or other original papers newly read; proposed local reason repair does not assert universal perceptual superiority.'},
   {'id':'U-M-prior-scope','status':'unverified scope','scope':'initial 17 actual prerequisite pages preserved plus7.6 read; other previous source and figures not broadly re-audited.'}],
 'counts': {'unique_origin_ids': 7, 'reader_final_required_origins': 2, 'reader_final_optional_origins': 4,
            'technical_optional_origins': 1, 'blocker':0, 'burden':2, 'necessary_units':2,
            'optional_origins_reviewed':5, 'optional_units_after_dedup':4,
            'optional_units_adopted':3, 'optional_origins_adopted':4,
            'optional_units_keep_original':1, 'optional_origins_keep_original':1,
            'deferred_units':0,'pending_check_issue_units':0,'unknown_scope_groups':5,
            'strong_passes_rechecked':13,'strong_passes_retained_or_limited':12,'strong_passes_narrowed_for_required_relations':1},
 'repair_scale': {'paragraph': '2 necessary local relations; 3 adopted optional units (10.11 scope phrase, shared11.5/11.6 scope summary,13.3 PAD phrase)',
                  'page_rewrite_required':False,'chapter_rewrite_required':False,'order_change_required':False,
                  'reason':'Current main mechanisms and transitions remain coherent; two gaps do not require a new section, reordering, complete reproduction cards or added experiments.'},
 'conclusion': '必要2項，均paragraph級；可選5個origin去重4項，其中採用3項/4origins，11.3保留原文。未知為5個核驗範圍而非5個教材缺陷。無page/chapter/order改寫必要。'
}
report_path = B / 'reports/cross-modalities.json'
report_path.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
seal_path = B / 'reports/cross-modalities-seal.json'
seal = {'schema_version':1, 'reviewer':'/root/continuity_modalities','group':'modalities',
        'phase':'post-seal cross review; no original report mutation', 'sealed_at':datetime.now(timezone.utc).isoformat(),
        'report_path':str(report_path),'report_sha256':sha(report_path),
        'source_reports':report_sources,'own_initial_sha256_unchanged':sha(B/files[3]),
        'own_main_notes_sha256_unchanged':sha(B/files[4]),
        'own_source_delivery_sha256_unchanged':sha(B/'work/continuity-modalities/source-delivery.jsonl'),
        'counts':out['counts']}
seal_path.write_text(json.dumps(seal,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'report':str(report_path),'sha256':seal['report_sha256'],'seal':str(seal_path),'counts':out['counts']},ensure_ascii=False))
