import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone

B = Path('/workspace/work/tutorial-audit-20261008')
R = B / 'reports'
S = B / 'freeze/sources'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def ev(page, line, quote=None, route='main'):
    p = S / (page + '.md')
    text = p.read_text().splitlines()[line - 1]
    if quote is None:
        quote = text
    assert quote in text, (page, line, quote)
    return dict(page_id=page, line=line, quote=quote, source_sha256=sha(p), route=route)

initial = json.loads((R / 'continuity-architecture-initial.json').read_text())
own_seal = json.loads((R / 'continuity-architecture-seal.json').read_text())
main_seal = json.loads((B / 'work/continuity-architecture/main-notes-seal.json').read_text())
assert sha(R / 'continuity-architecture-initial.json') == own_seal['initial_sha256']
assert sha(R / 'continuity-architecture-main-notes.json') == own_seal['main_notes_sha256'] == main_seal['sha256']
reader = json.loads((R / 'reader-architecture.json').read_text())
reader_main = json.loads((R / 'reader-architecture-main.json').read_text())
technical = json.loads((R / 'technical-architecture-initial.json').read_text())
technical_seal = json.loads((R / 'technical-architecture-initial.seal.json').read_text())
assert reader['issues'] == reader_main['issues']
assert sha(R / 'reader-architecture-main.json') == reader['main_report_seal']['sha256']
assert sha(R / 'technical-architecture-initial.json') == technical_seal['report_sha256']

reports = ['reader-architecture-main.json', 'reader-architecture.json',
           'technical-architecture-initial.json', 'continuity-architecture-initial.json']
all_initial_issues = reader['issues'] + technical['issues'] + initial['issues']
origin_lookup = {x['id']: x for x in all_initial_issues}

def origins(ids):
    out = []
    for oid in ids:
        x = origin_lookup[oid]
        report_names = []
        if oid.startswith('architecture-'):
            report_names = reports[:2]
        elif oid.startswith('technical-'):
            report_names = reports[2:3]
        else:
            report_names = reports[3:4]
        out.append(dict(id=oid, original_severity=x['severity'],
                        source_reports=[dict(path=str(R / n), sha256=sha(R / n)) for n in report_names]))
    return out

issues = []
issues.append(dict(
    id='cross-architecture-yarn-scale-purpose',
    origin_ids=['architecture-14.10-scale-purpose', 'technical-architecture-14.10-yarn-scale-purpose', 'CA-14.10-scale-need'],
    origin_records=origins(['architecture-14.10-scale-purpose', 'technical-architecture-14.10-yarn-scale-purpose', 'CA-14.10-scale-need']),
    page_ids=['14.10'],
    source_evidence=[ev('14.10',3), ev('14.10',9), ev('14.10',13), ev('14.10',15), ev('14.10',17), ev('14.10',26,route='extras')],
    actual_prerequisite_evidence=[ev('14.3',5), ev('14.3',26), ev('14.7',13), ev('14.7',15), ev('14.8',28), ev('14.9',17)],
    original_promise='延長文章時兼顧近遠；正文主動問「為什麼還有下面那一步」，並把分頻和尺度兩部分共同列為這裡介紹的YaRN。',
    mechanism_check=dict(status='已足夠', basis='快保原速、慢插值、中間過渡；正尺度放大既有分差、同分仍各一半；較集中不等較正確。手算和圖只支持這些局部作用。'),
    need_check=dict(status='需補說明', completed='分頻已接起保近處差角與把遠處範圍壓回的需求。尺度已接起可調集中度，但未指出延窗時要回應哪個新的讀取或預測需要。',
                    strongest_counterevidence='14.7說延長會增加位置/距離組合，且還有softmax；14.3已教softmax的總分母。讀者可在「原有分數不變、另加候選」這個另外選定的條件下推得原候選份量下降。14.10又說要配合訓練和長短驗收。',
                    counterevidence_limit='上述前提沒有選定當次要維持何種讀取分布，也沒有說位置改動後一定只增加固定分數的候選；把「因此應更集中」接回作者目的仍需審閱者另選條件和判準。不能用這個自補模型判原文已教，也不能宣稱更多候選一律要求放大。',
                    minimal_gap='為何延窗和改位置規則後，YaRN另考慮注意力尺度；這可以是指定長窗口預測評估的經驗用途，不要求普遍因果定理。'),
    external_cross_check=dict(path='/workspace/work/technical-architecture/yarn-2309.00071v3.txt',
                             sha256=sha('/workspace/work/technical-architecture/yarn-2309.00071v3.txt'),
                             actual_read_scope='本交叉輪實讀layout lines289–336與743–786；沒有稱整篇已讀。',
                             quote='In addition to the previous interpolation techniques, we also observe that introducing a temperature t',
                             supporting_lines=[295,296,305,329,330,331,332,754,756,757,758,771,772,777,778],
                             role='核對技術輪提出的最小修补是否有原始經驗依據；外部來源不回算main已教，亦不當本教材實測。'),
    final_severity='burden', disposition='採用',
    concrete_benefit='讀者除會算集中程度外，能說出長窗口方案為何加入第二個可調項，以及它只是一項需驗收的配方選擇。',
    minimum_repair='在14.10「為什麼」段補一至兩句，把延窗/改位置後需要重新核對注意力讀取與長文預測，接到作者在指定長窗口比較尺度的經驗發現；保留原本較集中不等正確、需適應與長短驗收。',
    repair_text_candidate='延長窗口並改變旋轉規則後，還要檢查注意力的讀取分布是否適合新的長文條件。YaRN作者在指定長窗口的文字預測評估中觀察到，調整softmax前的尺度可改善結果，因此把它與分頻插值搭配；這是其配方的經驗依據，並不保證越集中越好。',
    new_burden='新增一個有範圍的文獻經驗關係；可用已教的文字預測評估描述，避免另引entropy、perplexity公式或通用最佳係數。候選文字須由主agent核對最終表述，這裡未改教材。',
    repair_scale='paragraph', preserved_passes=['分頻的近遠需要', '兩部分操作機制', '同分練習與較集中不等正確限定'],
    discovery_status='三個封存前來源初判提出同一關係；此處互見後重新裁定，不以三票作理由。'))

issues.append(dict(
    id='cross-architecture-top2-choice-purpose',
    origin_ids=['architecture-15.7-top2-choice'],
    origin_records=origins(['architecture-15.7-top2-choice']), page_ids=['15.5','15.7'],
    source_evidence=[ev('15.5',3),ev('15.5',5),ev('15.5',24),ev('15.5',31,route='extras'),ev('15.7',3),ev('15.7',7),ev('15.7',26)],
    actual_prerequisite_evidence=[ev('15.3',3),ev('15.4',3),ev('15.5',26)],
    original_promise='15.5先說兩種規則合理且尺度不同，再承諾到15.7說明本教材top-2正規化、top-1保原比例的原因；不是要求證明某一算法普遍優越。',
    mechanism_check=dict(status='已足夠',basis='原比例與選中者重新正規化兩種加權和完整；top-1 p/p恆1消除這條任務梯度，保p可學；top-2兩個相對係數仍可變，不會自動堵同一路。'),
    need_check=dict(status='需補說明',completed='同寬單向量合併的需要由下一層期待與不能拼接/覆蓋支持。top-1保留gate的需要由固定輸出2、目標0充分支持。',
                    strongest_counterevidence='15.5明說「若希望選中兩位的比例合計為一」與歸一化結果在兩位輸出的混合之間；15.7說top-2不成恆係數；選讀明稱此為約定/明示選擇。',
                    counterevidence_limit='這些資料足以知道選中的规则和可學性，但「希望合計一」仍是新運算的直接產物，尚未交代當次為何要去掉未選中比例的總質量或以什麼合併含義完成示範。兩個都能保同寬的规则不會因此唯一選出top-2約定。折疊的「明示選擇」也沒有完成正文承諾的理由。',
                    minimal_gap='本次top-2選擇正規化的最低用途，與top-1例外分開說明。'),
    final_severity='burden', disposition='採用',
    concrete_benefit='能辨認top-2係數表示選中者之間的相對分工、與保留全路由比例的含義差異，並解釋為何本示範採其中一種；不用猜測有未呈現的品質优势。',
    minimum_repair='在15.5兩规则比較處補一小段當次選擇目的，例如是否要讓「選中者相對份量不變」時，不因未選中者分到的總量改變而把兩位輸出一起縮小；若只是為示範某種合併含義選的設定，據實交代示範目的與約定，並保留top-1為任務梯度例外。不能僅刪「原因」躲過選擇問題。',
    repair_text_constraint='上述用途是待採用的修補方向，不宣稱原文已給，也不能未確認實際設計目的就寫成作者已證明的品質優勢。最小可用修補是明示示範選定的合併含義；不需重跑所有top-2变体。',
    new_burden='約兩句新的選擇說明，沿用已教比例和輸出尺度；新增過度穩定性/品質理由會偷渡目的，故不採。',
    repair_scale='paragraph',preserved_passes=['同寬加權和及殘差區別','top-1任務梯度需要','top-2不成恆係數的性質'],
    discovery_status='由已封存reader origin提示後回查；technical和continuity初判沒有報此項，不據此否定它，也不冒稱銜接輪獨立發現。'))

def name_item(oid, evidence, role, purpose, classification, severity, disposition, benefit, repair, burden, prior=None, correction=None):
    x=origin_lookup[oid]
    issues.append(dict(id='cross-'+oid, origin_ids=[oid], origin_records=origins([oid]), page_ids=[x['page_id']],
                       original_promise='命名已教的當次概念與使用角色；此origin是名稱/表述建議，沒有提出缺少整套算法。',
                       source_evidence=evidence, actual_prerequisite_evidence=prior or [],
                       mechanism_check=dict(status='已足夠',basis=role),
                       need_check=dict(status='已足夠',basis=purpose),
                       name_completeness=classification, original_claim_correction=correction,
                       final_severity=severity, disposition=disposition,
                       concrete_benefit=benefit, minimum_repair=repair, new_burden=burden,
                       repair_scale='paragraph' if disposition=='採用' else None,
                       possible_scale_if_later_chosen='paragraph（詞句；不牽動頁/章/順序）',
                       discovery_status='封存reader可選origin互見後逐項裁定；原始reader route未讀的先備仍保留其路線事實。'))

name_item('architecture-14.1-name',[ev('14.1',5,'這種把位置寫進旋轉的方式叫 Rotary Position Embedding，常縮寫為 RoPE。')],
          '成對Q/K按位置旋轉、共同平移保相對角/長度/點積已完整。','句首新增文字而詞距不變的比較需要，14.1第3行已有獨立來源。',
          '英文全名已給；中文角色清楚；缺的是常用中文對名，沒有概念缺口。','optional','採用',
          '把「把位置寫進旋轉」與既有embedding中文詞彙直接對上，便於按中文方法名找同一段。','首個完整名稱旁補「旋轉位置嵌入」。','增加六字左右，不另講embedding或長窗能力。')
name_item('architecture-14.2-name',[ev('14.2',5)],
          '平方、均值、根、逐項除與共同偏移保留已教；RMS英全名也已給。','控制同位置特徵尺度、分辨減均值與保偏移的需要由第3行和對照支持。',
          'Root Mean Square/RMS與正規化角色明確；整個RMSNorm中文對名可補。','optional','採用',
          '把已教「均方根」及「正規化」組合成同一方法標記，避免把RMS統計量與整層名稱混記。','首處RMSNorm旁補「均方根正規化」。','僅一詞，不重列公式。')
name_item('architecture-14.3-name',[ev('14.3',7)],
          '各Q/K除自身L2長度，特徵軸與候選softmax軸區別完整。','比較方向、避免同角候選僅因長度不同獨占份量由第3行支持。',
          'L2 normalization英文方法名已明給；僅中文對名未列。','optional','採用',
          '讀者可把「正規化」中文動作和同段L2方法標記對應，與前一頁RMSNorm並列回憶。','補「L2正規化」。','一詞；不再展開范數理論。',
          correction='origin「正式雙語全名未列」過寬；英文L2 normalization已在正文，不補不存在的英名缺漏。')
name_item('architecture-14.4-mlp-name',[ev('14.4',3)],
          '矩陣之間插非線性與兩層純線性可合成的對照明確。','增加可表示的新關係由输入1/2的不同倍數及已教2.4需要支持。',
          '全書更早2.5正文已有完整雙語名稱及網路角色；reader局部路線未讀，不等於全書首次缺名。','none','保留原文',
          '重補同名不增加當頁線性/非線性理解；真正收益是確認可正常引用更早定義。','無必要新增；日後若做跳讀索引可連2.5。','在14.4再寫全名會重複名稱，不解新關係。',
          prior=[ev('2.5',26,'這樣的小網路叫**多層感知器**（Multilayer Perceptron，MLP）。')])
name_item('architecture-14.4-relu-name',[ev('14.4',5)],
          'ReLU保正截負、ReLU²再平方完整。','非線性增加轉折的需要已由本頁和2.4回答離零多遠的問題支持。',
          '全書更早2.4正文已有完整中英文/ReLU及操作。','none','保留原文',
          '既有讀者可直接沿用前文；重列英名不幫本頁比較GELU與ReLU²。','無需補全名。','重複先備而沒有新的操作收益。',
          prior=[ev('2.4',7,'本節用**整流線性單元**（Rectified Linear Unit，ReLU）：保留正數，把負數改為 0。')])
name_item('architecture-15.3-name',[ev('15.3',7)],
          '連續softmax權重混合所有expert完整；未把比例當正確率。','依輸入決定各套規則多少由第3行支持；本頁先理解router、未省稀疏工作已明說。',
          'soft routing英名及直接中文角色完整；中文標準對名未列，是低收益詞彙機會。','optional','保留原文',
          '「軟路由」可縮短對名，但現有「連續權重混合」對新手更直接，沒有新的理解或操作收益。','若全書將來統一雙語詞表，可補「軟路由」。','正文增加別名會多一個需記的標記；本輪保留直白描述。')
name_item('architecture-15.4-name',[ev('15.4',3)],
          '選最大k與真正只執行選中者分開，整數indices和連續weights角色清楚。','保多套規則、每token只花少數套計算的獨立需要已教。',
          'top-k routing英名已有直接中文操作；正式中文別名可選。','optional','保留原文',
          '「前k項路由」不比「只保留比例最大的k位」更清楚，未增加當次判斷成本是否已省的能力。','僅在術語表统一時考慮中文別名。','再加不熟的中文技術名可能讓讀者以為另有算法。')
name_item('architecture-15.6-names',[ev('15.6',29),ev('15.6',31)],
          'dispatch按來源token列送出、combine按原列號累加、dropless執行所有selected工作已各自立即中文解釋。','恢復原序與不丟同token多筆貢獻由覆蓋反例支持。',
          '英名与中文动作已一一對上；只有未指定的格式統一要求，未核實新的名稱缺口。','none','保留原文',
          '統一括號格式只改外觀，不新增對象或動作區別；沒有足以支持本輪改動的具體理解收益。','無需新增三份譯名或重寫角色。','額外名詞只重複已給的送出/加回/不丟工作。')

oid='architecture-15.8-scope-wording'
issues.append(dict(id='cross-'+oid,origin_ids=[oid],origin_records=origins([oid]),page_ids=['15.8'],
                   source_evidence=[ev('15.8',3),ev('15.8',22),ev('15.8',24)],actual_prerequisite_evidence=[ev('15.4',3)],
                   original_promise='解釋路由塌縮與如何辨識；受控例只證偏斜形成/辨識，不稱真實訓練必然塌縮。',
                   mechanism_check=dict(status='已足夠',basis='人工每列同排名即使平均p接近均勻仍12/0/0；bincount分母清楚。'),
                   need_check=dict(status='已足夠',basis='診斷實際參數是否收到工作、容量與拥塞風險；均勻不等最好品質。'),
                   counterevidence='「形成」可合理指固定排名造成選擇集中；前文明說人工反例，主文沒有稱測出回饋訓練過程，因此不升必要或技術錯誤。',
                   final_severity='optional',disposition='採用',
                   concrete_benefit='把固定排名造成集中與第3行描述的長期學習回饋過程分開，讀者更準確地說出這個程式究竟示範哪一步。',
                   minimum_repair='將「偏斜如何形成與如何辨識」精確成「固定排名如何造成工作集中，以及如何辨識」；保留人工/受控和非必然塌縮的限制。',
                   new_burden='一句內替換，不增加實驗或先備；原句並非錯誤。',repair_scale='paragraph',
                   discovery_status='封存reader可選origin，互見後採用精確化，不冒稱新增必要發現。'))

name_item('architecture-15.10-name',[ev('15.10',17)],
          '參數格數×4byte的局部儲存量與active proxy/峰值/速度分开。','區分裝置能否存下與每token經過多少套規則已有需要来源。',
          '全書5.9已明說FP32是32bit浮點格式、每格4bytes；15.10不是首次未教。','none','保留原文',
          '不重補已教格式；當頁與前文同一×4約定已可操作。','無需補32位元浮點名稱。','重複5.9定義，無新增成本界線。',
          prior=[ev('5.9',5,'FP32是用32個bit表示一個浮點數的格式，所以每個參數占4 bytes；這種數值儲存格式在程式中叫dtype。')])
name_item('architecture-16.2-names',[ev('16.2',3),ev('16.2',5)],
          '提示各位置Q與每次新位置Q仍讀所有舊KV的工作完整。','首字等待与逐步生成等待分開量的需要已有明确例子。',
          '英名及中文阶段工作已教；正式「預填充/解碼」別名可選。','optional','延後',
          '若后續全書查詢標準中文術語，「預填充/解碼」有對名收益；本頁理解兩種工作以「處理整段提示/剛加入token」已更直接。','將來有全書雙語名統一目標時補兩個别名，保持現有動作句。','預填充是新術語；decode又可能与ID轉文字的解碼混同，必須附本頁生成阶段含義，故本輪不逕增。')
name_item('architecture-16.4-name',[ev('16.4',26)],
          '四Q共享一KV端点和GQA分组清楚，未把头号当位置号。','長對話降低每層保存KV bytes的需要和训练/品质/总内存限制齐。',
          'Multi-Query Attention/MQA英文全名與中文角色已给；只有中文正式对名可补。','optional','採用',
          '与已给「分组查询注意力」并列对照，同一组命名中的MQA端点可用中文记忆，仍能对到图上保留多Q。','在MQA首处补「多查詢注意力」。','一词；保留全Q共享一KV解释，避免由中文名错推KV也是多组。')
name_item('architecture-16.7-float-names',[ev('16.7',5),ev('16.7',25,'第一行應看到原輸入仍torch.float32，輸出是torch.bfloat16。')],
          'FP16/BF16位元、范围/有效数精度和CPU舍入差异已明确。','省空间及可能支援硬体收益与小梯度下溢取舍已教，不必以名词补机制。',
          '兩種16bit格式的性质/意义已教；BF16与程式dtype bfloat16的別名映射尚可直接给，FP16的16位中文已给。','optional','採用',
          '让章节的BF16标签直接对应读者实际要填写的torch.bfloat16，而非猜第三种格式；FP16也能对到float16。','首处给FP16（float16，16位元浮點格式）/BF16（bfloat16）的别名关系，不增加不必要的命名史。','短括号；不再宣称缺少16位中文含义，也不增加Brain等来源未教的历史词。',
          correction='origin「完整雙語全名未列」不能解释成两格式中文性质未教；需要核实的是名字与dtype别名，不是更多范围推导。')
name_item('architecture-16.7-gradscaler-name',[ev('16.7',27)],
          'loss放大→同倍梯度→更新前除回的顺序和非改LR已完整。','FP16微小梯度下溢可能变0的独立需要已明确。',
          'GradScaler是工具类名、角色已经清楚；中文短对名可选。','optional','採用',
          '将英文工具名直接连到同段真正管理的梯度尺度，便于读者在流程中指认这个工具；操作仍用原解释。','补「梯度縮放器」对名。','一词，不改loss先放大的机制或重复整段数值。')
name_item('architecture-17.3-name',[ev('17.3',22)],
          'abs后mean逐格平均与正负不抵消完整。','权重MAE不是输出/任务上界，输入方向与尺度反例齐。',
          '中文定义/MAE已经给，首处英文展開未给；18.6后给英名不回写首处。','optional','採用',
          '让MAE三字母能直接对应mean/absolute/error与代码abs/mean，便于认报告栏名；不把此名字当任务品质准则。','17.3首个MAE旁补mean absolute error。','三英文词，不增统计算法。',
          prior=[ev('18.6',25,'MAE是mean absolute error（平均絕對差）')])
name_item('architecture-17.14-name',[ev('17.14',7),ev('17.14',27)],
          'fake forward值、detach近似反向、与round真导数不同已完整。','让量化前向仍有训练路径的需要由round格内导数0支持。',
          'STE中文名称/机制清楚；首处英文展開未给。','optional','採用',
          '把STE三字母与已有直通估计中文标记一一对应，读者能辨认论文/接口中的同一近似而不误当round真实导数。','补straight-through estimator。','三个英文词；不追加证明或训练。')
name_item('architecture-18.1-name',[ev('18.1',11)],
          '回答文字目标、硬目标与候选比例目标的信息区别已明示。','同原题仍可能收到不同学习讯号的需要由第3行支持。',
          '全書7.1正文已给Supervised Fine-Tuning、监督式微调、SFT及继续已有模型的角色；reader只读7.11是路线事实。','none','保留原文',
          '正常沿用既有术语即可；此处重列英文全名不会澄清硬/软目标差异。','不重复补全名。','多一份已教名字；不补真实机制。',
          prior=[ev('7.1',25,'用正確助理示範繼續調整已有模型，叫**監督式微調**（Supervised Fine-Tuning，SFT）。')])
name_item('architecture-18.8-name',[ev('18.8',7)],
          '读相同已知前文预测同一下一token、忽略直接KL仍影响后续的动作清楚。','防自由生成不同前文却盲按行比较的需要已明说。',
          'teacher forcing更早5.8已有英名和直接中文含义；18.8再次完整说明。正式「教師強制」并未增加当前意思。','none','保留原文',
          '现有「用已知前文餵入」比另加「教師強制」更直接，后者还可能误以为必须先调用蒸馏教师；没有足够新增理解收益。','保留英名与直接中文动作；若统一术语表，可列现有「使用標準前文」释义。','新增正式译名会重复甚至让teacher误联蒸馏teacher模型，故不采用。',
          prior=[ev('5.8',5)],
          correction='完整中文含义并非首次欠缺；没有正式逐字译名不等于角色未教。')

assert len(issues)==20
origin_ids=[i for x in issues for i in x['origin_ids']]
assert len(origin_ids)==len(set(origin_ids))==22
assert set(origin_ids)==set(origin_lookup)
assert sum(x['severity']=='optional' for x in reader['issues'])==18

passes=[
 dict(id='CP-14-relative-position-length',pages=['14.1','14.7','14.8','14.9','14.10'],
      evidence=[ev('14.1',3),ev('14.1',5),ev('14.7',15),ev('14.8',3),ev('14.8',7),ev('14.8',28),ev('14.9',15),ev('14.9',17),ev('14.10',7)],
      mechanism_status='已足夠',need_status='已足夠（14.10分頻部分）；尺度第二步另见必要项',
      shortest_inference='共同移位给Q/K同角增量，差角及长度不变→点积保留；16内容未删而m×8/16压位置，代价是邻距也压；快保/慢压响应近远两需。',
      strongest_counterexample='能算位置15不等于读好远线索；分数坐标0.5/7.5也不是原见过整数；共同旋转不保证任意整模型语义平移不变。正文明确这些条件，故保留通过。'),
 dict(id='CP-15-selected-work-combine-router',pages=['15.4','15.5','15.6','15.7'],
      evidence=[ev('15.4',3),ev('15.5',3),ev('15.5',24),ev('15.6',29),ev('15.6',31),ev('15.7',3),ev('15.7',7),ev('15.7',26)],
      mechanism_status='已足夠',need_status='只對省未选工作、同宽不丢贡献、top-1学习路判已足够；不包top-2选择目的',
      shortest_inference='先topk才execute选中者→未选者不做FFN；恢复token来源行用累加避免覆盖；top1要降低固定2贡献，保连续p可沿loss教router，p/p恒1则不能。',
      strongest_counterexample='全算后只留两份输出不省计算；按expert号恢复会乱token；top2非恒系数不构成选择它的完整need。这次明确收窄自己初判strong pass中的「延期理由回收」为top-1。'),
 dict(id='CP-16-causal-cache-gqa-packing-accumulation',pages=['16.3','16.4','16.5','16.6'],
      evidence=[ev('16.3',3),ev('16.3',31),ev('16.4',3),ev('16.4',5),ev('16.5',5),ev('16.5',25),ev('16.6',3),ev('16.6',25),ev('16.6',27)],
      mechanism_status='已足夠',need_status='已足夠',
      shortest_inference='因果旧特征不改→各层KV可重用但新位offset须延续；少未扩张KV组→按头数减少specific cache bytes；packed同文且因果隔离，并另处理边界targets；共总有效数保每token份量，末一次step。',
      strongest_counterexample='改首token仍沿旧cache、EOS自动隔离、每微批mean等权与每份step均会改变原问题；源文主动排除，packing示范不伪装已完成全部装填。'),
 dict(id='CP-16-attention-interface-storage-visibility-speculation',pages=['16.8','16.9','16.12','16.13','16.14'],
      evidence=[ev('16.8',5),ev('16.8',29),ev('16.9',5),ev('16.9',9),ev('16.12',11),ev('16.12',37),ev('16.12',39),ev('16.13',11),ev('16.13',25),ev('16.14',9),ev('16.14',32)],
      mechanism_status='已足夠',need_status='已足夠',
      shortest_inference='接口同允许/一次缩放/dropout才同题；online新最大时分子分母同倍缩保持完整加权；Flash换储存/搬移安排不删因果边，滑窗/块局部才改可见；greedy首错修正并丢错条件后缀保目标路径。',
      strongest_counterexample='大矩阵涂mask不保证T×W实际成本；两层3→5→7可达不保证保留日期；Python验证循环不证加速。源文各自有独立成本/访问/保持目标需要和具体限定。'),
 dict(id='CP-17-grid-scale-container-pack-qat-behavior',pages=['17.2','17.3','17.5','17.6','17.7','17.8','17.14','17.15'],
      evidence=[ev('17.2',3),ev('17.2',23),ev('17.5',3),ev('17.5',22),ev('17.6',21),ev('17.7',25),ev('17.8',22),ev('17.8',24),ev('17.14',7),ev('17.14',27),ev('17.14',29),ev('17.15',3),ev('17.15',17)],
      mechanism_status='已足夠',need_status='已足夠',
      shortest_inference='刻度定义码/实数关系、细格缩覆盖；大行共尺度抹小行→分尺保差但多metadata；限制码范围不改int8容器→须真实pack低高位；STE为训练路径近似而非真round导数，最终另pack；小平均改动仍可跨离散argmax/EOS边界。',
      strongest_counterexample='pack无损不恢复量化误差；full反量化float运算不自动低bit kernel；有梯度/预适应不保品质，matched小QAT实测反例保留，不要求每格4bit都更差。'),
 dict(id='CP-18-signals-temperature-alignment-target-cost',pages=['18.1','18.3','18.4','18.5','18.6','18.8','18.9','18.10','18.13','18.14'],
      evidence=[ev('18.1',3),ev('18.3',30),ev('18.4',9),ev('18.5',3),ev('18.5',31),ev('18.6',3),ev('18.6',27),ev('18.8',24),ev('18.8',26),ev('18.9',3),ev('18.9',29),ev('18.10',5),ev('18.13',5),ev('18.13',30),ev('18.14',5),ev('18.14',31)],
      mechanism_status='已足夠',need_status='已足夠',
      shortest_inference='同hard完整目标及训练条件无新增监督；soft目标提供非首偏好并改变q-p；次候选CE份量小→升T显示关系且不改排名；candidate身份/同前文/同答案预测位必对齐，labels仅mask KL、T²一次；错误teacher仍保CE和独立truth，teacher成本单列，四组隔离结构/讯号/精度。',
      strongest_counterexample='ID相同不保候选身份，shape相同不保同题；modal答案输入位置16/4与预测行15/3不同；教师一致率100%可任务1/3；同架构两学生无推论差不由多请求回收教师准备成本。原文均具体示出，未用更像教师推质量通过。')
]

cross_prereqs=[]
for page,purpose in [('2.4','核ReLU是否真的全書首次缺名，并查其非线性角色'),('2.5','核MLP完整双语名和角色'),('5.9','核FP32格式定义是否更早已教'),('7.1','核SFT完整双语名及继续已有模型的意义'),('5.8','核teacher forcing英文名及中文动作是否更早已教')]:
    cross_prereqs.append(dict(page_id=page,source_sha256=sha(S/(page+'.md')),actual_read_scope='本交叉輪完整main，跳过details；只为名称/角色先备，未称该页图或实测本轮已验证。',purpose=purpose))

counts={s:sum(x['final_severity']==s for x in issues) for s in ['burden','optional','none']}
dispositions={s:sum(x['disposition']==s for x in issues) for s in ['採用','延後','保留原文','待查']}
report=dict(
 schema_version=1, kind='sealed-source-initials-cross-review',group='architecture',reviewer='/root/continuity_architecture',
 created_at=datetime.now(timezone.utc).isoformat(), phase='互見後交叉；不是新增独立盲初判',
 instructions=dict(path=str(B/'cross-review-instructions.md'),sha256=sha(B/'cross-review-instructions.md'),actually_read='完整'),
 criteria_sha256=initial['criteria_sha256'],
 own_seals_reconfirmed=dict(initial_sha256=own_seal['initial_sha256'],main_notes_sha256=main_seal['sha256'],byte_matches=True,original_records_modified=False),
 read_reports=[dict(path=str(R/n),sha256=sha(R/n),
                    actual_read_scope=('全部20 issues；71页specific_basis逐页；18个origin所在页checkpoint的材料/预期/疑问/图需求；未重抄其他无争议checkpoint字段' if n=='reader-architecture-main.json' else
                                       '全部20 issues与main逐项相同核；全部71页optional_read assessment与optional_effects新问题/待修/补充理解；main seal核' if n=='reader-architecture.json' else
                                       '全部1 issue、71页source_judgment/quoted_basis/checks/unverified逐页，实际先备/repair/isolation与证据范围；未直接执行或全部重新读取引用JSON/implementation' if n=='technical-architecture-initial.json' else
                                       '本角色完整独立阅读并写封存的71页记录；本轮重查issue、覆盖、unknown、6组strongest passes与相关source')) for n in reports],
 same_group_independent_supplements=dict(found=[],note='reports目录匹配architecture独立supplement/cross文件没有既存补件；未读历史review或人类checks。'),
 cross_actual_prerequisites=cross_prereqs,
 source_read_scope='独立轮已实际全文读architecture71 main、封存後71 extras及记录的必要先备。本轮按每个origin回查相应main/extras和六组strongest pass原关系，不声称重新全文读71页。全书词名只在freeze/sources检索，匹配需用者再读main；未来页18.6只作为后续命名记录，未倒填17.3首处。',
 issues=issues, strong_pass_checks=passes,
 visual_scope=dict(independent_report=str(R/'continuity-architecture-initial.json'),
                   independent_source_image_scope=initial['visual_scope'],
                   cross_new_images_viewed=0,
                   note='所有候选是文字名称/need关系，未新提出图形/版面争议。本轮未把file/DOM当视觉证据；沿用本人独立轮已实看的25张640与14张360、列明原位截图。source3幅必要对应（15.6/16.4/18.6）和14.1截图已本人亲看后记录，不因同行的图一致结论扩大到全站。'),
 unknowns=[
   dict(scope='该局部修补对真人初学者的收益',status='未验证',effect='AI来源裁定不是学生试读，不能称已测出改善。'),
   dict(scope='每一原始JSON、private checkpoint/data、GPU速度峰值、跨seed/任务推广',status='未独立重验',effect='技术轮报告的限定技术核有其证据范围；本交叉只实际重读YaRN选定原文，不把同行数值验证变成本人独立实验，未知不升缺陷亦不无条件通行。'),
   dict(scope='整个71页新网站互动、runtime outputs/reading annotations、所有图360',status='未全验',effect='本人独立截图范围有限，新preview省略两类内容；不用纯PNG证明折叠可操作或完整阅读路径。'),
   dict(scope='top-2实际选定的设计目的',status='来源尚缺；修补表述需据实选择',effect='可用示范目的或真实合并需求补最小关系，不能从候选修补反写作者当初动机。'),
 ],
 reconciliation=dict(unique_origin_ids=22,necessary_origin_ids=4,optional_origin_ids=18,
                     deduplicated_relationships=20,final_necessary=counts['burden'],final_blocker=0,
                     final_optional=counts['optional'],closed_without_requested_change=counts['none'],
                     disposition_counts=dispositions,
                     counting_note='reader-main/final同origin不再计两项。14.10三origin合一；top2单origin独立成一。可选origin全部18逐项保留；4个全書已教名、15.6即时双语动作、18.8已教teacher forcing角色等6项不立新缺口。余12是可保留的词句/表述改善机会，其中含9采用、2保留、1延后，均非必要。',
                     final_required_scales=dict(paragraph=2,page=0,chapter=0,order=0),
                     optional_adopted_scales=dict(paragraph=sum(x['final_severity']=='optional' and x['disposition']=='採用' for x in issues),page=0,chapter=0,order=0),
                     rewrite_conclusion='两处必要段落关系修补；现有完整来源与strongest passes不支持整页、章或顺序大改。没有修改教材。'),
 independence=dict(cross_review_visible=True,old_historical_reviews_read=False,initial_reports_modified=False,
                   textbook_modified=False,training_run=False,subagents_spawned=False,new_independent_discovery_claimed=False),
)

out=R/'cross-architecture.json'
out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
seal=R/'cross-architecture-seal.json'
seal.write_text(json.dumps(dict(schema_version=1,group='architecture',reviewer='/root/continuity_architecture',
                                report_path=str(out),report_sha256=sha(out),sealed_at=datetime.now(timezone.utc).isoformat(),
                                source_initial_sha256=own_seal['initial_sha256'],main_notes_sha256=main_seal['sha256'],
                                unique_origin_ids=22,optional_origins_adjudicated=18,necessary_relationships=2,
                                meaning='互見後交叉报告独立字节封存；原initial/mainnotes未改，不能算第二份独立发现。'),ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(report=str(out),sha256=sha(out),counts=counts,dispositions=dispositions,
                      necessary_scales=report['reconciliation']['final_required_scales'],optional_adopted=report['reconciliation']['optional_adopted_scales']),ensure_ascii=False))
