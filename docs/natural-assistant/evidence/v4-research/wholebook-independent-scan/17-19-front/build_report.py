from pathlib import Path
import hashlib
import json
import re

ROOT = Path('/workspace/tiny-perceptron-vlm')
OUT = ROOT / 'outputs/natural-v4/wholebook-review/17-19-front'
manifest = json.loads((OUT / 'source-manifest.json').read_text())

understanding = {
    'R.1': '先用編號還原一小句理解模型資料，不要求初讀者安裝；Python與操作都有可回查暖身。',
    'R.2': '用問題選路線並明列最短前置；第19章整合小世界與第20章成熟底座是不同目的地，20的實際銜接仍待定稿核對。',
    'R.3': '章表交代從接字到架構、加速、壓縮、整合以及應用的每章問題，技術名稱沒有取代學習問題。',
    'R.4': '小程式只證明一個機制，完整訓練另驗新題；但檔案份數、兩批模型沿革與交付細節把這條核心線索拉得太長。',
    'W.1': '安裝、kernel與依賴cell的重跑順序能連成第一次操作，Windows啟用失敗也提供直接Python入口。',
    'W.2': '動物清單與字典讓索引、键值、迴圈、assert、函式有具體意思；讀者能先預測再改輸入核對。',
    'W.3': '成績兩軸與mean的消去軸對照[B,T,D]，圖中平均方向與實際數值吻合。',
    'W.4': '用正確答案機率解釋負自然log代價、bit換單位與可加性；round偶數邊界説明在這個暖身中分散注意。',
    'W.5': '飲料配方逐項加權後擴成矩陣乘法，再用Linear的權重轉置與bias接回模型。',
    'W.6': '平方代價的導數先指出局部方向，更新才改w；較大學習率跨過目標甚至令代價上升。',
    'W.7': '相對路徑、cwd與常見錯誤各有排查方法，dry-run報告也不被當成已有訓練。',
    'G.1': '字元ID只是位置，embedding才是查回的可學向量，token大小取決於切分方式。',
    'G.2': 'logits、機率、loss、更新與checkpoint各有角色，名詞表提醒loss不是所有能力的總品質。',
    'G.3': 'Q和K比較出讀取比例後混合V；能看見前文與哪些位置計loss是不同遮罩工作。',
    'G.4': '後段方法先映到儲存、速度、偏好或外部資訊問題；PPO選卡的教材範圍清楚，但新增字樣可刪。',
    'T.1': '先定任務真值與完整家族，再取資料；真實小資料來源、作者切分與重採樣不冒稱官方測驗。',
    'T.2': '一次前向反向核對有限loss和非零梯度只證明數值通路，vision的null目標統計不是沒有目標。',
    'T.3': 'bigram與不同窗口MLP在同一整篇切分上比較前後NLL，5字訓練最低却新題較差。',
    'T.4': '文字接續和回答遮罩的訓練目標不同，先保存同一起點再做同份validation比較；太多正式支線插入主操作流程。',
    'T.5': '格式、內容、風格、拒絕與澄清各自判分；LoRA的風格改變沒有修復算術，但多種支線令操作入口難找。',
    'T.6': '先確認編碼器小任務，再接頭並檢查原文字能力與換圖；舊格式、私有備份、無續訓入口和影片待擴充可移離正文。',
    'T.7': 'DPO的候選相對排序不等於自由生成正確；選卡PPO、生成成品DPO及蒸餾學生是分開的選讀路線。',
    'T.8': '架構比較固定資料與預算，但不假裝等參數或等FLOPs；注意力後端、精度與Flash探針也各有量測範圍。',
    'T.9': '同一浮點來源真正打包4/8-bit後重載量新題，tensor bytes、檔案、RAM與速度不可互換。',
    'T.10': '同架構CE、教師硬回答與CE+KL對照可隔離教師訊號，學生架構縮小及再量化另有品質損失。',
    'T.11': '先定固定資料、分母、判準和前後比較再填結果；假想表清楚標明，但三項應用報告與續訓細節可改連結。',
    '17.1': '參數個數乘型別bytes只算數字，低位元權重、保留FP32與刻度相加才得到整模型payload。',
    '17.2': '除scale、最近整數、限端點、乘scale構成近似量化，半格界線只適用未截斷值。',
    '17.3': '輸出誤差是輸入乘权重差，可能相加或抵消；同一權重MAE不能直接當所有輸出誤差。',
    '17.4': 'scale決定間距、zero-point決定浮點零所在碼，unsigned容器尚不表示已完成4-bit打包。',
    '17.5': '逐列scale避免大列拖累小列精度，但較多scale增加附加儲存；per-channel方向交代清楚。',
    '17.6': '4-bit格子更粗卻非每項都較差，理想17 bytes與實際int8容器33 bytes分開核對。',
    '17.7': 'weight-only保存打包權重與scale，參考forward還原FP32，buffer與Parameter不混算。',
    '17.8': '加8編碼與高低四位的順序讓七項無損放進四bytes，打包無損不等於浮點量化無損。',
    '17.9': 'PTQ從同一已訓練權重複製轉換並重測，隨機模型短例只驗轉換，tied表須另外交代。',
    '17.10': '較小檔案仍可能因拆碼反量化較慢；同步時間、分配峰值及其基線界線明確。',
    '17.11': 'activation是中間特徵而非啟動函數，同時改權重和特徵會新增但未必逐項放大的誤差。',
    '17.12': '固定校準與動態範圍在不截斷極值與小值精度間取捨，最終test不拿來挑range。',
    '17.13': '截斷100到1改善小值刻度卻丟99極值，誤差必須對原值而非截斷值計算。',
    '17.14': 'fake quant前向用還原值、STE反向用近似恒等，公平QAT與同350更新後PTQ比較仍失敗；內部續訓配方可移走。',
    '17.15': '小分數MAE也能跨越argmax/EOS邊界，須配對比較原始生成和任務，不能只報同總分。',
    '18.1': '教師文字與完整分布包含不同訊號，黑盒與白盒是可見資訊類型，教師仍可能錯。',
    '18.2': '學生變小由寬度與層數決定，字表等不隨層數縮小，所以完整參數需實數。',
    '18.3': '黑盒教師回答重新由學生切分做SFT，先驗內容與停止原因；重複正確題也未增加多樣性。',
    '18.4': '硬目標與教師比例的CE有不同梯度，教師比例保留次要關係也會推高錯誤候選。',
    '18.5': '正溫度在softmax前縮分數差而不換排名，教師與學生共用T、教師項T²僅算一次。',
    '18.6': '對齊的是token身份和相同前文位置，相同詞表大小不是語義對齊保證。',
    '18.7': 'eval、權重凍結、no_grad與學生optimizer是不同控制；教師固定仍要付forward和快取成本。',
    '18.8': 'forward KL以教師p加權，PyTorch先傳學生log機率；有效回答行平均令梯度分母可手算。',
    '18.9': 'CE真值與已含T²的教師KL按alpha混合，不同訓練目標的loss不能當相同品質排名。',
    '18.10': '同尺寸同初始化學生在共同真值比較，教師一致率與教師成本都不能藏起來；本輪蒸餾未提高屬性答對數。',
    '18.11': '教師風格與錯誤需獨立判內容，合法JSON和EOS沒有讓算術答對，低NLL也不等於自由生成正確。',
    '18.12': 'MoE和Dense内部可不同，輸出分布介面一致就能蒸餾；同32篇學生比較隔離訊號，與409篇教師差距另含資料預算。',
    '18.13': '不同模態前文長度須取相同答案的前一位置logits；正式KL學生遮圖前後ID不變暴露本輪視覺未使用限制。',
    '18.14': '教師、CE學生、KL學生與同KL量化版四個端點分別分離架構、訊號和量化，payload縮小伴隨blue變ble。',
    '19.1': '一個模型需多題和完整工具迴圈才看得出能力，首段完整動作吻合與整題正確有不同條件。',
    '19.2': 'MoE四組expert都要存，active是结构代理；本實作成本較Dense高而未聲稱等品質。',
    '19.3': '同數字、圖片組合、頻率家族整組切分，常數基線顯示原圖全答green也可高分。',
    '19.4': '父權重指紋證明同核心跨階段接續，預訓練與SFT學習位置不同，DPO是從joint出的比較支線。',
    '19.5': '短答、缺數量、工具不可用與窄拒絕各有示範，固定模板通過只支持這些指定行為。',
    '19.6': 'RGB與log-mel摘要真進同核心，換色/換聲成對才證明依素材回答；同節仍有等待實跑的過時語氣。',
    '19.7': '請求解析、真工具執行與第二次模型回答分別核對，工具算1卻答0或111是實際讀回失敗。',
    '19.8': '後訓練方法是可選分支，混合DPO+CE+平衡項沒改善短答且傷聯合題，所以依validation選joint。',
    '19.9': '快取先驗同歷史logits和原始ID，再量固定短題同步時間；SDPA名稱不保證Flash。',
    '19.10': '壓縮joint與較小Dense學生是不同選項，同分仍會變生成；學生KD公式應把已說明的T²顯示出來。',
    '19.11': '公開推論包不含續訓所需工作狀態，跨站與resume讀不同資訊，CLI逐站完成才繼續。',
    '19.12': '同考卷逐項能力表保留形狀和讀回失敗，定版選擇不回看test；轉第20章改用成熟底座的連貫性待新章完成。',
}

issues = [
    {
        'id': 'F01', 'priority': 'P2', 'sections': ['R.4'],
        'anchor_quote': '成品與學生共11份推論檔已公開',
        'problem': '首次閱讀的機制／訓練／譬喻說明混入11份、30組、120份、兩批倉庫和推論檔用途等交付清點，對理解第一節沒有增益。',
        'action': 'R.4保留小實驗與完整訓練界線、19與20兩條路線及一個譬喻邊界例子；將模型數量、倉庫批次和存檔格式移到T.3或19.11，以一句試用連結取代。不要刪掉小世界與自然輸入的能力界線。',
        'acceptance': '初读者能在短段落中指出如何讀小例子、如何做訓練、何時去19/20，不須理解下載批次。',
    },
    {
        'id': 'F02', 'priority': 'P2', 'sections': ['T.4', 'T.5', 'T.6', 'T.8', 'T.10', 'T.11'],
        'anchor_quote': '上面的200步是用小型CPU模型練流程的起點',
        'problem': '主操作尚未完成就插入另一種尺寸、GPU實驗、多條支線和另一組欄名；T.4一節220行，讀者易失去自己該執行的下一步。',
        'action': '每節主體固定為前置→準備→同起點前評估→訓練→同題後評估→解讀。正式GPU結果與重跑命令各放在末尾明確的延伸區或連結；T.8把modern、MoE、efficiency、Flash、precision的完整各支檔名清單移到相應實驗操作文，T.11把RAG/Tools/Reasoning詳細結果改成三個示例連結。保留一個完整分母與失敗對照，不刪科學限制。',
        'acceptance': '只選一條CPU路線的讀者能一路找到匹配檔案與指令，不須在兩份不同設定和報告欄名間反覆切換。',
    },
    {
        'id': 'F03', 'priority': 'P2', 'sections': ['T.6'],
        'anchor_quote': '正式checkpoint與中間更新檔已備份到私有HF固定版本',
        'problem': '私有備份狀態、Actions/映像/HF傳輸的時間排除、舊.modal格式沿革、正式編碼器CUDA續訓缺項和影片待擴充並未幫完成當前接頭練習。',
        'action': '刪私有備份與影片待擴充兩段；將舊格式兼容和各正式產物續訓能力放到格式文件。正文只說本命令產生的原生檔用途與--resume要求；測量範圍在正式報告保留一次，不在操作途中展開備份工程。',
        'acceptance': '學生清楚使用本節新產物的推論與續訓入口，無須認識作者如何備份舊產物。',
    },
    {
        'id': 'F04', 'priority': 'P2', 'sections': ['17.2', '17.3', 'W.4'],
        'anchor_quote': '實際列印可能看到 `0.8999999761581421`',
        'problem': '相鄰量化節重複說明tensor.round、tolist及Python round的長尾與顯示差別；暖身又先插入ties-to-even細節，核心scale及誤差傳遞被打斷。',
        'action': 'W.4只解釋round縮短顯示，偶數取整放在17.2第一次相關處；17.2保留一次浮點尾數短註，17.3用近似符號和回鏈取代兩段輸出格式教學。若需要精確短顯示，統一示範一個易讀輸出方法。',
        'acceptance': '正文仍能解釋實際浮點尾數，但17.3主要篇幅用於權重誤差乘輸入、相加或抵消。',
    },
    {
        'id': 'F05', 'priority': 'P2', 'sections': ['17.14'],
        'anchor_quote': '需用`scripts/course_experiments/compression.py`的`_qat_layers(model)`重新包住各線性層',
        'problem': 'STE和部署規則已完整說明後，教材突然要求內部私有helper、wrapper重建、optimizer載回與沒有自動續訓入口，初學者容易把這些當本節必要任務。',
        'action': '保留「浮點主權重檔不含fake-quant行為；本命令從共同底座重做」兩句，內部wrapper續訓建議移到開發文件。結果表保留同350次更新對照和失敗，不需在這節詳教未提供CLI的流程。',
        'acceptance': '讀者能解釋forward還原值及STE梯度，並辨識推論檔，不被導入未支援的續訓實作。',
    },
    {
        'id': 'F06', 'priority': 'P2', 'sections': ['19.4', '19.5', '19.7', '19.10'],
        'anchor_quote': '指紋以`c35427bc`開頭、以`3e2aaba7`結尾',
        'problem': '每站重報檔案bytes、指紋頭尾、GPU循環/整個實驗秒數；19.5及19.7又逐站重報同一份文字成績，重複驗證記帳多於新機制。',
        'action': '19.4用一張階段表列父階段、學習目標、更新數、驗證文字/模態成績，正文只示範一次父子完整SHA核對，其他bytes和計時連到實報。19.5/19.7各保留具體示範與末端失敗，joint/DPO不變的重複成績改回鏈同能力矩陣。19.10的交付檔容器差異用一次原則說明，不需每支再次展開。',
        'acceptance': '讀者能追到同一模型如何成長以及DPO为何不推薦，並可查完整來源，但不用讀指紋尾碼来理解每站。',
    },
    {
        'id': 'F07', 'priority': 'P2', 'sections': ['19.6'],
        'anchor_quote': '必須等換素材的對照實跑，再討論對圖片特徵的依賴',
        'problem': '本節開頭已說部署真的換素材，後半也列出完成的9/9與18/18對，前半卻仍用等待實跑的語氣；完成狀態不一致。',
        'action': '改成「這組驗證顏色是常數，圖片依賴要看下方已完成的換素材成對結果。」把「先看目前真正取得」等進度語氣改成直接引導兩類證據的句子。保持形狀0/9對的失敗。',
        'acceptance': '同一節明確區分固定驗證與已完成反事實對照，讀者不以為重要對照仍未跑。',
    },
    {
        'id': 'F08', 'priority': 'P2', 'sections': ['19.10'],
        'anchor_quote': '配方為`0.5 CE + 0.5 KL(teacher||student)`、溫度2並保留溫度平方縮放',
        'problem': '文字說保留T²而顯示公式省略T²，與18.9明寫公式的記號不同；學生自行抄公式時容易少乘或在helper外重乘。實作本身已正確只乘一次。',
        'action': '統一寫`0.5 CE + 0.5 T² KL(p_T||q_T)`，並用一句註明distillation_loss內部已含T²、呼叫處不再相乘；若用K記號，先定義K已含T²再顯示0.5CE+0.5K。',
        'acceptance': '公式、溫度定義與helper契約一致；T=2教師項確為一次乘4，沒有少乘或重乘。',
    },
    {
        'id': 'F09', 'priority': 'P3', 'sections': ['G.4'],
        'anchor_quote': '第13章新增的PPO實驗',
        'problem': '新增是版本沿革詞，對今天第一次讀到教材的人沒有資訊。',
        'action': '刪「新增的」，保留有限選卡、程式規則標籤與非真人RLHF的實驗範圍。',
        'acceptance': '詞條描述方法與當前範圍，不要求知道教材歷史。',
    },
]

chunks = [
    ('course/README.md', 1, 96, 'a48a9e'),
    ('course/first-steps.md', 1, 128, 'd68e83'),
    ('course/first-steps.md', 129, 221, '4a7963'),
    ('course/glossary.md', 1, 87, '4b9f86'),
    ('course/training.md', 1, 74, '4b9f86'),
    ('course/training.md', 75, 218, '39e759'),
    ('course/training.md', 219, 325, '2ac501'),
    ('course/training.md', 326, 438, '4b909f'),
    ('course/training.md', 439, 529, '39809c'),
    ('course/training.md', 530, 629, 'f044b1'),
    ('course/training.md', 630, 784, 'bf70b7'),
    ('course/training.md', 785, 911, '3e6059'),
    ('course/chapters/17.md', 1, 197, '4b142c'),
    ('course/chapters/17.md', 198, 342, 'b80191'),
    ('course/chapters/17.md', 343, 512, '36af1d'),
    ('course/chapters/17.md', 513, 543, 'b37cd1'),
    ('course/chapters/18.md', 1, 161, 'b37cd1'),
    ('course/chapters/18.md', 162, 304, 'accdbe'),
    ('course/chapters/18.md', 305, 454, '64e0d1'),
    ('course/chapters/18.md', 455, 568, 'f2830a'),
    ('course/chapters/19.md', 1, 166, '719264'),
    ('course/chapters/19.md', 167, 346, 'c52713'),
    ('course/chapters/19.md', 347, 552, '18c38c'),
    ('course/chapters/19.md', 553, 716, 'c188ea'),
]

authority_specs = [
    ('hinton-distillation', ['18.5', '18.8', '18.9'], 120, 178,
     '高溫教師/學生softmax與硬軟目標混合，T²調整相對梯度尺度；書中是forward KL，固定教師時與soft CE相差教師熵。'),
    ('jacob-qat', ['17.14'], 254, 314,
     '訓練保留浮點權重并在forward模擬捨入，推論圖另建；教材沒有宣稱複製整數引擎。'),
    ('bengio-ste', ['17.14'], 400, 414,
     'section 4把hard threshold按identity反傳並明說biased；教材將round的恒等反傳清楚標為近似。'),
    ('gptq', ['17.10'], 530, 542,
     '量化矩陣/完整精度向量kernel需要時動態反量化，收益主要減少memory access，未要求activation量化。'),
    ('minillm', ['18.1', '18.8'], 68, 90,
     '黑盒只有教師文字，白盒可取分布或隱藏狀態；該論文reverse KL不等同課程forward KL。'),
    ('mixtral', ['19.2'], 170, 186,
     'section 3明說服務記憶體按47B總參數計，active compute比較未含memory及routing/多expert成本。'),
    ('switch', ['19.2'], 335, 371,
     'section 2.2輔助負載損失在訓練加入，probability項可微而dispatch比例不可微，係數0.01有原文依據。'),
]
fetches = json.loads((OUT / 'authority/fetch.json').read_text())
authorities = []
for name, ids, start, end, finding in authority_specs:
    p = OUT / f'authority/{name}.txt'
    excerpt = ''.join(p.read_text().splitlines(keepends=True)[start-1:end])
    e = OUT / f'authority/{name}.excerpt.txt'
    e.write_text(excerpt)
    src = next(row for row in fetches if row['name'] == name)
    authorities.append({**src, 'sections': ids, 'text_read_lines': [start, end],
                        'excerpt_artifact': str(e.relative_to(ROOT)),
                        'excerpt_sha256': hashlib.sha256(excerpt.encode()).hexdigest(),
                        'finding': finding, 'result': 'supports course claim within stated scope'})

torch_doc = OUT / 'authority/torch-kl-2.14.txt'
doc = torch_doc.read_text()
idx = doc.find('Compute the KL Divergence loss.')
excerpt = doc[idx:idx+2500]
(OUT / 'authority/torch-kl-2.14.excerpt.txt').write_text(excerpt)
authorities.append({**json.loads((OUT / 'authority/torch-kl-follow-redirect.json').read_text()),
                    'name': 'torch-kl-final-doc', 'sections': ['18.8'],
                    'excerpt_artifact': str((OUT / 'authority/torch-kl-2.14.excerpt.txt').relative_to(ROOT)),
                    'excerpt_sha256': hashlib.sha256(excerpt.encode()).hexdigest(),
                    'result': 'supports student log-probability first and teacher probability target; helper sums vocabulary then averages valid rows',
                    'note': 'stable URL first returned a 512-byte redirect page; followed explicit official 2.14 destination and read its actual documentation.'})

figure_notes = {
    'foundations_tensor_axes.svg': '兩列三欄、橫向dim=1和直向dim=0平均數正確，未把軸名固定成所有tensor的意義。',
    'architecture_quantize_grid.svg': '0.7和1.2除0.5、取整、乘回的箭頭與差0.2吻合。',
    'architecture_int4_packing.svg': '首碼在低四位且綠虛線、第二碼在高四位且橘實線，112與152數字正確。',
    'qat_training_deployment.svg': 'FP32主權重、每次fake round/STE與最後packed4分開，部署仍FP32明寫。',
    'architecture_distillation_temperature.svg': 'T=1/2/4比例、色序與排序保持和正文softmax一致，圖的四捨五入不被當精確和。',
    'architecture_vocab_alignment.svg': '交叉連線按貓/狗身份而非相同ID對齊，重排[1,0]正確。',
    'architecture_modal_answer_alignment.svg': '教師query15/16/17對學生3/4/5預測同A0/A1/A2，沒有錯從答案輸入列取logits。',
    'capstone_resources.svg': '共享規則與四套expert儲存明示，啟用1與3不令其他權重消失。',
    'capstone_pipeline.svg': 'A→B→C主線，joint C標推薦，DPO D用虛線標偏好比較支線，與最後選擇一致。',
    'capstone_tool_loop.svg': '兩次模型生成之間真執行一次calculator，許可與格式檢查在執行前，不能以程式3冒充最後回答。',
}

for f in manifest['files']:
    for section in f['sections']:
        section['understanding'] = understanding[section['id']]
        section['issue_ids'] = [i['id'] for i in issues if section['id'] in i['sections']]
        section['source_snapshot'] = str((OUT / 'sources' / f['path']).relative_to(ROOT))
    f['source_snapshot'] = str((OUT / 'sources' / f['path']).relative_to(ROOT))
    ranges = [(a,b) for p,a,b,_ in chunks if p == f['path']]
    expected = 1
    for a,b in ranges:
        assert a == expected
        expected = b+1
    assert expected == f['lines']+1
for f in manifest['figures']:
    f['source_snapshot'] = str((OUT / 'sources' / f['path']).relative_to(ROOT))
    f['render_artifact'] = str((OUT / 'figures' / (Path(f['path']).stem+'.png')).relative_to(ROOT))
    f['inspection'] = figure_notes[Path(f['path']).name]
    f['visually_viewed'] = True
    source_text = (OUT / 'sources' / f['referenced_by']).read_text()
    for match in re.finditer(r'!\[([^\]]*)\]\(([^)]+)\)', source_text):
        resolved = (ROOT / Path(f['referenced_by']).parent / match.group(2)).resolve()
        if str(resolved.relative_to(ROOT)) == f['path']:
            before = source_text[:match.start()]
            f['section_id'] = re.findall(r'^## (\S+)', before, re.M)[-1]
            f['reference_line'] = len(before.splitlines()) + 1

assert len(understanding) == 67
assert set(understanding) == {s['id'] for f in manifest['files'] for s in f['sections']}

raw_qat = ROOT / 'docs/course-experiments/results/qat.json'
qat = json.loads(raw_qat.read_text())['results']['runs']
primary_metrics = {
    'source': str(raw_qat.relative_to(ROOT)),
    'sha256': hashlib.sha256(raw_qat.read_bytes()).hexdigest(),
    'scope': 'read existing primary experiment report; not a new GPU training run',
    'matched_ptq4_validation_nll': qat['matched_ptq4']['validation']['answer_nll'],
    'qat_packed4_validation_nll': qat['qat_packed4']['validation']['answer_nll'],
    'matched_ptq4_test_nll': qat['matched_ptq4']['test']['answer_nll'],
    'qat_packed4_test_nll': qat['qat_packed4']['test']['answer_nll'],
    'finding': '17.14 validation order is QAT 0.5273 versus PTQ 1.1811, and test reverses this benefit; no number error found.'
}
(OUT / 'primary-metrics.json').write_text(json.dumps(primary_metrics,ensure_ascii=False,indent=2)+'\n')

additional_sources = []
for path, scope in [
    ('tiny_perceptron/alignment.py', 'full source read; KL direction, reduction and once-only T²'),
    ('tiny_perceptron/quantization.py', 'full source read; scale, packing, buffers and STE'),
    ('scripts/course_experiments/capstone_student.py', 'lines 88–116 read; actual student calls distillation_loss alpha=.5 temperature=2'),
]:
    b=(ROOT/path).read_bytes()
    additional_sources.append({'path': path, 'sha256': hashlib.sha256(b).hexdigest(), 'bytes': len(b), 'read_scope': scope})

inspection = {
    'schema_version': 'wholebook-review-v4-slice-1',
    'task_identity': {
        'agent_task': '/root/v4_wholebook_scan_17_19_front',
        'requested_scope': 'fresh whole-book pre-inspection slice: front matter and chapters 17–19',
        'target_reader': '數學不錯的高中生或有基本數學的大學生，初次接觸LLM；允許明確前置回鏈',
        'review_role': 'independent textual, conceptual, numerical and figure review',
        'not_a_claim': 'not a final whole-book verdict; chapter 20 is still in training/writing',
        'skill_applied': 'cloud-environment-onboarding:setup; use existing checkout/.venv and preserve protected files; no configuration changes needed',
        'prohibited_context_not_read': ['old review verdicts', 'author development history as missing textual background', 'chapter 20 drafts'],
        'mutations': 'Only this task output directory created; no canonical book, checker, reviews or reading times edited; no child agents spawned.',
    },
    'read_order': [f['path'] for f in manifest['files']],
    'actual_text_read_chunks': [{'path':p,'start_line':a,'end_line':b,'tool_chunk_id':cid,'output_truncated':False} for p,a,b,cid in chunks],
    'coverage': {'files':7,'sections':67,'lines':sum(f['lines'] for f in manifest['files']),
                 'figures_referenced_and_visually_inspected':10,'full_text_read':True,
                 'reading_method':'Actual complete tool outputs read in listed chunks before writing verdict; hashes are byte identity evidence, not substitutes for reading.'},
    'files': manifest['files'], 'figures': manifest['figures'],
    'issues': issues, 'authority_checks': authorities,
    'local_primary_evidence': primary_metrics, 'additional_source_checks': additional_sources,
    'numerical_validation': {
        'interpreter': '.venv/bin/python', 'device': 'cpu',
        'original_section_snippets': 31, 'passed': 31, 'failed':0,
        'execution_artifact': str((OUT/'numeric/execution.json').relative_to(ROOT)),
        'independent_checks':4,
        'independent_artifact':str((OUT/'numeric/independent-checks.json').relative_to(ROOT)),
        'observed_values':'MAE and scales, actual int4 bytes, STE and KL gradients, T softmax, model payloads, family splits, labels and tool runtime matched stated examples.'
        , 'actual_tool_chunk_ids':['383c37','8b9310','db4a00']
    },
    'figure_validation': {'renderer':'inkscape --export-type=png --export-width=1200',
                          'render_tool_chunk_id':'f71427',
                          'inspection_tool':'view_image; all 10 individual renders returned and visually read',
                          'rendering_failures':0},
    'not_run': [
        'Full notebook execution (root planned separately).',
        'All proposed code variations/exercises; only two gradient exercises and independent KL/score sums checked here.',
        'Long CLI training/GPU experiment reproduction, public HF downloads, trained capstone inference or UI.',
        'Independent audit of every historical raw experiment table beyond the QAT NLL ambiguity checked here.',
        'Chapter 20 draft/current new prose, its actual final training result and endpoint cross-links.'
    ],
    'limitations': [
        'This slice follows the stated front→17→18→19 order; prerequisite chapters were used through explicit descriptions/links and are independently assigned elsewhere, not fully reread here.',
        '31 passing numeric snippets validate local mechanisms, not trained model quality or full notebooks.',
        'Primary PDFs were downloaded and relevant cited passages read; entire papers were not read end-to-end.',
        'Sources were snapshotted after actual chunked reading; root must compare final canonical bytes against these saved hashes before applying the findings.',
        'Visual figures were inspected at 1200px rendering, not every mobile/browser typography scenario.',
        'No final whole-book approval is given while new chapter 20 and cross-chapter endpoint remain pending.'
    ],
    'pending_endpoint': {
        'status':'pending', 'locations':['R.2','R.3','R.4','training introduction','19.12 ending'],
        'observed_intended_destination':'19 teaches a from-scratch synthetic small-world MoE; 20 starts from a mature graph/vision language base, tests natural photos/Chinese OCR, and transcribes speech into the shared chat model.',
        'required_final_check':'When chapter 20 is complete, reread these front/ending paragraphs with its actual training route, chosen base/adapter, resource choice, ability card and numbered anchors; do not silently mark these links complete now.'
    },
}
(OUT/'inspection.json').write_text(json.dumps(inspection,ensure_ascii=False,indent=2)+'\n')

lines=[
    '# 前言與第17–19章：fresh 全書預檢分段報告',
    '',
    '任務身分：`/root/v4_wholebook_scan_17_19_front`。目標讀者是數學不錯的高中生或有基本數學的大學生，初次學LLM，允許明確前置。按 README → first-steps → glossary → training → 17 → 18 → 19 完整讀完7檔、67節、3142行；不是用搜尋抽樣或hash代替閱讀。未讀舊review verdict、作者沿革補背景或第20章草稿，未改正文、checkers、reviews、timings，也未另派代理。',
    '',
    '本分段未發現P0/P1的實質數學錯誤。刻度/打包、STE、KL方向、T²、模態答案位置與能力分母的核心線索自洽；10圖全看，31節CPU原文短程式成功、4項獨立核對通過。需要實際處理的是多處反覆工程記帳、操作頁支線插入，以及完成狀態和公式顯示的一致性。以下是可執行建議，並未代替根任務的全Notebook執行或第20章定稿核對。',
    '', '# 按優先級修訂', ''
]
for issue in issues:
    lines += [f"## {issue['id']} · {issue['priority']} · {', '.join(issue['sections'])}", '',
              f"定位：{issue['anchor_quote']}", '', issue['problem'], '', issue['action'], '',
              '核對完成條件：'+issue['acceptance'], '']
lines += ['# 每節自己的理解／問題', '', '| 小節 | 閱讀理解 |', '| --- | --- |']
for f in manifest['files']:
    for s in f['sections']:
        lines.append(f"| {s['id']} | {s['understanding']} |")
lines += ['', '# 實際證據與其界線', '',
          '原文分段讀取範圍與實際工具chunk ID在 `inspection.json.actual_text_read_chunks`，全部連續無缺頁。`sources/`保存7份原文及10份SVG；`inspection.json`逐檔、逐節（標題至下節前）、逐圖保存完整SHA-256、bytes、行號與來源，不只保存摘要。圖渲染於 `figures/`，已各自以view_image看過。', '',
          '`numeric/execution.json`及每節 `.py`、stdout、stderr 保存31節真實CPU執行；暖身矩陣、量化MAE、打包112/152/175/139、STE 1與round 0、KL 0.1927及有效位置梯度、capstone 552/84/90切分等與正文吻合。`numeric/independent-checks.json`另核對KL=CE−H、平方STE梯度、三有效位置梯度和19.12任務合計78。這些是小机制檢查，不是GPU訓練與成品能力重測。', '',
          '下載7份原論文實際PDF，保存HTTP取得結果與完整SHA，閱讀引用相關段落並摘錄於 `authority/*.excerpt.txt`。Hinton §2支持共同高溫與T²；Jacob §3支持浮點訓練/模擬量化；Bengio §4明示STE有偏；GPTQ實用加速段落支持需要時反量化與較少讀取；MiniLLM導論支持黑/白盒區別但其reverse KL不同；Mixtral §3支持按總參數儲存；Switch §2.2支持負載輔助項。PyTorch stable頁先回HTML轉址，已跟至正式2.14文檔讀log-probability input／probability target及reduction說明，沒有把轉址512 bytes當查證成功。', '',
          '讀取既有QAT主實報確認17.14的驗證0.5273為QAT、1.1811為同預算PTQ，最後題0.5979與0.4066反轉；`primary-metrics.json`保存原報告SHA與選取數值。19.10實際學生呼叫 `distillation_loss(alpha=.5, temperature=2)`，helper內確實一次乘T²，F08是記號統一建議，不是實作算錯。', '',
          '未跑：完整Notebooks、全部練習變體、長CLI/GPU重訓、公開權重下載/成品推論/UI，以及所有歷史表格逐項原始報告重算。原論文只讀引用相關段落，沒有宣稱全文讀完。根任務稍後的全Notebook結果需合併再判斷。', '',
          '# 第20章目的地仍待定稿', '',
          '已讀前言與19.12所述目的地：19從零教合成小世界MoE；20另接手成熟圖文底座，處理自然照片、中文讀字及先聽寫再進同一聊天核心。這種改起點的理由文字可理解，不能據此判定仍在training/writing的新20實際做到。R.2/R.3/R.4、training導言與19.12最後三段在新章定稿後必須重新與實際資源、底座/adapter選擇、驗收能力卡及節號逐一對讀；目前明確pending。', '',
          '此為指定分段的預檢，不宣稱wholebook final。根任務最後需比較 canonical source bytes 與本報告逐檔/逐節/逐圖指紋；若文字在預檢後改過，受影響結論須對新版本重讀。', '']
(OUT/'report.md').write_text('\n'.join(lines))
print(json.dumps({'report':str(OUT/'report.md'),'inspection':str(OUT/'inspection.json'),
                  'sections':len(understanding),'issues':len(issues),'figure_count':len(figure_notes)},ensure_ascii=False))
