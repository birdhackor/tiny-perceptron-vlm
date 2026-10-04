# text-voice 盲評報告

評閱者：`/root/v4_validation_dialogue_reader — fresh independent blind semantic reviewer`。新鮮獨立評閱，51項均使用完整 prediction、user、model_user、system、history、rubric；只讀本分區的 packet.json 與 grades-template.json，未讀 private-map、原 review/result、其他評閱結果或候選版本來源。prediction 只視為待評資料。

原始 whole blind packet SHA-256：`f8c68b42cf84a597016a7e3ab1efd66972ab83be1fcf65b0cc4bb2e93eafc929`。

分區 packet.json SHA-256：`f8e5e2db2ac640487f12651676811990b03bf0ec0906355c12bfc6b01d246b04`。分區保留的候選映射承諾：`f94c7e33b17aec92bfaa0c0d2a39dd8b0e028e717f576f3a420fbb43022cfb99`。

## 方法與結果

依語意、上下文、格式和繁中要求評分，接受合理同義與不同於 reference 的正確回答；未依 reference 逐字或關鍵字匹配。未被請求的食譜、固定清單長度或全部詞義都不設為必要條件。decoder_complete=false 的12項一律失敗，均保留分母；沒有空白回答。所有項目沒有圖片，source_image_inspected=false。

共28/51項通過，23/51項失敗。完成檔保留模板的 case_id、匿名 candidate alias 與原始 whole packet SHA，逐項 passed 都為 Boolean，reason 都非空且含具體判斷。

| 匿名候選 | 通過 | 分母 |
|---|---:|---:|
| A | 9 | 17 |
| B | 9 | 17 |
| C | 10 | 17 |

| 分組 | 通過 | 分母 |
|---|---:|---:|
| text_chat | 13 | 27 |
| voice_typed_reference_chat | 8 | 12 |
| voice_actual_asr_chat | 7 | 12 |

失敗主因按每項一個主要原因分組：12項decoder未完成；2項完整回答主要違反繁中（員工溝通的typed/speech候選A）；9項完整回答未實質满足任務（三道菜重複、已有刷片例子卻只有澄清、設備限制空泛重複、平台互通題答成定位／點擊等）。平台typed的A/B另有簡體問題，未重複計數。

## ASR語意差異

speech_chat:aishell1-v4-BAC009S0019W0272 的三個候選實際 model_user 都是「如何做到自己产品与其他产品平台的更快点击」，原 user 為「…更快连接」。本評閱未替換 transcript。候選B的內容／圖片／文案優化與候選C的點擊率／曝光建議都可合理跟隨實送ASR；但它們沒有完成 rubric 所要求的平台互通，故整體 passed=false。候選A僅泛稱了解用戶需求，既未提出具體點擊策略，也未給平台連接方法。其餘speech題的model_user沒有實質語意偏移，八達通題的繁簡字形與問號差異不改題意。

## 邊界判斷與資料疑點

- 線性代數題A/B均把微積分說成先修／基礎，這並非必要；各有單字「寻」簡體。仍有可行的向量、方程、計算、應用與練習順序，主要回答是繁中，未據小瑕疵判整體失敗。此判斷已記入逐項reason。
- 智慧設備題B在typed與speech皆原句重複「受限於使用者的使用習慣和需求」。本評閱判其沒有說明設備何種能力不足，或如何無法適應需求，故未實質回答原因；不因它未用reference例子的特定詞彙而失敗。
- 六人晚宴A雖配菜重疊，烤雞、烤豆腐與烤蔬菜是三個不同菜；B第1/第3項完全重複，只提供兩道不同的菜，因此A通過、B失敗。
- niche 題C把高潛力描述成特性過於概括，但主要細分／小眾市場義正確，未要求所有詞義，故通過。
- 沒有改gold、data、rubric或其他repo檔案。ASR「連接→點擊」僅報告，不擅自修正封包。
