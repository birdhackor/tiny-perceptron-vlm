# 新版產物、資料與升版規格

P1獨立審查後的具體定義與處置見[CONTRACTS.md](CONTRACTS.md)；其模組邊界、計數、度量與資產通道細節優先。尚未完成正式模型工程、development profile或final驗收。

本規格於正式訓練與sealed final前凍結。工程probe只可用development。所有新元件從隨機初始化訓練；自家舊A續訓參數另標warm start，不稱重新預訓練。

## A：短片＋聲音時間對齊

- 新版payload video-audio-v1：4秒以內、最多16張32×32灰階影格、各影格相對共同起點的毫秒時間；16kHz waveform及實際有效長度；mask與抽樣規則公開。舊圖片／OCR／有限真人語音介面保持獨立可測。
- 局部聲音token的支持範圍最多250ms，禁止拿雙向GRU全域特徵假稱局部事件；因果token索引與物理時間分開。
- 主要問題：發聲時看見圓或方？同影格／聲音內容、只改時間的配對都答對；時長、影格間距、振幅及位置變體分家族；單模態刪除必須不可判定，不餵gold event ID。
- 正式比較：實際時間加法embedding／時間RoPE，各三個固定下游seed，基線、資料及計算條件明列。full TMRoPE的空間角色另CPU例，不由無空間變化的任務聲稱完整收益。
- 資料上限train12000家族、dev1000、final200配對家族；family先分再渲染所有變體。至少包含時長／間距留出及線索刪除。
- 升版目標：密封主配對正確率至少85%，各時長／間距面板至少75%，線索刪除不可判定至少90%；與chance及簡單控制分開報。只作有限合成影音能力，不外推開放影片理解。三seed完整結果與區間均保留，不只報最好一次。
- 新／舊任務分開評，歷史已見測試不作新版sealed final。保留能力新面板每舊任務至少100個獨立家族，母體不足時公開實數。升版須列出相對舊版退步；若主要舊任務退步超過5個百分點，維持舊版預設並發布研究候選，不拿平均掩飾。

## A的有限聲音生成模組

- 輸入有限文字／狀態的音型指令；輸出8kHz、最多2秒、最多4事件的音調／節奏／停頓waveform；非人聲TTS。
- 從零訓練單碼本encoder／codebook／causal decoder；教straight-through與更新方法。codec凍結後才產Talker targets，Talker訓練有限條件到聲音碼。
- train上限5000聲音程式家族、dev500、final200；完整頻率／時值／停頓組合留出，phase/amplitude/noise為同家族變體。codec與Talker都不可看final組合。資料可由固定生成器重建，不必提交所有waveform。
- 三個層次分開：codec重建、gold條件的生成、真A答案→公開音型約定→Talker→decoder整鏈。另有解析振盪器／靜音控制，檢查塌縮，不宣稱學習法比解析合成更好。
- 升版目標：final重建平均SNR至少10dB、gold條件事件頻率與順序正確至少85%；實際A條件整鏈至少80%，完整輸出合法率100%。有意義靜音段由事件分析器單獨處理，不能以靜音提高SNR。判準計算與分析器先用development驗證並凍結。
- 只有實際可播放chunk可報首段等待；雙向輸入仍是離線輸入，不稱full-duplex。

## C：共享材料的有限語義判斷

- state-only encoder不讀question/candidate，各問題／實質候選唯讀共享state。候選用自足描述，不允許「前一項」。共享參數獨立實質候選分支＋對稱none彙整保持清單重排等變；none正確性不自動保證。
- 固定UTF-8 bytes覆蓋：state≤512 bytes、question≤128、每候選≤128、最多8候選；超限明確拒絕。byte覆蓋不等於新語義；未見措辭／組合／概念分列。
- Choice、資訊足夠、rubric型別分開；unsupported世界範圍與目前列表none不同。任務仍有限客服／簡單狀態與算式世界，不能宣稱通用Jev複製。
- 訓練上限30000家族、model-dev2000；校準80／policy60／final60個獨立情境家族另外切分。來源留出若無真正不同來源，只稱模板／措辭留出，不能拿同生成器不同seed湊不同來源。
- calibration只選T，policy只選固定接受門檻；final不得再調。p_max與集中度不同，溫度不保證改善。各語言／來源／型別／接受數、錯誤、coverage、risk及區間分開；zero accepted risk未定義。
- 升版目標：主要Choice至少90%、資訊足夠與rubric各至少85%；候選重排機率／語義一一對應在fp32容忍1e-5內；共享／逐題重算中間狀態與機率容忍1e-5。固定policy在主要支援面板coverage至少50%且empirical accepted error≤5%，僅有限樣本觀察，不聲稱未來5%保證。任何面板無足夠接受數須公開，不用post-hoc curve挑分數。
- 正式C核心一次＋同架構同seed資料對照一次，不把tokenizer、架構與資料全部改變的產品對照叫pointer因果收益。

## B與Auto

- B保留自己的tokenizer／文法／權重／能力邊界。適配器只接受可無歧義轉換的輸入，其他拒絕／轉交。
- C只看到文字與附件存在資訊時，路由主張只限這些資訊；不假稱讀取影音內容。人工指定優先。
- gold路由與真C路由都跑全鏈，至少各100個凍結家族；分列路由／適配／A／B／聲音失敗。新增路由不能當作B既有加法能力已改善。

## 凍結、正式研究與發布

正式最低矩陣12jobs：A2表示×3seed=6；C核心與資料對照=2；codec+Talker=2；Kev及3B研究=2。額外infra/profiling/final皆計入US$60新round；官方rates與time/step/media上限先凍結最壞預約。核心sealed final不可為省額度刪掉；不足則明列未完成。本文數值是事前能力目標，不是已見結果；若development顯示量測定義錯誤，需在開final前保存修正理由與新凍結，不得看final降標準。

成果分為實作／訓練完成、能力達標、已正式發布三個事實。失敗候選照實教學，只有達標且版本／審閱／下載全過才替換正式預設。發布權重、完整optimizer/RNG checkpoint與資料生成版本分清楚。所有歷史結果不改。
