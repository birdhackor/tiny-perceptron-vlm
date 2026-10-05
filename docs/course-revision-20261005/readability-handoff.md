# 第4階段易讀性讀者輪交接

讀者輪完成時點：2026-10-05T09:56:25.462952+00:00。基準版本：`457e188be67b6256910eccbc5b7063fa54d34727`。

309個編號小節均有目前正文、導言與圖版本相符的獨立讀者pass報告；共309個不同的實際讀者task，現版trace有1,244個逐段checkpoint。已停止派工，保留原讀者task供根協調者必要時召回。

這是易讀性讀者輪的交接。根協調者已執行reader schema、新輪身分、durable trace三個正式gate，全部通過。格式檢查器原本將JSON／文字答案框誤算為可執行程式，現已區分並通過32項檢查器測試，讀者判定未被改動。身分與逐段版本檢查見[reader-round-audit.json](reader-round-audit.json)及[readability-trace-audit.json](readability-trace-audit.json)；本文件與工具計數不代替真實閱讀。factual、continuity、成品工程、資料準備與訓練尚未開始，新增程式與命令未因reader pass視為已執行。

逐節派工、凍結SHA、原問題與處理狀態見[review-progress.json](review-progress.json)。讀者自己的現版報告、所有歷史報告與不可回改trace在[reader-reviews](../reader-reviews/)。

本輪相對基準有110節正文小節內容改動；125筆作者處理紀錄包括讀者斷點、主軸整理與版面待辦，不是125個互斥問題。主要修改是材料與標籤對應、首次術語、示範範圍、練習觀察點、圖文位置，以及刪去無額外教學作用的歷史成績／流水。必要條件與反例保留，修改後均由原讀者新session實讀。

## 正文修改小節清單

| 來源 | 小節 |
| --- | --- |
| `course/chapters/01.md` | 1.11 |
| `course/chapters/02.md` | 2.4 |
| `course/chapters/03.md` | 3.4, 3.5, 3.7 |
| `course/chapters/04.md` | 4.3, 4.8 |
| `course/chapters/05.md` | 5.3, 5.6, 5.7, 5.13, 5.14, 5.15, 5.16, 5.17 |
| `course/chapters/06.md` | 6.5, 6.8 |
| `course/chapters/07.md` | 7.1, 7.4, 7.7, 7.11, 7.12, 7.16, 7.18 |
| `course/chapters/08.md` | 8.3, 8.8, 8.14, 8.15 |
| `course/chapters/09.md` | 9.10 |
| `course/chapters/10.md` | 10.2, 10.4, 10.10, 10.11 |
| `course/chapters/11.md` | 11.2, 11.4, 11.5, 11.9, 11.10, 11.12, 11.15, 11.16 |
| `course/chapters/12.md` | 12.1, 12.4, 12.6, 12.7, 12.8, 12.10, 12.12 |
| `course/chapters/13.md` | 13.8, 13.13, 13.16 |
| `course/chapters/14.md` | 14.1 |
| `course/chapters/15.md` | 15.13 |
| `course/chapters/16.md` | 16.8, 16.9, 16.12 |
| `course/chapters/17.md` | 17.5, 17.9, 17.12, 17.15 |
| `course/chapters/18.md` | 18.1, 18.3, 18.7, 18.10, 18.11, 18.12, 18.13, 18.14 |
| `course/chapters/19.md` | 19.1, 19.2, 19.4, 19.6, 19.7, 19.9, 19.10, 19.11, 19.12 |
| `course/chapters/20.md` | 20.4, 20.5, 20.6, 20.7, 20.9 |
| `course/chapters/0A.md` | A.2, A.5, A.9 |
| `course/chapters/0B.md` | B.1, B.2, B.3, B.4, B.7, B.8 |
| `course/chapters/0C.md` | C.1, C.2, C.4, C.5, C.6, C.7 |
| `course/README.md` | R.4 |
| `course/first-steps.md` | W.5 |
| `course/training.md` | T.2, T.3, T.4, T.5, T.6, T.7, T.8, T.9, T.10, T.11 |
| `course/glossary.md` | G.4 |

逐項改法與實際複查trace在每節records的`issues_handled`，預派作者修訂在`author_predispatch_edits`，不需以本摘要代替原記錄。

本輪亦調整7個SVG：`audio_calibration_test.svg`、`flash_allocated_memory.svg`、`rewrite-03-qkv-flow.svg`、`rewrite-06-05-shared-denominator.svg`、`rewrite-11-crop-evidence.svg`、`rewrite-A-rag-flow.svg`、`tokenizer_common_scale.svg`。涉及文字尺寸、標籤、向量笑臉與刪去不存在的流程節點；對應現版報告保存實際640／360圖檢與圖SHA。根端頁面抽查是獨立版面證據，不取代讀者判定。

## 下一輪待核對

progress保留26筆`pending_technical_questions`記錄，包含已移除舊例的保留記錄；需按每筆狀態與目前正文核對，不能把所有舊問題一律當成新教材主張。尤其本輪新增／調整的3.5、5.17、19.9／19.10、C.1與T.3／T.5／T.6／T.8／T.9相關示例或命令，均未在易讀性輪執行。T.9的新只讀張量bytes snippet、量化三版評估命令的載入／schema／共享儲存計數範圍等，已明確留給技術輪。

T.5的容量錯誤與同一讀者恢復原真實session已記在`execution_events`；先前不存在9.11／9.12的零單元派工錯誤保留於`dispatch_errors`，沒有計入309節。共同filesystem不是技術隔離，本輪只宣稱遵守分段揭露流程。

Notebook同步、正式gate與checkpoint由根協調者接手。協調者本輪未commit、push或改generated/sharednav。
