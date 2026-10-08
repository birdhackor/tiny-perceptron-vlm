# 教材局部修補與複查紀錄（2026-10-08）

本輪完成局部修補，保留章節架構、順序及原始實驗數據。33個小節的文字或圖表與4個操作／參考頁有調整；另同步英文README及新增工具紀錄勘誤。沒有整章重寫或重新訓練模型。

## 修了什麼

前次全325頁審閱收斂的11項必要補點及28項有收益的可選修改已落實；23項維持原文，3項可選潤飾延後。這次修後複查另外辨認7項局部必要關係，已經回修並回讀，沒有以後來的同行認可改寫原始分級。

主要補點是FFN、Adam／AdamW、重採樣、mel、YaRN與MoE的選用用途；把計分目標、正常答題、QAT屬性問答及商品位置題的實驗範圍說清楚；補齊教師權重和資料的固定配方前置；修正Git LFS安裝入口、下載動作、temperature 0與環境連結的操作約定。可選修改包括少量術語對名、刪除重複說明、來源定位、CER轉寫方向及窄螢幕補圖。

工具實驗的36／56筆generation metadata曾引用後續改動的messages；原始input_ids逐筆重建仍全部吻合。未來紀錄改存deepcopy，兩項回歸測試檢查快照與兩步工具回饋；舊JSON及分數保留，另附可重現補件。這項修正沒有把重建當成新模型實測，詳見[工具紀錄勘誤](tool-trace-errata.md)。

## 怎樣複查

[前次診斷](diagnosis.json)及其[原始cross報告](reviews/audit-before/relocation.json)保留原bytes。本輪作者紀錄在[authors](authors/)，與審閱角色分開。

[第一份修後凍結](reviews/freeze-01/manifest.json)涵蓋37個受影響canonical單位與2個附加頁。讀者先保存主文理解，再讀選讀／必要前文；全體讀者完成後，才依次進入技術、銜接與交叉審閱。各階段的實際封存檢查在該目錄的checks中。三頁早期context曝光未隱去，另派未提示的讀者重做受影響首讀；learning報告的16處引用字形差異由自己的cross紀錄附勘誤，沒有改寫舊報告。

交叉完成後的10頁回修在[callback-02](reviews/callback-02/manifest.json)。7位原讀者完整回讀自己相關頁，另外兩位實際技術／銜接角色各讀全部10頁。這是知情局部回查，沒有充當新的盲讀初判。實際報告與SHA見[最終callback封存](reviews/callback-02/checks/all-related-callbacks-sealed.json)；逐項處置、分級差異與限制見[協調者裁定](adjudication.json)。

## 完成的驗證與界線

- 287份Notebook產生及同步檢查通過；33份有文字變動。程式相對baseline僅9.6的兩處變數名改為non_refusal_rate，其他AST相同。
- 協定、匯出及估時相關測試共54項通過；Ruff檢查及2045檔格式檢查通過。審閱原始證據維持狹窄的archive排除，可重現工具補件仍受檢查。
- 325頁、4條路線的估時來源／圖檔SHA核對通過；37頁採實讀者提供的新AI估計，288筆未變資料保留。局部再修使用本人全文回讀的新估計，沒有只換hash。估計含理解與短思考，排除安裝、下載、操作與訓練等待；不是人類實測。
- 本機嚴格Zensical建置及326個HTML頁、287組下載／Colab入口、115936個內部連結、597個Notebook閱讀連結通過。預覽包含完整來源綁定的AI估時，沒有宣稱提供本輪執行輸出。
- 相關10頁在1280×800與390×844取得實際DOM／截圖，未見全頁溢出或缺圖；人工視讀範圍另記。教師折疊的連結、手動展開、Markdown呈現與MoE命令橫捲有實際紀錄。課綱上捲時的中央回頂按鈕遮圖已由單行設定修正並重測；右下TOC局部蓋字仍屬已記錄的可選限制。
- 147個原始技術輸入、113張圖的修後凍結版本及審閱準則指紋核對未變。原始資產、分數、dependency lock與舊審閱指標沒有改寫。

實際檢查記錄見[本機驗證](reviews/callback-02/checks/root-software-final.json)、[來源完整性](reviews/callback-02/checks/root-source-integrity-final.json)與[原位視讀](reviews/callback-02/checks/root-ui-visual-final.json)。沒有重新執行全部287個kernel、GPU／MPS／Colab訓練或真人新手測試。

舊的正式發布審閱閘門仍實際失敗：`skill/protocol criteria changed; previous results cannot be reused`。本輪是完成本機修補與受影響單位複查，沒有更新或繞過舊active指標，也沒有宣稱正式發布驗收通過。沒有commit、push或發布。
