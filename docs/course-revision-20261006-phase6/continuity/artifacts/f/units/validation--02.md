## 小實驗、正式訓練與成品各有自己的範圍

正文的短程式把一個機制縮到可以追蹤的大小，例如遮住未來、對齊答案位置或只更新LoRA分支。它們幫助理解原理，不直接代表自然圖片與一般中文的能力。[正式實驗入口](course-experiments/README.md)連到各項訓練資料、配置與實際報告。

[第19章v2](../course/chapters/19.md#19.12)已完成從隨機初始化的MoE與Dense訓練、公開推論輸出和各3,734題的固定最後測試。文字核心、視覺／語音入口與接頭都自行訓練；三類服飾與兩格位置、12字的指定連續1–4字區域、三種銀行客服語音主題及有限文字／工具任務各有分母。兩版都未通過全部原定能力判準。[完整結果](selftrained/results/v2-final-public-results.json)保留原判準及逐項成績：MoE完整工具往返0/276、語音回答42/90、語音後續對話26/60；圖片服飾辨識357/360與OCR252/324也不能代替這三項能力。

[公開CPU示範](selftrained/v2-public-cpu-commands.md)使用挑選的validation成功題，完成8次chat與1次history append，支持匿名下載、載入及介面運作。它不改最後測試分數。[本機stage wrapper驗證](selftrained/infrastructure/v2-local-stage/independent-review/review-final.md)檢查自己的新段、完整sidecars、失敗及精確續訓契約，使用width16合成CPU材料，沒有重新完成全量production訓練。

已發布的[舊合成示範](course-experiments/README.md)只涵蓋簡單圖形、純音、固定句型與受限工具，其歷史成績不能代替v2驗收。[第20章](../course/chapters/20.md)是成熟模型的延伸路線，使用已有的圖文底座與語音辨識器，分別檢查自然照片、中文讀字、文字聊天及語音聊天。兩條路的題目、權重來源與規模不同，分數不能直接當成同一場比賽。

每份能力結果都要一起讀資料範圍、分母、生成設定與判分方式。同一張圖的多個問答彼此相關，四個錄音問題也不能代表所有一般人的說話情境。第20章的[能力卡](../course/chapters/20.md#20.13)把這些限制放在得分旁邊。

