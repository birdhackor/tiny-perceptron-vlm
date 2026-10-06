## Modal GPU 與 HF checkpoint 實測

雲端訓練還要核對另一層：GPU是否真的執行更新，中途存檔是否能重新取得，以及續訓是否接上原狀態。早期連線實測使用31,584參數的小模型，完成80步更新，從私有Hugging Face備份重新取回第40步存檔，再接到第80步；該次對照與不中斷路線的最大權重差為0，見[原始結果](gpu-smoke-result.json)。

這項檢查支持那次小模型的更新、傳輸與接續，不提供自然圖片、語音或大模型品質的證明。第20章的訓練與選版使用自己的[完整操作指引](natural-assistant/v4/TRAINING.md)和[驗證決定](natural-assistant/v4/selection.json)，各自核對資料與設定。

