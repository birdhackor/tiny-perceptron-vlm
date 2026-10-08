# 教材實驗與證據

課程的短程式用來拆解一個機制；這裡的完整實驗則實際更新權重，再檢查沒有拿來訓練的題目。兩者分開記錄。模型能載入、代價有下降、能正常結束回答，以及答案正確，是四個不同的檢查，不能由其中一項代替其他項。

[實驗清單](plan.json)列出30組正式實驗、前置權重、固定資料包與對應小節，另列Flash Attention與自然語言選工具兩項補充實驗。[目前進度](progress.json)列出已有的正式報告，以及252節正文的兩種審閱版本狀態；它不替作者或審閱者產生通過紀錄。[費用保留紀錄](budget-reservations.json)包含重跑與失敗，每次GPU工作最多600秒、同時一件，累計上限US$10。保留額是保守的用量上限，不是實際帳單；整個Modal帳戶的費用變化也不能當成本專案的精確花費。

`results/<實驗名>.json` 保存程式版本、種子、環境、有效計分位置、全部留出題評估、選定生成例子、資料與權重指紋。30組正式實驗實際產生的完整目錄備份至私有 Hugging Face repository；其中哪些 checkpoint 含更新器、步數與隨機狀態，要按檔案格式與訓練入口核對。目錄已備份，不代表每個模型都能精確中斷續訓，見[5.7的續訓檢查](../../course/chapters/05.md#5.7)。正式報告的 `hf.revision` 指向權重備份版本，`hf.result_revision` 指向其後保存報告的版本；兩者不能互換。公開報告移除帳戶帳務觀察與私有分支的逐筆文字。

補充的[tool_choice報告](results/tool_choice.json)對應[B.5–B.8](../../course/chapters/0B.md#B.5)：143,616參數的小模型從中文問題生成DIRECT、TOOL或ASK。它只測動作選擇，沒有接上JSON參數與工具執行；能力／成本比較則用人工算例說明。原模板的留出數字對96/96選對，但新問法中6道該用工具的題全選ASK，保留這個失敗供教學。這一項CPU訓練約35.55秒、完整實驗約39.41秒，權重保存在本機，未加入原30組HF學生權重或私有備份；讀者可用`--experiment tool_choice --device cpu`重跑。

## 重跑一組實驗

從有 `pyproject.toml` 的專案根目錄執行；安裝與虛擬環境操作見[暖身](../../course/first-steps.md)。以下只列入口，不要求讀者一開始就跑完整訓練：

```bash
.venv/bin/python -m scripts.course_experiments.run --experiment simple_models --device cpu
.venv/bin/python -m scripts.course_experiments.run --experiment text_foundation --device cuda
```

預設輸出在 `outputs/course-experiments/course-v1/<實驗名>/`。有前置權重的組別從同一批次各實驗目錄讀取；也能以 `--dependencies` 明確指定根目錄。資料包依 `assets/training/manifest.json` 驗證來源與內容。`--step-scale` 小於1只檢查通路，不能取代正式報告；模態組別的本機CPU短訓也明記為通路檢查。沒有CUDA時指定GPU的正式入口會直接說明缺少裝置。

較長的Modal訓練使用手動觸發的 [course-experiments workflow](../../.github/workflows/course-experiments.yml)。工作一次只選一個實驗，結果對應段落修改完成後才進下一件。憑證留在GitHub與Modal Secret，公開教材與權重檔不含憑證。

## 學生權重與審閱

`releases/<實驗名>.json` 是逐檔釋出清單。`reviewed:false` 或 `approved:false` 表示草稿；發布入口會拒絕它。核准清單固定私有來源版本與每個檔案的指紋，匯出時剔除更新器、隨機狀態與訓練參照模型，留下推論必要設定。BPE模型同時提供匹配的分詞器；LoRA同時核對基底；多模態模型保留影像大小、patch與聲音配置。

公開的 `public-models.json` 記錄實際發布後的Hugging Face版本及匯出檔案指紋；下載入口只取選定模型並核對內容。資料與權重的許可分開列在模型卡，來源條款核驗見[發布許可紀錄](release-licenses.md)。PKU 衍生權重不進公開學生模型包，完整報告備份在私有 HF；較早 Actions 診斷包含公開來源的選取片段與模型生成，詳見許可紀錄。

完整實驗與段落修改完成後，先由每節獨立新讀者檢查理解與數據對應，再由另一批獨立審閱者核對事實、原始論文／權威文件與實測證據。各報告固定當節原文、圖解與引用證據的指紋；修改之後須重新核對。最後重建並執行全部Notebook，通過兩種審閱檢查與網站檢查後才發布。這些是代理審閱與程式驗證，並非真人學生測試。
