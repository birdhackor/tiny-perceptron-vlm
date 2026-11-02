# chapter-19 獨立技術原判的人工核對

審閱者：`/root/omni_p5_technical_chapter_19`，fresh。審閱對象是 canonical
`chapter-19`：`course/chapters/19.md` 第一個 `##` 之前的完整 2306 bytes；
不是 19.1–19.24 的逐節技術判定，也不是 reader 或 continuity 證據。

讀取 AGENTS.md、technical-review-guide、clear-tutorial SKILL／project-context、
REVIEW-PROTOCOL 與指定中立包後，完整讀章導讀及其 details。沒有讀其他角色的
審閱、作者對審閱的處置或預期答案。訓練／測試的原始收據與正式來源選擇紀錄
作為產品本身的主證據；其中提到的既有 independent review 沒有開啟。

## 主張與證據的人工判斷

1. 「MoE 助理讀文字、商品圖、字卡與錄音後用文字回答」是本專案有限產品的
   輸入輸出契約。`LimitedAssistant` 真正建立同一 LM 及 image／OCR／audio
   三個 encoder；`_embeddings` 把 encoder tokens 放入 LM 的保留槽，`generate`
   產生 token。encoder 的 CNN／GRU、projection、symbol projection 與位置表示
   都是本專案建立的神經模組。這支持「轉成特徵，再由接頭轉成向量」；接頭可在
   encoder 內，不要求它是一個另命名的獨立類別。「讀得懂」在這裡解作模型接受
   相容的向量，不能據此聲稱所有輸入都答對。

2. 隨機起點不只取自檔頭註解：檢查 trainer 的無來源初始化／既有本課 checkpoint
   分支，並讀取原 run `gha-37440329212-1` 的 execution、train-receipt、
   inference-manifest。origin 為 `all-neural-weights-random`，歷史含 pretrain、sft、
   vision、ocr、audio、joint；4000 次更新中依 validation 選 step 1000。selected
   checkpoint 與公開 safetensors 的 SHA 都和保留結果相符。這是從自有隨機起點
   分階段接續，並非每個階段重置所有參數。

3. 來源句只承諾「明確列出的資料來源」，不承諾素材自己創作。直接讀取三份
   provenance 的原始 metadata：Fashion-MNIST 固定 commit、Noto CJK Sans／Serif
   固定 commit、MInDS-14 zh-CN 固定 revision，均列 URL、指紋或來源文件。
   沒有將記錄筆數當成人數，也沒有據此重新作全面授權審計。

4. Looped 與 C 的介面由實際 `lesson_products/model.py`／`inference.py` 核對。
   Looped 讀有限語法並輸出 19 個答案類別；C 將 state、question、candidate meaning
   編碼後評分，以固定正溫度 softmax 回傳各候選機率。抽出實際 `parse_state`
   在隔離 namespace 執行：事件、加法、資訊缺失的客服狀態可解析，任意聊天被拒絕。
   沒有建立 Product、載入權重或執行神經網路。

5. 兩個新成品已做 GPU 訓練的句子由原 B／C execution 與原 training-receipt 支持。
   分別為 NVIDIA L4、seed7421／9041、1000／800 次更新、weights_changed=true。
   現有 B／C 權重、config、tokenizer 及 training-receipt 的 bytes 均與事前 settings
   freeze 相符。凍結設定存在於教材所指的 pretest commit 的真 Git object，且
   settings 時間早於 test-started marker，後者又早於 final report。

6. 獨立重算原始 85,302 列封包中的 product=B、split=test、depth=5 分數 argmax，
   得事件 2928／2956（1478 家族），高於固定 90%；加法 2／42（10 家族），低於
   85%。argmax 與保存的 answers 一致。C 的三型分別為 11797、10890、10890 題，
   既存 raw scores、selected 與正解一致；候選倒序記錄亦一致。這只支持有限舊版
   世界，不能外推為一般語言理解；共享家族題不是獨立樣本。保留 7068 bytes 的
   B gold／argmax 向量，另以完全只依賴本 proof 目錄的程式重算門檻。

7. 「19.12 用整份測試資料」由 complete journal、aggregate expected/count=3734
   及逐 task 分母合計核對，選定 safetensors SHA 亦一致。本頁只導覽能力與不足，
   沒有承諾 MoE 通過全部驗收；實際紀錄 capability_acceptance_passed=false、
   OCR exact252／324、voice42／90、真工具往返0／276 等失敗保留。沒有把章導讀的
   「可以實際使用」解讀成任意輸入可靠或所有原定門檻都過。

8. 完整讀取必要產品對照 19.13、成績界線 19.20，以及第20章導讀；核對其餘
   chapter19 headings、必要路徑與和本頁主張直接相關的來源段落／程式。19.1–12
   確實是原 MoE 流程，19.14–20 是共同家族、B、C、Auto、比較與下載／驗收。
   章導讀的回查範圍與目的相符。這些目的地的登錄、SHA、標題和導航有逐一紀錄；
   並沒有將只掃描的目的地冒稱另一份逐節完整審閱。

9. 新增 19.21–24 分別處理影音、聲音輸出、共享材料／候選、整合路由。
   每節確有本章內的 Python；解析全部 10 個 Python blocks（1301 行）而未執行。
   19.21–23 明確區分隨機／人工 CPU 小例、正式架構／訓練與 final 成績；
   19.24 的兩段是純 Python 人工控制流程，沒有字面 CPU 字樣，也不因此成為 GPU
   訓練證據。新 C／Auto 研究候選沒有替換舊預設，V2 與 B／C 舊版成績仍列出。
   這支持本頁的版本和證據分工，不等於我替這四節的所有機制、程式、門檻或能力
   作了逐節 pass；「看懂」仍是 reader 的獨立審閱工作。

## 實際頁面

用 Chromium 151.0.7922.34 實看 1440×1000 桌面及 390×844 手機，打開回查 details。
完整章導讀與24節目錄都存在，沒有水平溢出，必要文字與連結可辨認。再獨立走訪
35 個內部 links，全部 HTTP200，heading 對上 canonical freeze。網站將原 Markdown
章檔 anchor 正確轉為各節 HTML，因此 URL 沒有原始 `#19.12` 不是失連。

第一個 screenshot probe 的網址篩選沒有收進瀏覽器改寫成絕對 local URL 的 links；
它的原始 receipt 保留。另存 navigation_probe 與完整成功 receipt 補實查，不把第一份
空 targets 說成通過。兩張實看截圖在 Git 外，路徑／bytes／SHA 在 browser receipt；
只有收據進 repo。

## 未驗範圍

沒有訓練、retune、正式 final／teacher NN forward、Modal、workflow 或發布；沒有
載入 PyTorch 或任何模型權重。沒有重新量測任一成品，也沒有將保存分數重算說成
重做模型推論。V2 使用原始 aggregate／journal及訓練收據，未逐筆重評全部生成文字。
未重新下載公開大權重、驗 Colab、MPS、GPU kernel、真人閱讀時間、無障礙工具，
也不對其他 canonical 頁提供技術 pass。章導讀無 SVG／數學式／命令，故無圖一致性
項目；其產品與門檻主張有實質內容，不能整頁 NA。

原始分數的固定來源：
`https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/2c6a1062f3e66a602cf83eda6c703456ec18bca0/lesson-products/v1-20261016/raw/final-report-v1.predictions.jsonl.gz`。
本次讀的是現存原始下載，compressed SHA
`cce86e7adb505877caa8a122b07426fd5effc3037aec2a29c22a8e9f8309790c`，
uncompressed SHA `8b593421043b2d0ee8b81caea34612b468e8af7eafd5b84c552f8c25b0cf4896`，
兩者均與本專案原 publication-package／final-report 對上。其他原始輸入的
實際 path、bytes、SHA、JSON 定位和 run 版本在 inspection-recomputation-v1.json。
