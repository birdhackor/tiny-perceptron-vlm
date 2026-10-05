# 12.9 獨立實際查證紀錄

身分：`/root/phase4_factual_coordinator/factual_12_9`。新 context，未參與作者、第一次閱讀或其他小節技術審閱。只負責 12.9。

實際親讀現原稿 `course/chapters/12.md` 第 265–307 行的小節全部，包括 fence、練習與 details。原始 bytes 保存於 `extracted/section.md`；SHA-256 為 `409619baa10be2b64f80d549d541280646be65ae3ffa8ff7f97fba539d48d378`。最初凍結的完整章節在 `inputs/course/chapters/12.md`，指紋 `97250b5340e72b9b1b05c9cae2e0aed167904e5d3ad3ebe9fa14f7c210c37616` 只表示此次保存輸入，不冒稱日後整章版本。必要前置完整閱讀為 12.8、10.7、11.3，各自 bytes 已保存在 prerequisites。

先讀 factual-reviewer-instructions、checker 全部 schema、section_facts 全文、clear-tutorial SKILL 與 review-protocol。先 rg --files 定位原檔，沒有內容廣搜 repo 或 docs，沒有讀 canonical 舊報告、reader trace、作者額外結果解釋或他人 inspection。來源定位索引只讀 immutable original bytes 的 URL/路徑/hash/版本識別；沒有把索引當作已驗證的結論。

原碼先用 AST 定位，實際讀取：data.py 1–28；model.py 1–105；multimodal.py 1–174；attention.py 1–74；course_experiments/modalities.py 27–53、73–133、195–283、287–374、377–384、884–908。最後的 run_audio scope 解釋值第 909 行沒有讀取。上述範圍含必要資料規約、損失分母、模態展開、生成/EOS/非法 token 判準、凍結設定、步數及更新契約；沒有執行完整 recipe。完整原檔保持不變並保存指紋，沒有刪掉作者附註製造新原檔。

PyTorch 原始來源用 immutable revision `5c4886908584029761b579af026dcfb627c84070`，和實際 `.venv` 的 `2.14.1+cpu` 對應。本人親讀 Tensor.backward 第 566–625 行、CrossEntropyLoss 第 1200–1307 行、Linear 第 53–140 行。官方 pytorch/pytorch repository 是其 API 與自動微分契約的權威；三份原 bytes 又由各自 raw.githubusercontent.com HTTPS URL 重新取回，全部 HTTP 200 且 SHA 與保存原檔相等。原文支持 chain rule/累積 grad、ignore_index/有效目標分母與 affine shape，不能支持教材單音成功率。

本人從原始 Scheduled Sampling PDF 第 1 頁核對標題、四位作者、Google Research、`arXiv:1506.03099v3 [cs.LG] 23 Sep 2015`。親讀 abstract/Introduction、Sec. 2.1 equations (1)–(3)、Sec. 2.2/2.3（保存的 pdftotext 第 1–145 行）。作者原論文支持真實前一 token 的訓練條件與模型自生成前一 token 的推理條件差異；本文是 RNN 論文，沒有拿它聲稱本教材的 transformer 已學會或 scheduled sampling 有效。HTTPS 原文 URL 為 https://arxiv.org/pdf/1506.03099v3，存取日 2026-10-05。

原 fence 透過 section_facts 在 CPU offline helper 真正執行，exit 0、stderr 空、沒有 guard event；stdout 是 `(1, 19, 264)` 與非零接頭梯度 True。正式 code、command、stdout、環境與來源指紋在 original-cpu。短變體 check_cpu.py 只執行随机初始模型的 forward/backward：440/high 用 11 框、20 展開位置、19 預測位置，5 個有效目標位於 14–18，交叉熵逐目標平均 5.8564748764，接頭梯度 norm 0.0766692236。200/low 同樣 11 框，19 展開位置、18 預測位置，4 目標位於 14–17，loss 5.7397203445，梯度 norm 0.0247925986。所有參數在 backward 前後 torch.equal 精確不變。改未來的答案輸入 token，第一答案位置 logits 差為精確零，核因果遮罩。沒有 optimizer step、既有模型載入、訓練或既有模型重評。

原始 audio.json 完整保留，先列必要上層 keys/types，再讀具名原 measurement/samples/provenance pointers（清單在 raw-reaggregation.json）。沒有讀 `/results/scope` 或任何作者結果摘要。check_raw.py 真正以 generated_ids 截至 EOS 重新核 exact-match/decoded answer/invalid special token，重數 14 題中 11 題正確、EOS 14/14、目標 token 62、skipped 0。按原 records 核 train/validation/test 為 16/8/14，frequency family 互斥；歷史 history 300 步、有效目標 5409。當時記錄版本為 revision 22a0bb1b870e4df4af76630243ef55b7ca27840c、seed 42、Python 3.13.3、torch 2.14.1+cu126、cuda；這是核既有記錄，沒有重跑 GPU。必要當時 code SHA 與現原碼相等。

三個錯例為 row 4：290 Hz、振幅 0.25、0.1 秒；row 6：300 Hz、0.25、0.1 秒；row 7：300 Hz、0.5、0.12 秒。全在 290/300 Hz。振幅與時長完全成對變動，而且兩種時長在每個 split 都出現。因此「錯誤集中在邊界與時長變化」不能孤立支持時長作用，保留 revise。建議改成三個錯例都在 290/300 Hz，並說清楚成對變动無法分辨振幅與時長各自影響，不需新增訓練。

實際 Chromium 151.0.7922.173、Playwright `.venv` render 現頁於 1280×800 與 390×844，逐字原 fence 和 HTML pre code 相等；已 view desktop/mobile full screenshot。正文例子、數字、梯度與歷史成果的區分實際可見，本節 image 數為 0。手機 code 橫向滾動，正文閱讀可見。必要前置 SVG 也真正 render/view：12.8 圖保留 11 時間框、16 帶逐框變 8 值並標「三框代表圖」；10.7 圖示 1 marker→3 特徵、展開 7 位置後 input/target 各 6、assistant→73、73→EOS。圖面沒有與相關前置文字矛盾。正式 12.9 figure_sha256 為空，前置圖另存原檔與 hash。

工具限制與修復：第一次 Chromium 用 file:// 開前置 SVG 被 administrator policy 阻擋；保留第一次程式/stdout/stderr，以相同授權原 SVG bytes 用 page.set_content 渲染成功，沒有因此改科學判定。另一次保存 helper 的 copytree 跟隨 workspace symlinks，暫時複製了既有 120 個 checkpoint bytes；沒有 load/evaluate、訓練或阅读被複製的報告。已刪掉我自己產生的 copied workspace，保留 artifact-copy-correction.json，最終正式 artifact 樹沒有 .pt。此事件是檔案複製失誤，沒有接觸被禁的結論。

沒有模型/資料下載、GPU、付費運算、完整訓練 recipe 執行、正文/圖/實作修改或 commit。短 CPU 展示只支持計算與梯度通路；既有有限單音正確率不能外推真人語音理解。
