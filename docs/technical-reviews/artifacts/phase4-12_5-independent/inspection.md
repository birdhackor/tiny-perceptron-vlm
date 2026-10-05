# 12.5 獨立技術查證記錄

審閱身分：`/root/phase4_factual_coordinator/factual_12_5`。日期：2026-10-05。唯一負責小節為 `course/chapters/12.md#12.5`；未撰寫教材，未讀舊 canonical 報告、reader 報告、審閱結論、作者修正摘要或其他人的 inspection。定位時 `rg --files` 曾列出歷史檔名，沒有開啟其內容。沒有發生摘要污染。正式報告由本身完整新建後覆寫，沒有沿用既有 JSON。

先親讀 factual-reviewer-instructions.md、checker schema、section_facts.py 的擷取／執行／防護規約，以及 clear-tutorial skill 與 review-protocol。也依環境指示讀 cloud-environment-onboarding:setup 及 onboarding references；已有 .venv 與渲染工具可用，不需安裝或修改環境設定。

實際正文讀取範圍為第 12 章導言、12.1–12.4 全文和 12.5 全文（首次 frozen 章檔行 1–162）；未讀 12.6 以後正文。前置理解是振幅與頻率不同、取樣率換算時間、完整 unfold 框和 Hann 加權；12.5 進一步用加窗短框的 Fourier 係數表示隨時間的頻率成分。整章保存版本只是 frozen input：SHA-256 `97250b5340e72b9b1b05c9cae2e0aed167904e5d3ad3ebe9fa14f7c210c37616`，不表示親讀整章。正式小節 hash 是原始 UTF-8 bytes 的 `6c030b2e76c370f59171a82218228922b32b57ad67423158c3ed5c728e80ed0c`。

實作先用 AST 列出 multimodal.py 函式位置，再只讀檔首 import（行 1–11）與 tone 契約（44–46）。tone 建立 round(seconds*sample_rate) 個等間隔時間點，回傳 0.5*sin(2*pi*frequency*time)。本節没有原始訓練結果 JSON 或 shell recipe。未讀其他原始實報的附註；只列 original-source-locators 與 original-paper-locators 的頂層 key/type，再讀原始來源 locator 的 URL、版本、路徑、hash。實際使用的 locator 是 `/locators/44`，僅作 immutable 官方 _torch_docs.py 的定位；內容由官方 HTTPS 原始 URL 重新取得並親讀，不使用其他審閱者的結論。

官方來源為 PyTorch 官方 GitHub 原始碼：安裝 wheel 的 git commit `5c4886908584029761b579af026dcfb627c84070`。親讀 functional.py 的 stft 完整函式與 docstring（507–690）：定義、DFT 公式、center=True、reflect、onesided、normalized=False、複數輸出、N×T 軸順序與框數。官方下載與實際安裝 torch.functional 的 bytes 完全相同。_torch_docs.py 先由 AST 定位再讀 abs 219–241、argmax 7106–7156、mean 7195–7261、square 11031–11052、hann_window 13067–13111。這些 API 合組核對本節資料流，不將每個語法分成獨立實驗。另取得 v2.9.0 functional.py 作可用 fallback，但未以其未讀內容支持任何結論；使用的權威版本與安裝版本相同。

原 fence 以 section_facts helper 的 CPU／offline 防護執行，exit 0、guard_events 空、實際執行 fence 1，stdout 為 `(201, 21)`、11、440.0。親讀擷取的 bootstrap.py：本機分支只定位 repo、import torch、設執行緒及 random seed；Colab clone/pip 分支未走。短變體在 CPU 執行，不載入模型或資料：480 Hz→12，450 Hz→11 且 11、12 兩格都有較強功率；center=False→(201,18)，unfold→(18,400)；以逐項 complex sum 手算 DFT 的第 11 格、第 10 框，與 STFT 的 complex 絕對差 5.17e-6，小於 0.001；3+4j 的 abs 與平方為 5、25；幅值加倍使平方幅值四倍；兩單音 440/880 Hz 的最強兩格為 11/22。

數字分母與單位：3200=0.2*16000；400 點=0.025 s，160 點=0.01 s；非負格 400//2+1=201；center=True 的 T=1+3200//160=21，未補邊的 T=1+(3200-400)//160=18；頻率 f[k]=k*16000/400，間隔 40 Hz；mean(-1) 沿時間軸，用 21 框作平均。power 是 normalized=False 的平方 STFT 幅值；本節沒有宣稱它是 watts、PSD/Hz、dB、校準響度或不同配方可直接比較的值。

圖檔 raw bytes SHA-256 `a36139901176e9b8356a785a19138b1ee0b6ae8a61064f5d8eb71496cb0f8820`。親讀 SVG 全文，Inkscape 實際渲染並 view_image 看 PNG。另以 Chromium/Playwright 實際渲染當前 8765/12.5.html，viewport 1280×800 與 390×844，全頁截图均親自 view_image 查看。六個正文段落與原 fence 跟當前 Markdown 比對相符，HTTP 圖 bytes 跟 frozen SVG 完全相同。波形振幅 ±0.5、前 0.008 秒約 3.52 週期；161 個座標跟獨立 440 Hz 正弦取樣的 x/y 誤差均不超過 0.005001 px。頻譜示意的頻率縱軸 0/440/880 Hz、時間橫軸前五框中心 0/0.01/0.02/0.03/0.04 s、框長與步長均一致，且明寫亮度不是實測值，沒有把五框誤認成完整 21 框或把示意色當測量。

工具事件：第一個合併讀檔 command 因 exec-server transport disconnected 而未執行，拆成精確讀檔後成功。兩次直接 Chromium CLI 截圖卡在瀏覽器背景請求，未產生圖片；只終止本審閱的兩個 profile process，保留 stderr，改用有界 Playwright、限定本機請求後成功。Inkscape 出現 Pango/GtkRecentManager 警告，但圖片正常產生並可見。初次逆推 rounded SVG x 的 ad hoc 誤差檢查套用過緊 y 容忍差而失敗；改以未四捨五入的 sample index 各自計算 x/y，符合原座標的 0.01 px 顯示精度。這是查證方法的修正，沒有修改圖或教材。

限制與判定：查證的是固定頻谱計算與合成聲音算例，不是模型能力或真人語音評測。非整格頻率與加窗本來就可分散到邻格；圖不宣稱實測強度；最強頻率格也不是音高序列或詞彙編號。沒有訓練、重測既有模型、資料／模型下載、GPU／付費運算、完整 shell recipe、.pt 輸出、正文／圖／實作修改或 commit。未發現影响本節理解的實質錯誤，判定 pass；checker 僅在完整新報告成功生成並確認 reviewer_task 後單節執行。
