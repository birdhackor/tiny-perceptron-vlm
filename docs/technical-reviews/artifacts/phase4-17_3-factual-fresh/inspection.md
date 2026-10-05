# 本輪 17.3 獨立正確性審閱

Reviewer task: `/root/phase4_factual_coordinator/factual_17_3`。2026-10-05；fresh context，只審 17.3。未讀舊 technical/reader 報告、作者預寫修正摘要或舊判定；沒有污染／停止事件。沒有改正文、圖或訓練，也沒有委派子代理。

實際閱讀：審閱方法 `docs/review-tools/factual-reviewer-instructions.md`；checker schema；section_facts 擷取與執行介面；`.agents/skills/clear-tutorial/SKILL.md` 和 review-protocol。`course/chapters/17.md` 行 1–106（章導言、17.1–17.2 前文與 17.3 全節），另看到 17.4 標題行 107。17.1 的現稿實報不是本節待查 claim，也未開該原 JSON、重訓或重評模型。初始整章 raw bytes 已存 `originals/17-frozen.md`；它的 hash 僅表示此次 frozen input，不冒充往後章版本。正式 section fingerprint 對 `originals/section.md` 的原始 UTF-8 bytes 計算，保留換行。

先以 rg --files 定位方法、來源及候選原檔；初次檔名定位輸出含舊 artifacts 路徑但未開報告。之後外部權威資料直接由官方 HTTPS 取得，未讀來源庫摘要。普通 PyTorch API 定義合組核查，來源 commit 與本機 torch.version.git_version 完全相同：5c4886908584029761b579af026dcfb627c84070（2.14.1+cpu）。先 AST 定位 add_docstr/add_docstr_all 與 L1Loss，再逐項親讀所列範圍；定位永久保存在 `originals/official_ast_locators.json`。沒有把關聯 API 的一般 docstring 當成污染。

PyTorch `_torch_docs.py` 親讀 abs 219–241、mean 7195–7261、matmul 7908–7990、tensor 9582–9635、round 9897–9951、transpose 11942–11990。abs 是逐元素绝對值；未指定維度的 mean 取所有元素平均；兩個二維張量做矩陣乘法；round 保留輸入 dtype、ties-to-even，decimals 是數值捨入而非列印格式。

PyTorch `_tensor_docs.py` 親讀 item 2788–2805、round 4268–4275、float 5274–5284、tolist 5445–5465、T 6769–6783。T 對二維權重交換行列；tolist/item 返回 Python 數值，不能保證固定小數位。loss.py L1Loss 66–133 親讀 MAE 式與 reduction='mean' 的全部元素分母；本節四係數的分母為 4，並非兩筆輸入或兩個輸出。

AdaRound 原論文：Nagel et al., Up or Down? Adaptive Rounding for Post-Training Quantization；arXiv:2004.10568v2，2020-06-30；PDF 頁內印出的版本與題名、作者已核。PDF 原件由 https://arxiv.org/pdf/2004.10568 取得，版本化引用 https://arxiv.org/pdf/2004.10568v2。親讀 `pdftotext -layout` 輸出行 1–343（pp.1–5 開頭）：Eq.(1) 最近格、scale 與 clip；Sec.2 Eq.(2)–(8) 任務 loss 對共同權重擾動的依賴；Sec.3.2 Eq.(14)–(20) 的 z=Wx、輸入二階矩和 (delta W x)^2；Sec.3.3 Eq.(25) 與相鄰段落的前層誤差與 activation 處理。這些直接支持本節機制和限制；論文方法的近似假設不被當作無條件品質保證，其 ImageNet 實報也未在此重評。

數學獨立核對：delta W=[[1/5,1/5],[-1/5,1/5]]；X=[[1,1],[1,-1]]。第 b 筆第 o 個輸出差 D[b,o]=sum_f X[b,f]delta W[o,f]，因此 D=[[2/5,0],[0,-2/5]]。每格係數誤差1/5並不作為D的上限；一個可用上界是 max_f|delta W[o,f]| × sum_f|X[b,f]|。MAE=(4×1/5)/4=1/5。此無物理單位的例子中權重 MAE 是係數單位，輸出差是輸入乘係數的單位，兩者不可直接互換。

原 fence 以原 bytes 單獨真執行（exit 0），另由 verify.py 再執行並驗證，沒有補改原碼。原 stdout 顯示 MAE 0.2、原輸出約[[0.9,0.5],[0.5,-1.9]]、還原輸出[[0.5,0.5],[0.5,-1.5]]、逐項差約[[0.4,0],[0,-0.4]]。未做 tensor.round 時，有一個取消位置仍約5.96e-8，與現稿已交代的尾差一致。原值誤差以 atol 1e-6、rtol 0 檢查；x×10 的代數重排以 atol 1e-5、rtol 0 檢查。Fraction 獨立有理數推導則用精確相等。

必要 CPU 變體：x×10 對應[[4,0],[0,-4]]、權重 MAE 不變；x×(-1) 只反轉輸出誤差符號；x×0 得全零差；額外三筆輸入的 X[3,2] 得 Y[3,2]，防止兩個方形 tensor 隱藏軸解讀错误。verify.json 保存每個實際浮點值，未以預期代替觀察。restored 是 float32、每格4 byte，沒有 int8、bit packing、backward 或 optimizer step。這是數值示範，不是訓練／壓縮容量／模型品質／速度的驗收。

17.3 沒有引用圖；文字逐列數字、原碼和輸出已明確表示兩筆輸入和兩個輸出，不需要额外圖去判讀空間或箭頭。本輪 figure render/view 不適用，未聲稱渲染過未存在的圖。無 raw measurements、模型 weights、完整模型 scores 的必要存取；未下載資料或模型，沒有 GPU 作業。

本輪判定：pass。七個實質 claim 各有原始權威來源、數學與／或真執行的支持範圍；沒有尚未確認的實質主張。checker 只作格式與版本檢查，真偽判定由本次本人核查負責。
