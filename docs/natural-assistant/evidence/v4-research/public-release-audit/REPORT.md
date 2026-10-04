# v4 公開成品資料的獨立審核

Verdict: **PASS**。審核者 `/root/v4_public_release_auditor` 已實際完整閱讀並核對最終候選與 model card，於 2026-10-04T18:12:02.967994+00:00 保存批准檔。這是 release artifact audit：沒有訓練、選版、新推論、重新語義評分、Modal dispatch、HF upload、Git 指令或對外訊息。

## 審核的確切內容

- 未批准候選：`outputs/natural-v4/release-preflight/assistant-2b-v4.candidate.json`，9,286 bytes，SHA-256 `c529b559bcb71840bb954b5164388ad1c0ed16e498ba4e02bc1b9e42200119ab`。
- 批准檔：`docs/natural-assistant/releases/assistant-2b-v4.json`，9,284 bytes，SHA-256 `05229a7132e7c4045c5ffe4996135d27282218093ba998f0b3c48011878b7a38`。
- 卡片：8,722 UTF-8 bytes，SHA-256 `c319a97d90aca2a074410e3ece8c6325515e0c3a336660e4ca1d00ba6fab65fd`；JSON 的 `model_card` 與獨立 candidate Markdown 位元組一致。
- Canonical selection：`docs/natural-assistant/v4/selection.json`，SHA-256 `8570755345e08c55ab77c165b405f41f6e5012b4c37a93909472b7f17334c42f`。
- Manifest：SHA-256 `0c660490eb78bd82a8e092c2658646a6bae59c70058b6f5c2c944d138f732f60`。
- Pretest Git：`1e71b8abffb34eccd297747d39827135a627107a`。本人直接讀取、解壓並核對本機 Git commit/tree/blob 的 SHA-1，不執行 Git 指令；selection、manifest、protocol、ASR selection 四份目前檔案與該 commit 的原始 blob 完全相同。actual test 綁定同一 source commit 及 selection 指紋。

`promotion-receipt.json` 與 `candidate-final.diff` 證明：只有 `/approved`、`/reviewed` 兩個 Boolean 由 false 改 true；所有其他 JSON 值遞迴相同，整個卡片位元組相同。原始 JSON 排版亦保留，只替換兩個字面值。

## 實際核對與結論

本人寫的 `audit_local_evidence.py` 實際完成 1,881 項資料／指紋／發布契約核對，全數通過。這個數字是程式核對項數，不是受試者、獨立試驗或回答成功數。輸出為 `local-evidence-audit.json`，SHA-256 `29fce908366c824c8952bfa50ce536b60fb7786070cb2b9ee525e5bd81dfd496`。

| 固定最後測試用途 | 真實通過／原分母 |
| --- | ---: |
| 照片短描述 | 25/42 |
| 同組照片可見事實 | 58/84 |
| 其中動作 | 3/11 |
| 其中關係 | 21/29 |
| 有／無中文字 | 18/18 |
| 自然中文字完整轉寫 | 8/10 |
| 多區塊及行序 | 1/3 |
| 文字聊天 | 2/13 |
| 問句來源文字後聊天 | 2/4 |
| 真正 ASR 後聊天 | 2/4 |

178 份 LM 原始生成、22 段真人錄音、147 份語義判讀全部保留。147 = 126 照片回答 + 13 文字聊天 + 8 同四段問句的雙路聊天；31 份讀字／presence 題另走固定客觀判準。本人從 `case_decisions` 重算各組分母和通過數，逐項核对 combined 的 147 个原判讀、四位 grader 身分、非空理由及原照片检查欄位；没有改寫任何判讀或以本人的新看圖分數替代原證據。

42 張照片對应 126 份相關回答，不能當成 126 張獨立照片。manifest 原題型支援 11 動作題、29 關係題及 1 題保留前文的測試對話；描述性子集與原分數指紋一致。171/178 份回答最後原始 token 是實際 EOS；另外 7 份均為文字聊天截斷，仍在 13 題分母並判錯。EOS 停止與內容正確分開。原 grader 檔與 closure 聲明為四位未參與訓練與選版的 AI 助理、已逐張看圖，沒有冒充人類研究；本審核者不是這四位 grader，沒有另做語義盲評。

本人從真正 hypotheses 與 manifest 原 references 獨立重算 Levenshtein CER，並核對逐錄音 SHA：全組原樣 112/674、NFKC 去 Unicode 空白 96/661；18 段 FLEURS 是 112/632、96/619；4 段 AISHELL 均是 0/42。仍保留標點、大小寫與簡繁，不使用來源真值替換 ASR 後的聊天輸入。四段聽寫全正確而聊天兩路各 2/4，卡片沒有把兩件事混為一種成功率。

兩次真實訓練原紀錄均 completed：2,077 updates、4,154 row uses、1,605,632 trainable LoRA parameters；卡片的候選訓練數正確。固定驗證顯示候選照片成績提升，但存在截斷與語音後聊天 nonregression gate 不通過，依 pretest selection 保留 BASE。這只是核對既有選版，沒有再選候選。ASR 原比較實際是同 16 段 validation 的 small 117/510、turbo 50/510；換 ASR 沒有改稱 LoRA 訓練成果。

BASE 的 `source:null`、`files:[]`、`adapter_parameters:0` 與實際生成的零 trainable parameters 一致。Qwen 底座不同參數 2,127,532,032、turbo 808,878,080，合計 2,936,410,112；本人從固定模型 safetensors tensor shapes 重算並核對整份權重 SHA 與官方 LFS SHA。兩站不是原生音訊 VLM；沒有合成語音輸出。本包不輸出上游權重、本課 LoRA、optimizer、RNG、圖片或錄音。

卡片的新硬體與預算敘述對上 actual result/core：NVIDIA L4 BF16 LM、CPU ASR、不抽樣、最多 384 新 token、65,536–524,288 pixels，提示與預留新 token 合計不超過 2,048，所以保留 384 時提示最多 1,664。CPU student route 的預設 FP32 由 `student_options` 核對；卡片沒有聲稱 CPU 輸出與 BF16 逐字相同，也沒有把下載 bytes 當最低 RAM/VRAM。

## 公開版本與實讀授權

本人新發出的請求不帶 Authorization header。兩份固定模型 API 與 README 均實際 HTTP 200；API 返回完整指定 SHA、`private:false`、`gated:false`、`disabled:false`。README 真正的原始 frontmatter 分別為 `license: apache-2.0` 與 `license: mit`，与官方 metadata 相符。兩份固定 commit 均沒有獨立 `LICENSE` 檔，所發請求真實 404；本結論以發布者在固定版本 README 的授權聲明為據，不虛構授權檔。

| 官方固定來源 | 實讀 README SHA-256 | 結論 |
| --- | --- | --- |
| [Qwen/Qwen3-VL-2B-Instruct @ 89644892e4d85e24eaac8bacfd4f463576704203](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/tree/89644892e4d85e24eaac8bacfd4f463576704203) | `5fc5be1ca9a3910399bd6239ee5086ab5d82a2a59c5d2b00e887a8835cc110e4` | 公開，Apache-2.0 |
| [openai/whisper-large-v3-turbo @ 41f01f3fe87f28c78e2fbf8b568835947dd65ed9](https://huggingface.co/openai/whisper-large-v3-turbo/tree/41f01f3fe87f28c78e2fbf8b568835947dd65ed9) | `aaef74a740faca90fa1899c4233ebe17f2093b9846d0acf16d9131bf650e9585` | 公開，MIT |

本人以官方 `blobs=true` API 逐檔核對全部 23 個模型／配套檔的大小和 SHA（普通檔使用官方 Git blob SHA-1，權重使用 LFS SHA-256），總量正是 5,889,111,977 bytes；計數不含兩份 `.gitattributes`。這是固定模型 artifact 指紋核對，沒有重新產生模型回答。

本專案 `LICENSE` 的完整 MIT 實質文字與 copyright (c) 2026 birdhackor 確實包含於卡片。卡片明確限定頂部 MIT 指本包自行撰寫的說明與版本證明；Qwen Apache-2.0、Whisper MIT 與各資料的授權分別保留，不把全體模型或素材改標成專案 MIT。

## 訓練來源與署名

本人核對 DATA、manifest 的全部 source/annotation fingerprints、1,513 個所選檔案的完整大小與 SHA、實際 NOTICE/ATTRIBUTION 與既有來源記錄，沒有改資料或 gold。

- DOCCI：新讀官方網站確認 Google LLC 的圖片／標註 CC BY 4.0；另讀固定 HF README 與 [原論文 v1 §2.1](https://arxiv.org/html/2404.19753v1)，確認圖片由 Jason Baldridge 及其家人拍攝。論文來源 SHA-256 `bf2861b7019d21f8bf2ca2669411db6bfa5dc30bb51e24a188a16463d365524c`。本課 509 張照片的 439/28/42 split 與公開視覺相似群組一致，未宣稱群組是已證明的拍攝場次；AI 繁中改寫身份與修改告知保留。
- NVIDIA：新讀來源固定 commit `69696a1cc543ef3a0f8e9892a89c17293e915263` 的實際 README，CC BY 4.0；827 個文件裁切都只用于 train。來源 README SHA-256 `4e39811b7105c1101b472917fde0a1c5e0a23d244e100954f4eb17128315d962`。
- Commons：逐一從全部 70 張來源保存的官方 API extmetadata 核對作者、授權及限制與來源清單；56 張 CC BY 4.0、9 張 CC BY 3.0、5 張 CC0，另有 62 個訓練裁切。這是獨立核對保存來源，不冒充當天重抓全部 70 個網頁。每張來源 URL、原作者欄、license URL、修改及原 API 回應 SHA 均記錄於 local evidence audit。`commons-10979995` 的台灣彩券招牌另有 `Restrictions=trademarked`，已保存；CC BY 3.0 是照片著作權授權，不聲稱提供商標授權。卡片沒有重散佈該圖或主張商標權。
- OASST2：新讀固定 commit `179dd21fc55192153d94adb0e0ce8f69e222bf75` 的官方 README，Apache-2.0，SHA-256 `52bd472cf3cae04a4b21ecd5316f929741c47d991d4763b041c3a94de072dc31`。原文、NOTICE、修改 ledger 與另寫 80 題的作者來源保留；同對話樹不跨 split。
- FLEURS：新讀固定 commit `d7c758a6dceecd54a98cac43404d3d576e721f07` 的官方 README，CC BY 4.0，SHA-256 `878990a8b3832046622035b1b2e4f1effa138176876372fd5a2365d0a469f14c`；manifest 的 30 段只作聽寫，不虛構助手真值。
- AISHELL-1：本人新讀 [OpenSLR 33](https://www.openslr.org/33/) 原頁，其 corpus License 欄確實是 Apache License v.2.0，來源 SHA-256 `072e77c5f5a401eca59f7ba0c6608a4c6720f6c67227ab53c37f5745d7b019d9`；8 段既有真人問句、來源逐字稿、移除分詞空格及本課回應判準各有記錄。

所有本課來源家族保持 split disjoint，但未知底座預訓練與公共素材重疊不能排除。卡片沒有誇大為自然街景、方言、自由對話或普遍安全性成績；小量、相關題目、3 題行序、1 題前文與 4 段問句的限制都明寫。

## 發布與學生下載契約

本人閱讀 `scripts/modal_natural.py` 的 `validate_release`/`release_remote` 及 `scripts/fetch_natural_release.py`：前者在 BASE 只接受空來源與空 private file allowlist，發布時自己生成 README 與 release-provenance；上游以 token=False 核對公開 immutable pins。候選 false flags 会被真正的本機 validator 拒絕；完成本審核后同內容 true flags 通过其本機契約。

學生用的 public-release manifest 是發布後另一份 receipt：必須有完整 immutable 公開 commit、README/provenance 的正確 SHA/size、reviewed/anonymous flags、依賴版本、Python 3.12、runtime limits、兩份核心 source code SHA。BASE 会使用 `adapter=None`。fetch/verify 嚴格檢查固定 prefix、精確檔案集合、provenance 與檔案指紋，不將這份 approval 的 `files:[]` 混當學生下載清單，也不插入私有訓練結果。

本審核只批准 concrete artifact，**沒有實際發布 HF 或驗證新公開成品的匿名下載**。root 後續仍須執行已授權發布、匿名回讀、產生正式 public-release manifest 並完成學生指引／實際選定 CPU 試用記錄。這些是後續操作驗收，不是此處未完成的卡片事实修正。

## 修正邊界與原稿

原 8,923-byte 候選與 8,361-byte 卡片的 exact bytes 保留在 `initial-read-snapshot/`；它們由本人先閱讀，snapshot 寫入時因作者并發更新而以作者保存的 pre-scope-correction 原 bytes 補存，此取得範圍明寫在 `initial-read-receipt.json`。原候選 SHA `c4577293fc36915b4e77cdf62a28fbc8fa3b77838bdb123dab5d60dd8ed2ae16`、原卡片 SHA `bdd530b231847dc4e40f29cdbdd590cb928b2ef0b052f06182d5499892c22fda`。

本人真實讀到的中間稿 `ad26af65e17a84e0872b332bd413cc6d3f30eb4be4a3ef9689f7da09c415c8c0`（卡 `d95ff4ed93c9358286f2b11718476657025d131a645096d78b50b07f88ceafa9`）保留在 `token-budget-issue-snapshot/`。本人報告其中「總輸入預算 2,048」與 actual core 的 `input_tokens + max_new_tokens <= max_tokens` 不一致，作者自行修正；我沒有修改 card、runtime、scorer、data、gold 或 selection，也未批准中間稿。最後重新完整讀取、核對 c529 最終候選後才 PASS。

完整新 source responses 只放本目錄被 `.gitignore` 排除的 `research/`；公开內容為本人的分析、必要 URL、短授權事實與來源完整 SHA。未新增整本第三方文件到 Git。新來源請求的真 status、UTC retrieval time、URL、bytes、SHA 收在 `upstream-source-receipts.json`，其完整 SHA-256 `913eed41458a03b16203a86c488514ff66acf69cd57c5bb84a53e214a56df815`。
