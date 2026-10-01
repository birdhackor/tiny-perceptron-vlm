# 近期公開課程與教學規劃對照

研究日期：2026-10-01（Asia/Taipei）。範圍：2024-10-01 至 2026-10-01 公開的課程版本與教材。由三個 subagents 分別查台大／李宏毅、Harvard／Stanford、MIT，再交叉核對官方原始碼與現有大綱。

## 1. 查證範圍與限制

- Stanford、MIT、Harvard：讀取官方 GitHub 的課站原始碼、課綱、講義與 Notebook；來源固定到 commit。課程年份以官方內容核實。
- 台大／李宏毅：官方課站入口及 YouTube 連結由多份課程筆記、作業副本交叉確認；這些是非官方鏡像。2026 學生副本提供教材線索，尚未直接核實官方完整課綱。
- 網路代理對大學官網與 YouTube 回傳 403。另以環境內 Playwright + Chromium 無頭瀏覽器實測台大、Stanford、MIT，仍為 `ERR_TUNNEL_CONNECTION_FAILED`。
- 已閱讀可取得的課綱、教材與官方講者摘要，未觀看 YouTube 影片或取得完整影片逐字稿。影片入口的來源層級與日期證據分別列出。
- 「近兩年」限制的是課程版本；新課教到 CLIP、RNN 等較早方法仍可採用，但不將它們稱為近兩年發明。

## 2. 課程與年份證據

| 課程 | 年份／日期證據 | 已取得內容 | 查證層級 |
| --- | --- | --- | --- |
| 李宏毅 ML 2025 Spring | 筆記索引明列 2025 春；另一份作業副本列 2025 年 2–6 月日期。 | Agent、LLM 內部機制、預訓練／對齊、遺忘、推理、評估、模型編輯／融合、語音 LM。 | 非官方鏡像；原課站與影片日期待直接核實。 |
| 李宏毅 Introduction to GenAI and ML 2025 Fall | Notebook 副本明載課名、2025 Fall 與原課站入口。 | 文字接龍、chat template、多輪、Tool Use、紅樓夢 RAG 作業。 | 非官方作業副本；不視為完整官方課綱。 |
| 李宏毅 ML 2026 Spring | 學生修改版多份 Notebook 明載 ML2026 Spring。 | 微調遺忘、模型編輯／融合、Flow Matching、Spoken LM 等線索。 | 補充線索；未核定完整課綱或原片年份。 |
| Stanford CS336 Spring 2025 | 官方封存頁明列 Spring 2025，課表 4/1–6/3。 | 從零 tokenizer／Transformer、資源、scaling、資料、評估、SFT／RLHF、推理 RL；作業公開。 | 官方課綱與教材。 |
| Stanford CS336 Spring 2026 | 官方首頁與講義 repo 明列 Spring 2026，課表 3/30–6/3。 | 明列資料來源、filtering／dedup／mixing／synthetic data、RLVR、multimodality。 | 官方課綱、講義、影片入口。 |
| Harvard AC215 2025 | README：draft 2025/4/2，update 2025/6/16；秋季課表列 10/7 RAG、10/9 Agents、10/14 fine-tuning、10/21 data labeling／versioning。 | LLMOps、RAG、Agents、資料版本、微調、壓縮、部署。 | 官方公開課綱；尚未確認免登入的完整公開錄影。 |
| Harvard AC215 2026 | README：draft 2026/5/22；課表列 9/15 RAG、9/17 Agents、10/1 fine-tuning。 | latency／throughput／cost 與 monitoring 等規劃。 | 官方課綱草稿；多數連結為 `/not-yet/`，10/29、11/3 內容仍屬研究日之後的預定課程。 |
| MIT 6.S191 2025 | 官方頁：實體授課 2025/1/6–1/10；影片公開日期列 3/3–5/5。 | Sequence、CV、generative、RL、LLM；post-training 摘要明列資料 accuracy／diversity／complexity、SFT、偏好、model merging、評估。 | 官方課站、摘要、分年實驗、影片入口。 |
| MIT 6.S191 2026 | 官方頁列影片公開日期 2026/3/30–5/25；投影片保留 1 月授課日期。 | Sequence、CV、generative、RL、New Frontiers、Three Laws of AI、AI for Science、parallel training；風格微調與評估實驗。 | 官方課站、投影片、分年實驗、影片入口。 |

沒有將舊版 CS50 AI／MIT OCW 錄影因近期網站更新而算作新課。此次未取得 CS224N／CS25 近兩年可讀的第一手課綱，故未把它們列為已核實來源。

## 3. 對現有大綱的補充判斷

現稿已涵蓋 Dense／MoE、SFT／LoRA／DPO、FlashAttention、量化與蒸餾。補充優先處理資料、評估與能力來源的缺口，保留「一小節一個概念、可跳讀、可換小模型」的設計。

| 優先度 | 補充知識 | 近期教學證據 | 教學安排 |
| --- | --- | --- | --- |
| 高 | 最終 test 與 validation、完全／近似去重、資料污染 | CS336 2026 Lecture 12／14。 | 第 5 章，使用數百段合成文字；先 hash，再用字元片段集合的 Jaccard。 |
| 高 | SFT 錯標與覆蓋、災難性遺忘、replay | MIT 2025 post-training 摘要；李宏毅 2025 遺忘課索引與 2026 HW 線索。 | 第 7 章先 A→B 觀察 A 退步，再加入舊樣本，固定預算比較。 |
| 高 | 評分者可靠性、位置／長度偏差 | MIT 2025／2026 Lab 3；CS336 2026 Lecture 12。 | 第 8 章使用人工 rubric、正負 control 與預錄評分；不用線上 API 當必備依賴。 |
| 高 | 圖文對比學習 | CS336 2026 Lecture 17 的 CLIP→SigLIP→LLaVA。 | 第 10 章可選支線：四組圖片／描述→相似度矩陣→image-to-text CE→雙向目標。使用自行訓練的小 encoder。 |
| 高 | 視覺細節／token 預算與分項評估 | CS336 2026 Lecture 17；MIT 2026 Lab 2。 | 第 11 章以 resize／crop、patch size 和稀少形狀／顏色組合做對照。 |
| 中 | 不確定性與 calibration | MIT 2026 New Frontiers 投影片。 | 第 9 章用小分類器觀察 softmax 信心與正確率；和模型文字自述的信心分開。 |
| 中 | 容量、資料量與計算預算 | CS336 scaling 講義／作業；MIT 2026 parallel training 摘要。 | 第 5 章選修 2×2 小對照；tiny 實驗用來理解取捨，不能宣稱重現大型 scaling law。 |
| 延伸 | In-context learning、RAG、工具使用 | 李宏毅 2025 春／秋教材線索；Harvard AC215 2025／2026。 | SFT 後的獨立 A／B 單元；從本地文字與計算器開始。 |
| 延伸 | 推理步驟、候選取樣、verifier、test-time compute、RLVR | 李宏毅 2025 推理／推理長度／評估索引；CS336 2025 Assignment 5、2026 RLVR。 | 獨立 C 單元：先比較直接回答與中間步驟，再用程式驗證候選；最後才教最小 reward 更新。 |
| 延伸 | 串接式與直接音訊模型 | 李宏毅 2025 語音 LM 索引與 2026 作業副本。 | 第 12 章用預錄文字化結果比較資訊損失，完整語音模型另列成本。 |

值得學習的教學方式：

- 李宏毅近年索引以具體問題切入，例如「為什麼遺忘」「推理需要多長」。借用這種提問方式，每節先展示一個失敗或差異。
- CS336 把資料、資源、評估與模型元件放到同一個研究問題中。借用可對照的作業設計，縮成 CPU 玩具實驗。
- MIT Lab 3 先微調出可觀察的風格，再用已知符合／不符合風格的 control 檢查評分方式。借用實驗結構，保持模型與依賴小。
- Harvard 把成果、資料版本與執行成本說清楚。借用這些驗收方式；Docker、Kubernetes、雲端部署與大量框架仍屬另一種課程目標。

RNN／SSM／Mamba、model editing／merging、AnyRes／tiles、SigLIP loss、diffusion／flow matching 可保留研究選修。它們各有價值，優先度低於上述可直接補足理解缺口的實驗。

## 4. 台大／李宏毅的公開影片索引

以下主題與影片對照出自固定版本的 2025 春課程筆記索引；尚未直接驗證 YouTube 的標題、上傳日期與逐字稿。

| 主題 | YouTube |
| --- | --- |
| 生成式 AI 技術突破與未來 | [影片](https://www.youtube.com/watch?v=QLiKmca4kzI) |
| AI Agent 原理 | [影片](https://www.youtube.com/watch?v=M2Yg1kwPpts) |
| LLM 內部運作 | [影片](https://www.youtube.com/watch?v=Xnil63UDW2o) |
| Transformer 的競爭者 | [影片](https://www.youtube.com/watch?v=gjsdVi90yQo) |
| 預訓練與對齊 | [影片](https://www.youtube.com/watch?v=Ozos6M1JtIE) |
| 後訓練與遺忘 | [影片](https://www.youtube.com/watch?v=Z6b5-77EfGk) |
| DeepSeek-R1 類模型的推理來源 | [影片](https://www.youtube.com/watch?v=bJFtcwLSNxI) |
| 推理過程不用過長 | [影片](https://www.youtube.com/watch?v=ip3XnTpcxoA) |
| LLM 能力評估 | [影片](https://www.youtube.com/watch?v=s266BzGNKKc) |
| 模型編輯 | [影片](https://www.youtube.com/watch?v=9HPsz7F0mJg) |
| 模型融合 | [影片](https://www.youtube.com/watch?v=jFUwoCkdqAo) |
| 語音語言模型 | [影片](https://www.youtube.com/watch?v=gkAyqoQkOSk) |

官方入口：[2025 春](https://speech.ee.ntu.edu.tw/~hylee/ml/2025-spring.php)、[2025 秋](https://speech.ee.ntu.edu.tw/~hylee/GenAI-ML/2025-fall.php)、[2026 春](https://speech.ee.ntu.edu.tw/~hylee/ml/2026-spring.php)。

可追溯的非官方證據：

- [2025 春主題／原片索引](https://github.com/MLNLP-World/MachineLearning2025Spring--Notes/blob/3ec1979138e82b2087a1018c7b91f2db5d8f98b3/README.md)。
- [2025 春作業日期／主題](https://github.com/happyyzy/sol-for-LHY-ML2025-HWS/blob/a1a6b2ca7c1af61906cc93697a0c54b675865411/README.md)。
- [2025 秋 Notebook 副本](https://github.com/GBWen/ntu-hylee-GenAI-ML-2025-fall/tree/1f121800d08466e4c1def6e1c7137e9d16edf5d1)。
- [2026 春學生修改版 Notebook](https://github.com/squarejellyfish/ntu-ml-spring-2026/tree/9065b540561c185c09fb9e78ebf481b9ef4a745c)。

來源排除：`xmy3/hung-yi-lee-ml2025-notes` 的 58 講清單疑似舊課；`fireman333/li-hung-yi-ai-agent-2026` 是 AI 生成筆記，年份標示自相矛盾。兩者均未用來核定近期課綱。

## 5. Stanford、Harvard、MIT 的官方來源

### Stanford CS336

- [2025 官方封存課站](https://cs336.stanford.edu/spring2025/)／[固定版本課站原始碼](https://github.com/stanford-cs336/stanford-cs336.github.io/blob/25d740fd9060cc6613163b5b88ca88a5f64138ff/spring2025/index.html)。
- [2026 官方課站](https://cs336.stanford.edu/)／[固定版本課站原始碼](https://github.com/stanford-cs336/stanford-cs336.github.io/blob/25d740fd9060cc6613163b5b88ca88a5f64138ff/index.html)。
- [官方課站所連結的 2026 YouTube 播放清單](https://www.youtube.com/watch?v=JuoVZkPBiKk&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV)。
- [Lecture 12：evaluation／validity／judge](https://github.com/stanford-cs336/lectures/blob/de53a9f979a6ee35f7d13a5e1aadee5ea1afc58e/lecture_12.py)。
- [Lecture 14：deduplication／mixing／synthetic data](https://github.com/stanford-cs336/lectures/blob/de53a9f979a6ee35f7d13a5e1aadee5ea1afc58e/lecture_14.py)。
- [Lecture 17：CLIP／SigLIP／LLaVA／resolution](https://github.com/stanford-cs336/lectures/blob/de53a9f979a6ee35f7d13a5e1aadee5ea1afc58e/lecture_17.py)。
- [2026 講義庫：scaling、RLVR 等](https://github.com/stanford-cs336/lectures/tree/de53a9f979a6ee35f7d13a5e1aadee5ea1afc58e)。

### Harvard AC215

- [2025 官方課綱](https://github.com/Harvard-IACS/2025-AC215/blob/a0d3ed05aa749aa46c95c70a2f519db0eef2bb30/README.md)。
- [2025 RAG／Agents 課表](https://github.com/Harvard-IACS/2025-AC215/blob/a0d3ed05aa749aa46c95c70a2f519db0eef2bb30/_modules/week-06.md)。
- [2026 官方課綱草稿](https://github.com/Harvard-IACS/2026-AC215/blob/f00978fe6d008935ab87c7fda589b461bec6d236/README.md)。

課綱可公開取得；介紹影片連到 Canvas，其他連結亦未充分證明免登入公開錄影，本輪不把它列為已核實的 YouTube 公開課。

### MIT 6.S191

- [2025 官方課站](https://introtodeeplearning.com/2025/index.html)／[固定版本來源](https://github.com/MITDeepLearning/introtodeeplearning.com/blob/c3841a160a77d336391107cc71fd3b80555bfa22/2025/index.html)。
- [2026 官方課站](https://introtodeeplearning.com/2026/index.html)／[固定版本來源](https://github.com/MITDeepLearning/introtodeeplearning.com/blob/c3841a160a77d336391107cc71fd3b80555bfa22/2026/index.html)。
- [2025 Lab 3：LLM 風格微調與評估](https://github.com/MITDeepLearning/introtodeeplearning/blob/1a076fe1dcebf98bdf70f855e22930a9105467c0/lab3/LLM_Finetuning.ipynb)。
- [2026 Lab 3：LLM 風格微調與 judge controls](https://github.com/MITDeepLearning/introtodeeplearning/blob/535e0862f52c81288258fae0516c40968d07f603/lab3/LLM_Finetuning.ipynb)。
- [2026 Lab 2：資料偏差與分項評估](https://github.com/MITDeepLearning/introtodeeplearning/blob/535e0862f52c81288258fae0516c40968d07f603/lab2/PT_Part2_Debiasing.ipynb)。
- [2026 New Frontiers 投影片](https://github.com/MITDeepLearning/introtodeeplearning.com/blob/c3841a160a77d336391107cc71fd3b80555bfa22/2026/slides/6S191_MIT_DeepLearning_L6.pdf)。

官方課站所列的重點影片與公開日期：

| 主題 | 日期 | YouTube |
| --- | --- | --- |
| 2025 Introduction to LLM Post-Training | 2025-04-21 | [影片](https://www.youtube.com/watch?v=_HfdncCbMOE) |
| 2026 Deep Sequence Modeling | 2026-04-06 | [影片](https://www.youtube.com/watch?v=d02VkQ9MP44) |
| 2026 Deep Computer Vision | 2026-04-13 | [影片](https://www.youtube.com/watch?v=pqIcoskUuWs) |
| 2026 New Frontiers | 2026-05-04 | [影片](https://www.youtube.com/watch?v=ev7cLSd-ySE) |
| 2026 Massively Parallel Training | 2026-05-25 | [影片](https://www.youtube.com/watch?v=UZZD9d9YqnQ) |

具體規劃與小節見 [教學大綱](curriculum.md)。
