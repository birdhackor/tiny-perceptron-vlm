# 小小感知機：教學大綱與設計理由

這份大綱保留教學設計與研究取捨。目前有20章與A／B／C三條支線，共287份逐課Notebook；加上閱讀指南、暖身、訓練操作與名詞頁，共313個編號小節。從[教材入口](../course/README.md)開始閱讀，正文、圖解與同節Notebook使用同一份來源。

第19章主線v2已完成從隨機初始化的文字、視覺、OCR、語音及接頭訓練，MoE共5,447,107參數，Dense共2,288,067參數。兩版使用同一固定資料包的28,876題訓練、2,435題validation；凍結公開權重後，各做一次3,734題最後測試，仍未通過全部原定能力判準。完整成績與任務範圍見[19.12](../course/chapters/19.md#19.12)及[固定結果](selftrained/results/v2-final-public-results.json)，試用見[公開CPU操作](selftrained/v2-public-cpu-commands.md)，自己重做見[本機訓練指引](selftrained/TRAINING.md)。最後MoE段完成4,000步、選第1,000步；Dense段完成10,000步、選第1,000步，選定步數不等於總完成步數。

v2的MoE pretrain、MoE SFT、MoE joint與Dense joint四組推論輸出已公開，共16個配對檔案、MIT授權，固定在[HF revision 979cdf…](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/979cdfacc588ad0536f1c64fff96f264571cf054/selftrained/v2)。公開safe權重不含optimizer／RNG／sampler；重做主線接自己的完整checkpoint，不需要作者私人Volume。

既有局部實驗和舊合成整合也保留：30組正式實驗、CUDA Flash補驗、有限候選PPO與舊同底座MoE的配置、成績及限制在[歷史實驗紀錄](course-experiments/README.md)。舊[部署評估](course-experiments/results/capstone_deployment.json)中的joint與DPO各78／90題，舊[Dense學生](course-experiments/results/capstone_student.json)的示範／蒸餾各62／90與61／90題；這些是局部機制和歷史輔助證據，不是v2最後測試。[各章實驗清單](course-experiments/public-models.json)保留原30組／120份權重，[舊整合清單](course-experiments/capstone-public.json)保留8份階段／量化檔與3份Dense學生檔，各沿用自己的固定版本與指紋。

第20章另有[固定公開清單](natural-assistant/v4/public-release.json)，本版指定Qwen3-VL-2B-Instruct底座與Whisper-large-v3-turbo，不加LoRA修正。先依[20.2](../course/chapters/20.md#20.2)試用，操作與資源見[學生指引](natural-assistant/v4/STUDENT.md)，能力與限制見[20.13](../course/chapters/20.md#20.13)。它延伸已有上游能力，與自行訓練主線各自驗收。

共同基礎：**猜下一個字 → 理解上下文 → Attention → 可訓練、可驗證的 Dense Transformer → 文字對話**。先用字元 tokenizer，BPE 可回讀；之後按問題進入行為、多模態、架構、效率與壓縮單元。

**容易理解是最高優先度。** 每個小節只處理一個主要概念，改動一小段程式，並留下一個能觀察或驗證的結果。先建立直覺和簡單基準，再因具體問題引入後續設計；歷史背景放在對應概念旁。可以回到小模型、切換 checkpoint 或使用獨立實驗；章節排序依概念依賴與理解難度安排。

排序原則：驗證從第一次訓練開始；先在7.11看到同一模型接續對話訓練，再在7.17–7.18比較一般文字、示範、偏好與回饋，解釋常見預訓練／後訓練分工。之後用具體範例理解個性、遵循與安全；圖片和音訊接在已有的文字介面上。第13章先比較同題兩篇回答，再選擇直接偏好更新，或先教評分員、理解PPO的分步路線。現代Dense與MoE放在相鄰的架構單元，效率、量化與蒸餾各自從小實驗入手。

第19章最後把已學零件接到同一個有限任務助理：先看試用問題，再追蹤資料、父權重、預訓練、SFT、模態入口、聯合任務、工具與交付。前面章節仍可換模型或跳讀；本章才要求逐階段核對同一底座的來源。採MoE是為了實際展示選專家與整合，不假定它在小型硬體上必然比Dense省時或省記憶體。全部專家仍要儲存，路由開銷也要量測；v2的top-2 MoE用兩個FFN，Dense用一個，最後目標與選定歷史也不同，不能當成同參數、同計算的受控比較。量化、蒸餾及DPO的舊小世界分支保留各自配方與結果。

參考近兩年公開課後，補入資料去重、評估有效性、微調遺忘、圖文對比學習與視覺 token 預算；另外提供上下文／RAG、工具使用、推理驗證三個跳讀單元。來源與查證限制見 [近期公開課對照](public-course-review-2025-2026.md)。

再依實際重現者的公開提問，補強資料／梯度流程、UTF-8、mask／shift、有效監督與截斷、LR 搜尋及合成 OCR。既有概念使用更細的 tensor 表、單步觀察與故障練習；來源見 [學習者回饋對照](learner-feedback-review.md)。

本輪四個角度的獨立審閱後，保留大章順序，改用小節級首次閱讀路線；先看到計數生成、單例 SFT 與模態任務成果，再按問題回讀比較和除錯。新增詞表輸出層與圖像／音訊共同使用的最小驗收，補齊跳讀資產的能力前置。取捨見 [大綱審閱與修訂理由](curriculum-review-v0.8.md)。

## 1. 參考專案與採用的教學策略

本稿直接檢閱以下版本的文件與原始碼，未執行它們的完整訓練：

- nanochat：`92d63d4e8bb4df75c3b71618f31ddde2378b2bcd`。
- MiniMind-V：`1862b633fc082a723e78dbead9777545f09d960c`。
- MiniMind：`f659b55761b754d306bd140573493a6543cafd7f`。
- Hugging Face Smol Course：`f445ae5d9dd83f355ce6a2b1c6c71b588c8e1541`。
- PKU Safe RLHF：`e8cca16665ef2340ac92c6514f05519310251581`。
- PyTorch TorchAO：`a701b6a6058720c21f95908b7ae4a24bf0cae1b6`。

規劃時另檢閱 Stanford CS336 2025／2026、MIT 6.S191 2025／2026、Harvard AC215 2025／2026 的官方公開課綱／教材，以及李宏毅 2025／2026 課程的筆記與作業副本。當時大學官網與 YouTube 的直連受代理限制，無頭瀏覽器實測亦未改善；台大資料列為鏡像證據，Harvard 2026 尚未授課的單元列為當時預定內容。未將影片入口視為已觀看或核實上傳日期；這段研究紀錄也不代表那些來源的現況已重新查證。

| 觀察 | 本教學的調整 |
| --- | --- |
| nanochat 將 tokenizer、pretraining、evaluation、SFT、inference 串成完整流程。 | 每階段都有可執行成果，新增獨立的文字 SFT 章，再進入多模態。 |
| nanochat 的評估同時包含生成樣本、BPB 與任務分數；tokenizer 另有壓縮率比較。 | 早期固定提示詞與測試資料；換 tokenizer 時加入 BPB，避免直接比較不同切詞下的平均 token loss。 |
| nanochat 提供 CPU 示範，並以 depth 作為主要規模旋鈕。 | 提供小型預設配置，列出參數量、資料量、步數與資源需求；讀者初期只改一個變數。 |
| nanochat 的成熟模型已整合多項架構與效能設計。 | 先完成可讀的初始 Transformer；第 14 章回到小模型比較單項架構選擇，第 16 章再由量測引出效能設計。 |
| MiniMind-V 從既有 LLM 延伸，新增視覺編碼器、投影層與圖片佔位符替換。 | 先完成文字模型，再逐步引入視覺特徵、projector 與 embedding 序列組合；顯示每一步的 tensor shape。 |
| MiniMind-V 的資料程式明確產生 assistant 區段的 labels，其餘位置忽略 loss。 | 先在文字 SFT 中教會 loss mask，再將相同原則沿用到圖片與音訊。 |
| 此版 MiniMind-V 可直接 SFT；projector 預對齊選配，SFT 預設解凍 projector 與 LLM 首末層。 | 先用小實驗分辨對齊與問答的目的，再比較兩階段、直接 SFT 與不同凍結策略。 |
| MiniMind-V 保留純文字資料以維持語言能力。 | 多模態訓練前後都跑固定文字測試，再引入文字／多模態資料混合。 |
| MiniMind 有身份／領域 LoRA 範例與手寫 DPO；Smol Course 將指令微調與偏好對齊分開教。 | 第一次文字 SFT 就建立行為基準；先用理想回應範例教風格、誠實與安全，再以獨立單元比較偏好訓練。 |
| PKU Safe RLHF 分開建模 helpfulness 的 reward 與 safety 的 cost。 | 行為評估分別報告有用性、指令遵循、誠實與安全，並檢查過度拒絕；第13章用有限候選教評分員與PPO，完整逐步生成語言模型的PPO留作延伸。 |
| nanochat 的 GPT 採 Dense FFN；MiniMind／MiniMind-V 可將同一位置的 FFN 換成 routed experts。 | 先完成 Dense 基準，再獨立教 MoE 的 router、top-k、負載平衡與成本比較，保持其他模組一致。 |
| MiniMind 分開介紹教師生成答案的黑盒蒸餾與 CE + KL 的白盒蒸餾，並提供手寫訓練入口。 | 先重用 SFT 模仿教師回答，再逐步引入 soft targets、temperature 與 KL；另外明確選擇較小的學生架構。 |
| TorchAO 分開處理量化數值、低位元儲存格式與執行 kernel，QAT 也有 prepare／convert 階段。 | 從手寫數值實驗到實際儲存，再量測相容後端；分開報告模擬誤差、檔案大小、執行記憶體與速度。 |

常見實務做法必須連同模型規模、資料、硬體與量測方式一起說明。nanochat 使用 ReLU² MLP 和未共享的輸入／輸出權重；MiniMind-V 的預設使用 SiLU gating 和共享權重。這些差異適合做有條件的比較。

MiniMind-V 使用已訓練的 MiniMind LLM 與凍結的 SigLIP2 視覺編碼器。我們的主線自行訓練小型文字與視覺模型，用可控制的玩具任務學習原理；預訓練編碼器可列為後續自然圖片延伸單元。

本稿區分三類目標：**模態對齊**讓視覺／音訊特徵能被文字模型使用；**行為對齊**調整風格、指令遵循、誠實與安全；**架構比較**研究 Dense／MoE 等計算方式。風格與遵循能力各自評估，預設可生動有想法，遇到明確格式或限制時仍可靠地完成要求。

縮小部署成本則分兩條路：**量化**以較低位元表示權重等數值，通常保留原本的參數數量；**蒸餾**由教師提供學習訊號，配合較小學生架構減少參數與計算。MoE讓每個文字單位只啟動部分專家，LoRA減少微調參數，它們的資源效果各自量測。量化和蒸餾可組合，最後同時檢查品質、儲存與實際執行成本。

## 2. 教材呈現方式

**主教材採Zensical網站，正文旁提供同一小節的Notebook下載與Colab入口。完整模型、資料工具與長時間訓練放在Python模組和腳本。**

| 載體 | 用途 |
| --- | --- |
| 短 Notebook | 一小節一個主要問題；逐步看 tensor、曲線、attention 圖、圖片 patches 與聲音頻譜。 |
| Python 模組 | 維護完整模型與共用工具；各階段保留清楚可讀的實作版本。 |
| CLI 訓練／評估腳本 | 可重現的完整流程、checkpoint、長訓練及 CPU／GPU 配置。 |
| Zensical網站／Markdown來源 | 課程索引、完整解說、圖解與實際短程式結果；Notebook從相同正文產生。 |
| 後段推論 demo | 上傳圖像／音訊或輸入文字，體驗已完成的模型。 |

Notebook 的教學核心程式會直接呈現。進入完整模型後，明確指出對應的 Python 檔案與本節改動；相同版本的共用模型維持單一來源。每份 Notebook 必須能從新 kernel 由上到下執行。需要權重時，明示使用讀者自行訓練或課程提供的哪一個 checkpoint。

Notebook 首跑指南涵蓋開啟介面、選對專案 kernel 與 restart-and-run-all。目前 `notebook` dependency group 已包含 ipykernel、matplotlib、JupyterLab 與 nbclient；安裝及啟動步驟見[W.1](../course/first-steps.md#W.1)。網站各節提供Colab入口；本地CPU kernel驗證與Colab雲端實機驗證分開記錄，不能互相代替。

正文從眼前問題與例子開始；執行入口、安裝和完整配方放在需要時可查的操作頁。教材版本與練習副本分開，採用同一套內容來源維持網站與Notebook一致。

每小節用連貫段落帶讀者看一個問題、追蹤一個具體例子，再讀程式與結果。可以先預測、手算或看圖，但依當節概念選擇合適的說明順序，不套固定六步或每節相同的小標題。練習明定改哪裡、先預測什麼及如何核對；實際編寫與逐節回查依[讀者審閱標準](editorial-guide.md)。

「只改一個變數」用來理解機制。研究方法品質時，另宣告匹配的是參數、有效訓練 tokens、估計計算量或實測時間；為匹配預算而調整寬度、步數或 learning rate，須一併列出。結構／梯度／數值觀察可完成概念驗收，品質比較允許無差異或退步。

難點的固定呈現方式：

| 難點 | 必須看得見的內容 |
| --- | --- |
| 公式接程式 | 先單例／單 head；列軸名稱、shape 與一個手算元素，使用非方陣暴露方向錯誤。 |
| 一筆資料到 loss | raw record → render → token IDs → X/Y → masks → batch → logits／loss；每節只展開當前一步。 |
| 一次更新 | 數值擾動／chain rule → autograd → 清零／累積 → step；列參數梯度及權重變化。 |
| 自己的資料與配方 | 展示少量原始、保留與淘汰例子及理由；記錄各來源的實際筆數和有效監督 tokens。 |
| 對齊／凍結階段 | 列已有能力、可訓練參數、optimizer 收錄、grad norm、權重變化及本階段任務。 |
| 失敗定位 | 給一個成功例、一個只破壞一項設定的例子；讀者先用觀察定位，再修正。 |

文字訓練的一筆資料與一次更新；各節只展開當前一步：

![文字樣本先取得輸入X與下一步答案Y；可讀前文與指定計分位置分開，代價再用於一次權重更新](../course/figures/curriculum_learning_flow.svg)

X是模型收到的輸入，Y是對應的下一步答案。可讀前文的遮罩控制模型能看哪些位置；指定計分的位置則決定哪些答案要算入代價。圖中示範一次不累積舊梯度的更新；刻意累積梯度時，清零時機另外安排。

成功判準按實驗目的分別標示：程式可執行、資料／梯度有效、玩具 held-out 任務達標、較大自然語言任務品質。CPU smoke 可產生無意義文字；完整能力實驗另外明示資料與預算。

課程索引按下表區分「主題初讀」與「按需回讀」，完整訓練／硬體延伸另行標示。章節編號用於定位主題，首次閱讀按路線進行。初期保留直白預設值；理解單一問題後才加入配置、加速後端與硬體差異。每個主題選擇最容易看出效果的模型與資料。

### 閱讀與實作路線

順序讀者可按小節編號前進；跳讀者用[閱讀指南](../course/README.md#R.2)選主題。需要重做某個實驗時，再查[訓練操作](../course/training.md)的資料、父權重與執行步驟。這些路線只維護一份，避免大綱與正文各自出現不同的先後次序。

各節應在首次使用處提供當下需要的說明；前置連結只幫跳讀者回查，不取代任務、材料與答案的介紹。介面擴充或資料變更時，先說明本節新增的一件事。

## 3. 目前教材與新增主題

實際小節標題以[全課索引](../course/lessons.md)為準；章節目的與閱讀路線見[閱讀指南](../course/README.md)。這裡記錄設計取捨，不再維護一份容易與正文分歧的完整標題清單。

| 學習問題 | 放置位置 |
| --- | --- |
| 為什麼先大量讀文章，再練習當助理？ | 7.17–7.18；13章的回饋、PPO與DPO |
| 一個位置如何學不止一個未來token？ | 7.20–7.22的MTP；16.14的草稿與驗證 |
| 同一份題庫還能接受不同教法嗎？ | 18.1–18.4的原標籤、教師文字與教師分布；18.10的品質與成本 |
| 回答被截斷，和模型沒讀懂，是同一件事嗎？ | 7.19的輸入／輸出預算；8.14–8.16的範圍、格式與停止 |
| 為什麼有些圖片特徵能命名物件，卻無法讀字？ | 10–11章的空間、任務與資訊；11.17–11.18的有限商品、中文區域 |
| 真人說話，怎麼進入同一段聊天？ | 12.15–12.16的有限語音需求與共同紀錄 |
| 擴大長度、改位置、少算一點，各解決什麼？ | 14.7–14.10的context與位置延伸；16.12–16.13的視野和計算 |
| 長紀錄裝得下，還會漏資訊嗎？ | A.8–A.10的位置、資訊組合與更新 |
| 模型何時該選工具？ | B章的選擇、呼叫、結果回填與第二次回答 |
| 教過的零件能組成真正的小助理嗎？ | 19章的自行訓練整合；20章另作成熟模型延伸 |

主線成品的神經權重從隨機初始化開始；有限任務允許較小的模型，不能用成熟模型的能力代替驗收。三類模態共享語言核心與對話紀錄，但各自需要適合的素材、特徵序列與訓練訊號。v2視覺限三類服飾和公開兩格位置，OCR限12個已知繁體字與一個提供的連續1–4字區域，Sans／Serif兩種字型都已見於訓練。語音限地址、App與卡片三種銀行客服主題，直接由聲音特徵產生文字回答，不宣稱開放式逐字聽寫。錄音切分不等於說話者隔離，speaker ID未知。具體資料與留出規則見[固定資料清單](selftrained/v2-manifest.json)與[19.12](../course/chapters/19.md#19.12)。

## 4. 實作與驗收原則

教材先教成熟知識，再以局部機制例子建立直覺。原論文和官方文件支撐方法的用途與條件；小模型在少量資料上的成績，只能說明那次實作，不能推翻或保證一般效益。

每節使用一個清楚任務，先展示材料、答案和必要圖解，再引入程式與術語。完整配方、資料來源與逐題結果放在需要時可查的操作和證據頁。保留有助理解的錯例，刪除沒有教學用途的開發試錯。

易讀性、正確性與前後銜接分三輪獨立審閱；工程、訓練與能力驗收是之後的另一階段。文字、圖解或排版更新不觸發完整重訓，實作或能力宣稱改變才檢查相應任務。來源、版本與指紋可查，審閱者要實際看過內容和所需圖解，不能只換一個報告hash。

## 5. 原始碼參考

### 近期公開課

課程年份、官方／鏡像證據、影片入口與固定版本來源見 [近期公開課對照](public-course-review-2025-2026.md)。本輪新增小節的主要來源為 Stanford CS336 的 evaluation／data／multimodality，MIT 的 post-training／style evaluation／CV 分項實驗，Harvard 的 RAG／Agents 課綱，以及李宏毅的遺忘／推理／工具教材線索。

### 公開學習者回饋

具體提問、重現日記、日期與證據限制見 [學習者回饋對照](learner-feedback-review.md)。採用 nanochat、MiniMind／MiniMind-V、CS336 作業與 LLMs-from-scratch 的學習難點和觀察方法；歷史 bug、特定超參數與有限樣本的因果猜測另行標示。

### nanochat

- [README：完整流程與單一規模旋鈕](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/README.md)
- [runs/runcpu.sh：CPU 示範流程](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/runs/runcpu.sh)
- [runs/speedrun.sh：tokenizer → pretraining → evaluation → SFT](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/runs/speedrun.sh)
- [scripts/tok_eval.py：tokenizer 壓縮率與還原檢查](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/scripts/tok_eval.py)
- [nanochat/loss_eval.py：BPB](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/nanochat/loss_eval.py)
- [nanochat/tokenizer.py：對話渲染、loss mask 與可視化](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/nanochat/tokenizer.py)
- [nanochat/gpt.py：模型結構選擇](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/nanochat/gpt.py)
- [nanochat/engine.py：KV cache、prefill 與生成](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/nanochat/engine.py)
- [nanochat/flash_attention.py：FA3／SDPA 後端](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/nanochat/flash_attention.py)

### MiniMind-V

- [README：架構、資料與目前訓練策略](https://github.com/jingyaogong/minimind-v/blob/1862b633fc082a723e78dbead9777545f09d960c/README.md)
- [model/model_vlm.py：視覺編碼器、projector 與佔位符替換](https://github.com/jingyaogong/minimind-v/blob/1862b633fc082a723e78dbead9777545f09d960c/model/model_vlm.py)
- [dataset/lm_dataset.py：對話資料與 assistant-only labels](https://github.com/jingyaogong/minimind-v/blob/1862b633fc082a723e78dbead9777545f09d960c/dataset/lm_dataset.py)
- [trainer/trainer_utils.py：凍結策略與 checkpoint](https://github.com/jingyaogong/minimind-v/blob/1862b633fc082a723e78dbead9777545f09d960c/trainer/trainer_utils.py)
- [trainer/train_pretrain_vlm.py：projector 對齊入口](https://github.com/jingyaogong/minimind-v/blob/1862b633fc082a723e78dbead9777545f09d960c/trainer/train_pretrain_vlm.py)
- [trainer/train_sft_vlm.py：多模態 SFT 入口](https://github.com/jingyaogong/minimind-v/blob/1862b633fc082a723e78dbead9777545f09d960c/trainer/train_sft_vlm.py)
- [scripts/web_demo_vlm.py：Gradio 推論展示](https://github.com/jingyaogong/minimind-v/blob/1862b633fc082a723e78dbead9777545f09d960c/scripts/web_demo_vlm.py)

### MiniMind：行為微調、DPO 與 MoE

- [README：LoRA 身份／領域資料與後訓練介紹](https://github.com/jingyaogong/minimind/blob/f659b55761b754d306bd140573493a6543cafd7f/README.md)
- [trainer/train_dpo.py：手寫偏好 loss 與 reference model](https://github.com/jingyaogong/minimind/blob/f659b55761b754d306bd140573493a6543cafd7f/trainer/train_dpo.py)
- [model/model_minimind.py：Dense／MoE FFN、top-k routing 與 auxiliary loss](https://github.com/jingyaogong/minimind/blob/f659b55761b754d306bd140573493a6543cafd7f/model/model_minimind.py)
- [trainer/train_distillation.py：教師／學生、temperature、CE + KL 與 loss mask](https://github.com/jingyaogong/minimind/blob/f659b55761b754d306bd140573493a6543cafd7f/trainer/train_distillation.py)

### Hugging Face Smol Course：指令與偏好對齊

- [Instruction tuning](https://github.com/huggingface/smol-course/blob/f445ae5d9dd83f355ce6a2b1c6c71b588c8e1541/units/en/unit1/1.md)
- [SFT 與行為調整](https://github.com/huggingface/smol-course/blob/f445ae5d9dd83f355ce6a2b1c6c71b588c8e1541/units/en/unit1/3.md)
- [Preference alignment](https://github.com/huggingface/smol-course/blob/f445ae5d9dd83f355ce6a2b1c6c71b588c8e1541/units/en/unit2/1.md)
- [DPO](https://github.com/huggingface/smol-course/blob/f445ae5d9dd83f355ce6a2b1c6c71b588c8e1541/units/en/unit2/2.md)

### PKU Safe RLHF：安全目標與進階流程

- [README：helpfulness reward、safety cost 與偏好資料](https://github.com/PKU-Alignment/safe-rlhf/blob/e8cca16665ef2340ac92c6514f05519310251581/README.md)

### PyTorch TorchAO：量化、儲存格式與 QAT

- [README：量化與推論／訓練入口](https://github.com/pytorch/ao/blob/a701b6a6058720c21f95908b7ae4a24bf0cae1b6/README.md)
- [Quantization overview：數值、packing 與 kernels](https://github.com/pytorch/ao/blob/a701b6a6058720c21f95908b7ae4a24bf0cae1b6/docs/source/contributing/quantization_overview.rst)
- [QAT：fake quantization、prepare 與 convert](https://github.com/pytorch/ao/blob/a701b6a6058720c21f95908b7ae4a24bf0cae1b6/docs/source/workflows/qat.md)
