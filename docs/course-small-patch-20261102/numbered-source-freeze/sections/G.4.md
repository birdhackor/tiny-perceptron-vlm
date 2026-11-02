## G.4 後段主題先對應到問題

後段名詞很多，下表先把主要方法各對應到一個問題：想少算、想少存、想學較好回答，或想取得模型原本不知道的資料。這些目的可以組合，也可能互相影響；要先知道自己想改善什麼，再選方法。後面幾張表再收偏好學習、多次嘗試與機率判斷，以及第 19、20 章成品的用語。

| 名詞 | 想解決的問題 | 完整講解 |
| --- | --- | --- |
| Dense、MoE（Mixture of Experts，專家混合） | Dense 表示每個位置都用完整的一套 FFN，不依當前字只選部分配方。MoE 想保留多套規則，又讓每個 token 只花少數幾套的計算：準備多組 FFN（每組叫一位 expert），由 router 替每個 token 打分，只留比例最大的 k 位（top-k routing） | [4.4](chapters/04.md#4.4)、[第 15 章開頭](chapters/15.md)、[15.1](chapters/15.md#15.1)、[15.2](chapters/15.md#15.2)、[15.3](chapters/15.md#15.3)、[15.4](chapters/15.md#15.4) |
| 每 token 使用估計、每 token 參數代理 | 報告 MoE 有多大時，要寫明數了哪些部分：每 token 使用估計只數一層 MoE FFN 裡選中的 expert 加 router；每 token 參數代理是全部非 expert 參數加上每層選中 expert 的參數（實驗紀錄欄位 `active_parameter_proxy_per_token`），連整張 embedding 表與位置表都算入。沒被選中的 expert 仍要存著 | [15.10](chapters/15.md#15.10) |
| 思考投入（thinking effort，也叫 reasoning effort）；Auto | 使用者對解題資源的偏好；傳入的請求檔位、服務換成的執行設定與實際耗用要分開記。Auto 是服務先根據輸入替每題選預算 | [7.23](chapters/07.md#7.23)、[7.25](chapters/07.md#7.25)、[19.17](chapters/19.md#19.17) |
| 離線理想上界（oracle） | 多花計算最多能好多少：事後看正確答案，替每題挑答對又最省的檔位；部署時看不到正確答案，所以它不是可以上線的成績。C.3 的已知答案命中率（oracle coverage）是另一個量 | [7.25](chapters/07.md#7.25)、[19.18](chapters/19.md#19.18)、[C.3](chapters/0C.md#C.3) |
| Looped Transformer（循環式 Transformer） | 想少存參數：只保留一份區塊參數，把上一輪輸出的狀態交回它反覆處理。省下的是參數的儲存，每多一輪仍要多執行一次區塊 | [14.11](chapters/14.md#14.11) |
| 權重共享（weight tying） | 想少存一張表：輸出層直接用字的 embedding 表當權重。Looped 讓同一個區塊在各輪共用一組參數，是另一種讓同一份參數用在不止一處的做法 | [14.6](chapters/14.md#14.6)、[14.11](chapters/14.md#14.11) |
| 終止深度（exit depth）、逐輪監督（deep supervision） | 讀出答案的那一輪叫終止深度；一次展開到最大輪數，在多個中間狀態各算代價、再加權成一個總代價，叫逐輪監督 | [14.13](chapters/14.md#14.13)、[14.14](chapters/14.md#14.14) |
| 讀提示階段（prefill）、逐步生成階段（decode） | 找出等待花在哪裡：一開始處理整段提示叫讀提示階段，之後每次只處理剛加入的 token 叫逐步生成階段。這裡的 decode 不是 1.1 把 ID 轉回文字的解碼 | [16.2](chapters/16.md#16.2) |
| KV cache（快取） | 保存每層已算好的 K、V；生成下一個 token 時只替新位置做新的計算，但仍要讀前面所有已存的 K、V | [16.2](chapters/16.md#16.2)、[16.3](chapters/16.md#16.3) |
| 共享材料記憶體（shared memory） | 先只編碼材料得到M，各問題／候選分支唯讀同一份M；它是本課C-v2的共用表示，不必是decoder的prefix KV | [16.15](chapters/16.md#16.15)、[19.23](chapters/19.md#19.23) |
| 循環狀態（recurrent state）、混合模型（hybrid） | 某些層把讀過的資訊累積成可變狀態；Kev-4B混合這類DeltaNet層與完整attention，問題分支須一起隔離KV、卷積和循環狀態 | [16.15](chapters/16.md#16.15)、[20.14](chapters/20.md#20.14) |
| lookahead、首段等待（Time to First Audio，TTFA） | lookahead是決定目前輸出前還要等的未來上下文；首段等待記從請求到第一段實際可播放聲音的時間，和整段完成不同 | [16.16](chapters/16.md#16.16)、[20.16](chapters/20.md#20.16) |
| FlashAttention | 把分數、softmax 與 Value 加權分塊安排，減少大型中間表的保存與搬移，仍得到原來的加權結果 | [16.9](chapters/16.md#16.9) |
| CUDA | 想在 NVIDIA GPU 上計算：CUDA 是 NVIDIA GPU 的運算環境，命令寫 `--device cuda` 的配方需要它 | [16.8](chapters/16.md#16.8)、[操作頁開頭](training.md) |
| max_new_tokens | 這次最多新增幾個 token，不包含已送入的前文 | [7.19](chapters/07.md#7.19) |
| RoPE（旋轉位置嵌入）、位置插值（Position Interpolation） | 把向量的數字兩兩分組、依位置旋轉來表示位置；延長視窗時，先縮放位置座標再算位置資訊，叫位置插值 | [14.1](chapters/14.md#14.1)、[14.8](chapters/14.md#14.8) |
| TMRoPE（Time-aligned Multimodal Rotary Position Embedding） | 時間對齊的多模態旋轉位置表示：把向量分組分配給時間、高度、寬度座標；物理時間和token因果次序仍是兩個不同座標 | [14.19](chapters/14.md#14.19)、[20.15](chapters/20.md#20.15) |
| 滑動視窗注意力（sliding-window attention） | 每層每個位置只直接讀附近固定寬度的位置；較早的資訊可能經其他位置一層層傳過來 | [16.12](chapters/16.md#16.12) |
| 量化（quantization）、碼、刻度（scale）、還原值 | 想減少表示成本：把原數值換成少數幾個離散選項之一，存下的整數叫碼；相鄰兩個整數代表的數值差叫刻度，用刻度把碼換回的浮點數叫還原值 | [17.1](chapters/17.md#17.1)、[17.2](chapters/17.md#17.2) |
| 向量量化（Vector Quantization，VQ） | 把encoder向量換成最近的碼本向量；儲存所選列的ID，再用對應向量送decoder，不是替每個模型權重改存少數bit的同一個操作 | [12.18](chapters/12.md#12.18)、[17.16](chapters/17.md#17.16) |
| 直通估計（straight-through estimator，STE）、commitment loss | 硬選碼對encoder沒有可用的重建梯度，STE改用近似反向規則；commitment loss讓encoder靠近已選碼，更新路徑與碼本loss分開 | [17.16](chapters/17.md#17.16) |
| 殘差向量量化（Residual Vector Quantization，RVQ） | 第一個碼本近似後，下一個碼本再量化剩餘誤差；多份向量相加還原。碼本軸不同於時間軸 | [17.17](chapters/17.md#17.17) |
| 蒸餾（distillation）、教師模型 | 讓學生透過教師提供的訊號學習，例如教師的回答或候選比例；教師模型是另一個已訓練好、替新模型提供答案或機率的模型 | [9.13](chapters/09.md#9.13)、[18.1](chapters/18.md#18.1) |
| 屬性教師、風格教師、MoE 教師、圖片教師、圖音教師 | 第 18 章實驗用的五位教師：前三位只讀文字，圖片教師與圖音教師用在多模態實驗，會看圖片 | [18.1](chapters/18.md#18.1)、[18.13](chapters/18.md#18.13) |
| 字表對齊 | 逐欄比較教師與學生的機率之前，先確保兩邊同一欄代表同一個 token；欄數相同還不夠，每一欄的候選身份也要一致 | [18.1](chapters/18.md#18.1)、[18.6](chapters/18.md#18.6) |
| KL 散度（Kullback–Leibler divergence）、熵（H(p)） | 量兩個機率分布差多少：`KL(p‖q)=Σ p_i(log p_i-log q_i)`，兩邊完全相同時是 0。13.14 用它量策略模型偏離參考模型多少，18.8 用它讓學生的比例接近教師。18.8 選讀的熵 `H(p)=−Σ p_i log p_i` 用分布自己的機率衡量它的不確定程度 | [13.14](chapters/13.md#13.14)、[18.8](chapters/18.md#18.8) |
| 推測解碼（speculative decoding） | 想加速既有生成：草稿方便宜地提出候選，主模型（目標模型）決定哪些符合原本的生成規則 | [16.14](chapters/16.md#16.14) |
| 安全、安全標籤 | 想讓回答守住任務明定的邊界：安全標籤問每份回答各自是否通過規則，和問哪份相對較好的偏好標籤不能互相代替；只憑字面拒絕，會擋掉正常的請求 | [9.1](chapters/09.md#9.1)、[9.4](chapters/09.md#9.4) |
| 檢索增強生成（Retrieval-Augmented Generation，RAG） | 想用模型原本不知道的資料：先依問題找出相關段落，再放進模型的前文讓它回答；這只是這一次的輸入，不會寫進模型的參數 | [A.2](chapters/0A.md#A.2)、[A.3](chapters/0A.md#A.3) |
| 工具請求（英文也叫 tool call） | 想讓程式代算：模型生成外層程式讀得懂的工具名稱與參數（一段 JSON 文字），外層先解析、再驗證，兩關都通過才真正執行 | [B.1](chapters/0B.md#B.1)、[B.2](chapters/0B.md#B.2) |

例如「模型檔案太大」可先看量化與蒸餾；「模型已放得下，但接話很慢」可先查時間花在哪裡，再看 cache。檔案變小不等於所有硬體上都變快；少啟動幾位專家也不等於所有權重都能不存。

挑一個你目前想解決的問題，在上表選起點，讀那節的具體數值例子。最後用那個例子解釋為何選它；若只是因為名詞熱門，先回到要改善的量是儲存、時間還是回答品質。

### 偏好與強化學習

| 名詞 | 意思 | 完整講解 |
| --- | --- | --- |
| 獎勵（reward）、獎勵模型（reward model） | 作答後依指定規則打的分數叫獎勵，不自動等於回答的真實品質；先用偏好比較訓練、讀問題與回答後輸出分數的模型叫獎勵模型 | [7.18](chapters/07.md#7.18)、[13.10](chapters/13.md#13.10) |
| 強化學習（Reinforcement Learning，RL）、RLHF（人類回饋強化學習） | 讓模型自己作答、依得到的分數調整，叫強化學習；分數的源頭是人類偏好時，整個流程叫 RLHF，這個名稱交代回饋從哪裡來、流程怎麼接 | [7.18](chapters/07.md#7.18)、[13.17](chapters/13.md#13.17) |
| 近端策略最佳化（PPO）、舊策略、ratio、替代目標 | 強化學習的一種更新方法，設法避免一次把回答的機率改變太多，但不保證每次的改變都很小。收集作答時的策略叫舊策略，同一筆的新機率÷舊機率叫 ratio（機率比）；每筆算「ratio×優勢」與「裁切後的 ratio×優勢」兩項，取較小者當要提高的替代目標（surrogate objective） | [7.18](chapters/07.md#7.18)、[13.12](chapters/13.md#13.12)、[13.13](chapters/13.md#13.13) |
| 策略模型（policy） | 正在更新、之後實際作答的模型：讀入目前的情境，輸出每個可選動作的機率。在語言模型裡，動作是下一個 token；第 13 章的選卡例子裡，動作是選哪張回答卡 | [13.4](chapters/13.md#13.4)、[13.11](chapters/13.md#13.11) |
| 基準值（baseline）、優勢（advantage）、價值模型（critic） | 作答前估計平均會拿多少分叫基準值；實際獎勵減基準值叫優勢，正負表示高於或低於預期；只看題目情境、預測這個平均的模型叫價值模型（英文也叫 value model） | [13.11](chapters/13.md#13.11)、[13.14](chapters/13.md#13.14) |
| 參考模型（reference，也叫固定參考） | 保留這次訓練的起點、不跟著更新的副本：DPO 用它作固定基準，第 13 章的 PPO 用 KL 散度量策略模型偏離它多少 | [7.18](chapters/07.md#7.18)、[13.4](chapters/13.md#13.4)、[13.14](chapters/13.md#13.14) |
| 回答卡、選卡 | 事先寫好的候選回答，模型的回答就是其中一張；從幾張中挑出一張叫選卡。第 13 章用選回答卡的小任務把評分和作答分開；B 單元的每張卡是一個表示下一步動作的固定標記 | [13.10](chapters/13.md#13.10)、[B.5](chapters/0B.md#B.5) |
| 策略梯度（policy gradient）、REINFORCE | 用「優勢 × 所選動作的 log 機率」決定更新方向的方法；C.7 補充裡訓練用的 REINFORCE 是一種策略梯度方法 | [C.7](chapters/0C.md#C.7) |
| 可驗證獎勵強化學習（RLVR） | 用能直接驗算的正確答案給獎勵的強化學習；它會更新參數，和 C.5 參數不變的作答計算不同 | [C.7](chapters/0C.md#C.7) |
| 寬鬆評分、嚴格評分、獎勵鑽漏洞（reward hacking） | C.6 的兩條評分規則：嚴格評分要求整段文字是一個整數，而且等於正確答案；寬鬆評分只要按空白切開後有一段等於正確答案就給分，把 0 到 15 全部列出來的回答也得分。策略利用評分弱點拿分、卻沒完成真正任務，叫獎勵鑽漏洞 | [C.6](chapters/0C.md#C.6)、[13.16](chapters/13.md#13.16) |

查後訓練方法時，先分清材料來源與更新方法：人類回答比較可以提供回饋，PPO則說明一種更新方式。第13章用預寫回答卡讓一次選擇與得分清楚可見，實際文字生成需要再處理逐 token 的選擇。直接用偏好對更新的DPO，可從G.2的連結查起。

### 多試幾次、驗算與機率判斷

| 名詞 | 意思 | 完整講解 |
| --- | --- | --- |
| 驗證器（verifier）、驗算 | 依明確規則核對回答的程式叫驗證器，它做的核對叫驗算；說一個驗證器可靠時，要說清它驗了哪幾項 | [C.2](chapters/0C.md#C.2) |
| 集合命中、pass@k | 每題抽 k 份候選，至少一份正確叫集合命中；評測生成程式的模型時常用的 pass@k 問的是同一件事。原論文的估計式另有候選池大小 n，可以比 k 大 | [C.3](chapters/0C.md#C.3) |
| 已知答案命中率（oracle coverage） | 用已知的正確答案逐份檢查候選，集合命中的題數占全部題數的比例；要事先知道答案才算得出，不能當成能實際部署的選法的成績 | [C.3](chapters/0C.md#C.3) |
| 有型別的機率判斷、Jev、Choice、Noul、Score | 一次判斷由狀態材料、問題與候選三部分組成，回傳型別固定的機率。Jev 是 TypeSafe 公司提供的判斷服務，本課介紹其中三種型別：Choice 回傳選項分布，Noul 回傳 yes 機率，Score 回傳有次序的有限等級分布 | [9.11](chapters/09.md#9.11) |
| pointer head（指向式計分頭） | 用問題表示與候選表示的投影比較，替在場選項打分；重排性質取決於表示、位置與readout，不能只看點積公式 | [9.17](chapters/09.md#9.17)、[20.14](chapters/20.md#20.14) |
| none、unsupported | none指目前列表沒有合適答案；unsupported指任務在支援世界之外。單一none輸出不能自動告訴我們是哪個原因 | [9.17](chapters/09.md#9.17)、[19.23](chapters/19.md#19.23) |
| 信心（confidence）、集中度（concentration） | 本課信心是softmax最大的機率p_max，只比較在場候選；Jev Choice的confidence欄位另稱集中度，n>1時計算(最大機率−1/n)÷(1−1/n)。兩者都不是直接由答對紀錄算出的正確頻率 | [9.9](chapters/09.md#9.9)、[9.15](chapters/09.md#9.15)、[9.18](chapters/09.md#9.18) |
| 覆蓋率（coverage）、接受後錯誤率（accepted risk） | 接受題數除以全部預定題數是coverage；接受列中錯誤數除以接受數是risk。零接受時risk無法計算；事後看正誤挑門檻和固定規則的final結果分開 | [9.18](chapters/09.md#9.18)、[20.16](chapters/20.md#20.16) |
| 校準（calibration）、裁判核對、範圍校準（range calibration） | 校準指讓報出的機率和實際答對的比例相稱，例如報 0.8 的那些題大約八成答對。8.10 先用已知例子核對裁判的判準，叫裁判核對；17.12 用一批輸入記下中間特徵的範圍，叫範圍校準，和第 9 章的機率校準是兩件事 | [9.9](chapters/09.md#9.9)、[8.10](chapters/08.md#8.10)、[17.12](chapters/17.md#17.12) |
| 溫度縮放（temperature scaling）、校準資料（calibration split） | 在另一份未拿來更新參數的資料選正溫度T，raw logits除以T再做softmax。它保同題argmax，不保證換來源後校準改善，也不一般保跨題信心排序 | [9.10](chapters/09.md#9.10)、[9.18](chapters/09.md#9.18) |
| 決策資料（policy split）、決策規則 | policy資料用來選行動門檻，校準資料選T；要選模型時另留model-dev，final最後套凍結規則。這裡的policy資料不同於第13章的策略模型；19章Auto把機率判斷轉成有限路由與預算 | [9.14](chapters/09.md#9.14)、[9.18](chapters/09.md#9.18)、[19.23–19.24](chapters/19.md#19.23) |

### 第 19、20 章的成品與外部模型

| 名詞 | 意思 | 完整講解 |
| --- | --- | --- |
| MoE助理、Looped解題器、機率判斷器 | 第19章已發布的三個自家從隨機起點訓練的基線：A處理有限文字、圖、字卡與錄音；B解固定語法的狀態／加法題；C依狀態、問題與候選回傳機率。歷史成績各屬原版本 | [19.13](chapters/19.md#19.13)、[19.20](chapters/19.md#19.20) |
| A影音擴充、有限音調模組、C-v2 | 本輪新版設計：保留A基線另接影音模組，將真預測接有限音調生成；C-v2共享材料並重新訓練候選決策。正式能力依各自驗收，不能由CPU機制例推得 | [19.21–19.24](chapters/19.md#19.21) |
| C-v3裝置判斷器 | 目前建議的裝置決策成品，讀取3–6個裝置的明確欄位資料，回答動作、資訊是否齊全及固定規準評分；與舊文字C、C-v2研究版本分開驗收 | [19.25](chapters/19.md#19.25) |
| 助手（assistant）、助理 | 對話資料替每段話記一個角色（role），寫明這段話是誰說的；模型那一方的角色叫助手，角色欄位寫 assistant。第 19 章的 MoE 助理與第 20 章的助理，指的是整個成品 | [6.6](chapters/06.md#6.6)、[7.1](chapters/07.md#7.1)、[19.13](chapters/19.md#19.13)、[20.1](chapters/20.md#20.1) |
| 字卡 | 讀字用的中文字圖片，由作者製作；和第 13 章、B 單元的回答卡不同 | [11.18](chapters/11.md#11.18)、[19.13](chapters/19.md#19.13) |
| 跨切分重複的輸入句 | 同一段輸入（連同問題與候選）出現在不同資料份的列裡；家族沒有跨份，不等於模型從未看過這段文字 | [19.14](chapters/19.md#19.14) |
| 前處理器（processor） | Qwen 附的前處理程式：把圖片換成模型能讀的輸入，並套用 Qwen 的對話模板標出角色 | [20.7](chapters/20.md#20.7) |
| Qwen、Whisper | 第 20 章接手、別人已經訓練好的兩個成熟模型：Qwen3-VL-2B-Instruct 讀照片與文字、生成回答，也是這一章的底座；Whisper 做語音辨識，輸出轉寫文字 | [第 20 章開頭](chapters/20.md)、[20.1](chapters/20.md#20.1) |
| Kev、Qwen Omni | Kev是Jev式獨立開源決策模型；Qwen Omni處理多模態理解與文字／聲音輸出。第20章拿它們作成熟案例，本課主線不以其權重／輸出作訓練教師 | [20.14](chapters/20.md#20.14)、[20.15](chapters/20.md#20.15) |
| Thinker、Talker、Code2Wav | Omni中分別負責理解與文字、聲音碼生成及碼到波形的角色。兩代Talker接收的條件與最後解碼路線不同；Thinker名稱本身不表示必有推理文字 | [20.15](chapters/20.md#20.15) |

### 英文與縮寫索引

彙整本頁涉及的英文名詞與縮寫，依英文字母順序排列，不分大小寫，空格、連字號與符號不計。每個詞列一次，連到下定義的地方；括號裡是本課用的中文名稱或符號。decode 在本課有兩個意思，兩個都列出。

**A**　[accepted risk](chapters/09.md#9.18)（接受後錯誤率） · [activation](chapters/17.md#17.11)（中間特徵） · [activation checkpointing](chapters/16.md#16.10)（中間特徵重算） · [activation function](chapters/02.md#2.4)（啟用函式） · [Adam](chapters/05.md#5.4) · [adapter](chapters/08.md#8.9)（適配器） · [advantage](chapters/13.md#13.11)（優勢） · [ASR](chapters/12.md#12.11)（語音辨識） · [assistant](chapters/06.md#6.6)（助手） · [attention](chapters/03.md#3.1)（注意力） · [attention head](chapters/03.md#3.7)（注意力頭） · [attention weights](chapters/03.md#3.1)（讀取比例） · [audio encoder](chapters/12.md#12.8)（聲音入口） · [Auto](chapters/07.md#7.25) · [autoregressive](chapters/07.md#7.20)（逐 token 生成） · [auxiliary loss](chapters/04.md#4.5)（輔助代價） · [axis](first-steps.md#W.3)（軸）

**B**　[backpropagation](chapters/01.md#1.10)（反向傳播） · [baseline](chapters/13.md#13.11)（基準值） · [base model](chapters/07.md#7.11)（底座） · [batch](chapters/05.md#5.2) · [bias](chapters/02.md#2.3)（偏移） · [bigram 表](chapters/01.md#1.5)（接字表） · [bit](chapters/05.md#5.9)（位元） · [BOS](chapters/06.md#6.5)（開始標記） · [byte](chapters/05.md#5.9)（位元組）

**C**　[calibration](chapters/09.md#9.9)（校準） · [calibration split](chapters/09.md#9.10)（校準資料） · [causal mask](chapters/03.md#3.6)（因果遮罩） · [CE](chapters/09.md#9.14)（交叉熵） · [checkpoint](chapters/05.md#5.7)（檢查點） · [Choice](chapters/09.md#9.11) · [classification head](chapters/10.md#10.5)（分類頭） · [Code2Wav](chapters/20.md#20.15) · [codebook](chapters/12.md#12.18)（碼本） · [codec](chapters/12.md#12.18)（聲音編解碼） · [code point](chapters/06.md#6.1)（碼點） · [commitment loss](chapters/17.md#17.16) · [concentration](chapters/09.md#9.15)（集中度） · [confidence](chapters/09.md#9.9)（信心） · [context](chapters/02.md#2.5)（上下文） · [context window](chapters/02.md#2.5)（上下文視窗） · [control](chapters/12.md#12.4)（對照） · [coverage](chapters/09.md#9.18)（覆蓋率） · [critic](chapters/13.md#13.14)（價值模型） · [cross entropy](chapters/01.md#1.8)（交叉熵） · [CUDA](chapters/16.md#16.8) · [C-v3](chapters/19.md#19.25)（裝置判斷器）

**D**　decode（[解碼](chapters/01.md#1.1)、[逐步生成階段](chapters/16.md#16.2)） · [deep supervision](chapters/14.md#14.14)（逐輪監督） · [Dense](chapters/04.md#4.4) · [derivative](chapters/01.md#1.9)（導數） · [distillation](chapters/18.md#18.1)（蒸餾） · [DPO](chapters/07.md#7.18)（直接偏好最佳化）

**E**　[embedding](chapters/02.md#2.1)（embedding 表、embedding 向量） · [encode](chapters/01.md#1.1)（編碼） · [EOS](chapters/06.md#6.5)（結束標記） · [exit depth](chapters/14.md#14.13)（終止深度） · [expert](chapters/15.md#15.2)（專家）

**F**　[family](chapters/05.md#5.12)（家族） · [FFN](chapters/04.md#4.4)（前饋網路） · [FlashAttention](chapters/16.md#16.9) · [forward](chapters/01.md#1.10)（前向計算）

**G**　[gradient](chapters/01.md#1.9)（梯度） · [gradient descent](chapters/01.md#1.12)（梯度下降） · [greedy](chapters/01.md#1.15) · [ground truth](chapters/01.md#1.8)（正確答案）

**H**　[held-out](chapters/01.md#1.12)（留出） · [hidden](chapters/04.md#4.4)（中間寬度） · [hybrid](chapters/20.md#20.14)（混合模型）

**I**　[ID](chapters/01.md#1.1) · [image encoder](chapters/10.md#10.5)（圖片入口） · [inference](chapters/01.md#1.15)（推論）

**J**　[Jev](chapters/09.md#9.11) · [JSON](chapters/07.md#7.13)

**K**　[kernel](chapters/17.md#17.10)（計算程式） · [Kev](chapters/20.md#20.14) · [key](chapters/03.md#3.2)（K） · [KL](chapters/13.md#13.14)（KL 散度） · [KV cache](chapters/16.md#16.2)（快取）

**L**　[labels](chapters/05.md#5.2) · [language backbone](chapters/10.md#10.6)（文字核心） · [LayerNorm](chapters/04.md#4.3)（層正規化） · [learning rate](chapters/01.md#1.12)（學習率，lr） · [Linear](chapters/02.md#2.3)（線性層） · [LM head](chapters/04.md#4.6)（輸出層） · [LN](chapters/04.md#4.3)（層正規化） · [logits](chapters/01.md#1.8)（候選分數） · [lookahead](chapters/16.md#16.16)（未來上下文等待） · [Looped Transformer](chapters/14.md#14.11)（循環式 Transformer） · [LoRA](chapters/08.md#8.8) · [loss](chapters/01.md#1.8)（代價） · [loss mask](chapters/07.md#7.24)（代價遮罩）

**M**　[max_new_tokens](chapters/07.md#7.19) · [mel](chapters/12.md#12.6)（mel 頻帶） · [MLP](chapters/02.md#2.5)（多層感知器） · [modality](chapters/06.md#6.6)（模態） · [MoE](chapters/15.md)（專家混合） · [momentum](chapters/05.md#5.3)（動量） · [MTP](chapters/07.md#7.20)（多 token 預測）

**N**　[NaN](chapters/03.md#3.6) · [NLL](chapters/06.md#6.5)（negative log-likelihood，負對數似然） · [none](chapters/19.md#19.23)（列表無合適答案） · [Noul](chapters/09.md#9.11)

**O**　[Omni](chapters/20.md#20.15) · [optimizer](chapters/01.md#1.13)（最佳化器） · [optimizer state](chapters/05.md#5.7)（最佳化器狀態） · [oracle](chapters/07.md#7.25)（離線理想上界） · [oracle coverage](chapters/0C.md#C.3)（已知答案命中率） · [output layer](chapters/04.md#4.6)（輸出層）

**P**　[PAD](chapters/06.md#6.6)（填充） · [padding](chapters/07.md#7.6)（補齊） · [parameter](chapters/01.md#1.2)（參數） · [pass@k](chapters/0C.md#C.3) · [patch](chapters/10.md#10.2) · [patch embedding](chapters/10.md#10.3) · [pointer head](chapters/09.md#9.17)（指向式計分頭） · [policy](chapters/13.md#13.4)（策略模型） · [policy gradient](chapters/0C.md#C.7)（策略梯度） · [policy split](chapters/09.md#9.14)（決策資料） · [Position Interpolation](chapters/14.md#14.8)（位置插值） · [post-training](chapters/07.md#7.17)（後訓練） · [PPO](chapters/07.md#7.18)（近端策略最佳化） · [prefill](chapters/16.md#16.2)（讀提示階段） · [pretraining](chapters/07.md#7.11)（預訓練） · [processor](chapters/20.md#20.7)（前處理器） · [projector](chapters/10.md#10.6)（接頭）

**Q**　[quantization](chapters/17.md#17.1)（量化） · [query](chapters/03.md#3.2)（Q） · [Qwen](chapters/20.md#20.1)

**R**　[RAG](chapters/0A.md#A.2)（檢索增強生成） · [range calibration](chapters/17.md#17.12)（範圍校準） · [rank](chapters/08.md#8.8)（秩） · [ratio](chapters/13.md#13.12)（機率比） · [reasoning effort](chapters/07.md#7.23)（思考投入） · [recurrent state](chapters/16.md#16.15)（循環狀態） · [reference](chapters/13.md#13.4)（參考模型） · [REINFORCE](chapters/0C.md#C.7) · [residual connection](chapters/04.md#4.2)（殘差連線） · [reward](chapters/07.md#7.18)（獎勵） · [reward hacking](chapters/13.md#13.16)（獎勵鑽漏洞） · [reward model](chapters/13.md#13.10)（獎勵模型） · [RL](chapters/07.md#7.18)（強化學習） · [RLHF](chapters/07.md#7.18)（人類回饋強化學習） · [RLVR](chapters/0C.md#C.7)（可驗證獎勵強化學習） · [RMSNorm](chapters/14.md#14.2)（均方根正規化） · [RoPE](chapters/14.md#14.1)（旋轉位置嵌入） · [router](chapters/15.md#15.3)（路由器） · [RVQ](chapters/17.md#17.17)（殘差向量量化）

**S**　[sample rate](chapters/12.md#12.2)（取樣率） · [sampling](chapters/01.md#1.14)（抽樣） · [scale](chapters/17.md#17.2)（刻度） · [Score](chapters/09.md#9.11) · [SFT](chapters/07.md#7.1)（監督式微調） · [shape](first-steps.md#W.3) · [shared memory](chapters/16.md#16.15)（共享材料記憶體） · [shift](chapters/01.md#1.3)（下一位置對齊） · [sliding-window attention](chapters/16.md#16.12)（滑動視窗注意力） · [SNR](chapters/19.md#19.22)（重建訊雜比） · [softmax](chapters/01.md#1.7) · [source holdout](chapters/09.md#9.18)（來源留出） · [special token](chapters/06.md#6.6)（特殊 token） · [spectrogram](chapters/12.md#12.5)（頻譜圖） · [speculative decoding](chapters/16.md#16.14)（推測解碼） · [speech encoder](chapters/12.md#12.15)（語音入口） · [STE](chapters/17.md#17.16)（直通估計） · [support interval](chapters/12.md#12.17)（時間支持區間） · [surrogate objective](chapters/13.md#13.13)（替代目標）

**T**　[Talker](chapters/20.md#20.15) · [teacher forcing](chapters/05.md#5.8) · [temperature](chapters/01.md#1.15)（溫度） · [temperature scaling](chapters/09.md#9.10)（溫度縮放） · [tensor](first-steps.md#W.3)（張量） · [Thinker](chapters/20.md#20.15) · [thinking effort](chapters/07.md#7.23)（思考投入） · [timestamp](chapters/10.md#10.12)（時間戳） · [TMRoPE](chapters/14.md#14.19) · [token](chapters/06.md#6.1) · [tokenization](chapters/06.md#6.1)（切詞） · [tokenizer](chapters/06.md#6.1) · [tool call](chapters/0B.md#B.1)（工具請求） · [top-k routing](chapters/15.md#15.4) · [transcript](chapters/12.md#12.11)（轉寫文字） · [TTFA](chapters/16.md#16.16)（首段等待）

**U**　[unsupported](chapters/19.md#19.23)（支援範圍外） · [UTF-8](chapters/06.md#6.1)

**V**　[value](chapters/03.md#3.4)（V） · [value model](chapters/13.md#13.14)（價值模型） · [verifier](chapters/0C.md#C.2)（驗證器） · [vocabulary](chapters/01.md#1.1)（字表） · [VQ](chapters/17.md#17.16)（向量量化）

**W**　[weight](chapters/02.md#2.3)（權重） · [weight tying](chapters/14.md#14.6)（權重共享） · [Whisper](chapters/20.md#20.1) · [width](chapters/02.md#2.1)（特徵寬度）
