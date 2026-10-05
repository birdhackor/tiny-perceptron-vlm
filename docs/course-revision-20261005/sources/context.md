# 長上下文：第一階段來源筆記與微小節建議

查閱日期：2026-10-05。用途：確定新增內容、權威來源及教學順序；本檔不是課文、訓練結果或新能力報告。本次直接下載並閱讀下列原論文指定章節、固定版本官方文件及程式註解；沒有訓練、準備訓練資料、執行原論文評測或修改既有教材。

## 1. 先確定教學主線

主線應回答五個不同問題：「這次實際送入什麼」、「最多能再寫多少」、「位置超出訓練範圍時如何處理」、「計算和記憶體如何負擔」、「回答是否真正用到遠處線索」。前三項限制與資源改善都不能代替最後一項能力評測。

對已學會 token、causal mask、生成上限和 Q/K/V 的讀者，建議依序教授：輸入／輸出預算 → 能送入與能使用的差別 → RoPE 的相對旋轉 → 訓練長度與外推 → Position Interpolation → 局部注意力的可見範圍 → KV cache／FlashAttention → 按線索數及對話狀態設計評測。YaRN 在主線以兩個改良目的介紹，分頻公式、溫度擬合與動態縮放放延伸閱讀。

第 2 章現有固定長卡上下文是概念起點，適合先保留「沒有收到線索，不能靠多訓練補出線索」；完整長上下文單元須等讀者理解 Attention、位置和生成後再引入。既有 RoPE 14.1、prefill／KV cache 16.2–16.4 與 SDPA／FlashAttention 16.8–16.9 可作概念前置，不把所有長上下文內容塞回第 2 章。本階段只建議銜接，不改這些章節。

## 2. API 與長度：支持和不支持哪些話

### C01：context 與 `max_new_tokens` 分別限制什麼？

來源：Hugging Face Transformers **v4.57.1** 的 [`GenerationConfig` 原始文件註解](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/generation/configuration_utils.py#L104)，小標「Parameters that control the length of the output」；vLLM **v0.8.5** 的 [`ModelConfig` 文件註解](https://github.com/vllm-project/vllm/blob/v0.8.5/vllm/config.py#L238)，`max_model_len`。

- 支持：Transformers 的 `max_new_tokens` 是新生成 token 數上限，不含 prompt；`max_length` 描述 prompt 加生成結果的總長，兩者同時設定時 `max_new_tokens` 優先。vLLM 的 `max_model_len` 明確包含 prompt 和 output。
- 可教的保守預算：對保留完整歷史、共用單一總長度上限 C 的 decoder-only 推論，輸入 token 數 P 與預留新輸出 G 應滿足 `P + G <= C`。例如 C=16、P=12，最多預留4個新 token。這是小型數字例子，沒有指定真實模型。
- 實際 P 應數最終送入模型的 token ID，包括 system／歷史訊息和對話格式。`max_new_tokens` 是上限，不保證一定輸出這麼長；EOS 等停止條件可以提早結束。
- 不支持：「把 `max_new_tokens` 改大就會擴大上下文」；「每個 API 的 context 定義都一樣」；「token 數等於中文字數」；「設定為 C 就證明 C 範圍都有可靠能力」。滑動窗口、丟棄歷史、encoder-decoder 及特殊服務上限需另說各自契約。

### C02：訓練長度、設定上限與有效長度要分開

來源：PI §2.2、§3.1–3.3；RULER §4「Effective Context Size」；Lost in the Middle §1、§2.3；各原文連結見下文。

- 支持：模型在某個長度被訓練、軟體允許執行某個長度、模型在某項任務通過某個長度，是不同事實。RoPE 公式可以算出新位置，仍可能在未見過的範圍出現品質下降。
- RULER 的 effective length 是該論文選定任務、長度與通過門檻的結果；其門檻使用 Llama2-7B 在4K的表現。教材可以引用這個方法說明「必須先指定驗收任務和門檻」，不要把該數字當作跨所有用途的物理上限。
- 不支持：「訓練最長8K，所以8K內每個位置一定同樣可靠」；「某次密碼取回成功，所以所有8K問題都能回答」；「長序列 perplexity 降低，所以多步推理或長對話一定改善」。

## 3. 位置編碼和延伸長度

### C03：RoPE 提供相對旋轉，沒有自動附送無限長度能力

來源：Su et al., [*RoFormer: Enhanced Transformer with Rotary Position Embedding*，arXiv:2104.09864v5](https://arxiv.org/pdf/2104.09864v5)，§3.2.1–3.2.2，式13–16，Figure 1；§3.3、§3.4.3。Chen et al. 的 PI §2.1–2.2 補足外推限制。

- 支持：將 Q、K 的分量成對旋轉。固定原始 q、k 時，旋轉後點積的**位置依賴部分**可寫成相對位移 n−m；對兩位置加上相同位移，這個相對關係不變。每對分量有自己的角速度，不是所有維度每格都轉同一角度。
- 高中數學切入：先用同一平面的兩支箭頭與45°旋轉理解共同位移，再用兩個不同速度的指針理解多組旋轉。只需座標、點積、正弦餘弦，複數式及大矩陣留延伸。
- 不支持：「點積只取決於距離，與內容無關」；「所有實際 attention 權重必定隨距離單調下降」；「相對位置保證任意長度外推」。q、k 本身由內容與前層計算得到，softmax 還與其他候選位置有關。原論文長距離衰減的討論不能轉寫成這三句保證。
- PI §2.2 直接指出原始 LLaMA/RoPE 超出訓練上下文時的失敗，並討論 RoFormer 所給上界可能過寬。教材主文應說「未見過的旋轉／相對距離分布需要處理」，不用小模型失敗去重新證明已知外推問題。

### C04：Position Interpolation 壓縮的是位置座標，不是文字本身

來源：Chen et al., [*Extending Context Window of Large Language Models via Positional Interpolation*，arXiv:2306.15595v2](https://arxiv.org/pdf/2306.15595v2)，§2.2，§2.3式4與 Theorem 2.1，Figure 1–2；§3.1–3.5。

- 支持：從原訓練範圍 L 延到 L′，在計算 RoPE 前將位置 m 換成 `m × L / L′`。例如8格的座標範圍延到16格，位置0、2、14分別映成0、1、7；16個 token 都還在，位置座標較密。
- §2.3 的穩定性上界支持其動機；§3.1 原論文常用1000步微調，§3.2檢查長序列語言建模，§3.3檢查 passkey，§3.4檢查原上下文能力，§3.5檢查長文摘要。
- 可以說「位置插值配合適當微調，在論文的 LLaMA 設定成功延長上下文」；不要省略適應與評測。
- 不支持：「只改一個設定就一定保留原能力」；「插值把16個 token 合併成8個」；「原論文約600倍的上界差就是準確率提高600倍」；「所有模型都需要原論文相同步數」。Theorem 2.1 是注意力分數的數學界，沒有證明任意任務的理解品質。

### C05：YaRN 是分頻處理加注意力尺度調整

來源：Peng et al., [*YaRN: Efficient Context Window Extension of Large Language Models*，arXiv:2309.00071v3](https://arxiv.org/pdf/2309.00071v3)，§3.1–3.3、Definition 1–2，式14–15；§3.4；§4.1–4.4與Table 1。查閱的 PDF 明確標示 v3、2026-02-06。

- 支持：§3.2 按旋轉波長相對訓練長度，對快轉、慢轉和中間維度採不同縮放；§3.3 的 Definition 2 明確把「NTK-by-parts 插值」與「attention scaling」合稱 YaRN。保留較快轉動的一部分，有助避免把鄰近位置的差異一起壓得過小。
- 主線適合兩個問題：所有指針一律減速，近處差別會變小；讓不同指針承擔不同範圍，再調整 attention 分數尺度。完整頻率分段、ramp、溫度公式留延伸。
- §4.1 的128K配方仍用長上下文微調；§4.2 Table 1 另示範以64K訓練資料、指定縮放及接續微調後的128K語言建模評估。§4.3是單密碼評估，§4.4另查短上下文基準。
- 不支持：「YaRN 完全不需要訓練」；「用過 YaRN 就必定長對話可靠」；「原論文的10倍 token／2.5倍步數差可直接套到本成品」；「有128K語言建模結果就代表128K多步推理通過」。無微調與有微調結果須分別描述。
- 延伸注意：§3.4 提醒動態改變縮放時，舊 KV 若已套用原 RoPE 不能不加處理地沿用。這是實作一致性問題，不適合在第一個 RoPE 小節同時教授。

## 4. 注意力可見範圍與執行分塊

### C06：滑動窗口降低直接可見範圍，堆疊層可傳遞更遠資訊

來源：Jiang et al., [*Mistral 7B*，arXiv:2310.06825v1](https://arxiv.org/pdf/2310.06825v1)，**§2「Architectural details」**內的「Sliding Window Attention」「Rolling Buffer Cache」「Pre-fill and Chunking」，Figures 1–3與Table 1。該版本沒有獨立編號的§2.2，引用應用上述實際標題。

- 支持：每層只直接注意最近窗口；更早資訊可能經前層的中間表示間接傳遞。論文以 W×層數描述理論傳遞跨度。固定注意力跨度使該種模型的逐層 cache 可用滾動緩衝區，到窗口後停止增長。
- 取捨：固定窗口減少可見配對與 cache，直接查任意遠處原始 K/V 的機會也減少；間接傳遞需透過有限深度和表示。理論跨度不是在那個長度上所有細節都可精準取回。
- 論文的「cache memory usage by 8x, without impacting model quality」是其既有 SWA 模型相對同一注意力契約的 cache 實作敘述。**不能**改寫成「把任意 full-attention 模型刪掉七成／八成歷史，品質都不變」。
- 「Pre-fill and Chunking」把已知 prompt 分段送入，仍保留正確的跨段 cache 和原 SWA mask。它沒有宣稱每段彼此獨立就等價於完整處理。

### C07：「分塊」有三種不同含義，教學圖必須畫出 mask

來源：Transformers v4.57.1 的 [`masking_utils.py`](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/masking_utils.py#L77)，`causal_mask_function`、`sliding_window_overlay`、`chunked_overlay`、`chunked_causal_mask_function`；同版本 [`Llama4TextModel.forward`](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/llama4/modeling_llama4.py#L525)；Mistral §2；FlashAttention §3.1。

| 名稱與例子 | 是否改變可見配對 | 取捨與必須說明的界線 |
| --- | --- | --- |
| 滑動局部注意力 | 是；每個 query 的窗口跟著位置移動 | 限制本層直接可見歷史；需要再說層間傳遞。 |
| 區塊局部注意力，如官方 `chunked_overlay` | 是；同一區塊且不在未來的 key 才可見 | 區塊邊界切斷該層的直接連結；單靠局部層，不會自動跨區塊。 |
| prompt 分段 prefill 或 FlashAttention 的計算 tiling | 不必改；取決於是否保留原 cache／mask與正確合併 | 處理單位變小不代表丟掉跨塊資訊。FlashAttention 原版計算完整注意力。 |

- 官方 chunk mask 判斷 key/query 去掉左 padding 後的整除區塊編號相同，再與 causal mask 相交。可用8位置、每塊4位置畫圖：區塊局部在位置4開始新塊，滑動窗口則仍能連到位置3。圖只示範允許邊，不宣稱注意力權重和答案。
- Llama4 官方實作同時建立 `full_attention` 與 `chunked_attention` mask，依層的 `attention_type` 選用。因此不能從某一局部層推論整個混合模型完全不能跨區塊。
- 不支持：「分塊注意力都是 FlashAttention」；「chunked prefill 一律重置歷史」；「只把程式改成分批處理就會學到更長依賴」。

### C08：Transformer-XL 可作有跨段狀態的延伸例子

來源：Dai et al., [*Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context*，arXiv:1901.02860v3](https://arxiv.org/pdf/1901.02860v3)，§3.1「Vanilla Transformer Language Models」、§3.2「Segment-Level Recurrence with State Reuse」、§3.3「Relative Positional Encodings」，Figures 1–2、式1–2。

- 支持：獨立切段會造成 context fragmentation；Transformer-XL 在下一段重用前段的隱藏狀態，訓練時對重用狀態使用 stop-gradient。位置編碼也須處理跨段的位置識別。
- 適合延伸比較「沒有跨段狀態／有限跨段狀態／完整可見前文」，不必讓高中初學者先重寫 Transformer-XL。
- 不支持：「把任意既有模型的 chunk 接起來就得到 Transformer-XL」；「段級狀態重用與普通推論 KV cache 的訓練行為相同」；「有跨段狀態就保留全部歷史細節」。

## 5. 資源改善與能力改善

### C09：KV cache 重用已有計算，容量仍有成本

來源：Transformers **v4.57.1** [*Caching*](https://huggingface.co/docs/transformers/v4.57.1/cache_explanation)，「Attention matrices」「Cache class」「Cache storage implementation」「Cache position」；[*KV cache strategies*](https://huggingface.co/docs/transformers/v4.57.1/kv_cache)，「Default cache」「Fixed-size cache」「Cache offloading」「Quantized cache」。本次閱讀同 tag 的原始 Markdown。

- 支持：causal 推論中，先前 token 的 K/V 不需因未來新 token 到來重算；每層保存 cache，下一步計算新 token 的 Q/K/V，並使用已有 K/V。mask 要涵蓋 past+new，位置也須一致。
- dense、相同每層形狀且未量化的概略 cache 容量可由官方形狀直接推得：`2 × layers × batch × tokens × kv_heads × head_dim × bytes_per_value`；2代表 K 和 V。這是張量容量估算，不包含 allocator、暫存、額外索引及模型權重。GQA 應使用 K/V 頭數，不能直接用 Q 頭數。
- 對 dense 單 token 解碼，cache 可避免反覆處理整段前文；attention 仍要讀過往 K/V，單步可見長度的計算量通常隨長度線性成長。完整一段所有位置的注意力配對仍是平方量級；不能把單步與整段總成本混在一起。
- cache offloading 以傳輸換 GPU 記憶體；量化 cache 改變數值精度；固定 cache 以預留容量配合編譯。這些取捨不應混成「cache 必定同時更省記憶體、更快、更準」。
- 不支持：「cache 自己學會記憶新知識」；「cache 不占記憶體」；「開 cache 讓模型可以理解超出訓練長度的文字」；「推論 cache 能原樣開在所有訓練流程」。官方指南明示其所述 cache 以 inference 為用途。

### C10：FlashAttention 省中間資料與搬移，原版仍算完整注意力

來源：Dao et al., [*FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*，arXiv:2205.14135v2](https://arxiv.org/pdf/2205.14135v2)，§2.1–2.2，§3.1 Algorithm 1／Theorem 1，§3.2 Theorem 2，§3.3，§4.3與§5。

- 支持：將 Q/K/V 分塊放入較快的片上 SRAM，合併正確的 softmax 統計；反向時重算所需部分，避免在 HBM 保存完整 N×N中間矩陣。Theorem 1 仍給 `O(N²d)` FLOPs；減少的是中間儲存及 HBM 讀寫，不能說全部計算改成線性。
- 「exact」指同一完整注意力數學問題，不保證各種浮點運算順序逐 bit 相同。論文§3.3另提出 **block-sparse extension**，它才以稀疏模式改變原本配對；兩者分開教。
- 更低資源成本可以讓較長序列訓練／推論可行，然後才有機會靠資料和訓練獲得新能力。同一模型、同一有效 mask、同一輸入只換精確計算 kernel，不構成已獲得長程理解能力的證據。
- 不支持：「FlashAttention 自動擴大模型的位置範圍」；「CPU 的分塊 Python 例子已驗證 CUDA 加速」；「任何裝置、尺寸或 backend 都有原論文倍數收益」。§5討論專用 kernel 和硬體可移植性；速度和記憶體結論須附實際環境。

## 6. 能力評測：從取回一條資訊到持續對話

### C11：Lost in the Middle 是測試位置敏感性的依據

來源：Liu et al., [*Lost in the Middle: How Language Models Use Long Contexts*，arXiv:2307.03172v3](https://arxiv.org/pdf/2307.03172v3)，§2.1–2.3（multi-document QA）、§3.1及Figure 7（key-value retrieval）、§4.1–4.3；Figures 1、5、7–9。

- 支持：固定問題和有效證據，改變證據位置及上下文長度，能隔離「能送入但不穩定使用」的現象。原論文在多個當時模型／設定發現開頭或結尾較好、中間較差的曲線。
- 原文§2的 multi-document QA 有多個文件，**恰好一個包含答案**；不能把名稱翻成「必須整合多份證據的推理」。§3也指出部分模型能幾乎完美完成該合成取回任務，因此不能說每個模型和任務都必然有同一 U 型曲線。
- §4.2「Query-Aware Contextualization」在 key-value 任務改善很大，對 multi-document QA 沒有同樣解決全部問題。把查詢放在前後不是普遍保證；不能把§4的初步分析說成已查明唯一原因。
- 教材應使用「這是要測的位置效應」，不預先要求本課成品一定重現同一失敗。2023年的具體模型成績不代表2026年的模型排名。

### C12：RULER 把多條線索再分成干擾、完整取回與多步追蹤

來源：Hsieh et al., [*RULER: What's the Real Context Size of Your Long-Context Language Models?*，arXiv:2404.06654v3](https://arxiv.org/pdf/2404.06654v3)，§3.1–3.4，Table 2；§4「Effective Context Size」「Model Ranking Criteria」；§5「Task Error Analysis」。

| 原論文任務 | 真正測的內容 | 教材應避免的混淆 |
| --- | --- | --- |
| Single NIAH | 一個目標 key-value 的取回 | 不能宣稱已測到所有內容理解。 |
| Multi-keys NIAH | 多個相似 key 干擾，仍只需一個答案 | 題目有多條資訊，不等於答案需要多條證據。 |
| Multi-values NIAH | 同一 key 的所有 value | 要核對漏項，不能只靠答案中出現一個正確值。 |
| Multi-queries NIAH | 多個不同 key 的完整取回 | 一次多筆取回與關係推理分開報告。 |
| Variable Tracking | 追蹤多步綁定／共指鏈 | 明確需要組合不同位置的線索。 |
| Common／Frequent Words Extraction | 全文聚合及頻率 | 不是另一個單線索密碼查找。 |
| QA（SQuAD／HotpotQA 加干擾） | 一跳與多跳問答 | 保留各子任務分數，不靠總分遮住弱項。 |

- 支持：按長度、線索數、干擾類型及推理步數調整難度；論文共13個任務。多線索教學應先區分「查多筆」和「組合多筆」，再談長度。
- 不支持：「這13題類型就代表所有真實長上下文用途」；「其 effective length 門檻是通用標準」；「漏掉某個任務可用另一任務總平均代替」。原文為合成 benchmark，適合控制因素，仍須接真實任務。

### C13：LongBench 補上長文理解任務，分數仍要按任務解讀

來源：Bai et al., [*LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding*，arXiv:2308.14508v2](https://arxiv.org/pdf/2308.14508v2)，§3.1、§3.2.1、Table 1、§3.2.2；§4.1–4.2。

- 支持：21個資料集、6個類別，包含單文 QA、多文 QA、摘要、few-shot learning、合成任務及程式補全；語言包含英文與中文。§3.2.1 的多文 QA 包含多跳原始資料集，與 Lost in the Middle「只有一個答案文件」的受控測試不同。
- 主線只需用一個實際任務說明：摘要要保留分散全文的重要內容，多文 QA 要依任務整合證據；找一個密碼不足以驗收。
- Table 1 的長度分別以英文詞數和中文字元數描述，不能直接當本模型 tokenizer 的 token 數。指標也不同，如 F1、ROUGE-L、accuracy、edit similarity，不能說每種百分比分數有相同意義。
- 不支持：「抽一筆 LongBench 就等於完成完整 benchmark」；「公開測試資料一定完全沒有預訓練污染」。§3.2.2 的避免洩漏做法不是能排除所有模型污染的證明。

### C14：LongMemEval 提供長對話評測的必要面向

來源：Wu et al., [*LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory*，arXiv:2410.10813v2](https://arxiv.org/pdf/2410.10813v2)，§3.1–3.4，Figures 1–3；§4與§5.2–5.5；Appendix A.1、A.4。

- 支持：500個問題測五種能力：資訊擷取、跨 session 推理、知識更新、時間推理、資訊不存在時棄答。§3.1把帶時間的對話 session 逐一交給系統，§3.2另說明生成／人工編輯的證據對話與干擾歷史。
- §3.3以 LLM judge 評估彈性回答，並用人工 meta-evaluation 檢查一致性；可以參考能力分類，不能無條件把 judge 當真值。本課的有限封閉任務先用明確答案／規則核對，需開放判分時另做人工抽查。
- §3.4與Figure 3 明確分開：商業系統的 online memory、整段歷史 offline reading、只提供證據 session 的 oracle setting。教材也應分開「模型收到全文」與「RAG／記憶系統是否成功找回」。檢索失敗不能全算成模型長上下文理解失敗，全文失敗也不能用檢索分數代替。
- 不支持：「128K窗口就等於持續記得所有使用者資訊」；「這是一份完全自然錄製而未經合成的對話集」；「原論文30%–60%下降代表所有目前產品」。原文§3.4註明產品評測在2024年8月上半月，商業系統還用97題與較短歷史；不能把其數字混為500題完整測試。

## 7. 適合高中數學讀者的微小節順序

以下是第二階段可採用的新增小節設計，不是現在要執行的實驗。每節只解一個問題；使用8／16位置小數字與紙上可核對的例子，不以短訓小模型擔任成熟知識的重新證明。

| 順序／題目 | 讀者要帶走的單一概念 | 可觀察的小例子／圖 | 來源與位置 |
| --- | --- | --- | --- |
| 1. 輸入長度和輸出上限怎麼共用預算？ | 新輸出上限不會擴大窗口 | 16格中12格輸入、4格預留；再畫提前EOS | C01；已有token／生成後 |
| 2. 能放進來，為什麼不等於會使用？ | 設定容量與任務能力分開 | 同一句線索放開頭、中間、結尾，先指出哪些因素固定 | C02、C11；Attention後 |
| 3. 用旋轉表示相對位置 | 固定q/k共同移位不改旋轉差 | 兩箭頭和兩種轉速；用簡單點積核對 | C03；接現有14.1 |
| 4. 超過訓練範圍，哪裡變陌生？ | 公式可算和學過分布是兩件事 | 訓練位置0–7，延到0–15，標出新相對距離 | C03、C04 |
| 5. 位置插值怎麼塞回熟悉範圍？ | 座標縮放，token未刪除 | 16張卡片仍16張、刻度變成m/2；指出鄰近刻度也變近 | C04 |
| 6. 為什麼不讓所有旋轉一律變慢？ | YaRN分頻與attention尺度各解一個問題 | 快慢指針；僅示意尺度變動，不寫一頁NTK推導 | C05；公式延伸 |
| 7. 滑動窗口能直接看到多遠？ | 每層直接可見與多層傳遞不同 | 8位置、4格窗口mask，再畫兩層的間接路徑 | C06 |
| 8. 區塊邊界會切掉哪些連結？ | 區塊局部、分段prefill和運算tiling不同 | 同一8×8圖對照窗口／區塊／完整causal三個mask | C07；C08作延伸 |
| 9. KV cache 留下的是什麼？ | 重用逐層K/V節省重算，仍占容量 | 3 token到第4 token的cache形狀；用個數×每值位元組估容量 | C09；接效率單元 |
| 10. FlashAttention 怎樣省搬移？ | 同一完整答案分塊計算、合併softmax | 同一小矩陣完整計算與分塊合併；硬體收益引用來源後另量測 | C10；接現有16.9 |
| 11. 找一條、多條和長對話怎麼驗收？ | 評測按需要的資訊與狀態分層 | 單碼取回→全取回→兩步關係→最新更改→無資料 | C11–C14；回連RAG／成品 |

其中6的方法名稱與核心目的可在主線短講，技術推導放延伸；8的 Transformer-XL、9的 offloading／cache量化、10的 IO複雜度定理也放延伸。完整 benchmark 來源由作者查閱即可，不要求初學者先讀完論文。

## 8. 對有限成品任務的建議界線

先用一個有清楚答案的「學校活動手冊與對話更改」任務家族限定成品範圍。這裡只列驗收設計，不撰寫／製作資料，不選模型，不開始訓練。

| 層次 | 建議要驗收的行為 | 未通過時先排除什麼 |
| --- | --- | --- |
| 單線索 | 取回一個活動室代碼，位置移到首／中／尾 | 線索是否真在截斷後輸入、token數、答案格式、短版能否完成 |
| 多筆取回 | 列出指定活動的全部規則，明確核對缺項與多項 | 題目是否只需要一項；不可當作已測多步推理 |
| 多線索組合 | 以「14:00出發」「提前10分鐘集合」得出13:50 | 短上下文同題是否會算；避免把基本算術失敗誤認成長度問題 |
| 長對話更新 | 先說15:00，後明確改為16:00，詢問最新時間；另問某日當時的時間 | 指令政策與日期是否明確；最新值和歷史值不可混成同一答案 |
| 資訊不存在 | 未提的活動室代碼應回答資料不足 | 不讓評分規則獎勵亂猜；需和有線索題同時驗收 |

每個題型至少保留短版的同題基準，再依輸入長度、位置、線索數和干擾類型分開記錄；固定最終prompt格式、tokenizer、輸出預算及解碼設定。輸出錯誤要分清截斷、漏取、狀態更新、推理與格式，不能全部統稱「模型忘記」。主文使用一兩個能教會差異的成功／錯誤例子及任務界線；沒有可解釋教學價值的短訓失敗數據不放主文。

這些局部驗收可以確認本成品在限定任務上的表現，不能負責重新證明 RoPE外推、FlashAttention效率或其他業界已成熟的結論，也不能將有限任務包裝成通用長對話助理。

## 9. 查證限制與版本追溯

- 所有上列引用是原作者論文或官方原始碼／文件。閱讀集中在指出的章節，不宣稱逐式重證所有appendix，也沒有觀看影片或執行原作者的完整實驗。
- arXiv無版本入口下載的PDF已從頁首辨識實際版本，正式引用改用下表固定版本；HF與vLLM直接抓固定tag。這些來源足以支撐概念與方法目的，不是2026最新產品排名或本課成品的測試結果。
- 沒有在GPU測速度、記憶體或任何長度能力；原論文數字須附其模型、資料、硬體和評測條件，不直接當本專案預期值。
- 官方Transformers cache文件是教學說明；它的attention式排版不作新mask定義的唯一依據。mask幾何已另查同版本`masking_utils.py`實作。
- 在第五階段選定成品模型／runtime後，仍須查該模型原訓練長度、RoPE scaling、各層attention型態、runtime實際上限及截斷策略；本階段不猜這些值。長對話與檢索系統亦需固定各自測試契約。

直接取得之原論文PDF SHA-256如下，用來核對本次確實讀到的內容，不作執行結果或能力證據。原始檔暫存在`/tmp/context-sources-20261005/`；repository本任務只新增本筆記。

| 論文／固定版本 | SHA-256 |
| --- | --- |
| RoFormer 2104.09864v5 | `e9a481fbe1c8a20b7b1fa566b13102a1896c7829fa9a8b4c80528452a5ddaf79` |
| Position Interpolation 2306.15595v2 | `890702e3170e5b8e2fb55552bd86cd99b0aa4ddbb3670ae8c351faa575cfaf09` |
| YaRN 2309.00071v3 | `e7c0268a796138460c6ba2f67a7cba5bd922c401aea94c1fcd09c4b76b883c85` |
| Mistral 7B 2310.06825v1 | `dfbac4e7035344b305c947481f2e7e8a02f7a24a563917eb6e47f6591d14c5ae` |
| Transformer-XL 1901.02860v3 | `6aa192b22820267309e8e7c501c514a3a3055256d9c8691b106c7f53f9098671` |
| FlashAttention 2205.14135v2 | `ca7f9fda10b90fc05dd291a3accc85e9c1a4a860b99b31928dab03ed3fcb14e4` |
| Lost in the Middle 2307.03172v3 | `653b29619eae2ae4b361d7bdcdc06db4bedcb1b0eabe31814036beca8c1af0b1` |
| RULER 2404.06654v3 | `8a4bc6ca28d84570eec7f42652400c2121f4c8bc361701634e596772e396fc22` |
| LongBench 2308.14508v2 | `2ef5a32cc11976d7d706e24bd608e2dfe859f91b24ed1e02c9acc6a9cf6b87ce` |
| LongMemEval 2410.10813v2 | `05c5d055201466a241a56e082cdd02d39ad566fa04b3804891983e4e069a3fda` |
