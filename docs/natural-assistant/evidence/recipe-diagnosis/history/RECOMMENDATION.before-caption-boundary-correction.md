下一輪先採 fresh Qwen3-VL-2B-Instruct LoRA，保留原 manifest、seed 42、180 updates × 2 rows、q/v rank 8、alpha 16、pixel/token 上限、greedy decoding，只將 lr 1e-4 改為 3e-5。這是單變因驗證，並非已證明原 LR 是根因。原先提出 60–90 updates 是降低總更新量的保守候選；這輪若同時縮短步數，會混入訓練量效果，故不採用。4B 作備選，資料分布調整作另一個有新 manifest 的實驗。

本診斷只讀 train / validation 的實際 row、loss、raw generation 與官方資料；沒有模型 forward/generate、GPU、test row/output/image 審閱，也沒有讀他人的語意審閱報告。原始 manifest 的 split / file inventory 僅用來選出 train/validation。程式與資料不修改，全部產物在 ignored outputs 中。

觀測與限制：

| 已觀測 | 可支持的判斷 | 仍未證明的原因 |
| --- | --- | --- |
| 120 個照片 train rows 全用同一英文單句 caption prompt；validation 是繁中照片關係描述及細節問答 | 訓練與產品使用的語言、指令、回答範圍明顯不同 | 尚無同圖英文/中文 prompt 對照，不能單獨歸因於語言 |
| 30 個 chat rows 是 10 組相同問答各重複 3 次，另有 2 個 dialogue | 32 rows 不能當作 32 組獨立聊天經驗；對話分布窄 | 不代表聊天一定失效，須看獨立語意審閱 |
| 360 row visits 中 scene 153 (42.5%)，但佔 supervised tokens 3763/5791 (65.0%)；OCR 137 visits、presence 32、chat/dialogue 38 | row 比例不等於 token 比例。Loss 是每 update 內兩 rows 按 token 加權，再由 AdamW 更新，不能把全局 token 比例當作精確梯度影響比例 | 無每 row loss / gradient 的因果歸因 |
| Pure OCR updates token-weighted loss 0.00793；pure presence 0.00159；pure scene 2.105；scene 36/36 updates 被 clip，全體 139/180 被 clip | OCR/presence 在此有限資料上已容易；scene 的訓練壓力較大 | clipping 不等於不穩定，也不能證明 1e-4 過大 |
| 首 30 / 末 30 updates loss 約 2.052 / 1.261；第一輪136updates / 第二輪44updates約1.741 / 1.204 | train objective 有下降 | 隨機抽到的任務/例子不同，非固定 validation learning curve；下降不保證泛化 |
| 同 validation 的 OCR 9/9、presence 3/3 兩 variants 都全對；scene fact proxy 21/24 → 15/24 | 沒有有限 OCR 上的 adapter 增益，scene fact proxy 退步 | Keyword proxy 有語意限制；不是完整圖片能力分數 |
| 12個scene descriptions 截斷由1→5；raw四-token片段至少出現10次由0→5，adapter最高34次；24個fact沒有這種長循環 | 確有局部長循環，而且集中在描述。增加 max_new_tokens 不能修復生成循環 | 不代表每張照片都退步，也尚未判定 factual correctness |
| 12個description token median 262.5→274.5；24個fact 85.5→20 | 簡短主要出現在fact，不能聲稱全部照片回答都變短 | 短答案本身不是錯誤；與proxy退步並列而非因果 |

DOCCI 官方 pinned 原資料說明：人寫的英文長描述，特別涵蓋空間關係、計數與細節，平均 136 words。這裡選定120張 train 的原描述平均126.48 words，但實際 target 只取第一句，平均19.61 words；99/120 target 含 view/photo/shot，109/120 以 A/An 起頭。這是符合當時 caption 任務的合法派生，但丟掉了大部分關係與細節 supervision。Adapter 的「這是一張從…視角拍攝…」與重複局部屬性，與該 target 風格吻合；仍屬待實驗驗證的假說。

這輪採用的 LR 對照應核對逐 update row_ids、supervised_tokens 與 initial LoRA hashes 完全相同，僅 optimizer LR 不同；用相同 validation prompt、greedy、max_new_tokens=384 和相同 scoring。預先比較完整 raw answers 的語意、fact proxy、OCR/presence 與長循環，不只比較訓練 loss。它只能估計這個固定 checkpoint / seed / dataset / decoding 下的 LR 差異，不足以概括所有 seeds。若 adapter 沒有可辨識的品質提升，產品仍應保留 base 作選項或採 base，而不是因做過訓練而必須部署 adapter。

其他候選的選擇依據與風險：

| 候選 | 何時值得做 | 真實風險 / 比較方式 |
| --- | --- | --- |
| 2B 降 LR、縮至60–90 updates | LR 對照仍顯示過度風格改寫、保守短更新值得作獨立實驗時 | 同時換 LR/steps 會混因；最好在同 LR 下比較早期 checkpoint。少 steps 可能欠擬合，不能預設更好 |
| 4B Dense + 溫和 q/v LoRA | 2B 溫和更新仍有明確產品需求缺口，且時間/預算允許，先在同 validation 核 4B base，再測 fresh 4B adapter | 較大不保證這組資料較好；更長下載、載入與decode時間。仍會繼承英文caption→中文QA錯配。先核同processor設定、真GPU最大案例記憶體/吞吐，保留2B基準 |
| DOCCI train 的繁中指令、關係 / 計數 / 細節 QA 與聊天比例 | 語言/指令/答案範圍錯配需要處理，且願意做新資料實驗時 | 只用 train images/full captions，實際看圖確認標籤；不能翻譯validation答案回填。保留原文/翻譯/QA provenance，換新manifest及版本。新標籤可能錯、放大來源偏差；重複同圖需按image family算有效資料量 |

資料候選可先將相同的 train 圖片分配給繁中主要內容、關係、計數/可見細節等不同指令（例如40/40/40是設計起點，不是驗證過的最優比例），再補真正不同的聊天/多輪例子。OCR 已飽和的有限觀測支持不用盲目增加重複 OCR；仍需保留適當 OCR/presence replay 來檢查能力保留。不能因全局 token 占比就斷言應直接降低 scene 權重；匹配照片任務比単純降低 caption 比例更具針對性。

4B 官方備選證據：

- Repo `Qwen/Qwen3-VL-4B-Instruct`，固定公開 revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`；model card、config、index 與 preprocessing/generation metadata 已成功下載並留 SHA256 receipts。沒有下載完整 shards。
- 官方 class `Qwen3VLForConditionalGeneration` / model_type `qwen3_vl`；text 是36層 dense MLP，hidden=2560、intermediate=9728、Q heads32 / KV heads8、head_dim128，沒有 expert/router 設定。
- 用現有 CPU `torch 2.8.0+cpu / transformers 4.57.6 / peft 0.18.1 / accelerate 1.12.0`，`init_empty_weights` 及 tie weights 能完成全 meta model+LoRA 建構，沒有 forward。此項核對正式釋出4.57.6支持，不必依model card過時的「4.57.0尚未release」註記升級至source。
- Unique base params **4,437,815,808**（text4,022,468,096；vision415,347,712）。BF16 unique weight **8,875,631,616 bytes = 8.266 GiB**，與官方 safetensors index 的 total_size 完全一致，避免 tied embedding 重複計數。
- q/v共有72 modules：36個q `[4096,2560]`、36個v `[1024,2560]`。rank8 / alpha16 trainable params = `36×8×[(4096+2560)+(1024+2560)]` = **2,949,120**；base+LoRA4,440,764,928。LoRA若用FP32，weights+grad+兩個Adam states粗算16bytes/param=**45 MiB**；其他optimizer/runtime buffers不在此估計。
- Batch1 BF16 decoder KV 約147456bytes/token，2048token約0.28125GiB，僅KV，不含vision/prefill/workspaces。Train use_cache=False，不能把此推論KV直接加進train已量測值。
- 已量測2B train peak allocated **5.230 GiB**，base BF16 weights3.963GiB；4B僅新增base weights就多4.303GiB。把其他2B開銷原樣加上得到9.533GiB的粗外推，但4B的層數/hidden/MLP更大，activations、allocator reserved及workspaces會改變，因此不是GPU fit上界、不是已測4B peak。L4 24GB有合理的嘗試空間，仍須真實最大image/sequence案例實測才可宣稱fit。
- Dense4B不涉及MoE；任何MoE備選都要看**總**resident weights與runtime記憶體，active參數小不等於所需memory小。

官方model card對VL建議sampling (`temperature0.7/top_p0.8/top_k20/presence_penalty1.5`)，本runner則greedy；這是另一個可研究的deployment條件。兩個既有variants共享greedy，現在LR A/B也應維持它，不能換decoding來掩盖退步或說greedy已被證明是根因；presence_penalty亦不能未核API就照搬至HF generate。

預算方面先完成既定LR對照與validation，再依真實ledger與保留的final/external/release支出決定是否4B。現有16.45 reserve與正在跑的0.62 external CPU不是已付費實額；不能用2B的149.47秒train直接承諾4B成本，更不能消耗必須完成的測試/發布預留。這份診斷不啟動付費工作。

可重現證據：`training-and-output-analysis.json` / `analyze_training.py` 精確重現全部180updates的supervised token數，`meta-4b-analysis.json` / `meta_count_4b.py` 保存meta與LoRA核對，`docci-source-analysis.json`保存DOCCI原資料SHA驗證，`official-source-receipts.json`保存4B下載pins。檔案均已確認被git忽略。
