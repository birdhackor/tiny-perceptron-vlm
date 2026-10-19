# 實作放在哪一節

下表列出正文展示或說明的 `tiny_perceptron` 定義、所在的小節與寫法。三種寫法的格式與規則見[AUTHORING.md](AUTHORING.md)的「把實作放進正文」。

原樣節錄與精簡版以章節為準：程式區塊第一行的標記（`# 節錄自 …`、`# 精簡自 …`）寫明來源檔案與定義名稱，精簡版省略了什麼寫在標記的括號裡。在 repo 根目錄執行下面的指令可列出全部標記；這張表和它不一致時，以章節為準。

```bash
grep -n -E '^# (節錄自|精簡自) tiny_perceptron/' course/chapters/*.md
```

檔案欄是 `tiny_perceptron/` 底下的路徑。只說明的列沒有標記，寫法欄記下正文說明了什麼。

| 節 | 定義 | 檔案 | 寫法 |
| --- | --- | --- | --- |
| 1.2 | `split_documents` | data.py | 原樣節錄 |
| 2.5 | `ContextMLP` | simple.py | 只說明並連結：查表、拼接、兩個 Linear 與 tanh，以及參數數怎麼算 |
| 3.6 | `manual_attention` | attention.py | 原樣節錄 |
| 3.7 | `CausalAttention` | attention.py | 精簡版 |
| 4.4 | `DenseFFN` | modern.py | 精簡版 |
| 4.5 | `Block` | model.py | 精簡版 |
| 4.5 | `ModelConfig` | model.py | 只說明並連結：本章用到的五個欄位 |
| 4.6 | `TinyLM` | model.py | 精簡版 |
| 4.7 | `shifted` | data.py | 原樣節錄 |
| 5.2 | `loss_sum`、`masked_loss` | model.py | 原樣節錄 |
| 5.6 | `learning_rate` | training.py | 原樣節錄 |
| 5.7 | `save_checkpoint`、`load_checkpoint` | training.py | 只說明並連結：用表列出存進檔案的欄位 |
| 5.11 | `fingerprint` | data.py | 原樣節錄 |
| 6.6 | `ByteTokenizer` | data.py | 原樣節錄 |
| 6.6 | `ByteLevelBPE` | tokenization.py | 只說明並連結：特殊 token 占 ID 0 到 7，編普通文字時不比對特殊 token，編出這些 ID 就報錯 |
| 7.1 | `render_chat` | data.py | 原樣節錄 |
| 7.10 | `pad_batch` | data.py | 原樣節錄 |
| 7.11 | `toy_conversations` | data.py | 原樣節錄 |
| 8.8 | `LoRALinear` | alignment.py | 原樣節錄 |
| 9.10 | `reliability_bins` | alignment.py | 原樣節錄 |
| 10.1 | `scene` | multimodal.py | 原樣節錄 |
| 10.2 | `patchify`、`unpatchify` | multimodal.py | 原樣節錄 |
| 10.5 | `VisionEncoder` | multimodal.py | 原樣節錄 |
| 10.7 | `expand_modalities` | multimodal.py | 原樣節錄 |
| 10.8 | `MultiModalLM` | multimodal.py | 精簡版 |
| 11.14 | `picture_order_report` | natural_concepts.py | 只說明並連結：比較的兩張圖與回傳的三項 |
| 11.15 | `thin_stroke_report` | natural_concepts.py | 只說明並連結：細線圖怎麼縮小與回傳的欄位 |
| 11.16 | `edit_distance` | natural_concepts.py | 原樣節錄 |
| 11.16 | `text_error_report` | natural_concepts.py | 只說明並連結：參數順序與回傳的欄位 |
| 12.1 | `tone` | multimodal.py | 原樣節錄 |
| 12.6 | `mel_filter_bank` | multimodal.py | 原樣節錄 |
| 12.8 | `log_mel`、`AudioEncoder` | multimodal.py | 原樣節錄 |
| 12.13 | `audio_order_report` | natural_concepts.py | 只說明並連結：怎麼接兩段音、倒轉時間框與回傳的欄位 |
| 12.14 | `speech_stages` | natural_concepts.py | 只說明並連結：四個參數與回傳的欄位 |
| 12.15 | `AudioEncoder` | selftrained/model.py | 只說明並連結：三步怎麼整理時間與兩個回傳值 |
| 13.3 | `sequence_log_probability` | alignment.py | 原樣節錄 |
| 13.5 | `dpo_loss` | alignment.py | 原樣節錄 |
| 13.10 | `preference_loss` | posttraining.py | 原樣節錄 |
| 13.10 | `FiniteRewardModel` | posttraining.py | 原樣節錄 |
| 13.11 | `bandit_advantage` | posttraining.py | 原樣節錄 |
| 13.13 | `ppo_clipped_objective` | posttraining.py | 原樣節錄 |
| 13.14 | `exact_kl` | posttraining.py | 原樣節錄 |
| 13.15 | `FiniteResponsePolicy`、`FiniteValueModel` | posttraining.py | 原樣節錄 |
| 14.1 | `rope` | modern.py | 精簡版 |
| 14.2 | `RMSNorm` | modern.py | 原樣節錄 |
| 14.15 | `write_dataset` | lesson_products/data.py | 只說明並連結：重建的資料與資料 manifest 的 `config`、`files` 相同，`b.jsonl` 每列的 `split`、`world`、`events` 是什麼 |
| 15.11 | `MoEFFN` | modern.py | 原樣節錄 |
| 16.5 | `attention_mask` | attention.py | 原樣節錄 |
| 17.5 | `quantize_symmetric` | quantization.py | 原樣節錄 |
| 17.7 | `QuantizedLinear` | quantization.py | 原樣節錄 |
| 17.8 | `pack_int4`、`unpack_int4` | quantization.py | 原樣節錄 |
| 17.9 | `replace_linear_layers` | quantization.py | 原樣節錄 |
| 17.14 | `fake_quantize` | quantization.py | 原樣節錄 |
| 18.8 | `distillation_kl` | alignment.py | 原樣節錄 |
| 18.9 | `distillation_loss` | alignment.py | 原樣節錄 |
| 19.2 | `LimitedAssistant` | selftrained/model.py | 精簡版 |
| 19.2 | `SelftrainedConfig` | selftrained/model.py | 只說明：和預設值不同的四項與其餘預設 |
| 19.3 | `read_records` | selftrained/dataset.py | 只說明並連結：依序檢查必要欄位、`id`、`split`、`group_id` 與消息格式，同一個 `group_id` 出現在兩份資料就報錯、不回傳任何記錄 |
| 19.5 | `RecordEncoder` | selftrained/dataset.py | 精簡版 |
| 19.5 | `CharacterTokenizer` | selftrained/tokenizer.py | 只說明並連結：特殊 token 的編號與 `build` 收哪些字 |
| 19.6 | `decode_ocr` | selftrained/model.py | 原樣節錄 |
| 19.7 | `parse_tool_call`、`execute_tool_call` | selftrained/tools.py | 原樣節錄 |
| 19.14 | `family`、`split_of` | lesson_products/data.py | 原樣節錄 |
| 19.15 | `LoopedSolver` | lesson_products/model.py | 原樣節錄 |
| 19.16 | `CandidateScorer` | lesson_products/model.py | 原樣節錄 |
| 19.16 | `candidate_batch`、`grouped_scores` | lesson_products/model.py | 只說明並連結：怎麼檢查候選、每個候選排成一列，分數怎麼排回各題 |
| 19.17 | `rule_depth` | lesson_products/inference.py | 原樣節錄 |
| A.3 | `lexical_terms`、`retrieve` | retrieval.py | 原樣節錄 |
| B.2 | `call_tool` | retrieval.py | 原樣節錄 |
| B.4 | `tool_loop` | retrieval.py | 原樣節錄 |

節錄與精簡版裡用到、但沒有另外展示的常數與零件（例如 5.2 的 `IGNORE`、19.7 的 `CALL_KEYS`、19.2 的三個感知入口），在同一節的文字裡說明用途，不另列。

下列各節沒有 Python 程式區塊，各節後面是維持文字的理由：

- 11.17：機制已在 [11.14](../../course/chapters/11.md#11.14) 用程式示範：左右對調的兩張圖，4×4 平均摘要相同、完整像素序列不同；本節是設計例。
- 12.16：機制已在 [12.9](../../course/chapters/12.md#12.9) 與 [7.1](../../course/chapters/07.md#7.1) 用程式示範：`<audio>` 換成聲音特徵，依角色決定哪些位置計分。
- 19.1：需要下載權重：重做本節的回答要先從 Hugging Face 匿名下載公開 MoE 權重，命令在本機終端機執行。
- 19.8：機制已在 [13.5](../../course/chapters/13.md#13.5) 與 [13.13](../../course/chapters/13.md#13.13) 用程式示範：DPO 的 `dpo_loss`、PPO 的 `ppo_clipped_objective`；本節只談成品要選哪一種。
- 19.11：需要下載權重：本節的 CPU 示範要先從 Hugging Face 下載公開推論權重。
- 19.13：機制在 [19.2](../../course/chapters/19.md#19.2)、[19.15](../../course/chapters/19.md#19.15) 與 [19.16](../../course/chapters/19.md#19.16) 用程式示範：三個成品的模型各在其中一節節錄或精簡；本節只介紹三者的用途與輸入輸出。
- 19.18：內容是實測結果：表中的數字取自實驗紀錄與 CPU 執行紀錄。
- 19.19：需要下載權重：命令先下載 Looped 解題器與機率判斷器的權重，再在本機推論。
- 19.20：內容是實測結果：驗收表是設定凍結後首次測試的成績。
- 20.2：需要下載權重：試用要先下載兩個成熟模型，共約 5.48 GiB。
- 20.4：需要下載資料包：例子是第 20 章固定資料裡的真實照片，照片要照 [DATA.md](../natural-assistant/v4/DATA.md) 的〈取得固定快照，核對後再使用〉下載。
