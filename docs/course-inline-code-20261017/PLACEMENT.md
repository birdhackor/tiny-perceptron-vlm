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
| 19.5 | `RecordEncoder` | selftrained/dataset.py | 精簡版 |
| 19.5 | `CharacterTokenizer` | selftrained/tokenizer.py | 只說明並連結：特殊 token 的編號與 `build` 收哪些字 |
| 19.7 | `parse_tool_call`、`execute_tool_call` | selftrained/tools.py | 原樣節錄 |
| 19.14 | `family`、`split_of` | lesson_products/data.py | 原樣節錄 |
| 19.17 | `rule_depth` | lesson_products/inference.py | 原樣節錄 |
| A.3 | `lexical_terms`、`retrieve` | retrieval.py | 原樣節錄 |
| B.2 | `call_tool` | retrieval.py | 原樣節錄 |
| B.4 | `tool_loop` | retrieval.py | 原樣節錄 |

節錄與精簡版裡用到、但沒有另外展示的常數與零件（例如 5.2 的 `IGNORE`、19.7 的 `CALL_KEYS`、19.2 的三個感知入口），在同一節的文字裡說明用途，不另列。

下列 17 節沒有 Python 程式區塊：11.17、12.15、12.16、19.1、19.3、19.6、19.8、19.11、19.13、19.15、19.16、19.18、19.19、19.20、20.2、20.4、20.8。
