# 實作放在哪一節

下表列出正文程式 import、卻沒有在正文展示的 `tiny_perceptron` 頂層定義，以及建議展示的小節與寫法。行號與行數是本輪開始時（`b05ebda3`）的狀態，由 `ast` 解析得到；寫法欄是建議，作者可依本節教學需要調整，但要在報告說明理由。

寫法：**節錄**＝原樣節錄；**精簡**＝精簡版加 `assert` 核對；**說明**＝只說明並連到原始碼。三種寫法的格式見[AUTHORING.md](AUTHORING.md)。

| 展示的節 | 定義（檔案:行） | 行數 | 建議寫法 | 備註 |
| --- | --- | ---: | --- | --- |
| 1.2 | `split_documents`（data.py:89） | 11 | 節錄 | 先去重複再切分，正好回答本節問題 |
| 3.6 | `manual_attention`（attention.py:21） | 8 | 節錄 | 最後兩行處理整列都不可看的情況，說明它在第 7 章的補位時才會遇到 |
| 3.7 | `CausalAttention`（attention.py:31） | 44 | 精簡 | 只留單頭、手寫路線；省略 kv_heads、sdpa、RoPE、cache、segments |
| 4.4 | `DenseFFN`（modern.py:35） | 19 | 精簡或節錄 | relu2／swiglu 兩個分支屬第 14 章；精簡時只留 GELU |
| 4.5 | `Block`（model.py:31） | 20 | 精簡 | 省略 valid、positions、segments、cache、MoE 與 auxiliary |
| 4.5 | `ModelConfig`（model.py:15） | 14 | 說明 | 列出本章用到的欄位；其餘欄位對應後面哪一章 |
| 4.6 | `TinyLM`（model.py:53） | 37 | 精簡 | 字表、位置表、blocks、最後 LN、輸出層；省略 RoPE、權重共享、cache 與多模態輸入 |
| 4.7 | `shifted`（data.py:46） | 6 | 節錄 | |
| 5.1–5.2 | `loss_sum`、`masked_loss`（model.py:92、103） | 9＋3 | 節錄 | 相鄰，可一起節錄；需要 `IGNORE` 與 `F` |
| 5.6 | `learning_rate`（training.py:106） | 10 | 節錄 | |
| 5.7 | `save_checkpoint`、`load_checkpoint`（training.py:30、79） | 37＋25 | 說明 | 說明存了哪些欄位與為什麼；連到原始碼 |
| 5.11 | `fingerprint`（data.py:102） | 3 | 節錄 | |
| 6.6 | `ByteTokenizer`（data.py:14） | 15 | 節錄 | `SPECIALS` 是常數，正文用文字說明 |
| 7.1 | `render_chat`（data.py:54） | 15 | 節錄 | |
| 7.10 | `pad_batch`（data.py:71） | 16 | 節錄 | |
| 7.11 | `toy_conversations`（data.py:117） | 6 | 節錄或說明 | |
| 8.8 | `LoRALinear`（alignment.py:11） | 16 | 節錄 | |
| 9.10 | `reliability_bins`（alignment.py:66） | 16 | 節錄 | |
| 10.1 | `scene`（multimodal.py:177） | 14 | 節錄或說明 | 讓讀者知道合成圖怎麼畫出來 |
| 10.2 | `patchify`、`unpatchify`（multimodal.py:12、20） | 6＋8 | 節錄 | 相鄰，可一起節錄 |
| 10.5 | `VisionEncoder`（multimodal.py:30） | 12 | 節錄 | |
| 10.7 | `expand_modalities`（multimodal.py:84） | 17 | 節錄 | |
| 10.8 | `MultiModalLM`（multimodal.py:103） | 33 | 精簡 | 第 10 章只走圖片路線；聲音在第 12 章 |
| 11.14–11.16 | `natural_concepts` 的報告函式（natural_concepts.py:22、32、58） | 8–13 | 說明（`edit_distance` 可節錄） | 報告輔助；字錯率的編輯距離值得展示 |
| 12.1 | `tone`（multimodal.py:44） | 3 | 節錄 | |
| 12.6 | `mel_filter_bank`（multimodal.py:49） | 11 | 節錄 | |
| 12.8 | `log_mel`、`AudioEncoder`（multimodal.py:62、72） | 8＋10 | 節錄 | 相鄰，可一起節錄 |
| 12.13–12.14 | `audio_order_report`、`speech_stages`（natural_concepts.py:70、88） | 16、9 | 說明 | |
| 13.3 | `sequence_log_probability`（alignment.py:29） | 7 | 節錄 | |
| 13.5 | `dpo_loss`（alignment.py:38） | 5 | 節錄 | |
| 13.10 | `preference_loss`（posttraining.py:8） | 5 | 節錄 | |
| 13.11 | `bandit_advantage`（posttraining.py:24） | 5 | 節錄 | |
| 13.13 | `ppo_clipped_objective`（posttraining.py:31） | 20 | 節錄 | |
| 13.14 | `exact_kl`（posttraining.py:15） | 7 | 節錄 | |
| 13.15 | `FiniteResponsePolicy`、`FiniteRewardModel`、`FiniteValueModel`（posttraining.py:53、64、78） | 9–12 | 節錄 | 程式目前只在補充區，移回正文 |
| 14.1 | `rope`（modern.py:19） | 14 | 節錄或精簡 | `positions.ndim` 的二維分支給多模態位置用 |
| 14.2 | `RMSNorm`（modern.py:8） | 9 | 節錄 | |
| 15.11 | `MoEFFN`（modern.py:56） | 29 | 節錄 | |
| 16.5 | `attention_mask`（attention.py:10） | 9 | 節錄 | |
| 17.5 | `quantize_symmetric`（quantization.py:8） | 10 | 節錄 | |
| 17.7 | `QuantizedLinear`（quantization.py:47） | 18 | 節錄 | |
| 17.8 | `pack_int4`、`unpack_int4`（quantization.py:29、39） | 8＋6 | 節錄 | 相鄰，可一起節錄 |
| 17.9 | `replace_linear_layers`（quantization.py:67） | 8 | 節錄 | |
| 17.14 | `fake_quantize`（quantization.py:77） | 5 | 節錄 | |
| 18.8 | `distillation_kl`（alignment.py:45） | 11 | 節錄 | |
| 18.9 | `distillation_loss`（alignment.py:58） | 6 | 節錄 | |
| 19.2 | `SelftrainedConfig`、`LimitedAssistant`（selftrained/model.py:25、295） | 29、163 | 說明或精簡 | 只展示與本節問題有關的部分 |
| 19.5 | `RecordEncoder`、`CharacterTokenizer`（selftrained/dataset.py:108、selftrained/tokenizer.py:12） | 134、66 | 說明或精簡 | |
| 19.7 | `parse_tool_call`、`execute_tool_call`（selftrained/tools.py:57、76） | 13、14 | 節錄 | |
| A.3 | `lexical_terms`、`retrieve`（retrieval.py:8、12） | 2＋9 | 節錄 | 相鄰，可一起節錄 |
| B.2 | `call_tool`（retrieval.py:23） | 12 | 節錄 | |
| B.4 | `tool_loop`（retrieval.py:37） | 13 | 節錄 | |

`rope` 與 `RMSNorm` 目前沒有被正文 import；第 14 章用手算教，展示函式庫版本可讓讀者接上第 4 章的 `Block`。

沒有程式的 37 節，依計畫 D5 判斷是否補程式。初步建議補：7.19、7.20、7.22、7.23、8.14、8.15、8.16、9.11、9.13、11.18、14.7、14.9、14.10、14.11、14.18、16.13、A.8、A.9、A.10、19.17；維持文字：11.17、12.15、12.16、19.1、19.3、19.6、19.8、19.11、19.13、19.15、19.16、19.18、19.19、19.20、20.2、20.4、20.8。
