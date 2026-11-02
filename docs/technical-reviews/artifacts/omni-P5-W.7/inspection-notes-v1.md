# W.7 獨立技術核對記錄

Reviewer：`/root/omni_p5_technical_w_7`。只判定 NUMBERED freeze v3 的 W.7；未讀作者處置、reader 答案、其他 technical 報告或 coherence 判定。必要前置只讀 W.1、W.2、5.7 與 training 的命令約定；表列其他節只用來核對檔案索引承諾，未把它們當作另一次完整技術審閱。

## 檔案對照的人工判讀

`inspection-v2.json` 保存 17 列、69 個小節引用（含範圍展開與跨檔重複引用），以及獨立掃描到的 56 個「節錄自／精簡自」標記。所有標記都有被表格涵蓋。各列的最早列出小節依順序為 1.2、2.5、3.6、4.4、4.5、5.6、6.6、6.6、8.8、10.1、11.14、12.15、13.10、14.15、17.5、19.21、A.3，與聲稱的閱讀順序相符。這是「節錄或點名該檔並解說零件」索引，不聲稱每個 import 都列入，也不聲稱列出套件每個維護檔。

- data：split_documents、shifted、fingerprint、ByteTokenizer、render_chat、pad_batch、toy_conversations 的各節錄 AST 均對得上實檔。
- simple：2.5 點名 simple.py／ContextMLP，解說 embedding、拼接、hidden tanh、output 三部分；不是只 import。
- attention：3.6 manual_attention 與 16.5 attention_mask 對得上實檔；3.7 清楚標示 CausalAttention 的簡化與省略。
- modern：4.4 DenseFFN、14.1 rope 標明簡化；14.2 RMSNorm、15.11 MoEFFN 為相同 AST 的節錄。
- model：4.5 Block 與 4.6 TinyLM 標明簡化；5.2 loss_sum、masked_loss 對得上實檔。
- training：5.6 learning_rate 節錄相同；5.7 點名 training.py 並以欄位表解說 save_checkpoint 的模型、optimizer、step、RNG、metadata 等狀態。實檔的 save_checkpoint 建立 parent，使用暫存檔再 replace。
- capstone、tokenization：6.6 分別點名 capstone.py 的 tokenizer state 存取檢查與 tokenization.py 的 ByteLevelBPE 普通文字／控制 ID 邊界；實檔確有相應檢查，不只是匯入。
- alignment：六個引用小節分別節錄 LoRALinear、reliability_bins、sequence_log_probability、dpo_loss、distillation_kl、distillation_loss，對得上實檔。
- multimodal：八個引用小節涵蓋 scene、patchify/unpatchify、VisionEncoder、expand_modalities、MultiModalLM、tone、mel_filter_bank、log_mel/AudioEncoder。MultiModalLM 明標簡化，其餘標記的節錄相同。
- natural_concepts：11.14 picture_order_report/swapped_picture、11.15 thin_stroke_report、11.16 text_error_report/edit_distance、12.14 speech_stages 均有點名同檔並解釋輸入、計算或回傳欄；edit_distance 的節錄相同。
- selftrained：12.15 明確區分同名 AudioEncoder，解說時間壓縮與池化；19.2 LimitedAssistant、19.5 RecordEncoder 標明簡化；19.3 read_records 的 group_id 交叉切分檢查、19.6 decode_ocr、19.7 parse_tool_call/execute_tool_call、19.9 SDPA 運算路徑都有實檔定位與解說。19.6 的實檔 model.decode_ocr 與節錄相同；dataset.py 中同名包裝函式不是被節錄的版本。
- posttraining：五個引用小節涵蓋 preference_loss/FiniteRewardModel、bandit_advantage、ppo_clipped_objective、exact_kl、FiniteResponsePolicy/FiniteValueModel，標記的節錄相同。
- lesson_products：14.15 連到 data.py 並解說 write_dataset 重建後的 split/world/events；16.15 點名固定版 model.py 的 CandidateScorer/Block，正文解說舊 C 雙向混合的限制。19.14 family/split_of、19.15 Block/LoopedSolver、19.16 CandidateScorer、19.17 rule_depth 節錄相同，19.17 另逐步解說 inference.py 的 auto_solve。
- quantization：quantize_symmetric、QuantizedLinear、pack_int4/unpack_int4、replace_linear_layers、fake_quantize 都有實作節錄且 AST 相同。
- omni_products：19.21 的 AVConfig、mel_bank、local_audio、TimeAttention 與 AVModel 的初始化及 forward 內建；AVModel 只略去不影響這些計算的 config_sha256。19.22 內建 Event/validate_program、CodecConfig、SoundCodec 的初始化、量化、forward、encode/decode/chunks，及 program_schedule、ProgramTalker；只略去訓練維護用 reset_unused_training_codes。19.23 的 text_bytes、Candidate、Question、pad_bytes、collate_questions、ChoiceConfig、StateMemory、ByteEncoder、ChoiceModel 全部 AST 與實檔相同。所謂完整計算支持表列影音分支、聲音碼／Talker、候選判斷的教學與正式推論計算，不把資料搬運、訓練排程或發布工具也宣稱內嵌。這三節的程式沒有 import 專案內隱藏的模型計算。未執行這些正式模型。
- retrieval：A.3 lexical_terms/retrieve、B.2 call_tool、B.4 tool_loop 的節錄相同。

## 正文操作與過渡

先有檔案用途分類，再說起點，再用唯讀 Path 程式看結果，最後用錯誤與 dry-run 報告辨認實際做到哪一步；本次未發現技術上的跳步或互相矛盾。W.1 已給出 clone 後的資料夾名、uv 建立環境、直接 interpreter 路徑、啟用與 Tiny Perceptron kernel；W.2 已教 list/key/index，5.7 已教 checkpoint。展開表沒有成為操作程式的先決條件。

實驗的原段程式在專案根目錄與 notebooks 各以一般模式及互動模式執行，得到 True/True 與 False/False。例外各自以刻意失敗的獨立程序重現，保留檔案、行號與最後原因。ModuleNotFoundError 的程序用 -I -S 刻意不載入 site-packages，沒有卸載 torch；正常 CPU venv 有 torch。Windows 路徑僅做字面與 PureWindowsPath 檢查，未冒稱實際執行 Windows。

uv 0.12.19 的 temporary-project 離線 dry-run 明說會建立 `.venv`；沒有建立環境或 lockfile。這不是安裝重現或套件相容性證據。本機 `data/` 尚未建立不構成矛盾：表格說明資料產生後的用途，不保證 clone 即帶本機原始資料。

小型合成文字 dry-run 使用全新 8 寬、1 層、1 頭模型，一批、16 最大長度、CPU。攔截 AdamW.step 與 save_checkpoint，若試圖更新／存檔即失敗；實際兩者呼叫數為零，最後 JSON 是 dry-run-no-weight-update。它確有 backward 與非零梯度，不能稱模型已學會；未執行 --train 或任何固定 final-family 模型。

無 W.7 SVG、論文引用、速度／品質實測主張。沒有看桌面／手機實際頁面，也沒有驗 Windows、macOS、MPS、GPU、Colab、首次安裝、下載或發布。這些界線不影響本節已核實的本機路徑示例與軟體行為；不能把本報告外推為整頁或平台驗收。
