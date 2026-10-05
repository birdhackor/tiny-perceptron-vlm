# 指令遵循：三個新增小節的第一階段來源筆記

查閱日期：2026-10-05。範圍：InstructGPT、FLAN／instruction tuning、IFEval 原論文與官方驗收實作，聚焦內容、指定範圍、格式和結束。這是新增三小節的查證與規劃；不新增課文、不訓練、不製作資料、不實作驗收器。

## 1. 新增內容與既有內容的界線

原書7.1–7.5已有對話格式和有效監督，7.9有EOS／輸出上限，7.11有接續SFT，7.17–7.18有預訓練與後訓練；8.3–8.5已比較文風、依要求切換與JSON格式／答案正確性，8.10–8.12已有評分者問題。三個新增小節不再完整講一次SFT、RLHF或JSON語法。

建議補的是三段銜接：**認出資訊→依指令選擇回答範圍；已知SFT流程→示範確實涵蓋目標要求；單項規則通過→完整任務聯合驗收與檢查器界線。** 主文使用幾筆手寫對照即可說清楚，不需用沒有教學價值的短訓失敗重新證明已成熟的差異。

主例先是一張文字卡，明示「顏色紅、形狀圓」。同一事實在「只答顏色」「只答形狀」「只回指定欄位JSON」下，正確交付物不同。若後續銜接圖片／OCR，先保留識別成績，再檢查指令遵循；下列文字論文不是本成品視覺能力的證據。

## 2. 原論文查證

### I01：InstructGPT 已將正確任務、明確約束與內容真實性分項

來源：Ouyang et al., [*Training language models to follow instructions with human feedback*，arXiv:2203.02155v1](https://arxiv.org/pdf/2203.02155v1)。重點：§1；§3.1「High-level methodology」及Figure 2；§3.2「Dataset」；§3.6「Evaluation」；**Table 3、§4.1及Figure 4**；§4.3；Appendix B.2與Table 10。

- 支持：§1明確區分網頁下一token預測目標與「helpfully and safely follow user instructions」的目標。模型有內容知識／續寫能力，不表示會以使用者要求的方式交付。
- Table 3分開保存「Fails to follow the correct instruction / task」「Satisfies constraint provided in the instruction」「Hallucination」。§4.1及Figure 4也分項分析；明確約束的例子包括「兩段或更少」。這支持教材分開評估答哪項任務、是否符合要求、內容是否正確，不將其壓成「會／不會回答」。
- §3.1 Step 1以輸入prompt分布上**desired behavior的示範**做SFT；§3.2區分SFT demonstration、RM比較與沒有人工答案的PPO prompt。這支持「新增SFT要用實際執行目標要求的prompt／target」，不能把任何答對內容的句子都當作相同目標示範。
- §3.6說明labeler的理解可能與使用者意圖不同；Appendix B.2的標註準則還曾演變。規則和人工偏好評分也各有可測與不可測的範圍。
- §4.3明確記錄多個明確約束及指定句數仍可能失敗。不能把instruction tuning或RLHF描述成保證服從所有格式／範圍要求。
- 不支持：「內容知識與遵循能力在模型內完全獨立」；「所有約束都要用RLHF才學得會」；「SFT單獨一定達到InstructGPT整個SFT+RM+PPO流程的成績」；「遵循提高就等於內容真實性已被保證」。本教材可分項測量，不需宣稱已定位獨立的內部模組。

### I02：FLAN 的核心資料是有指令的input與對應target

來源：Wei et al., [*Finetuned Language Models Are Zero-Shot Learners*，arXiv:2109.01652v5](https://arxiv.org/pdf/2109.01652v5)，§2.1「Tasks & Templates」、Figure 4；§2.2「Evaluation Splits」；§2.3「Classification with Options」；**§4.3「Role of Instructions」及Figure 8**；§4.4。

- 支持：§2.1將62個資料集轉為帶自然語言指令的任務，每個資料集寫十個模板，input的指令與target共同定義模型要做什麼；部分模板還反轉任務，如情緒分類改成產生評論。
- §2.3將分類選項放入input，使模型知道期待哪些回答選項。內容理解／分類判斷與怎樣輸出其結果，在資料契約中都要明確。
- §4.3比較無指令、只加資料集名稱、有自然語言指令的微調；其設定中有指令表現更好。可教「輸入要帶本次要求，答案示範要真的完成它」，不要把prompt中新增要求視為在target之外自然長出的保證。
- §2.2按任務群組留出評估，§4.4另討論few-shot示範。**原論文也支持對未訓練過的任務／措辭泛化**，因此不能把「每個新句子都必須有一筆一模一樣的SFT示範」寫成必要條件。
- 本課合理的較窄資料要求：若要靠一組新增SFT教會限定選擇器／schema，就應在該組prompt與target中看得到並驗收這些要求；這是可核對的訓練設計，不是原論文證明「沒有專用示範的模型不可能遵循」。
- 不支持：「有任務指令就一定得到任意精確schema」；「zero-shot任務平均成績能代替正常結束／裸JSON／只有指定欄位的驗收」；「FLAN結果可直接預測本課小模型數字」。

延伸來源：Chung et al., [*Scaling Instruction-Finetuned Language Models*，arXiv:2210.11416v5](https://arxiv.org/pdf/2210.11416v5)，§2.1「Finetuning Data」及Figure 3、§2.2–2.3。該文使用多種instruction／few-shot／CoT格式，並明示一部分few-shot資料不含指令。可支撐資料格式和任務多樣性的重要性；三個新增小節不必再展開CoT、1,836任務配方或模型規模比較，也不能說所有instruction-tuning樣本都必須同一模板。

### I03：IFEval 能驗收可檢查要求，不是內容正確性的通用裁判

來源：Zhou et al., [*Instruction-Following Evaluation for Large Language Models*，arXiv:2311.07911v1](https://arxiv.org/pdf/2311.07911v1)，§1；**Table 1；§2.1–2.2；§3及Table 3；§4**。

- 支持：25類可驗證指令、541個prompt；同一prompt可能有多條要求。Table 1涵蓋字數、段數、JSON、指定選項、禁用詞與**End Checker：「以這句結尾，後面不再加字」**。
- §2.1同時處理要求衝突和措辭多樣性，最後逐筆人工檢查。教材也應先確定題目要求相容；「只回裸JSON」不能再要求JSON外額外結尾句。
- §3明確定義prompt-level accuracy為**該prompt全部可驗證要求通過**，instruction-level accuracy則逐條計算。某題4項只做到3項，可有3/4分項表現，但整題沒有完成全部要求。
- §2.2提供strict與loose。loose可移除Markdown修飾、第一行或最後一行及其組合，以降低false negative；原文明說這也可能引入false positive，例如去掉第一行後字數才符合。
- 能力界線：這些規則檢查規定的輸出特徵，不自動檢查事實、識別結果、推理和完整語義。例如`{"color":"blue"}`可符合JSON格式但顏色錯；指定結尾句存在也不代表前面真的完成任務。
- 不支持：「541個prompt涵蓋所有指令」；「規則都是百分之百客觀而且沒有邊界案例」；「strict必定等於下游程式的字串契約」；「IFEval文字評測已驗證圖片／音訊指令遵循」。§4把擴大指令多樣性與多模態列未來工作。

## 3. 原作者官方驗收程式：strict 仍要看每條規則的契約

讀取並固定 Google Research repository commit：`e49bbfe381c9c0e564b937f1c4e163a2273c65cc`（repository commit日期2026-09-30）。這是本次取得的官方實作版本，不將其與2023年實驗當成完全同一份執行結果。

來源：[evaluation_lib.py](https://github.com/google-research/google-research/blob/e49bbfe381c9c0e564b937f1c4e163a2273c65cc/instruction_following_eval/evaluation_lib.py)，`test_instruction_following_strict`、`test_instruction_following_loose`、`print_report`；[instructions.py](https://github.com/google-research/google-research/blob/e49bbfe381c9c0e564b937f1c4e163a2273c65cc/instruction_following_eval/instructions.py)，以下類別。

| 官方檢查器 | 實際判斷 | 本課若需要更窄契約，不能省略的部分 |
| --- | --- | --- |
| `JsonFormat` | 去掉可選Markdown code fence後`json.loads`可解析即可 | 不保證裸JSON、根節點為object、欄位集合、型別和欄位值正確。官方description明確允許Markdown ticks，不是程式錯誤。 |
| `ConstrainedResponseChecker` | 指定選項**包含**在回答文字中即可 | 若要求「只有該選項」，子字串搜尋不足；多餘解釋或多個選項不能因此通過。 |
| `EndChecker` | trim空白、外側雙引號，轉小寫後檢查suffix | 若使用者要求大小寫精確、禁止額外引號，需另定規則；更不能從文字suffix推斷模型自發EOS。 |
| `ResponseLanguageChecker` | 語言偵測器判斷主要語言 | 單一語言標籤不等於每個字都符合「ENTIRE response」的要求；很短回答和混語內容有邊界。 |

`strict`外層把每條`check_following(response)`的布林值記錄下來，再取`all`；是否真的嚴格，要看各檢查器定義。`loose`對每條指令逐一找八種變換中是否有一種可通過，不能將清理後通過改寫成原始完整輸出已符合接口。

本課應借用「可解釋規則＋逐項記錄＋整題全部通過」這個方法，依有限成品契約制定規則，並保留未修改原始回答。不要將只搜尋正確字詞／某個欄位／結尾短語稱為完整驗收。這是來源分析，現在不複製或改寫驗收實作。

## 4. 正常結束要再區分文字要求、模型EOS與程序截停

補充官方來源：Transformers **v4.57.1** [`GenerationConfig`文件註解](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/generation/configuration_utils.py)，「Parameters that control the length of the output」的`max_new_tokens`、`max_length`及`stop_strings`；`eos_token_id`；**`forced_eos_token_id`**。文件明確說後者可在`max_length`到達時強制生成EOS。來源版本與上下文筆記一致。

- 文字要求：「最後一句為某句，後面沒有別的文字」可以檢查回答suffix。這是IFEval End Checker的範圍。
- 模型結束：本課已定義的assistant結束token應由模型在內容完成後預測；保留原始生成ID與停止原因才能查證。單看解碼文字或最後一個EOS不足，還要排除強制EOS、stop string等程序控制。
- 預算截停：達到`max_new_tokens`只是生成預算用完；不證明模型學會何時結束。反之，模型提早輸出EOS也不證明所需內容齊全。

若新增SFT想教會本課所稱的正常結束，後續資料審查應檢查示範是否在正確位置結束、結束目標是否留在有效監督中、是否遭截斷；這是本課資料契約的審查要求。InstructGPT／FLAN／IFEval三篇本身沒有證明本專案的EOS mask實作正確，不能用它們替代第五階段的實際資料與runtime核對。

教學只回連7.3–7.4和7.9，不再新增一套EOS原理課。驗收時把「內容完整而結束」「內容未完先EOS」「程序到上限截停」「外部規則強制停止」分開記錄，不靠增加輸出上限掩飾結束問題。

## 5. 三個微小節的新增建議

以下只定內容與紙上例子；不建立訓練集、評測集或程式。

| 新小節 | 主要問題與必要內容 | 手寫例子／圖的建議 | 權威來源與回連 |
| --- | --- | --- | --- |
| 1. 認出內容之後，為什麼還要選回答範圍？ | 事實判斷、選擇器要求與交付物不同；正確但多答仍可能未遵循 | 同一「紅色、圓形」卡片接不同要求，畫內容箭頭和選擇要求箭頭匯合到回答；不宣稱模型內有兩個獨立模組 | InstructGPT Table 3／Figure 4；FLAN§2.3；回連8.1、8.4 |
| 2. 示範怎麼包含我們想教的要求？ | 已知SFT流程下核對prompt帶要求、target執行要求、正常結束仍受有效監督；用同內容不同要求避免只靠題材猜輸出 | 「只答顏色」應示範顏色；「只答形狀」應示範形狀。若全部target都寫完整描述，內容正確也不是這組要求的正確示範；留出新措辭與新組合 | InstructGPT§3.1–3.2；FLAN§2.1／§2.2／§4.3；回連7.3–7.4、7.9、7.11，不重講SFT／RLHF |
| 3. 怎樣確認整個要求都完成？ | 分項核對內容、範圍、格式、結束，整題取全部必要項交集；規則能查有限契約而非一般理解 | 同一題四項對照表：內容答對但多答、格式對但值錯、全部文字對但被上限截停、完整通過；保留原始回答再討論判斷 | IFEval Table 1／§2.2／§3與官方checker；回連8.5、8.10–8.12 |

分項與聯合完成率均應附題數，先測單一要求，再組合相容要求；留出新措辭與內容／要求的新組合。不能只報格式通過或只用內容substring命中，宣稱整體指令遵循已學會。上述驗收原則由成熟來源支持，本課局部小實驗只驗收自己的有限成品。

## 6. 查證限制與追溯

- 本次直接讀指定原論文章節及官方checker，未重新執行原作者模型、評測或任何訓練。2022／2023的模型成績不作2026產品排名或本成品預期數字。
- InstructGPT分項評估支持內容／要求的測量差異，不證明某項能力必須放在獨立網路；FLAN支持訓練中指令與target的設計，不證明每個新要求沒有專門示範就不可能泛化。
- IFEval的文字布林規則不直接測自然EOS、圖片識別、事實真偽與完整語義。官方JSON／結尾規則也不等於本課較窄的接口要求；主文引用時須跟著說明。
- 第五階段才核對選定成品資料的目標要求、有效監督、停止token／stop reason與真正留出題；本階段沒有開始這些工程工作。

原始閱讀檔暫存在`/tmp/instruction-sources-20261005/`；本任務在repository只新增此筆記。取得PDF的頁首確認版本如下；SHA-256可用於核對實際閱讀內容。

| 原論文／版本 | SHA-256 |
| --- | --- |
| InstructGPT 2203.02155v1 | `c1984bb50a5b90fddb895fdc3a0f72e5bc977148c9f63ef6040cbe7a3e1f0d98` |
| FLAN 2109.01652v5 | `7126b83c5590b9eb4b6e6027446cb88c0aeff132e47d88e9407982b1f7fad044` |
| Scaling Instruction-Finetuned 2210.11416v5 | `771f758c1b711c2a63ca2439e80ab90751351d721632897a058c0205ba9e2a22` |
| IFEval 2311.07911v1 | `5d7cedfd762af6d9ad13d8768de64fa65895b8ebc3f5e454454c5ffbd48124f2` |

上述固定commit的官方程式SHA-256：`evaluation_lib.py`為`35decc06000718487f44d7deafa6d3f48a8ec0886281edf40162c0265b7d248c`；`instructions.py`為`60e086f5342a03ce8e18b64bbcccf86308f523c08aa826707a562150a52f3edf`。它們與最初master取得的內容一致，未將可變branch當成固定版本。
