# 7.15 本輪親查來源與支持範圍

審閱身分：/root/phase4_factual_coordinator/factual_7_15；全新單節正確性審閱。未讀任何旧技術／讀者報告、判定或旁人筆記。親讀本輪方法、schema、section_facts helper 與 review-protocol；親讀目前 7.15 全節，另讀 7.12、7.11、5.7 的必要前置契約。7.15 無 SVG 或其他圖片引用，沒有需要渲染的實圖。

查證日：2026-10-05。外部來源均重新從原 URL 取得；HTTP URL、最終 URL、狀態、原 bytes SHA 在 sources/fetch-receipt.json。lookup-only 索引只看 URL／原文件定位，未沿用其中任何核實判斷。本輪 EWC 論文自行從 arXiv 取得，不依賴舊報告。

* Kirkpatrick 等《Overcoming catastrophic forgetting in neural networks》，https://arxiv.org/pdf/1612.00796v2。親讀 PDF 第一頁作者與 DeepMind／Imperial College 機構、第一頁 `arXiv:1612.00796v2 [cs.LG] 25 Jan 2017`、§1 首兩段、§2 前段與 Fig.1 圖說、§2.1 順序訓練與測試集比較；另核對原 arXiv abs 的 v2 提交歷史為 2017-01-25，標題與作者相符。§1 指出 sequential multiple tasks 時 task A 重要權重被改以適應 task B 可能導致遺忘；本節「可能」措辭比把遺忘寫成必然更有限。Fig.1 只提供機制示意，不能證明本課哪個參數造成哪道錯答。§2.1 以以前任务測試集的前後表現衡量舊能力保留；本課只報這一小次觀察，未假稱實作 EWC。
* CPython 官方 v3.13.5 `Doc/library/stdtypes.rst`，親讀 4611–4684 Mapping Types（dict 將 key 對應任意 objects、braces key:value 建立）、5614–5622 The Null Object（None 是單一 null object）。支持 Python 嵌套字典與 None 的語義，不賦予 None 機器學習分數含義。原 fence 只建立 dict/list 和 print，沒有模型或 optimizer。
* CPython 官方 v3.13.5 `Doc/tutorial/datastructures.rst`，親讀 12–35 `list.append`、`list.extend`、`list.insert`。支持給 A 題清單 append 一題只擴大清單。本輪 Python 3.13.5 正好同版本。
* PyTorch 官方 tag v2.9.0 `docs/source/notes/serialization.rst`，親讀 106–146 Saving and loading torch.nn.Modules：state_dict 包含 parameters / persistent buffers，load_state_dict 恢復狀態。這是概念／API契約來源，並未宣稱安裝版本也是 2.9.0。本輪執行為 torch 2.14.1+cpu。歷史 save_checkpoint / load_checkpoint 原碼 30–62、79–103 則親核序列化設定、權重與 load_state_dict(strict=True)。本節續用基模要求的是原權重起點；b-only 的 AdamW 在新任務建立新的 optimizer，沒有宣稱延續舊任務 optimizer 狀態或逐位相同軌跡。
* scikit-learn 官方 tag 1.7.0 `doc/modules/cross_validation.rst`，親讀 9–20（training/testing same data 是 methodological mistake，held-out X_test/y_test）、57–75（反覆用測試集調參洩漏，final test 與 validation 分開）。支持評估題不參與 B 更新。此來源不支持本課模型的任何成績，成績另讀原 JSON。

原實測親核：`docs/course-experiments/results/sft_ablation.json` 是 original experiment revision `52964f650787cef393d18647a350471637208f67`、seed42、complete_run、step_scale1、L4 / torch2.14.1+cu126 的已存原結果。原 SFT 結果来自 revision `a253d1262bf5f361f9ac4e19232ae752f0ecc7a3`。親讀前後 A/B 的 validation/test 原 `samples` 與 `generated_ids`、原 training 步數與受監督 token 數，另讀結果記錄的 code_sha256。用 git show 擷取前一原 revision 的 common.py / text.py / data.py / model.py / training.py 等，逐一驗證原結果的 code_sha256，相符才作重算。prepare_data.py 未被當時 runner 納入 code_sha256 範圍，但同 revision 原碼的資料重建六份 JSONL 均與原結果宣告的 SHA 完全一致。

重要結果與定義：A 原材料有 red/green/blue × circle/square × low/high 的12家族，每家族5題（describe/shape/color/pitch/joint），train45/9家族、validation5/1、test10/2；本節測試 A10 的說法是該獨立材料，不是前段兩條中文 placeholder，更不是7.12四格示例的新實驗。B 加數各0..7，64有向題按 min+max 分36家族，train49/28、validation8/4、test7/4。`2+3` 和 `3+2` 同訓練家族，所有 split 家族零交集。基模 A test 的原樣本與 original sft.after.test 完全相等；歷史 text.py 617–623 明確 load sft checkpoint、deepcopy(base)、只 B train 更新。

每一道重算 `exact` 採 EOS 以前 raw token IDs 等於正解 ByteTokenizer IDs（不是僅比較 decoder 文本）。EOS 另外記；因此 raw [3,65,2] decode 成9仍不 exact，而 [65] 可以 exact 但沒有EOS，已用歷史 evaluator/generator + 明示 scripted logits 短CPU變化親跑。原 A before5/10 after0/10，B before0/7 after0/7，兩邊測試 EOS 都全數正常。circle 問題生成 [65,2] 是9+EOS，支持內容錯，不是僅多一句禮貌話；沒有對風格品質的獨立評分。

NLL 重算用原 recorded nll_sum / 由原 render_chat 重建答案bytes+EOS的 token數：B test before98.81486511230469/14=7.058204650878906，after64.08580017089844/14=4.5775571550641745。五位小數各容忍0.000005；A test分母69。這不是每題平均，也不是生成正確率。B train完整單遍答案+EOS113tokens，500次每批16筆為8000次有放回抽樣，重播原 sampler 而不更新模型，受監督 token 恰18453，匹配原結果。train NLL8.001003653602263→0.10324167947705208，但 B留出 exact未提升。

支持限度：CPU只執行原placeholder、A append練習、歷史資料切分／原結果聚合／抽樣token分母、以及明示 scripted logits 的評分契約。未重新訓練、未進行原權重推論、未下載資料或模型、未存神經權重。NLL重算是原 nll_sum 的聚合核對，沒有宣称新重做 logits。單次小資料觀察只支持舊題 exact下降，不支持新能力學好、普遍必然遺忘、某特定權重的因果，或內容與一般風格能力一併消失。
