# 8.7 獨立核對筆記（2026-10-05）

審閱身分：`/root/phase4_factual_coordinator/factual_8_7`，全新獨立單節技術審閱。未讀任何舊技術／讀者報告正文、結論或來源庫摘要。本輪讀取目前 8.7 全節（原始 UTF-8 bytes，原稿行 215–247）、其圖與圖說；前置 8.3 與 8.6 全節；8.8 的首個 LoRA 示例與部分補充僅為了解相鄰語境，未審核該節。完整讀 T.5 入口與固定 style 分支。不是章節首節，未宣稱審閱本章導言。

## 本人讀取的原始權威資料

- Ouyang 等，Training language models to follow instructions with human feedback，arXiv:2203.02155v1，2022-03-04，https://arxiv.org/pdf/2203.02155v1 。原 PDF 首頁的 arXiv 版本、作者、OpenAI 機構與題名已親讀；`pdftotext -layout` 後讀 §3.6（PDF 頁 9–10）、Table 3（頁 9）及 Figure 10 的標註規約（頁 37）。§3.6 說明 preference 是主要評測之一，並另測 truthfulness、限制與其他特定代理指標；Table 3 分列 overall quality、correct instruction/task、constraint、hallucination；Figure 10 要求準確資訊、完成使用者意圖、清楚語言，且明示取捨依任務。這支持分開報告評測面向及說明判準，並不提供本課 0.2／2 的係數，也不直接驗證這七题的模型結果。來源本身是作者原論文。原 PDF SHA-256 `c1984bb50a5b90fddb895fdc3a0f72e5bc977148c9f63ef6040cbe7a3e1f0d98`，僅利用 locator-only index 找該原始 PDF，沒有看旁邊 notes/reports。另親取得 arXiv v1 原頁（HTTP 200），讀 `<title>`、明示 v1、提交日期與 `/pdf/2203.02155v1` 連結，確認來源版本；該頁只是來源身分證據，概念判斷仍來自親讀 PDF。
- PyTorch 官方 `torch/nn/modules/loss.py`，immutable revision `5c4886908584029761b579af026dcfb627c84070`，https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/torch/nn/modules/loss.py 。親讀行 1200–1243 的 CrossEntropyLoss 原始文件／公式：logits、class-index target、ignore_index 的指示函數、sum 與 mean 分母；沒有 class weight 時為有效位置的負自然對數機率平均。本人由原始 cache 取得 bytes 後，另從 immutable URL 親取得 HTTP 200，與 cache 原 bytes 完全相同，SHA-256 `415e4ccbbab63b9cc2b79f09a338cca6fa191c07138a4fb419e3c841c4748ae0`。本次安裝 2.14.1+cpu 的 `torch.version.git_version` 恰是這個 revision。這支持平均 loss 的定義，沒有把 loss 說成答案正確率或將長回答一律視為較低 loss。
- CPython 官方 `Doc/library/functions.rst`，tag `v3.13.5`，https://raw.githubusercontent.com/python/cpython/v3.13.5/Doc/library/functions.rst 。本人親取得 HTTP 200 並读 `print` 行 1603–1627 與 `round` 行 1760–1786；後者返回四捨五入後的數值（平手取偶數），不修改輸入數值。本 fence 在 `print` 參數中呼叫 `round(total,1)`，未用其返回值重新賦值或排名，故只改展示精度。官方版本與本次 Python 3.13.5 相同。list/dict/迴圈、讀分項、算總分與輸出作為一個相連程式 coverage claim，以整個原 fence 執行核對，沒有逐句拆成多個瑣碎 API claims。

## 原始實作與實測版本

親讀現行 `tiny_perceptron/data.py` 的 ByteTokenizer、render_chat，`tiny_perceptron/model.py` 的 loss_sum／masked_loss；讀 `scripts/course_experiments/common.py` 的 text_examples、_nll、evaluate_lm，`behavior.py` 的 _style_record、_style_metrics、run_style，以及 `run.py` 的 CPU/CUDA 裝置契約。

原始實測是 `docs/course-experiments/results/style.json`，revision `ae7bbbf95537d228a44810041d2a9e978360d369`，seed 42、Python 3.13.3、PyTorch 2.14.1+cu126、NVIDIA L4、step_scale 1、complete_run。本人親讀 JSON metadata 與 default_style_runs.concise/vivid.after.test 的所有七題、其 nll_sum、effective_tokens、原 generated_ids 及 rubric。另以 `git show` 讀當時的 behavior/common/text/data/model 原始碼，五檔 SHA 均與該 JSON 的 code_sha256 完全一致。現行 behavior 新增隱藏控制 ID 檢查；歷史報告的規則沒有該檢查，故本輪核算使用歷史函式，並獨立確認本次十四份生成都無非法控制 ID、UTF-8 bytes 可還原且 EOS 正常。本輪不宣稱歷史評測包含現行新增檢查。

原訓練 run_style 契約是同一內容模型分別 deepcopy，短答與固定積木句各訓練 450 步。T.5 的 GPU 完整配方僅查閱，沒有執行訓練、data/model 下載、GPU、付費、上傳。8.7 沒有 shell fence。本次 CPU 代替核對使用原 fence、短有界 logit 計算及原 JSON 核算，沒有保存或取得神經 weights。

原始 JSONL 的檔案指紋並非 `_digest` 的 canonical-JSON 指紋：`_save_splits` 對逐筆 `json.dumps(row, ensure_ascii=False) + '\n'` 的實際檔案 bytes 做 hash。本人的初次 audit script 錯用 `_digest`，因缺少它的依賴 `_json_bytes` 而失敗；初次 stderr/stdout 保留在 results/verification-attempt1.*。查閱真正 `_save_splits` 後按其序列化契約修正自己的 audit script。重建短答／長答 test JSONL SHA 完全等於原 report 的 data.test.sha256，修正後全部 assertions 通過。此為審閱工具修正，不是教材或原實驗的缺陷。

## 逐項結果與界限

1. 手寫分項固定；數學總分為短答 `w+1`、錯答 `3w`。差為 `1-2w`，w=0.2 時 1.2／0.6、w=2 時 3／6、w=0.5 時精確平手 1.5／1.5。0.49999 與 0.50001 的展示都可四捨五入到 1.5／1.5，但原數值排名相反，印出精度不能自行變成新的判準。誤差容忍 1e-12；0.5 的平手精確相等。没有梯度、参数更新器或模型訓練，評分係數與可訓練模型參數的區分正確。
2. Synthetic CPU logits 的形狀是 `[1,N,2]`（batch、有效位置、類別），固定算術位置 target=0 而 argmax=1，該位置 NLL 一直為 4.0181499279178094 nats。由 2 個有效位置變為 44 個、只加容易預測的 target=1，平均由 2.0181499279178094 降到 0.10905901882690068 nats/有效位置；同一算術位置仍錯。另加 -100 位置不改有效分母或總和；全 -100 时 helper 明確 ValueError。这只是平均組成機制；没有證明該歷史模型的 token loss 逐項值，也沒有隔離風格訓練的因果效應。
3. 原七題是同樣未訓練的加法題。短答 target 各有 1 byte 數字 + EOS＝2，總計14；長版每題 43 byte 回答 + EOS＝44，總計308。目標分母由理想回答而不是生成字數、完整回答數或訓練步數決定；短答有兩筆生成 `10` 仍不改目標分母。nll_sum／分母＝8.413567679268974／0.26804629239169037，打印五位小數容忍 5e-6，與8.41357／0.26805一致。nats 由原自然 log cross entropy 公式確認。本輪只重新核算原結果，沒有重訓／重推理来制造成績。
4. 兩支的七個生成數字全錯，內容0／7；短答數字格式與積木固定詞句各7／7。歷史 rubric 的 vivid style_correct 是指定固定詞句 substring；content_correct 是全形逗號前數字是否等於加法真值。它沒有評「洞察深度」「有助理解」或比喻跨題材恰當性。正文把7次固定句出現限定為固定寫法，並要求若要驗證助理解须另讀語意與關係，與證據範圍一致。
5. 圖本人以 Inkscape 1.4 實渲染到640與390像素寬，使用 view_image 看兩張 PNG。上下兩份回答風格1／3、正確1／0固定；下面0.2的1.2>0.6和2的3<6正確、中文字完整可讀，沒有實測圖／能力軸／百分比等混淆。圖是手算分數單位，不是準確率。本輪不宣稱完整桌面／手機網頁瀏覽檢查。

判定：pass。没有未解實質主張，也沒有修改正文、圖或執行 commit。可重現命令、原 fence、stdout/stderr、裝置與版本、來源 bytes 與各檔 hashes 皆保存在本節正式 artifact 目錄；checker 只核對格式及版本，不代替以上本人判斷。
