# 19.4 本輪親讀與查核紀錄

Reviewer task: `/root/phase4_factual_coordinator/factual_19_4`。全新獨立單節技術審閱；未參與作者工作、第一次閱讀或其他小節技術審閱。沒有讀舊 technical/reader 報告正文、既有判定或作者修正摘要。兩份 locator index 僅讀 key/type 與原始來源的 URL/path/SHA/version 定位，並親自核對原件。

## 原稿與讀取範圍

親讀 19.4 全節，首次位置為原檔第162–216行，原始 UTF-8 SHA-256 `d64afe83caddda70057f39961176ff0e302627ecbeac70629d823487085a6c9e`。保存 `inputs/section.md`；沒有正規化換行。`inputs/frozen-19.md` 是第一次讀取時的完整檔案 bytes，SHA `3a008303054604279f49a1c00b70792db43bc4136e5dba9380990f04cced3537`，只代表 frozen input，不代表其他小節目前版本。

為確認新／舊成品的範圍，另親讀本章導言和原稿19.1全節，保存原始 bytes。只用導言「新成品尚未完成訓練與驗收」及19.1主線設計／舊合成模型的區分來界定19.4；沒有把19.1的其他能力、命令或图纳入本節驗收。主線接入自訓練感知元件是後續成品計畫，舊合成實報不是它的能力證據。

本節沒有圖片／SVG引用，也沒有需要讀圖才能確認的圖片內容或空間標籤。計分對齊由原 fence、mask 與有效目標數直接展示，階段比較由資料表列出；因此本節圖解检查為不適用，沒有宣稱做過本節圖的 rendering。

## 原始碼定位與親讀

先以 filenames 定位，再以 AST 定位 class/function；沒有廣搜 docs 中的正文。親讀範圍：

- `tiny_perceptron/capstone.py`：1–121、124–375、428–605、608–694。包含資料家族切分、byte／角色對齊、batch 的 valid 與 labels 分離、偏好分數、凍結 reference、生成終止規則、工具往返、loader/exporter 與 provenance。一般方法限制／docstring 也可讀；未使用它們作既有結果判定。
- `tiny_perceptron/data.py`：1–86；byte 對應與IGNORE=-100。
- `tiny_perceptron/model.py`：1–105；Dense／MoE選擇、logits 軸與 sum/count 交叉熵。
- `tiny_perceptron/modern.py`：35–84；Dense與MoE分派及平衡項。
- `tiny_perceptron/attention.py`：10–18、31–74；valid 決定可見 key，沒有把 loss mask 當 attention mask。
- `tiny_perceptron/alignment.py`：29–42；回答序列 log probability 與 DPO 式。
- `scripts/course_experiments/capstone.py`：1–44、70–263、280–309。實際原方法包含父檔載入／資料版本檢查、前三站 CE+0.01 auxiliary、DPO+0.2 CE replay+0.01 auxiliary、只累加 replay labels 的 effective_tokens。沒有讀 deployment 結果／解釋值，也沒有執行任何 train_stage。
- `tiny_perceptron/multimodal.py` 僅讀 AST 的函式名稱與行號，執行時被匯入；未據此判定感知效果。CPU環境檔列出實際匯入模組及SHA。

## 原始 JSON 讀取 pointers

先列四實報 root 和 `/results`、`/results/data_manifest`、`/results/inference_export`、`/results/validation_summary`、`/results/parameters` 的 key/type。只讀下列具名 pointers：`/revision`、`/device`、`/seed`、`/gpu`、`/torch_version`、`/results/data_version`、`/results/stage`、`/results/requested_steps`、`/results/steps`、`/results/new_steps`、`/results/schedule_completed`、`/results/seed`、`/results/effective_tokens`、`/results/test_evaluated`、`/results/parent_checkpoint_sha256`、`/results/inference_export`、`/results/validation_summary`、`/results/data_manifest/counts`、`/results/data_manifest/sha256`、`/results/parameters/config`、`/results/code_sha256`、`/artifacts`。未讀 author notes／scope correction／舊 review 結論。

四份 validation 先列 root、record 和 trace 的 key/type，再檢查 `/count`、`/action_correct`、`/end_to_end_correct`、`/by_task`、`/protocol` 和 `/records/*/{id,family,task,expected_action,expected_final,action_trace,parsed_action,runtime,final_trace,answer,action_correct,end_to_end_correct}`。action/final trace 讀 `/prompt_ids`、`/generated_ids`、`/raw`、`/eos`、`/stop_reason`；這些是原始樣本與判準，不是作者評語。

每份 JSON 的完整 SHA 保留；336筆既有生成紀錄逐筆解碼、檢查特殊token／EOS、與自行重建資料的答案及題目核對，再重算評分與分項。沒有載入模型重新評分。

## 原始外部來源

正文指定 `1df335318bda03fd771807f66976953231d5a00b` 的四實報、四validation和兩份capstone程式均親自HTTPS讀取，與本地原檔逐byte相同，詳 `fetch-results.json`。原始code SHA也與四實報 `/results/code_sha256` 相同。

親讀 InstructGPT arXiv:2203.02155v1（第一頁作者／版本，§3.1步驟1、§3.3 SFT，另§5.5 next-word objective），確認從 pretrained model 接續示範訓練，以及next-word objective只是使用者行為的proxy；原論文不當作本repo的實測。親讀 Switch Transformer arXiv:2101.03961v3，§2 load balancing 式(4)–(6)和α=0.01；只支持平衡項的目的，repo使用top-2版本，不宣稱完全等同Switch top-1或保證均衡。親讀 DPO arXiv:2305.18290v3，§4式(7)、DPO outline／reference初始化；本repo加CE replay，沒有稱為純DPO或人類偏好結果。

親讀與已安裝PyTorch git revision一致的官方source：functional.py cross_entropy 3478–3533（ignore_index、sum、shape）、autograd.md 194–230（freezing）、random.py 49–76（manual_seed）、_tensor_docs.py 2656–2672、2784–2805、4121–4147、5023–5037；_torch_docs.py 5744–5776、11263–11291（isfinite、sum）。functional/autograd/random另由HTTPS原站取得並byte核對。來源URL、原始副本SHA、存取日期和first-page版本核對記錄在authority-provenance及正式報告。未讀舊審閱者的來源解釋。

## 實際查核結果與限制

原fence輸出逐字符合：問題為照抄數字29，答案DIRECT:29，兩次誤差有限，有效目標43／10。問題32 bytes、換行1 byte、答案9 bytes、EOS1 target，得到43；SFT答案9 bytes+EOS1得到10。問題加「甲」只讓pretrain變46，SFT仍10；答案加「9」則44／11。SFT全部prefix仍valid；修改29→28而保持labels不變會改變回答logits。直接logit梯度在IGNORE位置為零；沒有optimizer.step，參數未變。手算兩格有效CE的分母2與helper相符。

四站84題ID與資料SHA完全相同；完成更新為300／1400／600／100，整題正確為0／42／75／71。SFT文字42/42、模態0/42；joint文字42/42、模態33/42，九個失敗全部image_shape；DPO另外四個joint失敗。前三站和DPO的父檔SHA全部完整相等，第一站parent為null；最後90題在四實報中test_evaluated=false。

短CPU偏好算例確認chosen計分10 targets、rejected28、CE replay10；DPO另有policy與reference的兩份序列logprob，未被effective_tokens計數。第一版自寫算例錯估附加字串為21 bytes而測試失敗；已保留first-attempt原碼／stderr，改以6字×3=18 bytes核算後再跑通過。這是本輪算例預期的修正，沒有修改原教材或實報，也沒有 unresolved claim。

這些結果是指定單seed42合成任務的既有證據核對，GPU裝置由原實報記錄，現在CPU查核沒有重新訓練、GPU工作、下載資料集、載入完整checkpoint或產生新模型成績。0/84只表示這84題的對話協定未達該exact判準；歷史SFT/joint/DPO成績不證明新主線能力。不同目標與額外DPO計算也不能當等成本比較。沒有發現需要修改19.4的實質錯誤。

收到19.5局部改稿版本通知：本人未讀、不依賴19.5，故不需重查；本報告的whole章SHA只指首次frozen bytes，不被其他節改稿機械更新。
