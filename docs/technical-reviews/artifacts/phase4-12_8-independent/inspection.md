# 12.8 獨立本人 inspection

Reviewer: `/root/phase4_factual_coordinator/factual_12_8`; fresh context; 2026-10-05.
本次沒有讀舊 canonical technical report、reader report、作者額外解釋摘要或他人 inspection；`prior-report.opaque.json` 只由 copyfile 保存，未打開。沒有污染事件。沒有訓練、重測既有模型、讀取/建立 .pt、取得錄音/模型或使用 GPU。

## 真正讀取的範圍

- 現行 `course/chapters/12.md#12.8` 全部 UTF-8 原 bytes（含唯一 fence 與圖），本節 SHA-256 `d80f829ddaa5732899753c5d318938f4dec8282e9b5f00c3c58e5ba74e3c545e`。
- 必要教材前置：12.7、10.3、W.3；任務銜接：12.9、12.15。這些是目前教材，不是舊審閱。完整 12.md 當次 frozen input 存 inputs/course/chapters/12.md，SHA `97250b5340e72b9b1b05c9cae2e0aed167904e5d3ad3ebe9fa14f7c210c37616`；此 SHA 指保存當下完整章節，不冒充後來整章版本。
- 親讀 factual-reviewer-instructions.md、checker schema、section_facts.py、clear-tutorial SKILL.md、review-protocol.md。另遵循 developer 指定的 cloud-environment-onboarding:setup，讀 setup/SKILL.md 及 onboarding.md；既有 .venv 足以驗證，沒有配置變更。
- 原實作先 AST 定位：multimodal.py 的 tone 44–46、mel_filter_bank 49–59、log_mel 62–69、AudioEncoder 72–81，親讀 1–84（其他入口僅上下文）。完整 log-mel 契約含短輸入補零、Hann window、STFT、功率、三角 mel bank、1e-8 下限與自然 log。
- modalities.py 先 AST 定位及辨識字串。親讀 _hash/_manifest 37–53、_audio_records 195–216、_media 219–243、_loss_fn 260–283、run_encoders 387–445、_fit 73–153 中計算/更新/原測量規約。不讀 run_encoders 的 calibration、scope、limitation 等結果解釋值。普通分組註解與 timing scope 屬方法規約，已讀；沒有作者修正結論曝光。
- 原 encoders.json 先列上層 key/type、/results/audio 下層 key/type，以及 raw samples/manifest fields key/type。實際具名 pointers 全記 bounded-results.json 的 history_selected_pointers；只讀 provenance、config/classes、data/splits 的 count/sha256/records、training 原測量欄位和 before/validation/test 的 count/correct/samples。沒有讀 /results/audio/scope、calibration 或任何 notes/review/correction。完整原檔保存、SHA `4ed0c1a384802adca0c30a77a230b1fd2e7c6cbe6de5f3d2af583395fed0ed18`，未刪原附註。
- Locator JSON 只讀 keys/types 與具名 locators 的 URL/version/path/hash，作來源位置線索；所有引用的 PyTorch 原文由本人另從上游 immutable commit 下載並讀，不使用別人的 inspection。

## 本人核對的權威原文

所有原文的 URL、HTTP 結果、版本、SHA、日期保存在 official-fetch.json 和 minds-fetch.json。

PyTorch 是該 API 的官方維護者，上游 git commit `5c4886908584029761b579af026dcfb627c84070` 與實跑 torch.version.git_version 相同；執行為 Python 3.13.5 / torch 2.14.1+cpu，CUDA build=None、cuda_available=False。

- torch/functional.py 507–650：stft 的 input 1-D/2-D 規約、center=True、reflect padding、onesided real input、輸出 B?×N×T、center 時 T=1+L//hop。本案 L=1600、hop=160，因此 T=11，FFT bin=400//2+1=201。
- torch/_torch_docs.py 13067–13111 hann_window；11942–11990 transpose；2936–2956 clamp；6289–6316 log；7195–7261 mean。涵蓋視窗、交換兩軸、下限、自然 log 及跨 dim 的平均，並無靠單一籠統 API 引用代替。
- aten/src/ATen/TensorIndexing.h 660–681：單一 None index 分支明確回傳 self.unsqueeze(0)，支持 [None] 的前置一軸。執行另驗值/儲存皆未改。
- torch/nn/modules/linear.py 53–140：y=xA^T+b、僅改最後寬度、learnable Parameter/隨機初始化；normalization.py LayerNorm 105–236：normalized_shape 單整數只跨最後軸，形狀保持；activation.py GELU 778–823：xΦ(x)、形狀保持。這組 API 共同支持 frame-wise Linear→LayerNorm→Linear→GELU，不混合時間位置。
- PolyAI MInDS14 README immutable `40ce77cb32a384e4d50a568e1ec39ac804019d33`：親讀完整原 data card（含 metadata）；重點 869–923 zh-CN audio/intent_class，927–935 spoken intent detection；990–998 audio/intent_class 欄位。它是資料發布者的原資料規約，支持真人需求分類與人工单音閾值任務的材料/標註區別；未取得或播放錄音，未驗任何中文模型能力。

## 執行與數值

原 fence 由 section_facts.py 在隔離 CPU/offline process 實跑（original-executed/），exit=0，無 guard event，stdout 頻譜 (1,16,11)、時間特徵 (1,11,8)。原 fence SHA `5d76f94f04c33e1867dc4881c64ecfcc2c59e51294ba0de5f6c9ea27420f4ad1`。

bounded_cpu.py 是本人寫的短 CPU 變體。1600 samples/16000 Hz=0.1 秒；None 使 1600→1×1600；手工 reflect pad 200+200、unfold 400/hop160、rFFT 與 torch.stft 最大絕對差 0。獨立 scalar mel bank 最大差 1.4975667e-6；獨立 log-mel 最大差 1.9073486e-6，使用明記 atol/rtol 容忍浮點差。floor=1e-8、自然 log 下限約 -18.420681。

內部 encoder 輸出等於獨立 log_mel 交換軸再套原層的輸出，差=0；逐框個別計算吻合；所有參數 requires_grad=True，但 forward 前後參數逐項完全相同。沒有 optimizer/backward。只改 width=12 得 1×11×12；32-band 外部前處理和 AudioEncoder(32) 內部都是 32 帶，輸出 1×11×8。AudioEncoder 的 bands 自動傳給內部 log_mel；獨立的 spectrogram 列印要同改 bands 才能對照。顯式把16帶 tensor 傳入32帶 projection 出現預期 shape mismatch。batch2=2×11×8。

逐框反轉只反轉輸出位置；沿時間平均反轉前後一致，故保留時間位置與 time mean 不同。這是結構檢查，不保證下游已會辨識任何變化。bounded stderr 的 requires_grad scalar-conversion warning 來自記錄差值，未涉及梯度或更新，assertions/exit 均通過。

歷史 JSON 的16/8/14筆、8/4/7個frequency families、split digests、全部 f>300 的 high/low（300本身為low）及 sample correct 判準已實際重算。原方法的250步×8 samples=2000 effective targets；610 parameters=576 AudioEncoder(width16)+34 classifier。原記錄使用 torch2.14.1+cu126/cuda，与此次CPU檢查明確分開。歷史與現行multimodal.py字節hash相同，但本次沒有重新評測其.pt；250步與更新flags只是核對原記錄/原method。

## 圖的本人畫面 inspection 及限制

Inkscape 把現 SVG 渲染為 figure.png；本人使用 view_image 親看全圖。圖有 (1,16,11)、框1/2/3各16條示意帶、各自箭頭指8個值、(1,11,8)，並明記三框代表其餘依序、色格不是實錄音；與文字/實作吻合，沒有11字的標籤。配方共享由同一入口的圖文字表示，不是把三框合併。

另外讀取8765/12.8.html，確認新增 None 與內部 log-mel 句確實存在。Chromium桌面/手機首輪未產出截圖，僅停止本人user-data-dir process；改用timeout25s/virtual-time-budget有界重試仍exit124，未產圖。stderr實事件保存；不能宣稱桌面/手機頁面畫面已驗。這個瀏覽器執行限制不取代或否定已render/view的SVG技術一致性。沒有外部來源HTTP transient，所有官方原文request皆200。

## 判定

PASS：現行12.8的原fence、數值、軸與encoder整體處理、練習變化、圖及有限任務範圍均獨立查證。它展示隨機初值前向特徵形狀；不宣稱學會單音或中文需求。真人短句設計是後續有限任務安排，不是本節完成能力驗收。本次沒有改正文、圖、實作或commit。
