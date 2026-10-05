# 本人獨立查核 12.12

審閱身分：`/root/phase4_factual_coordinator/factual_12_12`，fresh context，唯一小節 12.12。本紀錄為本審閱者本人觀察，不是作者或他人的 inspection。2026-10-05。

讀取現行 `course/chapters/12.md` L373–425 全節、其圖與原 Python fence。必要前置親讀：同檔 12.11 全節 L341–372，12.9 L265–308；10.md 10.7 L213–250；11.md 11.8 L284–321。因精確範圍讀取同時看見少量下一節開頭，未據其作判定。完整 Markdown 原 bytes 僅保存 frozen input；正式 section SHA 是 `43692d920d81e1c207a707688ebb4392ea819bc6c788596d0fa4d5c758000513`。没有閱讀舊報告、reader 摘要、作者結果 scope 或修正摘要；定位器只用來取得 immutable 原始來源。

先讀 factual-reviewer-instructions、section_facts.py、check_technical_reviews.py 全 schema、clear-tutorial SKILL 與 review-protocol。先 rg --files 定位，再 AST 定位所需原實作。

原方法實讀範圍：multimodal.py L1–140（含生成器簽名/普通 API docstring），L177–190；model.py L1–105；data.py L1–29。modalities.py `_hash/_manifest` L37–53，`_fit` L73–126（optimizer、梯度、step、probe、普通 timing contract），`_vision_records/_audio_records/_media/_sequence/_loss_fn` L171–283，`_evaluate` L287–358（raw ID exact match、EOS、part 判準），`_modal/_freeze` L367–384，`run_encoders` L387–443，`run_joint` L913–954，另曾實讀 L455–482、L484–510 的普通 calibration 計算規約；未讀 L955–963 的作者 scope 結論。common.py Context.dependency L19–30。AST inventory 僅列方法名、行號與 Return 的鍵，不展開結果解釋字串。

原 JSON 先印上層鍵與型別，再僅查具名 pointers：joint `/schema_version`, `/experiment_id`, `/revision`, `/device`, `/seed`, `/torch_version`, `/python_version`, `/step_scale`, `/results/training/{steps,effective_tokens,effective_targets,weights_changed,nonzero_gradient_seen,cpu_smoke,config,modal_config,checkpoint}`, `/results/data/{seed,split_policy,splits}`（三 split 的 count、sha256、records），`/results/validation/{examples,correct,shape_correct,pitch_correct,eos_rate,effective_tokens,ablation,samples}`，`/results/evaluations/{none,blank_image,blank_audio,blank,swap_image,swap_audio}` 的同名 measurements/samples，`/artifacts/*/{path,sha256}`，`/code_sha256/{scripts~1course_experiments~1modalities.py,tiny_perceptron~1multimodal.py}`，`/assets`（空清單）。`/results/before`、`/modal`、`/hf` 僅鍵/type，不讀值；未讀 `/results/scope`、statuses、hf checkpoint_resume_scope。encoder JSON 實讀 `/revision`, `/results/audio/data/splits/train/records`, `/results/audio/training/{steps,weights_changed}`, `/code_sha256/scripts~1course_experiments~1modalities.py`；其餘所印只有 key/type。

核過 current modalities/multimodal 完整 SHA 與 joint 原始 code_sha256 相符。joint 原 run revision `0cb167669bfc1654016a391a7b940ae31ca7334a`，原 CUDA 紀錄 torch 2.14.1+cu126/Python 3.13.3/seed42，400 steps、18422 有效目標；這不是本次訓練。本次 `.venv` Python 3.13.5/torch2.14.1+cpu/git5c4886908584029761b579af026dcfb627c84070，只跑 random tiny model，沒有模型檔載入、生成評測或 parameter update。

親核 PyTorch 官方原 autograd notes `How autograd encodes the history` L12–34，官方 `_tensor.py` Tensor.backward L566–627。官方 immutable URL 的 autograd bytes HTTP200 與本次原 snapshot 完全一致。版本等於本次安裝 git；不用別人對文件的解讀。Geirhos 等論文親讀原 PDF title/first-page arXiv2004.07780v2（19May2020）、摘要/導言 pp1–2、§3 p4 對位置 shortcut 的 control、BoxI PDF p14、§7 recommendation p17；另親核官方 arXiv `/abs/2004.07780v2` 的 title/authors/this-version dateline/submission history，最新版 v5 與實讀 v2 分清。它只支持一般 shortcut 風險和受控測試的用意，不證明本 repo 模型的成績。

原 fence helper exit0，圖/音路徑皆 True。bounded audit 在四個新 random 模型中各做一次 backward：無 detach、只 detach 圖、只 detach 音、detach 兩者。未切斷時 norm 為 0.0370502323 / 0.0318056941；切斷某路只使那路 grad=None。所有參數前後完全相同。logits `(1,42,264)`，12 個有效目標（11 ASCII bytes 加 EOS）；完整展開長43，single shift 得42，答案標籤一致。full-vs-masked reduction 的 float32 誤差 4.76837e−7，容忍1e−6。

逐題 raw ID 解码與標準 exact_match 重新計算，不執行既有模型：none (shape,pitch,joint)=12,12,12；blank_image=6,12,6；blank_audio=12,6,6；blank=6,6,3，全部 /12。每組138 有效 target tokens；EOS12/12，無 generation_error/invalid tokens。validation=8,12,8 /12。兩種 swap 各12/12完整正確、12/12 pair兩端正確、只改對應部分，目標按新材料重標。blank 對的是原目標，不是聲稱空白/靜音自帶某一類。

train24、validation12、test12，四種 shape×pitch train各6、其他各3；原 split records SHA 逐一吻合。test偏移2/260或340Hz，train偏移0/180,220,380,420Hz。encoder 原 train records 已有260和340Hz，支持「不是所有線索皆未見」的限定；不拿12題概括真實聲音、未見條件或成品驗收。

視覺親核 Inkscape PNG、Chromium桌面1280×800/手機390×844的完整頁截圖，與現 source paragraphs/fence/table核對，served SVG bytes相等。圖是scene紅圓逐格256格的忠實素材（49紅格）；440Hz曲線清楚標示「示意」，沒有可讀頻率軸或時間軸，因此不把三個畫中周期當0.1s的44周期。箭頭將circle/high送到完整答案，沒有畫成模型已學會。手機圖全顯，程式/表格在窄容器可橫向查看；本次技術審閱不是重新做易讀性驗收。

工具過程保留：第一次自寫 masked-vs-full loss assertion用1e−7，因float32 reduction差異失敗，保留 `.initial` 原碼/stdout/stderr，改為有理據的1e−6後通過。第一次 structured runner將 `.venv/bin/python` symlink resolve到系統直譯器，造成無torch；保留 `.runner-first` 日誌，改用未解析的venv絕對路徑後exit0。這兩件是本審閱工具問題，未改教材。初次 Chromium CLI 遲遲未產出截圖，停止自己啟動的進程；兩個有界 CLI retry也timeout，日誌保留；改用相同官方 Chromium的Playwright，domcontentloaded及限時，外部可選fonts/analytics HTTPS中止、同local頁面成功實際render。原 paragraphs comparison曾因 markup空白及純image的空p轉figure而失敗，排除markup空白/空p後全7個實質段落與全部5個表列一致；原fence比較原碼一致。沒有發生權威來源 transport failure。

只白名單保存指定原稿、原JSON、小原碼、原官方檔、本人程式/日誌與圖像；不copy helper workspace，不跟symlinks、不保存新.pt。沒有資料/模型下載、GPU、付費或完整shell recipe執行；只取得官方資料作查證。無正文/圖/實作更動、無commit、無派child。

收協調者的純版本通知後，本人另讀目前12.9全節，保存 `original/prerequisite-12.9-current.md`，原 UTF-8 section SHA `0930ed04d140c96d076a3ca8af5d3334f20d684293d2f6eaff8f04830d991149`，當前原稿起於L265。12.12真正依賴的440Hz>300Hz high、audio marker以11條特徵替換、width接頭、IGNORE/shift以及backward不更新等背景，与本人的同SHA原實作和CPU audit一致。12.9的單音歷史成績敘述變動不被12.12拿來作分數或因果推論，本次未擴張去重新審核12.9的獨立實測主張。仍保留初讀整章frozen SHA，不機械改其歷史版本。
