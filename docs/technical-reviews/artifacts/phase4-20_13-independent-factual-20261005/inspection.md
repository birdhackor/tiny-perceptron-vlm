# 20.13 本人核對範圍與觀察

Reviewer: `/root/phase4_factual_coordinator/factual_20_13`。本輪只核20.13；完整讀取抽出的原始bytes及fence，必要前置讀20.2，未讀舊technical/reader報告。原20.md全檔只以bytes保存為frozen input，沒有聲稱完整讀整章。沒有改教材或圖，也沒有訓練、GPU、權重/訓練資料下載或真實成熟推論。

## 程式與原記錄

親讀methods：natural_assistant.py asset_path/row_assets/load_manifest 52–115；normalized/edit_distance/asr_cer_metrics/score_output 175–250；load_core 263–306；generate/parameter_counts 315–374；save_checkpoint 444–467；archive_checkpoint/training_row_at 490–523；run_train 的 optimizer/resume契約528–600；load_asr/transcribe 687–770。natural_ui.py AST定位後親讀99–115、131–169、168–184、183–213、338–352；確認ASR首次transcribe才load，文字route進同一核心。fetch_natural_release.py AST後讀model_pin/validate_manifest 78–187、verify_provenance/verify_release/fetch_release/check_runtime/student_options 278–432、main435–493。score_natural_v4_test.py讀constants1–45、expected_cases/eos81–172、binding173–260、blind_packet446–482、normalization/distance/asr_scores/score_blind531–698；validation helper eos104–120。普通methods/docstrings與一般限制均視為原方法，不視為作者額外結果評語。

audit.py實際CPU exit0，從178份raw生成與完整generated_token_ids、22份raw transcripts、147筆原semantic grades重算各分母/結束/精確OCR/語音CER/子集。manual布林是既有原標註，未宣稱重新獨立評完全部147筆；本人以下另QC原目標、題目、部分raw回答和照片。ASR選版只查16對validation原raw transcript/音檔sha/reference，未讀comparison.json的suggestion或作者結果解釋；small為raw138/524、normalized117/510，turbo為68/524、50/510，依既定雙CER降低規則驗證選版。原final test輸入與選版/模型pin/原raw結果的hash均親核。原Python fence另由section_facts helper fresh CPU執行exit0，並跑同bytes、換adapter名稱、增換行的短變體。

loader_contract.py實際執行原load_core/load_asr/student_options，但transformers和peft是記錄呼叫的tiny替身；因此只驗證revision、像素設定、dtype、adapter分支、eval/freeze傳遞。不是HF完整模型推論驗收。review .venv是Python3.13.5、torch2.14.1+cpu；正式成品需另一Python3.12環境和requirements-natural的torch2.8.0/transformers4.57.6等pins。check_runtime在review環境預期拒絕3.13，沒有把此拒絕判教材錯誤。

## 實際圖像QC

source本節沒有任何image/SVG引用，helper記0 SVG；所以教材figure consistency是具體NA。20.13以清單、檔案指紋與能力卡表格為主，不要求讀者想像新的空間箭頭。以下照片是對能力測量原樣本的QC，並非聲稱教材內另有圖。

已實際渲染並view photo-inspection.png及ocr-inspection.png。三張photo：test_03027是紅色跑車在展示空間、後方白/藍車；test_01539貓的伸前爪與小昆蟲仍隔空隙，對「正在撫摸」的原fail理由合理；test_01831為棒球場，畫面沒有足以斷定德州大學/德州的標識，原fail的越界資訊理由合理。對車款細分不額外宣稱獨立鑑定。

另以original detail直接view commons-153860428.jpg及commons-146481810.jpg。前者指定兩段確為「免費」與「熱點在此」，按要求保留二行；後者指定區域可見直排「蝦味鮮」及下面「活蝦料理」「外送專線」，順序/換行與gold一致。contact sheet也查看commons-171412167的「美人樹／新北市板橋區公所」及commons-193157372右上指定紅招牌「绝味鸭脖」，沒有把整照片其他字算進逐字抄寫。七份既有ignored原JPEG均只保存具名必要原件，對原件/永久copy/manifest SHA與bytes三方親核；未copytree或symlink整資料。

本人讀blind-packet具名cases0、6、1、4、128、130、139、141、140、142的題目/rubric/reference/prediction/decoder_complete；對应原grade只是model答題的primary標註，而非教材technical審閱。聊天成語例的正確內容與校對題額外改寫的失敗理由可直接按原rubric理解。電影例的原grade理由作為既有primary評分記錄，未另宣稱本輪電影資料研究。raw answers/scoring重新彙總支持25/42等歷史測量，不支持無限域或重新模型驗收。

## 外部原始來源

官方HF pinned API /id、/sha、/safetensors/parameters、/safetensors/total與兩個config親讀：Qwen2,127,532,032、Whisper808,878,080均精確一致。Whisper pinned model card117–142及config確認ASR角色與turbo版本；v4.57.6官方processor docs17–34、Qwen docs25–83、Whisper docs26–93原API親讀；official modeling_utils.py from_pretrained doc中revision可用git commit。PEFT v0.18.1 checkpoint17–73確認adapter不含底座且需base。這些外部來源才支持機制概念；本人新取得metadata/docs，不以來源庫摘要當證據。

supplied original-source-locators只按URL locator篩查本節hashlib/pinnedQwen/Whisper/PEFT版本，沒有匹配。original-paper-locators定位Whisper arXiv2212.04356v1 raw PDF；核其SHA6337bde…、本人實讀title/authors/第一頁v1日期與Section2.3 multitask format，支持轉寫任務/結束token界線，不支持turbo參數或本課成績。全文PDF/derived text已保存在official，原論文定位以頁3§2.3為準。

public_metadata_check.py實際HTTPS讀固定公開版兩個小檔，完整size/SHA一致。README僅不解碼地核bytes，避免讀作者capability敘述；provenance JSON只核source=null、selected=base、兩模型及其他pins。沒有建立新交付包，也沒有啟動serve。

必要操作資料只讀STUDENT headings及71–125，DATA headings及9–43、78–105，TRAINING headings及47–64；是資料來源/切分/使用原方法，未讀額外結果敘述。20.13最後要求新候選另留validation/test是條件性操作，沒有聲稱第5階段新工程已完成。已公開的是沿用官方底座且沒有adapter的既有配置。

## 操作性停止與更正

查來源時誤印manifest `/sources` 整個巢狀metadata，586310 token的tool輸出被truncated；立刻STOP並通知協調者。真STOP紀錄和operational-classification-and-resumption.json均保留。本人實際可見head/tail只原methods、AISHELL目標/rubric/provenance，沒有可見舊模型/教材review摘要；未顯示中段不稱讀過。原超量stdout未另留本機，不補造。協調者按actual visible scope分類為操作性失誤並允許同身份續查；續查改用/sources/*/path、type及必要/audio_rows/*具名原葉節點。此事件不是捏造的VOID或教材修正。
