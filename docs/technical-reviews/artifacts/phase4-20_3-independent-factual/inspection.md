# 20.3 本輪獨立查核紀錄

實際 reviewer task：`/root/phase4_factual_coordinator/factual_20_3`。派遣範圍為目前 `course/chapters/20.md#20.3`。未讀舊審閱正文、他人 verdict、作者附加結果評語或修正摘要；定位庫僅讀 URL/path/hash/version 定位。未改教材、未啟動 GPU、未下載模型權重或資料。

## 親讀範圍

- 完整 20.3：原稿第92–110行；必要前文第1–91行（章節導言、20.1 輸入路線及20.2 固定底座、版本和CPU載入設定）。前文的既有選擇敘述僅為教材上下文，不作本節模型成績證據。
- `TRAINING.md` 第1–32行：本節直接連結的GPU操作環境、權重估算與實測範圍說明。這份操作稿的硬體數字未在20.3重述，本輪不核其训练成績、完成狀態或峰值原始量測，不將連結視作本輪硬體能力驗收。
- 原實作先AST定位；親讀 `natural_assistant.py` 第25–33行、`load_core` 263–306、`parameter_counts` 370–374、`load_asr` 687–697，另讀評測控制與共用ASR/模型分支818–868、912–927。未讀結果彙總字串與既有分數。
- public-release原JSON先列上層keys/types，再只讀 `/base_model`、`/asr_model`、`/dependency_versions`、`/runtime`、`/adapter_parameters`。未讀審閱/批准判定欄位值。
- 官方模型metadata先列keys/types，再只讀 `/sha`、`/safetensors`；原config為純模型構造配置，完整讀取。
- LoRA原PDF第1頁核版本；親讀sec4.1 Eq.(3)及合併段落、sec4.2 practical benefits。本地原PDF副本SHA相同，另用HTTPS重新取原PDF核SHA一致。
- 官方Transformers v4.57.6 AST定位每個必要constructor後逐項讀：vision MLP/patch/merger/attention/block、text RMSNorm/attention/MLP/decoder、vision/text/model/conditional generation。另讀text attention forward 416–457、text MLP/decoder460–519、vision forward703–752、tied output keys1271–1283。
- PEFT v0.18.1 `PeftModel.from_pretrained` 389–439 的參數与契約；官方memory anatomy v4.57.1的Anatomy of Model's Memory與Forward Activations；Whisper固定版modelcard 115–137、210–239；Qwen固定版modelcard11–18、53以及來源載入示例定位。

## 實際檢查與支持範圍

原始fence由section_facts.py完整CPU執行成功。獨立check_parameters.py按官方構造計唯一參數；語言部份1,720,574,976、視覺部份406,957,056，合計2,127,532,032，與固定revision的官方safetensors metadata相符。輸出lm_head共用embedding不重算；視覺merger及3份deepstack merger皆納入。使用torch的element_size實測float32=4、float16/bfloat16=2、float64=8 bytes並另算GiB；與正文7.93/3.96相符。

LoRA小變體驗證W0x+BAx=(W0+BA)x，而只給BAx不同；只支持原方法的加法與合併機制，沒有模型能力指標。原load_core/load_asr/parameter_counts的AST函式使用小CPU替身實際執行，確認先載入固定底座再接adapter、基底凍結、adapter訓練開關、CPU非float32提前拒絕、ASR獨立固定入口。替身不是完整Transformers/PEFT整合測試。本輪.venv為Python3.13.5/PyTorch2.14.1+cpu，Transformers和PEFT未安裝；官方實作契約按教材固定4.57.6/0.18.1親讀，不冒稱在.venv執行了成熟模型。

load-contract初版手寫regex的字串跳脫較原碼多，初稿与stdout保留；後改為直接抽取原始LORA_TARGETS assignment，新增原路徑匹配assert再執行，正式引用為後一版。這是檢查工具修正，沒有教材問題。

## 圖與範圍

20.3沒有圖片、SVG引用、圖片素材或空間標籤；純量參數與byte乘法可由文字/fence完整說明。figure consistency具體NA，本輪沒有對其他節圖作驗收。

本節沒有新模型成績或第5階段工程已完成宣稱。數值僅為未壓縮純權重數字儲存，不包括序列/視覺activations、KV cache、workspace、梯度、adapter optimizer或Whisper。凍結底座不表示為全部底座配置梯度/optimizer；LoRA原文明確減少這些部分。重量相同精度估算不能給出最低VRAM、實際載入峰值、推論/訓練可行容量。圖文底座與ASR固定是比較方法的控制要求，不是已證明LoRA勝出的宣告。
