# 選到 base 時的發布與學生試用契約

本文件說明發布程式已支援的兩條路線，不代表 v4 已完成選版或對外發布。仍須依 frozen validation protocol 完成盲評，先保存 selection，再執行 test 與正式 release review。不能為了產生 adapter 而改寫選版結果。

## 維護者：若 selection 選 base

1. 確認已提交的 selection 的 `selected_variant` 是 `base`；該 selection 不含 adapter run、checkpoint 或 weight SHA。test 使用相同官方 base/ASR pins，adapter 參數留空。
2. 將正式 release approval 存在 `docs/natural-assistant/releases/`。它必須有實際審閱完成的 `approved: true`、`reviewed: true`、model card、release ID，以及從選版／實測取得的完整 `base_model`、`asr_model` repository 與 40 位 commit。base 特有欄位為：

   ```json
   {
     "selected_variant": "base",
     "source": null,
     "files": [],
     "adapter_parameters": 0
   }
   ```

   此片段不是可發布的完整 approval；未列的 review 與 pinned model 欄位不能省略。`0` 表示沒有附加 LoRA 參數，不表示 base 的參數量為零。model card 應說明成品使用選中的官方 base 加固定版本 ASR，並引用實際 test 結果，不宣稱為本次 LoRA 成品。
3. 提交 approval 與相同的 frozen data manifest。dispatch `natural-assistant.yml` 的 `stage=release`，填相同 `batch_id`、`manifest`、approval 路徑；adapter run/checkpoint 留空。沿用既有 HF release repository、Modal Secret 與累計預算帳本。這是 CPU 發布工作，仍走既有 reserve/finish，不能當成免費 Modal 操作。
4. 發布程式先確認 release repository 公開、base 與 ASR 的指定 commit 皆公開可讀，再產生 `README.md` 和 `release-provenance.json`。provenance 的 `selected_variant` 為 `base`、`source` 為 `null`；不讀私人訓練 repository，不上傳或造出 adapter，也不重新散布官方完整 base/ASR 權重。兩個文件都會從實際 public upload commit 匿名下載，核對大小與 SHA。
5. 根據這次實際 release receipt 建立／更新 `docs/natural-assistant/public-release.json`。保留原 student manifest 的 schema/version/runtime/dependency/code SHA 契約；`repo`、`revision` 與兩個 `files` 項目的 path/size/SHA 必須取自 receipt。每個文件補審閱後的 `output`、`license`、`redistribution_approved`。填 `selected_variant: "base"`、`adapter_parameters: 0`、相同 model pins、approval SHA、資料 manifest SHA 和本次 source Git SHA。公開 prefix 仍為 `natural-v3/<release_id>`，只為相容既有下載路徑，不代表沿用舊模型。
6. 用該完整 manifest 實際匿名 fetch/verify 後才提交可供學生使用的 public manifest。base 的必要文件為 README 與 provenance；可另外附已審阅的 runtime versions/license/notices。base manifest 會拒絕任何 adapter 檔案、非零 adapter count、錯誤選版／私人 adapter source、非固定 commit、額外檔案或 SHA 不合。

## 學生：兩種成品共用相同命令

以下命令從 repo 根目錄、既有 Python 3.12 `.venv-natural` 執行。第一次 serve 會從 manifest 指定的官方 commit 下載 base 權重，第一次使用語音時再載入固定 commit 的 ASR；fetch 下載的是本專案的公開發布文件及所需 adapter（若選到 adapter）。有離線完整 cache 時才加 `--local-files-only`。

```bash
.venv-natural/bin/python scripts/fetch_natural_release.py --list
.venv-natural/bin/python scripts/fetch_natural_release.py
.venv-natural/bin/python scripts/fetch_natural_release.py --verify
.venv-natural/bin/python scripts/fetch_natural_release.py --serve --device cpu
```

支援的 NVIDIA GPU 可以改用 `--serve --device cuda`；不支援 bfloat16 時加 `--dtype float16`。base 成品自動傳 `adapter=None`，介面與照片、文字、語音聊天流程相同；不用自行建立 adapter 目錄。舊版下載目錄已有內容時，請使用新 `--output` 目錄並在 verify/serve 使用同一目錄，避免把不同成品混在一起。

## 若 selection 選 adapter

維持現有 private backup 到 public inference-only export 的流程。新的 approval/manifest 可將 `selected_variant` 明列為 `adapter` 或選中的 `adapter-step-NNNNNN`；provenance 必須同步。adapter source 使用已核對的私人 HF commit、檔案使用實際 SHA/size，選中 archive 時用 `source_path: checkpoints/step-NNNNNN/adapter_model.safetensors` 及對應 config。公開 student manifest 仍須至少包含真實 adapter weights、config、README、provenance，adapter parameter count 必須是正整數。

未標 `selected_variant` 的既有 approval 與 public manifest 仍視為 adapter，原來六個 provenance 欄位與四個必要檔案繼續適用。兩條路線都不允許公開 optimizer/RNG/resume state。

## 本次驗證的範圍

`tests/test_natural_base_release.py` 使用 mock Hub/API 與 mock pretrained loader，真實執行 CPU publisher、檔案 hash、匿名 fetch、學生 options 和 UI/core 入口，並覆蓋 base/legacy adapter/指定 archive 的契約。這些測試不是模型能力證據，也不代表曾向 HF 發布。原有 public release、Modal runtime 與 UI tests 同時保留並執行。
