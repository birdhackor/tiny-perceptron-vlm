## 從隨機權重重做自己的訓練

[本機訓練指引](https://github.com/birdhackor/tiny-perceptron-vlm/blob/selftrained-v2/docs/selftrained/TRAINING.md)使用 `scripts/selftrained/train_local_stage.py`，核對整個固定資料包後，依序執行自己的pretrain、SFT、vision、OCR、audio、joint與最後weighted／native分支。後段載入自己的selected `best.pt`；中斷後同段exact resume載入自己的 `latest.pt`，恢復optimizer、RNG、sampler與步數。每次attempt保存實際指令、檔案指紋、日誌和真實execution／training receipts，不需要作者私人Volume或原GHA run ID。

公開safetensors只供推論，沒有optimizer、RNG或sampler；目前trainer也沒有公開safe權重的training-init載入器。從它們推論，不能叫作精確續訓。全量配方與一個batch的教學smoke不同；本機wrapper工程驗證用width16合成CPU資料檢查成功、失敗、中斷及續訓契約，沒有重跑全量production training或另證模型能力。正文或圖解改版不要求重訓全書；實作、資料或能力宣稱改變時才驗收相應任務。

