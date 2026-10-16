# 三成品與固定重做方式

A 沿用 [MoE v2 公開 CPU 操作](../selftrained/v2-public-cpu-commands.md)，其歷史權重、考卷與未達全部門檻的結論保持原版本。B 是有限兩盒狀態追蹤解題器；C 是有型別問題的共享候選評分器。二者都由隨機權重訓練，不載入外部預訓練神經模型。B 的三數加法泛化失敗須連同狀態追蹤結果閱讀；能接受某種輸入格式，不代表答得可靠。

## 先使用完成權重

正式產品與原始分數：[固定HF版本](https://huggingface.co/birdhackor/tiny-perceptron-course-models/tree/2c6a1062f3e66a602cf83eda6c703456ec18bca0/lesson-products/v1-20261016)，共32個檔案，採本專案MIT授權。B／C各有獨立推論清單；`raw/`另保存壓縮逐題分數及其大小／SHA清單，機制對照保留完整最後checkpoint。公開套件中的來源commit指向測試前凍結版本，教材後續回填不改寫那次模型與考卷。

在 repository 根目錄，沿用本專案的 CPU 安裝方式。正式固定版本由 [release.json](../../tiny_perceptron/lesson_products/release.json) 指定 HF commit、產品 prefix 與 artifact manifest SHA；下載後再逐檔核對。

```bash
uv sync --frozen --extra cpu
uv run --frozen --extra cpu python -m tiny_perceptron.lesson_products download --product b
uv run --frozen --extra cpu python -m tiny_perceptron.lesson_products download --product c
uv run --frozen --extra cpu python -m tiny_perceptron.lesson_products solve --state '兩盒追蹤;A=2;B=6;事件=交換;查詢=B' --loops 3
uv run --frozen --extra cpu python -m tiny_perceptron.lesson_products judge --state '卡片進度;申請編號=?' --question info
uv run --frozen --extra cpu python -m tiny_perceptron.lesson_products auto --state '兩盒追蹤;A=2;B=6;事件=交換;查詢=B' --effort auto
```

作者已在新建CPU環境與空產品目錄，實際完成匿名下載、上述三種命令、候選反轉、硬上限及三個HTTP入口；[操作收據](results/public-cpu-acceptance-v1.json)記錄版本、命令與輸出。這只驗證這組例子和操作通路，能力分數仍看固定最後報告。

第一次需連線匿名下載，後續 `solve/judge/auto` 只讀本機權重。B 的 `--loops` 可選1–6；Auto介面的low/medium/high分別要求1/3/5輪，auto才查凍結規則。`--max-loops`是硬上限，截短會明示`budget_limited`，不保證答案正確。C先作Choice與資訊充分性兩次判斷；資訊不足回`clarification_needed`，其他任務回`unsupported`。

C 的問題分 `choice/info/rubric`。候選語義ID分別為 `card,address,app,boxes,sum,none`、`yes,no`、`grade0,grade1,grade2`；候選意義見[生成器](../../tiny_perceptron/lesson_products/data.py)。可用 `--candidates no,yes` 改順序，不能任意加新語義。`probabilities`是當前問題／候選集合上的機率，`concentration`只量集中程度。三種問題各有獨立的開發集溫度；集中度不是答對機率。

可啟動只綁本機的JSON示範服務：

```bash
uv run --frozen --extra cpu python -m tiny_perceptron.lesson_products serve --port 8765
```

`GET /health`檢查已載入；`POST /solve`接收`state,loops`，`/judge`接收`state,question,candidates`，`/auto`接收`state,effort,max_loops`。它們使用同一套推論函式；這是有限語法示範，沒有一般聊天能力。

## 重做訓練

只想看教材機制可先跑第14章Notebook；完整訓練需要重建固定資料與較長運算。以下入口在自己的裝置執行，不呼叫Modal或HF；`--device`可指定`cpu/mps/cuda`。裝置需有相應PyTorch後端，只有實際跑過的環境可宣稱已驗證。

```bash
uv run --frozen --extra cpu python -m scripts.lesson_products.train_local --job docs/lesson-products/jobs/train-b-v1.json --output outputs/my-looped-b --device cpu
uv run --frozen --extra cpu python -m scripts.lesson_products.train_local --job docs/lesson-products/jobs/train-c-v1.json --output outputs/my-judge-c --device cpu
```

若使用已正確安裝CUDA版PyTorch的環境，保留相應的`uv --extra`並改`--device cuda`；不要用`--extra cpu`把CUDA依賴換掉。正式作者訓練在NVIDIA L4執行，實際步數、選中步數、隨機起點、資料SHA與精確下一步恢復檢查見各產品`training-receipt.json`。更換裝置、版本或seed可能產生不同數字。

資料首次重建在`data/lesson-products-v2`，之後核對已提交的v2 manifest。輸出需新目錄；同配方恢復用`--resume-from`指向原`best.pt/last.pt`目錄。推論`model.pt`沒有optimizer/RNG，不能代替續訓組。CLI拒絕job與checkpoint的seed、batch或learning rate不一致；原訓練器另核對best/last run identity、資料、tokenizer與配方。

固定三輪與抽樣1/3/5輪的機制配方分別在`mechanism-fixed3-v1.json`、`mechanism-odd135-v1.json`。兩者相同seed、初始化與1500次實際更新；計算工作量不同，不能冒稱等算力比較。它們與部署B的逐輪監督產品配方分開。

## 資料、凍結與結果

[共同契約](../course-revision-20261016/PRODUCT-SPEC.md)、[資料manifest](data-v2-manifest.json)及[實際token稽核](data-v2-token-audit.json)保留切分與指紋。B/C從同一世界衍生的題目繼承家族split；C仍有跨家族相同的通用觀測，相關數量公開，因此不能把其分數稱為對完全未見文字的泛化。

溫度只看calibration；Auto規則與權重選擇只看policy；最後test在設定SHA凍結後首次啟動。最後報告、逐題原始分數與凍結檔隨固定HF產品公開，可從原始分數重算成績。完整成績與限制回到[19.20](../../course/chapters/19.md#19.20)閱讀；不能把開發集或單一成功例當作最後驗收。

維護者回查執行commit、一次性傳輸分支與公開包指紋，見[發布紀錄](RELEASE.md)。
