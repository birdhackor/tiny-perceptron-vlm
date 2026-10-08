## A.2 不在訓練資料裡的資訊怎麼使用？

小小書店的架空公告寫「2027年3月1日搬到青街8號」。問新地址，答案應是青街8號，引用公告1；日期只是支持版本的另一欄，不應拿來代替地址。即使模型訓練時不知道搬遷，只要當前拿到公告，就有資料可用。

文件多時，先依問題找相關段落，再放進模型前文讓它回答，叫RAG（檢索增強生成）。檢索負責找資料，生成負責讀資料回答；來源編號用來回查。這些都是本次上下文，不能把提供公告當成永久寫入權重。

![本地公告先被檢索，再與問題送入模型，回答要回查來源。](../figures/rewrite-A-rag-flow.svg)

```python
source = {"id": "公告1", "date": "2027年3月1日", "address": "青街8號"}
context = f"小小書店在{source['date']}搬到{source['address']}。"
prompt = f"[來源{source['id']}] {context}\n問題：書店新地址？請依資料回答並標示來源。"
print(prompt)
print("由資料確定的答案", source["address"])
```

程式插入公告欄位，印出真正要提供的資料與紙上答案，沒有生成。若直接把 `source["address"]` 當模型回答，會跳過本節要查的閱讀能力。

同模型可分有公告／無公告比較，再只換地址，要求答案跟新資料變；來源編號也可單獨換，以查引用是否跟著更新。模型可能讀錯欄、混入舊資訊或補出沒有根據的細節，下一節先把找資料那一站拆開。

<details>
<summary>補充：獨立實驗與成對改動</summary>

正式RAG組用可逐字核對的小世界測這件事：店名`K017`、地址`C2`、來源`D1`，應答格式是`C2[D1]`。公告練習按店名分組，同一店名的例子留在同一組；完整分組與訓練設定見本節末的正式報告。模型由隨機權重建立，沒有接續前面的SFT或文字預訓練權重，而是按示範回答訓練成一個新模型。

訓練結束後才替12個未教過的店名重新抽地址與來源，再固定權重生成。有正確公告時答對5/12。將同一店名的公告只改地址，改後答對7/12，但原版與改版都答對只有5/12對；只改來源編號，改後答對5/12，兩版皆對只有2/12對。例如`K017`真的由`C2[D1]`改答`C3[D1]`，`K028`真的由`B9[D9]`改答`B9[D0]`；這些個別成功不足以保證每份新公告都跟得上。改動只作用於當次輸入文件與期待答案，沒有修改保存的資料切分或再次訓練。

在已啟用專案環境的根目錄可完整重跑這份配方；CPU時間與正式L4時間不同，有自己的CUDA環境才將裝置改為`cuda`：

```bash
python scripts/course_experiments/run.py --experiment rag --device cpu
```

`outputs/course-experiments/course-v1/rag/`保存`dataset.json`、`icl-dataset.json`、`retrieval-corpus.json`、`generations.json`、最終`model.pt`及四份中間checkpoint。改設定須另留結果；完整配方與失敗生成見[正式報告](https://github.com/birdhackor/tiny-perceptron-vlm/blob/main/docs/course-experiments/results/rag.json)。

</details>

