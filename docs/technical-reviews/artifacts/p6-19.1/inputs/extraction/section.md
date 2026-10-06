## 19.1 一個成品，為什麼要試好幾種問題？

先給助理兩段條件：「接下來請用兩點回答」，以及「App錯誤訊息一直重複出現」。第一句決定格式，第二句決定內容。這份公開 MoE 權重在 CPU 上實際生成了下面的回答，並產生表示回答結束的 EOS 編號。

| 交給模型的材料 | 內容 |
| --- | --- |
| 先前的 user 要求 | 接下來請用兩點回答。 |
| 先前的 assistant 確認 | 好，接下來用兩點回答。 |
| 目前的 user 問題 | App錯誤訊息一直重複出現。 |
| 模型這次的回答 | 1. 先檢查網路並重新啟動App。<br>2. 仍失敗再詢問官方客服。 |

這裡的「先前確認」是公開示範檔提供的對話歷史；最後兩點才是本次模型生成的內容。它同時用到了當前問題和先前格式。沒有圖片或聲音的這一題，先讓我們確認共同的文字入口與回答流程能運作。

<details>
<summary>操作：用自己的 CPU 重做這個公開權重例子</summary>

先安裝 [Git](https://git-scm.com/downloads) 與 [uv](https://docs.astral.sh/uv/getting-started/installation/)。第一次取得本章專案，可從終端機執行：

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --branch selftrained-v2 https://github.com/birdhackor/tiny-perceptron-vlm.git
cd tiny-perceptron-vlm
uv sync --frozen --extra cpu --extra selftrained
```

已有專案的讀者從 repository 根目錄執行最後一行即可。這個純文字例子不需解開圖片與錄音資料包。接著執行：

```bash
uv run --frozen --extra cpu --extra selftrained python scripts/selftrained/chat.py \
  --model-dir outputs/selftrained-v2/public/moe-joint \
  --asset-dir outputs/selftrained-v2/data \
  --repo birdhackor/tiny-perceptron-course-models \
  --revision 979cdfacc588ad0536f1c64fff96f264571cf054 \
  --prefix selftrained/v2/moe-joint \
  --manifest-sha256 f27856892b0c46ca81ba43bfadf7051ed6bd31d848e4868a05147e5799df1e5e \
  --messages docs/selftrained/examples/v2/text.messages.json \
  --task text --device cpu --max-new-tokens 128 --threads 2
```

程式第一次會匿名取得固定版本的四份公開檔案，核對 SHA-256 指紋後讀入權重，再依訊息檔真正生成回答；後續可重用下載的檔案。這是推論，不會更新模型。輸出為 JSON，先找 `answer`，應為上表的兩點；再看 `generations[0].eos`，應為 `true`。文字中的 `\n` 表示換行。`model.selected_step` 則指出發布的權重取自這段訓練的第1,000步。

這份訊息檔的 system 設定只說明有限助理的任務，沒有放入當前問題的答案。若要看完整輸入，可直接打開[示範訊息檔](../../docs/selftrained/examples/v2/text.messages.json)。其他圖片、讀字、語音與工具的 CPU 命令及資料準備見[公開操作頁](../../docs/selftrained/v2-public-cpu-commands.md)。

</details>

接著把同一核心用在別的材料。商品圖會先經過商品入口，指定字卡經過讀字入口，錄音經過聲音入口；入口把內容變成核心可讀的向量，放進它的輸入序列。下圖的箭頭都朝向同一核心，表示它們最後使用同一份文字權重來回答。

![成品資料流示意：文字與歷史直接進入共同核心；商品、指定字卡及聲音先經過各自入口，再由同一核心回答或提出工具請求。](../figures/p6-19-start-core.svg)

一個文字例子只走了一條路。整合檢查還要改材料、改問題和改歷史，逐項看答案是否跟著條件改變。

| 本章材料 | 要核對的行為 | 使用範圍 |
| --- | --- | --- |
| 短問答、商品清單與對話歷史 | 內容、選取範圍、格式和正常結束 | 有限中文指令與教過類型的新問法、組合 |
| Fashion-MNIST 灰階商品圖 | 商品類別；兩格場景的左右或上下關係 | 褲子、包、短靴三類；位置格由呼叫程式提供 |
| 12字「大小上下左右開關入出人口」組成的字卡 | 讀出指定區域，保留字序及要求的換行 | 一個已提供的連續1–4格區域；Sans、Serif兩種已學過的字型 |
| MInDS-14 中文真人錄音 | 用聲音內容決定客服短答，再沿用對話歷史 | 地址、App錯誤、卡片問題三種意圖；輸出文字回答 |
| 小整數計算要求 | 生成合法工具請求，讀取真結果，再生成答案 | 有界的計算器往返 |

例如，把兩格商品圖左右交換，回答也應交換；把「兩點」改成「一句」，內容仍應正確而格式改變。這樣才能區分「認出素材」「選對範圍」和「照要求回答」。上面的成功案例是可重做的示範，整體成功率仍要由各項獨立題目判定。

