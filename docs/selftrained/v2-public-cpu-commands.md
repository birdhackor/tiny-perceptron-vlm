# V2 公開模型 CPU 操作命令

以下 7 組情境已用公開權重完成一次 CPU smoke，共 8 次 chat 呼叫與 1 次 history append；所有呼叫成功，回答與保存的 GPU validation 示範相同。命令保留當時實際 argv 的模型、訊息、預處理和生成參數，只把環境的 Python 路徑改為 `uv`，把檔案位置改為 repo-relative。原始執行記錄：`docs/selftrained/infrastructure/v2-public-cpu-smoke-actual-review.json`。

這些是挑選出的成功 validation 情境。它們可用來確認安裝、讀取權重與介面是否能運作，不代表未知問題的成功率。該 MoE 的最後一次 heldout test 中，完整工具往返為 **0/276**、語音回答為 **42/90**、語音後續對話為 **26/60**。這些限制不因示範成功而改變。

## 取得依賴、資料和公開權重

所有命令都從 repository 根目錄執行。

```sh
uv sync --extra cpu --extra selftrained
git lfs pull --include="assets/training/selftrained-v2.tar.gz" --exclude=""
```

如果還沒解開資料包，使用現有的 checksum-bound unpacker：

```sh
uv run --extra cpu --extra selftrained python - <<'PY'
import json
from pathlib import Path
from scripts.selftrained.hf_transport import unpack_verified_archive

manifest = json.loads(Path("docs/selftrained/v2-manifest.json").read_text())
unpack_verified_archive(
    "assets/training/selftrained-v2.tar.gz",
    "outputs/selftrained-v2/data",
    manifest["package"],
)
PY
```

`chat.py` 會匿名取得四個公開 safe files，不需要私人 `.pt`、optimizer 或 Modal。指定的 HF revision 是 `979cdfacc588ad0536f1c64fff96f264571cf054`；MoE joint 的 inference manifest SHA 是 `f27856892b0c46ca81ba43bfadf7051ed6bd31d848e4868a05147e5799df1e5e`。這份公開權重選自該輪 step 1,000，而該輪訓練實際完成 4,000 steps。

`docs/selftrained/examples/v2` 的 10 個 JSON 只提供 user 訊息、先前格式確認與公開 ROI／位置資料。它們不包含目前的標準答案、評分標籤或保存的實際模型回答。

## 文字：App 問題與兩點格式

```sh
uv run --extra cpu --extra selftrained python scripts/selftrained/chat.py \
  --model-dir outputs/selftrained-v2/public/moe-joint \
  --asset-dir outputs/selftrained-v2/data \
  --repo birdhackor/tiny-perceptron-course-models \
  --revision 979cdfacc588ad0536f1c64fff96f264571cf054 \
  --prefix selftrained/v2/moe-joint \
  --manifest-sha256 f27856892b0c46ca81ba43bfadf7051ed6bd31d848e4868a05147e5799df1e5e \
  --messages docs/selftrained/examples/v2/text.messages.json \
  --task text \
  --device cpu \
  --max-new-tokens 128 \
  --threads 2
```

## OCR：讀取提供的文字區域

```sh
uv run --extra cpu --extra selftrained python scripts/selftrained/chat.py \
  --model-dir outputs/selftrained-v2/public/moe-joint \
  --asset-dir outputs/selftrained-v2/data \
  --repo birdhackor/tiny-perceptron-course-models \
  --revision 979cdfacc588ad0536f1c64fff96f264571cf054 \
  --prefix selftrained/v2/moe-joint \
  --manifest-sha256 f27856892b0c46ca81ba43bfadf7051ed6bd31d848e4868a05147e5799df1e5e \
  --messages docs/selftrained/examples/v2/ocr.messages.json \
  --task ocr \
  --device cpu \
  --max-new-tokens 128 \
  --image images/ocr/3a8d1184bb954b6f30c6fc09b31c678e.png \
  --roi 18,50,90,86 \
  --threads 2
```

## 圖片：三類服飾辨識

```sh
uv run --extra cpu --extra selftrained python scripts/selftrained/chat.py \
  --model-dir outputs/selftrained-v2/public/moe-joint \
  --asset-dir outputs/selftrained-v2/data \
  --repo birdhackor/tiny-perceptron-course-models \
  --revision 979cdfacc588ad0536f1c64fff96f264571cf054 \
  --prefix selftrained/v2/moe-joint \
  --manifest-sha256 f27856892b0c46ca81ba43bfadf7051ed6bd31d848e4868a05147e5799df1e5e \
  --messages docs/selftrained/examples/v2/vision_clothing.messages.json \
  --task vision_clothing \
  --device cpu \
  --max-new-tokens 128 \
  --image images/vision/c98cb897d639c880fde261eafc6d953d.png \
  --image-layout docs/selftrained/examples/v2/vision_clothing.layout.json \
  --threads 2
```

## 圖片：兩個指定位置的關係

```sh
uv run --extra cpu --extra selftrained python scripts/selftrained/chat.py \
  --model-dir outputs/selftrained-v2/public/moe-joint \
  --asset-dir outputs/selftrained-v2/data \
  --repo birdhackor/tiny-perceptron-course-models \
  --revision 979cdfacc588ad0536f1c64fff96f264571cf054 \
  --prefix selftrained/v2/moe-joint \
  --manifest-sha256 f27856892b0c46ca81ba43bfadf7051ed6bd31d848e4868a05147e5799df1e5e \
  --messages docs/selftrained/examples/v2/vision_relation.messages.json \
  --task vision_relation \
  --device cpu \
  --max-new-tokens 128 \
  --image images/vision/13913fb756b87b5c3e0f52e4ba194fec.png \
  --image-layout docs/selftrained/examples/v2/vision_relation.layout.json \
  --threads 2
```

## 工具：模型呼叫真正的計算器

```sh
uv run --extra cpu --extra selftrained python scripts/selftrained/chat.py \
  --model-dir outputs/selftrained-v2/public/moe-joint \
  --asset-dir outputs/selftrained-v2/data \
  --repo birdhackor/tiny-perceptron-course-models \
  --revision 979cdfacc588ad0536f1c64fff96f264571cf054 \
  --prefix selftrained/v2/moe-joint \
  --manifest-sha256 f27856892b0c46ca81ba43bfadf7051ed6bd31d848e4868a05147e5799df1e5e \
  --messages docs/selftrained/examples/v2/tool_call.messages.json \
  --task tool_call \
  --device cpu \
  --max-new-tokens 128 \
  --tools \
  --threads 2
```

## 語音輸入：回答有限客服問題

```sh
uv run --extra cpu --extra selftrained python scripts/selftrained/chat.py \
  --model-dir outputs/selftrained-v2/public/moe-joint \
  --asset-dir outputs/selftrained-v2/data \
  --repo birdhackor/tiny-perceptron-course-models \
  --revision 979cdfacc588ad0536f1c64fff96f264571cf054 \
  --prefix selftrained/v2/moe-joint \
  --manifest-sha256 f27856892b0c46ca81ba43bfadf7051ed6bd31d848e4868a05147e5799df1e5e \
  --messages docs/selftrained/examples/v2/voice_qa.messages.json \
  --task voice_qa \
  --device cpu \
  --max-new-tokens 128 \
  --audio audio/ef25d9a7a3a6ce3790a040ba.wav \
  --modality-message-index 3 \
  --threads 2
```

## 語音後續對話：使用真正生成的第一輪回答

先產生第一輪回答並保存完整對話；`--history-output` 的內容包含原 user 訊息的 audio 路徑和本次模型真正生成的 assistant 回答。

```sh
uv run --extra cpu --extra selftrained python scripts/selftrained/chat.py \
  --model-dir outputs/selftrained-v2/public/moe-joint \
  --asset-dir outputs/selftrained-v2/data \
  --repo birdhackor/tiny-perceptron-course-models \
  --revision 979cdfacc588ad0536f1c64fff96f264571cf054 \
  --prefix selftrained/v2/moe-joint \
  --manifest-sha256 f27856892b0c46ca81ba43bfadf7051ed6bd31d848e4868a05147e5799df1e5e \
  --messages docs/selftrained/examples/v2/voice_topic_continuation.messages.json \
  --task voice_topic_continuation \
  --device cpu \
  --max-new-tokens 128 \
  --audio audio/ef25d9a7a3a6ce3790a040ba.wav \
  --modality-message-index 3 \
  --history-output outputs/selftrained-v2/smoke/voice-first-history.json \
  --threads 2
```

helper 只讀取剛才保存的實際 history，再加入 `voice-continuation-user.json` 裡的 user 改寫要求：

```sh
uv run --extra cpu --extra selftrained python docs/selftrained/examples/v2/append-voice-continuation.py
```

再用新的對話檔繼續生成。音訊保留在原來的 user 訊息，不把語音當成新的問題，也不從示範輸出貼入第一輪答案。

```sh
uv run --extra cpu --extra selftrained python scripts/selftrained/chat.py \
  --model-dir outputs/selftrained-v2/public/moe-joint \
  --asset-dir outputs/selftrained-v2/data \
  --repo birdhackor/tiny-perceptron-course-models \
  --revision 979cdfacc588ad0536f1c64fff96f264571cf054 \
  --prefix selftrained/v2/moe-joint \
  --manifest-sha256 f27856892b0c46ca81ba43bfadf7051ed6bd31d848e4868a05147e5799df1e5e \
  --messages outputs/selftrained-v2/smoke/voice-next-messages.json \
  --task voice_topic_continuation \
  --device cpu \
  --max-new-tokens 128 \
  --threads 2
```

## 如何查看輸出

每個 chat 命令輸出一份 JSON：

- `answer`：目前的最終文字回答。
- `messages`：包含此次真正生成回答的對話，可保存作下一輪輸入。
- `generations`：每次模型生成的輸入訊息、生成 tokens 與原始文字；工具案例會有兩次生成。
- `tool_trace`：模型輸出的 call、實際 executor result 和最終回答。
- `model`：選用 checkpoint step、公開 payload hashes 和模型配置。
- `public_source`：指定的 immutable HF revision，並標示 `authentication: disabled`。

這次實際 CPU 執行觀察到的 `answer`：

| 情境 | 實際 CPU 回答 |
|---|---|
| text | 1. 先檢查網路並重新啟動App。<br>2. 仍失敗再詢問官方客服。 |
| ocr | 大小 |
| vision_clothing | 這是短靴。 |
| vision_relation | 左邊是褲子。 |
| tool_call | 結果是416。 |
| voice_qa | 先檢查網路並重新啟動App，仍失敗再詢問官方客服。 |
| voice_topic_continuation：第一輪 | 先檢查網路並重新啟動App，仍失敗再詢問官方客服。 |
| voice_topic_continuation：第二輪 | 1. 先檢查網路並重新啟動App。<br>2. 仍失敗再詢問官方客服。 |

計算器案例先由模型產生 `26 × 16` 的合法 JSON call，executor 實際回傳 416，再由模型生成 `結果是416。`；程式沒有先替模型從題目擷取答案。

圖片案例只涵蓋三類服飾、公開兩格位置，以及提供 ROI 的有限字集 OCR。語音是有限客服情境的語音輸入轉文字回答；本示範不代表一般 ASR 或語音輸出。此次語音完整回答雖然正確，獨立音訊分類 head 的 argmax 與標籤不同，不能把它說成分類 head 已通過。
