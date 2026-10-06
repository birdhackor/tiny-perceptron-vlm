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
