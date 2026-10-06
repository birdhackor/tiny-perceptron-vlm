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

