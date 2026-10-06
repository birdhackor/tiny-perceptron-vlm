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

