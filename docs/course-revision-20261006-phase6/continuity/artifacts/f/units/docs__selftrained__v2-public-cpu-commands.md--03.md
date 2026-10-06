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

