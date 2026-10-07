PENDING_CHECKPOINTS=$(
  .venv-natural/bin/python -c 'import json; from pathlib import Path; n = json.loads(Path("outputs/natural-my-v4/train/adapter/training.json").read_text())["completed_steps"]; assert 0 <= n < 2077, "先確認尚未完成2,077步"; print(",".join(str(step) for step in (1039, 2077) if step > n))'
) && \
.venv-natural/bin/python scripts/natural_assistant.py train \
  --output outputs/natural-my-v4/resumed \
  --manifest docs/natural-assistant/v4/manifest.json \
  --data-root data/natural-v4 \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --model-revision 89644892e4d85e24eaac8bacfd4f463576704203 \
  --asr-variant turbo \
  --device cuda --dtype bfloat16 \
  --min-pixels 65536 --max-pixels 524288 \
  --max-tokens 2048 --max-new-tokens 384 --seed 42 \
  --cache-dir .cache/natural-v4-models --local-files-only \
  --adapter outputs/natural-my-v4/train/adapter \
  --steps 2077 --learning-rate 0.00003 \
  --lora-rank 8 --gradient-accumulation 2 \
  --checkpoint-every 25 --checkpoint-steps "$PENDING_CHECKPOINTS" \
  --max-seconds 3300
