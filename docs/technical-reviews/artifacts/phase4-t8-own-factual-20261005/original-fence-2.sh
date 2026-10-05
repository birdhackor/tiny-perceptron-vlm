.venv/bin/python scripts/evaluate.py checkpoints/baseline.pt --data data/generated/toy-text/validation.jsonl --mode text --tokens 32 --limit all --output outputs/baseline-validation.json
.venv/bin/python scripts/evaluate.py checkpoints/rms.pt --data data/generated/toy-text/validation.jsonl --mode text --tokens 32 --limit all --output outputs/rms-validation.json
.venv/bin/python scripts/evaluate.py checkpoints/moe.pt --data data/generated/toy-text/validation.jsonl --mode text --tokens 32 --limit all --output outputs/moe-validation.json
