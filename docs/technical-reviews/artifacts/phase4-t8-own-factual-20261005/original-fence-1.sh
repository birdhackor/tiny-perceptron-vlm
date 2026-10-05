.venv/bin/python scripts/prepare_data.py --kind toy-text --seed 42
.venv/bin/python scripts/train.py --task text --data data/generated/toy-text/train.jsonl --train --steps 200 --seed 42 --output checkpoints/baseline.pt
.venv/bin/python scripts/train.py --task text --data data/generated/toy-text/train.jsonl --train --steps 200 --seed 42 --norm rms --output checkpoints/rms.pt
.venv/bin/python scripts/train.py --task text --data data/generated/toy-text/train.jsonl --train --steps 200 --seed 42 --experts 4 --top-k 2 --output checkpoints/moe.pt
