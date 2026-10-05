.venv/bin/python scripts/infer.py outputs/course-experiments/course-v1/distillation/sft-w32-ce.pt --chat --prompt "color=red;shape=square;pitch=high;joint?" --tokens 24 --device cpu --json
.venv/bin/python scripts/infer.py outputs/course-experiments/course-v1/distillation/sft-w32-ce_kl.pt --chat --prompt "color=red;shape=square;pitch=high;joint?" --tokens 24 --device cpu --json
