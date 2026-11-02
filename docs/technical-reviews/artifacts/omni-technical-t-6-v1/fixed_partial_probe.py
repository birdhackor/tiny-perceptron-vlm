"""Verify fixed-experiment partial scope without optimization or training."""
from datetime import datetime, UTC
import json
from pathlib import Path
import platform
import torch
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.multimodal import MultiModalLM
from scripts.course_experiments.modalities import _freeze

model=MultiModalLM(TinyLM(ModelConfig(width=32,layers=3,heads=1,max_length=128)))
_freeze(model,'partial')
names=[n for n,p in model.named_parameters() if p.requires_grad]
assert any(n.startswith('language.blocks.2.') for n in names)
assert not any(n.startswith(('language.blocks.0.','language.blocks.1.','audio_projector.')) for n in names)
report={'created_at':datetime.now(UTC).isoformat(),
        'command':'uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-t-6-v1/fixed_partial_probe.py',
        'environment':{'python':platform.python_version(),'torch':str(torch.__version__),'device':'cpu'},
        'input':{'width':32,'layers':3,'scope':'partial','modality':'vision'},
        'trainable_names':names,
        'result':'Only image projector and last language block are trainable; no optimizer or training was run.'}
Path(__file__).with_name('fixed-partial-probe.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False))
