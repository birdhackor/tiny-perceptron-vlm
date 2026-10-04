import hashlib
import json
from pathlib import Path
import torch
from tiny_perceptron.posttraining import FiniteResponsePolicy

torch.set_num_threads(2)
records = {}
for stage in ['pretrain', 'sft', 'joint', 'dpo']:
    path = Path(f'docs/course-experiments/capstone-evidence/{stage}/train-report.json')
    records[stage] = json.loads(path.read_text())
    for file, digest in records[stage]['code_sha256'].items():
        assert hashlib.sha256(Path(file).read_bytes()).hexdigest() == digest
for parent, child in [('pretrain', 'sft'), ('sft', 'joint'), ('joint', 'dpo')]:
    assert records[child]['parent_checkpoint_sha256'] == records[parent]['inference_export']['sha256']
policy = FiniteResponsePolicy()
logits = policy(torch.ones(1, 4))
assert list(logits.shape) == [1, 4]
out = {'code_matches_original_four_stage_receipts': True,
       'complete_parent_export_sha_chain': True,
       'finite_response_policy_shape': list(logits.shape),
       'scope': 'Source/provenance and four-candidate shape checks; no training or model-quality replication.'}
Path('docs/technical-reviews/artifacts/natural-v4-factual/R.4/provenance-results.json').write_text(json.dumps(out, indent=2)+'\n')
print(json.dumps(out, indent=2))
