"""Bounded CPU checks of R.4 mechanisms, actual contracts, and navigation.

No dataset downloads, training loops, optimizer updates, saved weights, or model
capability evaluations. Fixed tensors only test the existing computation.
"""

import hashlib
import json
import math
import platform
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn

from tiny_perceptron.attention import manual_attention
from tiny_perceptron.capstone import CapstoneModel, TOK, prompt_ids
from scripts.capstone_release import validate_manifest

torch.set_num_threads(1)
torch.manual_seed(41004)
assert torch.version.cuda is None

record = {
    "environment": {
        "python": sys.version,
        "executable": sys.executable,
        "torch": str(torch.__version__),
        "torch_git_version": str(torch.version.git_version),
        "cuda_build": str(torch.version.cuda),
        "device": "cpu",
        "platform": platform.platform(),
    },
    "scope": "Fixed CPU tensors and local manifest/navigation checks; no training or capability scores.",
}

# The input ID selects a table row, rather than supplying a semantic meaning.
embedding = nn.Embedding(3, 2)
with torch.no_grad():
    embedding.weight.copy_(torch.tensor([[1., 2.], [3., 4.], [5., 6.]]))
    looked_up = embedding(torch.tensor([2, 0, 2]))
expected_lookup = torch.tensor([[5., 6.], [1., 2.], [5., 6.]])
assert torch.equal(looked_up, expected_lookup)
record["embedding_lookup"] = looked_up.tolist()

# Independently calculate the scaled dot-product weights and weighted values.
q = torch.tensor([[[[1., 0.]]]], dtype=torch.float64)
k = torch.tensor([[[[1., 0.], [0., 1.]]]], dtype=torch.float64)
v = torch.tensor([[[[10., 0.], [0., 20.]]]], dtype=torch.float64)
allowed = torch.ones(1, 1, 1, 2, dtype=torch.bool)
out, weights = manual_attention(q, k, v, allowed)
p = math.exp(1 / math.sqrt(2)) / (math.exp(1 / math.sqrt(2)) + 1)
expected_weights = torch.tensor([[[[p, 1-p]]]], dtype=torch.float64)
expected_out = torch.tensor([[[[10*p, 20*(1-p)]]]], dtype=torch.float64)
assert torch.allclose(weights, expected_weights, atol=1e-12, rtol=0)
assert torch.allclose(out, expected_out, atol=1e-12, rtol=0)
changed_out, changed_weights = manual_attention(q.flip(-1), k, v, allowed)
assert torch.allclose(changed_weights, expected_weights.flip(-1), atol=1e-12, rtol=0)
assert not torch.equal(changed_out, out)
record["attention"] = {"weights": weights.tolist(), "output": out.tolist(), "changed_query_weights": changed_weights.tolist(), "changed_query_output": changed_out.tolist(), "absolute_tolerance": 1e-12}

# Capture what the actual Capstone forward passes into the one language core.
row = {"system": "", "user": "x?", "image": True, "audio": True}
ids = torch.tensor([prompt_ids(row)])
model = CapstoneModel().eval()
captured = []
hook = model.language.register_forward_pre_hook(lambda module, args, kwargs: captured.append(kwargs["embeddings"].detach().clone()), with_kwargs=True)
image = torch.zeros(1, 3, 16, 16)
audio = torch.arange(16, dtype=torch.float32).reshape(1, 16) / 16
state_before = {name: p.detach().clone() for name, p in model.named_parameters()}
with torch.no_grad():
    model(ids, images=image, audio_features=audio)
    model(ids, images=image+1, audio_features=audio)
hook.remove()
image_index = int((ids[0] == TOK.image_id).nonzero()[0])
audio_index = int((ids[0] == TOK.audio_id).nonzero()[0])
text_start = audio_index+1
assert image_index < audio_index < text_start
expected_image = model.image_projector(torch.zeros(1, 48))
expected_audio = model.audio_projector(audio)
assert torch.equal(captured[0][:, image_index], expected_image)
assert torch.equal(captured[0][:, audio_index], expected_audio)
ordinary = (ids != TOK.image_id) & (ids != TOK.audio_id)
assert torch.equal(captured[0][ordinary], model.language.embedding(ids)[ordinary])
assert not torch.equal(captured[0][:, image_index], captured[1][:, image_index])
assert torch.equal(captured[0][:, audio_index], captured[1][:, audio_index])
assert torch.equal(captured[0][ordinary], captured[1][ordinary])
assert all(torch.equal(state_before[n], p) for n, p in model.named_parameters())
try:
    model(ids, images=None, audio_features=audio)
except ValueError as exc:
    missing_error = str(exc)
else:
    raise AssertionError("Missing image features must raise ValueError")
record["capstone_prefix"] = {
    "token_ids": ids.tolist(), "image_index": image_index, "audio_index": audio_index,
    "question_text_start": text_start, "embedding_shape": list(captured[0].shape),
    "image_projection_matches": True, "audio_projection_matches": True,
    "ordinary_text_embeddings_preserved": True, "image_change_affects_only_image_slot_at_core_input": True,
    "weights_unchanged": True, "missing_image_error": missing_error,
    "scope": "The existing synthetic Capstone connector contract; it does not implement or validate the planned new encoders/tasks.",
}

manifest_path = ROOT / 'docs/course-experiments/capstone-public.json'
manifest = json.loads(manifest_path.read_bytes())
validate_manifest(manifest)
record["public_manifest"] = {"repo": manifest['repo'], "revision": manifest['revision'], "ids": [x['id'] for x in manifest['models']], "validation": "pass", "no_download_performed": True}

source = ROOT / "course/README.md"
raw = source.read_bytes()
matches = list(re.finditer(rb"(?m)^## [^\r\n]+", raw))
heading = next(i for i,m in enumerate(matches) if m[0].startswith(b"## R.4 "))
end = matches[heading+1].start() if heading+1 < len(matches) else len(raw)
section = raw[matches[heading].start():end]
body = section.decode('utf-8')
links = re.findall(r"\]\(([^)]+)\)", body)
link_checks=[]
for link in links:
    path, _, fragment = link.partition('#')
    target=(source.parent/path).resolve()
    assert target.is_file(), link
    text=target.read_text()
    if fragment:
        assert re.search(r"(?m)^## " + re.escape(fragment) + r" ", text),link
    link_checks.append({"target":link,"file_exists":True,"numbered_anchor_exists":True if fragment else None})
training_text=(ROOT/'course/training.md').read_text()
chapter19_text=(ROOT/'course/chapters/19.md').read_text()
record['navigation']={
    'section_sha256':hashlib.sha256(section).hexdigest(),
    'r4_links':link_checks,
    'r4_python_or_shell_fences':len(re.findall(r'(?m)^```',body)),
    'r4_image_references':len(re.findall(r'!\[|<img',body)),
    'training_page_fetch_capstone_occurrences':training_text.count('fetch_capstone.py'),
    'training_page_chapter19_link':'](chapters/19.md)' in training_text,
    'chapter19_download_command_present':'python scripts/fetch_capstone.py --stage joint' in chapter19_text,
    'note':'The operation page links chapter 19, but direct joint retrieval and usage commands are in chapter 19.',
}
print(json.dumps(record, ensure_ascii=False, indent=2))
