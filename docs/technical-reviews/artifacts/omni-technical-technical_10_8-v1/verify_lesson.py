"""Run the two frozen lesson blocks verbatim, then inspect their key contracts on CPU."""

import contextlib
from datetime import datetime, timezone
import hashlib
import inspect
import io
import json
from pathlib import Path
import platform
import re
import sys

import torch


ROOT = Path.cwd()
PACKET = ROOT / "docs/course-revision-20261029-omni/reviews/technical-stage/dispatch-briefs-v9/technical_10_8.json"
packet = json.loads(PACKET.read_text())
page = packet["page"]
raw = (ROOT / page["snapshot"]).read_bytes()
assert hashlib.sha256(raw).hexdigest() == page["source_sha256"]
blocks = re.findall(r"```python\n(.*?)\n```", raw.decode(), re.S)
assert len(blocks) == 2
namespace = {}
original_stdout = io.StringIO()
with contextlib.redirect_stdout(original_stdout):
    for index, block in enumerate(blocks):
        exec(compile(block, f"{page['snapshot']}:block-{index + 1}", "exec"), namespace)

lm = namespace["lm"]
ours = namespace["ours"]
library = namespace["library"]
ids = namespace["ids"]
sequence = namespace["sequence"]
scene = namespace["scene"]
tok = namespace["ByteTokenizer"]()
image = scene()
language_calls = []
vision_calls = []
projector_calls = []
embedding_calls = []


def language_hook(module, args, kwargs):
    language_calls.append({
        "positional_ids": args[0].tolist() if args else None,
        "embeddings_shape": list(kwargs["embeddings"].shape) if "embeddings" in kwargs else None,
        "keyword_names": sorted(kwargs),
    })


def vision_hook(module, args, output):
    vision_calls.append({"input_shape": list(args[0].shape), "output_shape": list(output.shape)})


def projector_hook(module, args, output):
    projector_calls.append({"input_shape": list(args[0].shape), "output_shape": list(output.shape)})


def embedding_hook(module, args, output):
    embedding_calls.append(args[0].tolist())


handles = [
    lm.register_forward_pre_hook(language_hook, with_kwargs=True),
    ours.vision.register_forward_hook(vision_hook),
    ours.image_projector.register_forward_hook(projector_hook),
    lm.embedding.register_forward_hook(embedding_hook),
]

with torch.no_grad():
    text_direct = lm(ids[None])["logits"]
    language_calls.clear()
    text_wrapped = ours(ids)["logits"]
    text_route = {
        "language_calls": list(language_calls),
        "vision_call_count": len(vision_calls),
        "projector_call_count": len(projector_calls),
    }
    assert len(language_calls) == 1 and not vision_calls and not projector_calls
    assert language_calls[0]["positional_ids"] == [[1, 20, 30]]
    assert torch.equal(text_direct, text_wrapped)
    language_calls.clear()
    embedding_calls.clear()
    image_wrapped = ours(sequence, image=image)
    image_route = {
        "language_calls": list(language_calls),
        "vision_calls": list(vision_calls),
        "projector_calls": list(projector_calls),
        "embedding_ids": list(embedding_calls),
    }
    image_library = library(sequence, image=image)
    assert torch.equal(image_wrapped["logits"], image_library["logits"])
    assert image_wrapped["labels"].shape == (1, 19)
    assert (image_wrapped["labels"] == -100).all()
    assert image_route["embedding_ids"] == [[1], [4], [73], [2]]
    assert image_route["language_calls"][0]["embeddings_shape"] == [1, 19, 8]
    labels = torch.tensor([-100, -100, -100, 73, 2])
    labelled_ours = ours(sequence, labels, image=image)
    labelled_library = library(sequence, labels, image=image)
    assert torch.equal(labelled_ours["labels"], labelled_library["labels"])
    assert labelled_ours["labels"].tolist() == [[-100] * 17 + [73, 2]]
    assert torch.equal(labelled_ours["logits"], image_wrapped["logits"])
    language_calls.clear()
    try:
        ours(torch.tensor([1, 5, 4]))
    except ValueError as error:
        exercise_error = f"{type(error).__name__}: {error}"
    else:
        raise AssertionError("Exercise must reject an image marker without an image")
    assert exercise_error == "ValueError: placeholder 缺少配對圖片"
    assert not language_calls
    try:
        ours.forward(sequence, None, image)
    except TypeError as error:
        positional_image_error = f"{type(error).__name__}: {error}"
    else:
        raise AssertionError("image is keyword-only")
    text_embeddings = lm.embedding(ids[None])
    embeddings_direct = lm(embeddings=text_embeddings)["logits"]
    assert torch.equal(embeddings_direct, text_direct)
    default_labels = torch.full_like(sequence, -100)
    assert default_labels.tolist() == [-100] * 5

for handle in handles:
    handle.remove()

# Reconstruct initial modules solely to check separate random visual parameters.
# No optimisation or training is performed.
torch.manual_seed(0)
initial_lm = namespace["TinyLM"](namespace["ModelConfig"](width=8))
initial_ours = namespace["MultiModalLM"](initial_lm)
initial_library = namespace["LibraryMultiModalLM"](initial_lm)
initial_separate = {
    "same_language_object": initial_ours.language is initial_library.language is initial_lm,
    "same_vision_object": initial_ours.vision is initial_library.vision,
    "same_projector_object": initial_ours.image_projector is initial_library.image_projector,
    "vision_weights_equal_before_loading": torch.equal(initial_ours.vision.projection.weight, initial_library.vision.projection.weight),
    "projector_weights_equal_before_loading": torch.equal(initial_ours.image_projector.weight, initial_library.image_projector.weight),
}
assert initial_separate == {
    "same_language_object": True,
    "same_vision_object": False,
    "same_projector_object": False,
    "vision_weights_equal_before_loading": False,
    "projector_weights_equal_before_loading": False,
}
matching_keys = sorted(ours.state_dict())
assert all(torch.equal(value, library.state_dict()[name]) for name, value in ours.state_dict().items())
assert all(name.startswith(("audio.", "audio_projector.")) for name in namespace["loaded"].unexpected_keys)
assert tok.image_id == 5 and tok.bos_id == 1 and tok.vocab_size == 264
assert all(8 <= token < tok.vocab_size for token in [20, 30])
assert text_direct.shape == (1, 3, 264)
assert image_wrapped["logits"].shape == (1, 19, 264)
assert ours.image_projector.weight.shape == (8, 16)
assert torch.equal(scene(), scene("red", "square"))

files = [
    "tiny_perceptron/multimodal.py", "tiny_perceptron/model.py",
    "tiny_perceptron/data.py", "tiny_perceptron/attention.py", "tiny_perceptron/modern.py",
]
result = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-technical_10_8-v1/verify_lesson.py",
    "environment": {
        "python": sys.version.split()[0], "torch": str(torch.__version__),
        "torch_git": str(torch.version.git_version), "device": "cpu",
        "default_dtype": str(torch.get_default_dtype()), "platform": platform.platform(),
    },
    "source_snapshot": page["snapshot"],
    "source_sha256": hashlib.sha256(raw).hexdigest(),
    "code_block_sha256": [hashlib.sha256(block.encode()).hexdigest() for block in blocks],
    "repository_code_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files},
    "original_stdout": original_stdout.getvalue().splitlines(),
    "original_assertions": "passed; both frozen Python blocks executed without modification",
    "text": {
        "input_ids": ids.tolist(), "input_dtype": str(ids.dtype),
        "logits_shape": list(text_direct.shape), "logits_dtype": str(text_direct.dtype),
        "element_count": text_direct.numel(), "equal": torch.equal(text_direct, text_wrapped),
        "max_abs_difference": (text_direct - text_wrapped).abs().max().item(),
        "route": text_route,
        "embeddings_route_equal": torch.equal(embeddings_direct, text_direct),
    },
    "image": {
        "sequence": sequence.tolist(), "image_shape": list(image.shape), "image_dtype": str(image.dtype),
        "default_scene_is_red_square": torch.equal(image, scene("red", "square")),
        "logits_shape": list(image_wrapped["logits"].shape),
        "equal": torch.equal(image_wrapped["logits"], image_library["logits"]),
        "max_abs_difference": (image_wrapped["logits"] - image_library["logits"]).abs().max().item(),
        "route": image_route,
        "full_like_ids_shape": list(default_labels.shape), "full_like_dtype": str(default_labels.dtype),
        "full_like_values": default_labels.tolist(),
        "default_returned_labels_shape": list(image_wrapped["labels"].shape),
        "default_returned_labels_all_ignore": bool((image_wrapped["labels"] == -100).all()),
        "supplied_labels": labels.tolist(), "shifted_returned_labels": labelled_ours["labels"].tolist(),
        "effective_target_positions": (labelled_ours["labels"] != -100).nonzero().tolist(),
        "returned_labels_equal_library": torch.equal(labelled_ours["labels"], labelled_library["labels"]),
    },
    "state_loading": {
        "missing_keys": namespace["loaded"].missing_keys,
        "unexpected_keys": namespace["loaded"].unexpected_keys,
        "matching_key_count": len(matching_keys), "all_matching_parameters_equal_after_loading": True,
        "initial_separate": initial_separate, "projector_weight_shape": list(ours.image_projector.weight.shape),
    },
    "python_signature": str(inspect.signature(ours.forward)),
    "positional_image_error": positional_image_error,
    "exercise_error": exercise_error,
    "exercise_language_calls": 0,
    "scope": "CPU float32, seed 0, default TinyLM configuration except width=8; no optimisation, training, generation-quality, GPU or general platform test",
    "result": "all assertions passed",
}
print(json.dumps(result, ensure_ascii=False, indent=2))
