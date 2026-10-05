"""Independent, bounded R.3 checks; no training, installs, or model downloads."""
import hashlib
import json
import math
import platform
import re
import sys
from pathlib import Path

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
import torch
from torch.nn import functional as F
from tiny_perceptron.attention import manual_attention
from tiny_perceptron.capstone import CapstoneModel, build_dataset, modality_tensors, prepare_batch
from tiny_perceptron.data import ByteTokenizer, render_chat
from tiny_perceptron.efficiency import online_attention
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.natural_assistant import messages_for
from tiny_perceptron.quantization import pack_int4, quantize_symmetric, unpack_int4
from tiny_perceptron.retrieval import call_tool, retrieve

torch.set_num_threads(1)
torch.manual_seed(0)
result = {"environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu", "cuda_available": torch.cuda.is_available(), "threads": torch.get_num_threads(), "seed": 0}}
chapters = []
notebook_failures = []
notebooks = 0
for path in sorted((ROOT / "course/chapters").glob("*.md")):
    text = path.read_text()
    sections = re.findall(r"^## ([A-Z\d]+\.\d+) (.+)$", text, re.M)
    chapters.append({"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "title": text.splitlines()[0], "sections": [identifier + " " + title for identifier, title in sections]})
    headings = list(re.finditer(r"^## .+$", text, re.M))
    for position, match in enumerate(headings):
        identifier, title = re.match(r"## ([A-Z\d]+\.\d+) (.+)", match[0]).groups()
        end = headings[position + 1].start() if position + 1 < len(headings) else len(text)
        body = text[match.start():end]
        notebook_path = ROOT / "notebooks" / path.stem / (identifier + ".ipynb")
        if not notebook_path.exists():
            notebook_failures.append({"section": identifier, "error": "missing notebook"})
            continue
        notebook = json.loads(notebook_path.read_text())
        notebooks += 1
        if "".join(notebook["cells"][0]["source"]).strip() != f"# {identifier} {title}":
            notebook_failures.append({"section": identifier, "error": "title mismatch"})
        python_blocks = [b.strip() for b in re.findall(r"```python\n(.*?)```", body, re.S)]
        notebook_blocks = ["".join(cell["source"]).strip() for cell in notebook["cells"] if cell["cell_type"] == "code" and not cell.get("metadata", {}).get("course_setup") and not cell.get("metadata", {}).get("course_figure")]
        if python_blocks != notebook_blocks:
            notebook_failures.append({"section": identifier, "error": "Python block mismatch"})
assert not notebook_failures, notebook_failures
result["curriculum"] = {"chapters": chapters, "chapter_count": len(chapters), "notebooks_checked": notebooks, "notebook_title_and_python_block_failures": notebook_failures, "note": "Read every chapter entry/title. Static parity check does not execute all lesson programs."}

alignment = []
for question, answer, expected_first, expected_id in [("Q", "A", 4, 73), ("QQ", "A", 5, 73), ("QQ", "B", 5, 74)]:
    x, y = render_chat([{"role": "user", "content": question}, {"role": "assistant", "content": answer}])
    first = int((y != -100).nonzero()[0])
    assert (first, int(x[first]), int(y[first])) == (expected_first, 4, expected_id)
    alignment.append({"question": question, "answer": answer, "x": x.tolist(), "y": y.tolist(), "first": first, "input_at_first": int(x[first]), "target_at_first": int(y[first]), "effective_targets": int((y != -100).sum())})
result["answer_alignment"] = alignment
q = torch.tensor([[1., 0.], [0., 1.]], dtype=torch.float64)
k = q.clone()
v = torch.tensor([[10.], [20.]], dtype=torch.float64)
weights = (q @ k.T / math.sqrt(2)).softmax(-1)
expected = weights @ v
online = online_attention(q, k, v, block_size=1)
sdpa = F.scaled_dot_product_attention(q[None, None], k[None, None], v[None, None], dropout_p=0.)[0, 0]
assert torch.allclose(expected, online, atol=1e-12) and torch.allclose(expected, sdpa, atol=1e-12)
result["attention"] = {"query": q.tolist(), "key": k.tolist(), "value": v.tolist(), "weights": weights.tolist(), "expected": expected.tolist(), "online": online.tolist(), "sdpa": sdpa.tolist(), "max_online_error": float((expected-online).abs().max()), "max_sdpa_error": float((expected-sdpa).abs().max()), "scope": "CPU float64 formula equality; no GPU or performance claim"}
model = TinyLM(ModelConfig(width=8)).eval()
ids = torch.tensor([[1, 2, 3, 4]])
with torch.no_grad():
    full = model(ids)["logits"][:, -1]
    prefix = model(ids[:, :3])
    step = model(ids[:, 3:], cache=prefix["cache"])
    cached = step["logits"][:, 0]
error = float((full-cached).abs().max())
assert torch.allclose(full, cached, atol=1e-6)
result["kv_cache"] = {"input": ids.tolist(), "config": model.description()["config"], "prefill_key_shape": list(prefix["cache"][0][0].shape), "decode_key_shape": list(step["cache"][0][0].shape), "new_logits_shape": list(step["logits"].shape), "max_error": error, "atol": 1e-6, "scope": "One seeded CPU float32 fixed-length inference example; no throughput or memory saving benchmark"}
integers = torch.tensor([-8, -1, 0, 1, 7, 2, 3], dtype=torch.int8)
packed = pack_int4(integers)
restored = unpack_int4(packed, integers.shape)
assert torch.equal(integers, restored)
weight = torch.tensor([-0.7, 0.2, 0.7, 1.2])
quantized, scale = quantize_symmetric(weight, bits=4)
result["quantization"] = {"original_integers": integers.tolist(), "packed": packed.tolist(), "integer_payload_bytes_before": integers.numel()*integers.element_size(), "integer_payload_bytes_after": packed.numel()*packed.element_size(), "unpacked": restored.tolist(), "float_weights": weight.tolist(), "q4": quantized.tolist(), "scale": float(scale), "reconstruction": (quantized*scale).tolist(), "max_absolute_error": float((weight-quantized*scale).abs().max()), "scope": "Packing is lossless for integers; float quantization loses precision; scale/container overhead and kernel speed excluded"}
teacher = ["unknown", "unknown", "unknown"]
truth = ["2", "4", None]
checks = [answer == expected if expected is not None else answer == "unknown" for answer, expected in zip(teacher, truth)]
result["teacher_error_derivation"] = {"questions": ["1+1", "2+2", "unspecified dinner"], "teacher": teacher, "copied_student": teacher.copy(), "teacher_agreement": 3/3, "task_passes": checks, "task_accuracy": sum(checks)/3, "scope": "Hand-constructed counterexample to agreement implying correctness; no student training or quality claim"}
docs = [{"id": "d1", "text": "貓喜歡曬太陽"}, {"id": "d2", "text": "書店地址是青街8號"}]
hits = retrieve("書店地址", docs, k=1)
assert hits[0]["id"] == "d2"
miss = retrieve("營業處在哪", docs, k=1)
assert miss == []
tool_result = call_tool({"name": "multiply", "arguments": {"a": 123, "b": 45}})
assert tool_result == 5535
rejections = []
for request in [{"name": "delete_all", "arguments": {"a": 2, "b": 3}}, {"name": "add", "arguments": {"a": True, "b": 3}}]:
    try:
        call_tool(request)
        raise AssertionError("Invalid tool request accepted")
    except ValueError as error:
        rejections.append({"request": request, "error": str(error)})
result["retrieval_and_tools"] = {"documents": docs, "query": "書店地址", "hits": hits, "synonym_query": "營業處在哪", "synonym_hits": miss, "multiply_request": {"a": 123, "b": 45}, "tool_result": tool_result, "rejections": rejections, "scope": "Lexical toy retrieval and allowlisted arithmetic; not semantic retrieval, general entailment, or autonomous model tool use"}
splits, _ = build_dataset()
row = next(row for row in splits["train"] if row["task"] == "joint")
image, audio = modality_tensors(row)
capstone = CapstoneModel().eval()
batch, labels = prepare_batch([row])
with torch.no_grad():
    logits = capstone(**batch)["logits"]
assert logits.shape[:2] == labels.shape
result["capstone_interface"] = {"row_id": row["id"], "task": row["task"], "image_shape": list(image.shape), "audio_summary_shape": list(audio.shape), "logit_shape": list(logits.shape), "label_shape": list(labels.shape), "experts": capstone.config.experts, "top_k": capstone.config.top_k, "scope": "Random-weight integrated interface, real generated pixel/tone input; no trained capability claim"}
history = [{"role": "user", "content": "接下來請用繁體中文回答。"}]
typed = {"user": "請讀出照片裡的招牌。", "history": history}
spoken = {"user": "請讀出照片裡的招牌。", "history": history}
typed_messages = messages_for(typed, ROOT)
spoken_messages = messages_for(spoken, ROOT)
assert typed_messages == spoken_messages
result["shared_chat_input"] = {"typed_messages": typed_messages, "spoken_messages": spoken_messages, "equal": typed_messages == spoken_messages, "scope": "Actual message assembler with manually equal transcript; no Qwen/Whisper download, transcription, generation or quality replication"}
print(json.dumps(result, ensure_ascii=False, indent=2))
