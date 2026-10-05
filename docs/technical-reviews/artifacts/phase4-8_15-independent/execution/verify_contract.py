"""Bounded CPU audit of the unchanged section's targets; no model inference or training."""
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.data import ByteTokenizer, render_chat, pad_batch

ART = Path(__file__).resolve().parents[1]
raw = (ART / "inputs/extracted/section.md").read_bytes()
assert hashlib.sha256(raw).hexdigest() == "32f9d4b1c79dff9a4928565d368b0a4273c631d2e16a3fc05a973b186762da65"
text = raw.decode("utf-8")
rows = tuple((int(n), language, phrase) for n, language, phrase in
             re.findall(r"^\| ([1-4]) \| (中文|英文) \| ([^|]+?) \|$", text, re.M))
assert tuple(n for n, _, _ in rows) == (1, 2, 3, 4)
material = "\n".join(phrase for _, _, phrase in rows)
material_hash = hashlib.sha256(material.encode()).hexdigest()
selected = {"全部": [phrase for _, _, phrase in rows],
            "中文": [phrase for _, lang, phrase in rows if lang == "中文"],
            "英文": [phrase for _, lang, phrase in rows if lang == "英文"]}
assert selected["中文"] == ["今日休館", "明日開放"]
assert selected["英文"] == ["Closed today", "Open tomorrow"]

def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result

def reject_constant(value):
    raise ValueError("nonstandard JSON constant: " + value)

def json_check(reply, language):
    try:
        obj = json.loads(reply, object_pairs_hook=unique_object, parse_constant=reject_constant)
        schema = (isinstance(obj, dict) and set(obj) == {"text"}
                  and isinstance(obj["text"], list)
                  and all(isinstance(item, str) for item in obj["text"]))
        return {"schema": schema, "scope_order_content": schema and obj["text"] == selected[language]}
    except (ValueError, TypeError):
        return {"schema": False, "scope_order_content": False}

fence = re.search(rb"```json\r?\n(.*?)```", raw, re.S)[1]
assert fence == (ART / "inputs/fence-1.json").read_bytes()
assert json_check(fence.decode(), "中文") == {"schema": True, "scope_order_content": True}

cases = []
for language in ("全部", "中文", "英文"):
    plain = "\n".join(selected[language])
    encoded = json.dumps({"text": selected[language]}, ensure_ascii=False, separators=(",", ":"))
    assert plain.splitlines() == selected[language]
    assert json_check(encoded, language) == {"schema": True, "scope_order_content": True}
    assert hashlib.sha256(material.encode()).hexdigest() == material_hash
    cases.append({"selection": language, "material_sha256": material_hash,
                  "selected_rows": [n for n, lang, _ in rows if language == "全部" or lang == language],
                  "line_count": len(plain.splitlines()), "plain_target": plain, "json_target": encoded})

wrong = [
    ("plain Chinese under JSON request", "\n".join(selected["中文"]), "中文", False, False),
    ("JSON all four rows under Chinese request", json.dumps({"text": selected["全部"]}, ensure_ascii=False), "中文", True, False),
    ("unchanged original Chinese target under English request", fence.decode(), "英文", True, False),
    ("extra field", '{"text":["今日休館","明日開放"],"note":"完成"}', "中文", False, False),
    ("string instead of list", '{"text":"今日休館\\n明日開放"}', "中文", False, False),
    ("reversed order", '{"text":["明日開放","今日休館"]}', "中文", True, False),
    ("omitted required line", '{"text":["今日休館"]}', "中文", True, False),
    ("prose outside JSON", '答案是：' + fence.decode(), "中文", False, False),
    ("Markdown fence around JSON", '```json\n' + fence.decode() + '```', "中文", False, False),
    ("duplicate text field", '{"text":[],"text":["今日休館","明日開放"]}', "中文", False, False),
    ("nonstandard NaN", '{"text":[NaN]}', "中文", False, False),
]
counterexamples = []
for name, reply, language, schema, target in wrong:
    observed = json_check(reply, language)
    assert observed == {"schema": schema, "scope_order_content": target}
    counterexamples.append({"name": name, "request_selection": language, "reply": reply, **observed})

torch.set_default_device("cpu")
tok = ByteTokenizer()
chat = []
requests = [
    ("全部照抄，一行一項，不加列號或說明", "全部", "plain"),
    ("只抄中文，一行一項，不加列號或說明", "中文", "plain"),
    ("只抄中文，用只有text欄位的JSON物件，欄位值是依原順序排列的文字清單；不要在JSON外加字", "中文", "json"),
    ("只抄英文，一行一項", "英文", "plain"),
    ("只抄英文，用只有text欄位的JSON物件，欄位值是依原順序排列的文字清單；不要在JSON外加字", "英文", "json"),
    ("英文不要，兩條中文照原樣給我", "中文", "plain"),
]
for request, language, form in requests:
    answer = ("\n".join(selected[language]) if form == "plain" else
              json.dumps({"text": selected[language]}, ensure_ascii=False, separators=(",", ":")))
    prompt = material + "\n要求：" + request
    x, y = render_chat([{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}])
    first = (y != -100).nonzero()[0].item()
    expected_ids = tok.encode(answer) + [tok.eos_id]
    assert x.device.type == y.device.type == "cpu"
    assert x[first].item() == tok.assistant_id
    assert y[first].item() == tok.encode(answer)[0]
    assert y[y != -100].tolist() == expected_ids
    assert y[-1].item() == tok.eos_id
    assert int((y != -100).sum()) == len(answer.encode("utf-8")) + 1
    assert tok.decode(x[2:first - 1].tolist()) == prompt
    # The actual truncation helper can retain answer bytes while dropping EOS.
    _, truncated_y, _ = pad_batch([(x, y)], max_length=len(x) - 1)
    assert (truncated_y != -100).any() and truncated_y[0, -1].item() != tok.eos_id
    chat.append({"request": request, "selection": language, "format": form,
                 "material_sha256": material_hash, "answer": answer,
                 "first_answer_prediction_position": first,
                 "first_input_role_id": x[first].item(), "first_answer_id": y[first].item(),
                 "answer_utf8_bytes": len(answer.encode("utf-8")), "effective_targets_including_eos": len(expected_ids),
                 "eos_target_id": y[-1].item(), "truncated_still_has_targets": True,
                 "truncated_eos_present": False})

result = {"scope": "Exact original JSON fence plus manual target-contract, selection/format counterexamples and CPU render_chat/pad_batch checks. No training, model inference, gradients, parameter updates, performance metric, or novel model capability was measured.",
          "source_sha256": hashlib.sha256(raw).hexdigest(),
          "original_fence_sha256": hashlib.sha256(fence).hexdigest(), "material_sha256": material_hash,
          "material_rows": rows, "selection_cases": cases, "counterexamples": counterexamples,
          "chat_records": chat,
          "evaluation_design": {"same_material_paraphrase": "same notice family; changed request wording only", "new_material": "must use a notice family absent from training to claim unseen-material evaluation", "executed_model_evaluation": False},
          "all_assertions_passed": True}
(ART / "execution/contract-results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
environment = {"python": sys.version, "executable": sys.executable, "torch": str(torch.__version__),
               "torch_git_version": str(torch.version.git_version), "cuda_build": str(torch.version.cuda),
               "cuda_available": str(torch.cuda.is_available()), "device": "cpu",
               "cwd": str(Path.cwd()), "data_py_sha256": hashlib.sha256((ROOT / "tiny_perceptron/data.py").read_bytes()).hexdigest()}
(ART / "execution/environment.json").write_text(json.dumps(environment, indent=2) + "\n")
print(json.dumps({"original_JSON": json_check(fence.decode(), "中文"),
                  "selection_line_counts": {v["selection"]: v["line_count"] for v in cases},
                  "counterexamples_checked": len(counterexamples), "chat_records_checked": len(chat),
                  "all_assertions_passed": True, "scope": result["scope"]}, ensure_ascii=False))
