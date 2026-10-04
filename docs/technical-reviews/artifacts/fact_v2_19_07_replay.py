"""Independent CPU replay for lesson 19.7; no fitting or weight modification."""

import contextlib
import hashlib
import io
import json
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.check_technical_reviews import sections  # noqa: E402
from tiny_perceptron.capstone import (  # noqa: E402
    TOK,
    build_dataset,
    calculator_runtime,
    evaluate_rows,
    load_capstone,
    parse_action,
    prompt_ids,
)
from tiny_perceptron.capstone_ui import create_server  # noqa: E402

PREFIX = ROOT / "docs/technical-reviews/artifacts/fact_v2_19_07"
SCRATCH = ROOT / "outputs/integration-runs/fact_v2_19_07"
SCRATCH.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(2)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(suffix, value):
    path = Path(str(PREFIX) + suffix)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return str(path.relative_to(ROOT))


def trace_check(trace):
    if trace is None:
        return None
    ids = trace["generated_ids"]
    return {
        "decode_matches": TOK.decode(ids) == trace["raw"],
        "eos_matches": trace["eos"] == bool(ids and ids[-1] == TOK.eos_id),
        "no_interior_control_tokens": all(i >= 8 for i in ids[:-1]),
        "generated_positions_including_eos": len(ids),
    }


def audit(original, rows):
    indexed = {row["id"]: row for row in rows}
    assert len(indexed) == len(rows)
    assert len(original["records"]) == len(rows)
    assert {r["id"] for r in original["records"]} == set(indexed)
    examined = []
    for r in original["records"]:
        row = indexed[r["id"]]
        action = parse_action(r["action_trace"])
        ac = r["action_trace"]["eos"] and r["action_trace"]["raw"] == row["answer"]
        result = None
        answer = None
        if action["status"] in ("direct", "ask"):
            answer = action["content"]
        elif action["status"] == "tool":
            result = calculator_runtime(action, available=row["available"])
            if result["status"] == "ok":
                followup = dict(row, image=None, audio=None)
                followup["user"] = f"原題：{action['a']}+{action['b']}。計算器回報：{result['result']}。請回答。"
                assert r["final_trace"]["prompt_ids"] == prompt_ids(followup)
                final = parse_action(r["final_trace"])
                if final["status"] == "direct":
                    answer = final["content"]
        if row["task"] == "calculator":
            operands = [int(v) for v in re.findall(r"\d+", row["user"])]
            truth = str(sum(operands))
        else:
            truth = row["answer"].split(":", 1)[1]
        e2e = bool(ac and answer == truth)
        assert r["parsed_action"] == action
        assert r["runtime"] == result
        assert r["expected_action"] == row["answer"]
        assert r["expected_final"] == truth
        assert r["answer"] == answer
        assert r["action_correct"] == ac
        assert r["end_to_end_correct"] == e2e
        assert r["action_trace"]["prompt_ids"] == prompt_ids(row)
        checks = {"first": trace_check(r["action_trace"]), "final": trace_check(r["final_trace"])}
        assert all(all(v for k, v in c.items() if k != "generated_positions_including_eos") for c in checks.values() if c)
        examined.append({"id": r["id"], "task": row["task"], "user": row["user"], "available": row["available"], "expected": truth, "action": r["action_trace"]["raw"], "runtime": result, "final": None if r["final_trace"] is None else r["final_trace"]["raw"], "answer": answer, "action_correct": ac, "end_to_end_correct": e2e, "token_checks": checks})
    assert original["action_correct"] == sum(r["action_correct"] for r in examined)
    assert original["end_to_end_correct"] == sum(r["end_to_end_correct"] for r in examined)
    calc = [r for r in examined if r["task"] == "calculator"]
    off = [r for r in examined if r["task"] == "unavailable"]
    returns = [r for r in examined if r["task"] == "tool_return"]
    return {
        "all_record_count": len(examined),
        "all_action_correct": original["action_correct"],
        "all_end_to_end_correct": original["end_to_end_correct"],
        "calculator": {"count": len(calc), "action_correct": sum(r["action_correct"] for r in calc), "runtime_ok": sum(r["runtime"] is not None and r["runtime"]["status"] == "ok" for r in calc), "final_correct_with_eos": sum(r["end_to_end_correct"] for r in calc)},
        "unavailable": {"count": len(off), "ask_correct_no_execution": sum(r["action_correct"] and r["runtime"] is None and r["action"].startswith("ASK:") for r in off)},
        "independent_tool_return": {"count": len(returns), "correct": sum(r["end_to_end_correct"] for r in returns)},
        "records": examined,
    }


env = {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu", "platform": platform.platform(), "threads": str(torch.get_num_threads()), "cuda_available": str(torch.cuda.is_available())}
if Path("/proc/cpuinfo").exists():
    env["cpu"] = next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name"))
code_hashes = {str(p): sha(ROOT / p) for p in ["tiny_perceptron/capstone.py", "tiny_perceptron/capstone_ui.py", "tiny_perceptron/data.py", "tiny_perceptron/model.py", "scripts/capstone.py", "scripts/course_experiments/capstone.py", "scripts/course_experiments/capstone_deployment.py", "scripts/check_technical_reviews.py"]}
read_sections = {}
for p, ids in [("course/chapters/19.md", {"19.1", "19.7"}), ("course/chapters/0B.md", {"B.1", "B.2", "B.3", "B.4"})]:
    for lesson, body in sections(ROOT / p):
        if lesson in ids:
            read_sections[lesson] = {"source": p, "sha256": hashlib.sha256(body.encode()).hexdigest(), "body": body}
save("_read_sections.json", read_sections)

example = {"raw": "TOOL:calculator:1+2", "eos": True}
action = parse_action(example)
stream = io.StringIO()
with contextlib.redirect_stdout(stream):
    print("解析後的請求", action)
    for available in (True, False):
        print("計算器可用", available, "實際執行結果", calculator_runtime(action, available=available))
    print("還沒有模型讀回結果，也沒有最終模型答案")
faults = []
for raw, eos in [("TOOL:unknown:1+2", True), ("TOOL:calculator:-1+2", True), ("TOOL:calculator:1000+2", True), ("TOOL:calculator:1+2", False), ("DIRECT:", True), ("TOOL:calculator:999+999", True)]:
    parsed = parse_action({"raw": raw, "eos": eos})
    faults.append({"raw": raw, "eos": eos, "parsed": parsed, "runtime": calculator_runtime(parsed)})
for a, b in [(True, 2), (1.0, 2), (1, -1), (1, 1000)]:
    request = {"status": "tool", "name": "calculator", "a": a, "b": b}
    faults.append({"action": request, "runtime": calculator_runtime(request)})
compact_json = json.dumps({"name": "calculator", "arguments": {"a": 1, "b": 2}}, separators=(",", ":"))
save("_mechanism.json", {"environment": env, "command": ".venv/bin/python docs/technical-reviews/artifacts/fact_v2_19_07_replay.py", "example_stdout": stream.getvalue(), "faults": faults, "serialization": {"colon_text": example["raw"], "colon_bytes": len(example["raw"].encode()), "compact_json": compact_json, "json_bytes": len(compact_json.encode()), "same_sum_for_order": {"2+1": 2 + 1, "1+2": 1 + 2}, "ordered_action_exact_match": "TOOL:calculator:1+2" == "TOOL:calculator:2+1"}})

data_path = ROOT / "docs/course-experiments/capstone-evidence/deployment/data.json"
data = json.loads(data_path.read_text())
splits, manifest = build_dataset(42)
assert data["splits"] == splits
assert data["manifest"] == manifest
split_summary = {"manifest": manifest, "data_sha256": sha(data_path), "family_disjoint": all(not (set(manifest["families"][a]["numbers"]) & set(manifest["families"][b]["numbers"])) for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]), "reserved_family_test_only": "numbers:1:2" in manifest["families"]["test"]["numbers"] and all("numbers:1:2" not in manifest["families"][s]["numbers"] for s in ["train", "validation"])}

stage_results = {}
models = {}
for stage in ("pretrain", "sft", "joint", "dpo"):
    checkpoint = next((ROOT / "outputs/integration-runs" / f"v2-{stage}/capstone-review").glob("*/*/model.pt"))
    model, payload = load_capstone(checkpoint)
    models[stage] = model
    assert payload["metadata"]["data_manifest"] == manifest
    before = {k: v.clone() for k, v in model.state_dict().items()}
    report_path = ROOT / "docs/course-experiments/capstone-evidence" / stage / "train-report.json"
    report = json.loads(report_path.read_text())
    reference_path = ROOT / "docs/course-experiments/capstone-evidence" / stage / "validation.json"
    reference = json.loads(reference_path.read_text())
    original_audit = audit(reference, splits["validation"])
    started = time.perf_counter()
    replay = evaluate_rows(model, splits["validation"])
    seconds = time.perf_counter() - started
    replay_path = save(f"_{stage}_validation_cpu.json", replay)
    replay_audit = audit(replay, splits["validation"])
    mismatches = [a["id"] for a, b in zip(reference["records"], replay["records"], strict=True) if a != b]
    tensors = [{"name": name, "shape": list(t.shape), "dtype": str(t.dtype), "numel": t.numel(), "sha256": hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest(), "finite": bool(torch.isfinite(t).all())} for name, t in payload["model"].items()]
    stage_results[stage] = {"checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": sha(checkpoint), "checkpoint_bytes": checkpoint.stat().st_size, "checkpoint_metadata": {k: v for k, v in payload.items() if k != "model"}, "tensor_manifest": tensors, "total_tensor_parameters": sum(t["numel"] for t in tensors), "training_report": report, "training_report_sha256": sha(report_path), "reference_path": str(reference_path.relative_to(ROOT)), "reference_sha256": sha(reference_path), "original_audit": original_audit, "cpu_replay_path": replay_path, "cpu_replay_audit": replay_audit, "cpu_vs_reference_record_mismatches": mismatches, "cpu_replay_seconds": seconds, "timing_scope": "One full evaluate_rows call on all 84 validation rows, including modality feature preparation, first generations, calculator calls, and second generations; no fitting, no warmup or latency benchmark.", "weights_unchanged": all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())}
    print(stage, replay_audit["calculator"], "record_mismatches", mismatches, flush=True)

joint = models["joint"]
reference_path = ROOT / "docs/course-experiments/capstone-evidence/deployment/test-joint.json"
reference = json.loads(reference_path.read_text())
started = time.perf_counter()
replay = evaluate_rows(joint, splits["test"])
elapsed = time.perf_counter() - started
save("_joint_test_cpu.json", replay)
test_result = {"reference_path": str(reference_path.relative_to(ROOT)), "reference_sha256": sha(reference_path), "reference_checkpoint_sha256": reference["checkpoint_sha256"], "original_audit": audit(reference, splits["test"]), "cpu_replay_audit": audit(replay, splits["test"]), "record_mismatches": [a["id"] for a, b in zip(reference["records"], replay["records"], strict=True) if a != b], "cpu_replay_seconds": elapsed, "timing_scope": "One complete 90-row CPU evaluate_rows call, including actual tool calls and second generations; no warmup, no GPU timing inference."}
print("test", test_result["cpu_replay_audit"]["calculator"], "record_mismatches", test_result["record_mismatches"], flush=True)

public = ROOT / "outputs/integration-runs/release-main/public-manifest.json"
public_manifest = json.loads(public.read_text())
item = next(x for x in public_manifest["files"] if x["output"] == "joint/model.pt")
public_path = hf_hub_download(repo_id=public_manifest["repo"], filename=item["path"], revision=public_manifest["revision"], token=False, local_dir=SCRATCH / "public")
assert sha(public_path) == item["sha256"]
public_model, public_payload = load_capstone(public_path)
public_result = {"repo": public_manifest["repo"], "revision": public_manifest["revision"], "filename": item["path"], "token": False, "sha256": sha(public_path), "manifest_sha256": sha(public), "bytes": Path(public_path).stat().st_size, "all_tensors_equal_to_experiment_joint": all(torch.equal(v, public_model.state_dict()[k]) for k, v in joint.state_dict().items()), "metadata": {k: v for k, v in public_payload.items() if k != "model"}}
assert public_result["all_tensors_equal_to_experiment_joint"]

cli = []
for question, disabled in [("1+2等於多少？", False), ("1+2等於多少？", True), ("0+1等於多少？", False), ("請算1加0。", False)]:
    command = [str(ROOT / ".venv/bin/python"), "scripts/capstone.py", "infer", "--checkpoint", str(public_path), "--prompt", question, "--device", "cpu"]
    if disabled:
        command.append("--calculator-disabled")
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
    cli.append({"command": command, "exit_code": completed.returncode, "stdout": json.loads(completed.stdout), "stderr": completed.stderr})

server = create_server(public_model, port=0, stage="joint")
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
http = []
try:
    for question, available in [("1+2等於多少？", True), ("1+2等於多少？", False), ("請算1加0。", True)]:
        body = {"prompt": question, "calculator_available": available}
        req = urllib.request.Request(f"http://127.0.0.1:{server.server_port}/api/chat", data=json.dumps(body, ensure_ascii=False).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as response:
            http.append({"request": body, "http_status": response.status, "response": json.load(response)})
finally:
    server.shutdown()
    server.server_close()
    thread.join()

figure_path = ROOT / "course/figures/capstone_tool_loop.svg"
figure = next(r for r in json.loads((ROOT / "outputs/reading-time/figures/manifest.json").read_text())["records"] if r["source"] == "course/figures/capstone_tool_loop.svg")
assert sha(figure_path) == figure["source_sha256"]
assert sha(ROOT / figure["render"]) == figure["render_sha256"]
render_path = Path(str(PREFIX) + "_figure.png")
shutil.copyfile(ROOT / figure["render"], render_path)
figure["persistent_render"] = str(render_path.relative_to(ROOT))
figure["inspection"] = "Reviewer read complete XML and personally viewed the 1600-pixel render: downward request → validation/execution → same-model final generation; exactly one calculation between two generations, clear artificial-example label; no clipped text, contradictory number, or success-rate scale. 1+2→3 agrees with integer runtime; no implication all real generations succeed."

save("_audit.json", {"command": ".venv/bin/python docs/technical-reviews/artifacts/fact_v2_19_07_replay.py", "environment": env, "code_hashes": code_hashes, "section_sha256": read_sections["19.7"]["sha256"], "split_check": split_summary, "stages": stage_results, "joint_test": test_result, "public_checkpoint": public_result, "cli": cli, "http": http, "figure": figure, "result": "All stored records audited against questions, ordered payloads, runtime arithmetic, token IDs/EOS, and exact end-to-end definition; real CPU replays and public checkpoint comparisons completed."})
print("audit saved", flush=True)
