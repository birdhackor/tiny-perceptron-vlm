"""獨立核對截斷、遮罩、資料指紋與已保存實驗；不訓練、不解包外部全文。"""

import ast
import hashlib
import inspect
import io
import json
import math
import platform
import re
import subprocess
import sys
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch
from scripts.course_experiments import common, run, text
from tiny_perceptron.data import IGNORE, ByteTokenizer, pad_batch, render_chat
from tiny_perceptron.model import loss_sum, masked_loss


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def capture_error(call):
    try:
        call()
    except ValueError as error:
        return {"type": type(error).__name__, "message": str(error)}
    raise AssertionError("預期應拒絕沒有有效目標的資料")


def command(argv):
    proc = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
    return {"argv": argv, "exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}


torch.set_num_threads(1)
torch.manual_seed(42)
tok = ByteTokenizer()
messages = [{"role": "user", "content": "很長的問題"}, {"role": "assistant", "content": "是"}]
example = render_chat(messages)
x, y = example
original_ids = [tok.bos_id, tok.user_id] + tok.encode("很長的問題") + [tok.eos_id, tok.assistant_id]
original_ids += tok.encode("是") + [tok.eos_id]
check(x.tolist() == original_ids[:-1], "輸入必須只移除原序列末項")
check(y.nonzero().numel() > 0, "範例需含標籤")
valid_indices = (y != IGNORE).nonzero().flatten().tolist()
check(valid_indices == [18, 19, 20, 21], "有效目標索引不符")
check(x[18].item() == tok.assistant_id, "第一個答案目標應由助手標記位置預測")
truncations = {}
for length in (2, 18, 19, 20, 21, 22):
    if length < 19:
        truncations[str(length)] = capture_error(lambda: pad_batch([example], max_length=length))
    else:
        bx, by, bv = pad_batch([example], max_length=length)
        check(int((by != IGNORE).sum()) == length - 18, "截斷後目標數不符")
        truncations[str(length)] = {
            "shape": list(bx.shape), "effective_targets": int((by != IGNORE).sum()),
            "x_last": int(bx[0, -1]), "y_last": int(by[0, -1]), "valid_count": int(bv.sum()),
        }

freeze = ROOT / "docs/course-review-20261019/freeze-01/sources/7.10.md"
body = freeze.read_text()
fences = re.findall(r"```python\n(.*?)```", body, re.S)
frozen_function = ast.parse(fences[1]).body[0]
actual_function = ast.parse(inspect.getsource(pad_batch)).body[0]
check(ast.dump(frozen_function) == ast.dump(actual_function), "頁面摘錄與專案函式不一致")
short = render_chat([{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}])
bx, by, bv = pad_batch([example, short])
check(list(bx.shape) == [2, 22] and bx.dtype == by.dtype == torch.long and bv.dtype == torch.bool,
      "batch 維度或 dtype 不符")
check(torch.equal(bx[1, :len(short[0])], short[0]), "短例必須在左側")
check(bool((bx[1, len(short[0]):] == 0).all()), "輸入右側應填 PAD")
check(bool((by[1, len(short[1]):] == IGNORE).all()), "標籤右側應忽略")
check(not bool(bv[1, len(short[0]):].any()), "補齊區 attention 標記應為 False")
check(int(bv[0].sum()) == 22 and int((by[0] != IGNORE).sum()) == 4,
      "attention 有效位置與 loss 有效位置必須分開")
invalid = (short[0], torch.full_like(short[1], IGNORE))
mixed_error = capture_error(lambda: pad_batch([short, invalid]))
check("第 1 筆" in mixed_error["message"], "混合 batch 仍需指出空答案那列")
empty_error = capture_error(lambda: pad_batch([]))

logits = torch.zeros((1, len(x), tok.vocab_size), dtype=torch.float64, requires_grad=True)
total, count = loss_sum(logits, y[None])
mean = masked_loss(logits, y[None])
mean.backward()
expected = math.log(tok.vocab_size)
check(abs(float(mean.detach()) - expected) < 1e-12, "均勻 logits 的平均代價應為 log 詞表大小")
check(int(count) == 4, "平均分母應為有效目標數")
check(bool((logits.grad[0, y == IGNORE] == 0).all()), "忽略目標不應產生 logits 梯度")
no_labels = torch.full_like(y[None], IGNORE)
loss_error = capture_error(lambda: masked_loss(logits.detach(), no_labels))
flow = []
try:
    pad_batch([example], max_length=2)
except ValueError as error:
    flow.append({"branch": "except", "text": str(error)})
else:
    raise AssertionError("預期錯誤時不可進入 else")
try:
    pad_batch([example], max_length=19)
except ValueError:
    raise AssertionError("保留有效目標時不應進入 except")
else:
    flow.append({"branch": "else", "formatted_row": f"第 {1} 筆"})

manifest_path = ROOT / "assets/training/manifest.json"
manifest = json.loads(manifest_path.read_bytes())
asset = next(a for a in manifest["assets"] if a["id"] == "ultrachat-sft")
archive = ROOT / asset["archive"]
local_archive_bytes = archive.read_bytes()
archive_bytes = local_archive_bytes
archive_retrieval = {"local_bytes": len(local_archive_bytes), "local_sha256": sha(local_archive_bytes),
                     "local_is_lfs_pointer": local_archive_bytes.startswith(b"version https://git-lfs.github.com/spec/v1")}
if archive_retrieval["local_is_lfs_pointer"]:
    url = "https://media.githubusercontent.com/media/birdhackor/tiny-perceptron-vlm/main/" + asset["archive"]
    argv = ["curl", "--fail", "--location", "--silent", "--show-error", url]
    proc = subprocess.run(argv, capture_output=True, check=True)
    archive_bytes = proc.stdout
    archive_retrieval.update(url=url, command_argv=argv, exit_code=proc.returncode,
                             accessed_on=datetime.now(UTC).date().isoformat(), full_archive_saved=False)
check(sha(archive_bytes) == asset["archive_sha256"], "資料包 SHA 不符")
file_receipts = []
with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as package:
    for entry in asset["files"]:
        raw = package.extractfile(entry["path"]).read()
        check(sha(raw) == entry["sha256"] and len(raw) == entry["bytes"], "包內檔案指紋不符")
        file_receipts.append({"path": entry["path"], "sha256": sha(raw), "bytes": len(raw)})
        if entry["path"].endswith("/train-first-100.jsonl"):
            raw_rows = [json.loads(line) for line in raw.decode().splitlines()]

captured = {}
def record_parts(ctx, parts, name):
    captured["parts"] = parts
    return {split: {"records": len(rows), "families": len({r["family"] for r in rows}),
                    "sha256": sha("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode())}
            for split, rows in parts.items()}

def record_model(ctx, **kwargs):
    captured["model_config"] = kwargs
    return object()

def record_training(model, records, ctx, **kwargs):
    captured["training_config"] = kwargs
    return {"steps": kwargs["steps"]}

ctx = SimpleNamespace(seed=42, step_scale=1.0)
with patch.object(text, "_asset_rows", return_value=raw_rows), \
     patch.object(text, "_save_splits", side_effect=record_parts), \
     patch.object(text, "new_lm", side_effect=record_model), \
     patch.object(text, "fit_lm", side_effect=record_training), \
     patch.object(text, "_evaluations", return_value={}):
    prepared = text._natural_chat_pilot(ctx)

result_path = ROOT / "docs/course-experiments/results/sft.json"
recorded = json.loads(result_path.read_bytes())
pilot = recorded["results"]["ultrachat_pilot"]
parts = captured["parts"]
check(prepared["source_records"] == prepared["prepared_records"] == 100, "來源或轉換筆數不符")
split_receipts = {}
families = {s: {r["family"] for r in rows} for s, rows in parts.items()}
for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
    check(not families[left] & families[right], "家族不得跨切分")
clipped = {"user": 0, "assistant": 0}
for split, rows in parts.items():
    check(prepared["data"][split]["sha256"] == pilot["data"][split]["sha256"], "重建切分指紋不符")
    examples = common.text_examples(rows, mode="sft", max_length=256)
    for row in rows:
        source = raw_rows[row["source_row"]]
        check([m["role"] for m in source["messages"][:2]] == ["user", "assistant"], "第一輪角色不符")
        for output, original in zip(row["messages"], source["messages"][:2], strict=True):
            check(output["content"] == original["content"].encode()[:120].decode(errors="ignore"),
                  "片段需保留完整 UTF-8 字元")
            clipped[output["role"]] += output["content"] != original["content"]
    split_receipts[split] = {
        **prepared["data"][split], "matches_recorded_sha256": True,
        "x_length_range": [min(len(xx) for xx, yy in examples), max(len(xx) for xx, yy in examples)],
        "effective_targets_range": [min(int((yy != IGNORE).sum()) for xx, yy in examples),
                                    max(int((yy != IGNORE).sum()) for xx, yy in examples)],
        "total_effective_targets": sum(int((yy != IGNORE).sum()) for xx, yy in examples),
        "all_final_targets_eos": all(int(yy[-1]) == tok.eos_id for xx, yy in examples),
    }

evaluation = {}
for split, saved in pilot["evaluation"].items():
    matches = 0
    for row, sample in zip(parts[split], saved["samples"], strict=True):
        check(sample["messages"] == row["messages"][:-1], "評估問題必須來自相同資料切分")
        check(sample["expected"] == row["messages"][-1]["content"], "評估答案必須來自相同資料切分")
        generated = sample["generated_ids"]
        raw = generated[:generated.index(tok.eos_id)] if tok.eos_id in generated else generated
        exact = raw == tok.encode(sample["expected"])
        check(exact == sample["exact"], "原始 ID 比對與保存的 exact 不一致")
        check(tok.decode(raw) == sample["generated"], "保存文字與原始 ID 解碼不符")
        matches += int(exact)
    check(matches == saved["matches"] == 0 and len(saved["samples"]) == saved["records"] == 10,
          "評估匹配數或題數不符")
    evaluation[split] = {
        "records": saved["records"], "matches_recomputed": matches,
        "effective_tokens": saved["effective_tokens"], "eos_rate": saved["eos_rate"],
        "first_generated": saved["samples"][0]["generated"],
        "generation_limit": 128, "comparison": "首個 EOS 前的原始 ID 與 ByteTokenizer.encode(expected) 逐項相同",
    }

code_files = ["tiny_perceptron/data.py", "tiny_perceptron/model.py", "scripts/course_experiments/text.py",
              "scripts/course_experiments/common.py", "scripts/course_experiments/run.py"]
code_receipts = {name: {"sha256": sha((ROOT / name).read_bytes()),
                        "recorded_sha256": recorded["code_sha256"][name],
                        "matches_recorded": sha((ROOT / name).read_bytes()) == recorded["code_sha256"][name]}
                 for name in code_files}
history_lookup = command(["git", "show", recorded["revision"] + ":scripts/course_experiments/run.py"])
listed = command(["git", "rev-list", "--objects", "--all", "--", "scripts/course_experiments/run.py"])
historical = None
for line in listed["stdout"].splitlines():
    fields = line.split(" ", 1)
    if len(fields) == 2 and fields[1] == "scripts/course_experiments/run.py":
        raw = subprocess.run(["git", "cat-file", "blob", fields[0]], cwd=ROOT, capture_output=True, check=True).stdout
        if sha(raw) == recorded["code_sha256"][fields[1]]:
            tree = ast.parse(raw)
            function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "execute")
            function_text = ast.get_source_segment(raw.decode(), function)
            historical = {"git_blob": fields[0], "sha256": sha(raw), "function": function_text,
                          "inspection": "計時從 experiment(context) 前開始，到 CUDA synchronize 後結束；包含 run_sft 所有支線。"}
            break
check(historical is not None, "必須找到與實驗紀錄 SHA 相同的 runner 原始碼")

cli_commands = [command([str(ROOT / ".venv/bin/python"), "scripts/fetch_training_assets.py", "--list"]),
                command([str(ROOT / ".venv/bin/python"), "-m", "scripts.course_experiments.run", "--list-assets", "sft"])]
check(all(r["exit_code"] == 0 for r in cli_commands), "唯讀 CLI 核對失敗")
calls = []
with patch.object(sys, "argv", ["scripts.course_experiments.run", "--experiment", "sft", "--device", "cuda"]), \
     patch.object(run, "execute", side_effect=lambda *a, **kw: calls.append((a, kw)) or
                  {"experiment_id": "sft", "elapsed_seconds": 0, "evidence_status": "未執行訓練", "artifacts": []}), \
     patch("builtins.print"):
    run.main()
check(calls[0][0][0:2] == ("sft", "cuda"), "命令列應傳遞實驗名稱與裝置")
check(calls[0][0][2] == ROOT / "outputs/course-experiments/course-v1/sft", "預設輸出位置不符")

report = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-7-10/probe.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                    "torch_git_version": torch.version.git_version, "device": "cpu", "platform": platform.platform()},
    "source_sha256": sha(freeze.read_bytes()),
    "example": {"question_utf8_bytes": list("很長的問題".encode()), "answer_utf8_bytes": list("是".encode()),
                "original_length": len(original_ids), "xy_length": len(x), "x": x.tolist(), "y": y.tolist(),
                "valid_target_indices": valid_indices, "assistant_index": 18, "truncations": truncations},
    "batch": {"excerpt_ast_equal": True, "shape": list(bx.shape), "short_length": len(short[0]),
              "input_dtype": str(bx.dtype), "target_dtype": str(by.dtype), "valid_dtype": str(bv.dtype),
              "mixed_empty_error": mixed_error, "empty_batch_error": empty_error, "try_flow": flow},
    "loss": {"effective_targets": int(count), "sum": float(total.detach()), "mean": float(mean.detach()),
             "expected_mean": expected, "tolerance": 1e-12, "ignored_gradient_zero": True, "empty_error": loss_error},
    "asset": {"archive_path": asset["archive"], "archive_sha256": sha(archive_bytes),
              "retrieval": archive_retrieval,
              "file_receipts": file_receipts, "full_source_saved": False},
    "ultrachat": {"source_records": len(raw_rows), "prepared_records": prepared["prepared_records"],
                  "clipped_records": clipped, "splits": split_receipts,
                  "model_config": captured["model_config"], "training_config": captured["training_config"],
                  "recorded_training_steps": pilot["training"]["steps"], "seed": recorded["seed"],
                  "evaluation_recomputed_from_saved_ids": evaluation,
                  "byte_boundary_example": text._utf8_prefix("很長的問題", 4),
                  "scope": "重建資料並重算保存的生成 ID；沒有重新產生模型輸出，沒有重新訓練。"},
    "experiment": {"path": result_path.relative_to(ROOT).as_posix(), "sha256": sha(result_path.read_bytes()),
                   "device": recorded["device"], "gpu": recorded["gpu"], "torch": recorded["torch_version"],
                   "python": recorded["python_version"], "elapsed_seconds": recorded["elapsed_seconds"],
                   "rounded_seconds": f"{recorded['elapsed_seconds']:.1f}", "timing_scope": recorded["timing_scope"],
                   "scope": "耗時是整個 sft 實驗，包含屬性問答與先練續寫再 SFT 等支線。"},
    "code_receipts": code_receipts, "historical_revision_lookup": history_lookup,
    "historical_runner": historical, "cli_readonly_checks": cli_commands,
    "cli_arguments_with_training_replaced": {"experiment": calls[0][0][0], "device": calls[0][0][1],
                                              "output": str(calls[0][0][2]), "trained": False},
    "checks_passed": True,
}
print(json.dumps(report, ensure_ascii=False, indent=2))
