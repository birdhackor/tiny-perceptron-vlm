"""Independent bounded CPU checks for 8.2; no training or saved weights."""
from pathlib import Path
import ast
import contextlib
import copy
import hashlib
import io
import json
import platform
import random
import re
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
ART = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.model import generate
from tiny_perceptron.tokenization import generation_report

torch.set_num_threads(1)
assert torch.version.cuda is None
assert not torch.cuda.is_available()

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

result = {"environment": {"python": platform.python_version(), "executable": sys.executable,
          "torch": str(torch.__version__), "torch_git_version": str(torch.version.git_version),
          "device": "cpu", "cuda_build": str(torch.version.cuda), "cwd": str(Path.cwd()),
          "threads": str(torch.get_num_threads())}, "training_performed": False,
          "weights_saved": False}

# Execute exact original fence, then the stated one-requirement exercise.
code = (ART / "original-fence/fence-1.py").read_bytes()
assert sha(code) == "5cb674c6501dd04ea47315f10b81500a293e182845c1716d5d8540643cc573a8"
tree = ast.parse(code)
assert {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)} == {"print"}
def run(raw):
    ns = {"__name__": "__main__"}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(raw, "8.2-original-or-stated-exercise", "exec"), ns)
    return output.getvalue(), ns
original, ns = run(code)
expected = "用一句話回答：2+2=?\n用貼切比喻回答：2+2=?\n兩次都使用同一份權重與同一個選字方式\n"
assert original == expected
variant = code.replace("用貼切比喻回答".encode(), "只回一個數字".encode())
(ART / "exercise-variant.py").write_bytes(variant)
observed_variant, variant_ns = run(variant)
assert observed_variant.splitlines() == [expected.splitlines()[0], "只回一個數字：2+2=?", expected.splitlines()[2]]
assert ns["question"] == variant_ns["question"] == "2+2=?"
result["original_and_exercise"] = {"original_stdout": original, "exercise_stdout": observed_variant,
    "original_bytes_sha256": sha(code), "exercise_bytes_sha256": sha(variant),
    "changes": "only requests[1]; question and requests[0] unchanged", "model_calls": 0,
    "scope": "string assembly and print only; final printed statement does not demonstrate a model comparison"}

# Directly exercise the repository's genuine generation helper with fixed logits.
# This constructed table fixture is not a trained language model and cannot
# establish natural-language style following or arithmetic ability.
class FixedConditionalScores(torch.nn.Module):
    def __init__(self, max_length=12):
        super().__init__()
        self.config = SimpleNamespace(max_length=max_length)
        table = torch.full((2, 9), -100.0)
        table[0, 7], table[0, 8] = 3.0, 2.0
        table[1, 7], table[1, 8] = 2.0, 3.0
        self.scores = torch.nn.Parameter(table)
        self.observed = []
    def forward(self, ids, cache=None):
        assert cache is None
        self.observed.append(ids.clone())
        row = (ids == 6).any(dim=1).long()
        logits = self.scores[row].unsqueeze(1).expand(-1, ids.shape[1], -1).clone()
        completed = (ids == 7).any(dim=1) | (ids == 8).any(dim=1)
        logits[completed] = -100.0
        logits[completed, :, 2] = 100.0
        return {"logits": logits, "cache": None}

model = FixedConditionalScores()
weight_before = model.scores.detach().clone()
left = torch.tensor([[1, 4, 5]])
right = torch.tensor([[1, 6, 5]])
left_copy, right_copy = left.clone(), right.clone()
greedy_left = generate(model, left, max_new_tokens=4)
greedy_right = generate(model, right, max_new_tokens=4)
assert greedy_left.tolist() == [[1, 4, 5, 7, 2]]
assert greedy_right.tolist() == [[1, 6, 5, 8, 2]]
assert any(x.tolist() == [[1, 4, 5, 7]] for x in model.observed)
assert model.training
assert model.scores.grad is None
for _ in range(3):
    assert torch.equal(generate(model, left, 4), greedy_left)
assert torch.equal(generate(model, left, 4, temperature=-1), greedy_left)
assert torch.equal(generate(model, left, 0), left)
assert generate(model, torch.tensor([[1, 4, 5, 7]]), 4).tolist() == [[1, 4, 5, 7, 2]]
model.config.max_length = left.shape[1]
assert torch.equal(generate(model, left, 4), left)
model.config.max_length = 4
assert generate(model, left, 4).tolist() == [[1, 4, 5, 7]]
model.config.max_length = 12
sampled = []
for seed in range(32):
    torch.manual_seed(seed)
    sampled.append(generate(model, left, 1, temperature=1.0)[0, -1].item())
assert set(sampled) == {7, 8}
assert torch.equal(model.scores.detach(), weight_before)
assert torch.equal(left, left_copy) and torch.equal(right, right_copy)
result["generation_contract"] = {"fixture": "constructed 2x9 logits table, not neural training evidence",
    "axes": "logits [batch, sequence position, vocabulary]; helper selects logits[:, -1] and argmax over vocabulary",
    "same_question_marker": 5, "only_prompt_marker_change": [4, 6], "greedy_left": greedy_left.tolist(),
    "greedy_right": greedy_right.tolist(), "same_prompt_greedy_repetitions_equal": True,
    "sampling_draws": 32, "sampling_counts": {str(x): sampled.count(x) for x in sorted(set(sampled))},
    "sampling_source_of_variation": "same prompt, same temperature and same parameters; random seed changed",
    "weights_unchanged_exact": True, "input_unchanged_exact": True, "grad_is_none": True,
    "boundary_checks": ["max_new_tokens=0", "context already at maximum", "one remaining context slot",
                         "EOS stops singleton sequence", "negative temperature takes greedy branch"],
    "tolerance": "exact integer sequences and torch.equal on parameter/input tensors"}

# Recompute saved observations without calling or loading its trained model.
saved = json.loads((ART / "inputs/current/docs/course-experiments/results/style.json").read_text())
revision = saved["revision"]
historical_hashes = {}
for name in ["scripts/course_experiments/behavior.py", "scripts/course_experiments/common.py",
             "scripts/course_experiments/text.py", "tiny_perceptron/model.py",
             "tiny_perceptron/data.py", "tiny_perceptron/tokenization.py"]:
    raw = (ART / "inputs/historical" / name).read_bytes()
    digest = sha(raw)
    assert digest == saved["code_sha256"][name]
    assert raw == subprocess.check_output(["git", "show", f"{revision}:{name}"], cwd=ROOT)
    historical_hashes[name] = digest

# Rebuild only pure historical data functions; never call run_style/fit_lm.
def selected_functions(relative, names, namespace):
    tree = ast.parse((ART / "inputs/historical" / relative).read_bytes())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), relative+":pure-functions-only", "exec"), namespace)
    return {n.name:n for n in nodes}
pure = {"json": json, "random": random, "hashlib": hashlib}
selected_functions("scripts/course_experiments/text.py", ["arithmetic_records"], pure)
selected_functions("scripts/course_experiments/common.py", ["split_records"], pure)
selected_functions("scripts/course_experiments/behavior.py", ["_conversation", "_style_record"], pure)
arithmetic = pure["split_records"](pure["arithmetic_records"](), seed=saved["seed"])
conditional = {split:[pure["_style_record"](row, style, True) for row in rows
                     for style in ("concise", "vivid", "json")] for split,rows in arithmetic.items()}
dates = []
for day in range(1,25):
    date = f"2026-10-{day:02d}"
    dates.append(pure["_conversation"](f"task=date;date={date};confirm", "已確認"+date+"。", "date-"+date, style="clarification"))
    dates.append(pure["_conversation"](f"task=date;id={day};date=?;confirm", "請提供日期。", "date-"+date, style="clarification"))
date_parts = pure["split_records"](dates,seed=saved["seed"])
conditional = {split:rows+date_parts[split] for split,rows in conditional.items()}
rebuilt_hashes = {}
for split,rows in conditional.items():
    data_bytes = "".join(json.dumps(row,ensure_ascii=False)+"\n" for row in rows).encode()
    recorded = saved["results"]["data"][split]
    assert sha(data_bytes) == recorded["sha256"]
    assert len(rows) == recorded["records"]
    rebuilt_hashes[split] = {"records":len(rows), "jsonl_sha256":sha(data_bytes)}
assert sha(json.dumps(conditional["train"],sort_keys=True,ensure_ascii=False).encode()) == saved["results"]["training"]["records_sha256"]
for split,rows in conditional.items():
    family_styles = {}
    for row in rows:
        if "a" in row: family_styles.setdefault((row["a"],row["b"]),set()).add(row["style"])
    assert all(styles == {"concise","vivid","json"} for styles in family_styles.values())
current_gen = next(n for n in ast.parse((ROOT/"tiny_perceptron/model.py").read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name=="generate")
historical_gen = next(n for n in ast.parse((ART/"inputs/historical/tiny_perceptron/model.py").read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name=="generate")
assert ast.dump(current_gen,include_attributes=False) == ast.dump(historical_gen,include_attributes=False)
tok = ByteTokenizer()
probes = saved["results"]["prompt_only_comparison_same_weights"]
assert set(probes) == {"concise", "vivid", "json"}
all_questions = []
recomputed = {}
raw_observations = []
for style, evaluation in probes.items():
    rows = evaluation["samples"]
    assert len(rows) == evaluation["records"] == 7
    questions, content, correct_style, exact, eos_count, token_count = [], 0, 0, 0, 0, 0
    for sample in rows:
        prompt = sample["messages"][0]["content"]
        m = re.fullmatch(r"style=(concise|vivid|json); (\d+)\+(\d+)=\?", prompt)
        assert m and m[1] == style
        a, b = int(m[2]), int(m[3]); question = f"{a}+{b}=?"
        questions.append(question)
        ids = sample["generated_ids"]
        assert ids and ids[-1] == tok.eos_id and ids.count(tok.eos_id) == 1
        assert all(8 <= token < tok.vocab_size for token in ids[:-1])
        visible = tok.decode(ids[:-1]); assert visible == sample["generated"]
        assert generation_report(tok, ids)["valid_answer_tokens"]
        is_exact = ids[:-1] == tok.encode(sample["expected"])
        assert is_exact == sample["exact"]
        truth = a+b
        if style == "json":
            value = json.loads(visible)
            style_ok = isinstance(value, dict) and set(value) == {"answer"} and type(value["answer"]) is int
            value_ok = style_ok and value["answer"] == truth
        elif style == "vivid":
            style_ok = "像把兩組積木合在一起再數" in visible
            value_ok = visible.split("，",1)[0].strip() == str(truth)
        else:
            style_ok = visible.strip().isdigit()
            value_ok = visible.strip() == str(truth)
        expected_target = str(truth)
        if style == "vivid": expected_target += "，像把兩組積木合在一起再數。"
        if style == "json": expected_target = json.dumps({"answer": truth})
        assert sample["expected"] == expected_target
        correct_style += int(style_ok); content += int(value_ok); exact += int(is_exact)
        eos_count += int(sample["eos"]); token_count += len(ids)
        raw_observations.append({"style": style, "prompt": prompt, "expected": expected_target,
            "raw_generated_ids": ids, "generated": visible, "style_correct": style_ok,
            "content_correct": value_ok, "exact": is_exact, "eos": sample["eos"]})
    assert exact == evaluation["matches"] == 0
    assert evaluation["exact_match"] == exact/7 and evaluation["eos_rate"] == eos_count/7 == 1
    assert correct_style == 7 and content == 0
    rubric = saved["results"]["after"]["test"]["rubric"][style]
    assert rubric["records"] == 7 and rubric["style_correct"] == correct_style and rubric["content_correct"] == content
    all_questions.append(questions)
    recomputed[style] = {"records": 7, "style_correct": correct_style, "content_correct": content,
        "raw_exact": exact, "eos": eos_count, "generated_byte_token_ids_including_eos": token_count,
        "first_prompt": rows[0]["messages"][0]["content"], "first_generated": rows[0]["generated"]}
assert all(q == all_questions[0] for q in all_questions)
assert all_questions[0][0] == "2+2=?"
assert saved["results"]["training"]["steps"] == 1000
assert saved["results"]["training"]["history"][-1]["step"] == 1000
result["historical_observation_audit"] = {"revision": revision,
    "results_sha256": sha((ART / "inputs/current/docs/course-experiments/results/style.json").read_bytes()),
    "historical_code_hashes_match_recorded": historical_hashes, "paired_questions": all_questions[0],
    "historical_dataset_reconstruction": rebuilt_hashes,
    "all_arithmetic_training_questions_have_three_style_conditions": True,
    "current_and_historical_generate_AST_identical": True,
    "denominators": {"distinct_arithmetic_questions": 7, "conditions_per_question": 3,
                     "generated_answers": 21, "conditional_training_steps": 1000,
                     "conditional_training_records": 185}, "recomputed": recomputed,
    "original_run_environment": {k:saved[k] for k in ["device","seed","torch_version","python_version","gpu"]},
    "tolerance": "exact raw IDs, decoded strings, record counts and integer numerators; rates exact 0 or 1",
    "scope": "audit of saved result and matching historical code; no checkpoint loaded, no rerun, no broad style ability",
    "raw_observations": raw_observations}

(ART / "checks-result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
print(json.dumps({k:v for k,v in result.items() if k != "historical_observation_audit"},ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in result["historical_observation_audit"].items() if k != "raw_observations"},ensure_ascii=False,indent=2))
