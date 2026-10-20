"""保存小節執行收據，重算固定算例、資料切分、監督量與既有生成的成績。"""

import ast
import hashlib
import json
import platform
import random
import re
import subprocess
import sys
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DEST = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(name, value):
    path = DEST / name
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path.read_bytes())}))


class Paragraphs(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paragraphs = []
        self.active = False
        self.current = ""
        self.ids = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.append(attrs["id"])
        if tag == "p":
            self.active = True
            self.current = ""
            self.locator = self.ids[-1] if self.ids else ""

    def handle_data(self, value):
        if self.active:
            self.current += value

    def handle_endtag(self, tag):
        if tag == "p" and self.active:
            self.paragraphs.append((self.locator, re.sub(r"\s+", " ", self.current).strip()))
            self.active = False


def inspect_sources():
    records = []
    configs = [
        (
            "https://arxiv.org/html/1706.03762v7",
            "arXiv:1706.03762v7",
            "Attention Is All You Need",
            "At each step the model is auto-regressive",
            "consuming the previously generated symbols as additional input when generating the next",
            "Model Architecture 與 Encoder and Decoder Stacks：decoder 逐項生成、先前輸出成為下一步輸入；因果遮罩只讓預測依賴已知輸出。這只證明可利用先前文字，不證明中間步驟一定有用。",
        ),
        (
            "https://arxiv.org/html/2305.04388v2",
            "arXiv:2305.04388v2",
            "Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting",
            "Large Language Models (LLMs) can achieve",
            "CoT explanations can systematically misrepresent the true reason for a model’s prediction",
            "Abstract、Introduction、Explanation Faithfulness、Limitations：作者改變提示中的偏置線索，觀察預測改變而解釋不提線索；偏向錯答時仍會產生合理化文字。研究測 GPT-3.5、Claude 1.0 的特定任務，只支持不能預設可見解釋忠實或正確，不支持所有模型每次都不忠實。",
        ),
    ]
    for url, version, title, needle, quote, note in configs:
        command = ["curl", "-L", "--fail", "--silent", "--show-error", url]
        raw = subprocess.check_output(command)
        parser = Paragraphs()
        parser.feed(raw.decode("utf-8"))
        matches = [(loc, text) for loc, text in parser.paragraphs if needle in text]
        assert len(matches) == 1, "原文定位必須唯一"
        locator, paragraph = matches[0]
        assert quote in paragraph, "摘錄必須存在於實際取得的原文"
        assert len(quote.split()) <= 25, "只保存支持主張的短摘錄"
        records.append({
            "url": url, "version": version, "title": title,
            "accessed_on": datetime.now().astimezone().date().isoformat(),
            "command_argv": command, "returncode": 0,
            "retrieved_html_sha256": sha(raw), "retrieved_bytes": len(raw),
            "locator": locator, "excerpt": quote, "inspection_note": note,
            "full_text_saved": False,
        })
    save("primary-sources.json", {"sources": records})


def copy_section_execution():
    origin = ROOT / "outputs/reviewer-tools/runs/20261020T073411397458Z"
    files = {}
    for name in ("stdout.txt", "stderr.txt", "environment.json", "execution.json", "extraction.json", "fence-1.py", "bootstrap.py"):
        raw = (origin / name).read_bytes()
        files[name] = {"sha256": sha(raw), "content": raw.decode("utf-8")}
    save("section-execution.json", {
        "command": 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/0C.md#C.1" --execute',
        "result": "執行成功；回答 token 數為 1、47；沒有圖；原文雜湊符合凍結版。",
        "origin": origin.relative_to(ROOT).as_posix(), "files": files,
    })


def audit():
    import torch
    from scripts.course_experiments import applications as app
    from scripts.course_experiments.common import records_sha256, split_records, text_examples
    from tiny_perceptron.data import IGNORE, ByteTokenizer, CharTokenizer, render_chat
    from tiny_perceptron.model import ModelConfig, TinyLM

    torch.set_num_threads(1)
    result_path = ROOT / "docs/course-experiments/results/reasoning.json"
    recorded = json.loads(result_path.read_bytes())
    source_raw = (ROOT / "docs/course-review-20261019/freeze-01/sources/C.1.md").read_bytes()
    assert sha(source_raw) == "4f5894f1ccf2cc68aec0dfe3b9a42f88b5cd6c33f080dc19d03966739a39008e"
    tok = ByteTokenizer()
    texts = ["5", "從2開始數三次：3、4、5，所以是5。", "2+3=5"]
    token_checks = []
    for text in texts:
        ids = tok.encode(text)
        assert ids == [value + 8 for value in text.encode("utf-8")]
        assert not any(value in (tok.bos_id, tok.eos_id, tok.user_id, tok.assistant_id) for value in ids)
        token_checks.append({"text": text, "characters": len(text), "bytes": len(text.encode("utf-8")), "tokens": len(ids), "per_character_bytes": [len(ch.encode("utf-8")) for ch in text]})
    assert [row["tokens"] for row in token_checks] == [1, 47, 5]
    other_length = len(CharTokenizer(texts[1]).encode(texts[1]))
    assert other_length == 19

    rows = [{"family": ":".join(map(str, sorted((a,b,c)))), "a": a, "b": b, "c": c, "question": f"({a}+{b})+{c}=?", "truth": a+b+c} for a in range(6) for b in range(6) for c in range(6)]
    assert rows == app._reasoning_records()
    groups = {}
    for row in rows:
        groups.setdefault(row["family"], []).append(row)
    keys = sorted(groups)
    random.Random(42).shuffle(keys)
    independent_splits = {name:[row for family in families for row in groups[family]] for name,families in (("train",keys[:44]),("validation",keys[44:50]),("test",keys[50:]))}
    splits = split_records(rows, 42)
    assert splits == independent_splits
    summary = {name:{"records":len(values), "families":len({row["family"] for row in values}), "sha256":records_sha256(values)} for name,values in splits.items()}
    assert summary == recorded["results"]["split"]
    family_sets = {name:{row["family"] for row in values} for name,values in splits.items()}
    assert not (family_sets["train"] & family_sets["validation"] or family_sets["train"] & family_sets["test"] or family_sets["validation"] & family_sets["test"])

    model = TinyLM(ModelConfig(**recorded["results"]["model_config"]))
    parameters = sum(p.numel() for p in model.parameters())
    assert parameters == 141568
    rng = random.Random(42)
    schedule = [rng.choices(range(167), k=24) for _ in range(900)]
    branches = {}
    for mode, expected in (("direct", 48803), ("steps", 466322)):
        branch = recorded["results"]["comparison"][mode]
        sft = app._reasoning_sft(splits["train"], mode)
        examples = text_examples(sft, "sft", max_length=128)
        counts = []
        for row, example in zip(sft,examples,strict=True):
            a,b,c = row["a"],row["b"],row["c"]
            answer = str(a+b+c) if mode=="direct" else f"{a}+{b}={a+b};{a+b}+{c}={a+b+c};answer={a+b+c}"
            assert row["messages"][1]["content"] == answer
            labels = example[1]
            expected_labels = tok.encode(answer)+[tok.eos_id]
            assert labels[labels!=IGNORE].tolist() == expected_labels
            counts.append(len(expected_labels))
        used = sum(counts[i] for batch in schedule for i in batch)
        assert used == expected == branch["training"]["effective_tokens"]
        assert records_sha256(sft) == branch["training"]["records_sha256"]
        assert branch["training"]["steps"] == 900 and branch["training"]["records"] == 167
        assert branch["training"]["parameters"] == parameters
        own_sampler = random.Random(42)
        sampled_examples = [own_sampler.choices(examples,k=24) for _ in range(900)]
        assert sum(int((labels != IGNORE).sum()) for batch in sampled_examples for _,labels in batch) == used
        budget = branch["budgets"][0]
        assert budget["candidate_count"] == 1
        assert [{key:sample[key] for key in splits["test"][0]} for sample in budget["samples"]] == splits["test"]
        predictions = []
        for row in budget["samples"]:
            assert row["generation"]["temperature"] == 0.7
            assert row["generation"]["max_new_tokens"] == (8 if mode=="direct" else 48)
            assert row["generation"]["candidate_count"] == 1 and len(row["candidates"]) == 1
            candidate = row["candidates"][0]
            ids = candidate["generated_ids"]
            raw = ids[:-1] if ids and ids[-1]==tok.eos_id else ids
            assert tok.decode(raw) == candidate["generated"]
            assert len(ids) == candidate["generated_tokens"] <= row["generation"]["max_new_tokens"]
            assert not candidate["invalid_special_tokens"]
            text = candidate["generated"].strip()
            match = re.fullmatch(r"-?[0-9]+",text) if mode=="direct" else re.search(r";answer=(-?[0-9]+)$",text)
            answer = int(match[0] if mode=="direct" else match[1]) if match else None
            correct = answer == row["a"]+row["b"]+row["c"]
            assert answer == candidate["final_answer"] and correct == candidate["final_correct"]
            if mode=="steps":
                numbers = list(map(int,re.findall(r"-?\d+",text)))
                u,v,w,x,y,z,final = numbers
                fully = u+v==w and x+y==z and w==x and z==final and (u,v,y)==(row["a"],row["b"],row["c"]) and correct
                assert fully == candidate["fully_verified"]
            predictions.append({"question":row["question"],"truth":row["truth"],"generated":text,"independent_final_answer":answer,"correct":correct})
        correct = sum(row["correct"] for row in predictions)
        assert correct == budget["candidate_final_accuracy"]["numerator"] == (1 if mode=="direct" else 11)
        branches[mode] = {"training_steps":900,"batch_size":24,"sampled_records":21600,"effective_targets":used,"records_sha256":records_sha256(sft),"test_records":len(predictions),"correct":correct,"candidate_count":1,"temperature":0.7,"max_new_tokens":8 if mode=="direct" else 48,"generated_tokens_in_saved_samples":sum(row["candidates"][0]["generated_tokens"] for row in budget["samples"]),"predictions":predictions}

    application_path = "scripts/course_experiments/applications.py"
    historical_commit = "dc8d1ac3cec62d33ad973e54ba7eaf44732d8c17"
    historical_raw = subprocess.check_output(["git","show",f"{historical_commit}:{application_path}"],cwd=ROOT)
    assert sha(historical_raw) == recorded["code_sha256"][application_path]
    def definitions(raw):
        return {node.name:node for node in ast.parse(raw).body if isinstance(node,ast.FunctionDef)}
    before = definitions(historical_raw)
    after = definitions((ROOT/application_path).read_bytes())
    names = ["run_reasoning","_reasoning_records","_reasoning_sft","_verify_reasoning","_reasoning_samples","_split_summary","_conversation","_training_steps","_fit_lm","_prompt_ids"]
    ast_equal = {name:ast.dump(before[name],include_attributes=False)==ast.dump(after[name],include_attributes=False) for name in names}
    assert all(ast_equal.values())
    sample_before = ast.get_source_segment(historical_raw.decode(), before["_sample"])
    sample_after = ast.get_source_segment((ROOT/application_path).read_text(),after["_sample"])
    normalized_after = sample_after.replace('"messages": copy.deepcopy(messages),','"messages": messages,')
    assert ast.dump(ast.parse(sample_before),include_attributes=False)==ast.dump(ast.parse(normalized_after),include_attributes=False)
    important_files = ["tiny_perceptron/data.py","tiny_perceptron/model.py","tiny_perceptron/attention.py","tiny_perceptron/modern.py","tiny_perceptron/training.py","scripts/course_experiments/common.py"]
    hash_checks = {name:sha((ROOT/name).read_bytes())==recorded["code_sha256"][name] for name in important_files}
    assert all(hash_checks.values())
    _,labels = render_chat([{"role":"user","content":"(1+2)+3=?"},{"role":"assistant","content":"6"}],tok)
    assert labels[labels!=IGNORE].tolist()==tok.encode("6")+[tok.eos_id]
    ratio = branches["steps"]["effective_targets"]/branches["direct"]["effective_targets"]
    assert round(ratio,2)==9.56 and round(ratio,1)==9.6
    save("audit.json", {
        "command":"PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-C-1/audit.py audit",
        "result":"所有斷言通過；未訓練模型或重新抽樣生成。",
        "environment":{"python":platform.python_version(),"torch":str(torch.__version__),"device":"cpu","platform":platform.platform()},
        "source_sha256":sha(source_raw), "experiment_input":{"path":result_path.relative_to(ROOT).as_posix(),"sha256":sha(result_path.read_bytes()),"revision_as_recorded":recorded["revision"],"device":recorded["device"],"gpu":recorded["gpu"],"torch":recorded["torch_version"],"python":recorded["python_version"],"seed":recorded["seed"],"status":recorded["status"],"evidence_status":recorded["evidence_status"]},
        "arithmetic_example":{"count_from":2,"successive_values":[2+i for i in range(1,4)],"answer":2+3,"sft_direct":"6","sft_steps":"1+2=3;3+3=6;answer=6"},
        "tokenization":token_checks,"alternative_character_tokenizer_length":other_length,
        "all_records":len(rows),"all_families":len(groups),"split":summary,"disjoint_families":True,
        "parameters":parameters,"model_config":recorded["results"]["model_config"],
        "initial_state_sha256":{mode:branch["base_state_sha256"] for mode,branch in recorded["results"]["comparison"].items()},
        "initialization_inspection":"run_reasoning 建立隨機 base，深複製 state_dict，再分別 load_state_dict；沒有預訓練權重載入。fit_lm 每支使用相同種子的獨立 random.Random，對同次序資料做 choices。",
        "sample_schedule_sha256":sha(json.dumps(schedule).encode()),"branches":branches,"effective_target_ratio":ratio,
        "code_provenance":{"historical_commit":historical_commit,"historical_applications_sha256":sha(historical_raw),"current_applications_sha256":sha((ROOT/application_path).read_bytes()),"same_function_ast":ast_equal,"sample_ast_equal_after_only_messages_copy_normalization":True,"sample_difference":"目前 _sample 只將 messages 欄位改成 deepcopy，生成迴圈與評分不變。","other_current_files_match_recorded_hashes":hash_checks},
        "limitations":["從保存的原始生成重算成績，不是重新在 L4 訓練或生成。","一次 seed=42、24 道測試題，每題一份回答，不能推論普遍能力。","兩支的監督 token 與生成上限不同，不能隔離寫步驟的單獨效果。","UTF-8 的三 byte 結論只套用於算例中實際出現的中文字和標點，不外推所有 Unicode 漢字。"],
    })


if __name__ == "__main__":
    if sys.argv[1:] == ["sources"]:
        inspect_sources()
    elif sys.argv[1:] == ["section"]:
        copy_section_execution()
    elif sys.argv[1:] == ["audit"]:
        audit()
    else:
        raise SystemExit("請指定 sources、section 或 audit")
