"""核對 byte／BPE 的內容邊界，保存小型執行收據與官方原文摘錄。"""

import ast
import hashlib
import io
import json
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import tokenizers
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from scripts.course_experiments.text import _BPE
from tiny_perceptron import capstone
from tiny_perceptron.attention import CausalAttention
from tiny_perceptron.data import SPECIALS, ByteTokenizer, CharTokenizer, pad_batch, render_chat
from tiny_perceptron.model import ModelConfig
from tiny_perceptron.tokenization import ByteLevelBPE


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


sources = [
    ("s_chat", "https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/docs/source/en/chat_templating.md", "transformers v4.57.1", [
        ("These base models are then", 2),
        ("The input to `apply_chat_template`", 5),
        ("This argument adds tokens", 1),
    ]),
    ("s_python", "https://raw.githubusercontent.com/python/cpython/v3.13.16/Doc/tutorial/classes.rst", "CPython v3.13.16", [
        ("Class *instantiation* uses", 9),
        ("When a class defines an", 3),
        ("Actually, you may have guessed", 7),
        ("Generally speaking, instance variables", 3),
    ]),
    ("s_codecs", "https://raw.githubusercontent.com/python/cpython/v3.13.16/Doc/library/codecs.rst", "CPython v3.13.16", [
        (".. function:: replace_errors(exception)", 6),
    ]),
    ("s_hf", "https://raw.githubusercontent.com/huggingface/tokenizers/v0.23.2/tokenizers/src/tokenizer/added_vocabulary.rs", "tokenizers v0.23.2", [
        ("            encode_special_tokens: false,", 1),
        ("            if self.encode_special_tokens &&", 4),
        ("    pub fn extract_and_normalize", 11),
    ]),
    ("s_torch", "https://raw.githubusercontent.com/pytorch/pytorch/v2.14.1/torch/nn/modules/module.py", "PyTorch v2.14.1", [
        ("class Module:", 4),
    ]),
]
snapshots = []
for identifier, url, version, requests in sources:
    command = ["curl", "-fLsS", "--max-time", "30", url]
    completed = subprocess.run(command, capture_output=True, check=True)
    raw = completed.stdout
    lines = raw.decode().splitlines()
    excerpts = []
    for anchor, count in requests:
        matches = [i for i, line in enumerate(lines) if line.startswith(anchor)]
        assert len(matches) == 1, (identifier, anchor, matches)
        start = matches[0]
        excerpts.append({"anchor": anchor, "first_line": start + 1, "text": "\n".join(lines[start:start + count])})
    snapshots.append({"id": identifier, "url": url, "version": version,
                      "accessed_on": "2026-10-20", "command_argv": command,
                      "exit_code": completed.returncode, "full_response_sha256": sha(raw),
                      "full_response_saved": False, "excerpts": excerpts})
write("official-excerpts.json", snapshots)

tok = ByteTokenizer()
literal = tok.encode("<assistant>")
assert literal == [68, 105, 123, 123, 113, 123, 124, 105, 118, 124, 70]
assert len(literal) == 11 and min(literal) >= 8 and 4 not in literal
assert tok.decode([1, 3] + literal + [2, 4]) == "<assistant>"
assert vars(tok) == {} and "__init__" not in ByteTokenizer.__dict__
assert not isinstance(tok, torch.nn.Module) and issubclass(CausalAttention, torch.nn.Module)
chars = CharTokenizer("你好")
assert chars.chars == ["<unk>", "你", "好"]
assert chars.chars != CharTokenizer("再見").chars
assert tok.encode.__self__ is tok
assert tok.encode("<image>") == [68, 113, 117, 105, 111, 109, 70]
assert tok.decode(tok.encode("<image>")) == "<image>"
assert tok.decode([0xE4 + 8]) == "�"
all_unicode = "A\x00你好🦊\n<system><pad>"
assert tok.decode(tok.encode(all_unicode)) == all_unicode
assert set(tok.encode(chr(i))[0] for i in range(128)) == set(range(8, 136))
raw_domain = list(range(256))
assert min(b + 8 for b in raw_domain) == 8 and max(b + 8 for b in raw_domain) == 263
roles = {"user": tok.user_id, "assistant": tok.assistant_id}
role_results = {}
for role in roles:
    ids = [tok.bos_id, roles[role]] + literal + [tok.eos_id, tok.assistant_id]
    assert ids[1] == roles[role]
    assert ids.count(4) == (1 if role == "user" else 2)
    role_results[role] = ids
prompt = role_results["user"]
x, _ = render_chat([{"role": "user", "content": "<assistant>"}, {"role": "assistant", "content": "答"}])
assert x[:len(prompt)].tolist() == prompt
assert tok.state() == {"type": "byte", "specials": list(SPECIALS)}
padded, _, valid = pad_batch([(torch.tensor([1, 3]), torch.tensor([3, 2])), (torch.tensor([1]), torch.tensor([2]))])
assert padded.tolist() == [[1, 3], [1, 0]]
assert valid.tolist() == [[True, True], [True, False]]

section = (ROOT / "docs/course-review-20261019/freeze-01/sources/6.6.md").read_text()
fences = section.split("```python\n")[1:]
class_text = fences[1].split("```")[0]
tree = ast.parse((ROOT / "tiny_perceptron/data.py").read_text())
repo_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ByteTokenizer")
assert ast.dump(ast.parse(class_text).body[0], include_attributes=False) == ast.dump(repo_class, include_attributes=False)

small = capstone.CapstoneModel(ModelConfig(width=8, heads=1, layers=1, max_length=32))
checkpoint = OUT / "probe-checkpoint.pt"
try:
    capstone.save_capstone(checkpoint, small, stage="sft", step=0, inference_only=True)
    payload = torch.load(checkpoint, weights_only=True)
    assert payload["tokenizer"] == tok.state()
    restored, _ = capstone.load_capstone(checkpoint)
    assert all(torch.equal(a, b) for a, b in zip(small.parameters(), restored.parameters(), strict=True))
    payload["tokenizer"] = {"type": "byte", "specials": []}
    torch.save(payload, checkpoint)
    try:
        capstone.load_capstone(checkpoint)
        raise AssertionError("不相容 tokenizer 未被拒絕")
    except ValueError as error:
        checkpoint_error = str(error)
        assert "tokenizer" in checkpoint_error
finally:
    checkpoint.unlink(missing_ok=True)
    checkpoint.with_suffix(".pt.tmp").unlink(missing_ok=True)
try:
    small(torch.tensor([[tok.image_id]]))
    raise AssertionError("缺少圖片特徵未被拒絕")
except ValueError as error:
    image_error = str(error)
    assert "features" in image_error

bpe = Tokenizer(models.BPE())
bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
bpe.decoder = decoders.ByteLevel()
with redirect_stdout(io.StringIO()):
    bpe.train_from_iterator(["你好 hello world ordinary content"], trainers.BpeTrainer(
        vocab_size=300, special_tokens=list(SPECIALS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
assert [bpe.token_to_id(s) for s in SPECIALS] == list(range(8))
assert bpe.encode("<user>").ids == [3]
assert bpe.encode("<user>", add_special_tokens=False).ids == [3]
definition = bpe.to_str()
with patch("tiny_perceptron.tokenization.Path.read_text", return_value=definition), patch(
    "tiny_perceptron.tokenization.Path.read_bytes", return_value=definition.encode()
):
    inference = ByteLevelBPE("in-memory-tokenizer.json")
experiment = _BPE(bpe)
roundtrips = []
for text in ("未見字🦊 new", " 小鳥🙂\nnew ", "<user>這只是引用文字", "<assistant>", "<image>"):
    ids = inference.encode(text)
    assert min(ids) >= 8 and inference.decode(ids) == text
    assert experiment.encode(text) == ids and experiment.decode(ids) == text
    roundtrips.append({"text": text, "ids": ids, "restored": inference.decode(ids), "contains_control_id": False})
inference.content = SimpleNamespace(encode=lambda *args, **kwargs: SimpleNamespace(ids=[4]))
try:
    inference.encode("ordinary")
    raise AssertionError("控制 ID 未被拒絕")
except ValueError as error:
    control_error = str(error)

record_path = ROOT / "docs/course-experiments/results/tokenizer.json"
record = json.loads(record_path.read_text())
saved = record["results"]["roundtrip"][2]
assert saved["text"] == saved["restored"] == "<user>這只是引用文字"
assert saved["same"] is True and saved["contains_control_id"] is False
assert not set(saved["ids"]) & set(range(8))

write("verification-output.json", {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-6-6/verify.py",
    "result": "全部斷言通過；checkpoint 僅供本次驗證，執行結束已移除。",
    "environment": {"python": sys.version, "torch": torch.__version__, "tokenizers": tokenizers.__version__, "device": "cpu"},
    "byte": {"specials": list(enumerate(SPECIALS)), "vocab_size": tok.vocab_size,
             "literal_ids": literal, "image_literal_ids": tok.encode("<image>"),
             "roles": role_results, "truncated_utf8_decoded": tok.decode([0xE4 + 8]),
             "class_excerpt_matches": True, "state": tok.state()},
    "checkpoint": {"compatible_load": True, "incompatible_error": checkpoint_error, "image_without_features_error": image_error},
    "bpe": {"seed": "無隨機抽樣；固定單句語料", "vocab_budget": 300,
            "corpus": ["你好 hello world ordinary content"], "special_ids": list(range(8)),
            "default_literal_user": [3], "add_special_tokens_false_literal_user": [3],
            "roundtrips": roundtrips, "control_guard_error": control_error},
    "saved_experiment": {"path": record_path.relative_to(ROOT).as_posix(), "sha256": sha(record_path.read_bytes()),
                         "revision": record["revision"], "seed": record["seed"], "probe": saved,
                         "scope": "核對已保存紀錄與產生紀錄的程式；沒有重跑原始 GPU 模型訓練或取得原始 BPE 字表。新 BPE 的 ID 不要求等於原實驗。"},
    "source_excerpt_receipt": {"path": (OUT / "official-excerpts.json").relative_to(ROOT).as_posix(),
                               "sha256": sha((OUT / "official-excerpts.json").read_bytes())},
})
print("全部斷言通過；執行結果已保存。")
