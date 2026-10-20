"""用固定版本的模板與 tokenizer，核對純文字答案後綴的實際目標。"""

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer
import torch

from tiny_perceptron.natural_assistant import encode_messages, encode_training_row, messages_for

REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"


def fetch(name):
    url = f"https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/resolve/{REVISION}/{name}"
    argv = ["curl", "-fsSL", "--max-time", "45", url]
    completed = subprocess.run(argv, capture_output=True, check=True)
    raw = completed.stdout
    return raw, {"url": url, "revision": REVISION, "command_argv": argv,
                 "exit_code": completed.returncode, "bytes": len(raw),
                 "sha256": hashlib.sha256(raw).hexdigest()}


class TextProcessor:
    """只組合上游 Jinja 模板與 Rust tokenizer，不代替圖片處理的驗證。"""

    def __init__(self, template, tokenizer):
        self.template = ImmutableSandboxedEnvironment().from_string(template)
        self.tokenizer = tokenizer

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False
        return self.template.render(messages=messages, add_generation_prompt=add_generation_prompt,
                                    add_vision_id=False, tools=None)

    def __call__(self, text, return_tensors):
        assert return_tensors == "pt"
        encoded = self.tokenizer.encode_batch(text, add_special_tokens=True)
        return {"input_ids": torch.tensor([item.ids for item in encoded], dtype=torch.long),
                "attention_mask": torch.tensor([item.attention_mask for item in encoded], dtype=torch.long)}


def main():
    template_raw, template_receipt = fetch("chat_template.json")
    tokenizer_raw, tokenizer_receipt = fetch("tokenizer.json")
    tokenizer = Tokenizer.from_str(tokenizer_raw.decode("utf-8"))
    processor = TextProcessor(json.loads(template_raw)["chat_template"], tokenizer)
    row = {"id": "qwen-text", "user": "Q", "answer": "A"}
    prompt_messages = messages_for(row, ROOT)
    full_messages = messages_for(row, ROOT, assistant=True)
    prompt_text = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    full_text = processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
    assert full_text.startswith(prompt_text)
    assert full_text[len(prompt_text):] == "A<|im_end|>\n"
    prompt = encode_messages(processor, prompt_messages, ROOT, generation_prompt=True)
    encoded = encode_training_row(processor, row, ROOT)
    active = encoded["labels"] != -100
    ids = encoded["labels"][active].tolist()
    tokens = [tokenizer.id_to_token(i) for i in ids]
    assert ids == [32, 151645, 198]
    assert tokenizer.decode([198], skip_special_tokens=False) == "\n"
    assert tokens[1] == "<|im_end|>"
    shifted = encoded["labels"][:, 1:]
    assert shifted[shifted != -100].tolist() == ids
    result = {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-20-7/qwen-text-probe.py",
        "environment": {"python": sys.version, "platform": platform.platform(), "device": "cpu",
                        "torch": str(torch.__version__), "tokenizers": importlib.metadata.version("tokenizers"),
                        "jinja2": importlib.metadata.version("jinja2"), "transformers": "not installed"},
        "original_receipts": {"template": template_receipt, "tokenizer": tokenizer_receipt},
        "wrapper_sha256": hashlib.sha256((ROOT / "tiny_perceptron/natural_assistant.py").read_bytes()).hexdigest(),
        "prompt_text": prompt_text, "full_text": full_text,
        "prompt_input_ids": prompt["input_ids"].tolist(),
        "input_ids": encoded["input_ids"].tolist(), "labels": encoded["labels"].tolist(),
        "supervised_ids": ids, "supervised_tokens": tokens,
        "supervised_text": tokenizer.decode(ids, skip_special_tokens=False),
        "supervised_after_shift": shifted[shifted != -100].tolist(),
        "newline_id": 198, "newline_text": tokenizer.decode([198], skip_special_tokens=False),
        "result": "答案 A、im_end 與 im_end 後的換行都是有效目標。",
        "scope": "使用固定上游模板及 tokenizer.json 的純文字路徑，呼叫專案真實 encode_messages/encode_training_row。未安裝或呼叫 Transformers AutoProcessor，未載入模型或圖片；完整視覺處理仍依上游原始碼核對。",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
