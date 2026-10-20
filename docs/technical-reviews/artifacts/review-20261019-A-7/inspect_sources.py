"""讀取原始來源，只保存版本收據與支持主張的短摘錄。"""

import hashlib
import html
import json
import platform
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path


DIRECTORY = Path(__file__).resolve().parent
SPECS = [
    (
        "hf_generation",
        "https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/generation/configuration_utils.py",
        "Transformers v4.57.1，GenerationConfig docstring",
        [r"        max_length \(", r"        max_new_tokens \(", r"        eos_token_id \("],
    ),
    (
        "hf_chat",
        "https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/docs/source/en/chat_templating.md",
        "Transformers v4.57.1，chat_templating.md 開頭及 Using apply_chat_template",
        [r"The critical insight", r"These base models", r"to a chat model get converted", r"There are many possible chat formats"],
    ),
    (
        "vllm_context",
        "https://raw.githubusercontent.com/vllm-project/vllm/v0.8.5/vllm/config.py",
        "vLLM v0.8.5，ModelConfig.max_model_len",
        [r"max_model_len: Maximum length of a sequence"],
    ),
    (
        "hf_truncation",
        "https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/docs/source/en/pad_truncation.md",
        "Transformers v4.57.1，truncation 參數說明",
        [r"- `True` or `'longest_first'`", r"- `'only_first'`"],
    ),
    (
        "lost_middle",
        "https://arxiv.org/html/2307.03172v3",
        "arXiv:2307.03172v3，Abstract、Experimental Setup、Results and Discussion",
        [
            r"We find that performance can degrade significantly.{0,230}",
            r"To modulate the position of relevant information.{0,270}",
            r"Although some models perform the synthetic key-value retrieval task perfectly.{0,200}",
        ],
    ),
]


def main():
    receipts = []
    for identifier, url, version, patterns in SPECS:
        command = ["curl", "-fLsS", "--proto", "=https", url]
        run = subprocess.run(command, check=True, capture_output=True)
        raw = run.stdout
        text = raw.decode("utf-8")
        snippets = []
        if identifier == "lost_middle":
            text = html.unescape(re.sub(r"<[^>]+>", " ", text))
            text = re.sub(r"\s+", " ", text)
            for pattern in patterns:
                match = re.search(pattern, text)
                if not match:
                    raise ValueError("原始論文缺少待核對的語句")
                snippets.append({"locator": version, "excerpt": match[0]})
        else:
            lines = text.splitlines()
            for pattern in patterns:
                matches = [i for i, line in enumerate(lines) if re.search(pattern, line)]
                if len(matches) != 1:
                    raise ValueError(f"無法唯一定位來源語句：{identifier} {pattern}")
                index = matches[0]
                count = 4 if identifier in {"hf_generation", "hf_truncation"} else 2 if identifier == "vllm_context" else 1
                snippets.append({"locator": f"{version}；原文讀取時第 {index + 1} 行起", "excerpt": "\n".join(lines[index:index + count])})
        receipts.append({
            "id": identifier,
            "url": url,
            "version": version,
            "accessed_on": datetime.now(UTC).date().isoformat(),
            "command_argv": command,
            "exit_code": run.returncode,
            "downloaded_bytes": len(raw),
            "original_sha256": hashlib.sha256(raw).hexdigest(),
            "checked_original": True,
            "excerpts": snippets,
        })
    result = {"command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-A-7/inspect_sources.py", "environment": {"python": platform.python_version(), "network_client": subprocess.check_output(["curl", "--version"], text=True).splitlines()[0]}, "sources": receipts}
    output = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    with (DIRECTORY / "source-receipts.json").open("x", encoding="utf-8") as handle:
        handle.write(output)
    print(output, end="")


if __name__ == "__main__":
    main()
