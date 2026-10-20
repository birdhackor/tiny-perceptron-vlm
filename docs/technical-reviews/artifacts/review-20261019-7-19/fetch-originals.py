"""讀取指定版本的上游原文，只保存收據與必要摘錄。"""

import hashlib
import json
import subprocess
from pathlib import Path


BASE = Path(__file__).resolve().parent
SOURCES = [
    (
        "transformers-config",
        "https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/generation/configuration_utils.py",
        "v4.57.1",
        [
            ("GenerationConfig.max_new_tokens", "        max_new_tokens (`int`, *optional*):", 2),
            ("GenerationConfig.stop_strings", "        stop_strings (`str or list[str]`, *optional*):", 2),
            ("GenerationConfig.forced_eos_token_id", "        forced_eos_token_id (", 3),
        ],
    ),
    (
        "vllm-config",
        "https://raw.githubusercontent.com/vllm-project/vllm/v0.8.5/vllm/config.py",
        "v0.8.5",
        [("ModelConfig.max_model_len", "        max_model_len: Maximum length of a sequence (including prompt and", 2)],
    ),
    (
        "chat-template",
        "https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/docs/source/en/chat_templating.md",
        "v4.57.1",
        [
            ("Introduction", "The critical insight needed", 4),
            ("apply_chat_template", "[`apply_chat_template`] takes this list", 1),
            ("add_generation_prompt", "This argument adds tokens", 2),
            ("Generation prompt compatibility", "Not all models require generation prompts", 1),
        ],
    ),
    (
        "eos-criterion",
        "https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/generation/stopping_criteria.py",
        "v4.57.1",
        [("EosTokenCriteria.__call__", "        is_done = isin_mps_friendly(input_ids[:, -1], self.eos_token_id)", 2)],
    ),
    (
        "generation-loop",
        "https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/generation/utils.py",
        "v4.57.1",
        [
            ("GenerationMixin._prepare_generated_length", "            generation_config.max_length = generation_config.max_new_tokens + input_ids_length", 1),
            ("GenerationMixin._sample: append before stop", "            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)", 6),
        ],
    ),
    (
        "forced-eos",
        "https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/generation/logits_process.py",
        "v4.57.1",
        [("ForcedEOSTokenLogitsProcessor.__call__", "        if cur_len == self.max_length - 1:", 4)],
    ),
]


receipts = []
for identifier, url, version, selections in SOURCES:
    command = ["curl", "-fsSL", url]
    completed = subprocess.run(command, check=True, capture_output=True)
    raw = completed.stdout
    lines = raw.decode("utf-8").splitlines()
    snippets = []
    for locator, needle, length in selections:
        matches = [index for index, line in enumerate(lines) if line.startswith(needle)]
        if len(matches) != 1:
            raise ValueError(f"原文定位必須唯一：{identifier} {locator}；實際符合數 {len(matches)}")
        start = matches[0]
        snippets.append({"locator": locator, "excerpt": "\n".join(lines[start : start + length])})
    receipts.append(
        {
            "id": identifier,
            "command_argv": command,
            "url": url,
            "version": version,
            "accessed_on": "2026-10-20",
            "returncode": completed.returncode,
            "full_source_sha256": hashlib.sha256(raw).hexdigest(),
            "full_source_bytes": len(raw),
            "snippets": snippets,
        }
    )

result = {"retrieval": "curl；全文只在記憶體中讀取，持久證據限於收據與必要摘錄。", "sources": receipts}
output = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
with (BASE / "original-source-receipts.json").open("x", encoding="utf-8") as handle:
    handle.write(output)
print(output, end="")
