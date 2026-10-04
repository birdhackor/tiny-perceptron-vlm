"""Expand frozen, source-attributed Chinese conversations without reselection."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs/natural-assistant/v4/data"
PROOF = ROOT / "docs/natural-assistant/evidence/v4-research/chat-speech"
PIN = "179dd21fc55192153d94adb0e0ce8f69e222bf75"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    ledger_path = DATA / "chat-curation-ledger.jsonl"
    original_path = DATA / "chat-original-conversations.json"
    ledger = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    originals = json.loads(original_path.read_text())
    conversations = list(ledger)
    for index, texts in enumerate(originals["conversations"], 1):
        conversations.append(
            {
                "family": f"project-chat-v4-{index:03d}",
                "split": "train",
                "source": "project-authored-traditional-chinese-v4",
                "license": "MIT",
                "synthetic": True,
                "messages": [
                    {"role": "user" if i % 2 == 0 else "assistant", "content": text} for i, text in enumerate(texts)
                ],
                "changes": [],
                "context_dependent_followup": len(texts) > 2,
                "provenance": originals["provenance"],
                "peer_review": originals.get("peer_review"),
            }
        )
    rows, seen_families = [], set()
    for conversation in conversations:
        family = conversation["family"]
        assert family not in seen_families, f"Duplicate selected tree: {family}"
        seen_families.add(family)
        messages = conversation["messages"]
        assert len(messages) >= 2 and len(messages) % 2 == 0
        assert [m["role"] for m in messages] == ["user", "assistant"] * (len(messages) // 2)
        for index in range(1, len(messages), 2):
            row = {
                "id": f"{family}-turn-{index // 2 + 1}",
                "family": family,
                "split": conversation["split"],
                "task": "chat",
                "source": conversation["source"],
                "license": conversation["license"],
                "synthetic": conversation["synthetic"],
                "user": messages[index - 1]["content"],
                "answer": messages[index]["content"],
                "history": messages[: index - 1],
                "system": "請用繁體中文清楚回應，遵循使用者的格式與上下文；資訊不足時先澄清，不虛構事實或親身經驗。",
                "context_dependent": index > 1 and conversation["context_dependent_followup"],
                "curation_ledger_family": family,
                "predeclared_semantic_rubric": "直接回應目前問題；遵循明示格式與限制；使用相關前文；不虛構背景或親身經驗。參考答案是可接受示例，不要求逐字匹配。",
                "evaluation_reference_kind": "Source-derived answer or project-authored reference; not a model-generated evaluation output",
            }
            if "source_message_ids" in conversation:
                row["source_message_ids"] = conversation["source_message_ids"][: index + 1]
                row["source_tree_id"] = conversation["source_tree_id"]
            peer_review = conversation.get("peer_review")
            if peer_review:
                overrides = peer_review.get("row_overrides", {}).get(row["id"], {})
                if set(overrides) - {"predeclared_semantic_rubric", "context_dependent"}:
                    raise ValueError(f"Unsupported peer-review override: {row['id']}")
                row.update(overrides)
                row["peer_review"] = {
                    "reviewer": peer_review["reviewer"],
                    "stage": peer_review["stage"],
                    "report": "docs/natural-assistant/evidence/v4-research/chat-gold-review/report.json",
                }
            rows.append(row)
    splits = {split: {r["family"] for r in rows if r["split"] == split} for split in ("train", "validation", "test")}
    assert not (
        splits["train"] & splits["validation"]
        or splits["train"] & splits["test"]
        or splits["validation"] & splits["test"]
    )
    target = DATA / "chat-labels.jsonl"
    target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    source_metadata = {
        "schema_version": 1,
        "dataset_version": "natural-chat-v4-1",
        "rows_file": str(target.relative_to(ROOT)),
        "rows_sha256": sha(target),
        "row_counts": dict(Counter(row["split"] for row in rows)),
        "tree_counts": {name: len(families) for name, families in splits.items()},
        "source_tree_count": len(ledger),
        "original_tree_count": len(originals["conversations"]),
        "source_multi_turn_tree_count": sum(len(c["messages"]) > 2 for c in ledger),
        "original_multi_turn_tree_count": sum(len(c) > 2 for c in originals["conversations"]),
        "context_dependent_row_counts": dict(Counter(r["split"] for r in rows if r["context_dependent"])),
        "ledger": {"path": str(ledger_path.relative_to(ROOT)), "sha256": sha(ledger_path)},
        "original_conversations": {"path": str(original_path.relative_to(ROOT)), "sha256": sha(original_path)},
        "upstream": {
            "name": "OpenAssistant/oasst2",
            "revision": PIN,
            "card_url": f"https://huggingface.co/datasets/OpenAssistant/oasst2/blob/{PIN}/README.md",
            "download_url": f"https://huggingface.co/datasets/OpenAssistant/oasst2/resolve/{PIN}/2023-11-05_oasst2_ready.messages.jsonl.gz",
            "full_download_bytes": 54301900,
            "full_download_sha256": "a9f240c4c77aa1378364f70d37e753c07ba284e247b019d700e1947a0e5da751",
            "license": "Apache-2.0",
            "license_scope": "Official dataset card licenses the dataset. Contributor terms were independently read; source annotations do not prove every community answer is accurate or independently human-authored.",
            "attribution": "OpenAssistant contributors / LAION; Köpf et al., OpenAssistant Conversations, https://arxiv.org/abs/2304.07327",
            "license_file": "docs/natural-assistant-v4/research/chat-speech.Apache-2.0-LICENSE",
            "notice_file": "docs/natural-assistant-v4/research/chat-speech.NOTICE.txt",
        },
        "curation": {
            "review": "Selected complete paths read by the data-curation agent; low-risk answers retained, explicit corrections recorded. This is a first editorial curation, not an independent expert audit.",
            "path_policy": "One selected path per source tree, preferring the highest-ranked accepted safe answer. Rank-1 selected when rank-0 is unsuitable; source ranks are preserved. Rejected continuations are omitted explicitly, not silently truncated.",
            "language_conversion": "opencc-python-reimplemented==0.1.7, s2tw, plus individually logged corrections; original exact source messages retained in evidence snapshot. Conversion-task input literals remain Simplified; reverse-string labels recomputed after conversion.",
            "split_policy": "Source-tree isolation; semantic near-duplicate reverse-string and four-line poetry groups placed together. SHA256 of stable split-v4-1/group gives deterministic ordering; Reserve one context-dependent source tree in each heldout split (219 validation / 113 test in research trace), then fill to 8 validation trees and 12 test trees by hash order; remainder train. All project-authored examples train only.",
            "duplicate_policy": "Each selected tree occurs once; assistant turns expand to separate rows. Repeated sampling by later training is not counted as new data.",
            "exclusions": "Clinical, legal, time-sensitive questions, invented model identity/experiences, factually wrong answers, unfulfilled constraints, and recognizable substantial third-party copied passages omitted.",
        },
        "limitations": [
            "Small instructional supplement, not a representative Chinese conversation corpus or proof of broad chat ability.",
            "OpenAssistant conversational examples can contain residual quality flaws; independent review remains necessary.",
            "A majority of new context-dependent examples are project-authored synthetic text; human/source distinction is explicit.",
            "Source-tree isolation and reviewed semantic-task grouping do not demonstrate that the foundation model never saw an upstream source; common instruction wording and related domains can recur across splits.",
            "No ASR audio is used to train Whisper here. Text and transcribed speech share the language-model route.",
        ],
        "source_message_snapshot": str((PROOF / "oasst2-selected-original-messages.jsonl").relative_to(ROOT)),
        "source_message_snapshot_sha256": sha(PROOF / "oasst2-selected-original-messages.jsonl"),
        "max_tokens_policy": "2048 current-turn training tokens; exact Qwen3-VL-2B chat-template/prefix-mask audit is separate, never silent truncation",
    }
    audit_path = DATA / "chat-token-audit.json"
    peer_review = originals.get("peer_review")
    if peer_review:
        source_metadata["curation"]["peer_review"] = {
            "reviewer": peer_review["reviewer"],
            "role": "independent chat-data peer reviewer, not source curator or final-book reviewer",
            "stage": peer_review["stage"],
            "rows_read": len(rows),
            "report": "docs/natural-assistant/evidence/v4-research/chat-gold-review/report.json",
            "edits": "docs/natural-assistant/evidence/v4-research/chat-gold-review/edits.json",
            "replay": "docs/natural-assistant/evidence/v4-research/chat-gold-review/apply-peer-review.py",
        }
        source_metadata["upstream"]["research_extract_notice_file"] = source_metadata["upstream"]["notice_file"]
        source_metadata["upstream"]["notice_file"] = "docs/natural-assistant/v4/data/chat-NOTICE.txt"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        if audit["rows_sha256"] == source_metadata["rows_sha256"] and not audit["failures"]:
            source_metadata["token_audit"] = {
                "path": str(audit_path.relative_to(ROOT)),
                "sha256": sha(audit_path),
                "passed_rows": audit["rows_audited"],
                "max_observed_tokens": audit["maximum_tokens"],
                "processor_revision": audit["model_revision"],
            }
    write_json(DATA / "chat-sources.json", source_metadata)
    print(
        json.dumps(
            {
                "rows": len(rows),
                "trees": len(conversations),
                "splits": source_metadata["row_counts"],
                "context_dependent_rows": source_metadata["context_dependent_row_counts"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
