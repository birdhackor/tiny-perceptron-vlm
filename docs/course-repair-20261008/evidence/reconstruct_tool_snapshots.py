"""重建指定歷史工具紀錄的輸入快照；不執行模型、不覆寫原始結果。"""

import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "docs/course-experiments/results/tools.json"
EXPECTED_SHA256 = "3befd7fcb9a1a48d850761ab6d0917ac42388dc3bbfe9f011f0b98c3752d7140"
TARGET = Path(__file__).with_name("tool-input-snapshot-reconstruction.json")


def serialize(messages):
    """保存當時ByteTokenizer協議：BOS、角色、UTF-8 byte+8、EOS及助理起點。"""
    roles = {"system": 7, "user": 3, "assistant": 4}
    ids = [1]
    for message in messages:
        ids += [roles[message["role"]]] + [byte + 8 for byte in message["content"].encode("utf-8")] + [2]
    return ids + [4]


def main():
    original = SOURCE.read_bytes()
    digest = hashlib.sha256(original).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError("來源不是本勘誤指定的歷史tools.json；停止重建。")
    data = json.loads(original)
    records = []
    for condition, key in (("normal", "samples"), ("one_step_cap", "one_step_cap_samples")):
        for episode_index, episode in enumerate(data["results"][key]):
            first = next(event["generation"] for event in episode["trace"] if "generation" in event)
            current = copy.deepcopy(first["messages"][:2])
            if [message["role"] for message in current] != ["system", "user"]:
                raise ValueError("起始角色不符合指定歷史協議。")
            if current[1]["content"] != episode["question"]:
                raise ValueError("起始題目與episode紀錄不符。")
            for event_index, event in enumerate(episode["trace"]):
                if "generation" not in event:
                    continue
                generation = event["generation"]
                ids = serialize(current)
                pointer = f"/results/{key}/{episode_index}/trace/{event_index}/generation"
                if ids != generation["input_ids"] or len(ids) != generation["input_tokens"]:
                    raise ValueError(f"{pointer} 重建與原始模型輸入不符。")
                records.append(
                    {
                        "original_json_pointer": pointer,
                        "condition": condition,
                        "question": episode["question"],
                        "step": event["step"],
                        "messages_at_generation": copy.deepcopy(current),
                        "actual_input_ids": generation["input_ids"],
                        "actual_input_tokens": generation["input_tokens"],
                        "original_metadata_differs": generation["messages"] != current,
                        "all_ids_verified_equal": True,
                    }
                )
                if event.get("executed") and event.get("finite_result"):
                    current += [
                        {"role": "assistant", "content": generation["samples"][0]["generated"]},
                        {"role": "user", "content": f"TOOL_RESULT:{event['tool_result']}"},
                    ]
    mismatches = sum(record["original_metadata_differs"] for record in records)
    if len(records) != 56 or mismatches != 36:
        raise ValueError("步數或受影響快照數與指定歷史紀錄不符。")
    if SOURCE.read_bytes() != original:
        raise ValueError("重建期間原始紀錄被更改。")
    result = {
        "kind": "append-only historical tool-input reconstruction",
        "original_file": SOURCE.relative_to(ROOT).as_posix(),
        "original_sha256": digest,
        "scope": "Historical metadata reconstruction only; no model invocation, training, new sampling or source overwrite.",
        "serialization": "ByteTokenizer: BOS=1; system=7, user=3, assistant=4; UTF-8 bytes+8; EOS=2; trailing assistant=4.",
        "method": "Start from saved system/user messages; append only earlier executed request and finite tool result; compare every reconstructed ID and token count with the original generation input.",
        "records_verified": len(records),
        "metadata_mismatches": mismatches,
        "records": records,
    }
    encoded = (json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode()
    if TARGET.exists() and TARGET.read_bytes() != encoded:
        raise ValueError("既存補件內容不同；不覆寫，請另存新版本。")
    if not TARGET.exists():
        TARGET.write_bytes(encoded)
    print(json.dumps({"records_verified": len(records), "metadata_mismatches": mismatches, "source_sha256": digest}))


if __name__ == "__main__":
    main()
