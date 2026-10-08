"""Append-only proof that historical step inputs can be recovered without training."""
import copy
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
original = root / "tools.json"
before = hashlib.sha256(original.read_bytes()).hexdigest()
data = json.loads(original.read_text())

def serialize(messages):
    ids = [1]
    roles = {"system": 7, "user": 3, "assistant": 4}
    for message in messages:
        ids += [roles[message["role"]]] + [byte + 8 for byte in message["content"].encode("utf-8")] + [2]
    return ids + [4]

records = []
for condition, key in (("normal", "samples"), ("one_step_cap", "one_step_cap_samples")):
    for episode_index, episode in enumerate(data["results"][key]):
        first = next(event["generation"] for event in episode["trace"] if "generation" in event)
        current = copy.deepcopy(first["messages"][:2])
        for event_index, event in enumerate(episode["trace"]):
            if "generation" not in event:
                continue
            generation = event["generation"]
            expected_ids = serialize(current)
            assert expected_ids == generation["input_ids"], (condition, episode_index, event_index)
            assert len(expected_ids) == generation["input_tokens"]
            records.append({
                "original_json_pointer": f"/results/{key}/{episode_index}/trace/{event_index}/generation",
                "condition": condition,
                "question": episode["question"],
                "step": event["step"],
                "historical_messages_reconstruction": copy.deepcopy(current),
                "actual_input_ids": generation["input_ids"],
                "actual_input_tokens": generation["input_tokens"],
                "original_metadata_was_modified": generation["messages"] != current,
                "all_ids_verified_equal": True,
            })
            if event.get("executed") and event.get("finite_result"):
                current += [
                    {"role": "assistant", "content": generation["samples"][0]["generated"]},
                    {"role": "user", "content": f"TOOL_RESULT:{event['tool_result']}"},
                ]

assert len(records) == 56
assert sum(record["original_metadata_was_modified"] for record in records) == 36
after = hashlib.sha256(original.read_bytes()).hexdigest()
assert before == after
output = {
    "kind": "independent append-only reconstruction feasibility evidence",
    "original_file": "tools.json",
    "original_sha256_before_and_after": before,
    "scope": "Historical raw-record reconstruction only; no model invocation, training, new sampling, or original result overwrite.",
    "serialization": "ByteTokenizer: BOS=1; system=7, user=3, assistant=4; UTF-8 bytes+8; message EOS=2; trailing assistant=4.",
    "method": "Start from preserved system/user messages, append only earlier executed request and actual finite tool result, and require byte-for-byte equality with every original step input_ids.",
    "records_verified": len(records),
    "metadata_mismatches": 36,
    "records": records,
}
(root / "tool-input-snapshot-reconstruction.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({key: value for key, value in output.items() if key != "records"}, ensure_ascii=False, indent=2))
