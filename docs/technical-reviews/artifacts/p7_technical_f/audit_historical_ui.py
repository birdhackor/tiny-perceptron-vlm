import collections
import hashlib
import json
from pathlib import Path

root = Path("docs/natural-assistant/evidence/v4-research/student-selected-public-ui")
report = json.loads((root / "actual-ui/report.json").read_text())
events = [json.loads(line) for line in (root / "actual-ui/observer.jsonl").read_text().splitlines()]
requests = json.loads((root / "actual-ui/browser/requests.json").read_text())
for source in events[0]["sources"]:
    name = "tiny_perceptron/" + Path(source["path"]).name
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == source["sha256"]
assert events[0]["inference_mock"] is False
counts = collections.Counter(event["kind"] for event in events)
assert counts["load_core_end"] == 1 and counts["load_asr_end"] == 1
assert counts["generate_end"] == 4 and counts["transcribe_end"] == 1
assert all(row["status"] == 200 for row in requests)
reset = next(event for event in events if event["kind"] == "operation_end" and event["route"] == "/api/reset")
assert reset["state"]["history_messages"] == reset["state"]["assets"] == reset["state"]["transcriptions"] == 0
print(json.dumps({"historical_command": report["command"], "dependencies": report["dependencies"], "event_counts": dict(counts), "loads": [event for event in events if event["kind"] in ("load_core_begin", "load_asr_begin")], "requests": [{"route": row["route"], "status": row["status"], "prediction": row["response"].get("prediction"), "asr": row["response"].get("asr", {}).get("transcript") if row["response"].get("asr") else None} for row in requests], "reset_state": reset["state"], "server_resources": next(event for event in events if event["kind"] == "process_resources"), "scope": "Reanalysis of original CPU server/browser raw events; no model load, new inference, installation, or success-rate judgement in this audit."}, ensure_ascii=False))
