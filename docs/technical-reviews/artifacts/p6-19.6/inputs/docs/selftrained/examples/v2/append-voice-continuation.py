"""Append a user request to the actual generated voice history.

Run from the repository root after chat.py --history-output.
"""

import json
from pathlib import Path

history_path = Path("outputs/selftrained-v2/smoke/voice-first-history.json")
history = json.loads(history_path.read_text(encoding="utf-8"))
request_path = Path("docs/selftrained/examples/v2/voice-continuation-user.json")
request = json.loads(request_path.read_text(encoding="utf-8"))
if request.get("role") != "user" or set(request) != {"role", "content"}:
    raise ValueError("The next message must be a public user request")
if not history or history[-1].get("role") != "assistant":
    raise ValueError("First generate and save the actual assistant reply")
history.append(request)
output_path = Path("outputs/selftrained-v2/smoke/voice-next-messages.json")
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
