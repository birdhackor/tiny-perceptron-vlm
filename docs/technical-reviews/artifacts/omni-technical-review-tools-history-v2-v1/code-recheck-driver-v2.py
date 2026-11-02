"""Phase 2: re-run the immutable v1 driver using the coordinator's scope-v3."""
import hashlib
from pathlib import Path

path = Path(__file__).with_name("code-recheck-driver-v1.py")
assert hashlib.sha256(path.read_bytes()).hexdigest() == '425a655df85b6820af510dd8ef3c5d3462ba1582efd781166ab4ec8cbea32da7'
code = path.read_text()
changes = [
    ("history-v2-review-scope-v2.json", "history-v2-review-scope-v3.json"),
    ("c73cba5795420d0aac2cd0db2af01ef9cf4df9b7dfab43120e20119bd02cd97c", "1031640666d937e4b49f96a24057830afb8dab7d58344b31a5ded80274d16076"),
    ("technical-review-tools-history-v2-code-recheck-v1", "technical-review-tools-history-v2-code-recheck-v2"),
    ("preview=ROOT/'outputs/omni-review-site-colab-preview-fix01'", "preview=ROOT/'outputs/omni-review-site-colab-preview-fix02'"),
    ("code-recheck-execution-v1.json", "code-recheck-execution-v2.json"),
]
for old, new in changes:
    assert code.count(old) == 1, old
    code = code.replace(old, new)
# Existing File-menu text is still present as the Notebook-screen alternative;
# the current complete footer was separately read before this execution.
exec(compile(code, str(__file__), "exec"), {"__name__": "__main__", "__file__": str(__file__)})
